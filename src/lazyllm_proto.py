"""LazyLLM baseline (Fu et al. 2024, arXiv 2407.14057), prefill-only faithful-minimal version in Engine style.

No official code exists (paper only). What is implemented (paper Sec. 3):
  * importance of prompt token i at layer l:  s_i^l = (1/H) sum_h A^l_{h, last, i}   (last prompt token's attention,
    softmax over the tokens still active at layer l, mean over heads);
  * token i is pruned at layer l+1 if s_i^l is below the k^l-th percentile of the active tokens (top-k keep);
  * progressive schedule: prune at a few layers, keeping more tokens early and fewer later; a token pruned at
    layer l stays absent from every layer >= l of this forward pass (its K/V for those layers never exist);
  * the last prompt token is never pruned (it is the query that produces the first answer token).
Deviation (documented): the Aux-Cache *revival* of pruned tokens during later decode steps is not implemented;
pruned tokens stay absent during decoding (static drop, as in FastKV-style eviction). This is the TTFT/prefill
part of LazyLLM, which is what a matched-prefill-compute comparison measures.

Accuracy-evaluation style exactly like Engine.prefill(mode="drop"): all positions are computed, pruned tokens are
masked out as keys in layers >= their prune layer; their hidden states above that layer are never read, so the
outputs equal the compute-saving implementation.

Saving (layer-token work of the prompt skipped), schedule [(l_1,r_1),...,(l_n,r_n)], r_0 = 1:
    S = sum_i (r_{i-1} - r_i) * (L - l_i) / L
"""
import math
import torch
import torch.nn.functional as F
from engine import DEV, rope


def lazy_saving(schedule, L):
    s, prev = 0.0, 1.0
    for l, r in schedule:
        s += (prev - r) * (L - l) / L; prev = r
    return s


def lazy_schedule(L, target=0.49, n_stages=3, layers=None):
    """Geometric progressive schedule: prune at layers l_i = round(i*L/(n+1)) (default L/4, L/2, 3L/4 for n=3) and
    keep r_i = rho^i of the prompt from l_i on; rho found by bisection so that lazy_saving == target."""
    layers = layers or [round(i * L / (n_stages + 1)) for i in range(1, n_stages + 1)]
    lo, hi = 0.0, 1.0
    for _ in range(60):
        rho = (lo + hi) / 2
        if lazy_saving([(l, rho ** (i + 1)) for i, l in enumerate(layers)], L) > target:
            lo = rho
        else:
            hi = rho
    return [(l, rho ** (i + 1)) for i, l in enumerate(layers)]


@torch.no_grad()
def lazy_prefill(eng, ids, m, schedule, max_new=0, last_only=False):
    """Drop-in for Engine.prefill: returns (logits, cache) usable by eng.decode. schedule: list of (layer, keep_frac)
    with 1 <= layer < L increasing and keep_frac (fraction of the row's valid prompt tokens still active from that
    layer on) decreasing. Sets eng.lazy_info = {"saving": realized layer-token saving, "active": [L] mean counts}."""
    model, L, H, KV, hd = eng.model, eng.L, eng.H, eng.KV, eng.hd
    B, P = ids.shape
    sched = dict(schedule)
    assert all(1 <= l < L for l in sched), "score at layer l-1 prunes at layer l"
    pos = (m.long().cumsum(1) - 1).clamp(min=0)
    h = model.model.embed_tokens(ids)
    cos, sin = model.model.rotary_emb(h, pos)
    nvalid = m.sum(1)
    active = m.clone()
    pruned_any = False
    fused = B == 1 and bool(m.all())
    causal = torch.ones(P, P, dtype=torch.bool, device=DEV).tril()
    eye = torch.eye(P, dtype=torch.bool, device=DEV)[None]
    T = P + max_new
    store = max_new > 0
    Kc = [torch.zeros(B, KV, T, hd, device=DEV, dtype=h.dtype) for _ in range(L)] if store else None
    Vc = [torch.zeros(B, KV, T, hd, device=DEV, dtype=h.dtype) for _ in range(L)] if store else None
    km = [torch.zeros(B, T, dtype=torch.bool, device=DEV) for _ in range(L)] if store else None
    sc = None
    work = 0.0
    act_counts = []
    for l in range(L):
        layer = model.model.layers[l]; att = layer.self_attn
        if l in sched:  # prune using scores of layer l-1, among currently active tokens
            nk = torch.ceil(sched[l] * nvalid.float()).long().clamp(min=1)
            s = sc.masked_fill(~active, float("-inf"))
            s[:, -1] = float("inf")  # last prompt token always kept
            order = s.argsort(-1, descending=True)
            rank = torch.empty_like(order); rank.scatter_(1, order, torch.arange(P, device=DEV).expand(B, P))
            active = active & (rank < nk[:, None])
            pruned_any = True
        act_counts.append(active.sum(1).float())
        x = layer.input_layernorm(h)
        q = rope(att.q_proj(x).view(B, P, H, hd).transpose(1, 2), cos, sin)
        k = rope(att.k_proj(x).view(B, P, KV, hd).transpose(1, 2), cos, sin)
        v = att.v_proj(x).view(B, P, KV, hd).transpose(1, 2)
        if fused and not pruned_any:
            o_ = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=att.scaling, enable_gqa=True)
        else:
            al = (causal[None] & active[:, None, :]) | eye
            o_ = eng._attend(q, k, v, al, att)
        if (l + 1) in sched:  # s_i^l: last token's attention over active keys, mean over heads
            lg = torch.einsum("bhd,bhkd->bhk", q[:, :, -1].float(),
                              k.repeat_interleave(H // KV, 1).float()) * att.scaling
            sc = lg.masked_fill(~active[:, None, :], float("-inf")).softmax(-1).mean(1)  # [B,P]
        a = h + att.o_proj(o_.transpose(1, 2).reshape(B, P, H * hd))
        h = a + layer.mlp(layer.post_attention_layernorm(a))
        if store:
            Kc[l][:, :, :P] = k; Vc[l][:, :, :P] = v; km[l][:, :P] = active
    logits = model.lm_head(model.model.norm(h[:, -1:] if last_only else h))
    ac = torch.stack(act_counts, 0)  # [L,B]
    eng.lazy_info = {"saving": float(1 - ac.sum() / (nvalid.float().sum() * L)),
                     "active_frac_per_layer": (ac.sum(1) / nvalid.float().sum()).tolist()}
    eng._mode, eng._j = "exact", 999
    return logits, (Kc, Vc, km, pos[:, -1] + 1, P)

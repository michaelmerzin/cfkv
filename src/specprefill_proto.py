"""SpecPrefill baseline (Liu et al. 2025, arXiv 2502.02789, ICML'25) in Engine style.

Official code: github.com/Jingyu6/speculative_prefill (MIT), a vLLM-0.6.3 monkey patch pinned to torch 2.4 /
transformers 4.50 and hard-wired to LlamaForCausalLM + Llama-3 EOS ids -> not runnable with our torch 2.7 /
transformers 5.x stack. This file re-implements its token selection 1:1
(speculative_prefill/vllm_patch/worker/look_ahead_spec_worker.py):
  1. the draft (same tokenizer) prefills the full prompt, then greedily decodes look_ahead-1 more tokens;
     queries = last prompt token + the generated tokens (look_ahead in total; truncated after an EOS);
  2. per layer/head: softmax(q K_ctx^T / sqrt(hd)) over the PROMPT keys only -> [L*H, c, S];
  3. avg_pool1d over the sequence (kernel 13, stride 1, pad 6); max over layers*heads; mean over look-ahead queries;
  4. chunked keep: split the prompt into chunks of 32, chunk score = mean, keep ceil(n_chunks*keep) best chunks;
  5. the target prefills only the kept tokens WITH THEIR ORIGINAL POSITION IDS; decoding continues at the original
     prompt length.
Deviation (documented): the last prompt token is always kept (the official code relies on it being selected; if it
were dropped the first answer token would be predicted from a different token).

Saving (target layer-token work skipped, net of the draft):
    S_net = (1 - n_kept/P) - c_draft * (P + look_ahead)/P,  c_draft = N_nonemb(draft) / N_nonemb(target)
"""
import math
import torch
import torch.nn.functional as F
from engine import DEV, rope, left_pad


def nonemb_params(model):
    return sum(p.numel() for n, p in model.named_parameters() if "embed_tokens" not in n and "lm_head" not in n)


def spec_keep_for(target_saving, c_draft, P=None, look_ahead=8):
    """keep fraction giving S_net == target_saving (P: typical prompt length; None ignores the look-ahead term)."""
    la = 0.0 if P is None else look_ahead / P
    return max(0.0, 1 - target_saving - c_draft * (1 + la))


@torch.no_grad()
def forward_pos(eng, ids, m, pos, max_new=0, last_only=False, next_pos=None, record_q=False):
    """Exact prefill of a left-padded batch with EXPLICIT position ids pos [B,P] (mirrors Engine.prefill 'exact' op
    for op). Returns (logits, cache[, (qlast per layer [B,H,hd], K per layer [B,KV,P,hd])])."""
    model, L, H, KV, hd = eng.model, eng.L, eng.H, eng.KV, eng.hd
    B, P = ids.shape
    h = model.model.embed_tokens(ids)
    cos, sin = model.model.rotary_emb(h, pos)
    fused = B == 1 and bool(m.all())
    if not fused:
        causal = torch.ones(P, P, dtype=torch.bool, device=DEV).tril()
        allowed = (causal[None] & m[:, None, :]) | torch.eye(P, dtype=torch.bool, device=DEV)[None]
    T = P + max_new
    store = max_new > 0
    Kc = [torch.zeros(B, KV, T, hd, device=DEV, dtype=h.dtype) for _ in range(L)] if store else None
    Vc = [torch.zeros(B, KV, T, hd, device=DEV, dtype=h.dtype) for _ in range(L)] if store else None
    km = [torch.zeros(B, T, dtype=torch.bool, device=DEV) for _ in range(L)] if store else None
    qs, ks = [], []
    for l in range(L):
        layer = model.model.layers[l]; att = layer.self_attn
        x = layer.input_layernorm(h)
        q = rope(att.q_proj(x).view(B, P, H, hd).transpose(1, 2), cos, sin)
        k = rope(att.k_proj(x).view(B, P, KV, hd).transpose(1, 2), cos, sin)
        v = att.v_proj(x).view(B, P, KV, hd).transpose(1, 2)
        if record_q:
            qs.append(q[:, :, -1]); ks.append(k)
        if fused:
            o_ = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=att.scaling, enable_gqa=True)
        else:
            o_ = eng._attend(q, k, v, allowed, att)
        a = h + att.o_proj(o_.transpose(1, 2).reshape(B, P, H * hd))
        h = a + layer.mlp(layer.post_attention_layernorm(a))
        if store:
            Kc[l][:, :, :P] = k; Vc[l][:, :, :P] = v; km[l][:, :P] = m
    logits = model.lm_head(model.model.norm(h[:, -1:] if last_only else h))
    npos = pos[:, -1] + 1 if next_pos is None else next_pos
    eng._mode, eng._j = "exact", 999
    cache = (Kc, Vc, km, npos, P)
    return (logits, cache, (qs, ks)) if record_q else (logits, cache)


@torch.no_grad()
def _lookahead_queries(deng, tok0, cache, n, stop_ids):
    """Greedy draft decode of n tokens after the prompt, recording each new token's per-layer query [B,H,hd].
    (Same loop as Engine.decode; the query of token t is the one used to generate token t+1.)"""
    model, L, H, KV, hd = deng.model, deng.L, deng.H, deng.KV, deng.hd
    Kc, Vc, km, pos, t = cache
    B = tok0.shape[0]
    tok, qs, done = tok0, [], torch.isin(tok0, stop_ids)
    cnt = torch.full((B,), 1, device=DEV)  # queries so far: the last prompt token
    for step in range(n):
        h = model.model.embed_tokens(tok[:, None])
        cos, sin = model.model.rotary_emb(h, pos[:, None])
        ql = []
        for l in range(L):
            layer = model.model.layers[l]; att = layer.self_attn
            x = layer.input_layernorm(h)
            q = rope(att.q_proj(x).view(B, 1, H, hd).transpose(1, 2), cos, sin)
            k = rope(att.k_proj(x).view(B, 1, KV, hd).transpose(1, 2), cos, sin)
            v = att.v_proj(x).view(B, 1, KV, hd).transpose(1, 2)
            ql.append(q[:, :, 0])
            Kc[l][:, :, t] = k[:, :, 0]; Vc[l][:, :, t] = v[:, :, 0]; km[l][:, t] = True
            o_ = deng._attend(q, Kc[l][:, :, :t + 1], Vc[l][:, :, :t + 1], km[l][:, None, :t + 1], att)
            a = h + att.o_proj(o_.transpose(1, 2).reshape(B, 1, H * hd))
            h = a + layer.mlp(layer.post_attention_layernorm(a))
        qs.append(ql)
        cnt = cnt + (~done).long()  # official: queries up to and including the step that emitted EOS
        tok = model.lm_head(model.model.norm(h))[:, 0].argmax(-1)
        done = done | torch.isin(tok, stop_ids)
        pos = pos + 1; t += 1
    return qs, cnt


@torch.no_grad()
def spec_select(deng, ids, m, keep, look_ahead=8, pool=13, chunk=32, stop_ids=None):
    """Returns a list (per row) of kept indices into the row's VALID prompt tokens (sorted, 0-based)."""
    B, P = ids.shape
    stop_ids = stop_ids if stop_ids is not None else torch.tensor([-1], device=DEV)
    pos = (m.long().cumsum(1) - 1).clamp(min=0)
    lg, cache, (q0, K) = forward_pos(deng, ids, m, pos, max_new=look_ahead, last_only=True, record_q=True)
    qla, cnt = _lookahead_queries(deng, lg[:, -1].argmax(-1), cache, look_ahead - 1, stop_ids)
    g = deng.H // deng.KV
    out = []
    for b in range(B):
        s0 = int((~m[b]).sum()); S = P - s0; c = int(cnt[b])
        Kb = torch.stack([K[l][b, :, s0:] for l in range(deng.L)], 0).repeat_interleave(g, 1).float()  # [L,H,S,hd]
        Qb = torch.stack([torch.stack([q0[l][b]] + [qla[st][l][b] for st in range(c - 1)], 1)
                          for l in range(deng.L)], 0).float()  # [L,H,c,hd]
        att = (Qb @ Kb.transpose(-1, -2) / math.sqrt(deng.hd)).softmax(-1)  # [L,H,c,S]
        att = att.flatten(0, 1)  # [L*H, c, S]
        if pool:
            att = F.avg_pool1d(att, kernel_size=pool, padding=pool // 2, stride=1)
        ti = att.max(0)[0].mean(0)  # [S]
        if chunk:
            ch = torch.split(ti, chunk)
            cs = torch.stack([x.mean() for x in ch])
            kc = math.ceil(len(ch) * keep)
            top = cs.topk(kc).indices
            idx = torch.cat([torch.arange(i * chunk, min((i + 1) * chunk, S), device=DEV) for i in top.tolist()])
        else:
            idx = ti.topk(math.ceil(S * keep)).indices
        idx = torch.unique(torch.cat([idx, torch.tensor([S - 1], device=DEV)]))  # sorted; last token kept
        out.append(idx)
    return out


@torch.no_grad()
def spec_prefill(eng, deng, ids, m, keep, look_ahead=8, pool=13, chunk=32, max_new=0, last_only=False,
                 stop_ids=None, pad_id=0):
    """Drop-in for Engine.prefill: (logits, cache) for eng.decode. keep>=1 -> exact prefill (no draft call).
    Sets eng.spec_info = {"kept_frac", "c_draft", "saving_gross", "saving_net"}."""
    B, P = ids.shape
    nvalid = m.sum(1)
    if keep >= 1:
        sel = [torch.arange(int(n), device=DEV) for n in nvalid]
        c_draft = 0.0
    else:
        sel = spec_select(deng, ids, m, keep, look_ahead, pool, chunk, stop_ids)
        c_draft = nonemb_params(deng.model) / nonemb_params(eng.model)
    rows, prow = [], []
    for b in range(B):
        s0 = P - int(nvalid[b])
        rows.append(ids[b, s0:][sel[b]].tolist()); prow.append(sel[b].tolist())
    ids2, m2 = left_pad(rows, pad_id)
    pos2 = torch.zeros_like(ids2)
    for b, pr in enumerate(prow):
        pos2[b, ids2.shape[1] - len(pr):] = torch.tensor(pr, device=DEV)
    lg, cache = forward_pos(eng, ids2, m2, pos2, max_new=max_new, last_only=last_only, next_pos=nvalid.long())
    kept = sum(len(r) for r in rows) / float(nvalid.sum())
    draft = c_draft * float(nvalid.sum() + (look_ahead * B if keep < 1 else 0)) / float(nvalid.sum())
    eng.spec_info = {"kept_frac": kept, "c_draft": c_draft, "saving_gross": 1 - kept, "saving_net": 1 - kept - draft}
    return lg, cache

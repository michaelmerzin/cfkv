"""Closed-form KV prediction engine (model-agnostic for Qwen2 / Llama decoder layers).

Semantics (deployment): a prompt of length p. Context positions C = all prompt positions except
the first valid one (attention sink), the last one (produces the first answer token) and an optional
exact recent window of w tokens before the last. For layers l >= j, the K/V of C come from
  exact   : the model's own (reference)
  predict : [rms(h_{f}) for f in feat_layers, 1] @ W   (ridge, closed form)
  naive   : layer l's own W_K/W_V applied to input_layernorm_l(h_j)   (= POP, arXiv 2602.03295)
  drop    : C removed from layers >= j (eviction)
Everything else (lower layers, sink, last prompt token, window, all generated tokens) is exact.
For accuracy evaluation we compute upper layers for all prompt positions; hidden states at C above j
are never read (their K/V are replaced), so outputs equal the compute-saving implementation.
"""
import math, os, time
import torch
import torch.nn.functional as F

DEV = "cuda"
ROOT = os.environ.get("CFKV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HUB = os.path.expanduser(os.environ.get("CFKV_HF_HUB", "~/.cache/huggingface/hub"))   # directory holding the Hugging Face model cache
PHUB = f"{ROOT}/cache/hf/hub"


def snap(base, name):
    d = f"{base}/{name}/snapshots"
    return f"{d}/{sorted(os.listdir(d))[0]}"


MODELS = {
    "qwen1.5b": lambda: snap(HUB, "models--Qwen--Qwen2.5-1.5B-Instruct"),
    "qwen7b": lambda: snap(HUB, "models--Qwen--Qwen2.5-7B-Instruct"),
    "llama8b": lambda: snap(PHUB, "models--unsloth--Llama-3.1-8B-Instruct"),
    "swiftkv8b": lambda: snap(PHUB, "models--Snowflake--Llama-3.1-SwiftKV-8B-Instruct"),
    "mistral7b": lambda: snap(PHUB, "models--unsloth--mistral-7b-instruct-v0.3"),
    "mistral24b": lambda: snap(PHUB, "models--mistralai--Mistral-Small-24B-Instruct-2501"),
}


def load_swiftkv():
    """Load Snowflake's SwiftKV checkpoint into a plain LlamaForCausalLM + its extra swiftkv weights.
    Semantics (ArcticTraining modeling_llama_swiftkv.py): for layers l >= num_key_value_layers, K/V of ALL
    tokens = {k,v}_proj_swiftkv(norm_swiftkv(h_j)); queries = q_proj_swiftkv(input_layernorm_l(h_l))."""
    import json, glob
    from safetensors import safe_open
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    path = MODELS["swiftkv8b"]()
    cfg = json.load(open(f"{path}/config.json"))
    j = cfg["num_key_value_layers"]; assert cfg.get("key_value_group_size", 1) == 1
    # same architecture as Llama-3.1-8B-Instruct: reuse its config (building LlamaConfig from kwargs trips
    # transformers-5 rope standardization); assert every architectural field matches SwiftKV's config
    config = LlamaConfig.from_pretrained(MODELS["llama8b"]())
    for k in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "head_dim", "rms_norm_eps", "vocab_size", "tie_word_embeddings"):
        assert getattr(config, k) == cfg[k], (k, getattr(config, k), cfg[k])
    rp = getattr(config, "rope_parameters", None) or {}
    assert rp.get("rope_theta", cfg["rope_theta"]) == cfg["rope_theta"], (rp, cfg["rope_theta"])
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(path)  # AutoTokenizer would try to resolve 'llama_swiftkv'
    model = LlamaForCausalLM.from_pretrained(path, config=config, dtype=torch.bfloat16, device_map=DEV,
                                             attn_implementation="sdpa").eval()
    model.requires_grad_(False); torch.set_grad_enabled(False)
    extra = {}
    for f in glob.glob(f"{path}/*.safetensors"):
        with safe_open(f, "pt", device="cuda") as sf:
            for k in sf.keys():
                if "swiftkv" in k:
                    extra[k] = sf.get_tensor(k).to(torch.bfloat16)
    import torch.nn as nn
    norm = type(model.model.norm)(config.hidden_size, eps=config.rms_norm_eps).to(DEV, torch.bfloat16)
    norm.weight.copy_(extra["model.norm_swiftkv.weight"])
    L = config.num_hidden_layers
    sw = {"j": j, "norm": norm,
          "q": {l: extra[f"model.layers.{l}.self_attn.q_proj_swiftkv.weight"] for l in range(j, L)},
          "k": {l: extra[f"model.layers.{l}.self_attn.k_proj_swiftkv.weight"] for l in range(j, L)},
          "v": {l: extra[f"model.layers.{l}.self_attn.v_proj_swiftkv.weight"] for l in range(j, L)}}
    return tok, model, sw


def load(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = MODELS[name]()
    # Mistral-Small-24B ships a Tekken tokenizer whose pre-tokenizer regex is wrong in tokenizer.json; transformers fixes it on
    # request (the other models' tokenizers have no such regex)
    tok = AutoTokenizer.from_pretrained(path, **({"fix_mistral_regex": True} if name == "mistral24b" else {}))
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=DEV,
                                                 attn_implementation="sdpa").eval()
    model.requires_grad_(False)
    torch.set_grad_enabled(False)
    return tok, model


def dims(model):
    a = model.model.layers[0].self_attn
    c = model.config
    return c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, a.head_dim


def rot(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)


def rope(x, cos, sin):  # x [B,h,T,d]; cos,sin [B,T,d]
    return x * cos[:, None] + rot(x) * sin[:, None]


def rms(h):
    h = h.float()
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)


def kv_pre(layer, h):
    x = layer.input_layernorm(h)
    return layer.self_attn.k_proj(x), layer.self_attn.v_proj(x)


def _shift(x, k, m):
    """x [B,P,d] (or [P,d]); value from position i-k (zero if before sequence start / pad)."""
    squeeze = x.dim() == 2
    if squeeze:
        x = x[None]; m = torch.ones(1, x.shape[1], dtype=torch.bool, device=x.device) if m is None else m[None]
    x = x * m[..., None]
    y = torch.zeros_like(x); y[:, k:] = x[:, :-k]
    return y[0] if squeeze else y


def _causal_mean(x, n, m):
    squeeze = x.dim() == 2
    if squeeze:
        x = x[None]; m = torch.ones(1, x.shape[1], dtype=torch.bool, device=x.device) if m is None else m[None]
    x = x * m[..., None]
    c = x.cumsum(1); cnt = m.float().cumsum(1)
    cs = c - torch.cat([torch.zeros_like(c[:, :n]), c[:, :-n]], 1) if c.shape[1] > n else c
    cn = cnt - torch.cat([torch.zeros_like(cnt[:, :n]), cnt[:, :-n]], 1) if cnt.shape[1] > n else cnt
    y = cs / cn.clamp(min=1)[..., None]
    return y[0] if squeeze else y


def features(hs, feat_layers, m=None):
    """hs: dict layer->[..., d] hidden (input to that layer). 'ctx' in feat_layers adds context
    features of the FIRST listed layer: its state at i-1, i-2 and the causal mean over the last 8."""
    fl = [f for f in feat_layers if not isinstance(f, str)]
    parts = [rms(hs[f]) for f in fl]
    if "ctx" in feat_layers:
        u = parts[0]
        parts += [_shift(u, 1, m), _shift(u, 2, m), _causal_mean(u, 8, m)]
    elif "ctx1" in feat_layers:   # memory-light variant: previous token's state only
        parts += [_shift(parts[0], 1, m)]
    one = torch.ones(*parts[0].shape[:-1], 1, device=parts[0].device)
    return torch.cat(parts + [one], -1)


# ---------------------------------------------------------------- fitting
def fit_grams(model, seqs, specs, c_dtype=torch.float64, sq=None):
    """seqs: list of token-id lists (every position except 0 is a context sample).
    specs: list of (j, feat_layers). Returns {spec: (G, C, n)}; G always fp64, C in c_dtype
    (fp32 halves the memory of the large cross-moment matrix for big models).
    sq: optional dict, filled with {j: per-column sum of squared targets} (for variance matching)."""
    L, H, KV, hd = dims(model)
    layers = model.model.layers
    acc = {s: [None, None, 0] for s in specs}
    for s in seqs:
        ids = torch.tensor([s], device=DEV)
        hs = model.model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
        sl = slice(1, len(s))
        tgt_cache = {}
        for spec in specs:
            j, fl = spec
            X = features({f: hs[f][0] for f in fl if not isinstance(f, str)}, fl)[sl].double()
            Ys = []
            for l in range(j, L):
                if l not in tgt_cache:
                    tgt_cache[l] = torch.cat(kv_pre(layers[l], hs[l][0, sl]), -1)
                Ys.append(tgt_cache[l])
            Y = torch.cat(Ys, -1).to(c_dtype)   # fp32 path never materializes an fp64 [F, out] temporary
            if sq is not None and (j, id(s)) not in sq.setdefault("_done", set()):
                sq["_done"].add((j, id(s)))
                y2 = Y.double().pow(2).sum(0) if c_dtype == torch.float64 else Y.float().pow(2).sum(0).double()
                sq[j] = y2 if j not in sq else sq[j] + y2
            a = acc[spec]
            a[0] = X.T @ X if a[0] is None else a[0] + X.T @ X
            XY = X.to(c_dtype).T @ Y
            a[1] = XY if a[1] is None else a[1].add_(XY)
            del XY
            a[2] += len(s) - 1
    return acc


def solve(G, C, alpha, per_feature=False):
    """Ridge. Default: lambda = alpha * mean diag(G) on every feature. per_feature: lambda_i = alpha * G_ii
    (ridge on standardized features), which a few massive-activation dimensions cannot inflate."""
    d = torch.diagonal(G)
    reg = alpha * d if per_feature else alpha * d.mean() * torch.ones_like(d)
    A = G + torch.diag(reg)
    Lc = torch.linalg.cholesky(A)
    del A
    if C.dtype == G.dtype:
        return torch.cholesky_solve(C, Lc).float()
    # C kept in lower precision: solve column blocks in fp64 to bound peak memory
    out = torch.empty(C.shape, device=C.device, dtype=torch.float32)
    for i in range(0, C.shape[1], 8192):
        out[:, i:i + 8192] = torch.cholesky_solve(C[:, i:i + 8192].double(), Lc).float()
    return out


# ---------------------------------------------------------------- batched prefill + decode
def left_pad(seqs, pad_id):
    P = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), P), pad_id, dtype=torch.long)
    m = torch.zeros(len(seqs), P, dtype=torch.bool)
    for b, s in enumerate(seqs):
        ids[b, P - len(s):] = torch.tensor(s); m[b, P - len(s):] = True
    return ids.to(DEV), m.to(DEV)


def context_mask(m, window=0, ctx_end=None):
    """m [B,P] valid (left padded). Context = valid, not first valid, index < end-1-window.
    ctx_end: optional per-row exclusive end index (for teacher-forced evaluation); default P."""
    B, P = m.shape
    idx = torch.arange(P, device=m.device)[None].expand(B, P)
    first = (~m).sum(1, keepdim=True)  # index of first valid
    end = torch.full((B, 1), P, device=m.device) if ctx_end is None else ctx_end[:, None]
    return m & (idx > first) & (idx < end - 1 - window)


class Engine:
    def __init__(self, model, tok, swift=None):
        self.model, self.tok, self.swift = model, tok, swift
        self.L, self.H, self.KV, self.hd = dims(model)

    def _attend(self, q, K, V, allowed, att):
        g = self.H // self.KV
        o = F.scaled_dot_product_attention(q, K.repeat_interleave(g, 1), V.repeat_interleave(g, 1),
                                           attn_mask=allowed[:, None], scale=att.scaling)
        return o

    def prefill(self, ids, m, ctx, j, mode, W=None, feat_layers=None, max_new=0, keep=0, obs=8, last_only=False,
                kv_src=None, pool=0, win=0, after=0):
        """Returns logits [B,P,V] and cache (K,V per layer preallocated to P+max_new, keymask)."""
        model, L, H, KV, hd = self.model, self.L, self.H, self.KV, self.hd
        B, P = ids.shape
        pos = (m.long().cumsum(1) - 1).clamp(min=0)
        h = model.model.embed_tokens(ids)
        cos, sin = model.model.rotary_emb(h, pos)
        fused = B == 1 and bool(m.all())  # unpadded single sequence: fused causal kernel, no PxP mask
        if not fused:
            causal = torch.ones(P, P, dtype=torch.bool, device=DEV).tril()
            allowed = causal[None] & m[:, None, :]
            allowed = allowed | torch.eye(P, dtype=torch.bool, device=DEV)[None]  # pads attend to self (no NaN)
        else:
            allowed = None
        T = P + max_new
        store_cache = max_new > 0  # prefill-only calls (MMLU, selection) never decode: skip the cache
        Kc = [torch.zeros(B, KV, T, hd, device=DEV, dtype=h.dtype) for _ in range(L)] if store_cache else None
        Vc = [torch.zeros(B, KV, T, hd, device=DEV, dtype=h.dtype) for _ in range(L)] if store_cache else None
        km = [torch.zeros(B, T, dtype=torch.bool, device=DEV) for _ in range(L)] if store_cache else None
        hist = {}
        pred = None
        for l in range(L):
            layer = model.model.layers[l]; att = layer.self_attn
            if feat_layers is not None and l in feat_layers:
                hist[l] = h
            if l == j and keep > 0 and mode != "exact":
                # SnapKV-style importance at layer j: last `obs` queries' attention to all keys
                x0 = layer.input_layernorm(h)
                q0 = rope(att.q_proj(x0[:, -obs:]).view(B, -1, H, hd).transpose(1, 2), cos[:, -obs:], sin[:, -obs:])
                k0 = rope(att.k_proj(x0).view(B, P, KV, hd).transpose(1, 2), cos, sin)
                sc = torch.einsum("bhqd,bhkd->bhqk", q0, k0.repeat_interleave(H // KV, 1)).float() * att.scaling
                if pool > 0 or win > 0:   # official-FastKV scoring: causal mask inside the observation window
                    kp = torch.arange(P, device=DEV)[None, :]; qp = (P - obs + torch.arange(obs, device=DEV))[:, None]
                    sc = sc.masked_fill((kp > qp)[None, None], float("-inf"))
                sc = sc.masked_fill(~m[:, None, None, :], float("-inf")).softmax(-1).mean((1, 2))  # [B,P]
                if pool > 0:   # average pooling over key positions (kernel `pool`, zero padding, stride 1)
                    sc = F.avg_pool1d(sc[:, None], kernel_size=pool, stride=1, padding=pool // 2)[:, 0, :P]
                sc = sc.masked_fill(~ctx, -1.0)
                if win > 0:    # the last `win` prompt tokens are always kept (inside the budget)
                    wm = torch.zeros(P, dtype=torch.bool, device=DEV); wm[P - win:] = True
                    sc = torch.where(wm[None] & ctx, torch.full_like(sc, 1e9), sc)
                if keep < 1:   # fractional budget: keep ceil(keep * #context tokens) per row
                    kr = (ctx.sum(1).float() * keep).ceil().long()
                    order = sc.argsort(dim=-1, descending=True)
                    rank = torch.empty_like(order); rank.scatter_(1, order, torch.arange(P, device=DEV).expand(B, P))
                    ctx = ctx & ~(rank < kr[:, None])
                else:
                    top = sc.topk(min(int(keep), P), dim=-1).indices
                    ctx = ctx.clone(); ctx.scatter_(1, top, False)
                self.last_ctx = ctx
            if l == j and mode == "predict":
                hist[j] = h
                Xf = features(hist, feat_layers, m)  # [B,P,F] fp32; per-layer slices of W below
            if l == j:
                hj = h
            x = layer.input_layernorm(h)
            if mode == "swiftkv" and l >= j:
                sw = self.swift
                if l == j:
                    hsw = sw["norm"](h)
                q = rope((x @ sw["q"][l].T).view(B, P, H, hd).transpose(1, 2), cos, sin)
                k = hsw @ sw["k"][l].T; v = hsw @ sw["v"][l].T
            else:
                q = rope(att.q_proj(x).view(B, P, H, hd).transpose(1, 2), cos, sin)
                k = att.k_proj(x); v = att.v_proj(x)
            keymask = m.clone()
            if l >= j and mode in ("predict", "naive", "scaffold"):
                if mode == "scaffold":  # exact upper-layer K/V of a same-length filler prompt (form, no content)
                    kc, vc = kv_src[l]
                elif mode == "predict":
                    o = (l - j) * 2 * KV * hd
                    kv_l = (Xf @ W[:, o:o + 2 * KV * hd]).to(h.dtype)
                    kc, vc = kv_l[..., :KV * hd], kv_l[..., KV * hd:]
                else:
                    kc, vc = kv_pre(layer, hj)
                sel = ctx[..., None]
                k = torch.where(sel, kc, k); v = torch.where(sel, vc, v)
            k = rope(k.view(B, P, KV, hd).transpose(1, 2), cos, sin)
            v = v.view(B, P, KV, hd).transpose(1, 2)
            al = allowed
            if l >= j + after and mode == "drop":   # after=1: the selection layer itself still sees every token
                keymask = m & ~ctx
                base = allowed if allowed is not None else torch.ones(P, P, dtype=torch.bool, device=DEV).tril()[None]
                al = (base & keymask[:, None, :]) | torch.eye(P, dtype=torch.bool, device=DEV)[None]
            if al is None:
                o_ = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=att.scaling, enable_gqa=True)
            else:
                o_ = self._attend(q, k, v, al, att)
            a = h + att.o_proj(o_.transpose(1, 2).reshape(B, P, H * hd))
            h = a + layer.mlp(layer.post_attention_layernorm(a))
            if store_cache:
                Kc[l][:, :, :P] = k; Vc[l][:, :, :P] = v; km[l][:, :P] = keymask
        logits = model.lm_head(model.model.norm(h[:, -1:] if last_only else h))
        self._mode, self._j = mode, j
        return logits, (Kc, Vc, km, pos[:, -1] + 1, P)

    @torch.no_grad()
    def scaffold_kv(self, ids, m, j, template_empty=None, filler_ids=None):
        """Control from 'Probing the Prompt KV Cache' (2605.30574): replace each prompt's content tokens by neutral
        filler of the same length, keeping the template (common prefix/suffix with the empty-content template),
        run the exact model on it, and return pre-RoPE K/V of every upper layer {l: (k, v)} [B,P,KV*hd]."""
        B, P = ids.shape
        fids = ids.clone()
        for b in range(B):
            valid = m[b].nonzero()[:, 0]
            s0 = int(valid[0]); seq = ids[b, s0:].tolist()
            pl = sl = 0
            if template_empty is not None:
                while pl < min(len(seq), len(template_empty)) and seq[pl] == template_empty[pl]:
                    pl += 1
                while (sl < min(len(seq) - pl, len(template_empty) - pl)
                       and seq[-1 - sl] == template_empty[-1 - sl]):
                    sl += 1
            else:
                pl = 1 if len(seq) > 1 else 0
            n = len(seq) - pl - sl
            fill = (filler_ids * (n // len(filler_ids) + 1))[:n]
            fids[b, s0 + pl:s0 + pl + n] = torch.tensor(fill, device=DEV)
        # run the exact model on the filler prompt with the engine's own position ids / masks
        out = {}
        h = self.model.model.embed_tokens(fids)
        pos = (m.long().cumsum(1) - 1).clamp(min=0)
        cos, sin = self.model.model.rotary_emb(h, pos)
        causal = torch.ones(P, P, dtype=torch.bool, device=DEV).tril()
        allowed = (causal[None] & m[:, None, :]) | torch.eye(P, dtype=torch.bool, device=DEV)[None]
        L, H, KV, hd = self.L, self.H, self.KV, self.hd
        for l in range(L):
            layer = self.model.model.layers[l]; att = layer.self_attn
            x = layer.input_layernorm(h)
            if l >= j:
                out[l] = (att.k_proj(x), att.v_proj(x))
            q = rope(att.q_proj(x).view(B, P, H, hd).transpose(1, 2), cos, sin)
            k = rope(att.k_proj(x).view(B, P, KV, hd).transpose(1, 2), cos, sin)
            v = att.v_proj(x).view(B, P, KV, hd).transpose(1, 2)
            o_ = self._attend(q, k, v, allowed, att)
            a = h + att.o_proj(o_.transpose(1, 2).reshape(B, P, H * hd))
            h = a + layer.mlp(layer.post_attention_layernorm(a))
        return out

    @torch.no_grad()
    def gemfilter_select(self, ids, r, k):
        """GemFilter (Shi et al., 2024): run layers 0..r-1, score prompt tokens by the last query's attention at
        layer r (mean over heads), keep the top-k in original order (last token always kept). Batch 1, no padding."""
        model, H, KV, hd = self.model, self.H, self.KV, self.hd
        B, P = ids.shape
        assert B == 1
        if P <= k:
            return ids
        pos = torch.arange(P, device=DEV)[None]
        h = model.model.embed_tokens(ids)
        cos, sin = model.model.rotary_emb(h, pos)
        for l in range(r):
            layer = model.model.layers[l]; att = layer.self_attn
            x = layer.input_layernorm(h)
            q = rope(att.q_proj(x).view(1, P, H, hd).transpose(1, 2), cos, sin)
            kk = rope(att.k_proj(x).view(1, P, KV, hd).transpose(1, 2), cos, sin)
            v = att.v_proj(x).view(1, P, KV, hd).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, kk, v, is_causal=True, scale=att.scaling, enable_gqa=True)
            a = h + att.o_proj(o.transpose(1, 2).reshape(1, P, H * hd))
            h = a + layer.mlp(layer.post_attention_layernorm(a))
        layer = model.model.layers[r]; att = layer.self_attn
        x = layer.input_layernorm(h)
        q = rope(att.q_proj(x[:, -1:]).view(1, 1, H, hd).transpose(1, 2), cos[:, -1:], sin[:, -1:])
        kk = rope(att.k_proj(x).view(1, P, KV, hd).transpose(1, 2), cos, sin).repeat_interleave(H // KV, 1)
        sc = (torch.einsum("bhqd,bhkd->bhqk", q, kk).float() * att.scaling).softmax(-1).mean((1, 2))[0]  # [P]
        top = sc.topk(k).indices
        top = torch.unique(torch.cat([top, torch.tensor([P - 1], device=DEV)]))  # sorted, last token kept
        return ids[:, top]

    @torch.no_grad()
    def extend(self, tokens, cache):
        """Teacher-force a list of new tokens (batch 1) through all layers on top of a prefill cache, exactly as
        decoding would; returns the logits after the last token and the updated cache. Used for follow-up turns
        that reuse a cached prefix."""
        model, L, H, KV, hd = self.model, self.L, self.H, self.KV, self.hd
        Kc, Vc, km, pos, t = cache
        swm = getattr(self, "_mode", None) == "swiftkv"
        logits = None
        for tk in tokens:
            h = model.model.embed_tokens(torch.tensor([[tk]], device=DEV))
            cos, sin = model.model.rotary_emb(h, pos[:, None])
            for l in range(L):
                layer = model.model.layers[l]; att = layer.self_attn
                x = layer.input_layernorm(h)
                q = rope(att.q_proj(x).view(1, 1, H, hd).transpose(1, 2), cos, sin)
                k = rope(att.k_proj(x).view(1, 1, KV, hd).transpose(1, 2), cos, sin)
                v = att.v_proj(x).view(1, 1, KV, hd).transpose(1, 2)
                Kc[l][:, :, t] = k[:, :, 0]; Vc[l][:, :, t] = v[:, :, 0]; km[l][:, t] = True
                o_ = self._attend(q, Kc[l][:, :, :t + 1], Vc[l][:, :, :t + 1], km[l][:, None, :t + 1], att)
                a = h + att.o_proj(o_.transpose(1, 2).reshape(1, 1, H * hd))
                h = a + layer.mlp(layer.post_attention_layernorm(a))
            logits = model.lm_head(model.model.norm(h))[:, 0]
            pos = pos + 1; t += 1
        return logits, (Kc, Vc, km, pos, t)

    def decode(self, first_tok, cache, max_new, stop_ids):
        model, L, H, KV, hd = self.model, self.L, self.H, self.KV, self.hd
        Kc, Vc, km, pos, t = cache
        B = first_tok.shape[0]
        out = [first_tok]
        done = torch.isin(first_tok, stop_ids)
        tok = first_tok
        for step in range(max_new - 1):
            h = model.model.embed_tokens(tok[:, None])
            cos, sin = model.model.rotary_emb(h, pos[:, None])
            swm = getattr(self, "_mode", None) == "swiftkv"
            for l in range(L):
                layer = model.model.layers[l]; att = layer.self_attn
                x = layer.input_layernorm(h)
                if swm and l >= self._j:
                    sw = self.swift
                    if l == self._j:
                        hsw = sw["norm"](h)
                    q = rope((x @ sw["q"][l].T).view(B, 1, H, hd).transpose(1, 2), cos, sin)
                    k = rope((hsw @ sw["k"][l].T).view(B, 1, KV, hd).transpose(1, 2), cos, sin)
                    v = (hsw @ sw["v"][l].T).view(B, 1, KV, hd).transpose(1, 2)
                else:
                    q = rope(att.q_proj(x).view(B, 1, H, hd).transpose(1, 2), cos, sin)
                    k = rope(att.k_proj(x).view(B, 1, KV, hd).transpose(1, 2), cos, sin)
                    v = att.v_proj(x).view(B, 1, KV, hd).transpose(1, 2)
                Kc[l][:, :, t] = k[:, :, 0]; Vc[l][:, :, t] = v[:, :, 0]; km[l][:, t] = True
                o_ = self._attend(q, Kc[l][:, :, :t + 1], Vc[l][:, :, :t + 1], km[l][:, None, :t + 1], att)
                a = h + att.o_proj(o_.transpose(1, 2).reshape(B, 1, H * hd))
                h = a + layer.mlp(layer.post_attention_layernorm(a))
            tok = model.lm_head(model.model.norm(h))[:, 0].argmax(-1)
            tok = torch.where(done, torch.full_like(tok, int(stop_ids[0])), tok)
            out.append(tok)
            done |= torch.isin(tok, stop_ids)
            pos = pos + 1; t += 1
            if bool(done.all()):
                break
        return torch.stack(out, 1)

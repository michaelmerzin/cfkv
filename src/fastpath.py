"""Compute-saving prefill (batch 1) + time-to-first-token benchmark.

prefill_fast: lower layers 0..j-1 on all P tokens (causal sdpa); at layer j one matmul predicts all
upper-layer pre-RoPE K/V for context tokens; upper layers run ONLY for the exact set
E = {first token} ∪ {window} ∪ {last token}. Returns last-token logits and the per-layer KV cache.
Correctness: its last-token logits must equal Engine.prefill(mode="predict") (same math, different
schedule). Timing: CUDA events, warmup, median over repeats; compared with the exact HF forward.
"""
import argparse, json, os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from engine import DEV, ROOT, load, dims, rope, features, kv_pre, Engine, fit_grams, solve, left_pad, context_mask


@torch.no_grad()
def prefill_fast(model, ids, j, W, feat_layers, window=0):
    L, H, KV, hd = dims(model)
    P = ids.shape[1]
    layers = model.model.layers
    pos = torch.arange(P, device=DEV)[None]
    h = model.model.embed_tokens(ids)
    cos, sin = model.model.rotary_emb(h, pos)
    Ks, Vs = [], []
    hist = {}
    g = H // KV
    for l in range(j):
        if l in feat_layers:
            hist[l] = h
        layer = layers[l]; att = layer.self_attn
        x = layer.input_layernorm(h)
        q = rope(att.q_proj(x).view(1, P, H, hd).transpose(1, 2), cos, sin)
        k = rope(att.k_proj(x).view(1, P, KV, hd).transpose(1, 2), cos, sin)
        v = att.v_proj(x).view(1, P, KV, hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=att.scaling, enable_gqa=True)
        a = h + att.o_proj(o.transpose(1, 2).reshape(1, P, H * hd))
        h = a + layer.mlp(layer.post_attention_layernorm(a))
        Ks.append(k); Vs.append(v)
    hist[j] = h
    # exact set E and context set C
    E = torch.cat([torch.zeros(1, dtype=torch.long, device=DEV),
                   torch.arange(max(1, P - 1 - window), P, device=DEV)]).unique()
    Cm = torch.ones(P, dtype=torch.bool, device=DEV); Cm[E] = False
    C = Cm.nonzero()[:, 0]
    X = features({f: hist[f][0] for f in feat_layers}, feat_layers)[C].to(h.dtype)
    pred = X @ (W if W.dtype == h.dtype else W.to(h.dtype))  # bf16 matmul, fp32 accumulate: [|C|, out]
    hE = h[:, E]
    cE, sE = cos[:, E], sin[:, E]
    cC, sC = cos[:, C], sin[:, C]
    allowed_E = torch.cat([C, E])[None, :] <= E[:, None]  # [|E|, |C|+|E|] causal by position
    for l in range(j, L):
        layer = layers[l]; att = layer.self_attn
        off = (l - j) * 2 * KV * hd
        kc = rope(pred[:, off:off + KV * hd].reshape(1, -1, KV, hd).transpose(1, 2), cC, sC)
        vc = pred[:, off + KV * hd:off + 2 * KV * hd].reshape(1, -1, KV, hd).transpose(1, 2)
        x = layer.input_layernorm(hE)
        q = rope(att.q_proj(x).view(1, -1, H, hd).transpose(1, 2), cE, sE)
        ke = rope(att.k_proj(x).view(1, -1, KV, hd).transpose(1, 2), cE, sE)
        ve = att.v_proj(x).view(1, -1, KV, hd).transpose(1, 2)
        K, V = torch.cat([kc, ke], 2), torch.cat([vc, ve], 2)
        o = F.scaled_dot_product_attention(q, K, V, attn_mask=allowed_E[None, None], scale=att.scaling,
                                           enable_gqa=True)
        a = hE + att.o_proj(o.transpose(1, 2).reshape(1, -1, H * hd))
        hE = a + layer.mlp(layer.post_attention_layernorm(a))
        Ks.append(K); Vs.append(V)  # cache order: [C..., E...] (positions tracked by caller)
    return model.lm_head(model.model.norm(hE[:, -1:])), (Ks, Vs, C, E)


@torch.no_grad()
def prefill_fast2(model, ids, j, W, feat_layers, window=0):
    """Same math as prefill_fast, faster schedule for the exact set. Keys are kept in position order
    [sink, context..., window..., last], so the window/last queries attend with a fused lower-right causal
    mask (no explicit mask, no gather); the sink query attends only to itself (output = its own value)."""
    from torch.nn.attention.bias import causal_lower_right
    L, H, KV, hd = dims(model)
    P = ids.shape[1]
    layers = model.model.layers
    pos = torch.arange(P, device=DEV)[None]
    h = model.model.embed_tokens(ids)
    cos, sin = model.model.rotary_emb(h, pos)
    hist = {}
    for l in range(j):
        if l in feat_layers:
            hist[l] = h
        layer = layers[l]; att = layer.self_attn
        x = layer.input_layernorm(h)
        q = rope(att.q_proj(x).view(1, P, H, hd).transpose(1, 2), cos, sin)
        k = rope(att.k_proj(x).view(1, P, KV, hd).transpose(1, 2), cos, sin)
        v = att.v_proj(x).view(1, P, KV, hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=att.scaling, enable_gqa=True)
        a = h + att.o_proj(o.transpose(1, 2).reshape(1, P, H * hd))
        h = a + layer.mlp(layer.post_attention_layernorm(a))
    hist[j] = h
    t0 = max(1, P - 1 - window)             # first exact position after the sink
    X = features({f: hist[f][0, 1:t0] for f in feat_layers}, feat_layers).to(h.dtype)   # context 1..t0-1
    Wd = W if W.dtype == h.dtype else W.to(h.dtype)
    Eidx = torch.cat([torch.zeros(1, dtype=torch.long, device=DEV), torch.arange(t0, P, device=DEV)])
    hE = h[:, Eidx]
    cE, sE = cos[:, Eidx], sin[:, Eidx]
    cC, sC = cos[:, 1:t0], sin[:, 1:t0]
    nq = P - t0                              # queries after the sink
    bias = causal_lower_right(nq, P)
    D = KV * hd
    for l in range(j, L):
        layer = layers[l]; att = layer.self_attn
        off = (l - j) * 2 * D
        kv = X @ Wd[:, off:off + 2 * D]
        kc = rope(kv[:, :D].reshape(1, -1, KV, hd).transpose(1, 2), cC, sC)
        vc = kv[:, D:].reshape(1, -1, KV, hd).transpose(1, 2)
        x = layer.input_layernorm(hE)
        q = rope(att.q_proj(x).view(1, -1, H, hd).transpose(1, 2), cE, sE)
        ke = rope(att.k_proj(x).view(1, -1, KV, hd).transpose(1, 2), cE, sE)
        ve = att.v_proj(x).view(1, -1, KV, hd).transpose(1, 2)
        K = torch.cat([ke[:, :, :1], kc, ke[:, :, 1:]], 2)   # position order
        V = torch.cat([ve[:, :, :1], vc, ve[:, :, 1:]], 2)
        g = H // KV  # CausalBias does not honour enable_gqa: expand K/V to all query heads explicitly
        o_rest = F.scaled_dot_product_attention(q[:, :, 1:], K.repeat_interleave(g, 1), V.repeat_interleave(g, 1),
                                                attn_mask=bias, scale=att.scaling)
        o_sink = ve[:, :, :1].repeat_interleave(H // KV, 1)   # softmax over a single key
        o = torch.cat([o_sink, o_rest], 2)
        a = hE + att.o_proj(o.transpose(1, 2).reshape(1, -1, H * hd))
        hE = a + layer.mlp(layer.post_attention_layernorm(a))
    return model.lm_head(model.model.norm(hE[:, -1:]))


def timeit(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen7b")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--js", default="14,19")
    ap.add_argument("--lengths", default="1024,4096,8192,16384")
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--n_calib", type=int, default=64)
    ap.add_argument("--compare_windows", type=int, default=0)
    ap.add_argument("--pop_cost", type=int, default=0,
                    help="also time a single-layer predictor (features = h^(j) only): the same matmul size as POP's per-layer "
                         "projections of h^(j), i.e. POP's prediction cost in this schedule; timed at w=0 and w=64 with CF-KV")
    args = ap.parse_args()
    run_dir = f"{ROOT}/results/raw/{args.run_id}"; assert not os.path.exists(run_dir); os.makedirs(run_dir)
    tok, model = load(args.model)
    L = dims(model)[0]
    js = [int(x) for x in args.js.split(",")]
    # small fit is enough for timing; correctness check uses the same W in both paths
    from round2 import wiki_texts, chat_pairs, chat_ids
    calib = [tok(t, add_special_tokens=False)["input_ids"][:1280] for t in
             wiki_texts("train-00000-of-00002.parquet", 40000, args.n_calib, 1)]
    specs = [(j, tuple(x for x in (j, j - 4, j - 8) if x >= 1)) for j in js]
    specs1 = [(j, (j,)) for j in js] if args.pop_cost else []
    acc = fit_grams(model, calib, specs + specs1)
    Ws32 = {j: solve(acc[s][0], acc[s][1], 1e-4) for j, s in zip(js, specs)}
    Ws = {j: w.to(torch.bfloat16) for j, w in Ws32.items()}
    Wp = {j: solve(acc[s][0], acc[s][1], 1e-4).to(torch.bfloat16) for j, s in zip(js, specs1)}
    res = {"run_id": args.run_id, "model": args.model, "L": L, "gpu": torch.cuda.get_device_name(0),
           "window": args.window, "check": {}, "ttft_ms": {}}
    # correctness: fast path == engine predict mode (last-token logits)
    eng = Engine(model, tok)
    long_text = " ".join(wiki_texts("test-00000-of-00001.parquet", None, 1, 7, chars=12000))
    ids = torch.tensor([tok(long_text, add_special_tokens=False)["input_ids"][:2000]], device=DEV)
    for j, s in zip(js, specs):
        fl = list(s[1])
        lf, _ = prefill_fast(model, ids, j, Ws[j], fl, args.window)
        m = torch.ones_like(ids, dtype=torch.bool)
        le, _ = eng.prefill(ids, m, context_mask(m, args.window), j, "predict", Ws32[j], fl)
        d = (lf[0, -1].float() - le[0, -1].float()).abs().max().item()
        same = bool(lf[0, -1].argmax() == le[0, -1].argmax())
        res["check"][j] = {"max_abs_logit_diff": d, "same_argmax": same}
        print(f"check j={j}: max|Δlogit| {d:.4f} same argmax {same}", flush=True)
        if args.compare_windows:
            for w in (0, 64):
                l2 = prefill_fast2(model, ids, j, Ws[j], fl, w)
                le2, _ = eng.prefill(ids, m, context_mask(m, w), j, "predict", Ws32[j], fl)
                d2 = (l2[0, -1].float() - le2[0, -1].float()).abs().max().item()
                s2 = bool(l2[0, -1].argmax() == le2[0, -1].argmax())
                res["check"][f"fast2_j{j}_w{w}"] = {"max_abs_logit_diff": d2, "same_argmax": s2}
                print(f"check fast2 j={j} w={w}: max|Δlogit| {d2:.4f} same argmax {s2}", flush=True)
    for P in map(int, args.lengths.split(",")):
        base = tok(" ".join(wiki_texts("train-00000-of-00002.parquet", 40000, 8, 11, chars=20000)),
                   add_special_tokens=False)["input_ids"]
        while len(base) < P:
            base = base + base
        ids = torch.tensor([base[:P]], device=DEV)
        r = {"exact_hf": timeit(lambda: model.model(input_ids=ids, use_cache=True))}
        for j, s in zip(js, specs):
            fl = list(s[1])
            r[f"fast_j{j}"] = timeit(lambda: prefill_fast(model, ids, j, Ws[j], fl, args.window))
            if args.pop_cost:
                r[f"fast_j{j}_w64"] = timeit(lambda: prefill_fast(model, ids, j, Ws[j], fl, 64))
                r[f"pop_j{j}"] = timeit(lambda: prefill_fast(model, ids, j, Wp[j], [j], 0))
                r[f"pop_j{j}_w64"] = timeit(lambda: prefill_fast(model, ids, j, Wp[j], [j], 64))
            if args.compare_windows:
                for w in (0, 64):
                    r[f"fast2_j{j}_w{w}"] = timeit(lambda: prefill_fast2(model, ids, j, Ws[j], fl, w))
                r[f"fast_j{j}_w64"] = timeit(lambda: prefill_fast(model, ids, j, Ws[j], fl, 64))
        res["ttft_ms"][P] = r
        print(f"P={P}: " + " | ".join(f"{k} {v:.1f} ms ({r['exact_hf']/v:.2f}x)" for k, v in r.items()), flush=True)
        json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)
        torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main()

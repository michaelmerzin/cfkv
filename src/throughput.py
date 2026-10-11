"""Batched prefill throughput: does the CF-KV speedup survive batching?

Equal-length prompts (no padding), batch B in {1,4,16}, P in {1024,4096}; prefill tokens/s of the standard HF
forward vs. the compute-saving CF-KV schedule (fastpath.prefill_fast generalized to a batch). Correctness: each
batch row must reproduce the batch-1 schedule's last-token logits.
"""
import argparse, json, os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from engine import DEV, ROOT, load, dims, rope, features, fit_grams, solve
from fastpath import prefill_fast, timeit
from round2 import wiki_texts


@torch.no_grad()
def prefill_fast_b(model, ids, j, W, feat_layers, window=0):
    """prefill_fast for a batch of equal-length, unpadded prompts [B,P]; returns last-token logits [B,1,V]."""
    L, H, KV, hd = dims(model)
    B, P = ids.shape
    layers = model.model.layers
    pos = torch.arange(P, device=DEV)[None].expand(B, P)
    h = model.model.embed_tokens(ids)
    cos, sin = model.model.rotary_emb(h, pos)
    hist = {}
    for l in range(j):
        if l in feat_layers:
            hist[l] = h
        layer = layers[l]; att = layer.self_attn
        x = layer.input_layernorm(h)
        q = rope(att.q_proj(x).view(B, P, H, hd).transpose(1, 2), cos, sin)
        k = rope(att.k_proj(x).view(B, P, KV, hd).transpose(1, 2), cos, sin)
        v = att.v_proj(x).view(B, P, KV, hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=att.scaling, enable_gqa=True)
        a = h + att.o_proj(o.transpose(1, 2).reshape(B, P, H * hd))
        h = a + layer.mlp(layer.post_attention_layernorm(a))
    hist[j] = h
    E = torch.cat([torch.zeros(1, dtype=torch.long, device=DEV),
                   torch.arange(max(1, P - 1 - window), P, device=DEV)]).unique()
    Cm = torch.ones(P, dtype=torch.bool, device=DEV); Cm[E] = False
    C = Cm.nonzero()[:, 0]
    X = features({f: hist[f] for f in feat_layers}, feat_layers)[:, C].to(h.dtype)
    pred = X @ (W if W.dtype == h.dtype else W.to(h.dtype))          # [B,|C|,out]
    hE = h[:, E]
    cE, sE = cos[:, E], sin[:, E]
    cC, sC = cos[:, C], sin[:, C]
    allowed_E = torch.cat([C, E])[None, :] <= E[:, None]
    for l in range(j, L):
        layer = layers[l]; att = layer.self_attn
        off = (l - j) * 2 * KV * hd
        kc = rope(pred[..., off:off + KV * hd].reshape(B, -1, KV, hd).transpose(1, 2), cC, sC)
        vc = pred[..., off + KV * hd:off + 2 * KV * hd].reshape(B, -1, KV, hd).transpose(1, 2)
        x = layer.input_layernorm(hE)
        q = rope(att.q_proj(x).view(B, -1, H, hd).transpose(1, 2), cE, sE)
        ke = rope(att.k_proj(x).view(B, -1, KV, hd).transpose(1, 2), cE, sE)
        ve = att.v_proj(x).view(B, -1, KV, hd).transpose(1, 2)
        K, V = torch.cat([kc, ke], 2), torch.cat([vc, ve], 2)
        o = F.scaled_dot_product_attention(q, K, V, attn_mask=allowed_E[None, None], scale=att.scaling, enable_gqa=True)
        a = hE + att.o_proj(o.transpose(1, 2).reshape(B, -1, H * hd))
        hE = a + layer.mlp(layer.post_attention_layernorm(a))
    return model.lm_head(model.model.norm(hE[:, -1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen7b")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--j", type=int, default=14)
    ap.add_argument("--lengths", default="1024,4096")
    ap.add_argument("--batches", default="1,4,16")
    ap.add_argument("--max_tokens", type=int, default=65536)
    args = ap.parse_args()
    run_dir = f"{ROOT}/results/raw/{args.run_id}"; assert not os.path.exists(run_dir); os.makedirs(run_dir)
    tok, model = load(args.model)
    j = args.j; fl = [x for x in (j, j - 4, j - 8) if x >= 1]
    calib = [tok(t, add_special_tokens=False)["input_ids"][:1280] for t in wiki_texts("train-00000-of-00002.parquet", 40000, 64, 1)]
    acc = fit_grams(model, calib, [(j, tuple(fl))])
    W = solve(acc[(j, tuple(fl))][0], acc[(j, tuple(fl))][1], 1e-4).to(torch.bfloat16); del acc
    texts = wiki_texts("train-00000-of-00002.parquet", 40000, 64, 11, chars=40000)
    res = {"run_id": args.run_id, "model": args.model, "j": j, "gpu": torch.cuda.get_device_name(0), "check": {}, "tput": {}}
    # correctness: batched rows == batch-1 schedule
    P0 = 1024
    ids = torch.stack([torch.tensor(tok(t, add_special_tokens=False)["input_ids"][:P0]) for t in texts[:2]]).to(DEV)
    for w in (0, 64):
        lb = prefill_fast_b(model, ids, j, W, fl, w)
        d = max((lb[b, -1].float() - prefill_fast(model, ids[b:b + 1], j, W, fl, w)[0][0, -1].float()).abs().max().item() for b in range(2))
        res["check"][f"w{w}"] = d
        print(f"check batched vs batch-1 w={w}: max|dlogit| {d:.4f}", flush=True)
    for P in map(int, args.lengths.split(",")):
        rows = []
        for t in texts:
            x = tok(t, add_special_tokens=False)["input_ids"]
            while len(x) < P:
                x = x + x
            rows.append(x[:P])
        for B in map(int, args.batches.split(",")):
            if B * P > args.max_tokens:
                continue
            ids = torch.tensor(rows[:B], device=DEV)
            r = {"exact_hf": timeit(lambda: model.model(input_ids=ids, use_cache=True))}
            for w in (0, 64):
                r[f"cfkv_w{w}"] = timeit(lambda: prefill_fast_b(model, ids, j, W, fl, w))
            res["tput"][f"P{P}_B{B}"] = {k: {"ms": v, "tok_per_s": B * P / v * 1000} for k, v in r.items()}
            print(f"P={P} B={B}: " + " | ".join(f"{k} {v:.1f} ms {B*P/v*1000:,.0f} tok/s ({r['exact_hf']/v:.2f}x)"
                                                for k, v in r.items()), flush=True)
            json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)
            del ids; torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

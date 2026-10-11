"""Per-layer analysis: how predictable are upper-layer K/V from h_j, and why does POP depend on the model?

For one model and depth j: fit the ridge map (same calibration recipe and features as round2; alpha from
round2's validation choice), then on held-out text (WikiText-103 validation + Alpaca val chats, all
positions >= 1) report per upper layer l:
  R2_K, R2_V     1 - SSE / SST (SST about the per-dimension mean) for ridge (ours) and POP
  drift_raw      mean_t ||h_l - h_j|| / ||h_j||
  drift_norm     mean_t ||rms(h_l) - rms(h_j)|| / ||rms(h_j)||   (what POP ignores after RMSNorm)
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from engine import DEV, ROOT, load, dims, fit_grams, solve, features, kv_pre, rms
from round2 import wiki_texts, chat_pairs, chat_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--j", type=int, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--run_id", required=True)
    args = ap.parse_args()
    run_dir = f"{ROOT}/results/raw/{args.run_id}"; assert not os.path.exists(run_dir); os.makedirs(run_dir)
    tok, model = load(args.model)
    L, H, KV, hd = dims(model); j = args.j
    fl = tuple(x for x in (j, j - 4, j - 8) if x >= 1)
    calib = [tok(t, add_special_tokens=False)["input_ids"][:1280] for t in
             wiki_texts("train-00000-of-00002.parquet", 40000, 160, 1)]
    for u, r in chat_pairs("calib", 1024):
        p, a = chat_ids(tok, u, r); calib.append(p + a)
    acc = fit_grams(model, calib, [(j, fl)])
    G, C, n = acc[(j, fl)]
    W = solve(G, C, args.alpha); del acc, G, C
    held = [tok(t, add_special_tokens=False)["input_ids"][:1280] for t in
            wiki_texts("validation-00000-of-00001.parquet", None, 16, 2)]
    for u, r in chat_pairs("val", 64):
        p, a = chat_ids(tok, u, r); held.append(p + a)
    D = KV * hd
    stats = {l: {k: 0.0 for k in ("sse_k_cf", "sse_v_cf", "sse_k_pop", "sse_v_pop")} for l in range(j, L)}
    sums = {l: {"k": torch.zeros(D, device=DEV, dtype=torch.float64), "k2": torch.zeros((), device=DEV, dtype=torch.float64),
                "v": torch.zeros(D, device=DEV, dtype=torch.float64), "v2": torch.zeros((), device=DEV, dtype=torch.float64)}
            for l in range(j, L)}
    drift = {l: [0.0, 0.0] for l in range(j, L)}
    N = 0
    layers = model.model.layers
    for s in held:
        ids = torch.tensor([s], device=DEV)
        hs = model.model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
        sl = slice(1, len(s))
        X = features({f: hs[f][0] for f in fl}, fl)[sl]
        pred = X @ W
        hj = hs[j][0, sl].float(); nj = hj.norm(dim=-1); rj = rms(hj)
        for l in range(j, L):
            k, v = [t.double() for t in kv_pre(layers[l], hs[l][0, sl])]
            o = (l - j) * 2 * D
            kc, vc = pred[:, o:o + D].double(), pred[:, o + D:o + 2 * D].double()
            kp, vp = [t.double() for t in kv_pre(layers[l], hs[j][0, sl])]
            st = stats[l]
            st["sse_k_cf"] += (kc - k).pow(2).sum().item(); st["sse_v_cf"] += (vc - v).pow(2).sum().item()
            st["sse_k_pop"] += (kp - k).pow(2).sum().item(); st["sse_v_pop"] += (vp - v).pow(2).sum().item()
            sm = sums[l]
            sm["k"] += k.sum(0); sm["k2"] += k.pow(2).sum(); sm["v"] += v.sum(0); sm["v2"] += v.pow(2).sum()
            hl = hs[l][0, sl].float()
            drift[l][0] += ((hl - hj).norm(dim=-1) / nj).sum().item()
            drift[l][1] += ((rms(hl) - rj).norm(dim=-1) / rj.norm(dim=-1)).sum().item()
        N += len(s) - 1
    out = {"run_id": args.run_id, "model": args.model, "j": j, "L": L, "alpha": args.alpha, "n_calib_positions": n,
           "n_heldout_positions": N, "layers": {}}
    for l in range(j, L):
        sm, st = sums[l], stats[l]
        sst_k = (sm["k2"] - sm["k"].pow(2).sum() / N).item(); sst_v = (sm["v2"] - sm["v"].pow(2).sum() / N).item()
        out["layers"][l] = {"R2_K_cf": 1 - st["sse_k_cf"] / sst_k, "R2_V_cf": 1 - st["sse_v_cf"] / sst_v,
                            "R2_K_pop": 1 - st["sse_k_pop"] / sst_k, "R2_V_pop": 1 - st["sse_v_pop"] / sst_v,
                            "drift_raw": drift[l][0] / N, "drift_norm": drift[l][1] / N}
        r = out["layers"][l]
        print(f"layer {l:2d}: R2_K cf {r['R2_K_cf']:.3f} pop {r['R2_K_pop']:.3f} | R2_V cf {r['R2_V_cf']:.3f} pop {r['R2_V_pop']:.3f}"
              f" | drift raw {r['drift_raw']:.3f} norm {r['drift_norm']:.3f}", flush=True)
    json.dump(out, open(f"{run_dir}/metrics.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

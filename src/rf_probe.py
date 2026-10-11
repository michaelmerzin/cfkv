"""What limits the linear CF-KV predictor? A small closed-form probe on Qwen2.5-1.5B (j=14, half depth).

All predictors are ridge regressions with a closed-form solution (per-feature ridge, alpha chosen on held-out
generic text); only the feature map changes:
  lin        [rms h14, rms h10, rms h6, 1]                      (current CF-KV features)
  +rf        lin + GELU(R s(x) + b), R fixed Gaussian, s = per-feature standardization, width D
  +prev      lin + rms h14 of the previous token                (cross-token information)
  +prev+rf   both
Metric: R^2 of the true pre-RoPE upper-layer keys/values on held-out tokens (per output coordinate, averaged), on
generic validation text, on all tokens of 5-shot MMLU validation prompts, and on the answer-option tokens only.
"""
import argparse, json, os, sys, time
import numpy as np, pyarrow.parquet as pq, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from engine import DEV, ROOT, PHUB, snap, load, dims, rms, kv_pre
from round2 import log, wiki_texts, chat_pairs, chat_ids


def mmlu_val_spans(tok, n, seed=1):
    base = snap(PHUB, "datasets--cais--mmlu") + "/all"
    dev = pq.read_table(f"{base}/dev-00000-of-00001.parquet").to_pylist()
    val = pq.read_table(f"{base}/validation-00000-of-00001.parquet").to_pylist()
    shots = {}
    for r in dev:
        shots.setdefault(r["subject"], []).append(r)
    fmt = lambda r: r["question"].strip() + "\n" + "".join(f"{c}. {t}\n" for c, t in zip("ABCD", r["choices"])) + "Answer:"
    out = []
    for i in sorted(np.random.default_rng(seed).choice(len(val), size=n, replace=False)):
        r = val[i]
        head = f"The following are multiple choice questions (with answers) about {r['subject'].replace('_', ' ')}.\n\n"
        pre = head + "".join(fmt(d) + f" {'ABCD'[d['answer']]}\n\n" for d in shots[r["subject"]][:5]) + r["question"].strip() + "\n"
        o0 = len(pre); txt = pre + "".join(f"{c}. {t}\n" for c, t in zip("ABCD", r["choices"]))
        enc = tok(txt + "Answer:", add_special_tokens=True, return_offsets_mapping=True)
        opt = [t for t, (a, b) in enumerate(enc["offset_mapping"]) if b > a and a >= o0 and b <= len(txt)]
        out.append((enc["input_ids"], opt))
    return out


@torch.no_grad()
def states(model, ids, j, feat_layers):
    hs = model.model(input_ids=torch.tensor([ids], device=DEV), output_hidden_states=True, use_cache=False).hidden_states
    L = dims(model)[0]; layers = model.model.layers
    base = torch.cat([rms(hs[f][0].float()) for f in feat_layers], -1)          # [P, 3d]
    prev = torch.zeros_like(base[:, :hs[j].shape[-1]]); prev[1:] = rms(hs[j][0].float())[:-1]
    Y = torch.cat([torch.cat(kv_pre(layers[l], hs[l][0]), -1).float() for l in range(j, L)], -1)
    return base[1:], prev[1:], Y[1:]                                             # drop the sink position


def make_psi(base, prev, use_prev, rf):
    parts = [base]
    if use_prev:
        parts.append(prev)
    if rf is not None:
        mu, sd, R, b = rf
        parts.append(F.gelu(((base - mu) / sd) @ R + b))
    parts.append(torch.ones(base.shape[0], 1, device=DEV))
    return torch.cat(parts, -1).double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--j", type=int, default=14)
    ap.add_argument("--n_mmlu", type=int, default=50)
    ap.add_argument("--wide_only", type=float, default=0, help="skip the small configs; run D=8192 at this scale")
    args = ap.parse_args()
    run_dir = f"{ROOT}/results/raw/{args.run_id}"; os.makedirs(run_dir, exist_ok=True)
    tok, model = load("qwen1.5b")
    L, H, KV, hd = dims(model); j = args.j; fl = [j, j - 4, j - 8]
    calib = [tok(t, add_special_tokens=False)["input_ids"][:1280] for t in wiki_texts("train-00000-of-00002.parquet", 40000, 160, 1)]
    for u, r in chat_pairs("calib", 1024):
        p, a = chat_ids(tok, u, r); calib.append(p + a)
    gen = [tok(t, add_special_tokens=False)["input_ids"][:1280] for t in wiki_texts("validation-00000-of-00001.parquet", None, 8, 2)]
    for u, r in chat_pairs("val", 16):
        p, a = chat_ids(tok, u, r); gen.append(p + a)
    mm = mmlu_val_spans(tok, args.n_mmlu)
    # feature standardization from a calibration subset (for the random features only)
    sub = torch.cat([states(model, s, j, fl)[0] for s in calib[:40]])
    mu, sd = sub.mean(0), sub.std(0).clamp_min(1e-4); din = sub.shape[1]; del sub
    g = torch.Generator(device=DEV).manual_seed(0)
    def rf_params(D, scale):
        return (mu, sd, torch.randn(din, D, generator=g, device=DEV) * (scale / din ** 0.5),
                torch.randn(D, generator=g, device=DEV) * 0.5)
    configs = {"lin": (False, None)}
    for s_ in (0.5, 1.0, 2.0):
        configs[f"rf2048_s{s_:g}"] = (False, rf_params(2048, s_))
    configs["prev"] = (True, None)
    # held-out targets cached once
    def heldout(seqs, opt=None):
        B, P, Y, M = [], [], [], []
        for k, s in enumerate(seqs):
            b, p, y = states(model, s, j, fl); B.append(b); P.append(p); Y.append(y)
            if opt is not None:
                m = torch.zeros(len(s) - 1, dtype=torch.bool, device=DEV); m[[t - 1 for t in opt[k] if t >= 1]] = True; M.append(m)
        return torch.cat(B).cpu(), torch.cat(P).cpu(), torch.cat(Y).cpu(), (torch.cat(M).cpu() if M else None)  # held on CPU
    Hs = heldout(gen[::2]); Hg = heldout(gen[1::2])   # alpha / scale chosen on Hs, reported on Hg
    Hm = heldout([x[0] for x in mm], [x[1] for x in mm])
    out_dim = Hg[2].shape[1]; per = 2 * KV * hd
    res = {"run_id": args.run_id, "j": j, "n_calib_seqs": len(calib), "heldout_tokens": {"selection": int(Hs[0].shape[0]), "generic": int(Hg[0].shape[0]),
           "mmlu": int(Hm[0].shape[0]), "options": int(Hm[3].sum().item())}, "r2": {}}

    def r2_parts(Yh, Y):
        sse = (Yh - Y).pow(2).sum(0); sst = (Y - Y.mean(0)).pow(2).sum(0).clamp_min(1e-9)
        r2 = 1 - sse / sst                                                       # per output coordinate
        ks = torch.cat([torch.arange(l * per, l * per + KV * hd) for l in range(L - j)]).to(DEV)
        vs = torch.cat([torch.arange(l * per + KV * hd, (l + 1) * per) for l in range(L - j)]).to(DEV)
        top = torch.cat([torch.arange(l * per, (l + 1) * per) for l in range(L - j - 4, L - j)]).to(DEV)
        return {"keys": r2[ks].mean().item(), "values": r2[vs].mean().item(), "top4_layers": r2[top].mean().item()}

    def predict(Hx, W, rows=None):
        Wf = W.float(); out = []
        idx = torch.arange(Hx[0].shape[0]) if rows is None else rows.nonzero()[:, 0]
        for i in range(0, len(idx), 4096):
            k = idx[i:i + 4096]
            out.append(make_psi(Hx[0][k].to(DEV), Hx[1][k].to(DEV), use_prev_cur[0], rf_cur[0]).float() @ Wf)
        return torch.cat(out)

    use_prev_cur, rf_cur = [None], [None]

    def fit_eval(name, use_prev, rf):
        use_prev_cur[0], rf_cur[0] = use_prev, rf
        t0 = time.time(); G = None; C = None
        for s in calib:
            b, p, y = states(model, s, j, fl)
            X = make_psi(b, p, use_prev, rf)
            G = X.T @ X if G is None else G + X.T @ X
            XY = X.T @ y.double(); C = XY if C is None else C + XY
        best = None
        for alpha in (1e-5, 1e-4, 1e-3, 1e-2):
            A = G.clone(); A.diagonal().mul_(1 + alpha)
            W = torch.cholesky_solve(C, torch.linalg.cholesky(A)); del A
            sc = r2_parts(predict(Hs, W), Hs[2].to(DEV))
            if best is None or sc["keys"] + sc["values"] > best[0]:
                best = (sc["keys"] + sc["values"], alpha, W)
        _, alpha, W = best
        r = {"alpha": alpha, "features": int(G.shape[0]), "fit_sec": time.time() - t0}
        r["generic"] = r2_parts(predict(Hg, W), Hg[2].to(DEV))
        r["mmlu_all"] = r2_parts(predict(Hm, W), Hm[2].to(DEV))
        r["mmlu_options"] = r2_parts(predict(Hm, W, Hm[3]), Hm[2][Hm[3]].to(DEV))
        r["selection_split"] = r2_parts(predict(Hs, W), Hs[2].to(DEV))
        res["r2"][name] = r
        log(f"{name:16s} F={r['features']:6d} alpha {alpha:g} | " + " | ".join(
            f"{k}: K {r[k]['keys']:.3f} V {r[k]['values']:.3f} top4 {r[k]['top4_layers']:.3f}" for k in ("generic", "mmlu_all", "mmlu_options")))
        json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)
        del G, C, W; torch.cuda.empty_cache()
        return r

    for name, (up, rf) in configs.items():
        if not args.wide_only or name in ("lin", "prev"):   # same held-out sets as the wide configs
            fit_eval(name, up, rf)
    # best random-feature scale (on generic held-out text), then a wider expansion and the combination
    s_best = args.wide_only or max((0.5, 1.0, 2.0), key=lambda s_: sum(res["r2"][f"rf2048_s{s_:g}"]["selection_split"].values()))
    log(f"best RF scale on generic held-out text: {s_best:g}")
    rf8 = rf_params(8192, s_best)
    fit_eval(f"rf8192_s{s_best:g}", False, rf8)
    fit_eval(f"prev+rf8192_s{s_best:g}", True, rf8)
    log("DONE")


if __name__ == "__main__":
    main()

"""Variance-matched keys, and selection on a held-out task split (MMLU validation, never the test set).

Ridge predicts the conditional mean, so the predicted keys have less spread than the real ones: per column,
var(k_hat) = R^2 var(k) on the calibration data. Softmax attention is nonlinear in the keys, so shrunken keys
give flatter attention; values enter linearly and are left as fitted. For each upper layer and KV head we
rescale the predicted keys around their mean by s = (var(k) / var(k_hat))^(gamma/2), computed in closed form
from the calibration statistics (gamma = 0 is plain CF-KV, gamma = 1 matches the real key variance).

Selection over (features, alpha, gamma) uses two held-out criteria, both reported:
  generic: last-token KL on WikiText/chat validation text (the protocol of round2.py);
  task:    last-token KL over the four answer letters on 5-shot MMLU *validation* prompts (shots from dev).
MMLU test is used only for the final evaluation of the selected configurations and a gamma ablation.
"""
import argparse, json, os, sys, time
import numpy as np, pyarrow.parquet as pq, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from engine import DEV, ROOT, PHUB, snap, load, dims, Engine, fit_grams, solve, left_pad, context_mask
from round2 import log, wiki_texts, chat_pairs, chat_ids, mmlu_items, run_mmlu, flop_saving


def mmlu_val_items(tok, n, seed=1):
    base = snap(PHUB, "datasets--cais--mmlu") + "/all"
    dev = pq.read_table(f"{base}/dev-00000-of-00001.parquet").to_pylist()
    val = pq.read_table(f"{base}/validation-00000-of-00001.parquet").to_pylist()
    shots = {}
    for r in dev:
        shots.setdefault(r["subject"], []).append(r)
    fmt = lambda r, ans=True: (r["question"].strip() + "\n" + "".join(f"{c}. {t}\n" for c, t in zip("ABCD", r["choices"]))
                               + "Answer:" + (f" {'ABCD'[r['answer']]}\n\n" if ans else ""))
    idx = np.random.default_rng(seed).choice(len(val), size=n, replace=False)
    out = []
    for i in sorted(idx):
        r = val[i]
        head = f"The following are multiple choice questions (with answers) about {r['subject'].replace('_', ' ')}.\n\n"
        out.append(tok(head + "".join(fmt(d) for d in shots[r["subject"]][:5]) + fmt(r, ans=False),
                       add_special_tokens=True)["input_ids"])
    return out


@torch.no_grad()
def last_logits(eng, cuts, v, pad_id, cols=None, bs=8):
    out = []
    for i in range(0, len(cuts), bs):
        ids, m = left_pad(cuts[i:i + bs], pad_id)
        lg, _ = eng.prefill(ids, m, context_mask(m, 0), v["j"], v["mode"], v.get("W"), v.get("fl"), last_only=True)
        lg = lg[:, -1].float()
        out.append((lg[:, cols] if cols is not None else lg).cpu())
    return torch.cat(out)


def kl_top1(ref, lg):
    a, b = F.log_softmax(ref, -1), F.log_softmax(lg, -1)
    return (a.exp() * (a - b)).sum(-1).mean().item(), (ref.argmax(-1) == lg.argmax(-1)).float().mean().item()


def key_stats(G, C, sq, n, W, j, L, KV, hd, chunk=8192):
    """Per (layer, kv head): var(k)/var(k_hat) summed over the head's dims, and the column means of k_hat."""
    mu_hat = torch.empty(W.shape[1], dtype=torch.float64, device=W.device)  # mean of predictions (last feature = 1)
    e2 = torch.empty_like(mu_hat)
    for i in range(0, W.shape[1], chunk):
        Wc = W[:, i:i + chunk].double(); GW = G @ Wc
        e2[i:i + chunk] = (Wc * GW).sum(0) / n
        mu_hat[i:i + chunk] = GW[-1] / n          # G's last row is sum(phi), so (G W)[-1] = sum(k_hat)
        del Wc, GW
    var_hat = e2 - mu_hat ** 2
    mu_y = C[-1].double() / n
    var_y = sq / n - mu_y ** 2
    ratio = torch.empty(L - j, KV, dtype=torch.float64)
    for l in range(L - j):
        o = l * 2 * KV * hd
        for h in range(KV):
            c = slice(o + h * hd, o + (h + 1) * hd)
            ratio[l, h] = var_y[c].sum() / var_hat[c].sum().clamp_min(1e-12)
    return ratio, mu_hat.float()


def scaled(W, ratio, mu_hat, gamma, KV, hd):
    """In place: rescale predicted keys around their mean by ratio^(gamma/2). Scaling around the mean keeps the
    mean, so applying d1 then d2 equals applying d1 + d2 (used to step through the gamma grid without copies)."""
    if gamma == 0:
        return W
    s = ratio.pow(gamma / 2).float()
    for l in range(ratio.shape[0]):
        o = l * 2 * KV * hd
        for h in range(KV):
            c = slice(o + h * hd, o + (h + 1) * hd)
            W[:, c] *= s[l, h]
            W[-1, c] += (1 - s[l, h]) * mu_hat[c]
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen7b")
    ap.add_argument("--j", type=int, default=32)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--gammas", default="0,0.5,1")
    ap.add_argument("--specs", default="single,pair8", help="comma list of single, pair8, ctx")
    ap.add_argument("--alphas", default="1e-5,1e-4,1e-3")
    ap.add_argument("--alphas_pf", default="1e-4,1e-3,1e-2", help="alphas for per-feature (standardized) ridge")
    ap.add_argument("--n_val_task", type=int, default=240)
    ap.add_argument("--n_mmlu", type=int, default=1000)
    ap.add_argument("--mmlu_bs", type=int, default=8)
    ap.add_argument("--resume", default="", help="run_id of an interrupted run whose held-out scores are reused")
    args = ap.parse_args()
    run_dir = f"{ROOT}/results/raw/{args.run_id}"; assert not os.path.exists(run_dir); os.makedirs(run_dir)
    tok, model = load(args.model); eng = Engine(model, tok)
    L, H, KV, hd = dims(model); j = args.j
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    letter_ids = [tok(f" {c}", add_special_tokens=False)["input_ids"][-1] for c in "ABCD"]
    res = {"run_id": args.run_id, "model": args.model, "j": j, "args": vars(args), "selection": {}}

    # ---- calibration: identical to round2 defaults (WikiText train 160 x 1280 + 1024 Alpaca chats)
    calib = [tok(t, add_special_tokens=False)["input_ids"][:1280] for t in wiki_texts("train-00000-of-00002.parquet", 40000, 160, 1)]
    for u, r in chat_pairs("calib", 1024):
        p, a = chat_ids(tok, u, r); calib.append(p + a)
    spec_of = {"single": (j, (j,)), "pair8": (j, (j, j - 8)), "ctx": (j, (j, "ctx")), "ctx1": (j, (j, "ctx1"))}
    specs = [spec_of[x] for x in args.specs.split(",")]
    sq = {}; t0 = time.time()
    acc = fit_grams(model, calib, specs, c_dtype=torch.float32, sq=sq)
    n = acc[specs[0]][2]
    log(f"fit {len(specs)} specs over {n} positions: {time.time()-t0:.0f}s")

    # ---- held-out selection sets (exact logits cached once)
    rng = np.random.default_rng(5); gen = []
    for t in wiki_texts("validation-00000-of-00001.parquet", None, 8, 2):
        ids_ = tok(t, add_special_tokens=False)["input_ids"][:1280]
        gen += [ids_[:c] for c in rng.integers(64, len(ids_), size=6)]
    for u, r in chat_pairs("val", 16):
        p, a = chat_ids(tok, u, r); gen += [(p + a)[:c] for c in rng.integers(len(p), len(p) + len(a), size=3)]
    task = mmlu_val_items(tok, args.n_val_task)
    ex_gen = last_logits(eng, gen, {"j": 999, "mode": "exact"}, pad_id)
    ex_task = last_logits(eng, task, {"j": 999, "mode": "exact"}, pad_id, letter_ids)

    def score(v):
        g = kl_top1(ex_gen, last_logits(eng, gen, v, pad_id))
        t = kl_top1(ex_task, last_logits(eng, task, v, pad_id, letter_ids))
        return {"gen_kl": g[0], "gen_top1": g[1], "task_kl": t[0], "task_top1": t[1]}

    prev = {}
    if args.resume:   # reuse held-out scores of an interrupted run with the same settings
        old = json.load(open(f"{ROOT}/results/raw/{args.resume}/metrics.json"))
        prev = old["selection"]; res["resumed_from"] = args.resume
        for k in ("feat_second_moment", "key_std_shrink", "check_g1"):
            if k in old:
                res[k] = old[k]
    if "pop" in prev:
        res["selection"]["pop"] = prev["pop"]
    else:
        res["selection"]["pop"] = score({"j": j, "mode": "naive"})
    log("POP  " + " ".join(f"{k} {x:.4f}" for k, x in res["selection"]["pop"].items()))
    gammas = [float(x) for x in args.gammas.split(",")]
    stats = {}   # key -> (spec, alpha, per_feature); W is re-solved from the grams when needed (never stored)
    for spec in specs:
        G, C, _ = acc[spec]
        dg = torch.diagonal(G)[:-1] / n
        res.setdefault("feat_second_moment", {})[str(spec[1])] = {
            "max_over_median": (dg.max() / dg.median()).item(), "mean_over_median": (dg.mean() / dg.median()).item(),
            "n_over_100x_median": int((dg > 100 * dg.median()).sum())}
        log(f"features {spec[1]}: E[x^2] max/median {res['feat_second_moment'][str(spec[1])]['max_over_median']:.0f}, "
            f"mean/median {res['feat_second_moment'][str(spec[1])]['mean_over_median']:.2f}, "
            f"dims >100x median {res['feat_second_moment'][str(spec[1])]['n_over_100x_median']}")
        grid = [(a, False) for a in map(float, args.alphas.split(","))] + \
               ([(a, True) for a in map(float, args.alphas_pf.split(","))] if args.alphas_pf else [])
        for alpha, pf in grid:
            key = f"f{'-'.join(map(str, spec[1]))}_{'p' if pf else 'a'}{alpha:g}"
            stats[key] = (spec, alpha, pf)
            todo = [g for g in sorted(gammas) if f"{key}_g{g:g}" not in prev]
            for g in sorted(gammas):
                if f"{key}_g{g:g}" in prev:
                    res["selection"][f"{key}_g{g:g}"] = prev[f"{key}_g{g:g}"]
                    log(f"{key}_g{g:g}  (resumed) " + " ".join(f"{k} {x:.4f}" for k, x in prev[f"{key}_g{g:g}"].items()))
            if not todo:
                continue
            W = solve(G, C, alpha, per_feature=pf)
            ratio, mu_hat = key_stats(G, C, sq[j], n, W, j, L, KV, hd)
            res.setdefault("key_std_shrink", {})[key] = (1 / ratio.sqrt()).mean(1).tolist()  # per layer, std(k_hat)/std(k)
            cur = 0.0
            for g in sorted(gammas):
                scaled(W, ratio, mu_hat, g - cur, KV, hd); cur = g
                if g == 1 and "check_g1" not in res:   # variance matching must give var(k_hat) = var(k)
                    r1, m1 = key_stats(G, C, sq[j], n, W, j, L, KV, hd)
                    res["check_g1"] = {"max_ratio_err": (r1 - 1).abs().max().item(), "max_mean_shift": (m1 - mu_hat).abs().max().item()}
                    log("check gamma=1:", res["check_g1"])
                if g not in todo:
                    continue
                sc = score({"j": j, "mode": "predict", "W": W, "fl": list(spec[1])})
                res["selection"][f"{key}_g{g:g}"] = sc
                log(f"{key}_g{g:g}  " + " ".join(f"{k} {x:.4f}" for k, x in sc.items()))
                json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)
            del W, ratio, mu_hat; torch.cuda.empty_cache()
    json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)

    cand = {k: v for k, v in res["selection"].items() if k != "pop"}
    pick = {"generic": min(cand, key=lambda k: cand[k]["gen_kl"]), "task": min(cand, key=lambda k: cand[k]["task_kl"])}
    # gamma ablation at the (features, alpha) of each selected configuration
    evals = {}
    for crit, k in pick.items():
        base = k.rsplit("_g", 1)[0]
        for g in gammas:
            evals[f"{base}_g{g:g}"] = (base, g)
    res["picked"] = pick
    log("picked:", pick)

    items = mmlu_items(tok, args.n_mmlu)
    res["mmlu"] = {}
    variants = [("exact", {"j": 999, "mode": "exact"}), ("pop", {"j": j, "mode": "naive", "w": 0})] + \
               [(name, bg) for name, bg in evals.items()]
    needed = {stats[b][0] for b, _ in evals.values()}
    for spec in list(acc):             # keep only the statistics the evaluation needs
        if spec not in needed:
            acc.pop(spec)
    torch.cuda.empty_cache()
    preds = {}
    for name, v in variants:
        t0 = time.time()
        if isinstance(v, tuple):
            base, g = v
            spec, alpha, pf = stats[base]
            G, C, _ = acc[spec]
            W = solve(G, C, alpha, per_feature=pf)
            ratio, mu_hat = key_stats(G, C, sq[j], n, W, j, L, KV, hd)
            del G, C
            v = {"j": j, "mode": "predict", "w": 0, "W": scaled(W, ratio, mu_hat, g, KV, hd), "fl": list(spec[1])}
            del W
        a, pr = run_mmlu(eng, items, v, letter_ids, pad_id, bs=args.mmlu_bs)
        preds[name] = pr
        agree = float(np.mean([x == y for x, y in zip(pr, preds["exact"])]))
        res["mmlu"][name] = {"acc": a, "agree": agree, "saved": flop_saving(items, v, L), "preds": pr}
        log(f"MMLU {name:22s} acc {a:.4f} agree {agree:.3f} ({time.time()-t0:.0f}s)")
        json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)
        del v; torch.cuda.empty_cache()
    log("DONE")


if __name__ == "__main__":
    main()

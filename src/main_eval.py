"""Main-table evaluation with ONE CF-KV configuration per model, plus token-selection baselines at matched compute.

For a model and half depth j: fit, then select features (1/3 layers, or 1/2 for --feat pair8), alpha and gamma
(variance-matched keys) on held-out text exactly as round2.py; evaluate on all eight main-table benchmarks:
  cfkv        CF-KV with the selected configuration (context tokens stop at j)
  fastkv      FastKV-style selection at matched saving: at layer j_s = L/4, keep the fraction r = 1/3 of context
              tokens with the highest attention from the last 8 queries; the rest are dropped from layers >= j_s
              (saving (1-r)(L-j_s)/L = 50%)
  selcfkv     the same selection, with the unselected tokens' KV at layers >= j_s predicted by CF-KV (fit at j_s)
POP, drop, pruning and SwiftKV rows are unchanged from the earlier runs and are not recomputed.
Resumes after preemption (skips finished (benchmark, method) entries).
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from engine import DEV, ROOT, load, Engine, fit_grams, solve, dims
from round2 import (log, wiki_texts, chat_pairs, chat_ids, last_kl, run_gsm8k, run_mmlu, mmlu_items, gsm8k_items,
                    same_num, parse_num, flop_saving)
from gen_bench import math_items, humaneval_items, cnn_items, math_ok, humaneval_ok, rouge_l, block_influence, Pruned
from mbpp_eval import mbpp_items, extract_code as mbpp_code, passes as mbpp_passes
from extra_bench import arc_items, hellaswag_items, run_arc, run_hellaswag


def select_config(model, eng, tok, pad_id, j, feat, gammas, c_fp32):
    L, H, KV, hd = dims(model)
    calib = [tok(t, add_special_tokens=False)["input_ids"][:1280] for t in wiki_texts("train-00000-of-00002.parquet", 40000, 160, 1)]
    for u, r in chat_pairs("calib", 1024):
        p, a = chat_ids(tok, u, r); calib.append(p + a)
    three = tuple(x for x in (j, j - 4, j - 8) if x >= 1)
    specs = [(j, three)] if feat == "three" else [(j, (j,))]
    if feat == "both":
        specs.append((j, tuple(x for x in (j, j - 4, j - 8) if x >= 1)))
    elif feat == "pair8":
        specs.append((j, (j, j - 8)))
    sq = {} if gammas != [0.0] else None
    acc = fit_grams(model, calib, specs, c_dtype=torch.float32 if c_fp32 else torch.float64, sq=sq)
    rng = np.random.default_rng(5); val = []
    for t in wiki_texts("validation-00000-of-00001.parquet", None, 8, 2):
        ids_ = tok(t, add_special_tokens=False)["input_ids"][:1280]
        val += [ids_[:c] for c in rng.integers(64, len(ids_), size=6)]
    for u, r in chat_pairs("val", 16):
        p, a = chat_ids(tok, u, r); val += [(p + a)[:c] for c in rng.integers(len(p), len(p) + len(a), size=3)]
    best = None
    for spec in specs:
        G, C, nn = acc[spec]
        for alpha in (1e-5, 1e-4, 1e-3):
            W = solve(G, C, alpha)
            if sq is not None:
                from kscale import key_stats, scaled
                ratio, mu_hat = key_stats(G, C, sq[j], nn, W, j, L, KV, hd)
            cur = 0.0
            for g in gammas:
                if g != cur:
                    scaled(W, ratio, mu_hat, g - cur, KV, hd); cur = g
                kl, _ = last_kl(eng, val, {"j": j, "mode": "predict", "W": W, "fl": list(spec[1])}, pad_id)
                if best is None or kl < best["kl"]:
                    if best is not None:
                        del best["W"]
                    # the best W waits on the CPU: two [F, out] matrices next to a large model do not fit on one GPU
                    best = {"kl": kl, "feat": list(spec[1]), "alpha": alpha, "gamma": g, "W": W.detach().to("cpu", copy=True)}
            del W; torch.cuda.empty_cache()
    del acc; torch.cuda.empty_cache()
    best["W"] = best["W"].to(DEV)
    log(f"j={j}: feats {best['feat']} alpha {best['alpha']:g} gamma {best['gamma']:g} val KL {best['kl']:.4f}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--js", required=True, help="depths, e.g. 22,16 (one third, one half)")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--feat", default="both", choices=["both", "pair8", "single", "three"])
    ap.add_argument("--sel_feat", default="both", choices=["both", "pair8", "single"])
    ap.add_argument("--c_fp32", type=int, default=0)
    ap.add_argument("--gammas", default="0,1")
    ap.add_argument("--methods", default="cfkv,fastkv,selcfkv", help="subset of exact,cfkv,pop,drop,fastkv,fastkv2,selcfkv,prune")
    ap.add_argument("--tasks", default="gsm8k,math500,humaneval,mbpp,cnndm,mmlu,arc,hellaswag")
    ap.add_argument("--gen_bs", type=int, default=32)
    ap.add_argument("--mc_bs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke tests: first N items per benchmark")
    args = ap.parse_args()
    run_dir = f"{ROOT}/results/raw/{args.run_id}"; os.makedirs(run_dir, exist_ok=True)
    mf = f"{run_dir}/metrics.json"
    res = json.load(open(mf)) if os.path.exists(mf) else {"run_id": args.run_id, "model": args.model, "args": vars(args), "results": {}}
    tok, model = load(args.model); eng = Engine(model, tok)
    L = dims(model)[0]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gc = model.generation_config.eos_token_id
    stops = set([tok.eos_token_id] + (gc if isinstance(gc, list) else [gc]))
    for s_ in ("<|im_end|>", "<|eot_id|>"):
        i = tok.convert_tokens_to_ids(s_)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            stops.add(i)
    stop_ids = torch.tensor(sorted(x for x in stops if x is not None), device=DEV)
    gammas = sorted(float(x) for x in args.gammas.split(","))
    methods = args.methods.split(",")
    j_sel = round(L / 4)   # selection layer for the token-selection baselines
    variants = {}
    b2 = None
    for j in [int(x) for x in args.js.split(",")]:
        # token selection at layer L/4 keeping fraction r so that its saving (1-r)(L-j_sel)/L equals (L-j)/L
        r = 1 - ((L - j) / L) / ((L - j_sel) / L)
        if "cfkv" in methods:
            b = select_config(model, eng, tok, pad_id, j, args.feat, gammas, args.c_fp32)
            res.setdefault("cfkv_config", {})[str(j)] = {k: v for k, v in b.items() if k != "W"}
            variants[f"cfkv_j{j}"] = {"j": j, "mode": "predict", "w": 0, "W": b["W"], "fl": b["feat"]}
        if "pop" in methods:
            variants[f"pop_j{j}"] = {"j": j, "mode": "naive", "w": 0}
        if "drop" in methods:
            variants[f"drop_j{j}"] = {"j": j, "mode": "drop", "w": 0}
        if "fastkv" in methods:
            variants[f"fastkv_j{j}"] = {"j": j_sel, "mode": "drop", "w": 0, "keep": r}
        if "fastkv2" in methods:   # closer to the official FastKV: kernel-7 pooled scores, 8 forced window tokens, TSP layer sees all tokens
            r2 = 1 - ((L - j) / L) / ((L - j_sel - 1) / L)
            variants[f"fastkv2_j{j}"] = {"j": j_sel, "mode": "drop", "w": 0, "keep": r2, "pool": 7, "win": 8, "after": 1}
        if "selcfkv" in methods:
            if b2 is None:
                b2 = select_config(model, eng, tok, pad_id, j_sel, args.sel_feat, gammas, args.c_fp32)
                res["selcfkv_config"] = {k: v for k, v in b2.items() if k != "W"}
            variants[f"selcfkv_j{j}"] = {"j": j_sel, "mode": "predict", "w": 0, "keep": r, "W": b2["W"], "fl": b2["feat"]}
    if "exact" in methods:   # the unmodified model (the "Full model" row)
        variants["exact"] = {"j": 999, "mode": "exact", "w": 0}
    if "prune" in methods:   # ShortGPT-style: remove the lowest block-influence layers for ALL tokens (same recipe as gen_bench.py)
        pcal = [tok(t, add_special_tokens=False)["input_ids"][:1280] for t in wiki_texts("train-00000-of-00002.parquet", 40000, 160, 1)[:64]]
        bi = block_influence(model, pcal); order = sorted(range(L), key=lambda l: bi[l]); res["block_influence"] = bi
        for pct in (25, 50):
            variants[f"prune{pct}"] = {"j": 999, "mode": "exact", "w": 0, "prune": order[:int(round(pct / 100 * L))]}
    res["keep_frac"] = {str(j): 1 - ((L - j) / L) / ((L - j_sel) / L) for j in [int(x) for x in args.js.split(",")]}
    json.dump(res, open(mf, "w"), indent=1)
    letter4 = [tok(f" {c}", add_special_tokens=False)["input_ids"][-1] for c in "ABCD"]
    letter5 = [tok(f" {c}", add_special_tokens=False)["input_ids"][-1] for c in "ABCDE"]
    for task in args.tasks.split(","):
        R = res["results"].setdefault(task, {})
        todo = [m for m in variants if m not in R]
        if not todo:
            continue
        if task == "gsm8k":
            items = gsm8k_items(tok, 500)
        elif task == "math500":
            items = math_items(tok, 250)
        elif task == "humaneval":
            items = humaneval_items(tok, 164)
        elif task == "mbpp":
            items = mbpp_items(tok, 500)
        elif task == "cnndm":
            items = cnn_items(tok, 250, 1536)
        elif task == "mmlu":
            items = mmlu_items(tok, 1000)
        elif task == "arc":
            items = arc_items(tok)
        elif task == "hellaswag":
            items = hellaswag_items(tok, 1000)
        if args.limit:
            items = items[:args.limit]
        for name in todo:
            v = variants[name]; t0 = time.time()
            cm = Pruned(model, v["prune"]) if "prune" in v else None
            if cm is not None:
                cm.__enter__(); E = Engine(model, tok); vv = {"j": 999, "mode": "exact", "w": 0}
            else:
                E, vv = eng, v
            if task in ("gsm8k", "math500", "humaneval", "mbpp", "cnndm"):
                max_new = {"gsm8k": 400, "math500": 1024, "humaneval": 512, "mbpp": 512, "cnndm": 160}[task]
                gen = [(x[0], x[1] if task == "gsm8k" else "nan") for x in items]
                acc_, _, texts = run_gsm8k(E, gen, vv, pad_id, stop_ids, max_new=max_new, bs=args.gen_bs)
                if task == "gsm8k":
                    sc = [float(same_num(parse_num(t), items[i][1])) for i, t in enumerate(texts)]
                elif task == "math500":
                    sc = [float(math_ok(t, items[i][1])) for i, t in enumerate(texts)]
                elif task == "humaneval":
                    sc = [float(humaneval_ok(t, items[i][1])) for i, t in enumerate(texts)]
                elif task == "mbpp":
                    sc = [float(mbpp_passes(mbpp_code(t), items[i][1])) for i, t in enumerate(texts)]
                else:
                    sc = [rouge_l(t, items[i][1]) for i, t in enumerate(texts)]
                saved = flop_saving(items, v, L) if cm is None else len(v["prune"]) / L
            elif task == "mmlu":
                a, pr = run_mmlu(E, items, vv, letter4, pad_id, bs=args.mc_bs)
                sc = [float(p == items[i][1]) for i, p in enumerate(pr)]; saved = flop_saving(items, v, L) if cm is None else len(v["prune"]) / L
            elif task == "arc":
                a, sc = run_arc(E, items, vv, letter5, pad_id, bs=args.mc_bs); saved = None if cm is None else len(v["prune"]) / L
            else:
                a, sc = run_hellaswag(E, items, vv, pad_id, bs_items=args.mc_bs); saved = None if cm is None else len(v["prune"]) / L
            if cm is not None:
                cm.__exit__(None, None, None)
            R[name] = {"score": float(np.mean(sc)), "items": [float(x) for x in sc], "saved": saved}
            log(f"{task.upper():10s} {name:12s} score {100 * np.mean(sc):.1f}" +
                (f" saved {100 * saved:.0f}%" if saved is not None else "") + f" ({time.time() - t0:.0f}s)")
            json.dump(res, open(mf, "w"), indent=1)
    log("DONE")


if __name__ == "__main__":
    main()

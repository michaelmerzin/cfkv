"""Calibration ablation: how much / which unlabeled text does the closed-form fit need?

Llama-3.1-8B-Instruct, j=16 (50% of layers skipped for prompt tokens), w=0, features [h_j, h_{j-4}, h_{j-8}].
Configs vary the calibration pool size (mixed WikiText+chat, as in the main runs) and its domain at a
matched ~100k tokens. For each: ridge fit, alpha chosen on the same validation prefixes as round2
(last-token KL), then MMLU (1000, 5-shot) and GSM8K (500, generation) with the chosen map.
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from engine import DEV, ROOT, load, Engine, fit_grams, solve, left_pad
from round2 import (wiki_texts, chat_pairs, chat_ids, last_kl, mmlu_items, gsm8k_items, run_mmlu, run_gsm8k)


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama8b")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--j", type=int, default=16)
    ap.add_argument("--configs", default="mixed_10k,mixed_30k,mixed_100k,mixed_400k,wiki_100k,chat_100k")
    ap.add_argument("--n_mmlu", type=int, default=1000)
    ap.add_argument("--n_gsm8k", type=int, default=500)
    args = ap.parse_args()
    run_dir = f"{ROOT}/results/raw/{args.run_id}"; assert not os.path.exists(run_dir); os.makedirs(run_dir)
    tok, model = load(args.model)
    eng = Engine(model, tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gc = model.generation_config.eos_token_id
    stops = set([tok.eos_token_id] + (gc if isinstance(gc, list) else [gc]))
    for s_ in ("<|im_end|>", "<|eot_id|>"):
        i = tok.convert_tokens_to_ids(s_)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            stops.add(i)
    stop_ids = torch.tensor(sorted(x for x in stops if x is not None), device=DEV)
    j = args.j
    fl = tuple(x for x in (j, j - 4, j - 8) if x >= 1)

    # pools (same sources and seeds as round2; shuffled once, prefixes taken for smaller budgets)
    wiki = [tok(t, add_special_tokens=False)["input_ids"][:1280] for t in
            wiki_texts("train-00000-of-00002.parquet", 40000, 160, 1)]
    chat = []
    for u, r in chat_pairs("calib", 1024):
        p, a = chat_ids(tok, u, r); chat.append(p + a)
    rng = np.random.default_rng(0)
    wiki = [wiki[i] for i in rng.permutation(len(wiki))]
    chat = [chat[i] for i in rng.permutation(len(chat))]
    ntok = lambda seqs: sum(len(s) - 1 for s in seqs)

    def take(pool, budget):
        out, n = [], 0
        for s in pool:
            if n >= budget:
                break
            out.append(s); n += len(s) - 1
        return out

    def build(cfg):
        kind, size = cfg.split("_")
        budget = {"10k": 10_000, "30k": 30_000, "100k": 100_000, "400k": 10 ** 9}[size]
        if kind == "wiki":
            return take(wiki, budget)
        if kind == "chat":
            return take(chat, budget)
        frac = ntok(wiki) / (ntok(wiki) + ntok(chat))  # keep the main-run mixture proportion
        return take(wiki, budget * frac) + take(chat, budget * (1 - frac))

    # validation prefixes (identical construction to round2)
    rng = np.random.default_rng(5); val = []
    for t in wiki_texts("validation-00000-of-00001.parquet", None, 8, 2):
        ids_ = tok(t, add_special_tokens=False)["input_ids"][:1280]
        val += [ids_[:c] for c in rng.integers(64, len(ids_), size=6)]
    for u, r in chat_pairs("val", 16):
        p, a = chat_ids(tok, u, r); val += [(p + a)[:c] for c in rng.integers(len(p), len(p) + len(a), size=3)]
    letter_ids = [tok(f" {c}", add_special_tokens=False)["input_ids"][-1] for c in "ABCD"]
    mmlu = mmlu_items(tok, args.n_mmlu); gsm = gsm8k_items(tok, args.n_gsm8k)

    res = {"run_id": args.run_id, "model": args.model, "j": j, "feat_layers": list(fl), "configs": {}}
    for cfg in args.configs.split(","):
        seqs = build(cfg)
        t0 = time.time()
        acc = fit_grams(model, seqs, [(j, fl)]); torch.cuda.synchronize()
        G, C, n = acc[(j, fl)]
        fit_s = time.time() - t0
        best = None
        for alpha in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
            W = solve(G, C, alpha)
            kl, ag = last_kl(eng, val, {"j": j, "mode": "predict", "W": W, "fl": list(fl)}, pad_id)
            if best is None or kl < best[0]:
                best = (kl, ag, alpha, W)
        kl, ag, alpha, W = best
        v = {"j": j, "mode": "predict", "w": 0, "W": W, "fl": list(fl)}
        m_acc, _ = run_mmlu(eng, mmlu, v, letter_ids, pad_id)
        g_acc, _, _ = run_gsm8k(eng, gsm, v, pad_id, stop_ids)
        res["configs"][cfg] = {"n_positions": n, "n_seqs": len(seqs), "fit_sec": fit_s, "alpha": alpha,
                               "val_kl": kl, "val_top1": ag, "mmlu": m_acc, "gsm8k": g_acc}
        log(f"{cfg:11s} positions {n:7d} fit {fit_s:5.1f}s alpha {alpha:g} val KL {kl:.4f} | MMLU {m_acc*100:.1f} GSM8K {g_acc*100:.1f}")
        json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)
        del acc, G, C, W; torch.cuda.empty_cache()
    log("DONE")


if __name__ == "__main__":
    main()

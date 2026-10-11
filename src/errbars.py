"""Error bars: variance over calibration samples + per-item results for bootstrap CIs.

For one model / depth / window: draw S independent calibration samples of the default size (~385k tokens,
same WikiText/chat mixture) from a pool ~3x larger, refit the closed-form map on each, choose alpha on the
fixed validation prefixes, and evaluate MMLU (1000) and GSM8K (500). The exact model and POP are evaluated
once. Per-item correctness is saved for every run so paired bootstrap CIs can be computed offline.
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from engine import DEV, ROOT, load, Engine, fit_grams, solve
from round2 import (wiki_texts, chat_pairs, chat_ids, last_kl, mmlu_items, gsm8k_items, run_mmlu, run_gsm8k,
                    parse_num, same_num)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama8b")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--j", type=int, default=16)
    ap.add_argument("--w", type=int, default=0)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--budget", type=int, default=385_000)
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
    j = args.j; fl = tuple(x for x in (j, j - 4, j - 8) if x >= 1)

    # pool ~3x the default calibration size (disjoint seeds from the main runs' excerpts are fine: all train split)
    wiki = [tok(t, add_special_tokens=False)["input_ids"][:1280] for t in
            wiki_texts("train-00000-of-00002.parquet", 40000, 480, 31)]
    chat = []
    for u, r in chat_pairs("calib", 2048):
        p, a = chat_ids(tok, u, r); chat.append(p + a)
    ntok = lambda ss: sum(len(x) - 1 for x in ss)
    frac = 205_000 / 385_000  # WikiText share of the default mixture

    def sample(seed):
        rng = np.random.default_rng(seed)
        out, n = [], 0
        for pool, share in ((wiki, frac), (chat, 1 - frac)):
            k = 0
            for i in rng.permutation(len(pool)):
                if k >= args.budget * share:
                    break
                out.append(pool[i]); k += len(pool[i]) - 1
        return out

    rng = np.random.default_rng(5); val = []
    for t in wiki_texts("validation-00000-of-00001.parquet", None, 8, 2):
        ids_ = tok(t, add_special_tokens=False)["input_ids"][:1280]
        val += [ids_[:c] for c in rng.integers(64, len(ids_), size=6)]
    for u, r in chat_pairs("val", 16):
        p, a = chat_ids(tok, u, r); val += [(p + a)[:c] for c in rng.integers(len(p), len(p) + len(a), size=3)]
    letter_ids = [tok(f" {c}", add_special_tokens=False)["input_ids"][-1] for c in "ABCD"]
    mmlu = mmlu_items(tok, 1000); gsm = gsm8k_items(tok, 500)

    def evaluate(v):
        a, pr = run_mmlu(eng, mmlu, v, letter_ids, pad_id)
        g, _, texts = run_gsm8k(eng, gsm, v, pad_id, stop_ids)
        return {"mmlu": a, "gsm8k": g,
                "mmlu_items": [int(pr[i] == mmlu[i][1]) for i in range(len(mmlu))],
                "gsm8k_items": [int(same_num(parse_num(t), gsm[i][1])) for i, t in enumerate(texts)]}

    res = {"run_id": args.run_id, "model": args.model, "j": j, "w": args.w, "pool_tokens": ntok(wiki) + ntok(chat),
           "runs": {}}
    for name, v in (("exact", {"j": 999, "mode": "exact"}), ("pop", {"j": j, "mode": "naive", "w": args.w})):
        t0 = time.time(); res["runs"][name] = evaluate(v)
        print(f"{name:8s} MMLU {res['runs'][name]['mmlu']*100:.1f} GSM8K {res['runs'][name]['gsm8k']*100:.1f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(f"{run_dir}/metrics.json", "w"))
    for seed in map(int, args.seeds.split(",")):
        seqs = sample(seed); t0 = time.time()
        acc = fit_grams(model, seqs, [(j, fl)]); G, C, n = acc[(j, fl)]; del acc
        best = None
        for alpha in (1e-5, 1e-4, 1e-3):
            W = solve(G, C, alpha)
            kl, _ = last_kl(eng, val, {"j": j, "mode": "predict", "W": W, "fl": list(fl)}, pad_id)
            if best is None or kl < best[0]:
                best = (kl, alpha, W)
        del G, C; torch.cuda.empty_cache()
        r = evaluate({"j": j, "mode": "predict", "w": args.w, "W": best[2], "fl": list(fl)})
        r.update(n_positions=n, alpha=best[1], val_kl=best[0], fit_sec=time.time() - t0)
        res["runs"][f"cfkv_seed{seed}"] = r
        print(f"seed {seed}: positions {n} alpha {best[1]:g} val KL {best[0]:.4f} | MMLU {r['mmlu']*100:.1f} GSM8K {r['gsm8k']*100:.1f}", flush=True)
        json.dump(res, open(f"{run_dir}/metrics.json", "w"))
        del best; torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

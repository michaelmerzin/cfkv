"""Which prompt tokens must stay exact? Structured exact sets on 5-shot MMLU (Qwen2.5-7B by default).

Hypothesis: large models compute the answer *at the option tokens* in their upper layers, so those tokens'
upper-layer K/V hold a result that no map from layer j can predict. Test: keep only a structured subset of
prompt tokens at full depth (in addition to the sink and the last token) and measure MMLU accuracy:
  opts        the test question's four option lines + "Answer:"
  stem        the test question's stem
  stem+opts   both (roughly what a 64-token window covers)
  shot_ans    the answer letters of the five in-context examples
  rand_k      |opts| random tokens from the in-context examples (size-matched control)
for CF-KV and POP. W: single-layer features at the alpha round2 selected for this model/depth.
"""
import argparse, json, os, sys, time
import numpy as np, pyarrow.parquet as pq, torch
sys.path.insert(0, os.path.dirname(__file__))
from engine import DEV, ROOT, PHUB, snap, load, dims, Engine, fit_grams, solve, left_pad, context_mask
from round2 import log, wiki_texts, chat_pairs, chat_ids, mmlu_items


def mmlu_spans(tok, n, seed=0):
    """Same items as round2.mmlu_items, with token index sets for each structured region."""
    base = snap(PHUB, "datasets--cais--mmlu") + "/all"
    dev = pq.read_table(f"{base}/dev-00000-of-00001.parquet").to_pylist()
    test = pq.read_table(f"{base}/test-00000-of-00001.parquet").to_pylist()
    shots = {}
    for r in dev:
        shots.setdefault(r["subject"], []).append(r)
    idx = np.random.default_rng(seed).choice(len(test), size=n, replace=False)
    out = []
    for i in sorted(idx):
        r = test[i]
        head = f"The following are multiple choice questions (with answers) about {r['subject'].replace('_', ' ')}.\n\n"
        pieces, spans = [head], {"shot_ans": []}
        for d in shots[r["subject"]][:5]:
            pieces.append(d["question"].strip() + "\n" + "".join(f"{c}. {t}\n" for c, t in zip("ABCD", d["choices"])) + "Answer:")
            a0 = sum(map(len, pieces)); pieces.append(f" {'ABCD'[d['answer']]}")
            spans["shot_ans"].append((a0, sum(map(len, pieces)))); pieces.append("\n\n")
        s0 = sum(map(len, pieces)); pieces.append(r["question"].strip() + "\n")
        o0 = sum(map(len, pieces)); pieces.append("".join(f"{c}. {t}\n" for c, t in zip("ABCD", r["choices"])) + "Answer:")
        e = sum(map(len, pieces))
        txt = "".join(pieces)
        enc = tok(txt, add_special_tokens=True, return_offsets_mapping=True)
        ids, off = enc["input_ids"], enc["offset_mapping"]
        tok_in = lambda a, b: [t for t, (x, y) in enumerate(off) if y > x and x < b and y > a]
        sets = {"stem": tok_in(s0, o0), "opts": tok_in(o0, e),
                "shot_ans": sorted({t for a, b in spans["shot_ans"] for t in tok_in(a, b)})}
        sets["stem+opts"] = sorted(set(sets["stem"]) | set(sets["opts"]))
        out.append((ids, r["answer"], sets, s0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen7b")
    ap.add_argument("--j", type=int, default=14)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    run_dir = f"{ROOT}/results/raw/{args.run_id}"; assert not os.path.exists(run_dir); os.makedirs(run_dir)
    tok, model = load(args.model); eng = Engine(model, tok)
    L = dims(model)[0]; j = args.j
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    letter_ids = [tok(f" {c}", add_special_tokens=False)["input_ids"][-1] for c in "ABCD"]
    items = mmlu_spans(tok, args.n)
    ref = mmlu_items(tok, args.n)
    assert all(a[0] == b[0] for a, b in zip(items, ref)), "token ids differ from round2.mmlu_items"
    rng = np.random.default_rng(0)
    for ids, _, sets, s0 in items:   # size-matched random control inside the in-context examples
        cand = [t for t in range(1, len(ids) - 1) if t not in set(sets["stem+opts"])]
        sets["rand_k"] = sorted(rng.choice(cand, size=min(len(sets["opts"]), len(cand)), replace=False).tolist())
    calib = [tok(t, add_special_tokens=False)["input_ids"][:1280] for t in wiki_texts("train-00000-of-00002.parquet", 40000, 160, 1)]
    for u, r in chat_pairs("calib", 1024):
        p, a = chat_ids(tok, u, r); calib.append(p + a)
    spec = (j, (j,))
    acc = fit_grams(model, calib, [spec], c_dtype=torch.float32)
    W = solve(acc[spec][0], acc[spec][1], args.alpha); del acc; torch.cuda.empty_cache()
    res = {"run_id": args.run_id, "model": args.model, "j": j, "alpha": args.alpha, "results": {},
           "mean_prompt_tokens": float(np.mean([len(x[0]) for x in items])),
           "mean_set_size": {k: float(np.mean([len(x[2][k]) for x in items])) for k in items[0][2]}}
    log("mean set sizes:", res["mean_set_size"], "prompt", res["mean_prompt_tokens"])

    def run(variant, exact_set):
        preds = []
        for i in range(0, len(items), args.bs):
            chunk = items[i:i + args.bs]
            ids, m = left_pad([x[0] for x in chunk], pad_id)
            ctx = context_mask(m, 0)
            if exact_set:
                P = ids.shape[1]
                for b, x in enumerate(chunk):
                    pad = P - len(x[0])
                    ctx[b, torch.tensor(x[2][exact_set], device=DEV, dtype=torch.long) + pad] = False
            lg, _ = eng.prefill(ids, m, ctx, variant["j"], variant["mode"], variant.get("W"), variant.get("fl"),
                                last_only=True)
            preds += lg[:, -1, letter_ids].argmax(-1).tolist()
        return preds

    ex = run({"j": 999, "mode": "exact"}, None)
    gold = [x[1] for x in items]
    res["results"]["exact"] = {"acc": float(np.mean([p == g for p, g in zip(ex, gold)]))}
    log(f"exact acc {res['results']['exact']['acc']:.4f}")
    for es in (None, "opts", "stem", "stem+opts", "shot_ans", "rand_k"):
        for name, v in (("cfkv", {"j": j, "mode": "predict", "W": W, "fl": [j]}), ("pop", {"j": j, "mode": "naive"})):
            t0 = time.time(); pr = run(v, es)
            key = f"{name}_{es or 'none'}"
            res["results"][key] = {"acc": float(np.mean([p == g for p, g in zip(pr, gold)])),
                                   "agree": float(np.mean([p == e for p, e in zip(pr, ex)]))}
            log(f"{key:18s} acc {res['results'][key]['acc']:.4f} agree {res['results'][key]['agree']:.3f} ({time.time()-t0:.0f}s)")
            json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)
    log("DONE")


if __name__ == "__main__":
    main()

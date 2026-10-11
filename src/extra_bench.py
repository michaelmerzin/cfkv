"""Extra benchmarks from SwiftKV's suite: ARC-Challenge (5-shot, letter logits) and HellaSwag (0-shot,
length-normalized log-likelihood of the four endings). Same fit/selection recipe as round2.

HellaSwag semantics: the context is the prompt (approximated except its last token and window); the
continuation tokens are scored exactly, as they would be during decoding.
"""
import argparse, json, os, re, sys, time, glob
import numpy as np, pyarrow.parquet as pq, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from engine import DEV, ROOT, HUB, PHUB, snap, load, load_swiftkv, Engine, fit_grams, solve, left_pad, context_mask
from round2 import wiki_texts, chat_pairs, chat_ids, last_kl


def arc_items(tok, n_shot=5):
    base = snap(PHUB, "datasets--allenai--ai2_arc") + "/ARC-Challenge"
    train = pq.read_table(f"{base}/train-00000-of-00001.parquet").to_pylist()
    test = pq.read_table(f"{base}/test-00000-of-00001.parquet").to_pylist()
    L = "ABCDE"
    def fmt(r, ans=True):
        ch = r["choices"]
        s = "Question: " + r["question"].strip() + "\n" + "".join(f"{L[i]}. {t}\n" for i, t in enumerate(ch["text"])) + "Answer:"
        return s + (f" {L[ch['label'].index(r['answerKey'])]}\n\n" if ans else "")
    shots = [r for r in train if len(r["choices"]["text"]) == 4][:n_shot]
    head = "".join(fmt(r) for r in shots)
    items = []
    for r in test:
        ids = tok(head + fmt(r, ans=False), add_special_tokens=True)["input_ids"]
        items.append((ids, r["choices"]["label"].index(r["answerKey"]), len(r["choices"]["text"])))
    return items


def _pre(t):
    t = t.strip().replace(" [title]", ". ")
    t = re.sub(r"\[.*?\]", "", t)
    return t.replace("  ", " ")


def hellaswag_items(tok, n, seed=0):
    f = glob.glob(f"{HUB}/datasets--Rowan--hellaswag/snapshots/*/data/validation-00000-of-00001.parquet")[0]
    rows = pq.read_table(f).to_pylist()
    idx = np.sort(np.random.default_rng(seed).choice(len(rows), size=n, replace=False))
    items = []
    for i in idx:
        r = rows[i]
        ctx = _pre(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize())
        ends = r["endings"] if isinstance(r["endings"], list) else eval(r["endings"])
        cids = tok(ctx, add_special_tokens=True)["input_ids"]
        conts = [tok(" " + _pre(e), add_special_tokens=False)["input_ids"] for e in ends]
        items.append((cids, conts, [len((" " + _pre(e)).encode()) for e in ends], int(r["label"])))
    return items


@torch.no_grad()
def run_arc(eng, items, v, letter_ids, pad_id, bs=8):
    correct = []
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    preds = {}
    for k in range(0, len(order), bs):
        bi = order[k:k + bs]
        ids, m = left_pad([items[i][0] for i in bi], pad_id)
        lg, _ = eng.prefill(ids, m, context_mask(m, v.get("w", 0)), v["j"], v["mode"], v.get("W"), v.get("fl"),
                            keep=v.get("keep", 0), last_only=True,
                            pool=v.get("pool", 0), win=v.get("win", 0), after=v.get("after", 0))
        for r, i in enumerate(bi):
            nch = items[i][2]
            preds[i] = int(lg[r, -1, letter_ids[:nch]].argmax())
    return float(np.mean([preds[i] == items[i][1] for i in range(len(items))])), [int(preds[i] == items[i][1]) for i in range(len(items))]


@torch.no_grad()
def run_hellaswag(eng, items, v, pad_id, bs_items=8):
    corr = []
    for k in range(0, len(items), bs_items):
        chunk = items[k:k + bs_items]
        seqs, meta = [], []
        for c, (cids, conts, blen, lab) in enumerate(chunk):
            for e, ct in enumerate(conts):
                seqs.append(cids + ct); meta.append((c, e, len(cids), len(ct)))
        ids, m = left_pad(seqs, pad_id)
        P = ids.shape[1]
        ends = torch.tensor([P - lc for (_, _, _, lc) in meta], device=DEV)  # exclusive end of the context region
        ctx = context_mask(m, v.get("w", 0), ctx_end=ends)
        lg, _ = eng.prefill(ids, m, ctx, v["j"], v["mode"], v.get("W"), v.get("fl"), keep=v.get("keep", 0),
                            pool=v.get("pool", 0), win=v.get("win", 0), after=v.get("after", 0))
        lp = F.log_softmax(lg.float(), -1)
        scores = {}
        for r, (c, e, lctx, lc) in enumerate(meta):
            tgt = ids[r, P - lc:]
            s = lp[r, P - lc - 1:P - 1].gather(-1, tgt[:, None]).sum().item()
            scores[(c, e)] = s / chunk[c][2][e]
        for c, (_, conts, _, lab) in enumerate(chunk):
            pred = max(range(len(conts)), key=lambda e: scores[(c, e)])
            corr.append(int(pred == lab))
    return float(np.mean(corr)), corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--js", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--n_hs", type=int, default=1000)
    ap.add_argument("--n_arc", type=int, default=0, help="0 = all 1172")
    ap.add_argument("--feat", default="both", choices=["both", "single"])
    ap.add_argument("--c_fp32", type=int, default=0, help="accumulate the cross-moment matrix in fp32 (large models)")
    args = ap.parse_args()
    run_dir = f"{ROOT}/results/raw/{args.run_id}"; os.makedirs(run_dir, exist_ok=True)
    prev = json.load(open(f"{run_dir}/metrics.json")) if os.path.exists(f"{run_dir}/metrics.json") else {}  # resume after preemption
    swift = None
    if args.model == "swiftkv8b":   # public distilled checkpoint: evaluated as is, no fitting
        tok, model, swift = load_swiftkv(); js = []
    else:
        tok, model = load(args.model)
    eng = Engine(model, tok, swift)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if swift is None:
        js = [int(x) for x in args.js.split(",")]
    calib = [tok(t, add_special_tokens=False)["input_ids"][:1280] for t in
             wiki_texts("train-00000-of-00002.parquet", 40000, 160, 1)]
    for u, r in chat_pairs("calib", 1024):
        p, a = chat_ids(tok, u, r); calib.append(p + a)
    specs = []
    for j in js:
        specs.append((j, (j,)))
        if args.feat == "both":
            specs.append((j, tuple(x for x in (j, j - 4, j - 8) if x >= 1)))
    acc = fit_grams(model, calib, specs, c_dtype=torch.float32 if args.c_fp32 else torch.float64) if specs else {}
    rng = np.random.default_rng(5); val = []
    for t in wiki_texts("validation-00000-of-00001.parquet", None, 8, 2):
        ids_ = tok(t, add_special_tokens=False)["input_ids"][:1280]
        val += [ids_[:c] for c in rng.integers(64, len(ids_), size=6)]
    for u, r in chat_pairs("val", 16):
        p, a = chat_ids(tok, u, r); val += [(p + a)[:c] for c in rng.integers(len(p), len(p) + len(a), size=3)]
    chosen = {}
    for j in js:
        best = None
        for spec in [s for s in specs if s[0] == j]:
            for alpha in (1e-5, 1e-4, 1e-3):
                W = solve(acc[spec][0], acc[spec][1], alpha)
                kl, _ = last_kl(eng, val, {"j": j, "mode": "predict", "W": W, "fl": list(spec[1])}, pad_id)
                if best is None or kl < best[0]:
                    best = (kl, spec, alpha, W)
        chosen[j] = best
        print(f"j={j}: feats {best[1][1]} alpha {best[2]:g} val KL {best[0]:.4f}", flush=True)
    del acc; torch.cuda.empty_cache()
    variants = [("exact", {"j": 999, "mode": "exact"})]
    if swift is not None:
        variants.append(("swiftkv", {"j": swift["j"], "mode": "swiftkv"}))
    for j in js:
        for mode in ("naive", "drop", "predict"):
            v = {"j": j, "mode": mode, "w": 0}
            if mode == "predict":
                v.update(W=chosen[j][3], fl=list(chosen[j][1][1]))
            variants.append((f"{mode}_j{j}_w0", v))
    letter_ids = [tok(f" {c}", add_special_tokens=False)["input_ids"][-1] for c in "ABCDE"]
    arc = arc_items(tok); hs = hellaswag_items(tok, args.n_hs)
    if args.n_arc:
        arc = arc[:args.n_arc]
    res = {"run_id": args.run_id, "model": args.model, "js": js,
           "chosen": {j: {"feat": list(c[1][1]), "alpha": c[2], "val_kl": c[0]} for j, c in chosen.items()},
           "n_arc": len(arc), "n_hellaswag": len(hs), "arc": dict(prev.get("arc", {})), "hellaswag": dict(prev.get("hellaswag", {}))}
    for name, v in variants:
        if name in res["arc"] and name in res["hellaswag"]:
            print(f"{name:16s} ARC-C {res['arc'][name]['acc']*100:.1f} HellaSwag {res['hellaswag'][name]['acc']*100:.1f} (resumed)", flush=True); continue
        t0 = time.time()
        a, ai = run_arc(eng, arc, v, letter_ids, pad_id)
        h, hi = run_hellaswag(eng, hs, v, pad_id)
        res["arc"][name] = {"acc": a, "items": ai}; res["hellaswag"][name] = {"acc": h, "items": hi}
        print(f"{name:16s} ARC-C {a*100:.1f} HellaSwag {h*100:.1f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(f"{run_dir}/metrics.json", "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

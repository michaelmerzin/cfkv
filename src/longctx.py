"""Long-context evaluation: RULER-style needle tasks (4k-32k) + LongBench QA (F1).

Deployment semantics as in round2: only the prompt is approximated; first token, optional exact window w
before the last prompt token, the last prompt token and all generated tokens are exact.
Models: llama8b / qwen7b (methods exact, naive=POP, drop, predict) or swiftkv8b (exact vs SwiftKV).
"""
import argparse, collections, json, os, re, string, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from engine import DEV, ROOT, load, load_swiftkv, Engine, fit_grams, solve, left_pad, context_mask
from round2 import wiki_texts, chat_pairs, chat_ids, last_kl, WT
import pyarrow.parquet as pq

WORDS = ("apple river window garden silver candle rocket forest violin pepper harbor marble lantern tiger "
         "cobalt meadow falcon quartz saddle ember walnut glacier piano orchid anchor thistle comet velvet "
         "canyon beacon").split()


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ NIAH (RULER-style)
class Haystack:
    def __init__(self, tok):
        t = pq.read_table(f"{WT}/test-00000-of-00001.parquet", columns=["text"]).column("text").to_pylist()
        t += pq.read_table(f"{WT}/validation-00000-of-00001.parquet", columns=["text"]).column("text").to_pylist()
        self.par = [x for x in t if x.strip() and not x.strip().startswith("=")]
        self.ntok = [len(tok(x, add_special_tokens=False)["input_ids"]) for x in self.par]

    def take(self, budget, rng):
        start = int(rng.integers(0, len(self.par) // 2))
        out, n, i = [], 0, start
        while n < budget:
            out.append(self.par[i % len(self.par)]); n += self.ntok[i % len(self.par)]; i += 1
        return out


def niah_item(tok, hay, task, length, rng):
    nums = [str(x) for x in rng.integers(1_000_000, 9_999_999, size=4)]
    keys = list(rng.choice(WORDS, size=4, replace=False))
    if task == "single":
        needles = [(keys[0], nums[0])]; qkey, ans = keys[0], [nums[0]]
    elif task == "multikey":
        needles = list(zip(keys, nums)); qi = int(rng.integers(0, 4)); qkey, ans = keys[qi], [nums[qi]]
    else:  # multivalue
        needles = [(keys[0], n) for n in nums]; qkey, ans = keys[0], nums
    head = ("Some special magic numbers are hidden within the following text. Make sure to memorize it. "
            "I will quiz you about the numbers afterwards.\n")
    if task == "multivalue":
        q = f"\nWhat are all the special magic numbers for {qkey} mentioned in the provided text?"
        prefix = f"The special magic numbers for {qkey} mentioned in the provided text are"
    else:
        q = f"\nWhat is the special magic number for {qkey} mentioned in the provided text?"
        prefix = f"The special magic number for {qkey} mentioned in the provided text is"
    overhead = 120
    pars = hay.take(length - overhead, rng)
    for k, v in needles:
        pos = int(rng.integers(0, len(pars) + 1))
        pars.insert(pos, f"One of the special magic numbers for {k} is: {v}.\n")
    user = head + "".join(pars) + q
    ids = chat_ids(tok, user) + tok(prefix, add_special_tokens=False)["input_ids"]
    return ids, ans


def niah_score(text, ans):
    return float(np.mean([a in text for a in ans]))


# ------------------------------------------------------------------ LongBench
LB = {
    "hotpotqa": ("Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
                 "The following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me "
                 "the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:", 32),
    "2wikimqa": ("Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
                 "The following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me "
                 "the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:", 32),
    "multifieldqa_en": ("Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on "
                        "the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:", 64),
}


def lb_items(tok, task, n, max_len):
    rows = [json.loads(l) for l in open(f"{ROOT}/data/longbench/data/{task}.jsonl")][:n]
    tmpl, max_gen = LB[task]
    out = []
    for r in rows:
        ctx_ids = tok(r["context"], add_special_tokens=False)["input_ids"]
        budget = max_len - 400
        if len(ctx_ids) > budget:  # LongBench middle truncation
            h = budget // 2
            ctx = tok.decode(ctx_ids[:h]) + tok.decode(ctx_ids[-h:])
        else:
            ctx = r["context"]
        out.append((chat_ids(tok, tmpl.format(context=ctx, input=r["input"])), r["answers"], max_gen))
    return out


def _norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred, golds):
    best = 0.0
    p = _norm(pred).split()
    for g in golds:
        gt = _norm(g).split()
        common = collections.Counter(p) & collections.Counter(gt)
        ns = sum(common.values())
        if ns == 0:
            continue
        prec, rec = ns / len(p), ns / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


# ------------------------------------------------------------------ run
def generate_one(eng, ids_, v, pad_id, stop_ids, max_new):
    ids, m = left_pad([ids_], pad_id)
    if v["mode"] == "gemfilter":  # select top-k tokens at layer r, then run the full model on them only
        ids = eng.gemfilter_select(ids, v["r"], v["k"]); m = torch.ones_like(ids, dtype=torch.bool)
        v = {"j": 999, "mode": "exact"}
    ctx = context_mask(m, v.get("w", 0))
    kv_src = eng.scaffold_kv(ids, m, v["j"], v.get("tmpl"), v["filler"]) if v["mode"] == "scaffold" else None
    lg, cache = eng.prefill(ids, m, ctx, v["j"], v["mode"], v.get("W"), v.get("fl"), max_new=max_new,
                            keep=v.get("keep", 0), last_only=True, kv_src=kv_src)
    seq = eng.decode(lg[:, -1].argmax(-1), cache, max_new, stop_ids)[0].tolist()
    cut = next((t for t, x in enumerate(seq) if x in stop_ids.tolist()), len(seq))
    return eng.tok.decode(seq[:cut])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama8b")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--js", default="16,22")
    ap.add_argument("--windows", default="0,64")
    ap.add_argument("--modes", default="naive,drop,predict")
    ap.add_argument("--tasks", default="niah,longbench")
    ap.add_argument("--lengths", default="4096,8192,16384,32768")
    ap.add_argument("--n_niah", type=int, default=20)
    ap.add_argument("--n_lb", type=int, default=50)
    ap.add_argument("--lb_max_len", type=int, default=31500)
    ap.add_argument("--calib_long", type=int, default=16)
    ap.add_argument("--gemfilter", default="", help="e.g. 13:1024 (filter layer : kept tokens)")
    ap.add_argument("--keeps", default="", help="attention-selected exact tokens, e.g. 2048 (select+drop / select+predict)")
    ap.add_argument("--no_plain", type=int, default=0, help="skip the plain naive/drop/predict window variants")
    ap.add_argument("--scaffold", type=int, default=0, help="add the filler-scaffold KV control (2605.30574)")
    args = ap.parse_args()
    run_dir = f"{ROOT}/results/raw/{args.run_id}"; assert not os.path.exists(run_dir); os.makedirs(run_dir)
    t0 = time.time()
    swift = None
    if args.model == "swiftkv8b":
        tok, model, swift = load_swiftkv()
    else:
        tok, model = load(args.model)
    eng = Engine(model, tok, swift)
    L = eng.L
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gc = model.generation_config.eos_token_id
    stops = set([tok.eos_token_id] + (gc if isinstance(gc, list) else [gc]))
    for s_ in ("<|im_end|>", "<|eot_id|>"):
        i = tok.convert_tokens_to_ids(s_)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            stops.add(i)
    stop_ids = torch.tensor(sorted(x for x in stops if x is not None), device=DEV)
    res = {"run_id": args.run_id, "model": args.model, "L": L, "args": vars(args)}

    variants = [("exact", {"j": 999, "mode": "exact"})]
    if swift is not None:
        variants.append(("swiftkv", {"j": swift["j"], "mode": "swiftkv"}))
    else:
        js = [int(x) for x in args.js.split(",")]
        calib = [tok(t, add_special_tokens=False)["input_ids"][:1280] for t in
                 wiki_texts("train-00000-of-00002.parquet", 40000, 160, 1)]
        calib += [tok(t, add_special_tokens=False)["input_ids"][:8192] for t in
                  wiki_texts("train-00000-of-00002.parquet", 40000, args.calib_long, 21, chars=45000)]
        for u, r in chat_pairs("calib", 1024):
            p, a = chat_ids(tok, u, r); calib.append(p + a)
        specs = [(j, tuple(x for x in (j, j - 4, j - 8) if x >= 1) + ("ctx",)) for j in js]
        tf = time.time(); acc = fit_grams(model, calib, specs); torch.cuda.synchronize()
        res["fit_sec"] = time.time() - tf; res["n_calib_positions"] = acc[specs[0]][2]
        log(f"fit {len(specs)} specs over {res['n_calib_positions']} positions in {res['fit_sec']:.0f}s")
        rng = np.random.default_rng(5); val = []
        for t in wiki_texts("validation-00000-of-00001.parquet", None, 8, 2):
            ids_ = tok(t, add_special_tokens=False)["input_ids"][:1280]
            val += [ids_[:c] for c in rng.integers(64, len(ids_), size=6)]
        for u, r in chat_pairs("val", 16):
            p, a = chat_ids(tok, u, r); val += [(p + a)[:c] for c in rng.integers(len(p), len(p) + len(a), size=3)]
        res["chosen"] = {}
        for j, spec in zip(js, specs):
            best = None
            for alpha in (1e-5, 1e-4, 1e-3):
                W = solve(acc[spec][0], acc[spec][1], alpha)
                kl, ag = last_kl(eng, val, {"j": j, "mode": "predict", "W": W, "fl": list(spec[1])}, pad_id)
                if best is None or kl < best[0]:
                    best = (kl, alpha, W)
            res["chosen"][j] = {"alpha": best[1], "val_kl": best[0], "feat": [str(x) for x in spec[1]]}
            log(f"j={j}: alpha {best[1]:g} val KL {best[0]:.4f}")
            for w in ([] if args.no_plain else map(int, args.windows.split(","))):
                for mode in args.modes.split(","):
                    v = {"j": j, "mode": mode, "w": w}
                    if mode == "predict":
                        v.update(W=best[2], fl=list(spec[1]))
                    variants.append((f"{mode}_j{j}_w{w}", v))
            if args.scaffold:
                f_ids = tok(" This is neutral filler text with no particular content.", add_special_tokens=False)["input_ids"]
                for w in map(int, args.windows.split(",")):
                    variants.append((f"scaffold_j{j}_w{w}", {"j": j, "mode": "scaffold", "w": w, "filler": f_ids,
                                                             "tmpl": chat_ids(tok, "")}))
            for kp in [int(x) for x in args.keeps.split(",") if x]:
                for mode in ("drop", "predict"):
                    v = {"j": j, "mode": mode, "w": 0, "keep": kp}
                    if mode == "predict":
                        v.update(W=best[2], fl=list(spec[1]))
                    variants.append((f"{mode}_j{j}_keep{kp}", v))
        del acc; torch.cuda.empty_cache()
    if args.gemfilter:
        r_, k_ = map(int, args.gemfilter.split(":"))
        variants.append((f"gemfilter_r{r_}_k{k_}", {"mode": "gemfilter", "r": r_, "k": k_, "j": 999}))
    json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)

    items = []
    if "niah" in args.tasks:
        hay = Haystack(tok); rng = np.random.default_rng(123)
        for length in map(int, args.lengths.split(",")):
            for task in ("single", "multikey", "multivalue"):
                for _ in range(args.n_niah):
                    ids, ans = niah_item(tok, hay, task, length, rng)
                    items.append(("niah", f"{task}_{length}", ids, ans, 32))
    if "longbench" in args.tasks:
        for task in LB:
            for ids, ans, mg in lb_items(tok, task, args.n_lb, args.lb_max_len):
                items.append(("lb", task, ids, ans, mg))
    lens = collections.defaultdict(list)
    for x in items:
        lens[x[1]].append(len(x[2]))
    res["prompt_tokens"] = {k: float(np.mean(v)) for k, v in lens.items()}
    log(f"{len(items)} items; mean prompt tokens per subset: " + " ".join(f"{k}={v:.0f}" for k, v in res["prompt_tokens"].items()))
    res["results"] = {}
    for name, v in variants:
        ts = time.time(); scores = collections.defaultdict(list); outs = []
        for kind, sub, ids, ans, mg in items:
            txt = generate_one(eng, ids, v, pad_id, stop_ids, mg)
            sc = niah_score(txt, ans) if kind == "niah" else qa_f1(txt, ans)
            scores[sub].append(sc); outs.append(txt)
        agg = {k: float(np.mean(x)) for k, x in scores.items()}
        res["results"][name] = {"scores": agg, "sec": time.time() - ts}
        json.dump({"outputs": outs}, open(f"{run_dir}/out_{name}.json", "w"))
        log(f"{name:18s} " + " ".join(f"{k}={x*100:.1f}" for k, x in sorted(agg.items())) + f" ({time.time()-ts:.0f}s)")
        json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)
    res["runtime_sec"] = time.time() - t0
    json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)
    log("DONE", res["runtime_sec"])


if __name__ == "__main__":
    main()

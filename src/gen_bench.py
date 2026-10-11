"""More generation benchmarks, plus a layer-pruning baseline.

Tasks (greedy decoding, w=0 on the prompt):
  math500  MATH-500 (HuggingFaceH4), zero-shot CoT, answer in \\boxed{}; normalized exact match (simplified).
  humaneval  HumanEval pass@1: the model writes the full function in a ```python block; executed with the tests.
  cnndm    CNN/DailyMail 3.0.0 test, 3-sentence summary of the article (truncated to --cnn_tokens); ROUGE-L F1.
Methods: exact, POP (naive), drop, CF-KV (predict; features/alpha/gamma chosen on validation as in round2.py), and
ShortGPT-style layer pruning (all tokens skip the n layers with the lowest block influence 1 - cos(h_in, h_out)
on calibration text; prune fractions via --prune).
"""
import argparse, json, os, re, subprocess, sys, tempfile, time
import numpy as np, pyarrow.parquet as pq, torch
sys.path.insert(0, os.path.dirname(__file__))
from engine import DEV, ROOT, PHUB, snap, load, load_swiftkv, Engine, fit_grams, solve, dims
from round2 import log, wiki_texts, chat_pairs, chat_ids, last_kl, run_gsm8k, flop_saving


# ------------------------------------------------------------------ data
def math_items(tok, n):
    rows = [json.loads(l) for l in open(snap(PHUB, "datasets--HuggingFaceH4--MATH-500") + "/test.jsonl")][:n]
    return [(chat_ids(tok, r["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."),
             r["answer"]) for r in rows]


def humaneval_items(tok, n):
    rows = pq.read_table(snap(PHUB, "datasets--openai--openai_humaneval") + "/openai_humaneval/test-00000-of-00001.parquet").to_pylist()[:n]
    return [(chat_ids(tok, "Complete the following Python function. Write the full function, including the signature, "
                           "in a single ```python code block.\n\n```python\n" + r["prompt"] + "```"), r) for r in rows]


def cnn_items(tok, n, max_tok):
    rows = pq.read_table(snap(PHUB, "datasets--abisee--cnn_dailymail") + "/3.0.0/test-00000-of-00001.parquet").to_pylist()[:n]
    out = []
    for r in rows:
        art = tok.decode(tok(r["article"], add_special_tokens=False)["input_ids"][:max_tok])
        out.append((chat_ids(tok, "Summarize the following news article in three sentences.\n\n" + art), r["highlights"]))
    return out


# ------------------------------------------------------------------ scoring
def last_boxed(t):
    i = t.rfind("\\boxed{")
    if i < 0:
        return None
    j, d = i + 7, 1
    while j < len(t) and d:
        d += {"{": 1, "}": -1}.get(t[j], 0); j += 1
    return t[i + 7:j - 1] if d == 0 else None


def norm_math(s):
    if s is None:
        return None
    s = s.strip().strip("$").replace(" ", "").replace("\\!", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").replace("^\\circ", "").replace("^{\\circ}", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s).rstrip(".")
    return s


def math_ok(text, gold):
    a, g = norm_math(last_boxed(text)), norm_math(gold)
    if a is None:
        return False
    if a == g:
        return True
    try:
        return abs(float(a) - float(g)) < 1e-6
    except ValueError:
        return False


def extract_code(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return m[0] if m else text


def humaneval_ok(text, row, timeout=10):
    code = extract_code(text)
    if f"def {row['entry_point']}" not in code:
        code = row["prompt"] + code
    prog = code + "\n\n" + row["test"] + f"\ncheck({row['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=os.environ.get("TMPDIR"), delete=False) as f:
        f.write(prog); path = f.name
    try:
        return subprocess.run([sys.executable, path], capture_output=True, timeout=timeout,
                              cwd=os.path.dirname(path)).returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


def rouge_l(pred, ref):
    a, b = re.findall(r"\w+", pred.lower()), re.findall(r"\w+", ref.lower())
    if not a or not b:
        return 0.0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for k, y in enumerate(b, 1):
            cur = dp[k]
            dp[k] = prev + 1 if x == y else max(dp[k], dp[k - 1])
            prev = cur
    l = dp[-1]
    if l == 0:
        return 0.0
    p, r = l / len(a), l / len(b)
    return 2 * p * r / (p + r)


# ------------------------------------------------------------------ layer pruning (ShortGPT)
@torch.no_grad()
def block_influence(model, seqs):
    L = dims(model)[0]; bi = torch.zeros(L, dtype=torch.float64); n = 0
    for s in seqs:
        hs = model.model(input_ids=torch.tensor([s], device=DEV), output_hidden_states=True, use_cache=False).hidden_states
        for l in range(L):
            bi[l] += (1 - torch.nn.functional.cosine_similarity(hs[l][0].float(), hs[l + 1][0].float(), dim=-1)).sum().item()
        n += len(s)
    return (bi / n).tolist()


class Pruned:
    """Temporarily remove layers from the model; the engine then runs the remaining ones for all tokens."""
    def __init__(self, model, drop):
        self.model, self.drop = model, set(drop)

    def __enter__(self):
        self.full = self.model.model.layers; self.nl = self.model.config.num_hidden_layers
        self.model.model.layers = torch.nn.ModuleList([l for i, l in enumerate(self.full) if i not in self.drop])
        self.model.config.num_hidden_layers = len(self.model.model.layers)
        return self.model

    def __exit__(self, *a):
        self.model.model.layers = self.full; self.model.config.num_hidden_layers = self.nl


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama8b")
    ap.add_argument("--js", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--tasks", default="math500,humaneval,cnndm")
    ap.add_argument("--n_math", type=int, default=250)
    ap.add_argument("--n_he", type=int, default=164)
    ap.add_argument("--n_cnn", type=int, default=250)
    ap.add_argument("--cnn_tokens", type=int, default=1536)
    ap.add_argument("--prune", default="0.25,0.5", help="fractions of layers removed for ALL tokens (ShortGPT)")
    ap.add_argument("--gammas", default="0,1")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--feat", default="both", choices=["both", "single"])
    ap.add_argument("--c_fp32", type=int, default=0)
    args = ap.parse_args()
    run_dir = f"{ROOT}/results/raw/{args.run_id}"; os.makedirs(run_dir, exist_ok=True)
    prev = json.load(open(f"{run_dir}/metrics.json")) if os.path.exists(f"{run_dir}/metrics.json") else {}  # resume after preemption
    swift = None
    if args.model == "swiftkv8b":   # public distilled checkpoint, evaluated as is
        tok, model, swift = load_swiftkv()
    else:
        tok, model = load(args.model)
    eng = Engine(model, tok, swift)
    L, H, KV, hd = dims(model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gc = model.generation_config.eos_token_id
    stops = set([tok.eos_token_id] + (gc if isinstance(gc, list) else [gc]))
    for s_ in ("<|im_end|>", "<|eot_id|>"):
        i = tok.convert_tokens_to_ids(s_)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            stops.add(i)
    stop_ids = torch.tensor(sorted(x for x in stops if x is not None), device=DEV)
    js = [int(x) for x in args.js.split(",")] if swift is None else []
    # calibration / selection: identical to round2 (features x alpha x gamma on held-out text)
    calib = [tok(t, add_special_tokens=False)["input_ids"][:1280] for t in wiki_texts("train-00000-of-00002.parquet", 40000, 160, 1)]
    for u, r in chat_pairs("calib", 1024):
        p, a = chat_ids(tok, u, r); calib.append(p + a)
    gammas = sorted(float(x) for x in args.gammas.split(","))
    sq = {} if gammas != [0.0] else None
    specs = ([(j, tuple(x for x in (j, j - 4, j - 8) if x >= 1)) for j in js] if args.feat == "both" else []) + [(j, (j,)) for j in js]
    acc = fit_grams(model, calib, specs, c_dtype=torch.float32 if args.c_fp32 else torch.float64, sq=sq) if specs else {}
    rng = np.random.default_rng(5); val = []
    for t in wiki_texts("validation-00000-of-00001.parquet", None, 8, 2):
        ids_ = tok(t, add_special_tokens=False)["input_ids"][:1280]
        val += [ids_[:c] for c in rng.integers(64, len(ids_), size=6)]
    for u, r in chat_pairs("val", 16):
        p, a = chat_ids(tok, u, r); val += [(p + a)[:c] for c in rng.integers(len(p), len(p) + len(a), size=3)]
    res = {"run_id": args.run_id, "model": args.model, "args": vars(args), "chosen": {}, "results": {}}
    chosen = {}
    for j in js:
        best = None
        for spec in [s for s in specs if s[0] == j]:
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
                    if best is None or kl < best[0]:
                        best = (kl, spec, alpha, g, W.clone())
                del W
        chosen[j] = best
        res["chosen"][j] = {"feat": list(best[1][1]), "alpha": best[2], "gamma": best[3], "val_kl": best[0]}
        log(f"j={j}: feats {best[1][1]} alpha {best[2]:g} gamma {best[3]:g} val KL {best[0]:.4f}")
    del acc; torch.cuda.empty_cache()
    bi = block_influence(model, calib[:64])
    order = sorted(range(L), key=lambda l: bi[l])
    res["block_influence"] = bi
    prune = [float(x) for x in args.prune.split(",") if x] if swift is None else []

    tasks = {}
    if "math500" in args.tasks:
        tasks["math500"] = (math_items(tok, args.n_math), 1024, lambda t, g: float(math_ok(t, g)))
    if "humaneval" in args.tasks:
        tasks["humaneval"] = (humaneval_items(tok, args.n_he), 512, lambda t, g: float(humaneval_ok(t, g)))
    if "cnndm" in args.tasks:
        tasks["cnndm"] = (cnn_items(tok, args.n_cnn, args.cnn_tokens), 160, lambda t, g: rouge_l(t, g))
    for tname, (items, max_new, score) in tasks.items():
        gen = [(ids, "nan") for ids, _ in items]
        variants = [("exact", {"j": 999, "mode": "exact"})]
        if swift is not None:
            variants.append(("swiftkv", {"j": swift["j"], "mode": "swiftkv"}))
        for j in js:
            for mode in ("naive", "drop", "predict"):
                v = {"j": j, "mode": mode, "w": 0}
                if mode == "predict":
                    v.update(W=chosen[j][4], fl=list(chosen[j][1][1]))
                variants.append((f"{mode}_j{j}_w0", v))
        for f in prune:
            variants.append((f"prune{int(round(100 * f))}", {"prune": order[:int(round(f * L))]}))
        done = prev.get("results", {}).get(tname, {})
        res["results"][tname] = {**done, "n": len(items), "mean_prompt_tokens": float(np.mean([len(x[0]) for x in items]))}
        for name, v in variants:
            if name in done:
                log(f"{tname.upper()} {name:16s} score {100 * done[name]['score']:.1f} (resumed)"); continue
            t0 = time.time()
            if "prune" in v:
                with Pruned(model, v["prune"]) as pm:
                    e2 = Engine(pm, tok)
                    _, _, texts = run_gsm8k(e2, gen, {"j": 999, "mode": "exact"}, pad_id, stop_ids, max_new=max_new, bs=args.bs)
                saved = len(v["prune"]) / L
            else:
                _, _, texts = run_gsm8k(eng, gen, v, pad_id, stop_ids, max_new=max_new, bs=args.bs)
                saved = flop_saving(items, v, L)
            sc = [score(t, items[i][1]) for i, t in enumerate(texts)]
            res["results"][tname][name] = {"score": float(np.mean(sc)), "items": sc, "saved": saved}
            json.dump({"texts": texts}, open(f"{run_dir}/{tname}_{name}.json", "w"))
            log(f"{tname.upper()} {name:16s} score {100 * np.mean(sc):.1f} saved {100 * saved:.0f}% ({time.time() - t0:.0f}s)")
            json.dump(res, open(f"{run_dir}/metrics.json", "w"))
    log("DONE")


if __name__ == "__main__":
    main()

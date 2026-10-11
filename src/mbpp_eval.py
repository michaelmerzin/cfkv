"""MBPP code generation (pass@1, greedy): generated code is executed against the problem's asserts in a
subprocess with a timeout. Same fit/selection recipe as round2; variants exact / POP / drop / CF-KV at w=0."""
import argparse, json, os, re, subprocess, sys, tempfile, time
import numpy as np, pyarrow.parquet as pq, torch
sys.path.insert(0, os.path.dirname(__file__))
from engine import DEV, ROOT, HUB, snap, load, load_swiftkv, Engine, fit_grams, solve
from round2 import wiki_texts, chat_pairs, chat_ids, last_kl, run_gsm8k


def mbpp_items(tok, n):
    rows = pq.read_table(snap(HUB, "datasets--google-research-datasets--mbpp") + "/full/test-00000-of-00001.parquet").to_pylist()[:n]
    out = []
    for r in rows:
        q = (f"{r['text']}\nYour code should pass these tests:\n" + "\n".join(r["test_list"]) +
             "\nWrite the Python function in a single ```python code block.")
        out.append((chat_ids(tok, q), r))
    return out


def extract_code(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return m[0] if m else text


def passes(code, row, timeout=10):
    prog = (row.get("test_setup_code") or "") + "\n" + code + "\n" + "\n".join(row["test_list"]) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=os.environ.get("TMPDIR"), delete=False) as f:
        f.write(prog); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout, cwd=os.path.dirname(path))
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--js", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--bs", type=int, default=50)
    ap.add_argument("--feat", default="both", choices=["both", "single"])
    ap.add_argument("--c_fp32", type=int, default=0)
    args = ap.parse_args()
    run_dir = f"{ROOT}/results/raw/{args.run_id}"; os.makedirs(run_dir, exist_ok=True)
    prev = json.load(open(f"{run_dir}/metrics.json")) if os.path.exists(f"{run_dir}/metrics.json") else {}  # resume after preemption
    swift = None
    if args.model == "swiftkv8b":
        tok, model, swift = load_swiftkv()
    else:
        tok, model = load(args.model)
    eng = Engine(model, tok, swift)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gc = model.generation_config.eos_token_id
    stops = set([tok.eos_token_id] + (gc if isinstance(gc, list) else [gc]))
    for s_ in ("<|im_end|>", "<|eot_id|>"):
        i = tok.convert_tokens_to_ids(s_)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            stops.add(i)
    stop_ids = torch.tensor(sorted(x for x in stops if x is not None), device=DEV)
    js = [int(x) for x in args.js.split(",")] if swift is None else []
    calib = [tok(t, add_special_tokens=False)["input_ids"][:1280] for t in
             wiki_texts("train-00000-of-00002.parquet", 40000, 160, 1)]
    for u, r in chat_pairs("calib", 1024):
        p, a = chat_ids(tok, u, r); calib.append(p + a)
    specs = ([(j, tuple(x for x in (j, j - 4, j - 8) if x >= 1)) for j in js] if args.feat == "both" else []) + [(j, (j,)) for j in js]
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
    items = mbpp_items(tok, args.n)
    gen_items = [(ids, "nan") for ids, _ in items]  # run_gsm8k scores numeric answers; placeholder never matches
    variants = [("exact", {"j": 999, "mode": "exact"})]
    if swift is not None:
        variants.append(("swiftkv", {"j": swift["j"], "mode": "swiftkv"}))
    for j in js:
        for mode in ("naive", "drop", "predict"):
            v = {"j": j, "mode": mode, "w": 0}
            if mode == "predict":
                v.update(W=chosen[j][3], fl=list(chosen[j][1][1]))
            variants.append((f"{mode}_j{j}_w0", v))
    res = {"run_id": args.run_id, "model": args.model, "n": len(items), "results": dict(prev.get("results", {}))}
    for name, v in variants:
        if name in res["results"]:
            print(f"MBPP {name:16s} pass@1 {res['results'][name]['pass@1']*100:.1f} (resumed)", flush=True); continue
        t0 = time.time()
        _, _, texts = run_gsm8k(eng, gen_items, v, pad_id, stop_ids, max_new=512, bs=args.bs)
        ok = [int(passes(extract_code(t), items[i][1])) for i, t in enumerate(texts)]
        res["results"][name] = {"pass@1": float(np.mean(ok)), "items": ok}
        json.dump({"texts": texts}, open(f"{run_dir}/mbpp_{name}.json", "w"))
        print(f"MBPP {name:16s} pass@1 {np.mean(ok)*100:.1f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(f"{run_dir}/metrics.json", "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

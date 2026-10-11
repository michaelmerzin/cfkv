"""Round 2: selection on val, then real task accuracy (MMLU 5-shot prefill, GSM8K generation)."""
import argparse, json, os, re, sys, time
import numpy as np, pyarrow.parquet as pq, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from engine import (DEV, ROOT, HUB, PHUB, snap, load, load_swiftkv, dims, Engine, fit_grams, solve, left_pad,
                    context_mask)

WT = snap(HUB, "datasets--Salesforce--wikitext") + "/wikitext-103-raw-v1"


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------ data (text level, re-tokenized per model)
def wiki_texts(fname, max_rows, n, seed, chars=7000):
    t = pq.read_table(f"{WT}/{fname}", columns=["text"]).column("text").to_pylist()
    if max_rows:
        t = t[:max_rows]
    s = "".join(x for x in t if x.strip())
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.choice(len(s) - chars, size=n, replace=False))
    return [s[a:a + chars] for a in starts]


def chat_pairs(split, k):
    from transformers import AutoTokenizer
    qtok = AutoTokenizer.from_pretrained(snap(HUB, "models--Qwen--Qwen2.5-1.5B-Instruct"))
    recs = torch.load(f"{ROOT}/data/gen_{split}.pt")
    out = []
    for r in recs[:k]:
        ptxt = qtok.decode(r["prompt_ids"])
        user = ptxt.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        resp = qtok.decode(r["gen_ids"], skip_special_tokens=True)
        out.append((user, resp))
    return out


def chat_ids(tok, user, resp=None):
    msgs = [{"role": "user", "content": user}]
    if "Mistral-Small" in getattr(tok, "name_or_path", ""):   # its default system prompt embeds the run date and ~170 tokens
        msgs.insert(0, {"role": "system", "content": "You are a helpful assistant."})
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pid = tok(p, add_special_tokens=False)["input_ids"]
    return pid if resp is None else (pid, tok(resp, add_special_tokens=False)["input_ids"])


def mmlu_items(tok, n, seed=0):
    base = snap(PHUB, "datasets--cais--mmlu") + "/all"
    dev = pq.read_table(f"{base}/dev-00000-of-00001.parquet").to_pylist()
    test = pq.read_table(f"{base}/test-00000-of-00001.parquet").to_pylist()
    shots = {}
    for r in dev:
        shots.setdefault(r["subject"], []).append(r)
    def fmt(r, ans=True):
        s = r["question"].strip() + "\n" + "".join(f"{c}. {t}\n" for c, t in zip("ABCD", r["choices"])) + "Answer:"
        return s + (f" {'ABCD'[r['answer']]}\n\n" if ans else "")
    idx = np.random.default_rng(seed).choice(len(test), size=n, replace=False)
    items = []
    for i in sorted(idx):
        r = test[i]
        head = f"The following are multiple choice questions (with answers) about {r['subject'].replace('_', ' ')}.\n\n"
        txt = head + "".join(fmt(d) for d in shots[r["subject"]][:5]) + fmt(r, ans=False)
        items.append((tok(txt, add_special_tokens=True)["input_ids"], r["answer"]))
    return items


def gsm8k_items(tok, n):
    rows = pq.read_table(snap(HUB, "datasets--openai--gsm8k") + "/main/test-00000-of-00001.parquet").to_pylist()[:n]
    out = []
    for r in rows:
        q = r["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append((chat_ids(tok, q), gold))
    return out


def parse_num(text):
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    cand = m[-1] if m else text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", cand)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def same_num(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


# ------------------------------------------------------------ evaluation helpers
def batches(items, bs):
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    for i in range(0, len(order), bs):
        yield [order[k] for k in range(i, min(i + bs, len(order)))]


def tf_kl(eng, seqs, variant, pad_id):
    """teacher-forced: seqs = [(ids, c)], context < c-1 (window applies), compare logits at c-1..n-2."""
    kls, ags = [], []
    for ids_, c in seqs:
        ids, m = left_pad([ids_], pad_id)
        ctx = context_mask(m, variant.get("w", 0), ctx_end=torch.tensor([c], device=DEV))
        lx, _ = eng.prefill(ids, m, ctx, 999, "exact")
        lv, _ = eng.prefill(ids, m, ctx, variant["j"], variant["mode"], variant.get("W"), variant.get("fl"))
        a, b = lx[0, c - 1:-1].float(), lv[0, c - 1:-1].float()
        lpa, lpb = F.log_softmax(a, -1), F.log_softmax(b, -1)
        kls.append((lpa.exp() * (lpa - lpb)).sum(-1)); ags.append((a.argmax(-1) == b.argmax(-1)).float())
    return torch.cat(kls).mean().item(), torch.cat(ags).mean().item()


def last_kl(eng, cuts, variant, pad_id, bs=16):
    """cuts: list of token-id prefixes. Deployment semantics: only the final token (and window) exact.
    Returns mean KL and top-1 agreement at the final position vs the exact model."""
    kls, ags = [], []
    for i in range(0, len(cuts), bs):
        ids, m = left_pad(cuts[i:i + bs], pad_id)
        ctx = context_mask(m, variant.get("w", 0))
        lx, _ = eng.prefill(ids, m, ctx, 999, "exact", last_only=True)
        lv, _ = eng.prefill(ids, m, ctx, variant["j"], variant["mode"], variant.get("W"), variant.get("fl"),
                            last_only=True)
        a, b = lx[:, -1].float(), lv[:, -1].float()
        lpa, lpb = F.log_softmax(a, -1), F.log_softmax(b, -1)
        kls.append((lpa.exp() * (lpa - lpb)).sum(-1)); ags.append((a.argmax(-1) == b.argmax(-1)).float())
    return torch.cat(kls).mean().item(), torch.cat(ags).mean().item()


def flop_saving(items, variant, L):
    """fraction of prefill layer-token work skipped: ctx tokens * (L-j) / (prompt tokens * L)."""
    if variant["mode"] == "exact":
        return 0.0
    w, j = variant.get("w", 0), variant["j"]
    tot = sum(len(x[0]) for x in items)
    k = variant.get("keep", 0)
    ctx = sum(max(0, len(x[0]) - 2 - w) * (1 - k) if 0 < k < 1 else max(0, len(x[0]) - 2 - w - k) for x in items)
    return ctx * (L - j - variant.get("after", 0)) / (tot * L)


def run_mmlu(eng, items, variant, letter_ids, pad_id, bs=16):
    preds = [None] * len(items)
    for bi in batches(items, bs):
        ids, m = left_pad([items[i][0] for i in bi], pad_id)
        ctx = context_mask(m, variant.get("w", 0))
        kv_src = (eng.scaffold_kv(ids, m, variant["j"], variant.get("tmpl"), variant["filler"])
                  if variant["mode"] == "scaffold" else None)
        lg, _ = eng.prefill(ids, m, ctx, variant["j"], variant["mode"], variant.get("W"), variant.get("fl"),
                            keep=variant.get("keep", 0), last_only=True, kv_src=kv_src,
                            pool=variant.get("pool", 0), win=variant.get("win", 0), after=variant.get("after", 0))
        p = lg[:, -1, letter_ids].argmax(-1).tolist()
        for i, x in zip(bi, p):
            preds[i] = x
    acc = float(np.mean([preds[i] == items[i][1] for i in range(len(items))]))
    return acc, preds


def run_gsm8k(eng, items, variant, pad_id, stop_ids, max_new=400, bs=50):
    outs = [None] * len(items)
    for bi in batches(items, bs):
        ids, m = left_pad([items[i][0] for i in bi], pad_id)
        ctx = context_mask(m, variant.get("w", 0))
        kv_src = (eng.scaffold_kv(ids, m, variant["j"], variant.get("tmpl"), variant["filler"])
                  if variant["mode"] == "scaffold" else None)
        lg, cache = eng.prefill(ids, m, ctx, variant["j"], variant["mode"], variant.get("W"), variant.get("fl"),
                                max_new=max_new, keep=variant.get("keep", 0), last_only=True, kv_src=kv_src,
                            pool=variant.get("pool", 0), win=variant.get("win", 0), after=variant.get("after", 0))
        seq = eng.decode(lg[:, -1].argmax(-1), cache, max_new, stop_ids)
        for k, i in enumerate(bi):
            s = seq[k].tolist()
            cut = next((t for t, x in enumerate(s) if x in stop_ids.tolist()), len(s))
            outs[i] = s[:cut]
    texts = [eng.tok.decode(o) for o in outs]
    acc = float(np.mean([same_num(parse_num(t), items[i][1]) for i, t in enumerate(texts)]))
    return acc, outs, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen1.5b")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--js", default="10,14,19")
    ap.add_argument("--win_mmlu", default="0,64,256")
    ap.add_argument("--win_gsm8k", default="0,32")
    ap.add_argument("--modes", default="naive,drop,predict")
    ap.add_argument("--tasks", default="mmlu,gsm8k")
    ap.add_argument("--n_mmlu", type=int, default=1000)
    ap.add_argument("--n_gsm8k", type=int, default=500)
    ap.add_argument("--n_calib_wiki", type=int, default=160)
    ap.add_argument("--n_calib_chat", type=int, default=1024)
    ap.add_argument("--gsm_bs", type=int, default=50)
    ap.add_argument("--mmlu_bs", type=int, default=16)
    ap.add_argument("--scaffold", type=int, default=0, help="add the filler-scaffold KV control (2605.30574)")
    ap.add_argument("--ctxfeat", type=int, default=0, help="also fit contextual-feature specs")
    ap.add_argument("--feat", default="both", choices=["both", "single", "pair8"],
                    help="single: only [h_j]; pair8: only [h_j, h_{j-8}] (large models)")
    ap.add_argument("--c_fp32", type=int, default=0, help="accumulate the cross-moment matrix in fp32")
    ap.add_argument("--keeps", default="", help="attention-selected exact tokens (w=0), e.g. 64,256")
    ap.add_argument("--gammas", default="0", help="key-variance matching strengths to select over (kscale.py), e.g. 0,1")
    args = ap.parse_args()
    run_dir = f"{ROOT}/results/raw/{args.run_id}"; assert not os.path.exists(run_dir); os.makedirs(run_dir)
    t0 = time.time()
    swift = None
    if args.model == "swiftkv8b":
        tok, model, swift = load_swiftkv()
    else:
        tok, model = load(args.model)
    eng = Engine(model, tok, swift)
    L, KV, hd = eng.L, eng.KV, eng.hd
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stops = set([tok.eos_token_id] + (model.generation_config.eos_token_id if isinstance(
        model.generation_config.eos_token_id, list) else [model.generation_config.eos_token_id]))
    for s in ("<|im_end|>", "<|eot_id|>"):
        i = tok.convert_tokens_to_ids(s)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            stops.add(i)
    stop_ids = torch.tensor(sorted(x for x in stops if x is not None), device=DEV)
    js = [int(x) for x in args.js.split(",")]
    res = {"run_id": args.run_id, "model": args.model, "L": L, "js": js, "stop_ids": stop_ids.tolist()}

    # ---- engine validation vs HF
    probe = chat_ids(tok, "Explain why the sky is blue in two sentences.")
    ids, m = left_pad([probe, probe[5:]], pad_id)
    lg, _ = eng.prefill(ids, m, context_mask(m), 999, "exact")
    ref = model(input_ids=ids, attention_mask=m.long()).logits
    diff = (lg[0].float() - ref[0].float()).abs().max().item()  # row 0 unpadded (HF ignores left-pad positions)
    g_hf = model.generate(input_ids=ids[:1, :], max_new_tokens=40, do_sample=False, temperature=None, top_p=None,
                          top_k=None, repetition_penalty=1.0)[0, ids.shape[1]:].tolist()
    lg1, cache = eng.prefill(ids[:1], m[:1], context_mask(m[:1]), 999, "exact", max_new=40)
    g_me = eng.decode(lg1[:, -1].argmax(-1), cache, 40, stop_ids)[0].tolist()
    k = min(len(g_hf), len(g_me))
    res["engine_check"] = {"prefill_max_abs_logit_diff": diff, "greedy_prefix_match": sum(
        1 for a, b in zip(g_hf[:k], g_me[:k]) if a == b) / max(k, 1)}
    log("engine check:", res["engine_check"])

    if swift is not None:
        js = []
    # ---- calibration (generic: WikiText train + Alpaca chat), fit
    calib = [tok(t, add_special_tokens=False)["input_ids"][:1280] for t in
             wiki_texts("train-00000-of-00002.parquet", 40000, args.n_calib_wiki, 1)]
    for u, r in chat_pairs("calib", args.n_calib_chat):
        p, a = chat_ids(tok, u, r); calib.append(p + a)
    specs = []
    for j in js:
        if args.feat == "pair8":
            specs.append((j, (j, j - 8)))
            continue
        specs.append((j, (j,)))
        if args.feat == "single":
            continue
        specs.append((j, tuple(x for x in (j, j - 4, j - 8) if x >= 1)))
        if args.ctxfeat:
            specs.append((j, tuple(x for x in (j, j - 4, j - 8) if x >= 1) + ("ctx",)))
    tf = time.time()
    gammas = sorted(float(x) for x in args.gammas.split(","))
    sq = {} if gammas != [0.0] else None
    acc = fit_grams(model, calib, specs, c_dtype=torch.float32 if args.c_fp32 else torch.float64, sq=sq) if specs else {}
    torch.cuda.synchronize()
    res["fit_sec"] = time.time() - tf
    res["n_calib_positions"] = acc[specs[0]][2] if specs else 0
    log(f"fit grams for {len(specs)} specs over {res['n_calib_positions']} positions: {res['fit_sec']:.0f}s")

    # ---- selection on val only (teacher-forced KL, w=0)
    # deployment-faithful val: prefixes cut at random points (only the last token exact)
    rng = np.random.default_rng(5)
    val = []
    for t in wiki_texts("validation-00000-of-00001.parquet", None, 8, 2):
        ids_ = tok(t, add_special_tokens=False)["input_ids"][:1280]
        val += [ids_[:c] for c in rng.integers(64, len(ids_), size=6)]
    for u, r in chat_pairs("val", 16):
        p, a = chat_ids(tok, u, r)
        val += [(p + a)[:c] for c in rng.integers(len(p), len(p) + len(a), size=3)]
    chosen = {}
    res["selection"] = {}
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
                    kl, ag = last_kl(eng, val, {"j": j, "mode": "predict", "W": W, "fl": list(spec[1])}, pad_id)
                    tag = f"j{j}_f{'-'.join(map(str, spec[1]))}_a{alpha:g}" + (f"_g{g:g}" if sq is not None else "")
                    res["selection"][tag] = {"kl": kl, "top1": ag}
                    if best is None or kl < best[0]:
                        best = (kl, ag, spec, alpha, W.clone() if len(gammas) > 1 else W, g)
                del W
        chosen[j] = best
        for mode in ("naive", "drop"):
            kl, ag = last_kl(eng, val, {"j": j, "mode": mode}, pad_id)
            res["selection"][f"j{j}_{mode}"] = {"kl": kl, "top1": ag}
        log(f"j={j}: chosen feats {best[2][1]} alpha {best[3]:g} gamma {best[5]:g} val KL {best[0]:.4f} top1 {best[1]:.3f} | "
            f"naive KL {res['selection'][f'j{j}_naive']['kl']:.3f} drop KL {res['selection'][f'j{j}_drop']['kl']:.3f}")
        if args.feat == "both":
            del acc[(j, (j,))]
    res["chosen"] = {j: {"feat_layers": list(c[2][1]), "alpha": c[3], "gamma": c[5], "val_kl": c[0], "val_top1": c[1]} for j, c in chosen.items()}
    del acc  # fit statistics are no longer needed once W is solved; frees several GB on large models
    torch.cuda.empty_cache()
    json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)

    filler_ids = tok(" This is neutral filler text with no particular content.", add_special_tokens=False)["input_ids"]
    tmpl_ids = chat_ids(tok, "")   # empty-content chat template: its common prefix/suffix is kept (GSM8K)

    def make_variants(windows):
        vs = [("exact", {"j": 999, "mode": "exact"})]
        if swift is not None:
            vs.append(("swiftkv", {"j": swift["j"], "mode": "swiftkv"}))
        for j in js:
            for w in map(int, windows.split(",")):
                for mode in args.modes.split(","):
                    v = {"j": j, "mode": mode, "w": w}
                    if mode == "predict":
                        v.update(W=chosen[j][4], fl=list(chosen[j][2][1]))
                    vs.append((f"{mode}_j{j}_w{w}", v))
            if args.scaffold:
                for w in map(int, windows.split(",")):
                    vs.append((f"scaffold_j{j}_w{w}", {"j": j, "mode": "scaffold", "w": w, "filler": filler_ids,
                                                       "tmpl": tmpl_ids}))
            for kp in [int(x) for x in args.keeps.split(",") if x]:
                for mode in args.modes.split(","):
                    v = {"j": j, "mode": mode, "w": 0, "keep": kp}
                    if mode == "predict":
                        v.update(W=chosen[j][4], fl=list(chosen[j][2][1]))
                    vs.append((f"{mode}_j{j}_keep{kp}", v))
        return vs

    letter_ids = [tok(f" {c}", add_special_tokens=False)["input_ids"][-1] for c in "ABCD"]
    tasks = args.tasks.split(",")
    if "mmlu" in tasks:
        items = mmlu_items(tok, args.n_mmlu)
        res["mmlu"] = {"n": len(items), "mean_prompt_tokens": float(np.mean([len(i[0]) for i in items]))}
        ex_pred = None
        for name, v in make_variants(args.win_mmlu):
            ts = time.time()
            a, pr = run_mmlu(eng, items, v, letter_ids, pad_id, bs=args.mmlu_bs)
            if name == "exact":
                ex_pred = pr
            agree = float(np.mean([x == y for x, y in zip(pr, ex_pred)]))
            fs = flop_saving(items, v, L)
            res["mmlu"][name] = {"acc": a, "agree_with_exact": agree, "prefill_flop_saving": fs, "sec": time.time() - ts}
            log(f"MMLU {name:18s} acc {a:.4f} agree {agree:.3f} saved {fs:.1%} ({time.time()-ts:.0f}s)")
            json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)
    if "gsm8k" in tasks:
        items = gsm8k_items(tok, args.n_gsm8k)
        res["gsm8k"] = {"n": len(items), "mean_prompt_tokens": float(np.mean([len(i[0]) for i in items]))}
        ex_out = None
        for name, v in make_variants(args.win_gsm8k):
            ts = time.time()
            a, outs, texts = run_gsm8k(eng, items, v, pad_id, stop_ids, bs=args.gsm_bs)
            if name == "exact":
                ex_out = outs
            ident = float(np.mean([o == e for o, e in zip(outs, ex_out)]))
            fs = flop_saving(items, v, L)
            res["gsm8k"][name] = {"acc": a, "identical_to_exact": ident, "prefill_flop_saving": fs, "sec": time.time() - ts}
            json.dump({"texts": texts}, open(f"{run_dir}/gsm8k_{name}.json", "w"))
            log(f"GSM8K {name:18s} acc {a:.4f} identical {ident:.3f} saved {fs:.1%} ({time.time()-ts:.0f}s)")
            json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)
    res["runtime_sec"] = time.time() - t0; res["gpu"] = torch.cuda.get_device_name(0)
    json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)
    with open(f"{ROOT}/results/registry.jsonl", "a") as f:
        f.write(json.dumps({"run_id": args.run_id, "model": args.model,
                            "mmlu": {k: v["acc"] for k, v in res.get("mmlu", {}).items() if isinstance(v, dict)},
                            "gsm8k": {k: v["acc"] for k, v in res.get("gsm8k", {}).items() if isinstance(v, dict)}}) + "\n")
    log("DONE", res["runtime_sec"])


if __name__ == "__main__":
    main()

"""SCBench (Li et al. 2024, arXiv 2412.10319; microsoft/SCBench) multi-turn shared-context evaluation.

For each sample the shared context + the FIRST query is prefilled ONCE (the cached prompt). Turn 1 is decoded
from that cache; every later turn is appended exactly on top of the cache (teacher-forced), and decoded greedily.
Previous-turn answers are the GOLD answers teacher-forced into the conversation (deterministic, independent of
the method's earlier mistakes). All follow-up tokens and generated tokens are exact; only the prefilled prompt is
approximated (j, w=0), as in reuse_test.py / reuse2.py.

Methods: exact; cfkv (predict); pop (naive); select{k}+drop and select{k}+predict (SnapKV-style selection computed
at prefill, i.e. it only sees the FIRST query).

Tasks (subset of SCBench, prompts follow SCBench/InfiniteBench style):
  kv          scbench_kv          Retr.KV: UUID key->value lookup in JSON. Metric: gold value string in output.
                                  Shortening: the JSON is SUBSAMPLED (all queried pairs kept, other pairs dropped
                                  uniformly at random, original order kept) to fit --max_ctx; not middle-truncated.
  qa_eng      scbench_qa_eng      En.QA on a novel. Metric: token F1 (LongBench/InfiniteBench qa_f1_score).
  choice_eng  scbench_choice_eng  En.MultiChoice (4 options). Metric: accuracy of the predicted letter.
  many_shot   scbench_many_shot   Many-shot ICL (BBH: date understanding, translation-error detection, shuffled
                                  objects). Metric: accuracy of the predicted option letter.
qa_eng / choice_eng / many_shot contexts longer than --max_ctx tokens are truncated in the MIDDLE (first half +
last half of the token budget), as SCBench does for its max_seq_length.
"""
import argparse, glob, json, os, re, sys, time
import numpy as np, pyarrow.parquet as pq, torch
sys.path.insert(0, os.path.dirname(__file__))
from engine import DEV, ROOT, load, Engine, fit_grams, solve, left_pad, context_mask, rope
from round2 import wiki_texts, chat_pairs, chat_ids, last_kl
from longctx import qa_f1, _norm

SCB = glob.glob(f"{ROOT}/cache/hf/hub/datasets--microsoft--SCBench/snapshots/*/")[0]
TASKS = {"kv": "scbench_kv", "qa_eng": "scbench_qa_eng", "choice_eng": "scbench_choice_eng",
         "many_shot": "scbench_many_shot"}
MAX_GEN = {"kv": 96, "qa_eng": 64, "choice_eng": 24, "many_shot": 24}


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ context shortening
def mid_truncate(tok, text, budget):
    text = text if len(text) <= budget * 12 else text[:budget * 6] + text[-budget * 6:]  # cheap pre-cut (chars)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return text, False
    return tok.decode(ids[:budget // 2]) + tok.decode(ids[-(budget - budget // 2):]), True


def kv_subsample(tok, text, keys, budget, seed):
    head = "JSON data:\n"
    assert text.startswith(head)
    d = json.loads(text[len(head):])
    items = list(d.items())
    ntok = len(tok(text, add_special_tokens=False)["input_ids"])
    if ntok <= budget:
        return text, False
    rng = np.random.default_rng(seed)
    qidx = {i for i, (k, _) in enumerate(items) if k in keys}
    assert len(qidx) == len(set(keys)), "queried key missing from JSON"
    others = np.array([i for i in range(len(items)) if i not in qidx])
    rng.shuffle(others)
    frac = budget / ntok
    while True:
        n = max(0, int(len(items) * frac) - len(qidx))
        keep = sorted(qidx | set(others[:n].tolist()))
        out = head + json.dumps({items[i][0]: items[i][1] for i in keep}, ensure_ascii=False)
        if len(tok(out, add_special_tokens=False)["input_ids"]) <= budget or n == 0:
            return out, True
        frac *= 0.97


# ------------------------------------------------------------------ prompts / gold / metrics
def strip_q(s):
    return s.strip().strip("'\"").strip()


def build_turns(task, r):
    """Returns (instruction+context template with {context}, list of (user_text_i, gold_answer_text_i, gold_i))."""
    turns = []
    for i, mt in enumerate(r["multi_turns"]):
        if task == "kv":
            q = mt["input"].strip()
            g = strip_q(mt["answer"]); turns.append((q, g, g))
        elif task == "qa_eng":
            q = f"Question: {strip_q(mt['input'])}\nAnswer the question concisely, based on the book."
            g = strip_q(mt["answer"]); turns.append((q, g, g))
        elif task == "choice_eng":
            opts = list(mt["options"])
            gold = strip_q(mt["answer"])
            gi = next((k for k, o in enumerate(opts) if strip_q(o) == gold), None)
            if gi is None:
                gi = next((k for k, o in enumerate(opts) if _norm(o) == _norm(gold)), None)
            if gi is None:
                return None
            q = (f"Question: {strip_q(mt['input'])}\n" + "".join(f"{c}. {o}\n" for c, o in zip("ABCD", opts)) +
                 "Answer with the letter of the correct option only.")
            turns.append((q, f"{'ABCD'[gi]}. {opts[gi]}", ("ABCD"[gi], opts[gi])))
        else:  # many_shot
            g = strip_q(mt["answer"])
            turns.append((mt["input"].strip(), f"({g})", g))
    return turns


def head_text(task, ctx, q1):
    if task == "kv":
        return f"Extract the value corresponding to the specified key in the JSON object below.\n\n{ctx}\n{q1}"
    if task in ("qa_eng", "choice_eng"):
        return f"Read the book and answer the questions about it.\n\n{ctx}\n\n{q1}"
    return f"{ctx}{q1}"   # many_shot: context already ends with "\n\n" after the last solved example


def score(task, pred, gold):
    if task == "kv":
        return float(gold in pred)
    if task == "qa_eng":
        return qa_f1(pred, [gold])
    if task == "choice_eng":
        letter, text = gold
        m = re.search(r"\b([A-D])\b", pred)
        if m:
            return float(m.group(1) == letter)
        return float(_norm(text) in _norm(pred))
    m = re.search(r"\(([A-Z])\)", pred) or re.search(r"\b([A-Z])\b", pred)
    return float(m is not None and m.group(1) == gold)


def load_items(tok, task, n, max_ctx, max_turns):
    rows = pq.read_table(f"{SCB}/{TASKS[task]}/test-00000-of-00001.parquet").to_pylist()
    idx = np.unique(np.linspace(0, len(rows) - 1, min(n, len(rows))).round().astype(int))  # spread over tasks
    items = []
    for i in idx:
        r = rows[i]
        turns = build_turns(task, r)
        if not turns:
            continue
        turns = turns[:max_turns]
        if task == "kv":
            keys = [re.search(r'Key: "([^"]+)"', t[0]).group(1) for t in turns]
            ctx, trunc = kv_subsample(tok, r["context"], keys, max_ctx, int(i))
        else:
            ctx, trunc = mid_truncate(tok, r["context"], max_ctx)
        msgs = [{"role": "user", "content": head_text(task, ctx, turns[0][0])}]
        prefix = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                     add_special_tokens=False)["input_ids"]
        prev, tails, ok = prefix, [], True
        for q, gtxt, _ in turns[1:]:
            msgs += [{"role": "assistant", "content": turns[len(tails)][1]}, {"role": "user", "content": q}]
            full = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                       add_special_tokens=False)["input_ids"]
            if full[:len(prev)] != prev:   # template must extend the previous prompt exactly
                ok = False; break
            tails.append(full[len(prev):]); prev = full
        if not ok:
            log(f"  skip {task} row {i}: chat template does not extend the prefix"); continue
        items.append({"row": int(i), "prefix": prefix, "tails": tails, "golds": [t[2] for t in turns],
                      "truncated": trunc})
    return items


# ------------------------------------------------------------------ chunked exact extension
@torch.no_grad()
def extend_chunk(eng, tokens, cache):
    """Same semantics as Engine.extend (teacher-force tokens exactly on top of the cache), but the chunk is
    processed in one causal pass per layer instead of token by token."""
    model, L, H, KV, hd = eng.model, eng.L, eng.H, eng.KV, eng.hd
    Kc, Vc, km, pos, t = cache
    n = len(tokens)
    h = model.model.embed_tokens(torch.tensor([tokens], device=DEV))
    p = pos[:, None] + torch.arange(n, device=DEV)[None]
    cos, sin = model.model.rotary_emb(h, p)
    col = torch.arange(t + n, device=DEV)
    causal = col[None, :] <= (t + torch.arange(n, device=DEV))[:, None]           # [n, t+n]
    for l in range(L):
        layer = model.model.layers[l]; att = layer.self_attn
        x = layer.input_layernorm(h)
        q = rope(att.q_proj(x).view(1, n, H, hd).transpose(1, 2), cos, sin)
        k = rope(att.k_proj(x).view(1, n, KV, hd).transpose(1, 2), cos, sin)
        v = att.v_proj(x).view(1, n, KV, hd).transpose(1, 2)
        Kc[l][:, :, t:t + n] = k; Vc[l][:, :, t:t + n] = v; km[l][:, t:t + n] = True
        allowed = km[l][:, None, :t + n] & causal[None]
        o_ = eng._attend(q, Kc[l][:, :, :t + n], Vc[l][:, :, :t + n], allowed, att)
        a = h + att.o_proj(o_.transpose(1, 2).reshape(1, n, H * hd))
        h = a + layer.mlp(layer.post_attention_layernorm(a))
    logits = model.lm_head(model.model.norm(h[:, -1:]))[:, 0]
    return logits, (Kc, Vc, km, pos + n, t + n)


def cut_stop(seq, stops):
    return seq[:next((i for i, x in enumerate(seq) if x in stops), len(seq))]


@torch.no_grad()
def run_sample(eng, it, v, stop_ids, pad_id, max_gen):
    """Prefill the shared prompt once, then answer every turn on top of the same cache. Returns per-turn texts."""
    stops = set(stop_ids.tolist())
    ids, m = left_pad([it["prefix"]], pad_id)
    ctx = context_mask(m, 0)
    extra = sum(len(t) for t in it["tails"]) + max_gen + 1
    lg, cache = eng.prefill(ids, m, ctx, v["j"], v["mode"], v.get("W"), v.get("fl"), max_new=extra,
                            keep=v.get("keep", 0), last_only=True)
    lg = lg[:, -1]
    outs = []
    for ti in range(len(it["golds"])):
        if ti > 0:
            lg, cache = extend_chunk(eng, it["tails"][ti - 1], cache)
        # decode writes its tokens past cache[4] but does not advance the cache tuple; the next extend overwrites
        # those slots with the gold answer (and attention only reads slots < t+1), so this is exact teacher forcing
        seq = eng.decode(lg.argmax(-1), cache, max_gen, stop_ids)[0].tolist()
        outs.append(eng.tok.decode(cut_stop(seq, stops)))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama8b")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--j", type=int, default=16)
    ap.add_argument("--tasks", default="kv,qa_eng,choice_eng,many_shot")
    ap.add_argument("--n", type=int, default=24, help="samples per task (spread evenly over the test split)")
    ap.add_argument("--max_turns", type=int, default=5)
    ap.add_argument("--keeps", default="256,2048")
    ap.add_argument("--max_ctx", type=int, default=32000, help="token budget for the shared context")
    ap.add_argument("--calib_long_len", type=int, default=8192,
                    help="token length of the 16 long calibration excerpts (8192 = reuse_test recipe; lower only for "
                         "small-GPU smoke tests)")
    ap.add_argument("--methods", default="", help="optional comma list to restrict methods (e.g. exact,cfkv)")
    args = ap.parse_args()
    run_dir = f"{ROOT}/results/raw/{args.run_id}"; os.makedirs(run_dir, exist_ok=True)
    prev = json.load(open(f"{run_dir}/metrics.json")) if os.path.exists(f"{run_dir}/metrics.json") else {}  # resume after preemption
    t0 = time.time()
    tok, model = load(args.model); eng = Engine(model, tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gc = model.generation_config.eos_token_id
    stops = set([tok.eos_token_id] + (gc if isinstance(gc, list) else [gc]))
    for s_ in ("<|im_end|>", "<|eot_id|>"):
        i = tok.convert_tokens_to_ids(s_)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            stops.add(i)
    stop_ids = torch.tensor(sorted(x for x in stops if x is not None), device=DEV)
    tasks = args.tasks.split(",")
    items = {}
    for task in tasks:
        items[task] = load_items(tok, task, args.n, args.max_ctx, args.max_turns)
        log(f"{task}: {len(items[task])} samples, prompt tokens mean "
            f"{np.mean([len(x['prefix']) for x in items[task]]):.0f} max {max(len(x['prefix']) for x in items[task])}, "
            f"turns {sum(len(x['golds']) for x in items[task])}, shortened {sum(x['truncated'] for x in items[task])}")
    res = {"run_id": args.run_id, "model": args.model, "j": args.j, "max_ctx": args.max_ctx,
           "max_turns": args.max_turns, "previous_turns": "gold teacher-forced", "tasks": {}}

    # ---- chunked extend == Engine.extend check (exact mode, short prefix)
    it0 = items[tasks[0]][0]
    short = it0["prefix"][-512:]; tail = (it0["tails"][0] if it0["tails"] else short[-16:])[:24]
    outs_ = []
    for fn in (eng.extend, lambda tk, c: extend_chunk(eng, tk, c)):
        ids, m = left_pad([short], pad_id)
        _, c = eng.prefill(ids, m, context_mask(m), 999, "exact", max_new=len(tail) + 1, last_only=True)
        outs_.append(fn(tail, c)[0].float())
    res["extend_check_max_abs_logit_diff"] = (outs_[0] - outs_[1]).abs().max().item()
    res["extend_check_argmax_equal"] = bool(outs_[0].argmax() == outs_[1].argmax())
    log(f"extend check: max|dlogit| {res['extend_check_max_abs_logit_diff']:.3f} "
        f"argmax equal {res['extend_check_argmax_equal']}")

    # ---- calibration / validation (identical to reuse_test.py)
    j = args.j; fl = tuple(x for x in (j, j - 4, j - 8) if x >= 1) + ("ctx",)
    calib = [tok(t, add_special_tokens=False)["input_ids"][:1280] for t in wiki_texts("train-00000-of-00002.parquet", 40000, 160, 1)]
    calib += [tok(t, add_special_tokens=False)["input_ids"][:args.calib_long_len] for t in wiki_texts("train-00000-of-00002.parquet", 40000, 16, 21, chars=45000)]
    for u, r in chat_pairs("calib", 1024):
        p, a = chat_ids(tok, u, r); calib.append(p + a)
    acc = fit_grams(model, calib, [(j, fl)]); G, C, _ = acc[(j, fl)]; del acc
    rng = np.random.default_rng(5); val = []
    for t in wiki_texts("validation-00000-of-00001.parquet", None, 8, 2):
        ids_ = tok(t, add_special_tokens=False)["input_ids"][:1280]
        val += [ids_[:c] for c in rng.integers(64, len(ids_), size=6)]
    for u, r in chat_pairs("val", 16):
        p, a = chat_ids(tok, u, r); val += [(p + a)[:c] for c in rng.integers(len(p), len(p) + len(a), size=3)]
    best = None
    for alpha in (1e-5, 1e-4, 1e-3):
        W = solve(G, C, alpha)
        kl, _ = last_kl(eng, val, {"j": j, "mode": "predict", "W": W, "fl": list(fl)}, pad_id)
        if best is None or kl < best[0]:
            best = (kl, alpha, W)
    del G, C; torch.cuda.empty_cache()
    W = best[2]
    res.update(alpha=best[1], val_kl=best[0], calib_long_len=args.calib_long_len)
    log(f"j={j} alpha {best[1]:g} val KL {best[0]:.4f} (fit+select {time.time()-t0:.0f}s)")

    variants = [("exact", {"j": 999, "mode": "exact"}), ("cfkv", {"j": j, "mode": "predict", "W": W, "fl": list(fl)}),
                ("pop", {"j": j, "mode": "naive"})]
    for k in map(int, args.keeps.split(",")):
        variants += [(f"select{k}+drop", {"j": j, "mode": "drop", "keep": k}),
                     (f"select{k}+predict", {"j": j, "mode": "predict", "keep": k, "W": W, "fl": list(fl)})]
    if args.methods:
        variants = [x for x in variants if x[0] in args.methods.split(",")]
    for task in tasks:
        R = res["tasks"][task] = {"n": len(items[task]), "rows": [x["row"] for x in items[task]],
                                  "mean_prompt_tokens": float(np.mean([len(x["prefix"]) for x in items[task]])),
                                  "n_shortened": int(sum(x["truncated"] for x in items[task])),
                                  "methods": dict(prev.get("tasks", {}).get(task, {}).get("methods", {}))}
        pf = f"{run_dir}/preds_{task}.json"
        preds = json.load(open(pf)).get("preds", {}) if os.path.exists(pf) else {}
        for name, v in variants:
            if name in R["methods"]:
                log(f"{task:10s} {name:18s} all {R['methods'][name]['all_turns']*100:5.1f} (resumed)"); continue
            ts = time.time(); per_turn = {}; texts = []
            for it in items[task]:
                outs = run_sample(eng, it, v, stop_ids, pad_id, MAX_GEN[task])
                texts.append(outs)
                for ti, (o, g) in enumerate(zip(outs, it["golds"])):
                    per_turn.setdefault(ti + 1, []).append(score(task, o, g))
            allsc = [s for x in per_turn.values() for s in x]
            follow = [s for t_, x in per_turn.items() if t_ > 1 for s in x]
            R["methods"][name] = {"all_turns": float(np.mean(allsc)),
                                  "turn1": float(np.mean(per_turn[1])),
                                  "followup": float(np.mean(follow)) if follow else None,
                                  "per_turn": {str(t_): float(np.mean(x)) for t_, x in sorted(per_turn.items())},
                                  "sec": time.time() - ts}
            preds[name] = texts
            fu = R["methods"][name]["followup"]
            log(f"{task:10s} {name:18s} all {np.mean(allsc)*100:5.1f} | turn1 {np.mean(per_turn[1])*100:5.1f} | "
                f"followup {'-' if fu is None else f'{fu*100:5.1f}'} | per-turn " +
                " ".join(f"{np.mean(x)*100:.0f}" for _, x in sorted(per_turn.items())) + f" ({time.time()-ts:.0f}s)")
            json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)
            json.dump({"golds": [[g if isinstance(g, str) else list(g) for g in it["golds"]] for it in items[task]],
                       "preds": preds}, open(f"{run_dir}/preds_{task}.json", "w"), indent=1)
    res["runtime_sec"] = time.time() - t0; res["gpu"] = torch.cuda.get_device_name(0)
    json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)
    log("DONE", round(res["runtime_sec"]))


if __name__ == "__main__":
    main()

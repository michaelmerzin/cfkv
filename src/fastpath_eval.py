"""Benchmark quality THROUGH the timed compute-saving implementation (fastpath.prefill_fast), with latency measured on the same prompts.

For one model and depth: select the CF-KV configuration exactly as main_eval.py does (features, alpha, gamma on held-out text), then
  MMLU (first n questions, 5-shot): last-token letter prediction from prefill_fast vs. from Engine.prefill (the evaluation engine of
    Table 1), accuracy of both, agreement, and per-prompt prefill latency (fast path vs. the exact HF forward);
  GSM8K (first n problems): prefill_fast + greedy decoding from its cache (Engine.decode), accuracy vs. the evaluation engine on the
    same problems, answer agreement, and per-prompt prefill latency.
Both at every requested exact window w. Batch 1 (the fast path is a batch-1 schedule).
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from engine import DEV, ROOT, load, Engine, dims, left_pad, context_mask
from round2 import log, mmlu_items, gsm8k_items, run_mmlu, run_gsm8k, parse_num, same_num
from fastpath import prefill_fast
from main_eval import select_config


def cuda_ms(fn):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); s.record(); out = fn(); e.record(); torch.cuda.synchronize()
    return out, s.elapsed_time(e)


def to_engine_cache(cache, P, max_new, KV, hd, dtype):
    """fast-path cache (per-layer K/V over all P prompt positions, RoPE applied) -> Engine.decode's preallocated cache."""
    Ks, Vs, _, _ = cache
    T = P + max_new
    Kc, Vc, km = [], [], []
    for K, V in zip(Ks, Vs):
        k = torch.zeros(1, KV, T, hd, device=DEV, dtype=dtype); v = torch.zeros_like(k)
        k[:, :, :P] = K; v[:, :, :P] = V
        m = torch.zeros(1, T, dtype=torch.bool, device=DEV); m[:, :P] = True
        Kc.append(k); Vc.append(v); km.append(m)
    return Kc, Vc, km, torch.tensor([P], device=DEV), P


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen7b")
    ap.add_argument("--j", type=int, default=14)
    ap.add_argument("--windows", default="0,64")
    ap.add_argument("--n_mmlu", type=int, default=1000)
    ap.add_argument("--n_gsm", type=int, default=200)
    ap.add_argument("--run_id", required=True)
    args = ap.parse_args()
    run_dir = f"{ROOT}/results/raw/{args.run_id}"; os.makedirs(run_dir, exist_ok=True)
    tok, model = load(args.model); eng = Engine(model, tok)
    L, H, KV, hd = dims(model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gc = model.generation_config.eos_token_id
    stops = set([tok.eos_token_id] + (gc if isinstance(gc, list) else [gc]))
    for s_ in ("<|im_end|>", "<|eot_id|>"):
        i = tok.convert_tokens_to_ids(s_)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            stops.add(i)
    stop_ids = torch.tensor(sorted(x for x in stops if x is not None), device=DEV)
    j = args.j
    best = select_config(model, eng, tok, pad_id, j, "both", [0.0, 1.0], 0)
    W32, fl = best["W"], list(best["feat"]); Wb = W32.to(torch.bfloat16)
    res = {"run_id": args.run_id, "model": args.model, "j": j, "config": {k: v for k, v in best.items() if k != "W"}, "mmlu": {}, "gsm8k": {}}
    letter4 = [tok(f" {c}", add_special_tokens=False)["input_ids"][-1] for c in "ABCD"]
    mm = mmlu_items(tok, 1000)[:args.n_mmlu]; gs = gsm8k_items(tok, 500)[:args.n_gsm]
    # warm-up (kernels, allocator)
    ids0 = torch.tensor([mm[0][0]], device=DEV)
    for _ in range(3):
        prefill_fast(model, ids0, j, Wb, fl, 0); model.model(input_ids=ids0, use_cache=True)
    for w in [int(x) for x in args.windows.split(",")]:
        v = {"j": j, "mode": "predict", "w": w, "W": W32, "fl": fl}
        # ---- MMLU
        t0 = time.time()
        _, pe = run_mmlu(eng, mm, v, letter4, pad_id)
        pf, tf, tx = [], [], []
        for ids, ans in mm:
            x = torch.tensor([ids], device=DEV)
            (lg, _), ms = cuda_ms(lambda: prefill_fast(model, x, j, Wb, fl, w)); tf.append(ms)
            pf.append(int(lg[0, -1, letter4].argmax()))
            _, ms = cuda_ms(lambda: model.model(input_ids=x, use_cache=True)); tx.append(ms)
        acc_f = float(np.mean([pf[i] == mm[i][1] for i in range(len(mm))])); acc_e = float(np.mean([pe[i] == mm[i][1] for i in range(len(mm))]))
        agree = float(np.mean([pf[i] == pe[i] for i in range(len(mm))]))
        res["mmlu"][str(w)] = {"n": len(mm), "acc_fastpath": acc_f, "acc_engine": acc_e, "agreement": agree,
                               "prefill_ms_fast_median": float(np.median(tf)), "prefill_ms_exact_median": float(np.median(tx)),
                               "mean_prompt_tokens": float(np.mean([len(x[0]) for x in mm])), "items_fast": pf}
        log(f"MMLU w={w}: fast path {100*acc_f:.1f} | engine {100*acc_e:.1f} | agreement {100*agree:.1f}% | prefill median "
            f"{np.median(tf):.1f} ms vs exact {np.median(tx):.1f} ms ({np.median(tx)/np.median(tf):.2f}x) ({time.time()-t0:.0f}s)")
        json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)
        # ---- GSM8K
        t0 = time.time()
        _, _, te = run_gsm8k(eng, gs, v, pad_id, stop_ids, max_new=400, bs=25)
        texts, tf = [], []
        for ids, _a in gs:
            x = torch.tensor([ids], device=DEV); P = x.shape[1]
            (lg, cache), ms = cuda_ms(lambda: prefill_fast(model, x, j, Wb, fl, w)); tf.append(ms)
            first = lg[0, -1].argmax()[None]
            seq = eng.decode(first, to_engine_cache(cache, P, 400, KV, hd, cache[0][0].dtype), 400, stop_ids)[0].tolist()
            cut = next((t for t, z in enumerate(seq) if z in stop_ids.tolist()), len(seq))
            texts.append(tok.decode(seq[:cut]))
        sf = [int(same_num(parse_num(t), gs[i][1])) for i, t in enumerate(texts)]
        se = [int(same_num(parse_num(t), gs[i][1])) for i, t in enumerate(te)]
        def _same(a, b):
            pa, pb = parse_num(a), parse_num(b)
            if pa is None or pb is None:
                return pa is None and pb is None
            return bool(same_num(pa, pb))
        ans_agree = float(np.mean([_same(a, b) for a, b in zip(texts, te)]))
        res["gsm8k"][str(w)] = {"n": len(gs), "acc_fastpath": float(np.mean(sf)), "acc_engine": float(np.mean(se)),
                                "answer_agreement": ans_agree, "prefill_ms_fast_median": float(np.median(tf)), "items_fast": sf}
        log(f"GSM8K w={w}: fast path {100*np.mean(sf):.1f} | engine {100*np.mean(se):.1f} | answer agreement {100*ans_agree:.1f}% | "
            f"prefill median {np.median(tf):.1f} ms ({time.time()-t0:.0f}s)")
        json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)
    log("DONE")


if __name__ == "__main__":
    main()

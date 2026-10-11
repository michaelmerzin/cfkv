"""Smoke test for lazyllm_proto / specprefill_proto on GSM8K prompts (TITAN V ok).
  1. no-pruning identity: lazy_prefill(schedule=[]) and spec_prefill(keep=1) vs Engine.prefill('exact')
     (logit diff + greedy decode match), batched (left-padded) and B=1;
  2. ~49%-saving settings: realized saving + GSM8K accuracy on a handful of prompts vs exact and FastKV-style drop."""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from engine import DEV, ROOT, PHUB, MODELS, snap, Engine, left_pad, context_mask
from round2 import gsm8k_items, parse_num, same_num
from lazyllm_proto import lazy_prefill, lazy_schedule, lazy_saving
from specprefill_proto import spec_prefill, spec_keep_for, nonemb_params

DRAFTS = {"qwen0.5b": lambda: snap(PHUB, "models--Qwen--Qwen2.5-0.5B-Instruct"),
          "llama1b": lambda: snap(PHUB, "models--unsloth--Llama-3.2-1B-Instruct")}


def load_path(path, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map=DEV, attn_implementation="sdpa").eval()
    model.requires_grad_(False); torch.set_grad_enabled(False)
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen1.5b"); ap.add_argument("--draft", default="qwen0.5b")
    ap.add_argument("--dtype", default="float32"); ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=320); ap.add_argument("--target", type=float, default=0.49)
    a = ap.parse_args()
    dt = getattr(torch, a.dtype)
    t0 = time.time()
    tok, model = load_path(MODELS[a.model](), dt); eng = Engine(model, tok)
    dtok, dmodel = load_path(DRAFTS[a.draft](), dt); deng = Engine(dmodel, dtok)
    print("loaded", time.time() - t0, torch.cuda.get_device_name(), a.dtype, flush=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stops = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    stop_ids = torch.tensor(sorted(s for s in stops if isinstance(s, int) and s >= 0), device=DEV)
    items = gsm8k_items(tok, a.n)
    txt = tok.decode(items[0][0])
    assert dtok(txt, add_special_tokens=False)["input_ids"] == items[0][0], "draft tokenizer mismatch"
    L = eng.L
    res = {"L": L, "dtype": a.dtype, "prompt_lens": [len(x[0]) for x in items]}

    # ---- 1. identity checks
    for B in (1, 4):
        ids, m = left_pad([x[0] for x in items[:B]], pad)
        lx, cx = eng.prefill(ids, m, context_mask(m), 999, "exact", max_new=48, last_only=True)
        gx = eng.decode(lx[:, -1].argmax(-1), cx, 48, stop_ids)
        for name, fn in (("lazy_empty", lambda: lazy_prefill(eng, ids, m, [], max_new=48, last_only=True)),
                         ("spec_keep1", lambda: spec_prefill(eng, deng, ids, m, 1.0, max_new=48, last_only=True,
                                                             stop_ids=stop_ids, pad_id=pad))):
            lv, cv = fn()
            gv = eng.decode(lv[:, -1].argmax(-1), cv, 48, stop_ids)
            res[f"id_{name}_B{B}"] = {"max_abs_logit_diff": (lv.float() - lx.float()).abs().max().item(),
                                      "decode_identical": bool(gv.shape == gx.shape and (gv == gx).all())}
            print(name, B, res[f"id_{name}_B{B}"], flush=True)

    # ---- 2. matched ~49% saving
    sch = lazy_schedule(L, a.target)
    c_draft = nonemb_params(dmodel) / nonemb_params(model)
    Pm = sum(res["prompt_lens"]) / len(items)
    keep_net = spec_keep_for(a.target, c_draft, Pm)
    res.update(lazy_sched=sch, lazy_nominal=lazy_saving(sch, L), c_draft=c_draft, spec_keep_net=keep_net)
    print("schedule", sch, "c_draft", c_draft, "keep_net", keep_net, flush=True)
    j = L // 2
    variants = {"exact": None, f"drop_j{j}": None, "lazy49": None, "spec_gross49": 1 - a.target,
                "spec_net49": keep_net}
    ids_all = [x[0] for x in items]
    for name in variants:
        ts = time.time()
        ids, m = left_pad(ids_all, pad)
        if name == "exact":
            lg, c = eng.prefill(ids, m, context_mask(m), 999, "exact", max_new=a.max_new, last_only=True); info = {"saving": 0}
        elif name.startswith("drop"):
            ctx = context_mask(m)
            lg, c = eng.prefill(ids, m, ctx, j, "drop", max_new=a.max_new, last_only=True)
            info = {"saving": float(ctx.sum()) * (L - j) / (float(m.sum()) * L)}
        elif name == "lazy49":
            lg, c = lazy_prefill(eng, ids, m, sch, max_new=a.max_new, last_only=True); info = dict(eng.lazy_info)
        else:
            lg, c = spec_prefill(eng, deng, ids, m, variants[name], max_new=a.max_new, last_only=True,
                                 stop_ids=stop_ids, pad_id=pad); info = dict(eng.spec_info)
        seq = eng.decode(lg[:, -1].argmax(-1), c, a.max_new, stop_ids)
        outs = []
        for s in seq.tolist():
            cut = next((t for t, x in enumerate(s) if x in stop_ids.tolist()), len(s)); outs.append(tok.decode(s[:cut]))
        acc = sum(same_num(parse_num(t), it[1]) for t, it in zip(outs, items)) / len(items)
        res[name] = {"acc": acc, **{k: v for k, v in info.items() if k != "active_frac_per_layer"},
                     "sec": time.time() - ts, "sample0_tail": outs[0][-160:]}
        print(name, json.dumps(res[name]), flush=True)
    os.makedirs(f"{ROOT}/results/baselines", exist_ok=True)
    json.dump(res, open(f"{ROOT}/results/baselines/smoke_{a.model}_{a.dtype}.json", "w"), indent=1)
    print("TOTAL", time.time() - t0)


if __name__ == "__main__":
    main()

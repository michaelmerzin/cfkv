"""Cache-reuse test v2: the follow-up question asks for a fact whose *form* differs from the first question's.

v1 (reuse_test.py) asked about a second magic-number needle, which shares the first needle's template, so tokens
selected for question A also covered B. Here turn 1 asks for a magic number (the header announces only numbers),
and turn 2 asks for a fact stated in ordinary prose elsewhere in the document ("<person> moved to the town of
<invented town> in <year>."). Selection that sees only turn 1 has no reason to keep that sentence.

Two-turn needle task (Llama-3.1-8B). Turn 1: haystack with 4 needles (distinct keys) + question about key A.
The prefix (up to the end of turn 1's generation prompt) is prefilled ONCE. Turn 2: the model's answer to A,
then a new user question about key B; these tokens and the answer to B are computed exactly on top of the
cached prefix. Score = fraction of turn-2 answers containing B's number.

Methods (all at j, w=0 on the prefix):
  exact; CF-KV (predict, query-independent); POP (naive);
  select+drop keeping k tokens chosen by turn-1 attention (FastKV/SnapKV-style: selection sees only question A);
  select+predict with the same selection;
  oracle select+drop: selection by question B (what selection would do if B were known at prefill).
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from engine import DEV, ROOT, load, Engine, fit_grams, solve, left_pad, context_mask
from round2 import wiki_texts, chat_pairs, chat_ids, last_kl
from longctx import Haystack, WORDS


FIRST = ("Mara", "Elias", "Noor", "Tobias", "Ines", "Kofi", "Lena", "Arjun", "Sofia", "Malik", "Hana", "Oskar")
LAST = ("Alvarez", "Brennan", "Castellano", "Dvorak", "Eriksen", "Fontaine", "Galloway", "Haddad", "Ivers", "Jansen")
SYL = ("bren", "low", "quel", "mar", "tosk", "vin", "dra", "hel", "orn", "sav", "kel", "thir", "ul", "ven", "gor")


def build(tok, hay, length, rng):
    num = str(int(rng.integers(1_000_000, 9_999_999)))
    key = str(rng.choice(WORDS))
    person = f"Dr. {rng.choice(FIRST)} {rng.choice(LAST)}"
    town = "".join(rng.choice(SYL, size=3)).capitalize()
    year = int(rng.integers(1950, 2015))
    pars = hay.take(length - 200, rng)
    pars.insert(int(rng.integers(0, len(pars) + 1)), f"One of the special magic numbers for {key} is: {num}.\n")
    pars.insert(int(rng.integers(0, len(pars) + 1)), f"{person} moved to the town of {town} in {year}.\n")
    head = ("Some special magic numbers are hidden within the following text. Make sure to memorize it. "
            "I will quiz you about the numbers afterwards.\n")
    qa = f"\nWhat is the special magic number for {key} mentioned in the provided text?"
    qb = f"Which town did {person} move to, according to the text?"
    ans = f"The special magic number for {key} mentioned in the provided text is {num}."
    user1 = head + "".join(pars) + qa
    msgs = [{"role": "user", "content": user1}, {"role": "assistant", "content": ans}, {"role": "user", "content": qb}]
    full = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
               add_special_tokens=False)["input_ids"]
    prefix = chat_ids(tok, user1)
    if full[:len(prefix)] != prefix:
        return None
    lead = f"According to the text, {person} moved to the town of"
    tail = full[len(prefix):] + tok(lead, add_special_tokens=False)["input_ids"]
    # oracle: selection sees question B in turn 1
    prefix_b = chat_ids(tok, head + "".join(pars) + "\n" + qb)
    tail_b = tok(lead, add_special_tokens=False)["input_ids"]
    return prefix, tail, prefix_b, tail_b, town


@torch.no_grad()
def answer(eng, prefix, tail, v, stop_ids, pad_id, max_new=24):
    ids, m = left_pad([prefix], pad_id)
    ctx = context_mask(m, 0)
    lg, cache = eng.prefill(ids, m, ctx, v["j"], v["mode"], v.get("W"), v.get("fl"), max_new=len(tail) + max_new,
                            keep=v.get("keep", 0), last_only=True)
    lg, cache = eng.extend(tail, cache)
    seq = eng.decode(lg.argmax(-1), cache, max_new, stop_ids)[0].tolist()
    return eng.tok.decode(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama8b")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--j", type=int, default=16)
    ap.add_argument("--lengths", default="8192,16384,32768")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--keeps", default="256,2048")
    args = ap.parse_args()
    run_dir = f"{ROOT}/results/raw/{args.run_id}"; assert not os.path.exists(run_dir); os.makedirs(run_dir)
    tok, model = load(args.model); eng = Engine(model, tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gc = model.generation_config.eos_token_id
    stops = set([tok.eos_token_id] + (gc if isinstance(gc, list) else [gc]))
    for s_ in ("<|im_end|>", "<|eot_id|>"):
        i = tok.convert_tokens_to_ids(s_)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            stops.add(i)
    stop_ids = torch.tensor(sorted(x for x in stops if x is not None), device=DEV)
    j = args.j; fl = tuple(x for x in (j, j - 4, j - 8) if x >= 1) + ("ctx",)
    calib = [tok(t, add_special_tokens=False)["input_ids"][:1280] for t in wiki_texts("train-00000-of-00002.parquet", 40000, 160, 1)]
    calib += [tok(t, add_special_tokens=False)["input_ids"][:8192] for t in wiki_texts("train-00000-of-00002.parquet", 40000, 16, 21, chars=45000)]
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
    print(f"j={j} alpha {best[1]:g} val KL {best[0]:.4f}", flush=True)
    variants = [("exact", {"j": 999, "mode": "exact"}), ("cfkv", {"j": j, "mode": "predict", "W": W, "fl": list(fl)}),
                ("pop", {"j": j, "mode": "naive"})]
    for k in map(int, args.keeps.split(",")):
        variants += [(f"select{k}+drop", {"j": j, "mode": "drop", "keep": k}),
                     (f"select{k}+predict", {"j": j, "mode": "predict", "keep": k, "W": W, "fl": list(fl)}),
                     (f"oracle_select{k}+drop", {"j": j, "mode": "drop", "keep": k, "oracle": True})]
    hay = Haystack(tok); rng = np.random.default_rng(321)
    items = []
    for L_ in map(int, args.lengths.split(",")):
        while sum(1 for x in items if x[0] == L_) < args.n:
            b = build(tok, hay, L_, rng)
            if b is not None:
                items.append((L_,) + b)
    res = {"run_id": args.run_id, "j": j, "alpha": best[1], "val_kl": best[0], "results": {}}
    for name, v in variants:
        t0 = time.time(); sc = {}
        for L_, prefix, tail, prefix_b, tail_b, gold in items:
            txt = answer(eng, prefix_b, tail_b, v, stop_ids, pad_id) if v.get("oracle") else \
                answer(eng, prefix, tail, v, stop_ids, pad_id)
            sc.setdefault(L_, []).append(float(gold in txt))
        res["results"][name] = {str(k): float(np.mean(x)) for k, x in sc.items()}
        print(f"{name:22s} " + " ".join(f"{k//1024}k={np.mean(x)*100:.0f}" for k, x in sorted(sc.items())) +
              f" ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

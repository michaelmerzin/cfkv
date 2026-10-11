"""Logic check of prefill_fast2 vs prefill_fast (same W, same inputs) in fp32 and in bf16."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from engine import DEV, load, fit_grams, solve
from fastpath import prefill_fast, prefill_fast2
from round2 import wiki_texts
tok, model = load(sys.argv[1] if len(sys.argv) > 1 else "qwen1.5b")
j = int(sys.argv[2]) if len(sys.argv) > 2 else 14
fl = [x for x in (j, j - 4, j - 8) if x >= 1]
calib = [tok(t, add_special_tokens=False)["input_ids"][:1280] for t in wiki_texts("train-00000-of-00002.parquet", 40000, 16, 1)]
acc = fit_grams(model, calib, [(j, tuple(fl))]); W = solve(acc[(j, tuple(fl))][0], acc[(j, tuple(fl))][1], 1e-4)
txt = " ".join(wiki_texts("test-00000-of-00001.parquet", None, 1, 7, chars=12000))
ids = torch.tensor([tok(txt, add_special_tokens=False)["input_ids"][:2000]], device=DEV)
for dt in (torch.float32, torch.bfloat16):
    model.to(dt); Wd = W.to(dt)
    for w in (0, 64):
        a, _ = prefill_fast(model, ids, j, Wd, fl, w); b = prefill_fast2(model, ids, j, Wd, fl, w)
        print(f"{dt} w={w}: max|fast-fast2| {(a[0,-1].float()-b[0,-1].float()).abs().max().item():.2e} "
              f"argmax equal {bool(a[0,-1].argmax()==b[0,-1].argmax())}", flush=True)

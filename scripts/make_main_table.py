"""Main results table with two operating points: (a) ~1/3 of the layers skipped for prompt tokens, (b) 1/2.
Reads the unified runs results/raw/me_* (main_eval.py: cfkv, pop, drop, fastkv, selcfkv at both depths), the
full-model scores of the earlier runs, SwiftKV (fixed at half depth) and ShortGPT (gen_bench.py).
Writes paper/table_main.tex (main text) and paper/table_main_full.tex (appendix, all methods)."""
import glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = [("GSM8K", "gsm8k"), ("MATH-500", "math500"), ("HumanEval", "humaneval"), ("MBPP", "mbpp"),
         ("CNN/DM", "cnndm"), ("MMLU", "mmlu"), ("ARC-C", "arc"), ("HellaSwag", "hellaswag")]
MODELS = [("Qwen2.5-1.5B", "qwen1.5b", 19, 14), ("Qwen2.5-7B", "qwen7b", 19, 14), ("Llama-3.1-8B", "llama8b", 22, 16),
          ("Mistral-7B", "mistral7b", 22, 16), ("Mistral-Small-24B", "mistral24b", 27, 20)]
# full-model scores (identical at both depths), from the earlier runs
EXACT = {"qwen1.5b": [69.6, 50.8, 59.8, 48.8, 20.1, 60.8, 76.3, 69.1], "qwen7b": [85.8, 65.2, 84.1, 65.6, 19.6, 74.1, 90.4, 81.7],
         "llama8b": [84.2, 42.8, 71.3, 59.2, 19.6, 67.6, 83.7, 80.3], "mistral7b": [57.4, 14.0, 40.2, 38.8, 19.4, 61.5, 78.0, 83.9]}
SWIFTKV = [85.8, 38.8, 65.9, 59.0, 19.8, 67.7, 82.8, 79.2]   # Llama-3.1-SwiftKV-8B, half depth

_B = [t for _, t in BENCH]
LEGACY = {}
SOURCES = {}   # tag -> {(task, name): folder}, filled by me_results (reported by main())
MIX_OK = {"mistral24b"}
RUN_GLOB = {"mistral24b": "m24_*"}   # rented-GPU runs (several jobs with disjoint cells); the full model is one of their variants

def me_results(tag):
    """One source folder per model (the one with the most cells), so a row never mixes runs on different hardware.
    mistral24b is the exception: its runs cover disjoint cells and are merged, in priority order of cell count."""
    runs = []
    for f in sorted(glob.glob(f"{ROOT}/results/raw/{RUN_GLOB.get(tag, f'me_{tag}_*')}/metrics.json")):
        if "smoke" in f:
            continue
        res = json.load(open(f)).get("results", {})
        n = sum(len(r) for r in res.values())
        if n:
            runs.append((n, os.path.basename(os.path.dirname(f)), res))
    runs.sort(key=lambda r: -r[0])
    if tag not in MIX_OK:
        runs = runs[:1]
    out, src = {}, {}
    for n, name, res in runs:
        for task, r in res.items():
            for k, v in r.items():
                if k not in out.get(task, {}):
                    out.setdefault(task, {})[k] = round(100 * v["score"], 1); src[(task, k)] = name
    if tag in LEGACY:
        lg = LEGACY[tag]
        for m_ in ("cfkv", "pop", "drop"):
            for t, v in zip(_B, lg[m_]):
                key = f"{m_}_j{lg['j']}"
                if key not in out.get(t, {}):
                    out.setdefault(t, {})[key] = v; src[(t, key)] = "legacy"
    SOURCES[tag] = src
    return out

def prune_results(tag):
    """ShortGPT cells: gen_bench runs (gen_*, three generation tasks) overridden by unified pruning runs
    (pr_*: main_eval.py --methods prune, all eight tasks)."""
    out = {}
    for pat in (f"gen_{tag}_*", f"pr_{tag}_*"):
        for f in sorted(glob.glob(f"{ROOT}/results/raw/{pat}/metrics.json")):
            for task, r in json.load(open(f)).get("results", {}).items():
                for name in ("prune25", "prune50"):
                    if name in r:
                        out.setdefault(task, {})[name] = round(100 * r[name]["score"], 1)
    return out

MAIN = [("POP", "pop"), ("FastKV", "fastkv"), ("ShortGPT", "prune"), ("\\ours{}", "cfkv")]
FULL = [("POP", "pop"), ("Drop", "drop"), ("FastKV", "fastkv"), ("ShortGPT", "prune"),
        ("\\ours{}", "cfkv"), ("Selection + \\ours{}", "selcfkv")]
TRAINING_FREE = {"pop", "drop", "fastkv", "prune", "cfkv", "selcfkv"}

def rows_for(tag, j, which, rows_def):
    me, pr = me_results(tag), prune_results(tag)
    if tag not in EXACT:
        EXACT[tag] = [me.get(t, {}).get("exact") for _, t in BENCH]
    for task, cells in me.items():     # unified-run pruning cells (main_eval --methods prune) take precedence
        for pk in ("prune25", "prune50"):
            if pk in cells:
                pr.setdefault(task, {})[pk] = cells[pk]
    rows = [("Full model", EXACT[tag], None)]
    for label, key in rows_def:
        if key == "prune":
            pk = "prune25" if which == "third" else "prune50"
            vals = [pr.get(t, {}).get(pk) for _, t in BENCH]
            label = f"ShortGPT ({'25' if which == 'third' else '50'}\\%)"
        else:
            vals = [me.get(t, {}).get(f"{key}_j{j}") for _, t in BENCH]
        rows.append((label, vals, key))
    return rows

fmt = lambda v: "--" if v is None else f"{v:.1f}"
def avg(vals):
    return None if any(v is None for v in vals) else sum(vals) / len(vals)

def build(rows_def, label, caption):
    nc = len(BENCH) + 2
    L = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}", caption, rf"\label{{{label}}}",
         r"\begin{tabular}{l" + "c" * len(BENCH) + "|c}", r"\toprule",
         " & " + " & ".join(b for b, _ in BENCH) + r" & Avg \\"]
    for which, title in (("third", r"(a) $\approx$1/3 of the layers skipped for prompt tokens ($\approx$31\% of prefill saved)"),
                         ("half", r"(b) 1/2 of the layers skipped for prompt tokens ($\approx$49\% of prefill saved)")):
        L += [r"\midrule", r"\multicolumn{%d}{c}{\textbf{%s}} \\" % (nc, title)]
        for name, tag, j3, j2 in MODELS:
            j = j3 if which == "third" else j2
            rows = rows_for(tag, j, which, rows_def)
            L += [r"\midrule", r"\multicolumn{%d}{l}{\textit{%s} ($j{=}%d$)} \\" % (nc, name, j)]
            for c in range(len(BENCH)):
                pass
            best = []
            for c in range(len(BENCH)):
                vs = [r[1][c] for r in rows if r[2] in TRAINING_FREE and r[1][c] is not None]
                best.append(max(vs) if vs else None)
            tf_avgs = [avg(r[1]) for r in rows if r[2] in TRAINING_FREE and avg(r[1]) is not None]
            best_avg = max(tf_avgs) if tf_avgs else None
            for lab, vals, key in rows:
                cells = []
                for c, v in enumerate(vals):
                    s = fmt(v)
                    if key in TRAINING_FREE and v is not None and v == best[c]:
                        s = r"\textbf{" + s + "}"
                    cells.append(s)
                a = avg(vals); sa = fmt(a)
                if key in TRAINING_FREE and a is not None and best_avg is not None and abs(a - best_avg) < 1e-9:
                    sa = r"\textbf{" + sa + "}"
                L.append(r"\quad " + lab + " & " + " & ".join(cells) + " & " + sa + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L) + "\n"


SHORT = {"gsm8k": "GSM", "math500": "MATH", "humaneval": "HE", "mbpp": "MBPP", "cnndm": "CNN", "mmlu": "MMLU", "arc": "ARC", "hellaswag": "HS"}

def build_wide(rows_def, label, caption):
    """Main-text version: one-third and half depth side by side (9 columns each incl. average), one row per method."""
    nb = len(BENCH)
    L = [r"\begin{table*}[t]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{2.1pt}", caption, rf"\label{{{label}}}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "c" * nb + "c|" + "c" * nb + "c}", r"\toprule",
         r" & \multicolumn{%d}{c|}{\textbf{(a) $\approx$1/3 of the layers skipped}} & \multicolumn{%d}{c}{\textbf{(b) 1/2 of the layers skipped}} \\" % (nb + 1, nb + 1),
         r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}" % (nb + 2, nb + 3, 2 * nb + 3),
         " & " + " & ".join(SHORT[t] for _, t in BENCH) + r" & Avg & " + " & ".join(SHORT[t] for _, t in BENCH) + r" & Avg \\"]
    for name, tag, j3, j2 in MODELS:
        L += [r"\midrule", r"\multicolumn{%d}{l}{\textit{%s} ($j{=}%d$ / $%d$)} \\" % (2 * nb + 3, name, j3, j2)]
        R3, R2 = rows_for(tag, j3, "third", rows_def), rows_for(tag, j2, "half", rows_def)
        # align rows by label stem (SwiftKV exists only at half depth)
        stems = []
        for r in R3 + R2:
            stem = r[0].split(" (")[0]
            if stem not in stems:
                stems.append(stem)
        by = lambda R: {r[0].split(" (")[0]: r for r in R}
        b3, b2 = by(R3), by(R2)
        def best(R):
            bs = []
            for c in range(nb):
                vs = [r[1][c] for r in R if r[2] in TRAINING_FREE and r[1][c] is not None]
                bs.append(max(vs) if vs else None)
            tf = [avg(r[1]) for r in R if r[2] in TRAINING_FREE and avg(r[1]) is not None]
            return bs, (max(tf) if tf else None)
        best3, ba3 = best(R3); best2, ba2 = best(R2)
        for stem in stems:
            half = []
            for R, bsel, bavg, B in ((R3, best3, ba3, b3), (R2, best2, ba2, b2)):
                r = B.get(stem)
                if r is None:
                    half.append(["--"] * (nb + 1)); continue
                lab, vals, key = r
                cells = []
                for c, v in enumerate(vals):
                    s_ = fmt(v)
                    if key in TRAINING_FREE and v is not None and v == bsel[c]:
                        s_ = r"\textbf{" + s_ + "}"
                    cells.append(s_)
                a = avg(vals); sa = fmt(a)
                if key in TRAINING_FREE and a is not None and bavg is not None and abs(a - bavg) < 1e-9:
                    sa = r"\textbf{" + sa + "}"
                half.append(cells + [sa])
            lab = (b3.get(stem) or b2.get(stem))[0]
            lab = lab.replace("ShortGPT", "ShortGPT").replace("FastKV", "FastKV")
            lab = lab.split(" (")[0]
            L.append(r"\quad " + lab + " & " + " & ".join(half[0]) + " & " + " & ".join(half[1]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(L) + "\n"

capw = (r"\caption{Main results at two operating points (only the sink and the last prompt token exact, $w=0$). (a) is POP's own setting"
        r" (about 31\% of prefill compute saved), (b) skips half of the layers (about 49\%). GSM = GSM8K (500, CoT), MATH = MATH-500 (250, CoT),"
        r" HE = HumanEval (164, pass@1), MBPP (500, pass@1), CNN = CNN/DailyMail (250, ROUGE-L), MMLU (1{,}000, 5-shot),"
        r" ARC = ARC-Challenge (1{,}172, 5-shot), HS = HellaSwag (1{,}000). POP and \ours{} skip the same layers for prompt tokens (\ours{}'s predictor"
        r" adds 5--12\% of the skipped compute) and FastKV's budget is set to the same saving; ShortGPT removes 25\% (a) or 50\% (b) of the layers"
        r" for all tokens and also saves decoding. \ours{} uses one configuration per model and depth (features, $\alpha$ and the optional key rescaling"
        r" $\gamma$), selected on held-out text and fixed for all benchmarks (Table~\ref{tab:configs}). Bold: best training-free method."
        r" Avg: unweighted mean over the eight benchmarks.}")
cap = (r"\caption{Main results at two operating points, with only the sink and the last prompt token exact ($w=0$)."
       r" (a) matches POP's own setting, (b) skips half of the layers. Generation: GSM8K (500, CoT), MATH-500 (250, CoT),"
       r" HumanEval (164, pass@1), MBPP (500, pass@1), CNN/DailyMail (250, ROUGE-L); multiple choice: MMLU (1{,}000, 5-shot),"
       r" ARC-Challenge (1{,}172, 5-shot), HellaSwag (1{,}000). All training-free methods save the same prefill compute"
       r" except ShortGPT (25\% in (a), 50\% in (b) of the layers), which removes layers for all tokens, prompt and generated, and also saves decoding. \ours{}:"
       r" features, $\alpha$ and $\gamma$ selected on held-out validation text. Bold: best training-free method."
       r" SwiftKV is distilled on 680M tokens and exists only at half depth. Avg: mean over the eight benchmarks.}")

def build_swiftkv():
    """Llama-3.1-8B at half depth, all eight benchmarks: the trained SwiftKV checkpoint next to the training-free methods."""
    j = 16
    rows = rows_for("llama8b", j, "half", [("POP", "pop"), ("\\ours{}", "cfkv")])
    by = {r[2]: r for r in rows if r[2]}
    entries = [("Full model", EXACT["llama8b"]), ("SwiftKV (trained, 680M tokens)", SWIFTKV),
               ("\\ours{} (training-free)", by["cfkv"][1]), ("POP (training-free)", by["pop"][1])]
    L = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{Comparison with the trained SwiftKV checkpoint on Llama-3.1-8B-Instruct at half depth ($j=16$, only the sink and the last prompt token exact, $w=0$). SwiftKV is distilled on 680M tokens and is a trained reference, not a training-free competitor; its official checkpoints cover only Llama models, so it is not part of Table~\ref{tab:main}. Benchmarks and protocol as in Table~\ref{tab:main}.}",
         r"\label{tab:swiftkv8}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "c" * len(BENCH) + "|c}", r"\toprule",
         " & " + " & ".join(b for b, _ in BENCH) + r" & Avg \\", r"\midrule"]
    for lab, vals in entries:
        a = sum(vals) / len(vals)
        L.append(lab + " & " + " & ".join(f"{v:.1f}" for v in vals) + f" & {a:.1f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(L) + "\n"

APPLY = "--apply" in sys.argv
suf = ".tex" if APPLY else ".new.tex"
capf = (r"\caption{All methods at both operating points (appendix version of Table~\ref{tab:main}); drop removes the context"
        r" from the upper layers, and selection + \ours{} predicts the tokens that selection discards.}")
open(f"{ROOT}/paper/table_main{suf}", "w").write("% generated by scripts/make_main_table.py\n" + build_wide(MAIN, "tab:main", capw))
open(f"{ROOT}/paper/table_swiftkv8{suf}", "w").write("% generated by scripts/make_main_table.py\n" + build_swiftkv())
open(f"{ROOT}/paper/table_main_full{suf}", "w").write("% generated by scripts/make_main_table.py\n" + build(FULL, "tab:main_full", capf))

# ---- report: empty cells, sources, win counts (training-free methods in the main table)
print("wrote", f"paper/table_main{suf}", "and", f"paper/table_main_full{suf}", "(use --apply to overwrite the live tables)")
for which, depth in (("third", 2), ("half", 3)):
    wins = tot = 0; losses = []; empty = []
    for name, tag, j3, j2 in MODELS:
        j = j3 if which == "third" else j2
        rows = rows_for(tag, j, which, MAIN)
        for c, (bn, bt) in enumerate(BENCH):
            tf = [(r[0], r[1][c]) for r in rows if r[2] in TRAINING_FREE]
            cf = next((v for l, v in tf if "CF-KV" in l or "ours" in l), None)
            if cf is None:
                empty.append(f"{name}/{bn}"); continue
            others = [v for l, v in tf if not ("CF-KV" in l or "ours" in l) and v is not None]
            if not others:
                empty.append(f"{name}/{bn} (baselines)"); continue
            tot += 1
            if cf >= max(others):
                wins += 1
            else:
                losses.append((name, bn, cf, max(others)))
    print(f"[{which}] CF-KV best training-free on {wins}/{tot} cells with a CF-KV value and at least one baseline")
    print("   losses:", [f"{a} {b}: {c:.1f} vs {d:.1f}" for a, b, c, d in losses])
    print("   cells with no CF-KV or no baseline:", len(empty), empty[:12], "..." if len(empty) > 12 else "")
for tag, src in SOURCES.items():
    print(f"source {tag}: {sorted(set(src.values()))}")

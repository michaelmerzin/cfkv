"""Main-text figure: accuracy relative to the full model vs fraction of layers skipped (0, ~1/3, 1/2), from the unified runs in Table 1.
Left: generation (GSM8K, MATH-500, HumanEval, MBPP), right: multiple choice (MMLU, ARC-C, HellaSwag). Per model: mean over the tasks of
score / full-model score; line: mean over the models with all cells; band: min-max over the models. Writes paper/figures/frontier_unified.{pdf,png}."""
import importlib.util, io, contextlib, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "axes.linewidth": 0.6})
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("mt", f"{ROOT}/scripts/make_main_table.py"); mt = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(mt)
BN = [t for _, t in mt.BENCH]
GROUPS = {"Generation (GSM8K, MATH-500, HumanEval, MBPP)": ["gsm8k", "math500", "humaneval", "mbpp"],
          "Multiple choice (MMLU, ARC-C, HellaSwag)": ["mmlu", "arc", "hellaswag"]}
METHODS = [("cfkv", "CF-KV", "#1f5fbf", "-", "o"), ("pop", "POP", "#e07b00", "--", "s"), ("fastkv", "FastKV", "#2a9d4b", ":", "^")]
MODELS = mt.MODELS
XS = [0.0, 1 / 3, 0.5]
fig, axes = plt.subplots(1, 2, figsize=(3.15, 1.75), sharey=True)
for ax, (gname, tasks) in zip(axes, GROUPS.items()):
    for key, lab, col, ls, mk in METHODS:
        per_model = []
        for name, tag, j3, j2 in MODELS:
            ys = [1.0]
            for which, j in (("third", j3), ("half", j2)):
                rows = mt.rows_for(tag, j, which, mt.MAIN); d = {r[2]: r[1] for r in rows if r[2]}
                rel = [d[key][BN.index(t)] / mt.EXACT[tag][BN.index(t)] for t in tasks]
                ys.append(float(np.mean(rel)))
            per_model.append(ys)
        a = np.array(per_model) * 100
        ax.fill_between(XS, a.min(0), a.max(0), color=col, alpha=0.12, lw=0)
        ax.plot(XS, a.mean(0), color=col, ls=ls, marker=mk, ms=3, lw=1.1, label=lab)
    ax.set_title(gname.split(" (")[0], fontsize=7.5)
    ax.set_xticks(XS); ax.set_xticklabels(["0", "1/3", "1/2"]); ax.set_xlabel("fraction of layers skipped", fontsize=7)
    ax.grid(alpha=0.25, lw=0.4); ax.set_ylim(0, 108)
axes[0].set_ylabel("accuracy / full model (%)", fontsize=7)
axes[0].legend(frameon=False, fontsize=6.5, loc="lower left")
fig.tight_layout(pad=0.3)
os.makedirs(f"{ROOT}/paper/figures", exist_ok=True)
fig.savefig(f"{ROOT}/paper/figures/frontier_unified.pdf"); fig.savefig(f"{ROOT}/paper/figures/frontier_unified.png", dpi=200)
for ax, (gname, tasks) in zip(axes, GROUPS.items()):
    print(gname)
    for line in ax.get_lines():
        print("  ", line.get_label(), [round(float(y), 1) for y in line.get_ydata()])

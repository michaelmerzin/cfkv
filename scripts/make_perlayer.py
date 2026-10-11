"""Regenerate paper/figures/perlayer.{pdf,png} (per-layer R^2 of true K/V: CF-KV vs POP) from results/raw/analysis_*.
TrueType (fonttype 42) fonts so the PDF has no Type 3 fonts."""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [("Qwen2.5-1.5B", "qwen1.5b_j14", 14, 28), ("Qwen2.5-7B", "qwen7b_j14", 14, 28), ("Llama-3.1-8B", "llama8b_j16", 16, 32)]
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#555555", "xtick.color": "#444444", "ytick.color": "#444444"})
fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.0), sharey="row")
for c, (name, tag, j, L) in enumerate(RUNS):
    d = json.load(open(sorted(glob.glob(f"{ROOT}/results/raw/analysis_{tag}_*/metrics.json"))[0]))["layers"]
    ls = sorted(int(k) for k in d if int(k) > j)
    for r, (key, lab) in enumerate((("K", "keys"), ("V", "values"))):
        ax = axes[r, c]; ax.grid(axis="y", color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
        ax.axhline(0, color="#aaaaaa", lw=1)
        cf = [d[str(l)][f"R2_{key}_cf"] for l in ls]; pop = [max(d[str(l)][f"R2_{key}_pop"], -1.0) for l in ls]
        ax.plot(ls, cf, color=BLUE, marker="o", ms=3.5, lw=2, label="Closed-form (ours)")
        ax.plot(ls, pop, color=ORANGE, marker="s", ms=3.5, lw=2, ls="--", label="POP")
        low = d[str(ls[-1])][f"R2_{key}_pop"]
        if low < -1:
            ax.text(ls[-1], -0.82, f"{low:.1f}", ha="center", fontsize=7, color="#444444")
        ax.set_ylim(-1.05, 1.05)
        if r == 0:
            ax.set_title(f"{name} ($j$={j} of {L})", fontsize=8.5)
        if c == 0:
            ax.set_ylabel(f"{lab}: $R^2$")
        if r == 1:
            ax.set_xlabel("upper layer $\\ell$")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(f"{ROOT}/paper/figures/perlayer.pdf"); fig.savefig(f"{ROOT}/paper/figures/perlayer.png", dpi=160)
print("ok")

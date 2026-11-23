"""Figure: size of the whole human brain network, the H01 temporal-cortex sample, and our reservoir subgraph (log scale).
Whole brain: literature estimates (about 86 billion neurons, Azevedo et al. 2009; synapses 10^14 to 10^15, estimates vary).
H01: Shapson-Coe et al., Science 384, eadk4858 (2024). Subgraph: this work (A_300_s0)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
BLUE, ORANGE, SLATE = "#2a78d6", "#eb6834", "#7a8794"
INK, MUTE, LINE = "#0b0b0b", "#52514e", "#c9c8c2"
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

# label, sublabel, neurons, synapses (value or (low, high)), colour
rows = [
    ("Whole human brain", "literature estimate", 86e9, (1e14, 1e15), SLATE),
    ("H01: 1 mm³ of temporal cortex", "counted by electron microscopy", 16087, 149871669, BLUE),
    ("Our reservoir subgraph", "300 neurons from H01", 300, 5163, ORANGE),
]
ys = [2, 1, 0]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.8), dpi=200, gridspec_kw=dict(wspace=0.1))
fig.patch.set_facecolor("white")


def fmt(v):
    if v >= 1e9:
        return f"{v / 1e9:.0f} billion"
    if v >= 1e6:
        return f"{v / 1e6:.0f} million"
    return f"{v:,.0f}"


def ratio_text(a, b):
    r = a / b
    if r >= 1e6:
        return f"÷ {r / 1e6:.1f} million"
    if r >= 1e4:
        return f"÷ {r / 1e3:,.0f} thousand"
    return f"÷ {r:,.0f}"


for ax, k, xl, xt, xlabel in ((ax1, 2, (50, 1e13), [1e2, 1e4, 1e6, 1e8, 1e10, 1e12], "neurons (log scale)"),
                              (ax2, 3, (1e3, 1e17), [1e3, 1e5, 1e7, 1e9, 1e11, 1e13, 1e15], "synapses (log scale)")):
    ax.set_xscale("log")
    ax.set_xlim(*xl)
    ax.set_ylim(-0.7, 2.75)
    prev = None
    for row, y in zip(rows, ys):
        v, col = row[k], row[4]
        ax.plot([xl[0], np.sqrt(v[0] * v[1]) if isinstance(v, tuple) else v], [y, y], color="#e7e6e0", lw=1, zorder=1)
        if isinstance(v, tuple):
            lo, hi = v
            ax.plot([lo, hi], [y, y], color=col, lw=7, alpha=0.35, solid_capstyle="round", zorder=2)
            ax.scatter([lo, hi], [y, y], s=70, facecolors="white", edgecolors=col, linewidths=2, zorder=3)
            ax.text(hi, y + 0.2, "10¹⁴ to 10¹⁵ (estimates vary)", va="bottom", ha="right", fontsize=10, fontweight="bold", color=INK)
            vv = np.sqrt(lo * hi)
        else:
            ax.scatter([v], [y], s=110, color=col, edgecolors="white", linewidths=1.2, zorder=3)
            ax.text(v * 2.2, y, fmt(v), va="center", fontsize=11, fontweight="bold", color=INK)
            vv = v
        if prev is not None:
            pv, py, praw = prev
            ax.plot([pv, vv], [py, y], color=MUTE, lw=0.9, ls=(0, (3, 3)), zorder=1)
            if isinstance(praw, tuple):
                txt = f"÷ {praw[0] / vv / 1e6:.1f} to {praw[1] / vv / 1e6:.0f} million"
            else:
                txt = ratio_text(pv, vv)
            ax.text(np.sqrt(pv * vv) * 1.15, (py + y) / 2 + 0.06, txt, fontsize=9.5, color=MUTE, ha="left", va="center",
                    bbox=dict(fc="white", ec="none", pad=1.2))
        prev = (vv, y, v)
    ax.set_xticks(xt)
    ax.set_xticklabels(["10" + str(int(round(np.log10(t)))).translate(SUP) for t in xt], fontsize=9, color=MUTE)
    ax.minorticks_off()
    ax.set_xlabel(xlabel, fontsize=9.5, color=MUTE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(axis="x", colors=LINE, length=3)
    ax.xaxis.grid(True, color="#ecebe6", lw=0.6, zorder=0)
    ax.set_yticks([])
ax1.set_title("Neurons", loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=10)
ax2.set_title("Synapses", loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=10)
for row, y in zip(rows, ys):
    ax1.annotate(row[0], xy=(0, y + 0.1), xycoords=("axes fraction", "data"), xytext=(-12, 0), textcoords="offset points",
                 ha="right", va="bottom", fontsize=11, fontweight="bold", color=INK, annotation_clip=False)
    ax1.annotate(row[1], xy=(0, y + 0.06), xycoords=("axes fraction", "data"), xytext=(-12, 0), textcoords="offset points",
                 ha="right", va="top", fontsize=8.8, color=MUTE, annotation_clip=False)
fig.text(0.03, 0.925, "How small is our reservoir compared with a human brain?", fontsize=15, fontweight="bold", color=INK)
fig.text(0.03, 0.882, "Log scales: each gridline is a factor of 100. Hollow markers are literature estimates, filled markers are counts.",
         fontsize=9.5, color=MUTE)
fig.text(0.03, 0.115, "Whole brain to our subgraph: about 290 million times fewer neurons (86 billion ÷ 300) and between 19 and 190 billion "
         "times fewer synapses (10¹⁴ to 10¹⁵ ÷ 5,163).", fontsize=9.3, color=INK)
fig.text(0.03, 0.075, "Sources: whole brain, Azevedo et al., J. Comp. Neurol. 513, 532 (2009) for neurons; the synapse range is a rough literature figure. "
         "H01, Shapson-Coe et al., Science 384, eadk4858 (2024).", fontsize=7.8, color=MUTE)
fig.text(0.03, 0.045, "H01 synapses are all detected synapses; most do not join two neurons whose cell bodies are in the volume (in our export, "
         "148,140 of 160 million rows, 0.09%, join two identified neurons). Subgraph synapses: rows between its 300 neurons.",
         fontsize=7.8, color=MUTE)
fig.subplots_adjust(left=0.2, right=0.97, top=0.8, bottom=0.27)
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(OUT, f"fig_scale.{ext}"), facecolor="white")
print("saved")

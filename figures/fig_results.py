"""Final-queue results: pooled success rate per condition, with the three subgraphs as dots.
Reads ../results/final_queue_v3_v3b.csv. Run from this folder: python fig_results.py [outdir]"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
rows = list(csv.DictReader(open("../results/final_queue_v3_v3b.csv")))
label = {"open_loop": "Open loop (untrained)", "bias": "Bias only (3 numbers)", "esn": "ESN (random wiring)",
         "shuffle_permute": "Shuffled H01, permuted signs", "h01_permute": "H01 wiring, permuted signs",
         "direct": "Direct readout (no reservoir)", "shuffle_data": "Shuffled H01, measured signs",
         "h01_data": "H01 wiring, measured signs", "er": "Random graph"}
rows.sort(key=lambda r: -float(r["pooled_percent"]))
INK, MUTE, LINE = "#1d2330", "#5b6475", "#d9dce3"
BLUE, GREY, ORANGE = "#0b6bb5", "#9aa3b5", "#b85a00"
fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=200)
for i, r in enumerate(rows):
    y = len(rows) - 1 - i
    pct = float(r["pooled_percent"])
    h01 = r["condition"].startswith("h01")
    col = BLUE if h01 else (ORANGE if r["condition"] == "open_loop" else GREY)
    ax.barh(y, pct, height=0.55, color=col, zorder=2)
    ax.text(106, y, f"{r['pooled_successes']}/{r['pooled_episodes']}", va="center", fontsize=9, color=INK)
    for k in ("subgraph_0", "subgraph_1", "subgraph_2"):
        if r[k]:
            ax.plot(float(r[k]) / 30 * 100, y, "o", ms=4.5, mfc="white", mec=INK, mew=1, zorder=3)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([label[r["condition"]] for r in rows][::-1], fontsize=9.5, color=INK)
ax.set_xlim(0, 120); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Success rate on 30 evaluation episodes (%)", fontsize=9.5, color=MUTE)
ax.grid(axis="x", color=LINE, lw=0.8, zorder=0)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(LINE); ax.tick_params(length=0, colors=MUTE)
ax.set_title("Final queue: bars are the pooled rate over three subgraphs; dots are the individual subgraphs",
             fontsize=9.5, color=INK, loc="left")
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"{OUT}/fig_results.{ext}")

"""Figure: the H01 300-neuron subgraph used as the reservoir, next to its two graph controls.
Reads webots_push/subgraphs and webots_push/controllers/h01_dataset_policy/reservoir.py (same code the robot uses)."""
import os, sys
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapes import scatter_by_color

ROOT = "../webots"
sys.path.insert(0, os.path.join(ROOT, "controllers", "h01_dataset_policy"))
import reservoir as R

NAME = sys.argv[1] if len(sys.argv) > 1 else "A_300_s0"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
BLUE, ORANGE, GREY, INK, MUTE = "#2a78d6", "#eb6834", "#a8a7a2", "#0b0b0b", "#52514e"

base = R.load_subgraph(os.path.join(ROOT, "subgraphs"), NAME)
base = R.apply_sign_policy(base, "data")
panels = [("H01 subgraph", R.control_graph(base, "h01")),
          ("Degree-preserving rewiring", R.control_graph(base, "shuffle", seed=0)),
          ("Matched random graph", R.control_graph(base, "er", seed=0))]


def stats(g):
    D = nx.DiGraph()
    D.add_nodes_from(range(g["n"]))
    D.add_edges_from(zip(g["pre"], g["post"]))
    U = D.to_undirected()
    recip = sum(1 for a, b in D.edges if D.has_edge(b, a)) / D.number_of_edges()
    return D, U, nx.average_clustering(U), recip


fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.4), dpi=200)
fig.patch.set_facecolor("white")
for ax, (title, g) in zip(axes, panels):
    D, U, clus, recip = stats(g)
    pos = nx.spring_layout(U, seed=3, k=2.2 / np.sqrt(g["n"]), iterations=400)
    arr = nx.rescale_layout(np.array([pos[i] for i in range(g["n"])]), scale=1.0)
    pos = {i: arr[i] for i in range(g["n"])}
    deg = dict(U.degree())
    size = [4 + 2.4 * deg[i] for i in range(g["n"])]
    if title == "Matched random graph":
        col = [(BLUE if s > 0 else ORANGE) if k else GREY for s, k in zip(g["sign"], base["known"])]  # shuffled signs: known flags no longer match
        col = [BLUE if s > 0 else ORANGE for s in g["sign"]]
    else:
        col = [(BLUE if s > 0 else ORANGE) if k else GREY for s, k in zip(g["sign"], base["known"])]
    segs = [(pos[a], pos[b]) for a, b in D.edges]
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, colors="#8a8984", linewidths=0.3, alpha=0.4, zorder=1))
    scatter_by_color(ax, [pos[i][0] for i in range(g["n"])], [pos[i][1] for i in range(g["n"])], size, col,
                     edgecolors="white", linewidths=0.35, zorder=2)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08)
    ax.axis("off")
    ax.set_title(title, fontsize=12, color=INK, pad=8, fontweight="bold")
    ax.text(0.5, -0.02, f"{g['n']} neurons · {len(g['pre'])} connections\nclustering {clus:.3f} · reciprocal {recip:.1%}",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=MUTE)
for ax, lab in zip(axes, "abc"):
    ax.text(0.0, 1.02, lab, transform=ax.transAxes, fontsize=14, fontweight="bold", color=INK)

n_inh = sum(1 for s, k in zip(base["sign"], base["known"]) if k and s < 0)
n_exc = sum(1 for s, k in zip(base["sign"], base["known"]) if k and s > 0)
n_unk = sum(1 for k in base["known"] if not k)
handles = [Line2D([], [], marker="o", ls="", color=BLUE, markeredgecolor="white", markersize=8, label=f"excitatory ({n_exc})"),
           Line2D([], [], marker="^", ls="", color=ORANGE, markeredgecolor="white", markersize=8, label=f"inhibitory ({n_inh})"),
           Line2D([], [], marker="D", ls="", color=GREY, markeredgecolor="white", markersize=8, label=f"sign unknown, treated as excitatory ({n_unk})")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.005), labelcolor=INK)
fig.text(0.5, 0.055, "Node size ∝ number of connections. Lines: directed synaptic connections (arrowheads omitted). "
         "Positions are force-directed layouts, not anatomical. In (c) signs are reassigned at random.",
         ha="center", fontsize=8.5, color=MUTE)
fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.2, wspace=0.03)
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(OUT, f"fig_h01_graph_{NAME}.{ext}"), facecolor="white")
print("saved", os.path.join(OUT, f"fig_h01_graph_{NAME}.png"))

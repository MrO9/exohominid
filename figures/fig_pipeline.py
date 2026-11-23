"""Figure: from the human temporal cortex sample to the reservoir subgraph (H01 pipeline).
Numbers in panels a-c are from Shapson-Coe et al., Science 384, eadk4858 (2024); panels d-e are our own extraction (LOG.md).
The tissue, imaging and reconstruction drawings are schematics drawn here, not images from the paper."""
import os, sys
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch, Circle, FancyBboxPatch
from matplotlib.collections import LineCollection

ROOT = "../webots"
sys.path.insert(0, os.path.join(ROOT, "controllers", "h01_dataset_policy"))
import reservoir as R
from shapes import scatter_by_color

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a8a7a2"
INK, MUTE, LINE = "#0b0b0b", "#52514e", "#c9c8c2"
rng = np.random.default_rng(4)

fig = plt.figure(figsize=(14, 9.2), dpi=200)
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(2, 6, left=0.03, right=0.985, top=0.855, bottom=0.115, hspace=0.34, wspace=0.35,
                      height_ratios=[1, 1.02])
ax_a = fig.add_subplot(gs[0, 0:2]); ax_b = fig.add_subplot(gs[0, 2:4]); ax_c = fig.add_subplot(gs[0, 4:6])
ax_d = fig.add_subplot(gs[1, 0:4]); ax_e = fig.add_subplot(gs[1, 4:6])
for a in (ax_a, ax_b, ax_c, ax_e):
    a.set_axis_off()


def title(ax, letter, text, sub=None):
    ax.text(0.0, 1.07, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", color=INK, va="bottom")
    ax.text(0.07, 1.07, text, transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=INK, va="bottom")
    if sub:
        ax.text(0.07, 1.015, sub, transform=ax.transAxes, fontsize=8.8, color=MUTE, va="bottom")


# ---------------------------------------------------------------- a: tissue
ax = ax_a
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
layers = [("1", 0.9), ("2", 0.7), ("3", 1.7), ("4", 0.8), ("5", 1.3), ("6", 1.2)]
shade = ["#f1efe9", "#e6e3da", "#ddd9cd", "#cfcabc", "#ddd9cd", "#e6e3da"]
y = 9.0
x0, x1 = 0.9, 3.9
for (name, h), c in zip(layers, shade):
    ax.add_patch(Rectangle((x0, y - h), x1 - x0, h, fc=c, ec="white", lw=1.2))
    ax.text(x0 - 0.12, y - h / 2, name, ha="right", va="center", fontsize=8.5, color=MUTE)
    y -= h
ax.add_patch(Rectangle((x0, y - 1.3), x1 - x0, 1.3, fc="#f7f7f5", ec="white", lw=1.2, hatch="////"))
ax.text(x0 - 0.12, y - 0.65, "WM", ha="right", va="center", fontsize=8, color=MUTE)
for cx, cy in [(1.5, 3.0), (2.4, 4.6), (3.3, 2.6), (2.0, 6.1)]:
    ax.add_patch(Circle((cx, cy), 0.15, fc=BLUE, ec="white", lw=0.6, zorder=3))
    ax.plot([cx, cx], [cy, 8.8], color=BLUE, lw=0.8, alpha=0.7, zorder=2)
    for dy in (0.5, 1.1):
        ax.plot([cx, cx - 0.3], [cy + dy, cy + dy + 0.45], color=BLUE, lw=0.6, alpha=0.7, zorder=2)
        ax.plot([cx, cx + 0.3], [cy + dy, cy + dy + 0.45], color=BLUE, lw=0.6, alpha=0.7, zorder=2)
    ax.plot([cx, cx], [cy, cy - 0.8], color=BLUE, lw=0.5, alpha=0.5, zorder=2)
ax.annotate("", xy=(4.4, 9.0), xytext=(4.4, 0.7), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
ax.text(4.65, 4.85, "about 3 mm", fontsize=8.2, color=INK, va="center", rotation=90)
ax.text((x0 + x1) / 2, 9.3, "pia", ha="center", fontsize=8.2, color=MUTE)
ax.text(5.5, 8.9, "left anterior middle\ntemporal gyrus", fontsize=9, color=INK, va="top", fontweight="bold")
ax.text(5.5, 7.1, "surgical sample from a\n45-year-old female, removed\nto reach an epileptic focus\nin the hippocampus", fontsize=8, color=MUTE, va="top")
ax.text(5.5, 3.9, "slab 170 µm thick,\nall six layers", fontsize=9, color=INK, va="top")
ax.text(5.5, 2.4, "1.05 mm³ after correcting\nsectioning compression", fontsize=9, color=INK, va="top")
ax.text((x0 + x1) / 2, -0.3, "schematic, layer widths not to scale", fontsize=7.3, color=MUTE, ha="center")
title(ax, "a", "Human temporal cortex sample", "H01, surgical tissue")

# ---------------------------------------------------------------- b: imaging
ax = ax_b
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
n = 13
for i in range(n):
    dx, dy = i * 0.33, i * 0.2
    ax.add_patch(Polygon([[0.8 + dx, 2.2 + dy], [4.6 + dx, 2.2 + dy], [5.6 + dx, 3.6 + dy], [1.8 + dx, 3.6 + dy]],
                         closed=True, fc="#e9e7df" if i % 2 else "#f4f3ee", ec="#8a8984", lw=0.5))
# multibeam footprint on the front section: 61 beams as a hexagonal tile
cx0, cy0 = 3.2 + 0 * 0.33, 2.9
hx, hy = [], []
for r in range(-4, 5):
    for q in range(-4, 5):
        if max(abs(q), abs(r), abs(-q - r)) <= 4:
            hx.append(cx0 + 0.26 * (q + 0.5 * r)); hy.append(cy0 + 0.15 * r)
assert len(hx) == 61
ax.scatter(hx, hy, s=5, c=BLUE, zorder=3)
ax.annotate("61 parallel beams image\none tile at a time", xy=(cx0 + 0.9, cy0 + 0.2), xytext=(5.4, 1.15),
            fontsize=8.2, color=MUTE, arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.6), va="center")
ax.annotate("", xy=(8.7, 6.2), xytext=(7.2, 4.9), arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
ax.text(8.75, 6.35, "cutting\ndirection", fontsize=8, color=MUTE, va="bottom")
ax.text(0.2, 9.55, "5,019 sections", fontsize=11, fontweight="bold", color=INK, va="top")
ax.text(0.2, 8.65, "mean 33.9 nm thick, on tape", fontsize=8.8, color=MUTE, va="top")
ax.text(0.2, 8.0, "multibeam scanning electron microscope", fontsize=8.8, color=MUTE, va="top")
ax.text(0.2, 7.35, "4 × 4 nm² pixels, aligned in 3D", fontsize=8.8, color=MUTE, va="top")
ax.text(5.0, 0.15, "1.4 petabytes of images", fontsize=11, fontweight="bold", color=INK, ha="center")
title(ax, "b", "Electron microscopy", "serial sections of the whole slab")

# ---------------------------------------------------------------- c: reconstruction
ax = ax_c
ax.set_xlim(0, 10); ax.set_ylim(0, 10)


def tree(ax, x, y, ang, length, depth, col, lw):
    if depth == 0:
        return
    x2, y2 = x + length * np.cos(ang), y + length * np.sin(ang)
    ax.plot([x, x2], [y, y2], color=col, lw=lw, solid_capstyle="round", zorder=2)
    for da in (-0.55, 0.5):
        tree(ax, x2, y2, ang + da + rng.normal(0, 0.12), length * 0.7, depth - 1, col, lw * 0.75)


cells = [((2.4, 5.4), "#2a78d6"), ((5.4, 4.9), "#1baf7a"), ((7.7, 6.0), "#6250d6")]
for (cx, cy), col in cells:
    for k in range(5):
        tree(ax, cx, cy, k * 2 * np.pi / 5 + rng.uniform(0, 0.5), 0.62, 3, col, 0.9)
    ax.add_patch(Circle((cx, cy), 0.24, fc=col, ec="white", lw=0.8, zorder=4))
ax.plot([2.65, 3.4, 4.1, 5.0], [5.3, 4.7, 5.1, 4.95], color=BLUE, lw=0.7, ls=(0, (1, 1)), zorder=1)
ax.plot([5.6, 6.4, 7.0, 7.45], [5.1, 5.6, 5.8, 5.9], color="#1baf7a", lw=0.7, ls=(0, (1, 1)), zorder=1)
for px, py, c in [(4.1, 5.1, BLUE), (3.4, 4.7, BLUE), (6.4, 5.6, ORANGE), (7.0, 5.8, ORANGE)]:
    scatter_by_color(ax, [px], [py], 24, [c], edgecolors="white", linewidths=0.6, zorder=5)
ax.text(0.2, 9.65, "Automatic reconstruction", fontsize=11, fontweight="bold", color=INK, va="top")
ax.text(0.2, 8.85, "flood-filling networks trace every process; pieces", fontsize=8.4, color=MUTE, va="top")
ax.text(0.2, 8.3, "are merged (version c3); synapses detected and typed", fontsize=8.4, color=MUTE, va="top")
ax.scatter([0.35], [2.55], s=22, c=BLUE, edgecolors="white", linewidths=0.5)
ax.text(0.6, 2.55, "excitatory-type synapse", fontsize=7.8, color=MUTE, va="center")
scatter_by_color(ax, [5.0], [2.55], 22, [ORANGE], edgecolors="white", linewidths=0.5)
ax.text(5.25, 2.55, "inhibitory-type", fontsize=7.8, color=MUTE, va="center")
ax.text(0.2, 1.75, "149,871,669 synapses  ·  57,180 cells with a nucleus", fontsize=8.8, color=INK, va="top", fontweight="bold")
ax.text(0.2, 1.05, "16,087 neurons, 32,315 glia, 8,100 vessel cells", fontsize=8.4, color=INK, va="top")
ax.text(0.2, 0.4, "104 neurons proofread in the paper", fontsize=8, color=MUTE, va="top")
title(ax, "c", "Segmentation and synapses", "published H01 resource")

# ---------------------------------------------------------------- d: funnel (our extraction)
ax = ax_d
stages = [("soma-labelled cell IDs in the synapse export", 73896, "73,896"),
          ("neurons by our rule\n(at least 5 soma rows and 20 dendrite-receiver rows)", 16599, "16,599"),
          ("neurons with a connection to another neuron\n(87,706 directed connections)", 14683, "14,683"),
          ("giant strongly connected core\n(every neuron reaches every other)", 10133, "10,133"),
          ("subgraph used as the reservoir\n(grown from one neuron, 2,981 connections)", 300, "300")]
ys = np.arange(len(stages))[::-1] * 1.0
for (lab, v, txt), yy in zip(stages, ys):
    ax.plot([100, v], [yy, yy], color="#bfd5f2" if v != 300 else "#f7c6b1", lw=2.2, zorder=1, solid_capstyle="butt")
    ax.scatter([v], [yy], s=95, color=BLUE if v != 300 else ORANGE, edgecolors="white", linewidths=1.2, zorder=3)
    ax.text(v * 1.22, yy, txt, va="center", ha="left", fontsize=10.5, fontweight="bold", color=INK, zorder=4,
            bbox=dict(fc="white", ec="none", pad=1.5))
    ax.text(105, yy + 0.2, lab, va="bottom", ha="left", fontsize=8.6, color=MUTE, linespacing=1.1)
ax.set_xscale("log")
ax.set_xlim(100, 1.4e5)
ax.set_ylim(-0.45, len(stages) - 0.05)
ax.set_yticks([])
ax.set_xticks([100, 1000, 10000, 100000]); ax.set_xticklabels(["100", "1,000", "10,000", "100,000"], fontsize=8.5, color=MUTE)
ax.set_xlabel("cells (log scale)", fontsize=8.8, color=MUTE)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(LINE)
ax.tick_params(axis="x", colors=LINE, length=3)
ax.xaxis.grid(True, color="#ecebe6", lw=0.6, zorder=0)
# the paper's neuron count as a reference mark
ax.axvline(16087, ymin=0.0, ymax=0.98, color=MUTE, lw=0.8, ls=(0, (3, 3)), zorder=1)
ax.text(16087 * 1.04, 0.35, "16,087 neurons counted\nby nucleus in the paper", fontsize=7.8, color=MUTE, va="bottom")
ax.text(0.0, 1.07, "d", transform=ax.transAxes, fontsize=14, fontweight="bold", color=INK, va="bottom")
ax.text(0.035, 1.07, "From the synapse table to a subgraph", transform=ax.transAxes, fontsize=11.5, fontweight="bold",
        color=INK, va="bottom")
ax.text(0.035, 1.015, "this work: 166 table shards, 160,259,769 synapse rows read", transform=ax.transAxes, fontsize=8.8,
        color=MUTE, va="bottom")

# ---------------------------------------------------------------- e: subgraph and reservoir
ax = ax_e
g = R.apply_sign_policy(R.load_subgraph(os.path.join(ROOT, "subgraphs"), "A_300_s0"), "data")
D = nx.DiGraph(); D.add_nodes_from(range(g["n"])); D.add_edges_from(zip(g["pre"], g["post"]))
U = D.to_undirected()
pos = nx.spring_layout(U, seed=3, k=2.2 / np.sqrt(g["n"]), iterations=400)
arr = np.array([pos[i] for i in range(g["n"])])
arr = (arr - arr.mean(0)) / np.abs(arr - arr.mean(0)).max(0)
deg = dict(U.degree())
col = [(BLUE if s > 0 else ORANGE) if k else GREY for s, k in zip(g["sign"], g["known"])]
sub = fig.add_axes([0, 0, 1, 1]); sub.set_axis_off()
bb = ax_e.get_position()
gax = fig.add_axes([bb.x0, bb.y0 + bb.height * 0.36, bb.width, bb.height * 0.64]); gax.set_axis_off()
gax.add_collection(LineCollection([(arr[a], arr[b]) for a, b in D.edges], colors="#8a8984", linewidths=0.25, alpha=0.35))
scatter_by_color(gax, arr[:, 0], arr[:, 1], [4 + 2.2 * deg[i] for i in range(g["n"])], col, edgecolors="white", linewidths=0.3, zorder=2)
gax.set_xlim(-1.08, 1.08); gax.set_ylim(-1.08, 1.08)
sub.remove()
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.text(5, 3.55, "190 excitatory · 27 inhibitory · 83 sign unknown", ha="center", fontsize=8.2, color=INK)
ax.add_patch(FancyArrowPatch((5, 2.9), (5, 2.15), arrowstyle="-|>", mutation_scale=11, color=INK, lw=1))
ax.add_patch(FancyBboxPatch((0.4, 0.2), 9.2, 1.85, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#f4f3ee", ec=LINE, lw=0.8))
ax.text(5, 1.5, "recurrent matrix W of a reservoir", ha="center", fontsize=9, fontweight="bold", color=INK)
ax.text(5, 0.72, "weight = sign of the sending neuron × √(synapse rows)", ha="center", fontsize=8.2, color=MUTE)
ax.text(0.0, 1.07, "e", transform=ax.transAxes, fontsize=14, fontweight="bold", color=INK, va="bottom")
ax.text(0.07, 1.07, "The reservoir", transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=INK, va="bottom")
ax.text(0.07, 1.015, "300 neurons, wiring from H01", transform=ax.transAxes, fontsize=8.8, color=MUTE, va="bottom")

# ---------------------------------------------------------------- arrows between stages and footer
def arrow(x0, x1, y):
    fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14,
                                   color=INK, lw=1.1))

pa, pb, pc = ax_a.get_position(), ax_b.get_position(), ax_c.get_position()
ymid = pa.y0 + pa.height * 0.5
arrow(pa.x1 + 0.003, pb.x0 + 0.012, ymid)
arrow(pb.x1 + 0.003, pc.x0 + 0.012, ymid)
fig.text(0.03, 0.968, "From human cortex to a robot's recurrent network", fontsize=15, fontweight="bold", color=INK, va="center")
fig.text(0.03, 0.941, "Panels a to c summarise the published H01 dataset; panels d and e are our extraction of a connected subgraph.",
         fontsize=9.5, color=MUTE, va="center")
fig.text(0.03, 0.04, "a-c: numbers from Shapson-Coe et al., Science 384, eadk4858 (2024); drawings are schematics made for this figure. "
         "d-e: our extraction from the c3 synapse export.", fontsize=7.8, color=MUTE)
fig.text(0.03, 0.018, "Excitatory/inhibitory sign: from the synapse type label, with type 1 read as inhibitory (a working assumption, see text); "
         "neurons without enough evidence are shown grey and treated as excitatory.", fontsize=7.8, color=MUTE)
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(OUT, f"fig_pipeline.{ext}"), facecolor="white")
print("saved")

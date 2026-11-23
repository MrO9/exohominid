"""Figure: three networks side by side, zooming from a human brain to the H01 temporal-cortex neurons to our reservoir subgraph.
Panel a is an ILLUSTRATION (no data: a brain outline with random dots and links); panels b and c are drawn from the real data
(all neurons we identified in H01 with their connections; our 300-neuron subgraph). Layouts are force-directed, not anatomical."""
import os, sys
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, ConnectionPatch, PathPatch, Polygon
from matplotlib.path import Path
from scipy.interpolate import splprep, splev

ROOT = ".."
sys.path.insert(0, os.path.join(ROOT, "webots", "controllers", "h01_dataset_policy"))
import reservoir as R
from shapes import scatter_by_color

OUT = os.path.join(ROOT, "figures")
BLUE, ORANGE, GREY, VIOLET, SLATE = "#2a78d6", "#eb6834", "#a8a7a2", "#6250d6", "#7a8794"
INK, MUTE, LINE = "#0b0b0b", "#52514e", "#c9c8c2"
rng = np.random.default_rng(7)

fig = plt.figure(figsize=(15, 6.6), dpi=200)
fig.patch.set_facecolor("white")
axs = [fig.add_axes([0.02 + i * 0.327, 0.2, 0.30, 0.6]) for i in range(3)]
for a in axs:
    a.set_axis_off()

# ------------------------------------------------------------------ a: illustrative brain
ax = axs[0]
pts = np.array([[0.00, 0.45], [0.05, 0.62], [0.17, 0.78], [0.38, 0.9], [0.62, 0.92], [0.82, 0.84], [0.95, 0.68], [1.0, 0.5],
                [0.97, 0.36], [0.86, 0.29], [0.74, 0.24], [0.62, 0.13], [0.46, 0.1], [0.32, 0.17], [0.22, 0.27], [0.1, 0.32]])
tck, _ = splprep(np.vstack([pts[:, 0], pts[:, 1]]), s=0, per=True)
bx, by = splev(np.linspace(0, 1, 400), tck)
outline = np.column_stack([bx, by])
path = Path(outline)
ax.add_patch(Polygon(outline, closed=True, fc="#f1f0ec", ec="#8a8984", lw=1.2, zorder=0))
cer = np.array([[0.74, 0.22], [0.8, 0.1], [0.92, 0.08], [0.98, 0.17], [0.93, 0.27], [0.82, 0.29]])
ctck, _ = splprep(cer.T, s=0, per=True)
cx, cy = splev(np.linspace(0, 1, 100), ctck)
ax.add_patch(Polygon(np.column_stack([cx, cy]), closed=True, fc="#e4e2dc", ec="#8a8984", lw=1.0, zorder=0))
# dots, denser near the surface (the cortex)
cand = rng.uniform([0, 0.05], [1, 0.95], size=(30000, 2))
inside = cand[path.contains_points(cand)]
d = np.min(np.linalg.norm(inside[:, None, :] - outline[None, ::8, :], axis=2), axis=1)
keep = rng.uniform(size=len(inside)) < np.exp(-d / 0.09) * 0.9 + 0.12
dots = inside[keep][:3200]
# short links to nearest neighbours plus long-range arcs
from scipy.spatial import cKDTree
tree = cKDTree(dots)
_, nn = tree.query(dots, k=4)
segs = [(dots[i], dots[j]) for i in range(len(dots)) for j in nn[i, 1:3]]
ax.add_collection(LineCollection(segs, colors="#8a8984", linewidths=0.25, alpha=0.35, zorder=1))
for _ in range(220):
    i, j = rng.integers(0, len(dots), 2)
    if np.linalg.norm(dots[i] - dots[j]) < 0.35:
        continue
    t = np.linspace(0, 1, 30)
    mid = (dots[i] + dots[j]) / 2 + np.array([0, 0.18]) * rng.uniform(0.6, 1.2)
    curve = np.outer((1 - t) ** 2, dots[i]) + np.outer(2 * (1 - t) * t, mid) + np.outer(t ** 2, dots[j])
    ax.plot(curve[:, 0], curve[:, 1], color="#8a8984", lw=0.3, alpha=0.3, zorder=1)
ax.scatter(dots[:, 0], dots[:, 1], s=2.2, c=SLATE, zorder=2, linewidths=0)
# where H01 sits (not to scale: the real sample is about a millionth of the brain)
tpos = np.array([0.52, 0.2])
ax.add_patch(Rectangle(tpos - 0.035, 0.07, 0.07, fc="none", ec=VIOLET, lw=1.6, zorder=4))
ax.set_xlim(-0.04, 1.04); ax.set_ylim(0.0, 1.0); ax.set_aspect("equal")
ax.text(0.5, -0.04, "Whole human brain", transform=ax.transAxes, ha="center", va="top", fontsize=12.5, fontweight="bold", color=INK)
ax.text(0.5, -0.115, "about 86 billion neurons, 10¹⁴ to 10¹⁵ synapses", transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=INK)
ax.text(0.5, -0.18, "illustration: 3,200 dots drawn, no data", transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=MUTE)

# ------------------------------------------------------------------ b: real H01 neurons
ax = axs[1]
lay = np.load(os.path.join(OUT, "layout_h01_neurons.npz"))
ids, xy = [str(i) for i in lay["ids"]], lay["xy"]
xy = (xy - np.median(xy, axis=0))
sc = np.percentile(np.abs(xy), 99.5)
xy = xy / sc
pos = dict(zip(ids, xy))
n_all = int(lay["n_all"])
edges = []
with open(os.path.join(ROOT, "outputs", "graph_v2", "graph_edges.tsv")) as f:
    f.readline()
    for l in f:
        p = l.split()
        if p[0] in pos and p[1] in pos and p[0] != p[1]:
            edges.append((pos[p[0]], pos[p[1]]))
ax.add_collection(LineCollection(edges, colors="#7b9cc4", linewidths=0.12, alpha=0.18, zorder=1))
ax.scatter(xy[:, 0], xy[:, 1], s=0.9, c=BLUE, alpha=0.8, zorder=2, linewidths=0)
sub_ids = [l.split()[0] for l in open(os.path.join(ROOT, "webots", "subgraphs", "nodes_A_300_s0.tsv")).read().splitlines()[1:]]
sx = np.array([pos[i] for i in sub_ids if i in pos])
ax.scatter(sx[:, 0], sx[:, 1], s=5, c=VIOLET, zorder=3, linewidths=0)
c0, r0 = sx.mean(axis=0), 1.35 * np.abs(sx - sx.mean(axis=0)).max()
ax.add_patch(Rectangle(c0 - r0, 2 * r0, 2 * r0, fc="none", ec=VIOLET, lw=1.6, zorder=4))
ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal")
ax.text(0.5, -0.04, "Neurons identified in H01", transform=ax.transAxes, ha="center", va="top", fontsize=12.5, fontweight="bold", color=INK)
ax.text(0.5, -0.115, f"{len(ids):,} of {n_all:,} neurons, {len(edges):,} connections (largest connected group)", transform=ax.transAxes,
        ha="center", va="top", fontsize=9.5, color=INK)
ax.text(0.5, -0.18, "real data: 1 mm³ of temporal cortex; layout not anatomical", transform=ax.transAxes, ha="center", va="top",
        fontsize=8.5, color=MUTE)

# ------------------------------------------------------------------ c: real subgraph
ax = axs[2]
g = R.apply_sign_policy(R.load_subgraph(os.path.join(ROOT, "webots", "subgraphs"), "A_300_s0"), "data")
D = nx.DiGraph(); D.add_nodes_from(range(g["n"])); D.add_edges_from(zip(g["pre"], g["post"]))
U = D.to_undirected()
p3 = nx.spring_layout(U, seed=3, k=2.2 / np.sqrt(g["n"]), iterations=400)
arr = np.array([p3[i] for i in range(g["n"])])
arr = (arr - arr.mean(0)) / np.abs(arr - arr.mean(0)).max(0)
deg = dict(U.degree())
col = [(BLUE if s > 0 else ORANGE) if k else GREY for s, k in zip(g["sign"], g["known"])]
ax.add_collection(LineCollection([(arr[a], arr[b]) for a, b in D.edges], colors="#8a8984", linewidths=0.3, alpha=0.38, zorder=1))
scatter_by_color(ax, arr[:, 0], arr[:, 1], [5 + 2.6 * deg[i] for i in range(g["n"])], col, edgecolors="white", linewidths=0.35, zorder=2)
ax.add_patch(Rectangle((-1.12, -1.12), 2.24, 2.24, fc="none", ec=VIOLET, lw=1.6, zorder=0))
ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal")
ax.text(0.5, -0.04, "Our reservoir subgraph", transform=ax.transAxes, ha="center", va="top", fontsize=12.5, fontweight="bold", color=INK)
ax.text(0.5, -0.115, "300 neurons, 2,981 connections, 5,163 synapses", transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=INK)
ax.text(0.5, -0.18, "real data; blue excitatory, orange inhibitory, grey sign unknown", transform=ax.transAxes, ha="center", va="top",
        fontsize=8.5, color=MUTE)

# ------------------------------------------------------------------ zoom lines and labels
for (a_from, a_to, p_from, p_to) in ((axs[0], axs[1], (0.52 + 0.035, 0.2 + 0.035), (-1.15, 1.15)),
                                     (axs[1], axs[2], (c0[0] + r0, c0[1] + r0), (-1.12, 1.12))):
    for sgn in (1, -1):
        pf = (p_from[0], p_from[1] if sgn == 1 else p_from[1] - (0.07 if a_from is axs[0] else 2 * r0))
        pt = (p_to[0], p_to[1] if sgn == 1 else -1.15 if a_to is axs[1] else -1.12)
        fig.add_artist(ConnectionPatch(xyA=pf, coordsA=a_from.transData, xyB=pt, coordsB=a_to.transData, color=VIOLET, lw=0.9,
                                       ls=(0, (4, 3)), alpha=0.8, zorder=10))
fig.text(0.02, 0.945, "Zooming from a human brain to our reservoir", fontsize=16, fontweight="bold", color=INK)
fig.text(0.02, 0.905, "Three networks, each a small piece of the one before. Only the left panel is an illustration; the other two are drawn from the data.",
         fontsize=10, color=MUTE)
for ax, lab, x in zip(axs, "abc", (0.02, 0.347, 0.674)):
    fig.text(x, 0.835, lab, fontsize=14, fontweight="bold", color=INK)
fig.text(0.02, 0.025, "Sources: brain counts, literature estimates (Azevedo et al. 2009 for neurons); H01, Shapson-Coe et al., Science 384, eadk4858 (2024). "
         "Sizes of the violet boxes are not to scale: the H01 sample is about a millionth of the brain.", fontsize=8, color=MUTE)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_nested.{ext}"), facecolor="white")
print("saved")

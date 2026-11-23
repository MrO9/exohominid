"""Schematic and data figures for the LaTeX paper (paper/figs). Run from this folder:  python paper_figs.py [outdir]
Blue = excitatory, orange = inhibitory, grey = sign unknown. Data figures read ../webots/subgraphs."""
import os, sys, math, csv
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon, Ellipse
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from scipy.interpolate import splprep, splev
from shapes import scatter_by_color

OUT = sys.argv[1] if len(sys.argv) > 1 else "../paper/figs"
os.makedirs(OUT, exist_ok=True)
SUB = "../webots/subgraphs"

PEACH, BLUEF, GREYF = "#fbe3d3", "#dce9f7", "#efefef"
STROKE, INK, MUTE = "#5a5a5a", "#222222", "#6b6b6b"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a8a7a2"
GREEN = "#2e8b57"
plt.rcParams["font.family"] = "DejaVu Sans"


def canvas(w, h, xl, yl, dpi=220):
    fig = plt.figure(figsize=(w, h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, xl); ax.set_ylim(0, yl); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", fc=GREYF, ec=STROKE, fs=9, dashed=False, bold=False, r=0.12, lw=1.2, color=INK, va="center"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                       ls="--" if dashed else "-", zorder=2)
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va=va, fontsize=fs, color=color,
                fontweight="bold" if bold else "normal", zorder=3, linespacing=1.35)
    return p


def arrow(ax, a, b, text=None, dashed=False, fs=8, off=(0, 0.14), color=STROKE, lw=1.3, rad=0.0, style="-|>", ha="center"):
    ar = FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, lw=lw, color=color,
                         ls="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}", zorder=4, shrinkA=0, shrinkB=0)
    ax.add_patch(ar)
    if text:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, ha=ha, va="center", fontsize=fs, color=MUTE, zorder=5)


def robot_icon(ax, cx, cy, s=1.0, col="#555555"):
    ax.add_patch(Circle((cx, cy + 0.52 * s), 0.11 * s, fc=col, ec="none", zorder=4))
    ax.add_patch(Rectangle((cx - 0.17 * s, cy + 0.12 * s), 0.34 * s, 0.30 * s, fc=col, ec="none", zorder=4))
    for sx in (-1, 1):
        ax.plot([cx + sx * 0.17 * s, cx + sx * 0.28 * s], [cy + 0.40 * s, cy + 0.16 * s], color=col, lw=2.6 * s, solid_capstyle="round", zorder=4)
        ax.plot([cx + sx * 0.07 * s, cx + sx * 0.09 * s], [cy + 0.12 * s, cy - 0.30 * s], color=col, lw=2.6 * s, solid_capstyle="round", zorder=4)


def mini_net(ax, cx, cy, w, h, n=18, seed=2, p=0.18):
    rng = np.random.default_rng(seed)
    pts = np.c_[cx + (rng.random(n) - .5) * w, cy + (rng.random(n) - .5) * h]
    col = [ORANGE if rng.random() < 0.18 else BLUE for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < p:
                ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color="#9a9a95", lw=0.4, alpha=.6, zorder=3)
    scatter_by_color(ax, pts[:, 0], pts[:, 1], 16, col, edgecolors="white", linewidths=0.4, zorder=4)


def load_sub(name):
    nodes = {}
    with open(f"{SUB}/nodes_{name}.tsv") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            nodes[row["id"]] = (int(row["sign"]), row["sign_source"])
    ids = list(nodes)
    idx = {k: i for i, k in enumerate(ids)}
    edges = []
    with open(f"{SUB}/edges_{name}.tsv") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            edges.append((idx[row["pre"]], idx[row["post"]], int(row["weight"])))
    sign = np.array([nodes[k][0] for k in ids])
    known = np.array([nodes[k][1] == "data" for k in ids])
    return ids, sign, known, edges


# ------------------------------------------------------------------ Figure 1: overview
def fig_overview():
    fig, ax = canvas(13, 3.6, 13, 3.6)
    xs = [0.2, 2.85, 5.5, 8.15, 10.8]
    w, h, y = 2.0, 1.6, 1.25
    box(ax, xs[0], y, w, h, "", BLUEF)
    box(ax, xs[1], y, w, h, "", GREYF)
    box(ax, xs[2], y, w, h, "", PEACH, lw=1.6)
    box(ax, xs[3], y, w, h, "", PEACH, ec=ORANGE, lw=1.8)
    box(ax, xs[4], y, w, h, "", BLUEF)
    ax.text(xs[0] + w / 2, y + h - 0.2, "human cortex (H01)", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(xs[0] + w / 2, y + 0.55, "1.05 mm$^3$, 16,087 neurons\n149.9 M synapses", ha="center", fontsize=8.2, color=INK, linespacing=1.4)
    ax.text(xs[1] + w / 2, y + h - 0.2, "signed neuron graph", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(xs[1] + w / 2, y + 0.55, "16,599 neurons\n87,706 connections", ha="center", fontsize=8.2, color=INK, linespacing=1.4)
    ax.text(xs[2] + w / 2, y + h - 0.2, "reservoir (fixed)", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    mini_net(ax, xs[2] + w / 2, y + 0.58, 1.8, 0.7, n=22)
    ax.text(xs[3] + w / 2, y + h - 0.2, "readout (trained)", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(xs[3] + w / 2, y + 0.55, "903 numbers\n$\\rightarrow$ 3 corrections", ha="center", fontsize=8.2, color=INK, linespacing=1.4)
    ax.text(xs[4] + w / 2, y + h - 0.2, "DARwIn-OP + box", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    robot_icon(ax, xs[4] + 0.6, y + 0.2, 0.85)
    ax.add_patch(Rectangle((xs[4] + 1.2, y + 0.22), 0.55, 0.55, fc="#d8b98a", ec=STROKE, lw=1, zorder=4))
    arrow(ax, (xs[0] + w, y + h / 2), (xs[1], y + h / 2), "read\ntables", off=(0, 0.3), fs=7.2)
    arrow(ax, (xs[1] + w, y + h / 2), (xs[2], y + h / 2), "300\nneurons", off=(0, 0.3), fs=7.2)
    arrow(ax, (xs[2] + w, y + h / 2), (xs[3], y + h / 2), "state", off=(0, 0.2), fs=7.2)
    arrow(ax, (xs[3] + w, y + h / 2), (xs[4], y + h / 2), "3 gait\nfixes", off=(0, 0.3), fs=7.2)
    arrow(ax, (xs[4] + w / 2, y), (xs[2] + w / 2, y), "gyroscope, accelerometer, gait phase", dashed=True, rad=-0.28, off=(0, -0.78), fs=8)
    ax.text(xs[2] + w / 2, 0.12, "matched controls: degree-preserving shuffle, random graph, ESN, permuted signs, no reservoir",
            ha="center", fontsize=8, color=MUTE, style="italic")
    ax.text(0.2, 3.35, "From a human cortical connectome to a robot controller", fontsize=10.5, fontweight="bold", color=INK)
    fig.savefig(f"{OUT}/fig_overview.png"); plt.close(fig)


# ------------------------------------------------------------------ Figure: excitatory vs inhibitory
def fig_ei():
    ids, sign, known, edges = load_sub("A_300_s0")
    n = len(ids)
    W = np.zeros((n, n))
    for a, b, wt in edges:
        W[b, a] += sign[a] * math.sqrt(wt)
    order = np.lexsort((np.arange(n), np.where(~known, 1, np.where(sign > 0, 0, 2))))   # excitatory, unknown, inhibitory
    Wp = W[np.ix_(order, order)]
    grp = np.where(~known, 1, np.where(sign > 0, 0, 2))[order]
    n_exc, n_unk, n_inh = (grp == 0).sum(), (grp == 1).sum(), (grp == 2).sum()

    fig = plt.figure(figsize=(13, 4.6), dpi=220)
    # (a) synapse schematic
    axa = fig.add_axes([0.01, 0.04, 0.36, 0.86]); axa.set_xlim(0, 6); axa.set_ylim(0, 5); axa.axis("off"); axa.set_aspect("equal")
    fig.text(0.015, 0.945, "(a) Two kinds of synapse", fontsize=10.5, fontweight="bold", color=INK, va="center")
    for yy, col, lab, sgn in ((3.45, BLUE, "excitatory sender", +1), (1.35, ORANGE, "inhibitory sender", -1)):
        axa.scatter([0.7], [yy], s=1900 if sgn > 0 else 2500, c=col, marker="o" if sgn > 0 else "^", edgecolors="white", linewidths=1.2, zorder=3)
        axa.scatter([3.55], [yy], s=1900, c=GREY, marker="o", edgecolors="white", linewidths=1.2, zorder=3)
        axa.plot([1.08, 3.17], [yy, yy], color=col, lw=2.2, zorder=2)
        axa.text(0.7, yy - 0.62, lab, ha="center", fontsize=8.2, color=INK)
        axa.text(3.55, yy - 0.62, "receiving neuron", ha="center", fontsize=8.2, color=INK)
        axa.text(2.1, yy + 0.2, "synapse", ha="center", fontsize=7.5, color=MUTE)
        t = np.linspace(0, 1, 60)
        base = yy
        trace = sgn * 0.55 * np.exp(-((t - 0.3) / 0.14) ** 2) * (1 - 0.1 * t)
        axa.plot(4.35 + t * 1.5, base + trace, color=col, lw=2.0, zorder=3)
        axa.plot([4.35, 5.85], [base, base], color="#bbbbbb", lw=0.8, ls=":", zorder=1)
    axa.text(5.1, 4.1, "pushes the\nreceiver up", ha="center", fontsize=8, color=BLUE, linespacing=1.3)
    axa.text(5.1, 2.0, "pushes the\nreceiver down", ha="center", fontsize=8, color=ORANGE, linespacing=1.3)
    axa.text(0.0, 0.28, "Type 1 read as inhibitory is an assumption (see text).", fontsize=7.4, color=MUTE, style="italic")
    # (b) motif
    axb = fig.add_axes([0.385, 0.04, 0.27, 0.86]); axb.set_xlim(0, 5); axb.set_ylim(0, 5); axb.axis("off"); axb.set_aspect("equal")
    fig.text(0.385, 0.945, "(b) How signs combine in a reservoir", fontsize=10.5, fontweight="bold", color=INK, va="center")
    pos = {"A": (0.7, 3.6), "B": (0.7, 2.3), "C": (0.7, 1.0), "R": (3.2, 2.3)}
    cols = {"A": BLUE, "B": BLUE, "C": ORANGE, "R": GREY}
    for k, (x, y) in pos.items():
        axb.scatter([x], [y], s=1500 if k != "C" else 2000, c=cols[k], marker="^" if k == "C" else "o", edgecolors="white", linewidths=1.2, zorder=3)
        axb.text(x, y - (0.06 if k == "C" else 0), k, ha="center", va="center", fontsize=8.5, color="white", fontweight="bold", zorder=4)
    for k, lab in (("A", "$+\\sqrt{n_{AR}}$"), ("B", "$+\\sqrt{n_{BR}}$")):
        x, y = pos[k]
        axb.add_patch(FancyArrowPatch((x + .36, y - 0.02), (pos["R"][0] - .38, pos["R"][1] + (0.1 if k == "A" else 0.02)), arrowstyle="-|>", mutation_scale=11, lw=1.6, color=BLUE, zorder=2))
        axb.text((x + pos["R"][0]) / 2, (y + pos["R"][1]) / 2 + (0.32 if k == "A" else 0.2), lab, fontsize=8, color=BLUE, ha="center")
    x, y = pos["C"]
    tip = (pos["R"][0] - .30, pos["R"][1] - .22)
    axb.plot([x + .36, tip[0]], [y + .08, tip[1]], color=ORANGE, lw=1.6, zorder=2)
    d = np.array([tip[0] - (x + .36), tip[1] - (y + .08)]); d /= np.linalg.norm(d); nrm = np.array([-d[1], d[0]]) * 0.16
    axb.plot([tip[0] - nrm[0], tip[0] + nrm[0]], [tip[1] - nrm[1], tip[1] + nrm[1]], color=ORANGE, lw=2.8, zorder=2, solid_capstyle="butt")
    axb.text(1.7, 1.75, "$-\\sqrt{n_{CR}}$", fontsize=8, color=ORANGE, ha="center")
    axb.text(3.2, 1.55, "$x_R \\leftarrow \\tanh\\left(\\sum_j W_{Rj}\\,x_j\\right)$", ha="center", fontsize=8.6, color=INK)
    axb.text(3.2, 0.75, "weight = sender's sign\n$\\times$ $\\sqrt{\\mathrm{synapse\\ rows}}$", ha="center", fontsize=8.2, color=MUTE, linespacing=1.4)
    axb.text(0.0, 0.1, "A, B, C send; R receives. One sign per sender.", fontsize=7.4, color=MUTE, style="italic")
    # (c) real matrix
    axc = fig.add_axes([0.70, 0.12, 0.27, 0.70])
    cmap = LinearSegmentedColormap.from_list("sg", [ORANGE, "#ffffff", BLUE])
    lim = 2.0
    axc.imshow(Wp, cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest", aspect="equal")
    axc.set_xticks([]); axc.set_yticks([])
    for s in axc.spines.values():
        s.set_color("#bbbbbb")
    for xb in (n_exc, n_exc + n_unk):
        axc.axvline(xb - 0.5, color="#888888", lw=0.5, ls=":")
    axc.set_xlabel("sender neuron (sorted by sign)", fontsize=8.5, color=MUTE, labelpad=3)
    axc.set_ylabel("receiving neuron", fontsize=8.5, color=MUTE, labelpad=3)
    # group bar on top
    axg = fig.add_axes([0.70, 0.83, 0.27, 0.025]); axg.axis("off"); axg.set_xlim(0, n); axg.set_ylim(0, 1)
    for a, b, c in ((0, n_exc, BLUE), (n_exc, n_exc + n_unk, GREY), (n_exc + n_unk, n, ORANGE)):
        axg.add_patch(Rectangle((a, 0), b - a, 1, fc=c, ec="none"))
    fig.text(0.70, 0.945, "(c) The recurrent matrix of subgraph s0", fontsize=10.5, fontweight="bold", color=INK, va="center")
    fig.text(0.70, 0.895, f"{n_exc} excitatory (circle), {n_unk} unknown (diamond, treated as +), {n_inh} inhibitory (triangle)", fontsize=7.0, color=MUTE, va="center")
    fig.savefig(f"{OUT}/fig_ei.png"); plt.close(fig)
    return n_exc, n_unk, n_inh


# ------------------------------------------------------------------ Figure: three subgraphs
def sub_stats(name):
    ids, sign, known, edges = load_sub(name)
    n = len(ids)
    E = len(edges)
    rows = sum(w for _, _, w in edges)
    es = {(a, b) for a, b, _ in edges}
    recip = sum(1 for a, b in es if (b, a) in es) / E
    n_inh = int(((sign < 0) & known).sum()); n_known = int(known.sum())
    return dict(n=n, E=E, rows=rows, deg=2 * E / n, recip=recip, inh=n_inh, known=n_known, unk=n - n_known,
                ids=ids, sign=sign, known_arr=known, edges=edges)


def fig_subgraphs():
    names = ["A_300_s0", "A_300_s1", "A_300_s2"]
    succ = {"A_300_s0": (30, 29), "A_300_s1": (30, 27), "A_300_s2": (13, 29)}   # measured signs, permuted signs (final queue, of 30)
    fig = plt.figure(figsize=(13, 6.6), dpi=220)
    stats = {}
    for k, nm in enumerate(names):
        S = sub_stats(nm); stats[nm] = S
        ax = fig.add_axes([0.02 + k * 0.325, 0.36, 0.30, 0.56])
        D = nx.DiGraph(); D.add_nodes_from(range(S["n"])); D.add_edges_from([(a, b) for a, b, _ in S["edges"]])
        U = D.to_undirected()
        pos = nx.spring_layout(U, seed=3, k=2.2 / math.sqrt(S["n"]), iterations=400)
        arr = nx.rescale_layout(np.array([pos[i] for i in range(S["n"])]), scale=1.0)
        deg = dict(U.degree())
        segs = [(arr[a], arr[b]) for a, b in D.edges]
        ax.add_collection(LineCollection(segs, colors="#8a8984", linewidths=0.3, alpha=0.4, zorder=1))
        col = [(BLUE if s > 0 else ORANGE) if kn else GREY for s, kn in zip(S["sign"], S["known_arr"])]
        scatter_by_color(ax, arr[:, 0], arr[:, 1], [4 + 2.4 * deg[i] for i in range(S["n"])], col, edgecolors="white", linewidths=0.35, zorder=2)
        ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"subgraph s{k}", fontsize=12, fontweight="bold", color=INK, pad=4)
    # table
    axt = fig.add_axes([0.02, 0.02, 0.96, 0.30]); axt.set_xlim(0, 13); axt.set_ylim(0, 3.2); axt.axis("off")
    rows = [("connections (edges)", lambda S, nm: f"{S['E']:,}"),
            ("synapse rows (total weight)", lambda S, nm: f"{S['rows']:,}"),
            ("mean in+out degree", lambda S, nm: f"{S['deg']:.1f}"),
            ("reciprocal connections", lambda S, nm: f"{S['recip']:.1%}"),
            ("inhibitory of signed neurons", lambda S, nm: f"{S['inh']} of {S['known']} ({S['inh'] / S['known']:.1%})"),
            ("neurons with no sign evidence", lambda S, nm: f"{S['unk']} ({S['unk'] / S['n']:.0%})"),
            ("success, measured signs / permuted signs (of 30)", lambda S, nm: f"{succ[nm][0]} / {succ[nm][1]}")]
    colx = [4.9, 7.5, 10.1]
    axt.text(0.0, 3.05, "300 neurons each", fontsize=9, color=MUTE, fontweight="bold", va="center")
    for j, (lab, fn) in enumerate(rows):
        y = 2.65 - j * 0.40
        axt.add_patch(Rectangle((0, y - 0.19), 12.9, 0.38, fc="#f4f4f4" if j % 2 == 0 else "white", ec="none", zorder=0))
        axt.text(0.1, y, lab, fontsize=9, color=INK, va="center")
        for k, nm in enumerate(names):
            axt.text(colx[k] + 0.9, y, fn(stats[nm], nm), fontsize=9, color=INK, va="center", ha="center",
                     fontweight="bold" if (j == 6 and nm == "A_300_s2") else "normal")
    for k, nm in enumerate(names):
        axt.text(colx[k] + 0.9, 3.05, f"s{k}", fontsize=9.5, fontweight="bold", color=INK, ha="center", va="center")
    fig.savefig(f"{OUT}/fig_subgraphs.png"); plt.close(fig)
    return stats


# ------------------------------------------------------------------ Figure: reservoir -> policy -> robot
def fig_policy():
    fig, ax = canvas(13.6, 5.0, 13.6, 5.0)
    # sensors
    box(ax, 0.2, 1.45, 2.3, 2.5, "", GREYF)
    ax.text(1.35, 3.72, "sensor input $u_t$", ha="center", fontsize=9.5, fontweight="bold")
    for k, t in enumerate(["3 gyroscope", "3 accelerometer", "2 gait phase (sin, cos)", "4 simulator inputs\n(zeroed in the final queue)"]):
        ax.text(1.35, 3.25 - k * 0.5 - (0.08 if k == 3 else 0), t, ha="center", va="center", fontsize=8.2, color=INK, linespacing=1.2)
        if k < 3:
            ax.plot([0.3, 2.4], [3.0 - k * 0.5, 3.0 - k * 0.5], color="#cfcfcf", lw=0.8)
    # input weights + reservoir
    arrow(ax, (2.5, 2.7), (3.35, 2.7), "$W_{in}$ random,\n25% density", off=(0, 0.45), fs=7.8)
    box(ax, 3.35, 1.2, 3.1, 3.0, "", PEACH, lw=1.6)
    ax.text(4.9, 3.98, "reservoir (fixed)", ha="center", fontsize=9.5, fontweight="bold")
    mini_net(ax, 4.9, 2.7, 2.5, 1.45, n=36, seed=5, p=0.10)
    ax.text(4.9, 1.45, "300 neurons, H01 wiring $W$\n$x\\leftarrow(1-a)x+a\\tanh(Wx+W_{in}u)$", ha="center", fontsize=7.8, linespacing=1.4)
    arrow(ax, (6.45, 2.7), (7.15, 2.7), "state $x_t$", off=(0, 0.3), fs=7.8)
    # readout
    box(ax, 7.15, 1.7, 1.9, 2.0, "", PEACH, ec=ORANGE, lw=2.0)
    ax.text(8.1, 3.45, "linear readout", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(8.1, 2.85, "$W_{out}\\,[x;1]$", ha="center", fontsize=9)
    ax.text(8.1, 2.3, "trained:\n903 numbers", ha="center", fontsize=8.2, color=ORANGE, fontweight="bold", linespacing=1.3)
    arrow(ax, (9.05, 2.7), (9.7, 2.7))
    # outputs
    box(ax, 9.7, 1.45, 1.55, 2.5, "", GREYF)
    ax.text(10.47, 3.72, "3 corrections", ha="center", fontsize=9, fontweight="bold")
    for k, t in enumerate(["turn  ($\\leq$ 0.6)", "ankle lean ($\\leq$ 0.08 rad)", "stride ($1\\pm0.25$)"]):
        ax.text(10.47, 3.2 - k * 0.55, t, ha="center", va="center", fontsize=7.6)
    # walker
    box(ax, 11.75, 3.0, 1.55, 1.4, "", BLUEF)
    ax.text(12.52, 4.12, "open-loop CPG gait", ha="center", fontsize=8.3, fontweight="bold", va="center")
    t = np.linspace(0, 1, 50)
    ax.plot(11.9 + t * 1.25, 3.5 + 0.2 * np.sin(2 * math.pi * t), color="#0b6bb5", lw=2.0)
    arrow(ax, (11.25, 3.2), (11.75, 3.55), "add", off=(0.0, 0.3), fs=7.8)
    arrow(ax, (12.52, 3.0), (12.52, 2.3), "joint targets", off=(0.62, 0.0), fs=7.8)
    # robot
    box(ax, 11.6, 0.2, 1.85, 2.1, "", BLUEF)
    ax.text(12.52, 2.05, "DARwIn-OP (Webots)", ha="center", fontsize=8, fontweight="bold")
    robot_icon(ax, 12.2, 0.72, 0.62)
    ax.add_patch(Rectangle((12.65, 0.45), 0.42, 0.42, fc="#d8b98a", ec=STROKE, lw=1, zorder=4))
    # feedback
    arrow(ax, (11.6, 0.7), (1.35, 0.7), dashed=True, lw=1.3)
    ax.plot([1.35, 1.35], [0.7, 1.45], color=STROKE, ls="--", lw=1.3, zorder=4)
    ax.add_patch(FancyArrowPatch((1.35, 0.9), (1.35, 1.45), arrowstyle="-|>", mutation_scale=11, lw=1.3, color=STROKE, ls="--", zorder=4))
    ax.text(6.4, 0.45, "sensor readings at the next time step (32 ms)", ha="center", fontsize=8.2, color=MUTE)
    ax.text(0.2, 4.72, "Reservoir policy: from sensors to gait corrections", fontsize=10.5, fontweight="bold")
    ax.text(5.3, 4.72, "fixed part: no weight of the recurrent network is trained", fontsize=8.2, color=MUTE, style="italic")
    fig.savefig(f"{OUT}/fig_policy.png"); plt.close(fig)


# ------------------------------------------------------------------ Figure: task (top view)
def fig_task():
    fig = plt.figure(figsize=(9.5, 3.8), dpi=220)
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.82])
    ax.set_xlim(-0.35, 1.75); ax.set_ylim(-0.72, 0.62); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((0.5, 0.30), 0.6, 0.15, fc=BLUE, alpha=0.12, ec="none", zorder=1))
    ax.add_patch(Rectangle((0.5, -0.45), 0.6, 0.15, fc=BLUE, alpha=0.12, ec="none", zorder=1))
    ax.add_patch(Rectangle((0.5, -0.3), 0.6, 0.6, fc="#e4cba3", ec=STROKE, lw=1.4, zorder=3))
    ax.text(0.8, 0.0, "cube 0.6 m\n0.7 kg", ha="center", va="center", fontsize=9, zorder=5)
    ax.add_patch(Rectangle((0.9, -0.3), 0.6, 0.6, fc="none", ec=STROKE, lw=1.0, ls="--", zorder=2))
    ax.text(1.31, 0.0, "after a\nsuccessful\npush", ha="center", va="center", fontsize=7.6, color=MUTE)
    ax.plot([0.9, 0.9], [-0.5, 0.5], color=GREEN, lw=2.0, zorder=6)
    ax.text(0.93, 0.5, "goal line: box front face\nmoved 0.40 m", fontsize=8.2, color=GREEN, va="bottom", ha="left")
    ax.add_patch(Rectangle((-0.08, -0.12), 0.16, 0.24, fc="#555555", ec="none", zorder=4))
    ax.add_patch(Circle((0.0, 0.0), 0.05, fc="white", ec="none", zorder=5))
    arrow(ax, (0.1, 0.0), (0.45, 0.0), color=INK, lw=1.4)
    ax.text(0.28, 0.07, "walks", fontsize=8.2, ha="center")
    ax.text(0.0, -0.22, "DARwIn-OP", fontsize=8.5, ha="center")
    ax.annotate("", xy=(0.5, -0.58), xytext=(0.0, -0.58), arrowprops=dict(arrowstyle="<->", color=MUTE, lw=1.0))
    ax.text(0.25, -0.63, "0.5 m to the near face", fontsize=8, color=MUTE, ha="center", va="top")
    ax.text(0.5, 0.5, "box centre shifted\nsideways up to $\\pm$0.15 m", fontsize=8.2, color=BLUE, ha="right", va="bottom")
    fig.text(0.02, 0.93, "Pushing task, seen from above", fontsize=10.5, fontweight="bold")
    fig.savefig(f"{OUT}/fig_task.png"); plt.close(fig)


# ------------------------------------------------------------------ Figure: training (evolution strategy)
def fig_training():
    fig, ax = canvas(13, 4.6, 13, 4.6)
    ax.text(0.2, 4.35, "Training: an evolution strategy on the readout only", fontsize=10.5, fontweight="bold")
    y0, hh = 2.45, 1.6
    ym = y0 + hh / 2
    box(ax, 0.2, y0, 1.9, hh, "", PEACH, ec=ORANGE, lw=1.8)
    ax.text(1.15, y0 + hh - 0.28, "readout $\\theta$", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(1.15, y0 + 0.62, "903 numbers\nstart at 0 =\nopen-loop gait", ha="center", fontsize=8.2, linespacing=1.3)
    arrow(ax, (2.1, ym), (2.85, ym))
    box(ax, 2.85, y0, 2.25, hh, "", GREYF)
    ax.text(3.97, y0 + hh - 0.28, "1. sample", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(3.97, y0 + 0.62, "8 noise vectors $\\epsilon$ give\n16 candidates\n$\\theta \\pm 0.05\\,\\epsilon$ (antithetic)", ha="center", fontsize=8.2, linespacing=1.4)
    arrow(ax, (5.1, ym), (5.85, ym))
    box(ax, 5.85, y0, 2.35, hh, "", BLUEF)
    ax.text(7.02, y0 + hh - 0.28, "2. run in Webots", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(7.02, y0 + 0.62, "3 episodes per candidate,\nsame 3 box offsets for all;\nscore = mean episode score", ha="center", fontsize=8.2, linespacing=1.4)
    arrow(ax, (8.2, ym), (8.95, ym))
    box(ax, 8.95, y0, 2.2, hh, "", GREYF)
    ax.text(10.05, y0 + hh - 0.28, "3. update", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(10.05, y0 + 0.62, "centred ranks of the 16\nscores weight the noise:\n$\\theta\\leftarrow\\theta+\\eta\\,\\hat g$, $\\eta=0.03$", ha="center", fontsize=8.2, linespacing=1.4)
    yl = 1.95
    ax.plot([10.05, 10.05, 1.15], [y0, yl, yl], color=STROKE, ls="--", lw=1.3, zorder=4)
    ax.add_patch(FancyArrowPatch((1.15, yl), (1.15, y0), arrowstyle="-|>", mutation_scale=11, lw=1.3, color=STROKE, ls="--", zorder=4))
    ax.text(5.6, yl + 0.2, "repeat for 60 generations", ha="center", fontsize=8.6, color=MUTE)
    box(ax, 0.2, 0.15, 6.2, 1.15, "", PEACH, lw=1.4)
    ax.text(3.3, 1.05, "4. every 3 generations: validate the centre $\\theta$", ha="center", fontsize=9.2, fontweight="bold")
    ax.text(3.3, 0.55, "10 separate fixed box offsets (seeds 5000+i); the zero readout is the first candidate,\nso the kept checkpoint is never worse than open loop on validation", ha="center", fontsize=8.0, linespacing=1.35)
    arrow(ax, (6.4, 0.72), (7.35, 0.72))
    box(ax, 7.35, 0.15, 5.4, 1.15, "", BLUEF)
    ax.text(10.05, 1.05, "5. final evaluation", ha="center", fontsize=9.2, fontweight="bold")
    ax.text(10.05, 0.55, "best checkpoint on 30 fixed box offsets\n(not used for training or validation)", ha="center", fontsize=8.0, linespacing=1.35)
    fig.savefig(f"{OUT}/fig_training.png"); plt.close(fig)



# ================================================================== abstract bodies (original drawings, not copies of any product)
MAN, MAN_E = "#5b5b5b", "#262626"          # mannequin
ROB, ROB_E = "#4a5360", "#222a33"          # abstract robot


def cap(ax, p, q, w, fc, ec, z=3, edge=0.035):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=ec, lw=(w + 2 * edge) * 72, solid_capstyle="round", zorder=z)
    ax.plot([p[0], q[0]], [p[1], q[1]], color=fc, lw=w * 72, solid_capstyle="round", zorder=z + 0.1)


def joint(ax, p, r, fc="#8d8d8d", ec="#262626", z=5):
    ax.add_patch(Circle(p, r, fc=fc, ec=ec, lw=1.0, zorder=z))


def mannequin(ax, cx, y0, H):
    u = H / 3.0
    hip = (cx, y0 + 1.25 * u); sh = (cx, y0 + 2.2 * u)
    head = (cx, y0 + 2.75 * u)
    for sx in (-1, 1):
        cap(ax, (cx + sx * 0.17 * u, hip[1]), (cx + sx * 0.23 * u, y0 + 0.65 * u), 0.2 * u, MAN, MAN_E)
        cap(ax, (cx + sx * 0.23 * u, y0 + 0.65 * u), (cx + sx * 0.25 * u, y0 + 0.1 * u), 0.17 * u, MAN, MAN_E)
        ax.add_patch(FancyBboxPatch((cx + sx * 0.25 * u - 0.15 * u, y0 - 0.02 * u), 0.3 * u, 0.1 * u, boxstyle="round,pad=0,rounding_size=0.04", fc=MAN, ec=MAN_E, lw=1.0, zorder=3))
        cap(ax, (cx + sx * 0.46 * u, sh[1] - 0.08 * u), (cx + sx * 0.62 * u, y0 + 1.45 * u), 0.14 * u, MAN, MAN_E)
        cap(ax, (cx + sx * 0.62 * u, y0 + 1.45 * u), (cx + sx * 0.66 * u, y0 + 0.95 * u), 0.12 * u, MAN, MAN_E)
        joint(ax, (cx + sx * 0.46 * u, sh[1] - 0.08 * u), 0.08 * u, MAN, MAN_E)
    cap(ax, (cx, hip[1] + 0.05 * u), (cx, sh[1] - 0.05 * u), 0.62 * u, MAN, MAN_E)
    cap(ax, (cx - 0.38 * u, sh[1]), (cx + 0.38 * u, sh[1]), 0.2 * u, MAN, MAN_E)
    cap(ax, (cx, sh[1]), (cx, y0 + 2.5 * u), 0.14 * u, MAN, MAN_E)
    ax.add_patch(Ellipse(head, 0.42 * u, 0.5 * u, fc=MAN, ec=MAN_E, lw=1.2, zorder=4))
    ax.add_patch(Ellipse((head[0] - 0.07 * u, head[1] + 0.09 * u), 0.14 * u, 0.2 * u, fc="#7a7a7a", ec="none", alpha=0.6, zorder=5))


def robot_front(ax, cx, y0, H):
    u = H / 3.0
    hip = (cx, y0 + 1.2 * u); sh = (cx, y0 + 2.1 * u)
    for sx in (-1, 1):
        cap(ax, (cx + sx * 0.2 * u, hip[1]), (cx + sx * 0.24 * u, y0 + 0.62 * u), 0.2 * u, ROB, ROB_E)
        cap(ax, (cx + sx * 0.24 * u, y0 + 0.62 * u), (cx + sx * 0.26 * u, y0 + 0.12 * u), 0.17 * u, ROB, ROB_E)
        joint(ax, (cx + sx * 0.24 * u, y0 + 0.62 * u), 0.09 * u, "#9aa3ad", ROB_E)
        ax.add_patch(FancyBboxPatch((cx + sx * 0.26 * u - 0.19 * u, y0 - 0.02 * u), 0.38 * u, 0.11 * u, boxstyle="round,pad=0,rounding_size=0.04", fc=ROB, ec=ROB_E, lw=1.0, zorder=3))
        cap(ax, (cx + sx * 0.5 * u, sh[1] - 0.05 * u), (cx + sx * 0.7 * u, y0 + 1.5 * u), 0.15 * u, ROB, ROB_E)
        cap(ax, (cx + sx * 0.7 * u, y0 + 1.5 * u), (cx + sx * 0.76 * u, y0 + 0.98 * u), 0.13 * u, ROB, ROB_E)
        joint(ax, (cx + sx * 0.5 * u, sh[1] - 0.05 * u), 0.1 * u, "#9aa3ad", ROB_E)
        joint(ax, (cx + sx * 0.7 * u, y0 + 1.5 * u), 0.075 * u, "#9aa3ad", ROB_E)
    ax.add_patch(FancyBboxPatch((cx - 0.34 * u, y0 + 1.15 * u), 0.68 * u, 1.0 * u, boxstyle="round,pad=0,rounding_size=0.1", fc=ROB, ec=ROB_E, lw=1.3, zorder=4))
    ax.add_patch(Circle((cx, y0 + 1.7 * u), 0.07 * u, fc="#5fd0ff", ec="none", zorder=5))
    cap(ax, (cx, y0 + 2.15 * u), (cx, y0 + 2.35 * u), 0.14 * u, ROB, ROB_E)
    ax.add_patch(FancyBboxPatch((cx - 0.3 * u, y0 + 2.3 * u), 0.6 * u, 0.5 * u, boxstyle="round,pad=0,rounding_size=0.12", fc=ROB, ec=ROB_E, lw=1.3, zorder=4))
    ax.add_patch(FancyBboxPatch((cx - 0.22 * u, y0 + 2.46 * u), 0.44 * u, 0.13 * u, boxstyle="round,pad=0,rounding_size=0.05", fc="#cfeaf7", ec="none", zorder=5))


def brain_outline(cx, cy, w):
    pts = np.array([[0.00, 0.45], [0.05, 0.62], [0.17, 0.78], [0.38, 0.9], [0.62, 0.92], [0.82, 0.84], [0.95, 0.68], [1.0, 0.5],
                    [0.97, 0.36], [0.86, 0.29], [0.74, 0.24], [0.62, 0.13], [0.46, 0.1], [0.32, 0.17], [0.22, 0.27], [0.1, 0.32]])
    tck, _ = splprep(pts.T, s=0, per=True)
    bx, by = splev(np.linspace(0, 1, 300), tck)
    return np.column_stack([cx + (bx - 0.5) * w, cy + (by - 0.5) * w])


def brain_network(ax, cx, cy, w, seed=4):
    rng = np.random.default_rng(seed)
    out = brain_outline(cx, cy, w)
    path = Path(out)
    ax.add_patch(Polygon(out, closed=True, fc="#f1f0ec", ec="#8a8984", lw=1.3, zorder=1))
    cand = np.c_[rng.uniform(out[:, 0].min(), out[:, 0].max(), 4000), rng.uniform(out[:, 1].min(), out[:, 1].max(), 4000)]
    pts = cand[path.contains_points(cand, radius=-0.02 * w)][:130]
    from scipy.spatial import cKDTree
    _, nn = cKDTree(pts).query(pts, k=4)
    segs = [(pts[i], pts[j]) for i in range(len(pts)) for j in nn[i, 1:4]]
    ax.add_collection(LineCollection(segs, colors="#8a8984", linewidths=0.5, alpha=0.5, zorder=2))
    col = [ORANGE if rng.random() < 0.2 else (GREY if rng.random() < 0.2 else BLUE) for _ in range(len(pts))]
    scatter_by_color(ax, pts[:, 0], pts[:, 1], 14, col, edgecolors="white", linewidths=0.4, zorder=3)
    return out


def subgraph_net(ax, cx, cy, r, name="A_300_s0"):
    ids, sign, known, edges = load_sub(name)
    n = len(ids)
    D = nx.DiGraph(); D.add_nodes_from(range(n)); D.add_edges_from([(a, b) for a, b, _ in edges])
    U = D.to_undirected()
    pos = nx.spring_layout(U, seed=3, k=2.2 / math.sqrt(n), iterations=400)
    P = np.array([pos[i] for i in range(n)]); P = P - np.median(P, axis=0)
    rad = np.linalg.norm(P, axis=1); P = P / np.percentile(rad, 85) * 0.82
    rad = np.linalg.norm(P, axis=1); P = np.where(rad[:, None] > 0.95, P / rad[:, None] * 0.95, P)
    arr = P * r + np.array([cx, cy])
    ax.add_collection(LineCollection([(arr[a], arr[b]) for a, b in D.edges], colors="#8a8984", linewidths=0.3, alpha=0.4, zorder=2))
    col = [(BLUE if sg > 0 else ORANGE) if k else GREY for sg, k in zip(sign, known)]
    deg = dict(U.degree())
    scatter_by_color(ax, arr[:, 0], arr[:, 1], [4 + 1.2 * deg[i] for i in range(n)], col, edgecolors="white", linewidths=0.3, zorder=3)


def fig_overview():
    fig, ax = canvas(13, 6.0, 13, 6.0)
    mx, rx = 2.7, 10.3
    mannequin(ax, mx, 0.55, 2.9)
    robot_front(ax, rx, 0.55, 2.9)
    out = brain_network(ax, mx, 4.65, 2.3)
    # highlighted patch in the brain and zoom lines to the subgraph
    hp = np.array([mx + 0.45, 4.35])
    ax.add_patch(Rectangle(hp - 0.1, 0.2, 0.2, fc="none", ec="#6250d6", lw=1.8, zorder=6))
    subgraph_net(ax, rx, 4.65, 0.95)
    ax.add_patch(Circle((rx, 4.65), 1.05, fc="none", ec="#6250d6", lw=1.8, zorder=1))
    ax.plot([hp[0] + 0.1, rx - 1.05 * 0.97], [hp[1] + 0.1, 4.65 + 0.28], color="#6250d6", ls="--", lw=1.0, zorder=1)
    ax.plot([hp[0] + 0.1, rx - 1.05 * 0.97], [hp[1] - 0.1, 4.65 - 0.28], color="#6250d6", ls="--", lw=1.0, zorder=1)
    ax.text((mx + rx) / 2, 5.45, "one connected region of 300 neurons", ha="center", fontsize=8.6, color="#6250d6", style="italic")
    # main arrow
    arrow(ax, (mx + 1.55, 1.9), (rx - 1.5, 1.9), lw=3.2)
    ax.text((mx + rx) / 2, 2.42, "its wiring becomes the recurrent core\nof the robot's controller", ha="center", fontsize=10, color=INK, fontweight="bold", linespacing=1.4)
    ax.text((mx + rx) / 2, 1.28, "fixed reservoir + trained linear readout\ncorrecting an open-loop gait", ha="center", fontsize=8.8, color=MUTE, linespacing=1.4)
    ax.text(mx, 0.16, "human cortex wiring (H01 connectome)", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(rx, 0.16, "simulated humanoid robot", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(mx, 5.88, "brain wiring (illustration)", ha="center", fontsize=8, color=MUTE, va="top", style="italic") if False else None
    # legend
    for k, (m, c, t) in enumerate([("o", BLUE, "excitatory"), ("^", ORANGE, "inhibitory"), ("D", GREY, "sign unknown")]):
        ax.scatter([5.15 + k * 1.15], [0.72 - 0.0], s=30 * (1.35 if m == "^" else 1), c=c, marker=m, edgecolors="white", linewidths=0.5, zorder=5)
        ax.text(5.28 + k * 1.15, 0.72, t, fontsize=7.4, color=MUTE, va="center")
    fig.savefig(f"{OUT}/fig_overview.png"); plt.close(fig)


def robot_side(ax, x, y0, H, alpha=1.0, push=True):
    fc, ec = ROB, ROB_E
    h = lambda px, py: (x + px * H, y0 + py * H)
    kw = dict(fc=fc, ec=ec)
    def c(p, q, w, z=3):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=ec, lw=(w * H + 0.07) * 72, solid_capstyle="round", zorder=z, alpha=alpha)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=fc, lw=w * H * 72, solid_capstyle="round", zorder=z + 0.1, alpha=alpha)
    # legs (back leg first)
    c(h(0.0, 0.42), h(-0.07, 0.21), 0.1); c(h(-0.07, 0.21), h(-0.13, 0.04), 0.085)
    ax.add_patch(FancyBboxPatch(h(-0.2, 0.0), 0.16 * H, 0.04 * H, boxstyle="round,pad=0,rounding_size=0.02", fc=fc, ec=ec, lw=1, zorder=3, alpha=alpha))
    c(h(0.0, 0.42), h(0.13, 0.22), 0.1, 4); c(h(0.13, 0.22), h(0.09, 0.04), 0.085, 4)
    ax.add_patch(FancyBboxPatch(h(0.03, 0.0), 0.16 * H, 0.04 * H, boxstyle="round,pad=0,rounding_size=0.02", fc=fc, ec=ec, lw=1, zorder=4, alpha=alpha))
    # torso leaning forward
    c(h(0.0, 0.44), h(0.1, 0.74), 0.27, 5)
    # head
    ax.add_patch(FancyBboxPatch(h(0.075, 0.79), 0.19 * H, 0.17 * H, boxstyle="round,pad=0,rounding_size=0.05", fc=fc, ec=ec, lw=1.2, zorder=6, alpha=alpha))
    ax.add_patch(FancyBboxPatch(h(0.18, 0.865), 0.085 * H, 0.04 * H, boxstyle="round,pad=0,rounding_size=0.015", fc="#cfeaf7", ec="none", zorder=7, alpha=alpha))
    # arm reaching forward
    c(h(0.12, 0.7), h(0.27, 0.58), 0.075, 7); c(h(0.27, 0.58), h(0.4, 0.52), 0.065, 7)
    ax.add_patch(Circle(h(0.12, 0.7), 0.04 * H, fc="#9aa3ad", ec=ec, lw=1, zorder=8, alpha=alpha))
    return h(0.4, 0.52)


def fig_task():
    S = 4.0                                    # inches per metre
    Hr = 0.339 * S
    fig = plt.figure(figsize=(10.5, 4.3), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10.5); ax.set_ylim(0, 4.3); ax.axis("off")
    g = 0.55
    ax.plot([0.2, 10.3], [g, g], color="#999999", lw=1.4, zorder=1)
    ax.add_patch(Rectangle((0.2, g - 0.12), 10.1, 0.12, fc="#eeeeee", ec="none", zorder=0))
    x0 = 0.95
    xf = x0 + 0.5 * S                          # near face of the cube
    # box, goal, ghost
    ax.add_patch(Rectangle((xf, g), 0.6 * S, 0.6 * S, fc="#e4cba3", ec=STROKE, lw=1.5, zorder=2))
    ax.text(xf + 0.12, g + 0.6 * S - 0.12, "cube\n0.6 m, 0.7 kg", ha="left", va="top", fontsize=9.5, zorder=3, linespacing=1.4)
    xg = xf + 0.4 * S
    ax.add_patch(Rectangle((xg, g), 0.6 * S, 0.6 * S, fc="none", ec=STROKE, lw=1.0, ls="--", zorder=1))
    ax.plot([xg, xg], [g, g + 0.6 * S + 0.25], color=GREEN, lw=2.2, zorder=4)
    ax.text(xg + 0.06, g + 0.6 * S + 0.2, "goal: near face\nmoved 0.40 m", fontsize=8.4, color=GREEN, va="bottom", linespacing=1.3)
    # robots
    robot_side(ax, x0, g, Hr, alpha=0.3)
    tip = robot_side(ax, xf - 0.4 * Hr, g, Hr, alpha=1.0)
    arrow(ax, (x0 + 0.3, g + Hr + 0.25), (xf - 0.4 * Hr - 0.05, g + Hr + 0.25), "walks", off=(0, 0.17), fs=8.5, color=INK, lw=1.4)
    ax.text(x0, g + Hr + 0.05, "start", fontsize=7.8, color=MUTE, ha="center")
    ax.text(1.65, g + Hr + 0.75, "DARwIn-OP (abstract drawing)", fontsize=8.2, color=MUTE, ha="center")
    ax.annotate("", xy=(xf, g - 0.25), xytext=(x0, g - 0.25), arrowprops=dict(arrowstyle="<->", color=MUTE, lw=1.0))
    ax.text((x0 + xf) / 2, g - 0.33, "0.5 m from the robot's root to the near face", fontsize=8.2, color=MUTE, ha="center", va="top")
    # top-view inset (offset)
    ix, iy, sc = 7.75, 2.75, 1.15
    ax.add_patch(Rectangle((ix - 0.05, iy - 1.05), 2.75, 2.1, fc="#fafafa", ec="#dddddd", lw=1.0, zorder=0))
    ax.text(ix + 0.05, iy + 0.82, "top view", fontsize=8.2, fontweight="bold", color=INK, va="center")
    bxl, byc = ix + 1.25, iy - 0.05
    for dy in (-0.15, 0.15):
        ax.add_patch(Rectangle((bxl, byc + dy * sc - 0.6 * sc / 2), 0.6 * sc, 0.6 * sc, fc="none", ec=BLUE, lw=0.9, ls=":", zorder=1))
    ax.add_patch(Rectangle((bxl, byc - 0.3 * sc), 0.6 * sc, 0.6 * sc, fc="#e4cba3", ec=STROKE, lw=1.0, zorder=2))
    ax.add_patch(Rectangle((ix + 0.25, byc - 0.07), 0.14, 0.14, fc="#555555", ec="none", zorder=3))
    arrow(ax, (ix + 0.42, byc), (bxl - 0.06, byc), color=INK, lw=1.0)
    ax.text(ix + 1.2, iy - 0.88, "box centre shifted sideways\nby up to $\\pm$0.15 m", fontsize=7.8, color=BLUE, ha="center", va="center", linespacing=1.3)
    fig.text(0.02, 0.955, "Pushing task: the robot walks to the cube and pushes it 0.40 m", fontsize=10.5, fontweight="bold", va="center")
    fig.savefig(f"{OUT}/fig_task.png"); plt.close(fig)


# ================================================================== Figure 1 (heads and necks)
def human_head(ax, cx, y0, S=1.0):
    """slender faceless mannequin head, long neck, sloping shoulders; y0 = bottom of the shoulders"""
    fc, ec = MAN, MAN_E
    sh_y = y0 + 0.55 * S
    # shoulders / upper chest (sloping)
    sx = np.array([-1.55, -1.0, -0.45, 0, 0.45, 1.0, 1.55]) * S
    sy = np.array([0.0, 0.52, 0.66, 0.74, 0.66, 0.52, 0.0]) * S
    body = np.column_stack([np.r_[cx + sx, cx + 1.55 * S, cx - 1.55 * S], np.r_[y0 + sy, y0, y0]])
    ax.add_patch(Polygon(body, closed=True, fc=fc, ec=ec, lw=1.3, zorder=3))
    ax.plot([cx - 0.5 * S, cx, cx + 0.5 * S], [y0 + 0.38 * S, y0 + 0.3 * S, y0 + 0.38 * S], color="#7c7c7c", lw=0.9, zorder=4)
    # neck
    nk = np.array([[-0.2, 0.62], [-0.17, 1.38], [0.17, 1.38], [0.2, 0.62]]) * S
    ax.add_patch(Polygon(nk + [cx, y0], closed=True, fc=fc, ec=ec, lw=1.3, zorder=2))
    # head: long egg, narrow chin
    t = np.linspace(0, 2 * np.pi, 200)
    a, b = 0.40 * S, 0.80 * S
    hx = a * np.sin(t) * (1 + 0.16 * np.cos(t))
    hy = b * np.cos(t) * 1.0
    hc = (cx, y0 + 1.38 * S + b * 0.82)
    ax.add_patch(Polygon(np.column_stack([hc[0] + hx, hc[1] + hy]), closed=True, fc=fc, ec=ec, lw=1.4, zorder=4))
    ax.add_patch(Ellipse((hc[0] - 0.14 * S, hc[1] + 0.3 * S), 0.16 * S, 0.34 * S, fc="#7c7c7c", ec="none", alpha=0.55, zorder=5))
    return hc, b


def robot_head(ax, cx, y0, S=1.0):
    fc, ec = ROB, ROB_E
    ax.add_patch(FancyBboxPatch((cx - 1.1 * S, y0), 2.2 * S, 0.6 * S, boxstyle="round,pad=0,rounding_size=0.28", fc=fc, ec=ec, lw=1.3, zorder=3))
    for sgn in (-1, 1):
        joint(ax, (cx + sgn * 0.78 * S, y0 + 0.3 * S), 0.16 * S, "#9aa3ad", ec)
    ax.add_patch(Circle((cx, y0 + 0.3 * S), 0.07 * S, fc="#5fd0ff", ec="none", zorder=5))
    ax.add_patch(FancyBboxPatch((cx - 0.17 * S, y0 + 0.55 * S), 0.34 * S, 0.8 * S, boxstyle="round,pad=0,rounding_size=0.06", fc="#69727d", ec=ec, lw=1.2, zorder=2))
    for k in range(3):
        ax.plot([cx - 0.17 * S, cx + 0.17 * S], [y0 + (0.75 + 0.18 * k) * S] * 2, color=ec, lw=0.7, zorder=3)
    hy = y0 + 1.3 * S
    ax.add_patch(FancyBboxPatch((cx - 0.62 * S, hy), 1.24 * S, 1.1 * S, boxstyle="round,pad=0,rounding_size=0.28", fc=fc, ec=ec, lw=1.4, zorder=4))
    ax.add_patch(FancyBboxPatch((cx - 0.46 * S, hy + 0.46 * S), 0.92 * S, 0.26 * S, boxstyle="round,pad=0,rounding_size=0.1", fc="#cfeaf7", ec="none", zorder=5))
    ax.add_patch(Ellipse((cx - 0.3 * S, hy + 0.92 * S), 0.2 * S, 0.12 * S, fc="#7d8794", ec="none", alpha=0.7, zorder=5))
    return (cx, hy + 0.55 * S)


def fig_overview():
    fig, ax = canvas(13, 6.4, 13, 6.4)
    mx, rx = 2.7, 10.3
    y0 = 0.45
    hc, b = human_head(ax, mx, y0, 1.0)
    rc = robot_head(ax, rx, y0, 1.0)
    brain_cy = 4.95
    brain_network(ax, mx, brain_cy, 2.3)
    hp = np.array([mx + 0.45, brain_cy - 0.3])
    ax.add_patch(Rectangle(hp - 0.1, 0.2, 0.2, fc="none", ec="#6250d6", lw=1.8, zorder=6))
    subgraph_net(ax, rx, brain_cy, 0.95)
    ax.add_patch(Circle((rx, brain_cy), 1.05, fc="none", ec="#6250d6", lw=1.8, zorder=1))
    ax.plot([hp[0] + 0.1, rx - 1.05 * 0.97], [hp[1] + 0.1, brain_cy + 0.28], color="#6250d6", ls="--", lw=1.0, zorder=1)
    ax.plot([hp[0] + 0.1, rx - 1.05 * 0.97], [hp[1] - 0.1, brain_cy - 0.28], color="#6250d6", ls="--", lw=1.0, zorder=1)
    ax.text((mx + rx) / 2, brain_cy + 0.78, "one connected region of 300 neurons", ha="center", fontsize=8.6, color="#6250d6", style="italic")
    arrow(ax, (mx + 1.7, 2.1), (rx - 1.7, 2.1), lw=3.2)
    ax.text((mx + rx) / 2, 1.45, "fixed reservoir + trained linear readout\ncorrecting an open-loop gait", ha="center", fontsize=8.8, color=MUTE, linespacing=1.4)
    ax.text(mx, 0.12, "human cortex wiring (H01 connectome)", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(rx, 0.12, "simulated humanoid robot", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    for k, (m, c, t) in enumerate([("o", BLUE, "excitatory"), ("^", ORANGE, "inhibitory"), ("D", GREY, "sign unknown")]):
        ax.scatter([5.15 + k * 1.15], [2.85], s=30 * (1.35 if m == "^" else 1), c=c, marker=m, edgecolors="white", linewidths=0.5, zorder=5)
        ax.text(5.28 + k * 1.15, 2.85, t, fontsize=7.4, color=MUTE, va="center")
    fig.savefig(f"{OUT}/fig_overview.png"); plt.close(fig)


# ---- outline-style human (after anatomical outline drawings) and robot with two ring eyes
def human_head(ax, cx, y0, S=1.0):
    from scipy.interpolate import PchipInterpolator
    LW, INK2 = 2.6, "#111111"
    def L(xs, ys, **k):
        ax.plot(cx + np.array(xs) * S, y0 + np.array(ys) * S, color=INK2, lw=k.pop("lw", LW), solid_capstyle="round", solid_joinstyle="round", zorder=k.pop("z", 4), **k)
    def smooth(pts, n=60):
        p = np.array(pts); t = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
        tt = np.linspace(0, t[-1], n); return PchipInterpolator(t, p[:, 0])(tt), PchipInterpolator(t, p[:, 1])(tt)
    # silhouette: left side from bottom of left arm up over shoulder, neck, head, and mirrored
    left = [(-1.62, 0.0), (-1.64, 0.42), (-1.55, 0.72), (-1.28, 0.9), (-0.9, 0.98), (-0.55, 1.04), (-0.3, 1.14), (-0.2, 1.3), (-0.19, 1.6)]
    lx, ly = smooth(left)
    rx_, ry_ = -lx, ly
    # head
    t = np.linspace(0, 2 * np.pi, 240); a, b = 0.40, 0.78
    hx = a * np.sin(t) * (1 + 0.12 * np.cos(t)); hy = b * np.cos(t)
    hc = (0, 1.52 + b * 0.82)
    fillx = np.r_[lx, rx_[::-1]]; filly = np.r_[ly, ry_[::-1]]
    ax.fill(cx + fillx * S, y0 + filly * S, color="white", zorder=2)
    ax.fill(cx + (hc[0] + hx) * S, y0 + (hc[1] + hy) * S, color="white", zorder=3)
    L(lx, ly); L(rx_, ry_)
    L([-1.62, 1.62], [0, 0], lw=LW * 0.0001, alpha=0)
    L(hc[0] + hx, hc[1] + hy)
    # ears
    for sg in (-1, 1):
        e = np.linspace(-np.pi / 2, np.pi / 2, 30)
        L(sg * (0.385 + 0.07 * np.cos(e)), hc[1] + 0.02 + 0.16 * np.sin(e) * 1.0, lw=LW * 0.8, z=3)
    # neck-to-head join and inner lines (collarbones, chest, sternal notch)
    L(*smooth([(-0.28, 1.12), (-0.55, 1.08), (-0.9, 1.0)]), lw=LW * 0.55)
    L(*smooth([(0.28, 1.12), (0.55, 1.08), (0.9, 1.0)]), lw=LW * 0.55)
    L(*smooth([(-1.12, 0.72), (-0.95, 0.52), (-0.6, 0.38), (-0.2, 0.34)]), lw=LW * 0.6)
    L(*smooth([(1.12, 0.72), (0.95, 0.52), (0.6, 0.38), (0.2, 0.34)]), lw=LW * 0.6)
    L(*smooth([(-1.2, 0.95), (-1.12, 0.72), (-1.15, 0.4)]), lw=LW * 0.55)
    L(*smooth([(1.2, 0.95), (1.12, 0.72), (1.15, 0.4)]), lw=LW * 0.55)
    L([-0.04, 0.04], [0.97, 0.97], lw=LW * 0.4)
    return (cx, y0 + hc[1] * S), b


def robot_head(ax, cx, y0, S=1.0):
    fc, ec = ROB, ROB_E
    ax.add_patch(FancyBboxPatch((cx - 1.1 * S, y0), 2.2 * S, 0.6 * S, boxstyle="round,pad=0,rounding_size=0.28", fc=fc, ec=ec, lw=1.3, zorder=3))
    for sgn in (-1, 1):
        joint(ax, (cx + sgn * 0.78 * S, y0 + 0.3 * S), 0.16 * S, "#9aa3ad", ec)
    ax.add_patch(FancyBboxPatch((cx - 0.17 * S, y0 + 0.55 * S), 0.34 * S, 0.8 * S, boxstyle="round,pad=0,rounding_size=0.06", fc="#69727d", ec=ec, lw=1.2, zorder=2))
    for k in range(3):
        ax.plot([cx - 0.17 * S, cx + 0.17 * S], [y0 + (0.75 + 0.18 * k) * S] * 2, color=ec, lw=0.7, zorder=3)
    hy = y0 + 1.3 * S
    ax.add_patch(FancyBboxPatch((cx - 0.7 * S, hy), 1.4 * S, 1.3 * S, boxstyle="round,pad=0,rounding_size=0.34", fc=fc, ec=ec, lw=1.4, zorder=4))
    for sgn in (-1, 1):                      # two round ring eyes
        ex, ey = cx + sgn * 0.3 * S, hy + 0.62 * S
        ax.add_patch(Circle((ex, ey), 0.23 * S, fc="#2a2f6e", ec="#7f8dff", lw=3.2, zorder=5))
        ax.add_patch(Circle((ex, ey), 0.09 * S, fc="#0d0f2b", ec="none", zorder=6))
    ax.add_patch(Circle((cx, hy + 0.2 * S), 0.045 * S, fc="#111111", ec="none", zorder=5))
    ax.add_patch(Ellipse((cx - 0.38 * S, hy + 1.12 * S), 0.24 * S, 0.12 * S, fc="#7d8794", ec="none", alpha=0.7, zorder=5))
    return (cx, hy + 0.65 * S)


# ---- outline-style human (line drawing like an anatomy outline) and robot with two round eyes
LINE = "#1c1c1c"


def human_head(ax, cx, y0, S=1.0):
    lw = 2.0
    # torso/shoulders outline: smooth trapezius into sloped shoulders and upper arms
    X = np.array([-1.55, -1.5, -1.35, -1.05, -0.7, -0.38, -0.2, 0.2, 0.38, 0.7, 1.05, 1.35, 1.5, 1.55])
    Y = np.array([0.0, 0.35, 0.55, 0.66, 0.74, 0.86, 1.05, 1.05, 0.86, 0.74, 0.66, 0.55, 0.35, 0.0])
    t = np.linspace(0, 1, len(X)); tt = np.linspace(0, 1, 200)
    from scipy.interpolate import PchipInterpolator
    xs = PchipInterpolator(t, X)(tt); ys = PchipInterpolator(t, Y)(tt)
    ax.fill(np.r_[cx + xs * S, cx + 1.55 * S, cx - 1.55 * S], np.r_[y0 + ys * S, y0, y0], fc="white", ec="none", zorder=2)
    for m in (xs < -0.19, xs > 0.19):
        ax.plot(cx + xs[m] * S, y0 + ys[m] * S, color=LINE, lw=lw, zorder=3, solid_capstyle="round")
    ax.plot([cx - 1.55 * S, cx - 1.55 * S], [y0, y0 + 0.02 * S], color=LINE, lw=lw)
    # collarbones and chest lines
    for sg in (-1, 1):
        ax.plot(cx + sg * np.array([0.22, 0.5, 0.9]) * S, y0 + np.array([0.86, 0.76, 0.72]) * S, color=LINE, lw=1.3, zorder=3)
        ax.plot(cx + sg * np.array([0.12, 0.5, 0.95, 1.2]) * S, y0 + np.array([0.28, 0.2, 0.3, 0.5]) * S, color=LINE, lw=1.3, zorder=3)
    ax.plot([cx, cx], [y0, y0 + 0.5 * S], color=LINE, lw=1.0, alpha=0.0)
    # neck
    ax.fill([cx - 0.19 * S, cx - 0.17 * S, cx + 0.17 * S, cx + 0.19 * S], [y0 + 1.02 * S, y0 + 1.55 * S, y0 + 1.55 * S, y0 + 1.02 * S], fc="white", ec="none", zorder=2)
    ax.plot([cx - 0.19 * S, cx - 0.16 * S], [y0 + 1.02 * S, y0 + 1.58 * S], color=LINE, lw=lw, zorder=3)
    ax.plot([cx + 0.19 * S, cx + 0.16 * S], [y0 + 1.02 * S, y0 + 1.58 * S], color=LINE, lw=lw, zorder=3)
    # head with ears
    a, b = 0.40 * S, 0.74 * S
    th = np.linspace(0, 2 * np.pi, 200)
    hx = a * np.sin(th) * (1 + 0.14 * np.cos(th)); hy = b * np.cos(th)
    hc = (cx, y0 + 1.55 * S + b * 0.9)
    for sg in (-1, 1):
        ax.add_patch(Ellipse((hc[0] + sg * 0.43 * S, hc[1] - 0.05 * S), 0.12 * S, 0.3 * S, fc="white", ec=LINE, lw=1.6, zorder=3))
    ax.add_patch(Polygon(np.column_stack([hc[0] + hx, hc[1] + hy]), closed=True, fc="white", ec=LINE, lw=lw, zorder=4))
    return hc, b


def robot_head(ax, cx, y0, S=1.0):
    fc, ec = ROB, ROB_E
    ax.add_patch(FancyBboxPatch((cx - 1.1 * S, y0), 2.2 * S, 0.6 * S, boxstyle="round,pad=0,rounding_size=0.28", fc=fc, ec=ec, lw=1.3, zorder=3))
    for sgn in (-1, 1):
        joint(ax, (cx + sgn * 0.78 * S, y0 + 0.3 * S), 0.16 * S, "#9aa3ad", ec)
    ax.add_patch(FancyBboxPatch((cx - 0.17 * S, y0 + 0.55 * S), 0.34 * S, 0.8 * S, boxstyle="round,pad=0,rounding_size=0.06", fc="#69727d", ec=ec, lw=1.2, zorder=2))
    for k in range(3):
        ax.plot([cx - 0.17 * S, cx + 0.17 * S], [y0 + (0.75 + 0.18 * k) * S] * 2, color=ec, lw=0.7, zorder=3)
    hy = y0 + 1.3 * S
    ax.add_patch(FancyBboxPatch((cx - 0.7 * S, hy), 1.4 * S, 1.25 * S, boxstyle="round,pad=0,rounding_size=0.3", fc=fc, ec=ec, lw=1.4, zorder=4))
    for sgn in (-1, 1):
        ex, ey = cx + sgn * 0.33 * S, hy + 0.62 * S
        ax.add_patch(Circle((ex, ey), 0.24 * S, fc="#6d8cff", ec="#2f46b8", lw=1.3, zorder=5))
        ax.add_patch(Circle((ex, ey), 0.14 * S, fc=fc, ec="none", zorder=6))
    ax.add_patch(Circle((cx, hy + 0.2 * S), 0.05 * S, fc="#10151b", ec="none", zorder=5))
    return (cx, hy + 0.6 * S)


if __name__ == "__main__":
    fig_overview(); print("EI groups", fig_ei()); st = fig_subgraphs()
    for k, v in st.items():
        print(k, v["E"], v["rows"], round(v["deg"], 1), f"{v['recip']:.1%}", v["inh"], v["known"], v["unk"])
    fig_policy(); fig_task(); fig_training()

r"""Per-neuron outgoing synapse types (used to give each neuron one excitatory/inhibitory sign).

Usage (PowerShell):
    python h01_cell_signs.py
    python h01_cell_signs.py C:\path\to\h01_edges_out C:\path\to\h01_graph_out_v2\cells.tsv

Defaults: h01_edges_out and h01_graph_out_v2/cells.tsv next to this script
(it looks next to the script first, then in the folder you pass).
Neuron rule (from the version 2 summary): rows labelled SOMA >= 5 and rows as DENDRITE receiver >= 20.
Reads every edge file again (about 15 minutes), writes h01_graph_out_v2\neuron_signs.tsv:
    id, out_rows, out_type1, out_type2, good_rows, good_type1, good_type2
(out_* = all rows where this neuron is the sender; good_* = only AXON -> DENDRITE/SOMA/AIS rows)
Needs only the standard library.
"""
import glob
import gzip
import os
import sys
import time
from collections import defaultdict

here = os.path.dirname(os.path.abspath(__file__))
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "h01_edges_out")
cells_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "h01_graph_out_v2", "cells.tsv")
out_path = os.path.join(os.path.dirname(cells_path), "neuron_signs.tsv")

N = set()
with open(cells_path) as f:
    hdr = f.readline().split()
    ix = {h: i for i, h in enumerate(hdr)}
    for line in f:
        p = line.split()
        if int(p[ix["pre_SOMA"]]) + int(p[ix["post_SOMA"]]) >= 5 and int(p[ix["post_DENDRITE"]]) >= 20:
            N.add(p[0])
print(f"neurons by rule: {len(N)}", flush=True)

edge_files = sorted(glob.glob(os.path.join(root, "edges_*.tsv.gz")))
GOOD_POST = ("DENDRITE", "SOMA", "AIS")
acc = defaultdict(lambda: [0, 0, 0, 0, 0, 0])  # rows, t1, t2, good rows, good t1, good t2
t0 = time.time()
for i, p in enumerate(edge_files):
    with gzip.open(p, "rt") as f:
        for line in f:
            a, b, t, pc, qc = line.rstrip("\n").split("\t")
            if a not in N:
                continue
            r = acc[a]
            r[0] += 1
            r[1 if t == "1" else 2] += 1
            if pc == "AXON" and qc in GOOD_POST:
                r[3] += 1
                r[4 if t == "1" else 5] += 1
    if (i + 1) % 20 == 0 or i + 1 == len(edge_files):
        print(f"  {i + 1}/{len(edge_files)} files, {time.time() - t0:.0f} s", flush=True)

with open(out_path, "w") as f:
    f.write("id\tout_rows\tout_type1\tout_type2\tgood_rows\tgood_type1\tgood_type2\n")
    for c in sorted(N, key=int):
        f.write(c + "\t" + "\t".join(map(str, acc[c] if c in acc else [0] * 6)) + "\n")

# a small summary to paste back
has = [c for c in N if c in acc and acc[c][0] >= 10]
fr = [acc[c][1] / acc[c][0] for c in has]
hist = [sum(1 for x in fr if min(9, int(x * 10)) == k) for k in range(10)]
good = [c for c in N if c in acc and acc[c][3] >= 10]
frg = [acc[c][4] / acc[c][3] for c in good]
histg = [sum(1 for x in frg if min(9, int(x * 10)) == k) for k in range(10)]
lines = [
    f"neurons: {len(N)}",
    f"neurons with >=10 outgoing rows: {len(has)}; share of type-1 rows per neuron, in tenths: {hist}",
    f"neurons with >=10 good outgoing rows: {len(good)}; share of type-1 rows per neuron, in tenths: {histg}",
    f"neurons with no outgoing rows at all: {sum(1 for c in N if c not in acc)}",
]
text = "\n".join(lines)
print("\n" + text)
with open(os.path.join(os.path.dirname(cells_path), "neuron_signs_summary.txt"), "w") as f:
    f.write(text + "\n")
print("\nSaved to", os.path.dirname(cells_path))

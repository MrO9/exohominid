r"""Build the soma-cell graph from the edge files made by h01_edges.py (no downloading).

Usage (PowerShell):
    python h01_build_graph.py
    python h01_build_graph.py C:\path\to\h01_edges_out      (if the folder is elsewhere)

Reads h01_edges_out\somas_*.txt and edges_*.tsv.gz (about 160 million rows; allow 15 to 40 minutes).
Writes into h01_graph_out\ (next to the folder it read):
    soma_ids.txt        every neuron_id that has class_label SOMA in any shard
    graph_edges.tsv     pre, post, n_rows, n_type1, n_type2   (only edges with BOTH ends in soma_ids)
    summary.txt         the numbers to paste back to me (small)
Needs only the standard library.
"""
import gzip
import glob
import os
import sys
import time
from collections import Counter, defaultdict

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "h01_edges_out")
out_dir = os.path.join(os.path.dirname(root.rstrip("\\/")), "h01_graph_out")
os.makedirs(out_dir, exist_ok=True)

# the 104 proofread cells (from the SWC zip), to check how they sit in the graph
PROOF = set(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "proofread_104_ids.txt")).read().split()) \
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "proofread_104_ids.txt")) else set()

# step 1: union of soma ids
soma_files = sorted(glob.glob(os.path.join(root, "somas_*.txt")))
edge_files = sorted(glob.glob(os.path.join(root, "edges_*.tsv.gz")))
S = set()
for p in soma_files:
    with open(p) as f:
        S.update(line.strip() for line in f if line.strip())
print(f"soma files: {len(soma_files)}, edge files: {len(edge_files)}, distinct soma ids: {len(S)}", flush=True)
with open(os.path.join(out_dir, "soma_ids.txt"), "w") as f:
    f.write("\n".join(sorted(S, key=int)) + "\n")

# step 2: one pass over every edge row
total = 0
membership = Counter()                 # (pre in S, post in S) -> rows
edge_rows = defaultdict(lambda: [0, 0, 0])   # (pre, post) -> [rows, type1, type2], both in S
out_type = defaultdict(lambda: [0, 0])       # sender in S -> [type1 rows, type2 rows], any receiver
class_pairs_SS = Counter()
t0 = time.time()
for i, p in enumerate(edge_files):
    with gzip.open(p, "rt") as f:
        for line in f:
            a, b, t, pc, qc = line.rstrip("\n").split("\t")
            total += 1
            ina, inb = a in S, b in S
            membership[(ina, inb)] += 1
            if ina:
                out_type[a][0 if t == "1" else 1] += 1
            if ina and inb:
                e = edge_rows[(a, b)]
                e[0] += 1
                e[1 if t == "1" else 2] += 1
                class_pairs_SS[(pc, qc)] += 1
    if (i + 1) % 10 == 0 or i + 1 == len(edge_files):
        print(f"  {i + 1}/{len(edge_files)} files, {total} rows, {time.time() - t0:.0f} s", flush=True)

with open(os.path.join(out_dir, "graph_edges.tsv"), "w") as f:
    f.write("pre\tpost\tn_rows\tn_type1\tn_type2\n")
    for (a, b), (n, t1, t2) in edge_rows.items():
        f.write(f"{a}\t{b}\t{n}\t{t1}\t{t2}\n")

# step 3: summary
def quant(vals, qs=(0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1)):
    vals = sorted(vals)
    return {q: vals[min(len(vals) - 1, int(q * len(vals)))] for q in qs} if vals else {}

outdeg, indeg, wout, win = Counter(), Counter(), Counter(), Counter()
for (a, b), (n, t1, t2) in edge_rows.items():
    outdeg[a] += 1; indeg[b] += 1; wout[a] += n; win[b] += n
L = []
L.append(f"edge files read: {len(edge_files)}; rows read: {total}")
L.append(f"distinct soma ids: {len(S)}")
L.append(f"rows by (sender in soma set, receiver in soma set): {dict(membership)}")
L.append(f"distinct soma->soma edges: {len(edge_rows)}; total synapse rows on them: {sum(v[0] for v in edge_rows.values())}")
nodes = set(outdeg) | set(indeg)
L.append(f"soma cells with at least one soma->soma edge: {len(nodes)}")
L.append(f"edge weight (synapse rows per edge) quantiles: {quant([v[0] for v in edge_rows.values()])}")
L.append(f"out-degree quantiles over cells with an out-edge: {quant(list(outdeg.values()))}")
L.append(f"in-degree quantiles over cells with an in-edge: {quant(list(indeg.values()))}")
L.append(f"top 10 out-degree: {outdeg.most_common(10)}")
L.append(f"top 10 in-degree: {indeg.most_common(10)}")
L.append(f"soma->soma class pairs (top 10): {class_pairs_SS.most_common(10)}")
# is row type a property of the sending cell? fraction of type-1 rows per sender with >= 20 rows
fr = [v[0] / (v[0] + v[1]) for v in out_type.values() if v[0] + v[1] >= 20]
hist = Counter(min(9, int(x * 10)) for x in fr)
L.append(f"senders in soma set with >=20 rows: {len(fr)}; share of type-1 rows per sender, in tenths (0=0-10%, 9=90-100%): "
         f"{[hist.get(k, 0) for k in range(10)]}")
if PROOF:
    inP = [c for c in PROOF if c in S]
    L.append(f"proofread cells that are in the soma set: {len(inP)} of {len(PROOF)}")
    pe = sum(1 for (a, b) in edge_rows if a in PROOF and b in PROOF)
    L.append(f"soma->soma edges between two proofread cells: {pe}")
    L.append(f"soma->soma edges touching a proofread cell: {sum(1 for (a, b) in edge_rows if a in PROOF or b in PROOF)}")
text = "\n".join(L)
print("\n" + text)
with open(os.path.join(out_dir, "summary.txt"), "w") as f:
    f.write(text + "\n")
print("\nSaved to", out_dir)

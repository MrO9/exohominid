r"""Build the soma-cell graph, version 2: keeps class information so we can filter noise.

Usage (PowerShell):
    python h01_build_graph_v2.py
    python h01_build_graph_v2.py C:\path\to\h01_edges_out    (if the folder is elsewhere)

Reads h01_edges_out\somas_*.txt and edges_*.tsv.gz (about 160 million rows; allow 15 to 25 minutes).
Writes into h01_graph_out_v2\ (next to the folder it read):
    cells.tsv           one row per soma-labelled id: how often it appears as sender/receiver, by class label
    graph_edges.tsv     soma->soma edges: counts of rows, types, and class-pair quality
    summary.txt         the small file to paste back to me
Does not touch the version 1 output. Needs only the standard library.

Row quality, from the (sender class, receiver class) pair:
    good  = sender AXON and receiver DENDRITE, SOMA or AIS
    unk   = anything else that involves an UNKNOWN label
    bad   = the rest (for example DENDRITE to DENDRITE, SOMA to SOMA, AXON to AXON)
"""
import glob
import gzip
import os
import sys
import time
from collections import Counter, defaultdict

here = os.path.dirname(os.path.abspath(__file__))
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "h01_edges_out")
out_dir = os.path.join(os.path.dirname(root.rstrip("\\/")), "h01_graph_out_v2")
os.makedirs(out_dir, exist_ok=True)

proof_file = os.path.join(here, "proofread_104_ids.txt")
PROOF = set(open(proof_file).read().split()) if os.path.exists(proof_file) else set()

CLASSES = ["AXON", "DENDRITE", "SOMA", "UNKNOWN", "AIS", "CILIA", "OTHER"]
CIDX = {c: i for i, c in enumerate(CLASSES)}
GOOD_POST = ("DENDRITE", "SOMA", "AIS")


def cidx(c):
    return CIDX.get(c, 6)


soma_files = sorted(glob.glob(os.path.join(root, "somas_*.txt")))
edge_files = sorted(glob.glob(os.path.join(root, "edges_*.tsv.gz")))
S = set()
for p in soma_files:
    with open(p) as f:
        S.update(line.strip() for line in f if line.strip())
print(f"soma files: {len(soma_files)}, edge files: {len(edge_files)}, soma-labelled ids: {len(S)}", flush=True)

total = 0
membership = Counter()
type_by_pair = Counter()                       # (type, pre class, post class) over ALL rows
as_pre = defaultdict(lambda: [0] * 7)          # id in S -> rows as sender, by sender class
as_post = defaultdict(lambda: [0] * 7)         # id in S -> rows as receiver, by receiver class
edges = defaultdict(lambda: [0] * 9)           # (a, b) both in S ->
#   [rows, type1, type2, good, unk, bad, good_type1, good_type2, 0]
out_good = defaultdict(lambda: [0, 0])         # sender in S -> [type1, type2] over its good rows (any receiver)
t0 = time.time()
for i, p in enumerate(edge_files):
    with gzip.open(p, "rt") as f:
        for line in f:
            a, b, t, pc, qc = line.rstrip("\n").split("\t")
            total += 1
            type_by_pair[(t, pc, qc)] += 1
            ina, inb = a in S, b in S
            membership[(ina, inb)] += 1
            good = pc == "AXON" and qc in GOOD_POST
            if ina:
                as_pre[a][cidx(pc)] += 1
                if good:
                    out_good[a][0 if t == "1" else 1] += 1
            if inb:
                as_post[b][cidx(qc)] += 1
            if ina and inb:
                e = edges[(a, b)]
                e[0] += 1
                e[1 if t == "1" else 2] += 1
                if good:
                    e[3] += 1
                    e[6 if t == "1" else 7] += 1
                elif pc == "UNKNOWN" or qc == "UNKNOWN":
                    e[4] += 1
                else:
                    e[5] += 1
    if (i + 1) % 10 == 0 or i + 1 == len(edge_files):
        print(f"  {i + 1}/{len(edge_files)} files, {total} rows, {time.time() - t0:.0f} s", flush=True)

# ---- write tables
with open(os.path.join(out_dir, "cells.tsv"), "w") as f:
    f.write("id\t" + "\t".join("pre_" + c for c in CLASSES) + "\t" + "\t".join("post_" + c for c in CLASSES) + "\n")
    for c in sorted(S, key=int):
        f.write(c + "\t" + "\t".join(map(str, as_pre[c])) + "\t" + "\t".join(map(str, as_post[c])) + "\n")
with open(os.path.join(out_dir, "graph_edges.tsv"), "w") as f:
    f.write("pre\tpost\tn_rows\tn_type1\tn_type2\tn_good\tn_unk\tn_bad\tn_good_type1\tn_good_type2\n")
    for (a, b), e in edges.items():
        f.write(f"{a}\t{b}\t" + "\t".join(map(str, e[:8])) + "\n")


# ---- helpers for the summary
def quant(vals, qs=(0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1)):
    vals = sorted(vals)
    return {q: vals[min(len(vals) - 1, int(q * len(vals)))] for q in qs} if vals else {}


def giant_scc(edge_list):
    """Size of the largest strongly connected component (iterative Kosaraju)."""
    adj, radj, nodes = defaultdict(list), defaultdict(list), set()
    for a, b in edge_list:
        adj[a].append(b)
        radj[b].append(a)
        nodes.add(a)
        nodes.add(b)
    seen, order = set(), []
    for s in nodes:
        if s in seen:
            continue
        stack = [(s, iter(adj[s]))]
        seen.add(s)
        while stack:
            v, it = stack[-1]
            for w in it:
                if w not in seen:
                    seen.add(w)
                    stack.append((w, iter(adj[w])))
                    break
            else:
                order.append(v)
                stack.pop()
    comp, best = set(), 0
    for s in reversed(order):
        if s in comp:
            continue
        comp.add(s)
        stack, size = [s], 0
        while stack:
            v = stack.pop()
            size += 1
            for w in radj[v]:
                if w not in comp:
                    comp.add(w)
                    stack.append(w)
        best = max(best, size)
    return len(nodes), best


L = []
L.append(f"edge files read: {len(edge_files)}; rows read: {total}; soma-labelled ids: {len(S)}")
L.append(f"rows by (sender in set, receiver in set): {dict(membership)}")

# type versus class pair over ALL rows: this is the excitatory/inhibitory evidence
pair_rows = Counter()
pair_t1 = Counter()
for (t, pc, qc), n in type_by_pair.items():
    pair_rows[(pc, qc)] += n
    if t == "1":
        pair_t1[(pc, qc)] += n
L.append("\nTYPE-1 SHARE BY CLASS PAIR (all rows; pair: rows, share of type 1):")
for (pc, qc), n in pair_rows.most_common(14):
    L.append(f"   {pc} -> {qc}: {n}, {pair_t1[(pc, qc)] / n:.3f}")
for pair in (("AXON", "SOMA"), ("AXON", "AIS"), ("AXON", "DENDRITE")):
    n = pair_rows.get(pair, 0)
    if n:
        L.append(f"   (focus) {pair[0]} -> {pair[1]}: {n}, share of type 1 = {pair_t1[pair] / n:.3f}")

# how real-neuron-like are the 73,896 ids?
n_soma = [as_pre[c][2] + as_post[c][2] for c in S]
n_dend_in = [as_post[c][1] for c in S]
def hist(vals, edges_):
    out, prev = [], None
    for hi in edges_:
        out.append(sum(1 for v in vals if (prev is None or v > prev) and v <= hi))
        prev = hi
    out.append(sum(1 for v in vals if v > prev))
    return out
L.append("\nCELL PROFILE over soma-labelled ids")
L.append(f"rows labelled SOMA per id, buckets (<=1, <=4, <=9, <=24, <=99, more): {hist(n_soma, [1, 4, 9, 24, 99])}")
L.append(f"rows as DENDRITE receiver per id, buckets (<=0, <=9, <=49, <=199, <=999, more): {hist(n_dend_in, [0, 9, 49, 199, 999])}")
L.append(f"ids with >=5 soma rows: {sum(1 for v in n_soma if v >= 5)}; "
         f"ids with >=5 soma rows and >=20 dendrite-receiver rows: "
         f"{sum(1 for c in S if as_pre[c][2] + as_post[c][2] >= 5 and as_post[c][1] >= 20)}")

# edge graph variants
variants = {
    "all edges": lambda e: True,
    "n_rows >= 2": lambda e: e[0] >= 2,
    "n_good >= 1": lambda e: e[3] >= 1,
    "n_good >= 2": lambda e: e[3] >= 2,
    "n_bad == 0": lambda e: e[5] == 0,
    "n_bad == 0 and n_rows >= 2": lambda e: e[5] == 0 and e[0] >= 2,
}
L.append("\nGRAPH VARIANTS (edges, cells with an edge, giant strongly connected component):")
for name, keep in variants.items():
    el = [k for k, e in edges.items() if keep(e)]
    n_nodes, big = giant_scc(el) if el else (0, 0)
    L.append(f"   {name}: {len(el)} edges, {n_nodes} cells, giant SCC {big}")

# per-sender type share on clean (good) rows only
fr = [v[0] / (v[0] + v[1]) for v in out_good.values() if v[0] + v[1] >= 20]
h = Counter(min(9, int(x * 10)) for x in fr)
L.append(f"\nSENDERS with >=20 good (AXON->DENDRITE/SOMA/AIS) rows: {len(fr)}; share of type-1 rows per sender, "
         f"in tenths: {[h.get(k, 0) for k in range(10)]}")
fe = [e[6] / (e[6] + e[7]) for e in edges.values() if e[6] + e[7] >= 5]
he = Counter(min(9, int(x * 10)) for x in fe)
L.append(f"EDGES with >=5 good rows: {len(fe)}; share of type-1 rows per edge, in tenths: {[he.get(k, 0) for k in range(10)]}")

if PROOF:
    L.append(f"\nproofread cells in the soma-labelled set: {sum(1 for c in PROOF if c in S)} of {len(PROOF)}")
    pe = [(a, b) for (a, b) in edges if a in PROOF or b in PROOF]
    L.append(f"edges touching a proofread cell: {len(pe)}; between two proofread cells: "
             f"{sum(1 for a, b in pe if a in PROOF and b in PROOF)}")

text = "\n".join(L)
print("\n" + text)
with open(os.path.join(out_dir, "summary.txt"), "w") as f:
    f.write(text + "\n")
print("\nSaved to", out_dir)

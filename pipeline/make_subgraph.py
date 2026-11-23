r"""Cut a connected subgraph of H01 neurons for the reservoir controller.

Inputs (from the logged pipeline, version 2 outputs):
    graph_edges.tsv    soma-labelled cell pairs with synapse-row counts and class-pair quality counts
    cells.tsv          per-id class label counts (defines which ids are neurons)
    neuron_signs.tsv   per-neuron outgoing type counts (gives each neuron one sign)

Usage:
    python make_subgraph.py --size 1000 --tier A --seed 0 --out subgraphs/
Options:
    --tier A | B | C      A: all edges among neurons; B: only edges with no implausible class pair (n_bad == 0);
                          C: only edges with at least 2 synapse rows
    --size N              number of neurons to keep (grown as one connected region)
    --seed K              replicate number (picks a different starting neuron, reproducibly)
    --inputs DIR          folder holding the three tables (default ../outputs/graph_v2)
Output (in --out): sg_<tier>_<size>_s<seed>.tsv.gz style files, written as two plain files:
    nodes_<name>.tsv   id, sign (+1 or -1), sign_source (data or unknown), n_good_rows
    edges_<name>.tsv   pre, post, weight (synapse rows)
    stats_<name>.txt   size, edges, density, strongly connected core, share inhibitory, etc.
Standard library only.
"""
import argparse
import heapq
import os
import random
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=1000)
ap.add_argument("--tier", choices=["A", "B", "C"], default="A")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--inputs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "graph_v2"))
ap.add_argument("--out", default="subgraphs")
ap.add_argument("--min_good_rows", type=int, default=10, help="good rows needed to trust a neuron's sign")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
name = f"{args.tier}_{args.size}_s{args.seed}"

# neurons (same rule as the log: SOMA rows >= 5 and DENDRITE-receiver rows >= 20)
neurons = set()
with open(os.path.join(args.inputs, "cells.tsv")) as f:
    hdr = f.readline().split()
    ix = {h: i for i, h in enumerate(hdr)}
    for line in f:
        p = line.split()
        if int(p[ix["pre_SOMA"]]) + int(p[ix["post_SOMA"]]) >= 5 and int(p[ix["post_DENDRITE"]]) >= 20:
            neurons.add(p[0])

# signs from good rows only
sign, good_rows = {}, {}
with open(os.path.join(args.inputs, "neuron_signs.tsv")) as f:
    f.readline()
    for line in f:
        p = line.split()
        n_good, g1 = int(p[4]), int(p[5])
        good_rows[p[0]] = n_good
        if n_good >= args.min_good_rows:
            sign[p[0]] = -1 if g1 / n_good >= 0.5 else 1    # type 1 = inhibitory (working assumption)

# edges among neurons for the chosen tier
w = {}
with open(os.path.join(args.inputs, "graph_edges.tsv")) as f:
    f.readline()
    for line in f:
        a, b, n, t1, t2, good, unk, bad, g1, g2 = line.split()
        if a not in neurons or b not in neurons:
            continue
        n, bad = int(n), int(bad)
        if args.tier == "B" and bad != 0:
            continue
        if args.tier == "C" and n < 2:
            continue
        w[(a, b)] = n

und = defaultdict(dict)                   # undirected total weight
for (a, b), n in w.items():
    und[a][b] = und[a].get(b, 0) + n
    und[b][a] = und[b].get(a, 0) + n


def scc_nodes(nodes, edge_list):
    """Nodes of the largest strongly connected component (iterative Kosaraju)."""
    adj, radj = defaultdict(list), defaultdict(list)
    for a, b in edge_list:
        adj[a].append(b)
        radj[b].append(a)
    seen, order = set(), []
    for s in nodes:
        if s in seen:
            continue
        st = [(s, iter(adj[s]))]
        seen.add(s)
        while st:
            v, it = st[-1]
            for x in it:
                if x not in seen:
                    seen.add(x)
                    st.append((x, iter(adj[x])))
                    break
            else:
                order.append(v)
                st.pop()
    done, best = set(), []
    for s in reversed(order):
        if s in done:
            continue
        done.add(s)
        st, comp = [s], []
        while st:
            v = st.pop()
            comp.append(v)
            for x in radj[v]:
                if x not in done:
                    done.add(x)
                    st.append(x)
        if len(comp) > len(best):
            best = comp
    return set(best)


# work inside the giant strongly connected component of the tier graph
core = scc_nodes(list(und), list(w))
rng = random.Random(args.seed)
deg = {v: sum(1 for x in und[v] if x in core) for v in core}
top = sorted(core, key=lambda v: -deg[v])[:50]
seed_node = top[0] if args.seed == 0 else rng.choice(top)

# greedy growth: always add the outside neuron most strongly tied to the current set
chosen = {seed_node}
gain = defaultdict(int)
heap = []
def push_neighbours(v):
    for x, n in und[v].items():
        if x in core and x not in chosen:
            gain[x] += n
            heapq.heappush(heap, (-gain[x], -deg[x], x))
push_neighbours(seed_node)
while len(chosen) < args.size and heap:
    g, _, x = heapq.heappop(heap)
    if x in chosen or -g != gain[x]:
        continue                       # stale heap entry
    chosen.add(x)
    push_neighbours(x)

sub = {k: n for k, n in w.items() if k[0] in chosen and k[1] in chosen}
inner = scc_nodes(list(chosen), list(sub))

with open(os.path.join(args.out, f"nodes_{name}.tsv"), "w") as f:
    f.write("id\tsign\tsign_source\tn_good_rows\n")
    for v in sorted(chosen, key=int):
        f.write(f"{v}\t{sign.get(v, 1)}\t{'data' if v in sign else 'unknown'}\t{good_rows.get(v, 0)}\n")
with open(os.path.join(args.out, f"edges_{name}.tsv"), "w") as f:
    f.write("pre\tpost\tweight\n")
    for (a, b), n in sub.items():
        f.write(f"{a}\t{b}\t{n}\n")

n_known = sum(1 for v in chosen if v in sign)
n_inh = sum(1 for v in chosen if sign.get(v) == -1)
recip = sum(1 for (a, b) in sub if (b, a) in sub)
stats = [
    f"name: {name}",
    f"neurons in graph tier {args.tier}: {len(set(a for a, _ in w) | set(b for _, b in w))}; giant strongly connected core: {len(core)}",
    f"start neuron: {seed_node}",
    f"neurons: {len(chosen)}; edges: {len(sub)}; synapse rows on them: {sum(sub.values())}",
    f"mean in+out degree: {2 * len(sub) / len(chosen):.2f}; density (edges / n(n-1)): {len(sub) / (len(chosen) * (len(chosen) - 1)):.4f}",
    f"largest strongly connected component inside the subgraph: {len(inner)}",
    f"reciprocal edges: {recip / max(1, len(sub)):.3f}",
    f"neurons with a data-derived sign: {n_known} ({n_known / len(chosen):.1%}); of those inhibitory: {n_inh} ({n_inh / max(1, n_known):.1%})",
    f"edge weight (synapse rows) mean {sum(sub.values()) / max(1, len(sub)):.2f}, max {max(sub.values()) if sub else 0}",
]
with open(os.path.join(args.out, f"stats_{name}.txt"), "w") as f:
    f.write("\n".join(stats) + "\n")
print("\n".join(stats))

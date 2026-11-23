"""Force-directed layout of the real H01 neuron graph (all neurons by our rule, tier A edges), cached to layout_h01_neurons.npz."""
import sys, time
import numpy as np, networkx as nx
base = "../outputs/graph_v2/"
neurons = set()
with open(base + "cells.tsv") as f:
    hdr = f.readline().split(); ix = {h: i for i, h in enumerate(hdr)}
    for l in f:
        p = l.split()
        if int(p[ix["pre_SOMA"]]) + int(p[ix["post_SOMA"]]) >= 5 and int(p[ix["post_DENDRITE"]]) >= 20:
            neurons.add(p[0])
G = nx.Graph()
G.add_nodes_from(sorted(neurons, key=int))
with open(base + "graph_edges.tsv") as f:
    f.readline()
    for l in f:
        p = l.split()
        if p[0] in neurons and p[1] in neurons and p[0] != p[1]:
            G.add_edge(p[0], p[1])
print("graph", G.number_of_nodes(), G.number_of_edges(), flush=True)
cc = max(nx.connected_components(G), key=len)
H = G.subgraph(cc)
print("largest component", len(cc), flush=True)
t = time.time()
pos = nx.spring_layout(H, seed=1, iterations=int(sys.argv[1]) if len(sys.argv) > 1 else 50)
print("layout seconds", time.time() - t, flush=True)
ids = list(H.nodes())
np.savez("layout_h01_neurons.npz", ids=np.array(ids), xy=np.array([pos[i] for i in ids]),
         n_all=G.number_of_nodes(), e_all=G.number_of_edges())
print("saved", flush=True)

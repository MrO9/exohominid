"""ExoHominid reservoir: a leaky recurrent network whose wiring comes from an H01 subgraph (or from a control graph).
Standard library only (the Webots Python may not have numpy).

Graph dict: {"n": int, "pre": [..], "post": [..], "w": [..] (positive synapse-row counts), "sign": [+1/-1 per node],
             "known": [bool per node], "name": str}
Edge i -> j means neuron pre[i] sends to neuron post[i]. The sign belongs to the SENDER (Dale's law), except in the
"esn" control, where every edge gets its own random sign.
"""
import math
import os
import random


# ------------------------------------------------------------------ loading
def load_subgraph(folder, name):
    """folder holds nodes_<name>.tsv and edges_<name>.tsv written by make_subgraph.py."""
    ids, sign, known = [], [], []
    with open(os.path.join(folder, f"nodes_{name}.tsv")) as f:
        f.readline()
        for line in f:
            p = line.split()
            ids.append(p[0])
            sign.append(int(p[1]))
            known.append(p[2] == "data")
    ix = {c: i for i, c in enumerate(ids)}
    pre, post, w = [], [], []
    with open(os.path.join(folder, f"edges_{name}.tsv")) as f:
        f.readline()
        for line in f:
            a, b, n = line.split()
            if a in ix and b in ix and a != b:
                pre.append(ix[a])
                post.append(ix[b])
                w.append(float(n))
    return {"n": len(ids), "pre": pre, "post": post, "w": w, "sign": sign, "known": known, "name": name, "ids": ids}


# ------------------------------------------------------------------ sign policies (applied to node signs)
def apply_sign_policy(g, policy, seed=0):
    """data      : known sign from the H01 synapse types; neurons without evidence are excitatory (default)
       random17  : each neuron inhibitory with probability 0.17 (keeps the typical cortical share, ignores the data)
       permute   : the data signs shuffled among the neurons (same share, wrong neurons)
       allpos    : all excitatory
       flipped   : the data signs reversed (tests whether the type labelling matters)"""
    g = dict(g)
    s = list(g["sign"])
    rng = random.Random(seed + 7)
    if policy == "data":
        pass
    elif policy == "random17":
        s = [-1 if rng.random() < 0.17 else 1 for _ in s]
    elif policy == "permute":
        rng.shuffle(s)                           # same inhibitory share as the data, assigned to random neurons
    elif policy == "allpos":
        s = [1] * len(s)
    elif policy == "flipped":
        s = [-x for x in s]
    else:
        raise ValueError(f"unknown sign policy {policy}")
    g["sign"] = s
    g["sign_policy"] = policy
    return g


# ------------------------------------------------------------------ control graphs
def control_graph(g, kind, seed=0):
    """h01     : the graph itself
       shuffle : same neurons, same in- and out-degrees, same weights, same signs, but wiring randomised
                 (double edge swaps); keeps degree structure, destroys everything else
       er      : same number of neurons and edges, endpoints uniformly random, same weight list, same sign share
       esn     : a standard echo-state network: same size and mean degree, Gaussian weights, one random sign per edge"""
    rng = random.Random(seed + 101)
    n, E = g["n"], len(g["pre"])
    out = dict(g)
    if kind == "h01":
        return out
    pre, post, w = list(g["pre"]), list(g["post"]), list(g["w"])
    if kind == "shuffle":
        have = set(zip(pre, post))
        for _ in range(10 * E):
            i, j = rng.randrange(E), rng.randrange(E)
            a, b, c, d = pre[i], post[i], pre[j], post[j]
            if a == d or c == b or (a, d) in have or (c, b) in have:
                continue
            have.discard((a, b))
            have.discard((c, d))
            have.add((a, d))
            have.add((c, b))
            post[i], post[j] = d, b
        out.update(pre=pre, post=post, w=w)
    elif kind in ("er", "esn"):
        have, np_, nq_ = set(), [], []
        while len(np_) < E:
            a, b = rng.randrange(n), rng.randrange(n)
            if a != b and (a, b) not in have:
                have.add((a, b))
                np_.append(a)
                nq_.append(b)
        if kind == "er":
            rng.shuffle(w)
            sg = list(g["sign"])
            rng.shuffle(sg)
            out.update(pre=np_, post=nq_, w=w, sign=sg)
        else:
            out.update(pre=np_, post=nq_, w=[abs(rng.gauss(0.0, 1.0)) + 1e-6 for _ in range(E)],
                       sign=[1] * n, edge_sign=[rng.choice((-1, 1)) for _ in range(E)])
    else:
        raise ValueError(f"unknown graph kind {kind}")
    out["control"] = kind
    return out


# ------------------------------------------------------------------ reservoir
def spectral_radius(rows, n, iters=150, seed=1):
    """Magnitude of the dominant eigenvalue by power iteration (geometric mean of the growth over the last 100 steps)."""
    rng = random.Random(seed)
    v = [rng.gauss(0, 1) for _ in range(n)]
    logsum, cnt = 0.0, 0
    for k in range(iters):
        u = [sum(wt * v[i] for i, wt in row) for row in rows]
        nrm = math.sqrt(sum(x * x for x in u))
        if nrm < 1e-12:
            return 0.0
        if k >= iters - 100:
            logsum += math.log(nrm / (math.sqrt(sum(x * x for x in v)) or 1.0))
            cnt += 1
        v = [x / nrm for x in u]
    return math.exp(logsum / cnt)


class Reservoir:
    """x <- (1-leak) x + leak * tanh( rho * W x + W_in u ).   W: signed, from the graph, rescaled to spectral radius 1."""

    def __init__(self, g, n_in, rho=0.9, leak=0.3, in_scale=0.5, in_density=0.25, weight_mode="sqrt", seed=0):
        self.n, self.n_in, self.leak = g["n"], n_in, leak
        n = self.n
        mag = {"sqrt": math.sqrt, "binary": lambda x: 1.0, "count": lambda x: x}[weight_mode]
        rows = [[] for _ in range(n)]
        es = g.get("edge_sign")
        for k, (a, b, wt) in enumerate(zip(g["pre"], g["post"], g["w"])):
            s = es[k] if es is not None else g["sign"][a]
            rows[b].append((a, s * mag(wt)))
        sr = spectral_radius(rows, n)
        self.radius_raw = sr
        k = rho / sr if sr > 0 else 0.0
        self.rows = [[(i, wt * k) for i, wt in row] for row in rows]
        rng = random.Random(seed + 5)                        # input wiring: identical recipe for every graph
        self.in_map = [[(j, rng.uniform(-1.0, 1.0) * in_scale) for j in range(n) if rng.random() < in_density]
                       for _ in range(n_in)]
        self.x = [0.0] * n
        self.n_edges = len(g["pre"])

    def reset(self):
        self.x = [0.0] * self.n

    def step(self, u):
        drive = [0.0] * self.n
        for d, ud in enumerate(u):
            if ud != 0.0:
                for j, wt in self.in_map[d]:
                    drive[j] += wt * ud
        x, a, tanh = self.x, self.leak, math.tanh
        new = []
        for j, row in enumerate(self.rows):
            s = drive[j]
            for i, wt in row:
                s += wt * x[i]
            new.append((1.0 - a) * x[j] + a * tanh(s))
        self.x = new
        return new


class NoReservoir:
    """Control without any recurrent network.
    "direct": the readout sees the 12 inputs themselves (a plain linear policy, no memory, no hidden units).
    "bias"  : the readout sees only the constant 1, so it can only learn one fixed correction (turn, lean, stride)."""

    def __init__(self, kind, n_in):
        self.kind, self.n_in = kind, n_in
        self.n = n_in if kind == "direct" else 0
        self.n_edges, self.radius_raw = 0, 0.0

    def reset(self):
        pass

    def step(self, u):
        return list(u) if self.kind == "direct" else []


def build(g_name, folder, kind="h01", sign_policy="data", n_in=12, rho=0.9, leak=0.3, in_scale=0.5,
          weight_mode="sqrt", seed=0):
    if kind in ("direct", "bias"):
        res = NoReservoir(kind, n_in)
        return res, f"no reservoir ({kind}): readout over {res.n} inputs plus a constant, {3 * (res.n + 1)} trained numbers"
    base = load_subgraph(folder, g_name)
    g = control_graph(apply_sign_policy(base, sign_policy, seed), kind, seed)
    res = Reservoir(g, n_in, rho=rho, leak=leak, in_scale=in_scale, weight_mode=weight_mode, seed=seed)
    n_inh = sum(1 for s in g["sign"] if s < 0)
    info = (f"reservoir {g_name} graph={kind} signs={sign_policy} weights={weight_mode}: {res.n} neurons, {res.n_edges} edges, "
            f"inhibitory {n_inh} ({n_inh / res.n:.0%}), raw spectral radius {res.radius_raw:.2f} -> rho {rho}, leak {leak}")
    return res, info

"""Summary statistics for an H01 synapse Avro file (works on a partial download).

Usage:  python h01_stats.py head.avro
Needs:  pip install fastavro
"""
import sys
from collections import Counter

from fastavro import reader


def get(d, *keys):
    """Safely walk nested dicts; return None if anything is missing."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


path = sys.argv[1]

n = 0
types = Counter()
pre_types = Counter()
post_types = Counter()
labels = Counter()
conf = []
xs, ys, zs = [], [], []
pre_ids, post_ids = Counter(), Counter()
pre_mismatch = post_mismatch = 0
examples = []
self_loops = 0
missing_ids = 0
truncated = False

with open(path, "rb") as f:
    try:
        for row in reader(f):
            n += 1
            pre = row.get("pre_synaptic_site") or {}
            post = row.get("post_synaptic_partner") or {}
            types[row.get("type")] += 1
            pre_types[pre.get("type")] += 1
            post_types[post.get("type")] += 1
            labels[(pre.get("class_label"), post.get("class_label"))] += 1
            c = row.get("confidence")
            if c is not None:
                conf.append(c)
            loc = row.get("location") or {}
            for lst, k in ((xs, "x"), (ys, "y"), (zs, "z")):
                if loc.get(k) is not None:
                    lst.append(loc[k])
            a, b = pre.get("neuron_id"), post.get("neuron_id")
            if a is None or b is None:
                missing_ids += 1
            else:
                pre_ids[a] += 1
                post_ids[b] += 1
                if a == b:
                    self_loops += 1
            for side, nid, base in (("pre", a, pre.get("base_neuron_id")),
                                    ("post", b, post.get("base_neuron_id"))):
                if nid is not None and base is not None and str(nid) != str(base):
                    if side == "pre":
                        pre_mismatch += 1
                    else:
                        post_mismatch += 1
                    if len(examples) < 4:
                        examples.append((side, nid, base))
    except Exception as e:  # a cut-off file ends mid-block
        truncated = True
        print(f"(stopped reading after {n} rows: {type(e).__name__}: {e})")

print(f"\nRows read: {n}" + ("  [file was cut off, that is expected]" if truncated else ""))
print("\nRow-level 'type' counts:", dict(types))
print("pre_synaptic_site 'type' counts:", dict(pre_types))
print("post_synaptic_partner 'type' counts:", dict(post_types))
print("(pre class_label, post class_label) counts:", dict(labels))

conf.sort()
if conf:
    print("\nConfidence: min %.3f  median %.3f  max %.3f" % (conf[0], quantile(conf, 0.5), conf[-1]))
    for t in (0.5, 0.8, 0.9):
        frac = sum(1 for c in conf if c >= t) / len(conf)
        print(f"  share with confidence >= {t}: {frac:.1%}")

for name, vals in (("x", xs), ("y", ys), ("z", zs)):
    if vals:
        print(f"location {name}: min {min(vals)}  max {max(vals)}")

print(f"\nUnique senders (neuron_id): {len(pre_ids)}")
print(f"Unique receivers (neuron_id): {len(post_ids)}")
print(f"Rows where sender == receiver: {self_loops}")
print(f"Rows with a missing neuron_id: {missing_ids}")
print(f"neuron_id != base_neuron_id:  sender {pre_mismatch} rows, receiver {post_mismatch} rows")
for ex in examples:
    print("   example:", ex)
print("\nMost frequent senders:", pre_ids.most_common(5))
print("Most frequent receivers:", post_ids.most_common(5))

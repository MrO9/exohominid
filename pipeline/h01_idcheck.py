"""ID check: do the 104 proofread cell IDs appear as neuron_id or base_neuron_id?

Usage:  python h01_idcheck.py shard0.avro [more shards...]
Needs:  pip install fastavro   (already installed)
Reads each file one row at a time; takes about a minute per shard.
Writes idcheck_out.txt next to this script.
"""
import os
import sys
from collections import Counter

from fastavro import reader

IDS = set("""546925828
547114083
620880207
678539249
751294744
810151953
810970127
883843993
955432427
1072605926
1304599410
1465400601
1669770671
1684504313
1787105373
1830470325
1874244336
1900905433
1930502394
2001418787
2047644309
2074305737
2103991145
2104047922
2195261734
2252715458
2306214824
2325998949
2451406889
2530864375
2820958695
2848552900
2888180252
2889815798
2903686765
3093748739
3111823553
3139317430
3178243558
3329117097
3383650350
3432315514
3470629528
3503540481
3519995546
3571083397
3680152874
3751392341
3761379470
3812058320
3836078296
3954491075
3955003482
4010150634
4039221194
4125702786
4138580687
4157825456
4188575291
4197933517
4260528559
4365276903
4420044370
4437316933
4467761196
4476359994
4522237537
4535129012
4554213569
4582188328
4607885827
4641147055
4668874666
4711567524
4711641825
4830551047
4853956860
4873639924
4932057570
5013648003
5015035926
5060444657
5061320552
5136107765
5173982155
5175880292
5215113731
5216806368
5319188363
5352961465
5355296383
5439194879
5536146025
5584343344
5654281423
5687162964
5805562981
5832764277
5965472721
6126390733
6833911543
7052969349
7196644737
7893125555""".split())

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "idcheck_out.txt")
rows = 0
hits = Counter()          # (side, field) -> rows where that field is one of the 104 IDs
row_type_by_hit = Counter()   # (side, field, row type) -> rows
site_type_by_hit = Counter()  # (side, field, site type) -> rows
class_by_hit = Counter()      # (side, field, class_label) -> rows
per_id_pre, per_id_post = Counter(), Counter()
both = 0                  # rows where sender AND receiver are in the 104

for path in sys.argv[1:]:
    with open(path, "rb") as f:
        for row in reader(f):
            rows += 1
            pre = row.get("pre_synaptic_site") or {}
            post = row.get("post_synaptic_partner") or {}
            flags = {}
            for side, site in (("pre", pre), ("post", post)):
                for field in ("neuron_id", "base_neuron_id"):
                    v = site.get(field)
                    if v is not None and str(v) in IDS:
                        hits[(side, field)] += 1
                        row_type = row.get("type")
                        row_type_by_hit[(side, field, row_type)] += 1
                        site_type_by_hit[(side, field, site.get("type"))] += 1
                        class_by_hit[(side, field, site.get("class_label"))] += 1
                        flags[(side, field)] = str(v)
                        if field == "neuron_id":
                            (per_id_pre if side == "pre" else per_id_post)[str(v)] += 1
            if ("pre", "base_neuron_id") in flags and ("post", "base_neuron_id") in flags:
                both += 1

lines = [f"Rows read: {rows}", ""]
for key in sorted(hits):
    lines.append(f"rows where {key[0]}.{key[1]} is one of the 104: {hits[key]}")
lines.append(f"rows where BOTH sender and receiver base_neuron_id are in the 104: {both}")
lines.append("")
lines.append("by row type: " + str(dict(row_type_by_hit)))
lines.append("by site type: " + str(dict(site_type_by_hit)))
lines.append("by class label: " + str(dict(class_by_hit)))
lines.append("")
lines.append(f"distinct proofread IDs seen as neuron_id sender: {len(per_id_pre)}, receiver: {len(per_id_post)}")
text = "\n".join(lines)
print(text)
with open(out_path, "w") as f:
    f.write(text + "\n")
print("\nSaved to", out_path)

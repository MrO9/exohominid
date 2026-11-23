"""Stream all H01 c3 synapse shards, one at a time, and keep a compact edge list.

For every shard: download (about 197 MB), read every synapse, write a small gzip file of
edges, then DELETE the shard. Nothing large is kept. Safe to stop (Ctrl+C) and rerun: shards
that already have an output file are skipped.

Usage (PowerShell):
    python h01_edges.py                 all shards, output next to the script
    python h01_edges.py 0 40            only shards 0..39 (to split work over PCs)
Optional: set the output folder with the environment variable H01_OUT.

Output folder h01_edges_out\\ contains, per shard N:
    edges_NNN.tsv.gz   columns: pre_neuron_id, post_neuron_id, row_type, pre_class, post_class
    somas_NNN.txt      neuron_ids that appear with class_label SOMA in that shard
and progress_log.txt (one line per finished shard: time, rows, kept rows, seconds).
Kept rows: both neuron_ids present and different. Confidence is already >= 0.5 in the export.
Needs: fastavro (installed). Everything else is standard library.
"""
import gzip
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

from fastavro import reader

BASE = "https://storage.googleapis.com/h01-release/data/20210729/c3/synapses/exported/export%012d"
here = os.path.dirname(os.path.abspath(__file__))
out_dir = os.environ.get("H01_OUT") or os.path.join(here, "h01_edges_out")
os.makedirs(out_dir, exist_ok=True)
tmp = os.path.join(out_dir, "current_shard.avro")
start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
stop = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(os.path.join(out_dir, "progress_log.txt"), "a") as f:
        f.write(line + "\n")


def download(url, dest):
    """Return True if downloaded, False if the shard does not exist (404)."""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            err = e
        except Exception as e:
            err = e
        log(f"  download problem ({err}); retry {attempt + 1}/5")
        time.sleep(10 * (attempt + 1))
    raise RuntimeError("download failed 5 times: " + url)


for n in range(start, stop):
    edges_path = os.path.join(out_dir, f"edges_{n:03d}.tsv.gz")
    if os.path.exists(edges_path):
        continue
    t0 = time.time()
    if not download(BASE % n, tmp):
        log(f"shard {n} does not exist (404): finished, last shard is {n - 1}")
        break
    size = os.path.getsize(tmp)
    rows = kept = 0
    somas = set()
    part = edges_path + ".part"
    with gzip.open(part, "wt") as out, open(tmp, "rb") as f:
        for row in reader(f):
            rows += 1
            pre = row.get("pre_synaptic_site") or {}
            post = row.get("post_synaptic_partner") or {}
            a, b = pre.get("neuron_id"), post.get("neuron_id")
            pc, qc = pre.get("class_label"), post.get("class_label")
            if pc == "SOMA" and a is not None:
                somas.add(a)
            if qc == "SOMA" and b is not None:
                somas.add(b)
            if a is None or b is None or a == b:
                continue
            kept += 1
            out.write(f"{a}\t{b}\t{row.get('type')}\t{pc}\t{qc}\n")
    with open(os.path.join(out_dir, f"somas_{n:03d}.txt"), "w") as f:
        f.write("\n".join(str(s) for s in sorted(somas)) + ("\n" if somas else ""))
    os.replace(part, edges_path)   # the output file only appears when the shard is complete
    os.remove(tmp)
    log(f"shard {n}: {size/1e6:.0f} MB, rows {rows}, kept {kept}, soma ids {len(somas)}, {time.time()-t0:.0f} s")

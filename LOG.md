# H01 connectome-to-controller project: process log

Purpose: a dated record of every step, command, result and decision, so the work can be reproduced and described on the project website and in the paper. Entries are appended in order. Nothing here is edited after the fact; corrections are added as new entries.

Conventions
- "Confirmed" means we saw it in output. "Hypothesis" means a guess that still needs a check.
- Times are America/Chicago.
- Dataset: H01 (human temporal cortex, Google). License of released data: CC BY 4.0. Cite the H01 paper and the SegCLR paper when publishing.

---

## 0. Scope decision (2026-11-18)

- Goal: build a controller for a simulated robot (Webots) whose recurrent wiring comes from a subgraph of the H01 connectome, and see what happens. Not aiming for novelty; aiming to do it, publish the results and discuss them.
- Constraint change: leave the earlier single-file, standard-library-only rule. A multi-file project is allowed; more compute (lab machines) may be used if needed.
- Prior art noted but not evaluated: a third-party Hugging Face repo (NathanRoll/h01-cortex-snn) hosts an H01 connectivity matrix (claims 16,087 neurons, 76,903 connections, built from 166 synapse shards, confidence >= 0.50). Unverified by us. We do not rely on it.

## 1. Documentation reviewed (2026-11-18)

Two pages from the H01 release site, saved as PDFs:
- Released Data: https://h01-release.storage.googleapis.com/data.html
- SegCLR Embeddings: https://h01-release.storage.googleapis.com/embeddings.html

Confirmed from those pages:
- Data lives in a public Google Cloud Storage bucket: `gs://h01-release/data/...` (CC BY 4.0).
- 104 manually proofread cells: `data/20210601/proofread_104/` (mesh, skeletons, subcompartments, SWC zip).
- c3 segmentation (conservative, suggested for connectomics): `data/20210601/c3/`; c2 is more aggressive with more merge errors.
- c3 synapses as points with excitatory or inhibitory type: `data/20210729/c3/synapses/precomputed`.
- c3 synaptic-connections database, Avro format: `data/20210729/c3/synapses/exported/`.
- SegCLR embeddings: sharded CSV zips, about 1 TB, `data/20220326/c3/embeddings/segclr_csvzips/`.
- Raw EM: 1.4 PB (from the Google Research blog); not used.

## 2. File listing of the synapse export (2026-11-18 ~18:17)

Command (opened in a browser; this is the public Cloud Storage JSON API):
`https://storage.googleapis.com/storage/v1/b/h01-release/o?prefix=data/20210729/c3/synapses/exported/&maxResults=5`

Confirmed (first page of 5 results):

| Object | Size (bytes) | MD5 (hex, converted from the API's base64) |
| --- | --- | --- |
| `exported/export000000000000` | 197,344,616 | 15f1a2748fb57b52a67f0e7eec48b92b |
| `exported/export000000000000.json` | 757,111,597 | 484db47da94730edb36986addfae8db2 |
| `exported/export000000000001` | 197,434,059 | ff436fcd5563e8e44293531e942da0fa |
| `exported/export000000000001.json` | 757,457,938 | 0e8604ba909940530748e7570a4a81ab |

- Created 2023-08-09. Storage class NEARLINE.
- Pattern: one extensionless file and one `.json` file per shard.
- Hypothesis (later confirmed in section 3): the extensionless files are Avro and the `.json` files are a text copy of the same records.
- Not yet checked: total number of shards (third-party page says 166).

## 3. Peeking at the Avro file (2026-11-18 ~18:25-18:31)

Environment: Windows PC (PowerShell), Python from the Microsoft Store, `pip install fastavro`.

Commands:
1. `curl.exe -r 0-5000000 -o head.avro "https://storage.googleapis.com/h01-release/data/20210729/c3/synapses/exported/export000000000000"`
   - Result: first 5,000,001 bytes saved as `head.avro` (4.76 MB shown by curl).
2. `python h01_peek.py head.avro` (script version in `scripts/h01_peek.py`).

Confirmed:
- `export000000000000` is an Avro container file (the script read its schema).
- The schema: one record per synapse. Top-level fields: `pre_synaptic_site`, `post_synaptic_partner`, `location`, `type`, `contact_area`, `confidence`, `bounding_box`.
- Each of `pre_synaptic_site` and `post_synaptic_partner` contains: `neuron_id`, `id`, `num_voxels`, `centroid {x,y,z}`, `type`, `subtype`, `confidence`, `bounding_box`, `mask`, `class_label`, `base_neuron_id`.
- In the first five rows: `class_label` was AXON for the sender and DENDRITE for the receiver; `contact_area`, `num_voxels` and `mask` were empty (None); `type` was 2 on every row.

## 4. Statistics on the first 5 MB (2026-11-18 ~18:33)

Command: `python h01_stats.py head.avro` (script in `scripts/h01_stats.py`; full output saved in `outputs/stats_head_avro.txt`).

Confirmed (25,204 rows read before the cut-off):
- Confidence: minimum exactly 0.500 (the export is already filtered at 0.5); median 0.970; 92.1% >= 0.8; 85.6% >= 0.9.
- Row-level `type`: value 2 on 14,937 rows (59%), value 1 on 10,267 rows (41%).
- Site `type`: sender is 1 and receiver is 2 on 24,293 rows; swapped (sender 2, receiver 1) on 911 rows (3.6%).
- Class-label pairs: AXON to DENDRITE 20,137 (about 80%); AXON to UNKNOWN 2,496; UNKNOWN to DENDRITE 835; AXON to AXON 523; DENDRITE to DENDRITE 502; others smaller (see output file).
- Position: x 69,591 to 479,417; y 41,519 to 289,315; z 256 to 5,125. So one shard spans the whole volume (it is not a slab).
- Unique senders 24,337; unique receivers 23,181; no sender appears more than twice, so a single shard has very few synapses per cell. Building a subgraph requires reading every shard.
- No row has sender == receiver.
- 863 rows (3.4%) have a missing `neuron_id`.
- `neuron_id` differs from `base_neuron_id` on the sender side in 10,706 rows and on the receiver side in 14,977 rows.

Open questions (hypotheses, not facts):
1. Which of `neuron_id` and `base_neuron_id` identifies the cell. To be tested using the 104 proofread cell IDs.
2. Whether row-level `type` 1 and 2 mean inhibitory and excitatory, and which is which. The 59/41 split is not the lopsided split expected for excitatory versus inhibitory, so no assumption is made.
3. Whether the 911 swapped rows have a wrong direction label.

Decisions so far:
- Plan to use only rows with a non-missing `neuron_id` and confidence >= 0.5 (already true of the file); class-label filter to be decided.
- Plan to stream all shards one at a time (about 197 MB each), keep only rows touching chosen cells, delete the shard afterwards. Rough total if there are 166 shards: about 33 GB transferred, never stored at once.

## 5. Next steps (as of 2026-11-18 ~18:36)

1. List the proofread-cell skeleton files to get the 104 cell IDs:
   `https://storage.googleapis.com/storage/v1/b/h01-release/o?prefix=data/20210601/proofread_104/skeletons/&maxResults=200&fields=items(name,size)`
2. Download one whole shard (about 197 MB): `curl.exe -o shard0.avro "https://storage.googleapis.com/h01-release/data/20210729/c3/synapses/exported/export000000000000"`; verify its MD5 against section 2.
3. Write and run an ID-check script (104 IDs against `neuron_id` and `base_neuron_id`).
4. Decide the subgraph size from real connection counts (candidate sizes to compare: about 100, 1,000, 10,000 neurons).

---

## 6. Proofread-cell folder listing (2026-11-18 ~18:37)

Command (browser): `https://storage.googleapis.com/storage/v1/b/h01-release/o?prefix=data/20210601/proofread_104/skeletons/&maxResults=200&fields=items(name,size)`

Confirmed:
| Object | Size (bytes) |
| --- | --- |
| `proofread_104/skeletons/0.shard` | 11,313,211 |
| `proofread_104/skeletons/1.shard` | 7,703,068 |
| `proofread_104/skeletons/2.shard` | 11,004,341 |
| `proofread_104/skeletons/3.shard` | 7,424,863 |
| `proofread_104/skeletons/info` | 249 |
| `proofread_104/skeletons/104_proofread_neurons_swc.zip` | 59,832,937 |

- The cell IDs are NOT in the file names: the skeletons are stored as four Neuroglancer-style shard files plus a small `info` file, not one file per cell. (The "shard" format name is from the file names; the exact format is not yet confirmed.)
- Total listed size is small (about 97 MB with the zip), so downloading all of it is cheap.
- Plan: the SWC zip (about 60 MB) probably has one file per cell, and the file names may be the cell IDs. Hypothesis, to be checked with `scripts/h01_zip_names.py`, which only lists the zip contents.

Next: download the zip, run the script, record the names (and whether they look like the IDs in the synapse file) in the next entry.

(Append new entries below. For every step: date and time, exact command, what was seen, what was decided.)

## 7. Contents of the SWC zip (2026-11-18 ~18:54)

Commands (PowerShell, from `C:\Users\chisi`):
1. `curl.exe -o 104_proofread_neurons_swc.zip "https://storage.googleapis.com/h01-release/data/20210601/proofread_104/skeletons/104_proofread_neurons_swc.zip"` (57.06 MB).
2. First attempt of `python C:\Users\chisi\Downloads\h01_zip_names.py` failed twice: an old copy of the script needed a path argument, then the zip sat in `C:\Users\chisi` rather than Downloads (curl saves to the current folder). Fixed by running with the zip in Downloads.
3. `python C:\Users\chisi\Downloads\h01_zip_names.py` (output saved in `outputs/zip_names.txt`).

Confirmed:
- The zip holds 3,327 files named `<cellID>.<n>.swc` (for example `1072605926.0.swc`). Every name matches that pattern.
- There are exactly 104 distinct cell IDs (the prefix before the first dot), matching the "104 proofread cells". Each cell has 1 to 267 SWC files (largest: 1684504313 with 267, 3519995546 with 237, 4711567524 with 206). The extra files per cell are probably separate skeleton pieces; not yet checked.
- IDs are 9 to 10 digits, for example the smallest 546925828 and the largest 7893125555. Saved sorted in `outputs/proofread_104_ids.txt`.
- First SWC lines are standard SWC columns (index, type, x, y, z, radius, parent); coordinates look like voxel units (x about 86,848).

Hypothesis (not yet tested): these 104 IDs appear as `base_neuron_id` or `neuron_id` in the synapse export. Test next.

Next: download one whole shard (about 197 MB, check size 197,344,616 bytes) and run `scripts/h01_idcheck.py` on it.

## 8. ID check on shard 0 (2026-11-18 ~18:59)

Commands (PowerShell): `curl.exe -o shard0.avro ".../exported/export000000000000"`, moved to Downloads, then
`python C:\Users\chisi\Downloads\h01_idcheck.py C:\Users\chisi\Downloads\shard0.avro` (script `scripts/h01_idcheck.py`, 104 IDs built in; output saved verbatim in `outputs/idcheck_out_shard0.txt`).

Confirmed (shard 0 only, 998,235 rows):
- The 104 proofread IDs match `neuron_id` far more often than `base_neuron_id`: as receiver 509 rows vs 13; as sender 9 rows vs 1. Decision: use `neuron_id` as the cell identifier.
- The proofread cells are almost entirely receivers: 509 receiver rows against 9 sender rows. 61 of the 104 cells appear as receivers, 9 as senders. Receiver class labels: DENDRITE 496, SOMA 11, UNKNOWN 2.
- Site-level `type` looks like a role flag, not excitatory/inhibitory: for these cells the receiver side is type 2 in 500 of 509 rows (and sender type 1 in 6 of 9); earlier stats gave sender 1 / receiver 2 in 96% of all rows. Hypothesis (supported, not proven): site `type` = pre/post role.
- Row-level `type` (1 or 2) is split 226/283 on receiver rows of the proofread cells. Which value is excitatory, if either, is still UNKNOWN. Not assumed.

Not measured: my "both ends in the 104" counter used `base_neuron_id` (0 rows). It was the wrong field, so it says nothing about links between the 104 cells using `neuron_id`. Rough expectation only (not a measurement): with 9 sender rows per shard, the 104-cell subgraph is very sparse.

Decision: do not build on the 104 cells alone. Next: stream all shards once and keep an edge list of every row with both `neuron_id`s present (script `scripts/h01_edges.py`: download one shard, write `edges_NNN.tsv.gz` and `somas_NNN.txt`, delete the shard; resumable; optional start/stop shard numbers). Then pick the subgraph (cells with a soma, ranked by connection counts) from the edge list.
Script test: logic checked with a stand-in reader on 3 fake rows (self-loop and missing-id rows dropped, soma id recorded, shard-not-found ends the run); not yet run on real data.

## 9. Edge-extraction test on shard 0 (2026-11-18 19:09)

Command: `python C:\Users\chisi\Downloads\h01_edges.py 0 1` (script `scripts/h01_edges.py`).
Output line: `shard 0: 197 MB, rows 998235, kept 962656, soma ids 4417, 115 s`.

Confirmed:
- Script runs on real data. 962,656 of 998,235 rows kept (96.4%); the rest have a missing `neuron_id` or sender == receiver.
- 4,417 distinct `neuron_id`s appear with class_label SOMA in this one shard (about 0.4% of rows).
- Time: 115 s per shard (download + parse). If there are about 166 shards: about 5.3 hours for the whole pass.
- Shard file deleted after processing (by design); outputs `h01_edges_out/edges_000.tsv.gz`, `somas_000.txt`, `progress_log.txt` in Downloads.

Decision: run the full pass (`python C:\Users\chisi\Downloads\h01_edges.py`, no arguments; shard 0 is skipped because its output exists).

## 10. Full edge pass started (2026-11-18 ~21:57)

Command: `python C:\Users\chisi\Downloads\h01_edges.py` (no arguments; shard 0 skipped).
First lines of output:
- 21:59:35  shard 1: 197 MB, rows 998694, kept 962960, soma ids 4306, 151 s
- 22:02:46  shard 2: 198 MB, rows 999623, kept 963851, soma ids 4463, 191 s

Observed: 151 to 191 s per shard (slower than shard 0's 115 s). At about 170 s per shard the remaining shards would take roughly 7 to 8 hours if there are about 166. Keep-rates stay near 96.4%. Total shard count still unknown (the run stops at the first missing shard).

## 11. Full edge pass finished (2026-11-19 03:41; user pasted the log at 10:11)

Output: `outputs/edges_progress_log.txt` (the full progress log for shards 1 to 165; shard 0 was run separately in section 9).

Confirmed:
- There are exactly 166 shards, numbered 0 to 165. The script asked for shard 166 and got 404 ("finished, last shard is 165"). This matches the third-party figure of 166.
- Rows in all shards: 166,216,068 (165,217,833 in shards 1-165 plus 998,235 in shard 0).
- Rows kept (both neuron_ids present, sender != receiver): 160,259,769 (96.4%). Kept share was 96.4% in every shard.
- Soma ids per shard: between 4,271 and 4,544 (shard-by-shard counts, not the union).
- Time: 5.74 hours of processing for shards 1-165 (about 2 min each), run 21:57 to 03:41. Shard 1-2 were slower (151 and 191 s); the rest about 122-129 s.
- Shards range 197-198 MB, no failed downloads in the log.

Next: `scripts/h01_build_graph.py` (reads the local edge files only): union of soma ids, keeps only edges with both ends in that set, and prints degree, weight and class-pair statistics. It also tests whether the row-level `type` behaves like a property of the sending cell (a per-sender share of type-1 rows that clusters near 0% or 100% would support excitatory/inhibitory; a spread-out share would not). Logic tested on synthetic files only.

## 12. Soma-cell graph v1 (2026-11-19 ~12:00)

Command: `python C:\Users\chisi\Downloads\h01_build_graph.py` (628 s for the edge pass). Outputs saved in `outputs/graph_v1/` (`summary.txt`, `graph_edges.tsv`, `soma_ids.txt`).

Confirmed (from summary.txt, 160,259,769 rows read):
- 73,896 distinct neuron_ids carry a SOMA class label at least once. This is far more than the roughly 16,000 neurons expected for H01, so a SOMA label at a synapse site is NOT a clean "this is a neuron" filter. (My earlier guess that the union would be about 16k was wrong.)
- Rows by (sender in set, receiver in set): neither 130,314,981; receiver only 28,739,844; sender only 864,436; both 340,508.
- Graph v1: 175,106 distinct directed edges from 340,508 synapse rows, on 37,196 cells. Edge weight (rows per edge): median 1, 75th percentile 2, 99th 13, maximum 338. Out-degree median 2 (max 110), in-degree median 3 (max 75).
- Class pairs on these rows: DENDRITE to DENDRITE 66,501; UNKNOWN to DENDRITE 64,156; AXON to DENDRITE 46,173; SOMA to SOMA 31,971; and others. Only about 14% are AXON to DENDRITE. SOMA to SOMA and DENDRITE to DENDRITE synapses are not expected biologically; likely label noise or merge errors in the segmentation (hypothesis).
- 53 of the 104 proofread cells are in the set; 0 edges between two proofread cells (as predicted); 637 edges touch one.
- Row-level `type` per sender (13,501 senders with at least 20 rows): share of type-1 rows by tenths: [27, 212, 420, 615, 959, 1590, 2544, 3618, 2684, 832]. Broad and single-humped around 70-80%. Not clustered at 0% and 100%, so `type` does not look like a clean property of the sending cell in this noisy set.

Additional analysis (run in the sandbox on graph_edges.tsv; not in the user's script, numbers reproducible from the file):
- Components (undirected): one giant component of 35,064 of 37,196 cells (832 components in all).
- Strongly connected: the giant strongly connected component has 18,447 cells and 142,429 edges (mean total degree 15.4). Reciprocal edges: 13.6% of edges.
- Edges with n_rows >= 2: 55,359 edges on 26,135 cells; n_rows >= 3: 28,398 on 19,738.
- Picking the top N cells by weighted degree gives poorly connected sets at small N: for N=100, 24 cells have no edge inside the set and the largest strongly connected piece is 43; for N=1000 it is 675 (all edges) or 262 (weight >= 2 edges). So hub-picking does not make a good recurrent network at small sizes.
- A connected "ball" grown from the best-connected seed inside the giant strongly connected core is denser: mean total degree 8.1 (N=100), 11.4 (N=1000), 18.3 (N=3000).
- Edge-level type check, edges with at least 5 rows (11,905 edges): share of type-1 rows in tenths: [197, 95, 147, 93, 188, 234, 468, 441, 1872, 8170]. Most strong edges are almost entirely type 1 and a small group (about 2%) is almost entirely type 2. Which value is excitatory is still not established.

Decisions (made 2026-11-19, to be confirmed with the user):
1. Do not use the SOMA label alone to define "cells". Add per-edge class-pair information (version 2 of the build script) so implausible rows can be dropped, then re-derive the cell set.
2. Subgraphs will be grown as connected regions inside the giant strongly connected core, not picked as top-N hubs.
3. Treat edges with a single synapse row as low confidence; run both with and without them.
4. Leave excitatory/inhibitory unassigned for now. Fallback: sign cells by a rule we state openly (for example the type with the larger share on a cell's outgoing synapses) and report results with a sign-agnostic control.

## 13. Version 2 of the graph builder written (2026-11-19 ~12:15)

Script: `scripts/h01_build_graph_v2.py`. Reads the same local edge files; writes to `h01_graph_out_v2\` (does not overwrite version 1). Expected run time 15 to 25 minutes (version 1: 628 s for the edge pass).

What is new compared with version 1:
- Per-edge row quality from the class pair: good = sender AXON and receiver DENDRITE/SOMA/AIS; unk = involves UNKNOWN; bad = everything else (for example DENDRITE to DENDRITE, SOMA to SOMA).
- `cells.tsv`: per soma-labelled id, rows as sender and as receiver broken down by class label (to judge which ids behave like real neurons).
- Type-1 share by class pair over ALL 160 million rows. Test idea: in cortex, synapses onto cell bodies and the axon initial segment (AXON to SOMA, AXON to AIS) are mostly inhibitory, so a strong type bias there versus AXON to DENDRITE would point to which type value is inhibitory. This is a test, not an assumption.
- Graph variants with their giant strongly connected component (all edges; n_rows >= 2; n_good >= 1; n_good >= 2; n_bad == 0; n_bad == 0 and n_rows >= 2).
- Per-sender and per-edge type-1 share on good rows only.

Tested only on synthetic edge files (3 files, 23,761 rows); it ran and wrote all three outputs. Not yet run on real data.

## 14. Graph builder version 2 on real data (2026-11-19 ~12:44)

Command: `python C:\Users\chisi\Downloads\h01_build_graph_v2.py` (882 s for the edge pass). Outputs saved in `outputs/graph_v2/` (`summary.txt`, `cells.tsv`, `graph_edges.tsv`).

Confirmed (from summary.txt and the two tables):
- Share of type-1 rows by class pair, over all 160,259,769 rows: AXON to DENDRITE 0.366 (132,914,844 rows); AXON to SOMA 0.795 (526,143 rows); AXON to AIS 0.852 (263,956 rows); UNKNOWN to SOMA 0.829. Soma and axon-initial-segment targets are strongly type 1; dendrite targets are mostly type 2.
- The release documentation (Released Data page) says synapse points carry an "excitatory or inhibitory type" (confirmed in the PDF text). It does not say which number is which (not found in the two saved pages).
- Evidence on the number mapping (hypothesis, supported): in cortex, synapses on cell bodies and on the axon initial segment are mostly inhibitory, and synapses on dendrites mostly excitatory. The data match that if type 1 = inhibitory and type 2 = excitatory. Caveat: 36.6% type 1 on dendrites is a higher inhibitory share than the usual textbook figure, so the classifier or the labels may be imperfect. Not proven.
- Per sender with at least 20 good rows (2,130 senders), type-1 share by tenths: [599, 605, 277, 159, 70, 55, 38, 37, 72, 218]. About 57% of these senders are below 20% type 1 and about 10% above 90%. Consistent with two cell populations plus a mixed middle.
- Cell profile: 73,896 soma-labelled ids. Rule "at least 5 rows labelled SOMA and at least 20 rows as DENDRITE receiver" selects 16,599 ids. This is close to the roughly 16,000 neurons H01 is described as containing (a third-party page said 16,087), which supports using this rule as the neuron definition (supported, not verified). Other thresholds on the same data: soma>=5 and dend>=50: 15,457; soma>=10 and dend>=50: 13,559; soma>=3 and dend>=20: 18,024; soma>=1 and dend>=20: 20,019.
- Graph variants over all 37,196 soma-labelled cells (giant strongly connected component, SCC): all edges 175,106 edges, SCC 18,447; rows>=2 55,359 edges, SCC 7,290; good>=1 39,071 edges, SCC 2,924; good>=2 6,584 edges, SCC 7; no bad rows 112,738 edges, SCC 13,993; no bad rows and rows>=2 33,136 edges, SCC 4,600.

Additional analysis (sandbox, from the saved tables; reproducible):
- Restricting to the 16,599 neurons: 87,706 edges among 14,683 neurons (mean total degree 11.9; giant SCC 10,133 neurons). With rows >= 2: 23,078 edges, SCC 2,319. With no bad rows: 42,853 edges, SCC 5,644. With good >= 1: 30,752 edges, SCC 2,755. With no bad rows and rows >= 2: 9,858 edges, SCC 123. Edge weight (synapse rows): median 1, mean 1.69.
- Only 2,421 of the 16,599 neurons have at least 20 outgoing rows labelled AXON. Most neurons' outgoing synapses carry other labels (UNKNOWN or DENDRITE on the sender side). Hypothesis: axon labelling is incomplete, so the class filter removes many real connections. This is a trade-off, to be handled as an experimental factor (edge-quality tier) and not decided once and for all.

Decisions (to confirm with the user):
1. Neuron set = rule soma>=5 and dendrite-receiver>=20 (16,599 cells).
2. Edge-quality tiers: A = all edges among neurons; B = no bad class pairs; C = rows >= 2. Run the controller experiments on at least A and B.
3. Excitatory/inhibitory: working assumption type 1 = inhibitory, type 2 = excitatory, labelled as an assumption everywhere; sign each neuron once (one sign for all its outgoing connections) from the type counts of its outgoing rows; include a sign-agnostic control (all weights positive) and a sign-flipped control.
4. Next: `scripts/h01_cell_signs.py` (about 15 minutes) writes per-neuron outgoing type counts.

## 15. Neuron signs (2026-11-19 ~13:00)

Command: `python C:\Users\chisi\Downloads\h01_cell_signs.py` (377 s). Outputs saved in `outputs/graph_v2/` (`neuron_signs.tsv`, `neuron_signs_summary.txt`).

Confirmed:
- 16,599 neurons by the rule soma>=5 and dendrite-receiver>=20. 364 have no outgoing rows at all.
- Using ALL outgoing rows (any class labels), the type-1 share per neuron is one broad hump around 70-80% (tenths: [23, 185, 398, 637, 976, 1720, 2581, 3260, 2263, 855], 12,898 neurons with at least 10 rows). Most of those rows have non-AXON sender labels, where type is not a clean per-neuron property. Signing neurons from all rows would label most of them inhibitory. Do not do this.
- Using only good rows (AXON to DENDRITE/SOMA/AIS), 3,365 neurons have at least 10 such rows. Their type-1 share by tenths: [922, 830, 472, 266, 124, 119, 81, 72, 123, 356]: a large group near 0-30% and a distinct group near 90-100%, with a thinner middle.

Additional analysis (sandbox, from `neuron_signs.tsv`; reproducible):
| good rows at least | neurons | type-1 share >= 70% | <= 30% | mixed |
| 3 | 6,491 | 1,110 (17.1%) | 3,724 (57.4%) | 1,657 |
| 5 | 5,033 | 830 (16.5%) | 3,101 (61.6%) | 1,102 |
| 10 | 3,365 | 551 (16.4%) | 2,261 (67.2%) | 553 |
| 20 | 1,910 | 304 (15.9%) | 1,390 (72.8%) | 216 |
| 50 | 579 | 104 (18.0%) | 425 (73.4%) | 50 |
- As the number of good rows per neuron grows, the mixed share shrinks (about 25% to 9%) while the high-type-1 share stays near 16-18%. This is what two cell populations (Dale's law) with measurement noise would look like. A group of 16-18% fits the usual figure for inhibitory neurons in cortex (about 15-25%), which supports type 1 = inhibitory. Still a hypothesis, not proof.
- 43,230 of the 87,706 edges among neurons (49%) start at a neuron whose sign is known (at least 10 good rows). Those 3,365 neurons have on average 12.8 outgoing edges in the graph against 3.4 for the others; they are the hubs with well-reconstructed axons. 12,476 neurons have at least one outgoing edge.

Decisions (to confirm with the user):
1. Sign rule: sign each neuron from its good rows only, and only if it has at least 10 of them (type-1 share >= 0.5 gives inhibitory, otherwise excitatory). Treat the rest as sign unknown.
2. Sign policy is an experimental factor: (a) "known": data sign where known, excitatory otherwise; (b) "prior": unknown signs drawn at random with 17% inhibitory (fixed seed); (c) "abs": all weights positive; (d) "flip": all known signs reversed (checks that the assumption type 1 = inhibitory is not driving results).

## 16. First subgraphs cut (2026-11-19 ~13:20)

Script: `h01_data/make_subgraph.py` (standard library only). Inputs: `outputs/graph_v2/` tables. Grows one connected region inside the giant strongly connected core of the chosen edge tier: starts at the best-connected neuron (seed 0) or a random one of the 50 best-connected (seeds 1, 2), then repeatedly adds the neuron most strongly tied to the set so far. Signs: from `neuron_signs.tsv` (at least 10 good rows; type 1 share >= 0.5 gives inhibitory); neurons without a data sign are written as +1 with sign_source "unknown" so the policy can be chosen later.

Command pattern: `python make_subgraph.py --size N --tier A|B|C --seed S --out ../subgraphs` for N in 100, 300, 1000, 3000; tiers A (all edges among neurons), B (no implausible class pairs), C (at least 2 synapse rows); seeds 0, 1, 2. 36 subgraphs, 3 files each (nodes, edges, stats) in `subgraphs/`.

Confirmed (seed 0 shown; neurons, edges, mean in+out degree, strongly connected component inside, share inhibitory among data-signed neurons):
| tier | N | edges | mean degree | SCC | inhibitory |
| A | 100 | 638 | 12.8 | 94 | 13.8% |
| A | 300 | 2,981 | 19.9 | 287 | 12.4% |
| A | 1000 | 10,697 | 21.4 | 975 | 13.3% |
| A | 3000 | 30,131 | 20.1 | 2,931 | 18.5% |
| B | 100 | 501 | 10.0 | 78 | 19.0% |
| B | 300 | 1,794 | 12.0 | 266 | 17.7% |
| B | 1000 | 6,340 | 12.7 | 925 | 18.0% |
| B | 3000 | 16,100 | 10.7 | 2,902 | 20.5% |
| C | 100 | 283 | 5.7 | 51 | 19.7% |
| C | 300 | 881 | 5.9 | 247 | 23.8% |
| C | 1000 | 2,820 | 5.6 | 888 | 22.0% |
| C | 3000 | 5,609 (only 2,319 neurons available) | 4.8 | 2,319 | 20.9% |
- Tier core sizes: A 10,133 neurons, B 5,644, C 2,319 (so size 3000 is not possible in tier C; it returns all 2,319).
- Growing a region gives much denser graphs than picking hubs (for example tier A, 100 neurons: 94 of 100 in one strongly connected component, against 43 of 100 for the top-hub pick in section 12).
- Only 67% of neurons in the 1000-neuron tier A graph have a data-derived sign (80% at 100 neurons, 49% at 3000).

## 17. Webots project, milestone 1: crawl under a table (2026-11-19 ~13:30)

User decisions: make a new world and a new DARwIn-OP environment (from their `walk.wbt` and an empty controller named `h01_dataset_policy`); multiple Python files allowed; task = DARwIn-OP crawls under a table; push-a-box idea to be added later if the crawl works (pickup dropped: the robot has no hands).

Files (all in `webots/`):
- `worlds/crawl_table.wbt`: R2025a world based on `walk.wbt` (same protos, floor, background, time step 16 ms). Robot placed face down (rotation 0 1 0 1.5708) at x = -0.9, `supervisor TRUE`. Table: top 0.8 m x 0.9 m x 0.04 m with its underside 0.25 m above the floor, x from 1.0 to 1.8, four legs at the corners (gap between legs 0.8 m). Green goal line at x = 2.0.
- `controllers/h01_dataset_policy/`: `config.py` (mode, scenario, table numbers, episode times, search settings), `robot_io.py` (devices, joint limits, reset, state), `crawl_gait.py` (open-loop sine gait, 27 numbers in [-1, 1]), `episode.py` (run and score one episode), `search.py` (evolution strategy, writes `results/`), `h01_dataset_policy.py` (entry: preview / test / search).

Design choices:
- Joint sign directions are unknown for a face-down robot, so every gait amplitude and offset is a signed number scaled to the real joint limits; the search finds the directions (same idea as the sign search in the walking controller).
- Score = forward distance minus 2 x sideways drift minus 0.3 x heading error, minus 1 if it rolls off its front, plus 1 for passing the goal line. No contact sensor for the table: the table physically blocks a body that is too high, so a jammed robot simply scores little. (A direct table-contact measure is not available in this setup; noted as a limitation.)
- Joints other than shoulder, elbow, hip, knee and pelvis roll stay at neutral.
- The face-down orientation (rotation 0 1 0 pi/2, head toward +x) is derived by hand from the right-hand rule, not yet checked in Webots; the preview mode prints `prone_z` and the heading so it can be checked.

Test status: the controller logic ran against a stand-in Webots module (search loop, csv, best json, test and preview modes all executed). It has not been run in Webots.

## 18. Evaluation scope extended: box pushing (2026-11-19 ~13:15)

User request: also evaluate with loco-manipulation of a box by pushing it (in addition to crawling under the table). Not built yet; to be added after the first Webots preview confirms the robot's size and orientation, so the box size and table gap are set from real numbers.

Planned task set for comparing the H01 reservoir against controls: (1) open-floor crawl, (2) crawl under the table, (3) push a box along the floor to a goal zone, (4) stretch: push the box through the gap under the table while crawling behind it.
Planned push scoring: how far the box moved toward the goal, how far it drifted sideways, and whether the robot stayed behind it; box pose read through the supervisor (`getFromDef("BOX")`). Box: a light cube, lower than the robot's face-down height so the head, shoulders or arms meet it, and lower than the 0.25 m table gap for task 4.

## 19. First Webots preview and package v2 (2026-11-19 ~13:20 to 13:45)

Project name chosen by the user: ExoMind (a nod to Destiny). Used as the project/paper name from now on; folder and controller names unchanged.

Preview output from the user's Webots run (R2025a, python.exe controller, `MODE = "preview"`), saved as the console text in this entry:
- Controller started and read all devices; the joint limit table printed (20 joints). Examples: ShoulderR [-3.14, +3.14], ShoulderL [-3.14, +2.85], ArmLowerR [-1.65, +1.16], ArmLowerL [-1.18, +1.63], PelvR [-1.01, +1.01], LegUpperR [-1.77, +0.45], LegUpperL [-0.50, +1.68], LegLowerR [-0.02, +2.25], LegLowerL [-2.25, +0.03], AnkleR [-1.24, +1.38], Head [-0.36, +0.94]. Left and right limits are mirror images for most joints (so left and right joints use opposite signs for the same motion).
- After 2 s at all-zero joints: root x -0.991, y -0.031, z 0.166 m; prone_z -0.93; heading -0.03 rad; gyro 0; accelerometer [0.0, -0.23, 0.09] (scaled).
- Screenshot (user): the table on the left, the robot on the right, lying down with the head toward the table.

Interpretation (mine, partly a hypothesis):
- Confirmed: orientation is as designed (face down, head toward +x): prone_z -0.93 and heading about 0.
- Not flat: prone_z -0.93 means the body axis is about 22 degrees off vertical, and the root origin sits 0.166 m above the floor. Hypothesis: with every joint at 0 the feet or arms prop part of the body up. The root origin is near the hips, not the chest, so root height does not equal body thickness. The robot's true face-down height (top surface above floor) has NOT been measured.
- The user doubted the robot would clear the table (gap 0.25 m). It cannot be judged from the root height alone.

Changes in package v2 (`h01_webots_m2.zip`):
1. The table height is now set by the controller at every reset (`TABLE_CLEARANCE` in `config.py`, starting at 0.30 m instead of 0.25). Table top and legs were given DEF names; legs are 0.6 m tall boxes with tops flush with the top plate, so any gap from about 0.05 to 0.5 m is possible. This lets us find the lowest gap the robot gets through and report it as a result (a candidate main metric).
2. Preview now includes a pose probe: it tries ankle poses, then hip poses, holds each for about 2 s, prints root height and prone_z, and names the flattest face-down pose. That dictionary goes into `REST_POSE` in `config.py` and becomes the neutral pose for the whole project.
3. `reset()` takes the table gap and the rest pose.
Test status: logic checked against a stand-in Webots module (preview with pose probe, table moves, no errors); not yet rerun in Webots.

## 20. Pivot: crawling replaced by pushing a box (2026-11-19 ~13:26 to 13:50)

User decision (13:26): switch the Webots task from crawling to loco-manipulation by pushing a box, because the user already did a crawling project and has not done a manipulation project in Webots.

Consequences and choices:
- The crawl package (`webots/`, packages m1 and m2) is archived, not deleted. Its preview run showed the robot face down but not flat (prone_z -0.93, root 0.166 m). Nothing from it is used now.
- The new task starts from the user's existing standing walker. `darwin_rl.py` (v20.3) is included unchanged; the new controller subclasses its `Walker` class and loads the saved calibration and best gait from `darwin_state.json` (keys read: `sg`, `z_stand`, `z_floor`, `crouch`, `lean0`, `gait`). The user must copy their own `darwin_state.json` into the controller folder.
- Package `h01_webots_push_m1.zip`: `worlds/push_box.wbt` (robot at the origin facing +x; box 0.2 x 0.4 x 0.25 m, mass 0.3 kg, placed 0.55 m ahead; green goal line at x = 0.95, i.e. 0.40 m of pushing; time step 16 ms as before), and `controllers/h01_dataset_policy/` with `config.py`, `pusher.py` (class Pusher(Walker): box placement and reading through the supervisor, one scored episode), `h01_dataset_policy.py` (modes "check" and "baseline", csv and log output), and `darwin_rl.py`.
- Score per episode: box distance moved along x, minus 0.5 x sideways box drift, minus 0.3 x box turning, minus 1 if the robot fell, minus 0.5 if the box tipped; success = box moved at least 0.40 m, robot upright, box not tipped.
- Plan for the H01 reservoir: it will output small corrections added to the 12 leg joint targets of the saved gait (the same residual idea as the earlier TD3 residual policy), plus later arm posture changes. Inputs: joint angles, gyro, accelerometer, torso tilt, and the box position relative to the robot. The box position will come from the supervisor (privileged information, to be stated as such in the paper; camera-based sensing is a possible later step).
- Open choices: arm posture during pushing (baseline keeps the walking arm pose), box mass (0.3 kg is a first guess; friction force is roughly 3 N), training method for the readout (not decided).
- The Webots world file keeps `selfCollision TRUE` as in the user's `walk.wbt`; the earlier calibration may have been made with a different setting (not known).

Test status: the glue code ran against a stand-in walker (both modes, csv and log written, no errors). The real `darwin_rl.py` imports cleanly (checked against a stand-in Webots module). Nothing has been run in Webots.

## 21. Push-box changes after the first Webots look (2026-11-19 ~13:37 to 13:55)

User feedback (13:37, with a screenshot of the scene): the box started much too far away (the user moved it by -1.20 m by hand), Darwin should push with the arms raised, and the box was too small; the user suggested the 0.6 m cube from the Webots object library.
What the screenshot shows: the orange box stands right beside the robot, with the robot touching its side, and the green goal line is off to the right. The exact numbers and the console output of that run were not available, so the cause of the original distance problem is not known. One real cause is likely: the controller places the box at every reset, so a manual move in the scene tree is overwritten at the next episode start.

Changes (package `h01_webots_push_m2.zip`):
1. Box: 0.6 m cube (same size as the Webots wooden box), built as a plain Solid with a controllable mass (0.5 kg for now; the main difficulty knob). The box and the goal line are now placed by the controller at every reset from `config.py`: `BOX_FACE_X` = distance from the robot's root to the box's FRONT face (default 0.20 m, so the arms reach it within a step or two), `GOAL_DX` = 0.40 m, goal line drawn at the front face's target. Manual moves in the scene tree no longer matter.
2. Arms: new `PUSH_ARMS` setting and an `arms_scan` mode. The joint directions for raising the arms forward are unknown, so the scan tries 18 postures (shoulders both directions, same-sign and mirrored, elbows bent as in walking or straight), 2 episodes each against the box, ranks them by mean score (box distance, drift, falls) and saves the best to `results/best_arms.json`; `baseline` then uses it (`PUSH_ARMS = "best"`). Arms are held fixed (no swing) when a posture is set.
3. `LEAN_EXTRA` (rad) added to the saved forward lean, in case raised arms make the robot tip.
Known risk (hypothesis): raising the arms moves the centre of mass forward and the saved calibration was made with the walking arm pose, so some postures may make it fall. The scan reports falls, so this shows up in the results.
Test status: logic checked against a stand-in walker (all three modes, scan to best file to baseline); nothing run in Webots yet.

## 22. User's arm pose and box friction (2026-11-19 16:53)

User report (by looking at the arms in Webots): the arms-raised pose has ShoulderL = 3.00 and ShoulderR = 3.00 (same sign on both sides, in rad), with ArmLowerL and ArmLowerR "right in the middle value" of their ranges. From the joint limits printed earlier: ArmLowerR [-1.65, +1.16] gives a middle of -0.245 rad; ArmLowerL [-1.18, +1.63] gives +0.225 rad. ArmUpper was not reported; assumed 0 (hypothesis, to be confirmed).
Notes (mine):
- A shoulder angle of 3.0 rad is about 172 degrees, so the arms point nearly straight up, not forward. With the robot's shoulder at about 0.33 m, hands would reach roughly the top edge of the 0.6 m cube. Pushing at the top edge of a tall cube tips it over unless its friction is low (for a cube 0.6 m wide and tall, pushing at the top tips before it slides if the friction coefficient is above about 0.5; the Webots default is 1, so this would be the case).
Changes (package `h01_webots_push_m3.zip`):
1. `config.py`: `PUSH_ARMS` default is now the user's pose: shoulders (3.0, 3.0), upper arms (0, 0), elbows "mid" (computed from each elbow's joint limits at run time).
2. `arms_scan` now tries shoulders 3.0, 2.4, 2.0, 1.6, 1.2 (both the same sign) x elbow mid-range or straight (10 postures, 2 episodes each), so we can compare the user's pose with lower arm heights.
3. World: the box has contact material "box" and `WorldInfo.contactProperties` sets friction 0.4 between "box" and "default" (the floor and the robot). This makes the box slide before it tips when pushed high, and lowers the push force needed (about 2 N for 0.5 kg). Easy to revert: delete the `contactProperties` block and the `contactMaterial` line.
Test status: stand-in test only (baseline and scan ran; elbow "mid" resolved without error). Not run in Webots.

## 23. ArmUpper values (2026-11-19 16:56)

User report: ArmUpperL = 0.77 and ArmUpperR = -0.68 rad. These are the two joints' limits (ArmUpperL [-2.25, +0.77], ArmUpperR [-0.68, +2.30]), so the upper arms are at their stops, on the same side as the walking pose (+0.6 / -0.6) but further. Set in `config.py` as (-0.67, 0.76) (just inside the limits so the controller's 5 mrad safety margin does not change them), and used in every `arms_scan` posture. Package `h01_webots_push_m4.zip`. Test status: syntax check only for this small change; the pipeline was tested on a stand-in earlier (section 22).

## 24. Robot started ahead of the box; drift metric (2026-11-19 17:2x)
Report (user): robot starts ahead of the box, box not in front at start, some drift; user's edited config.py has PUSH_ARMS shoulders (1.5, -1.5), upper (-0.67, 0.76), elbow "mid".
Cause: Walker.reset_to pins the robot at `home_xy`, read from the saved world file. The user's console printed `[init] world start pose: x=3.525 y=0.269`, while the box and goal were placed in absolute coordinates near the origin.
Fix: `ROBOT_HOME = (0.0, 0.0)` in config.py overrides home_xy; box and goal line are placed relative to it (box centre = home + face_x + 0.3). Also: `density -1` with `mass 0.5` in the world (silences the density/mass warning); `robot_yaw` added to metrics and csv (quantifies drift together with robot_dy); arms_scan candidates now (R,L) shoulder pairs (1.5,-1.5), (1.2,-1.2), (1.8,-1.8), (3.0,3.0), (3.0,-3.0) x elbow mid/straight.
Test: stand-in walker only (no Webots here): box centre at (0.5, 0, 0.302) with home (0,0); csv has robot_yaw column. Real results pending from the user (check, arms_scan, baseline).
Package: h01_webots_push_m5.zip.

Follow-up (17:14): user wants the box at least 0.5 m from the robot. BOX_FACE_X set to 0.5 (box front face 0.5 m ahead of ROBOT_HOME); package m6.

## 25. First baseline result in Webots (2026-11-19 17:20, user run, m6, BOX_FACE_X 0.5)
Open-loop saved gait, arms (1.5,-1.5)/(-0.67,0.76)/mid, 5 episodes x 400 steps (12.8 s), box start 0.5 m ahead, goal 0.4 m past the box face.
ep  fitness  box_dx  box_dy  box_yaw  robot_dx  robot_dy  robot_yaw  fell  success
0   0.402    0.434   0.021   0.070    0.208     0.143     1.809      no    yes
1  -0.473    0.565   0.015   0.101    0.642     0.484    -3.017      yes   no
2   0.201    0.238  -0.025   0.082    0.064     0.285     1.810      no    no
3  -0.598    0.443  -0.018   0.108    0.971    -0.647    -0.530      yes   no
4   0.480    0.492   0.005   0.031    0.837     0.661     2.448      no    yes
Contact at 2.4-2.6 s every time; no tipping; box pushed nearly straight (|box_dy| <= 0.025). 2 of 5 successes, 2 of 5 falls. Robot yaw ends far from 0 (up to ~3 rad): after contact the robot veers and turns, and keeps walking (not a controller bug, the gait is open loop). User observation: after the box crosses the line the robot walks off in another direction.
Baseline conclusion: pushing works physically; the baseline fails mainly through falls and heading drift. These are the targets for the reservoir (heading and balance feedback). Episode-to-episode spread is large with identical settings (physics is chaotic after contact), so comparisons need many episodes per controller.

## 26. Arm scan and no-box drift (2026-11-19 17:24, user run, m6)
Files from the user's results folder: run_log.txt (append-only, contains earlier test runs too), check_no_box.csv, baseline.csv, arms_scan.csv, best_arms.json.
No-box drift (check, box far, open-loop saved gait): ep1 robot_dx 1.28, dy -0.15, yaw +1.48; ep2 dx 1.39, dy +0.13, yaw +1.16 (12.8 s). Ep0 "fell" after 0.03 s with no motion: first-episode start-up artefact, excluded. So the gait turns about +1.2 to +1.5 rad (70-85 deg) per 12.8 s on its own.
Arm scan (2 episodes per pose, mean fitness): R+1.2/L-1.2 mid 0.976 (0 falls, box_dx 1.23 and 0.77); R+1.2 straight 0.331; R+1.8 straight 0.238; R+1.8 mid -0.014 (1 fall); R+3/L-3 straight -0.433; R+3/L-3 mid -0.581; R+3/L+3 mid -0.589; R+1.5 mid -0.727 (2/2 fell); R+1.5 straight -0.727 (2/2 fell); R+3/L+3 straight -0.854. Best saved to results/best_arms.json.
Caveats: n=2 per pose; pose (1.5,-1.5) fell 4/4 in the scan but 2/5 in the earlier baseline, so noise is large. Plan: confirm with 20 episodes per pose (1.2 mid vs 1.5 mid) via MODE "baseline" before fixing the arm pose for all later experiments.

## 27. Baseline, 20 episodes, arms R+1.2/L-1.2/upper (-0.67,0.76)/elbow mid (2026-11-19 17:28, user run, m6)
(user's console output; episodes 0-19, box face 0.5 m, goal 0.4 m past the face, 400 steps)
Results (n=20): 11 successes (55%), 5 falls (25%; eps 0, 5, 7, 11, 13), 4 slow runs that did not fall but pushed the box only 0.30-0.38 m. Mean fitness 0.31 (median 0.41, sd 0.51); mean box_dx 0.59 m (0.30 to 1.09). Box always pushed straight (|box_dy| <= 0.022), never tipped (0 of 20). Robot yaw at the end ranges from -2.5 to +3.0 rad: heading is uncontrolled and differs from the no-box check (+1.2 to +1.5), because contact pushes the robot around.
Decision: use arms (1.2, -1.2) mid as the fixed pose for all later experiments. The 1.5 comparison run (20 episodes) was not made; the scan ranking and these 20 episodes are the evidence.

## 28. Reservoir, policy, trainer, stop rule (2026-11-19 17:35, package m7)
Files (webots_push/controllers/h01_dataset_policy/): reservoir.py, policy.py; changed pusher.py, h01_dataset_policy.py, config.py; webots_push/subgraphs/ holds tiers A, B, C x sizes 100, 300, 1000 x seeds 0-2 (nodes_, edges_, stats_).
Reservoir: x <- (1-leak) x + leak tanh(W x + W_in u); W from the subgraph, entry = sign of the sending neuron (Dale's law) x sqrt(synapse rows), rescaled so the dominant eigenvalue magnitude (power iteration) is rho = 0.9; leak 0.3; no bias. Input wiring: each of 12 inputs reaches a random 25% of nodes with uniform weights in +-0.5, same recipe and seed for every graph.
Inputs (12): gyro xyz, accelerometer xyz (z minus 1 g), gait phase sin/cos, gap to the box's near face and its sideways offset in the robot frame, robot heading relative to its start, sideways offset from its start line. The last four come from the supervisor (privileged; to be stated in the paper).
Outputs (3, tanh, linear readout with bias): steering (the walker's own Walker.turn knob, max 0.6), extra ankle lean (max 0.08 rad), stride multiplier (1 +- 0.25 on hip and knee amplitude). Zero readout = exactly the saved open-loop gait.
Controls (same code, set by RES_GRAPH): h01; shuffle (double edge swaps, same degrees, weights, signs); er (same size and edge count, random endpoints, weights and signs permuted); esn (same size and mean degree, Gaussian weights, a random sign per edge). Sign policies RES_SIGN: data (unknown -> excitatory), random17, allpos, flipped. Edge strength RES_WEIGHT: sqrt, binary, count.
Task changes: STOP_AT_GOAL ends an episode when the box has moved 0.40 m; box start offset sideways uniform +-0.10 m by seed (episode i of every evaluation uses seed i, so all controllers see the same 30 starts); training uses other seeds (100000+).
Score (training and comparison): success = 1 + 0.5 (1 - t_goal/T) - 0.5|box_dy| - 0.3 box_yaw; failure = 0.6 box progress + 0.2 approach - 0.3 min(1,|robot_lat|) - 0.5|box_dy| - 1 if fell - 0.5 if tipped. The old fitness is still logged in the code but not written to csv.
Trainer: evolution strategy on the readout only (3 x (N+1) numbers), antithetic pairs, centred ranks, 16 candidates, 2 episodes per candidate with common box offsets, sigma 0.05, step 0.05, 60 generations by default; the centre is saved every generation (results/readout_<tag>.json) and resumes after a restart. Tag = graph_subgraph_sign_weight_seed.
Test (stand-in walker with a toy drift of +0.12 rad/s, not Webots): zero readout reproduces the baseline numbers exactly; a random readout changes the heading; the trainer raised mean score from 0.22 to 1.22 and the evaluation gave 4/4 successes. This tests the code only, not the robot. Reservoir step cost 0.6 ms for 1000 neurons (10.7k edges) on this machine; negligible next to the simulation.
Observation on the subgraphs: with sign "data" the share of inhibitory neurons is only 9-11% (neurons without enough evidence default to excitatory).

## 29. Project name (2026-11-19 17:43)
User decision: the project/paper is called ExoHominid (ExoMind was taken). Keep the "human" framing: recurrent wiring is a subgraph of the H01 human cortex connectome. Code and zip names keep the old "exomind"/"push" labels for now. Neuron counts for the record: 73,896 soma-labelled ids -> 16,599 neurons by the rule (>= 5 SOMA rows, >= 20 DENDRITE-receiver rows) -> 14,683 neurons with at least one edge in tier A -> giant strongly connected core 10,133; subgraphs cut so far: 100, 300, 1000 and 3000 neurons (tiers A, B, C; 3 seeds each; the robot package contains 100, 300, 1000).

## 30. Rename in code (2026-11-19 17:46)
Replaced 'ExoMind' by 'ExoHominid' in h01_dataset_policy.py, reservoir.py, README.md and push_box.wbt (header and world title). File and folder names unchanged. Package m8.

## 31. First Webots results: baseline (30 ep), res_check, training of the H01-300 reservoir (2026-11-19 17:50-18:26, user run, m8)
Baseline (stop rule on, arms R+1.2/L-1.2 mid, box offsets +-0.10 m by seed 0-29): SUMMARY n=30 success 18/30 (60%), falls 3, mean score +0.830, mean |box_dy| 0.007, mean time to goal 5.1 s. Failures: 3 falls (eps 0, 9, 25) and 9 slow runs where the robot veered (robot_yaw about -1.5 to +2.6) and the box moved only 0.27-0.39 m by the end of 12.8 s. Ep 0 fell after 0.03 s with the box untouched: a start-up artefact (first reset after the world was saved with the robot lying down; the pose is stale for the first step). Fix in m9: the fall check is skipped for the first 3 steps of an episode.
res_check (A_300_s0, h01, signs data, sqrt weights): 300 neurons, 2981 edges, 27 inhibitory (9%), raw spectral radius 7.14, 0.2 ms per step. Zero readout: 2/5 success (same distribution as the baseline: the 5 episodes use the same box offsets, so the values differ from the baseline only by the simulation's own run-to-run variation). Random readout: 0/5 success, 3 falls: corrections change the walk and can break it.
Training (ES, 16 candidates x 2 episodes, sigma 0.05, lr 0.05, 60 generations, seed 0): 21 s per generation at first, 33 s later; total 33 min. Mean candidate score rose from -0.7 (gens 2-8) to about +1.0-1.25 (gens 11+); best candidate at or near the score ceiling 1.25-1.26 from gen 10 on. The centre score (2 episodes) was often +1.25 but fell to +0.3 or -0.9 in about 9 of the generations after gen 10 (13, 14, 20, 26, 27, 41, 46, 51, 52): unstable or noisy. Physics warnings "physics step could not be computed correctly" appeared in generations 1, 2 and 14-15 (probably large corrections early in training and in the collapsed generations). Final centre at gen 60: +1.208. Saved: results/readout_h01_A_300_s0_data_sqrt_seed0.json (last generation).
Not yet shown: a 30-episode evaluation of the trained readout on the same box offsets as the baseline (mode eval). The training scores come from different offsets (seeds 100000+) and 2 episodes, so they do not yet prove an improvement over the baseline.
m9 changes: fall check skipped for the first 3 steps; the trainer also stores the best centre seen (best_theta, best_generation, best_centre) and eval can use it (config.EVAL_WHICH = "last" or "best").

## 32. Evaluation of the trained H01-300 reservoir (2026-11-19 18:34, user run, m9 code, readout "last" = generation 60)
Same 30 box offsets (seeds 0-29) as the baseline. SUMMARY: n=30 success 26/30 (87%), falls 4 (eps 10, 13, 16, 28), mean score +1.000, mean |box_dy| 0.003, mean time to goal 6.9 s.
Baseline on the same offsets: success 18/30 (60%), falls 3 (one of them the start-up artefact in ep 0), mean score +0.830, mean |box_dy| 0.007, mean time to goal 5.1 s.
Reading: the gain comes from the 9 slow, veering baseline runs, which are gone (all successful runs of the reservoir end with robot yaw within about +-0.7 rad, mostly within 0.15); falls did not go down (4 vs 3), and the trained policy is slower (6.9 s vs 5.1 s to the goal, the score's time bonus is small). Fisher exact test 26/30 vs 18/30: p printed in the reply; this is a single training run (one seed), one graph, no controls yet, and one 30-episode evaluation: it shows the readout learned to steer, not yet that H01 wiring is the reason.
Next: same training and evaluation for the controls (shuffle, er, esn), the sign policies, and several seeds, with identical budget and the same 30 evaluation offsets.

## 33. Queue mode (2026-11-19 18:41)
Added MODE "queue" (h01_dataset_policy.py: tag_now, read_eval_csv, queue_runs, queue_report, queue; config.py: QUEUE_SEEDS, QUEUE_SUBGRAPH_BY_SEED, QUEUE_SUBGRAPH, QUEUE_CONDITIONS). For each seed k in QUEUE_SEEDS and each (graph, sign) in QUEUE_CONDITIONS: set RES_GRAPH, RES_SIGN, RES_SEED = k and RES_SUBGRAPH = A_300_s<k> (so seed k is a different H01 region and every control of seed k is built from the same subgraph), train (resumes if interrupted), evaluate on the 30 fixed box offsets, and rewrite results/queue_summary.csv (one row per finished run) and log a table by condition. Order is round-robin over seeds. Finished evaluations (30 rows in results/eval_<tag>.csv) are skipped, so the earlier h01 seed-0 run is reused and the queue can be stopped and restarted.
Default queue: seeds 0,1,2 x (h01/data, shuffle/data, esn/data, er/data, h01/random17, h01/allpos) = 18 runs, 17 still to do at about 33 min each, roughly 9.5 hours.
Tested with the stand-in walker on 100-neuron subgraphs (2 seeds x 3 conditions, 2 generations): all runs trained, evaluated, tabulated; a second call skipped everything.
Note: the seed-0 h01 run was trained with the m8 code (before the first-three-steps fall-check change); later runs use the m9 pusher.

## 34. Paper figure of the subgraph and controls (2026-11-19 18:55)
Script figures/fig_graph.py (usage: python fig_graph.py <subgraph name> [out dir]); uses the same reservoir.py loaders and control builders as the robot code. Output: figures/fig_h01_graph_<name>.png/.pdf/.svg for A_300_s0 and A_100_s0. Panels: (a) the H01 subgraph, (b) degree-preserving rewiring, (c) matched random graph (same size and edge count, signs reassigned). Node colour: excitatory (blue), inhibitory (orange), sign unknown (grey, treated as excitatory). Layouts are force-directed (networkx spring, seed 3), not anatomical. Colours checked with the dataviz palette validator (blue/orange pass, CVD distance 24.7).
Numbers printed under the panels (A_300_s0): H01 clustering 0.125, reciprocal edges 6.6%; rewired 0.084, 3.0%; random 0.065, 3.3%. (A_100_s0): 0.207, 12.2%; 0.162, 6.3%; 0.134, 3.4%. So the H01 subgraph has more local clustering and about twice as many reciprocal connections than its controls, which keep the same degrees.

## 35. Pipeline figure from the tissue to the subgraph (2026-11-19 19:05)
Read pages 1-8 of the H01 paper (Shapson-Coe et al., Science 384, eadk4858, 2024; file supplied by the user). Facts used in figures/fig_pipeline.py (png, pdf, svg): sample from the left anterior middle temporal gyrus of a 45-year-old female, removed to reach an epileptic focus in the hippocampus; 170 um thick slab, all six layers, about 3 mm from layer 1 to white matter; 5,019 sections, mean 33.9 nm, multibeam SEM (61 beams), 4 x 4 nm2 pixels, 1.4 PB, 1.05 mm3; FFN segmentation, c3 agglomeration (the one used here); 149,871,669 synapses detected, 111,272,315 classified excitatory and 38,599,354 inhibitory (corrected estimate 102.5 M, 67.1% excitatory; 50.3 M, 32.9% inhibitory; classification correct 86.89% / 84.98%); 57,180 cells with nuclei: 16,087 neurons, 32,315 glia, 8,100 vessel-related; 104 neurons proofread (c2 vs c3 comparison).
Panels d-e use our numbers: 166 shards, 160,259,769 rows, 73,896 soma-labelled ids, 16,599 neurons by our rule (paper: 16,087 counted by nucleus; a consistency check, not the same list), 87,706 directed connections and 148,140 synapse rows among those neurons (0.09% of all rows), 14,683 neurons with a connection, giant strongly connected core 10,133, subgraph 300 neurons with 2,981 connections (190 excitatory, 27 inhibitory, 83 unknown sign).
Checks on our assumptions against the paper: (1) our export has 160.3 M rows against the paper's 149.9 M detected synapses: not the same count, reason not established; (2) type 1 share of our AXON->DENDRITE rows is 36.6% and 79.5% / 85.2% for AXON->SOMA / AXON->AIS; the paper reports that excitatory synapses dominate dendrites and notes that somata (0.394%) and AIS (0.197%) are rare targets; the enrichment of type 1 on soma and AIS fits type 1 = inhibitory, but the overall type-1 share (about 37% of rows, compared with 25.8% of detected synapses classified inhibitory, 32.9% after the paper's correction) is not an exact match, so the type mapping stays a working assumption; (3) the share of inhibitory neurons in our subgraphs (9-13%) is below the paper's 29.1% interneurons among neurons (4,688 of 16,087), which suggests inhibitory neurons are under-represented among neurons with enough clean outgoing evidence, or that sign assignment misses some of them.
The drawings in panels a-c are our own schematics, not images from the paper.

## 36. Scale figure (2026-11-19 19:30)
figures/fig_scale.py (png, pdf, svg): whole human brain (about 86 billion neurons, Azevedo et al. 2009, recalled from memory; synapses 10^14 to 10^15, a rough literature range, to be checked against a source before submission), H01 (16,087 neurons, 149,871,669 synapses, Shapson-Coe et al. 2024), our subgraph (300 neurons, 5,163 synapse rows). Ratios: neurons brain/H01 5.3 million, H01/subgraph 54, brain/subgraph about 290 million; synapses brain/H01 0.7 to 7 million, H01/subgraph 29 thousand, brain/subgraph 19 to 190 billion.
To do before submission: verify the two whole-brain numbers against the original sources (I did not look them up in this session).

## 37. Nested network figure (2026-11-19 19:50)
figures/make_layout.py: force-directed layout (networkx spring_layout, 50 iterations, seed 1, 464 s) of the 16,599 neurons by our rule with their tier A connections; 14,521 neurons lie in the largest connected group and are drawn (87,601 directed connections); cached in figures/layout_h01_neurons.npz. figures/fig_nested.py draws three panels: (a) ILLUSTRATION of a whole brain (outline with 3,200 random dots, nearest-neighbour links and long-range arcs; no data), (b) the real H01 neuron graph with the 300 subgraph neurons marked in violet, (c) the 300-neuron subgraph coloured by sign. Violet boxes and dashed zoom lines show the nesting (not to scale). Output: figures/fig_nested.png/.pdf.

## 38. Queue, partial results (seed 0), 2026-11-19 20:07
- h01 data-signs: 26/30 (87%), 4 falls, eval checkpoint 'last' (m8 code, no best_theta).
- shuffle: 3/30 (10%), 27 falls, 'best' gen 24. Training never learned (candidate means about -0.85).
- esn: 22/30 (73%), 7 falls, score +0.829, 'best' gen 20.
- er: 22/30 (73%), 2 falls, score +1.000, 'best' gen 33 (last gen 60). Learned from about gen 17, centre unstable until about gen 38.
- Caveats: single seed; h01 evaluated with 'last', controls with 'best' (chosen by noisy 2-episode centre score); ES centre collapses sporadically. h01 vs er/esn gap (4 episodes) is not significant. Shuffle result may be an optimisation failure, not a structural effect.
- Next: re-evaluate every finished run with one rule on the same 30 offsets; wait for more seeds; run 5/18 (random17 signs) started 20:06.

## 39. Queue finished (18 runs), 2026-11-20
Successes out of 30 per seed (s0,s1,s2): h01 data 26*,9,0 (35/90, 39%); shuffle 3,8,0 (11/90, 12%); esn 22,28,28 (78/90, 87%); er 22,29,15 (66/90, 73%); h01 random17 20,28,15 (63/90, 70%); h01 allpos 1,5,3 (9/90, 10%). (*seed-0 h01 used 'last' checkpoint, all others 'best'.) Open-loop baseline 60%.
Findings: H01 wiring with data signs does NOT beat the controls; the standard ESN is best. All-positive signs fail (signs matter). Several runs end below the open-loop baseline (0% < 60%), so the ES drifts away from the safe zero readout: a training/selection flaw, not only a graph effect. Queue table in the script only shows h01 seed 0 (aggregation bug). Hypotheses: raw spectral radius 7.8-9.9 means hub-dominated dynamics; sign assignment (only ~9% inhibitory in the data) hurts; unverified.

## 40. Trainer v2 (2026-11-20)
Changes: checkpoint chosen by 10 fixed validation offsets (seeds 5000+i, disjoint from training and the 30 eval offsets), checked every 3 gens and at the end; zero readout (open loop) is the starting candidate so a run can never end below it on validation; ES_EPISODES 2->3, ES_LR 0.05->0.03; RUN_TAG "_v2" keeps old results separate; queue_report/skip check now finds eval_*_best.csv (old bug: summary csv had one row); new sign policy "permute" (data signs shuffled among neurons); allpos dropped (known failure), conditions: h01 data/permute/random17, shuffle, er, esn. Tested with a fake robot (logic only).

## 41. Queue v2, 16 of 18 runs done (2026-11-21 08:40)
Successes /30 (s0,s1,s2): h01 data 29,20,28 (77/90, 86%); h01 permute 30,30,30 (100%); h01 random17 30,30,18 (78/90, 87%); shuffle 30,29,30 (89/90, 99%); er 30,30,(running); esn 29,30,(pending). Open-loop validation 5-9/10 per run.
Trainer v2 fixed the failures: nearly every control is at ceiling (97-100%), so the task no longer discriminates between graphs. H01 with data signs is the lowest group; shuffle (same data signs) is 99%, permute 100%. No evidence that the H01 wiring helps. Training takes ~105 s per generation (about 1.5-2 h per run), not 33 min.

## 42. v3 harder task: design and pre-registration (written 2026-11-21 before any v3 run)
Why: v2 hit a ceiling (97-100% for most controls), so the task could not separate graphs. v3 adds: heavier box (BOX_MASS), wider offset range (BOX_Y_RANGE), Gaussian input noise (SENSE_NOISE), and no privileged inputs (PRIVILEGED=False: box gap/side, heading, lateral offset set to 0; only IMU + gait phase remain). New mode "calibrate" runs the open-loop baseline over CALIBRATE_GRID.
Setting rule: the (mass, offset, noise) setting is chosen ONLY from open-loop baseline results, so that open loop succeeds 30-60%. No reservoir result may be used to choose or tune the setting.
Hypothesis H1: with real H01 wiring and measured signs, success exceeds each of shuffle, er, esn, h01-permute.
Primary comparison: pooled success over 3 seeds x 30 eval episodes per condition (same 30 box offsets for all), plus per-seed paired differences. Conditions: h01 data, h01 permute, shuffle data, er data, esn data (15 runs, trainer v2 settings, RUN_TAG _v3).
Decision: all results will be reported whichever way they go, including the v1 and v2 queues. If H1 fails again, the paper reports that. Limitation: 3 subgraphs/seeds only.

## 43. Queue v2 complete (2026-11-21 11:02)
Successes /30 (s0,s1,s2), pooled /90: h01 data 29,20,28 (86%); h01 permute 30,30,30 (100%); h01 random17 30,30,18 (87%); shuffle 30,29,30 (99%); er 30,30,29 (99%); esn 29,30,23 (91%). Task at ceiling; no H01 advantage. Next: v3 calibrate (see section 42).

## 44. Calibrate, first grid (2026-11-21 12:09), open loop, 20 episodes each
0.5 kg / +-0.10 m: 12/20 (60%, 4 falls; matches the earlier baseline, so the mass setter does not disturb anything). 1.0 kg / +-0.15, 1.5 / +-0.15, 2.0 / +-0.20, 3.0 / +-0.20: 0/20 each, every episode a fall within ~0.05 m of box movement. Cliff between 0.5 and 1.0 kg. Second grid: masses 0.6-0.9 at +-0.10, and 0.7, 0.8 at +-0.15.
Note: open loop is not affected by noise or PRIVILEGED, so calibration only fixes mass and offset; the noise level has to be set by a rule too (see reply).

## 45. Calibrate second grid and v3 setting (2026-11-21 12:12), chosen from open-loop results only
Open loop /20: 0.6 kg +-0.10: 18 (90%); 0.7 +-0.10: 6 (30%, 9 falls); 0.8 +-0.10: 0; 0.9 +-0.10: 0; 0.7 +-0.15: 10 (50%, 5 falls); 0.8 +-0.15: 0.
Chosen: BOX_MASS 0.7 kg, BOX_Y_RANGE +-0.15 m (inside the 30-60% band; 0.7/+-0.10 is at its edge). PRIVILEGED False, SENSE_NOISE 0.05 (set before any reservoir run).
Feasibility rule (written before running): one 20-generation run of the standard ESN, seed 0, tag _v3feas. If its validation never beats open loop, the task is not learnable without privileged inputs and v3 falls back to PRIVILEGED True (noise kept); this check does not use H01.
Incident: config.py sent for the second calibrate grid was truncated by a regex bug (lines after CALIBRATE_GRID lost); calibrate ran fine because it only needs the earlier lines. Rebuilt in full as config_1/2/3.

## 46. v3 open-loop baseline (2026-11-21 12:14), 0.7 kg, +-0.15 m, 30 episodes
10/30 success (33%), 16 falls, mean score +0.226, 5.1 s to goal. In band (30-60%). The calibrate run on the same setting and the same first 20 offsets gave 10/20; this run gives 5/20 on those episodes, so repeated open-loop evaluations are not identical (physics/state differences between runs). Treat gaps of a few episodes as noise. Next: feasibility run (config_2).

## 47. v3 feasibility result (2026-11-21 12:34), ESN seed 0, 20 gens, no privileged inputs, noise 0.05, 0.7 kg, +-0.15 m
Open-loop validation 3/10 (+0.133). NEW BEST at gen 9: validation +0.594 (6/10), so the pre-registered feasibility rule is met (validation beat open loop). Eval of the gen-9 checkpoint on 30 offsets: 12/30 success (40%), falls 2, score +0.548, vs open loop 10/30 (33%), 16 falls, score +0.226. Gain is mostly fewer falls; success barely above baseline (within noise). Remaining failures: robot turns away (|robot_yaw| ~2.5-3, box barely moves), expected without heading/box inputs. Training was unstable after gen 12 (centre and candidates down to -0.7; frequent physics warnings), the validation-selected checkpoint rescued it. Decision: proceed with config_3_full (15 runs, 60 gens, same trainer settings for all conditions).

## 48. v3 queue: simulator failure, runs 8-15 invalid (2026-11-22 01:09)
Valid runs 1-7 (successes /30): seed 0: h01 data 30, permute 29, shuffle 29, er 18, esn 29; seed 1: h01 data 30, permute 27. Open-loop baseline at this setting 10/30 (33%). Even with no privileged inputs and noise 0.05, most trained runs reach 90-100%: ceiling again, except er s0 (60%).
Run 8 (shuffle s1): after repeated physics warnings (gen 3-4, 23:29-23:30) every episode scored -1.034 (instant fall) from gen 5 on, also for the zero readout in runs 9-15 (open-loop validation 0/10). So the physics world was broken until restart; runs 8-15 (shuffle/er/esn seed 1, all five seed 2) are INVALID, not results. The QUEUE TABLE printed at the end averages these zeros and must be ignored.
Fix (code): sim_broken() reloads the world (p.worldReload) when the open-loop validation is 0 successes with score < -1.0, or three generations in a row have mean and centre < -1.02; the run restarts from scratch, max 3 reloads per run; queue deletes finished evaluations that are all instant falls and redoes them; queue_report skips them. Redo: 8 runs, about 10 h.

## 49. v3 queue, 13 of 15 valid runs done (2026-11-22 09:25)
Successes /30 (s0,s1,s2): h01 data 30,30,13 (73/90, 81%); h01 permute 29,27,29 (85/90, 94%); shuffle 29,30,16 (75/90, 83%); er 18,23,(running, gen 28/60, validation currently below open loop); esn 29,30,(pending). Open loop 10/30. No auto-reload was needed (0 broken-world events after the fix).
Seed 2 is the hard one for the two graphs with data signs (13 and 16), while permute on the same wiring got 29. h01 data vs shuffle (same signs): 73 vs 75, no difference. h01 data vs permute: 73 vs 85 (Fisher p about 0.01, descriptive only; episodes are not independent). Data signs on subgraph s2 may be the problem; one subgraph, not established.

## 50. v3 queue complete, 15 valid runs (2026-11-22 11:23)
Successes /30 (s0,s1,s2), pooled /90: h01 data 30,30,13 = 73 (81%); h01 permute 29,27,29 = 85 (94%); shuffle 29,30,16 = 75 (83%); er 18,23,13 = 54 (60%); esn 29,30,30 = 89 (99%). Open loop 10/30 (33%).
Pre-registered H1 (h01 data beats shuffle, er, esn, permute): NOT supported. h01 data vs shuffle p=0.85 (tie), vs er p about 0.006 (better), vs esn and vs permute worse (p < 0.01). Best: ESN. Order: esn > h01 permute > shuffle ~ h01 data > er.
Note: the seed-2 subgraph is weak for the data-sign graphs (13, 16) and for er (13); permute and esn are fine there.
Missing controls: shuffle with permuted signs (isolates wiring vs signs), and a no-reservoir (bias-only / direct linear) policy.

## 51. v3b controls: plan (written 2026-11-22 13:10, before any run)
Questions: (a) does the H01 wiring matter once signs are random? Compare h01 permute vs shuffle permute (same signs treatment, scrambled wiring). (b) how much of the jump over open loop is due to the reservoir? Compare every reservoir with "direct" (linear readout of the 12 inputs, no hidden units, 39 numbers) and "bias" (one fixed turn/lean/stride correction, 3 numbers).
Same settings as v3 (0.7 kg, +-0.15 m, no privileged inputs, noise 0.05, trainer v2, 60 gens, 3 seeds). 9 new runs (shuffle+permute, direct, bias; 3 seeds each), 15 finished runs skipped. All outcomes will be reported.

## 52. v3b controls, 23 of 24 runs done (2026-11-22 21:34); last run (bias seed 2) was still at gen 2/60
Successes /30 (s0,s1,s2): shuffle+permute 29,29,29 = 87/90 (97%); direct (linear readout of the 12 inputs, 39 numbers) 29,30,26 = 85/90 (94%); bias (3 numbers, no inputs, no reservoir) 29,30,(running) = 59/60 so far (98%). Earlier: esn 89, h01 permute 85, shuffle data 75, h01 data 73, er 54 (/90). Open loop 33%.
Findings: (1) the 3-number bias policy reaches 97-100%, so a constant correction (turn/lean/stride) fixes the open-loop gait; the reservoir is not needed for this task. (2) shuffle+permute (97%) >= h01 permute (94%): the wiring makes no measurable difference. (3) Differences between reservoir conditions mostly reflect how easy 900 readout numbers are to train, not computation. The v3 task cannot test the connectome hypothesis.
Planned: v4 task where a fixed correction cannot work: a hidden random per-episode gait bias (turn/lean) that the policy must infer from IMU history. Setting rule: bias-only and direct must succeed <= ~50% there (chosen from those controls and open loop only, before any reservoir run).

## 52. v3b controls, 23 of 24 runs done (2026-11-22 21:35; bias seed 2 still training)
Successes /30 (s0,s1,s2): shuffle+permute 29,29,29 (87/90, 97%); direct (linear readout of the 12 inputs, no reservoir) 29,30,26 (85/90, 94%); bias (3 numbers, one fixed turn/lean/stride correction, no inputs, no reservoir) 29,30,pending (59/60 so far).
Findings: (1) wiring: h01 permute 85/90 vs shuffle permute 87/90, no difference. (2) The reservoir is not needed: a constant gait correction reaches 97-100%, and a memoryless linear readout 94%. The jump from 33% (open loop) to ~97% comes from retuning the gait for the heavier box, not from sensing or recurrence. (3) Reservoirs with data signs (81%) or random wiring + shuffled signs (60%) do worse than the constant correction, so they can add harm.
Consequence: the proper baseline is the bias-only policy, not the untuned open loop; the task does not need feedback. A harder test must vary the conditions per episode (box mass, friction, start heading) so one fixed correction cannot fit all.

## 53. FINAL v3 table, all 24 runs valid (2026-11-22 22:58)
Successes /30 (s0,s1,s2), pooled /90: bias 29,30,30 = 89 (99%); esn 29,30,30 = 89 (99%); shuffle+permute 29,29,29 = 87 (97%); h01 permute 29,27,29 = 85 (94%); direct 29,30,26 = 85 (94%); shuffle data 29,30,16 = 75 (83%); h01 data 30,30,13 = 73 (81%); er 18,23,13 = 54 (60%). Open loop 10/30 (33%). Falls per episode: bias 0.00, direct 0.00, esn 0.01, h01 data 0.10, er 0.20.
Conclusions: (1) the bias-only policy (3 numbers) ties the best reservoir; the task does not need feedback or recurrence. (2) wiring: h01 permute vs shuffle permute 85 vs 87 (p=0.7), no effect. (3) measured signs hurt on subgraph s2; random-assignment signs fine. (4) er (random wiring + shuffled signs) is the weakest, probably noise injection. (5) Pre-registered H1 not supported.
Decisions: publish these results on bioRxiv; follow-up on DARwIn-OP pick-and-place with domain randomization and a liquid-network reservoir, gated by simple baselines. Paper draft started 2026-11-22 (Docs).

## 54. Paper draft started (2026-11-22 23:50)
Draft written as a Docs artifact (https://claude.ai/artifact/PcvQ6MLXhguHhWihSmDxcq): abstract, introduction, related work (verified-to-exist papers only), methods, results (final queue chart + comparison table + earlier queues), discussion with limitations and a careful whole-brain-emulation paragraph, availability, references.
Open items before posting: verify author lists, volumes and pages for references 2-6 and 8; add hemibrain, MICrONS, C. elegans connectome, FFN papers; resolve 160.3M vs 149.9M row discrepancy and the type-1 sign mapping; check H01 licence for redistribution; clean local paths from logs; verify the whole-brain volume figure (1.05 mm3 vs about 1.2 million mm3).
Not yet in the paper: figures from earlier turns (pipeline, scale, nested brain-to-subgraph).

## 55. Paper draft edits after user feedback (2026-11-22)
- Abstract now opens with the question ("Can the wiring of a human cortical microcircuit serve as the recurrent core of a robot controller?"); leading-result sentence removed; "negative results" replaced by "results"; abstract, introduction and discussion now say the task is specific and easy to train, so a more complex test is needed to evaluate wiring; "Follow-up" renamed "Next steps" (more refined, more difficult tasks).
- Row counts: the paper's 149,871,669 is its classified-synapse total (111,272,315 + 38,599,354); ours is 160,259,769 export rows after our filters. Not a typo in the paper; cause not established (unverified idea: rows of unclassified or other types). Limitations bullet removed; Methods keeps one neutral sentence (our export has somewhat more rows; we use row counts as edge weights).

## 56. Paper draft edits, round 2 (2026-11-23)
- Robot and task paragraph now frames the pushing task as the evaluation of the policy (simple manipulation test that exercises whole-body humanoid motion).
- Next steps: removed the DARwIn-OP forearm pick-and-place, the liquid-network plan and the follow-up-project wording; now only "more refined, more difficult tasks, conditions changing per episode, same controls". Removed the liquid-network sentence from Related work and references Hasani 2022 and Correll 2026 (no longer cited); Lechner et al. description no longer says "liquid".
- Added a "Long-term outlook" paragraph under "What a connectome does and does not give": hope for non-destructive full-brain wiring graphs, a network built on it, fed individual-specific data, with a general robot model attached; stated as a hope, not tested, with caveats (H01 is a small fragment from removed tissue; no method for living brains; unknown whether sufficient).

## 57. Repository and project page (2026-11-23)
- Built /home/claude/exohominid (zip sent): README with honest status, MIT licence, CITATION.cff, .gitignore, requirements (figures only), docs/ project page (layout after the Open-TeleVision-style page used for the earlier paper) and docs/REPRODUCE.md, webots/ (world, controller with final-queue config, calibrated walker and darwin_state.json, tier-A 100/300/1000 subgraphs), pipeline/ (H01 scripts, local paths removed), outputs/graph_v2 (derived tables), results/ (final queue, earlier queues, pairwise tests), figures/ (scripts and images incl. new fig_results), cleaned LOG.md copy.
- Checks: make_subgraph.py regenerates edges_A_300_s0.tsv byte-identically; reservoir build works offline for h01, shuffle, er, esn, bias; page checked at 1100 px and 390 px width; Fisher p-values recomputed (match the paper).
- Not tested: the Webots controller inside the repo layout (only the offline reservoir build and relative subgraph path were tested).
- Paper Data availability paragraph updated (repo released, MIT; raw tables not redistributed).

## 58. LaTeX paper draft (2026-11-23)
- Wrote paper/paper.tex in the preamble and layout of the earlier preprint (article 11 pt, 1 in margins, natbib numbers, booktabs tables), with refs.bib. 13 pages, no overfull boxes.
- New figures (figures/paper_figs.py): fig_overview, fig_ei (synapse types, sign combination, real recurrent matrix of s0), fig_subgraphs (three subgraphs, structural statistics recomputed from the TSVs; they match the paper table), fig_policy (sensors, reservoir, readout, gait, robot), fig_task, fig_training (evolution strategy). Reused: fig_nested, fig_pipeline (H01 to reservoir), graph-controls figure, fig_results.
- Figure 8 (robot frames) is a placeholder until the user supplies recordings.
- Bibliography entries for Dorkenwald, Shiu, Suarez are marked "to be verified"; added Jaeger 2001, Lukosevicius and Jaeger 2009, Maslov and Sneppen 2002, Salimans et al. 2017 from memory (to verify).

## 59. Figure 1 redesign and shapes (2026-11-23)
- Figure 1 redrawn per the user's idea: abstract human mannequin with a brain-wiring illustration above, an arrow to an abstract robot with the 300-neuron subgraph above it, dashed zoom lines between. The robot and mannequin are original schematic drawings, not traced from the DARwIn-OP product photos the user supplied (licensing).
- Node shapes by sign in all network figures: excitatory circles, inhibitory triangles, unknown diamonds (figures/shapes.py; fig_graph, fig_pipeline, fig_nested, paper_figs).
- Task figure (Figure 7) redrawn as a side view to scale (robot 0.339 m, cube 0.6 m, 0.5 m gap, goal at 0.40 m) with a top-view inset for the sideways offset.

## 60. Citations, videos, Figure 1 heads (2026-11-23)
- Removed the unpublished self-citation; FlyWire entry (Nature 634:124-138, doi 10.1038/s41586-024-07558-y, user's link) completed; conn2res completed.
- Added connectome-reservoir refs (Suarez 2021, Damicelli 2022, d'Andrea 2022, Morra & Daley 2022, Costi 2025) and Lappalainen 2024, Vaxenburg 2025, Januszewski 2018 (latter three from search metadata/memory: verify volumes/pages).
- Four recordings cropped to the 3D view (docs/assets/vid): baseline, baseline_fast, two training runs (not trained-policy rollouts). Fig 8 now uses baseline frames.
- Figure 1 redrawn: slender mannequin head/neck and robot head/neck, brain and subgraph above, connecting text removed.

## 61. Training-frames figure, project page and README (2026-11-23)
- Added Figure (fig_trained_frames) from the H01 training recording; caption states it is qualitative, not a success count (the recordings are training runs, not rollouts of the final trained policy).
- Project page rebuilt with all paper figures (overview, pipeline, synapse types, three subgraphs, policy, task, training), four videos with posters, frame strips. README gets GIFs and figures. figures/paper_figs.py main block moved to the end so it regenerates the final figures; make_frames.py added; requirements gain networkx and Pillow.

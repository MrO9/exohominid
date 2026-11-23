# Reproducing ExoHominid

There are two levels. Level 1 reruns the robot experiments from the subgraphs in this repository. Level 2 rebuilds the graph from the raw H01 tables (optional; about 32 GB of downloads and several hours).

## Level 1: robot experiments (Webots)

Requirements: Webots R2025a. The controller needs only the Python standard library.

1. Open `webots/worlds/push_box.wbt`. The controller is `h01_dataset_policy` (in `webots/controllers/`).
2. `darwin_state.json` (the calibration and searched gait of the DARwIn-OP walker) is included next to `darwin_rl.py`.
3. Edit `MODE` in `webots/controllers/h01_dataset_policy/config.py`, save, and reset the simulation, in this order:
   - `"check"`: five episodes with the box far away; confirms the walker loads and walks.
   - `"baseline"`: open-loop gait against the box, `EVAL_EPISODES` episodes on fixed box offsets.
   - `"res_check"`: builds the reservoir from the `RES_*` settings; zero readout must match the baseline.
   - `"train"`: trains one readout with the evolution strategy; `"eval"` evaluates it.
   - `"queue"`: trains and evaluates every run listed in `QUEUE_SEEDS` and `QUEUE_CONDITIONS` (restartable). The shipped settings are the final queue: 0.7 kg box, offsets ±0.15 m, `PRIVILEGED = False`, sensor noise 0.05, 60 generations, validation checkpointing.
4. Results are written to `webots/controllers/h01_dataset_policy/results/` (`run_log.txt`, per-run csv files, `queue_summary.csv`). The queue runs eight conditions on three subgraphs and takes many hours.

Notes:
- Repeated open-loop evaluations are not identical (physics is not fully deterministic), so differences of a few episodes are within noise.
- If the physics world enters a state where every episode scores -1.034, the controller detects it, writes a counter file and reloads the world (up to three times per run).
- Old csv files from other settings are not comparable; delete `results/` when you change the box mass or offset range, or change `RUN_TAG`.

Offline check without Webots (builds each reservoir and prints its size):

```
cd webots/controllers/h01_dataset_policy
python -c "import reservoir as R; print(R.build('A_300_s0','../../subgraphs',kind='h01',sign_policy='data')[1])"
```

## Level 2: from the raw H01 tables to the subgraphs

All scripts are in `pipeline/` and use the standard library only. They were run in this order (details and timings in `LOG.md`):

1. `h01_edges.py`: streams the 166 shards of the c3 synapse export one at a time, keeps a compact edge list, deletes each shard (about 197 MB each). Restartable.
2. `h01_build_graph_v2.py`: finds neurons and builds the directed graph (`outputs/graph_v2/`).
3. `h01_cell_signs.py`: assigns one sign per neuron from its output synapse types.
4. `make_subgraph.py`: grows the 100-, 300- and 1,000-neuron subgraphs from the giant strongly connected core.

`h01_peek.py`, `h01_stats.py`, `h01_idcheck.py` and `h01_zip_names.py` are the exploration scripts used while checking the data; the `LOG.md` entries explain each.


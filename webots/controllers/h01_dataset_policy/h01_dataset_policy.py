"""ExoHominid push-box controller entry point. MODE is set in config.py."""
import csv
import json
import math
import os
import random
import time

import darwin_rl as dr

import config
from pusher import Pusher
from policy import N_IN, ReservoirPolicy
import reservoir

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "run_log.txt"), "a") as f:
        f.write(line + "\n")


COLS = ["episode", "score", "success", "fell", "tipped", "t_goal", "t_contact", "box_dx", "box_dy", "box_yaw",
        "robot_dx", "robot_dy", "robot_yaw", "seconds", "box_y0"]


def box_offset(seed):
    """Sideways start offset of the box for episode seed (same numbers for every controller)."""
    return random.Random(10007 + seed).uniform(-config.BOX_Y_RANGE, config.BOX_Y_RANGE)


def evaluate(p, gait, box_x, label, policy=None, n=None, seed0=0, write=True):
    """n episodes, episode i with box offset seed0+i. Returns the list of result dicts; writes results/<label>.csv."""
    n = n or config.EVAL_EPISODES
    os.makedirs(RES, exist_ok=True)
    rows = []
    path = os.path.join(RES, f"{label}.csv")
    fh = open(path, "a", newline="") if write else None
    w = csv.writer(fh) if write else None
    if write and os.path.getsize(path) == 0:
        w.writerow(COLS)
    for i in range(n):
        y0 = box_offset(seed0 + i)
        r = p.run_episode(gait, box_x, policy=policy, box_y=y0)
        r["box_y0"] = y0
        rows.append(r)
        if write:
            w.writerow([i, round(r["score"], 4), int(r["success"]), int(r["fell"]), int(r["tipped"]),
                        "" if r["t_goal"] is None else round(r["t_goal"], 2),
                        "" if r["t_contact"] is None else round(r["t_contact"], 2),
                        round(r["box_dx"], 4), round(r["box_dy"], 4), round(r["box_yaw"], 3),
                        round(r["robot_dx"], 4), round(r["robot_dy"], 4), round(r["robot_yaw"], 3),
                        round(r["seconds"], 2), round(y0, 3)])
            fh.flush()
        log(f"{label} ep {i}: score {r['score']:+.3f} success {int(r['success'])} fell {int(r['fell'])} "
            f"box_dx {r['box_dx']:.3f} box_dy {r['box_dy']:+.3f} robot_yaw {r['robot_yaw']:+.2f}"
            + ("" if r["t_goal"] is None else f" t_goal {r['t_goal']:.1f}s"))
    if fh:
        fh.close()
    log(summarize(rows, label))
    return rows


def summarize(rows, label):
    n = len(rows)
    ok = [r for r in rows if r["success"]]
    tg = [r["t_goal"] for r in ok if r["t_goal"] is not None]
    return (f"SUMMARY {label}: n={n} success {len(ok)}/{n} ({len(ok) / n:.0%}) falls {sum(r['fell'] for r in rows)} "
            f"mean score {sum(r['score'] for r in rows) / n:+.3f} mean |box_dy| {sum(abs(r['box_dy']) for r in rows) / n:.3f} "
            f"mean time to goal {sum(tg) / len(tg) if tg else float('nan'):.1f}s")


def calibrate(p, gait):
    """Open-loop baseline for each (box mass, offset range): choose a setting where the task has head room."""
    n = 20
    for mass, yr in config.CALIBRATE_GRID:
        config.BOX_MASS, config.BOX_Y_RANGE = mass, yr
        rows = evaluate(p, gait, config.BOX_FACE_X, f"calibrate_m{mass}_y{yr}", n=n)
        ok = sum(int(r["success"]) for r in rows)
        log(f"CALIBRATE box mass {mass} kg, offset range +-{yr} m: open loop {ok}/{n} success, falls {sum(int(r['fell']) for r in rows)}")


def run_check(p, gait):
    """The old no-box walk: how the open-loop gait drifts."""
    rows = []
    for i in range(config.EPISODES):
        r = p.run_episode(gait, config.FAR_BOX_FACE_X)
        rows.append(r)
        log(f"check_no_box ep {i}: robot_dx {r['robot_dx']:.3f} robot_dy {r['robot_dy']:+.3f} robot_yaw {r['robot_yaw']:+.3f} "
            f"fell {int(r['fell'])}")


def best_arms_path():
    return os.path.join(RES, "best_arms.json")


def resolve_arms():
    """The arm posture asked for in config.PUSH_ARMS (None = walking pose)."""
    a = config.PUSH_ARMS
    if a == "best":
        try:
            with open(best_arms_path()) as f:
                return json.load(f)["spec"]
        except Exception:
            return None
    return a


def arms_scan(p, gait):
    """Try several arm postures against the box and keep the one that pushes it furthest. Signs of the shoulder
    joint are not known in advance, so both directions are tried."""
    elbows = {"elbow mid-range": "mid", "elbow straight": (0.0, 0.0)}
    cands = []
    pairs = [(1.5, -1.5), (1.2, -1.2), (1.8, -1.8), (3.0, 3.0), (3.0, -3.0)]   # (right, left) shoulder angles, rad
    for sr, sl in pairs:
        for en, e in elbows.items():
            cands.append((f"shoulders R {sr:+.1f} L {sl:+.1f}, {en}",
                          {"shoulder": (sr, sl), "upper": (-0.67, 0.76), "elbow": e}))
    os.makedirs(RES, exist_ok=True)
    rows = []
    with open(os.path.join(RES, "arms_scan.csv"), "a", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate", "episode", "fitness", "box_dx", "box_dy", "fell", "tipped"])
        for name, spec in cands:
            p.apply_arms(spec)
            fits = []
            for i in range(2):
                r = p.run_episode(gait, config.BOX_FACE_X)
                fits.append(r["fitness"])
                w.writerow([name, i, round(r["fitness"], 4), round(r["box_dx"], 4), round(r["box_dy"], 4),
                            int(r["fell"]), int(r["tipped"])])
                fh.flush()
                log(f"arms {name} ep {i}: box_dx {r['box_dx']:+.3f} fell {r['fell']} fitness {r['fitness']:+.3f}")
            rows.append((sum(fits) / len(fits), name, spec))
    rows.sort(key=lambda r: -r[0])
    log("ARMS SCAN RESULT (mean fitness over 2 episodes, best first):")
    for f, name, spec in rows:
        log(f"   {f:+.3f}  {name}")
    with open(best_arms_path(), "w") as f:
        json.dump({"name": rows[0][1], "spec": rows[0][2], "fitness": rows[0][0]}, f, indent=1)
    log(f"saved the best posture ({rows[0][1]}) to {best_arms_path()}")


def build_policy():
    folder = os.path.normpath(os.path.join(HERE, config.SUBGRAPH_DIR))
    res, info = reservoir.build(config.RES_SUBGRAPH, folder, kind=config.RES_GRAPH, sign_policy=config.RES_SIGN,
                                n_in=N_IN, rho=config.RES_RHO, leak=config.RES_LEAK, in_scale=config.RES_IN_SCALE,
                                weight_mode=config.RES_WEIGHT, seed=config.RES_SEED)
    log(info)
    tag = f"{config.RES_GRAPH}_{config.RES_SUBGRAPH}_{config.RES_SIGN}_{config.RES_WEIGHT}_seed{config.RES_SEED}{config.RUN_TAG}"
    return ReservoirPolicy(res), tag


def readout_path(tag):
    return os.path.join(RES, f"readout_{tag}.json")


def res_check(p, gait):
    pol, tag = build_policy()
    t0 = time.time()
    for k in range(50):
        pol.res.step([0.1] * N_IN)
    log(f"reservoir speed: {(time.time() - t0) / 50 * 1000:.1f} ms per step "
        f"(an episode has {config.EPISODE_STEPS} steps; the simulation step is {p.ts} ms)")
    pol.reset()
    log("zero readout (must match the baseline):")
    evaluate(p, gait, config.BOX_FACE_X, "res_check_zero", policy=pol, n=5, write=False)
    rng = random.Random(3)
    pol.set_params([rng.gauss(0, 0.05) for _ in range(pol.n_params)])
    log("random readout (the walk must change):")
    evaluate(p, gait, config.BOX_FACE_X, "res_check_random", policy=pol, n=5, write=False)


def validate(p, gait, pol, theta):
    """Mean score and successes of a readout on fixed validation box offsets (disjoint from training and from the final eval)."""
    pol.set_params(theta)
    rows = [p.run_episode(gait, config.BOX_FACE_X, policy=pol, box_y=box_offset(5000 + i)) for i in range(config.ES_VAL_EPISODES)]
    return sum(r["score"] for r in rows) / len(rows), sum(int(r["success"]) for r in rows)


def sim_broken(p, tag, path, hist, why):
    """The physics world can end up in a state where every episode is an instant fall (seen once in the v3 queue after
    repeated physics warnings). The fix is to reload the world. The run that was being trained is restarted from scratch
    (its readout drifted while every score was equal). After 3 reloads for the same run, stop."""
    cnt_path = os.path.join(RES, f"reloads_{tag}.txt")
    n = 0
    if os.path.exists(cnt_path):
        with open(cnt_path) as fh:
            n = int(fh.read().strip() or 0)
    log(f"[!] simulator looks broken ({why}); reload number {n + 1} for {tag}")
    if n >= 3:
        raise SystemExit(f"[!] {tag}: the world was reloaded 3 times and still fails. Check the world by hand, then delete {cnt_path}")
    with open(cnt_path, "w") as fh:
        fh.write(str(n + 1))
    for q in (path, hist):
        if os.path.exists(q):
            os.remove(q)
    p.worldReload()                                     # the controller is restarted; the queue resumes by itself
    raise SystemExit("world reload requested")


def train(p, gait):
    pol, tag = build_policy()
    P = pol.n_params
    theta = [0.0] * P
    start_gen, best, best_theta, best_gen = 0, -1e9, None, 0     # best = validation score of best_theta
    path = readout_path(tag)
    if config.ES_RESUME and os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        theta, start_gen = d["theta"], d.get("generation", 0)
        best_theta, best_gen, best = d.get("best_theta"), d.get("best_generation", 0), d.get("best_val", -1e9)
        log(f"resuming {tag} at generation {start_gen}")
    if best_theta is None:                      # generation 0: the open-loop walk (zero readout) is the first candidate
        best, nok = validate(p, gait, pol, [0.0] * P)
        best_theta, best_gen = [0.0] * P, 0
        log(f"validation of the zero readout (open loop): score {best:+.3f}, {nok}/{config.ES_VAL_EPISODES} successes")
        if best < -1.0 and nok == 0:                    # open loop never fails like this in a healthy world
            hist0 = os.path.join(RES, f"train_{tag}.csv")
            sim_broken(p, tag, path, hist0, "the open-loop walk falls in every validation episode")
    rng = random.Random(config.RES_SEED * 1000 + start_gen + 1)
    half = config.ES_POP // 2
    hist = os.path.join(RES, f"train_{tag}.csv")
    if not os.path.exists(hist):
        with open(hist, "w") as f:
            f.write("generation,mean_score,best_score,center_score,seconds\n")
    t0 = time.time()
    dead = 0
    for gen in range(start_gen, config.ES_GENS):
        eps = [[rng.gauss(0, 1) for _ in range(P)] for _ in range(half)]
        cands = [(e, 1.0) for e in eps] + [(e, -1.0) for e in eps]
        seeds = [100000 + gen * 10 + k for k in range(config.ES_EPISODES)]      # same box offsets for every candidate
        scores = []
        for e, sgn in cands:
            pol.set_params([theta[i] + sgn * config.ES_SIGMA * e[i] for i in range(P)])
            s = 0.0
            for sd in seeds:
                s += p.run_episode(gait, config.BOX_FACE_X, policy=pol, box_y=box_offset(sd))["score"]
            scores.append(s / len(seeds))
        order = sorted(range(len(scores)), key=lambda i: scores[i])
        util = [0.0] * len(scores)
        for rank, i in enumerate(order):
            util[i] = rank / (len(scores) - 1) - 0.5                    # centred ranks
        for i in range(P):
            g = sum(util[k] * cands[k][1] * cands[k][0][i] for k in range(len(cands)))
            theta[i] += config.ES_LR * g / (len(cands) * config.ES_SIGMA) * 2.0
        pol.set_params(theta)
        cs = sum(p.run_episode(gait, config.BOX_FACE_X, policy=pol, box_y=box_offset(sd))["score"]
                 for sd in seeds) / len(seeds)
        vtxt = ""
        if (gen + 1) % config.ES_VAL_EVERY == 0 or gen + 1 == config.ES_GENS:
            vs, vok = validate(p, gait, pol, theta)
            vtxt = f"  validation {vs:+.3f} ({vok}/{config.ES_VAL_EPISODES})"
            if vs > best:                       # strictly better than everything so far, including open loop
                best, best_theta, best_gen = vs, list(theta), gen + 1
                vtxt += "  NEW BEST"
        dead = dead + 1 if (sum(scores) / len(scores) < -1.02 and cs < -1.02) else 0
        if dead >= 3:
            sim_broken(p, tag, path, hist, "three generations in a row with every episode an instant fall")
        el = time.time() - t0
        log(f"train {tag} gen {gen + 1}/{config.ES_GENS}: candidates mean {sum(scores) / len(scores):+.3f} "
            f"best {max(scores):+.3f} centre {cs:+.3f}{vtxt}  ({el / (gen + 1 - start_gen):.0f} s per generation)")
        with open(hist, "a") as f:
            f.write(f"{gen + 1},{sum(scores) / len(scores):.4f},{max(scores):.4f},{cs:.4f},{el:.0f}\n")
        os.makedirs(RES, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"tag": tag, "generation": gen + 1, "theta": theta, "best_theta": best_theta,
                       "best_generation": best_gen, "best_val": best,
                       "settings": {k: getattr(config, k) for k in dir(config) if k.startswith(("RES_", "ES_"))}}, f)
    log(f"training finished; readout saved to {path}")


def eval_readout(p, gait):
    pol, tag = build_policy()
    path = readout_path(tag)
    if not os.path.exists(path):
        raise SystemExit(f"[!] no trained readout {path}: run mode 'train' first with the same RES_ settings")
    with open(path) as f:
        d = json.load(f)
    which = config.EVAL_WHICH if d.get("best_theta") else "last"
    pol.set_params(d["best_theta"] if which == "best" else d["theta"])
    log(f"eval {tag}: readout '{which}' (last generation {d.get('generation')}, best-centre generation {d.get('best_generation')})")
    tag = tag + ("_best" if which == "best" else "")
    evaluate(p, gait, config.BOX_FACE_X, f"eval_{tag}", policy=pol)


def tag_now():
    return f"{config.RES_GRAPH}_{config.RES_SUBGRAPH}_{config.RES_SIGN}_{config.RES_WEIGHT}_seed{config.RES_SEED}{config.RUN_TAG}"


def eval_csv_path():
    for suffix in ("_best", ""):
        q = os.path.join(RES, f"eval_{tag_now()}{suffix}.csv")
        if os.path.exists(q):
            return q
    return os.path.join(RES, f"eval_{tag_now()}.csv")


def read_eval_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def eval_looks_broken(path):
    rows = read_eval_csv(path)[-config.EVAL_EPISODES:]
    return len(rows) > 0 and all(int(r["fell"]) == 1 and float(r["box_dx"]) < 0.03 for r in rows)


def queue_runs():
    """Runs in round-robin order over the seeds, so a partly finished queue is still balanced."""
    runs = []
    for seed in config.QUEUE_SEEDS:
        for graph, sign in config.QUEUE_CONDITIONS:
            sub = config.QUEUE_SUBGRAPH.format(seed=seed) if config.QUEUE_SUBGRAPH_BY_SEED else config.RES_SUBGRAPH
            runs.append({"RES_GRAPH": graph, "RES_SIGN": sign, "RES_SEED": seed, "RES_SUBGRAPH": sub})
    return runs


def queue_report(runs):
    """Rewrite results/queue_summary.csv from every finished evaluation and log the table by condition."""
    out, by_cond = [], {}
    for r in runs:
        for k, v in r.items():
            setattr(config, k, v)
        path = eval_csv_path()
        if not os.path.exists(path):
            continue
        rows = read_eval_csv(path)[-config.EVAL_EPISODES:]
        if len(rows) < config.EVAL_EPISODES or eval_looks_broken(path):
            continue
        ok = sum(int(x["success"]) for x in rows)
        falls = sum(int(x["fell"]) for x in rows)
        score = sum(float(x["score"]) for x in rows) / len(rows)
        tg = [float(x["t_goal"]) for x in rows if x["t_goal"] not in ("", None)]
        out.append([tag_now(), r["RES_GRAPH"], r["RES_SIGN"], r["RES_SEED"], r["RES_SUBGRAPH"], len(rows), ok, falls,
                    round(score, 4), round(sum(tg) / len(tg), 2) if tg else ""])
        by_cond.setdefault((r["RES_GRAPH"], r["RES_SIGN"]), []).append((ok / len(rows), score, falls / len(rows)))
    with open(os.path.join(RES, "queue_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "graph", "sign", "seed", "subgraph", "episodes", "successes", "falls", "mean_score",
                    "mean_time_to_goal"])
        w.writerows(out)
    log("QUEUE TABLE (mean over the finished seeds; success rate, falls per episode, mean score):")
    for (g, s), v in by_cond.items():
        n = len(v)
        log(f"   {g:8s} signs {s:9s} seeds {n}: success {sum(x[0] for x in v) / n:.0%} "
            f"(per seed {[f'{x[0]:.0%}' for x in v]}) falls {sum(x[2] for x in v) / n:.2f} score {sum(x[1] for x in v) / n:+.3f}")


def queue(p, gait):
    """Train and evaluate every (graph, sign, seed) combination in config.QUEUE_*, one after the other.
    Safe to stop and restart: finished training and finished evaluations are skipped, unfinished training resumes."""
    runs = queue_runs()
    log(f"QUEUE: {len(runs)} runs ({len(config.QUEUE_SEEDS)} seeds x {len(config.QUEUE_CONDITIONS)} conditions), "
        f"about {config.ES_GENS * 33 / 60:.0f} min of training each")
    for n, r in enumerate(runs):
        for k, v in r.items():
            setattr(config, k, v)
        evalp = eval_csv_path()
        done = os.path.exists(evalp) and len(read_eval_csv(evalp)) >= config.EVAL_EPISODES
        if done and eval_looks_broken(evalp):
            log(f"QUEUE run {n + 1}/{len(runs)}: {tag_now()}: the saved evaluation is all instant falls (broken world); deleting it and redoing the run")
            for q in (evalp, readout_path(tag_now()), os.path.join(RES, f"train_{tag_now()}.csv")):
                if os.path.exists(q):
                    os.remove(q)
            done = False
        log(f"QUEUE run {n + 1}/{len(runs)}: {tag_now()}" + ("  (already finished, skipped)" if done else ""))
        if done:
            continue
        train(p, gait)
        eval_readout(p, gait)
        queue_report(runs)
    queue_report(runs)
    log("QUEUE finished")


def main():
    p = Pusher()
    gait = p.load_calibration()
    log(f"mode {config.MODE}, episodes {config.EPISODES}, box front face {config.BOX_FACE_X} m ahead, goal {config.GOAL_DX} m, arms {config.PUSH_ARMS}, "
        f"control step {p.ts} ms")
    if config.MODE == "arms_scan":
        arms_scan(p, gait)
    else:
        p.apply_arms(resolve_arms())
        if config.MODE == "check":
            run_check(p, gait)
        elif config.MODE == "baseline":
            evaluate(p, gait, config.BOX_FACE_X, "baseline_stop" if config.STOP_AT_GOAL else "baseline")
        elif config.MODE == "calibrate":
            calibrate(p, gait)
        elif config.MODE == "res_check":
            res_check(p, gait)
        elif config.MODE == "train":
            train(p, gait)
        elif config.MODE == "eval":
            eval_readout(p, gait)
        elif config.MODE == "queue":
            queue(p, gait)
        else:
            raise SystemExit(f"[!] unknown MODE {config.MODE}")
    log("done; the robot now keeps standing so you can look at the scene")
    q = p.gait_pose(p.stand_gait(), 0.0, 0.0)
    while True:
        p.command(q)
        p.tick()


main()

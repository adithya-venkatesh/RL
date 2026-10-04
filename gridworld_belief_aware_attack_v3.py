#!/usr/bin/env python3
"""Train 6 belief-aware attackers + 500-seed eval → worst-of-22.

Attackers (maximin cells only):
  pes Q + maximin test  × defender seeds 99999/99998/99997
  PB-GT Q + maximin test × defender seeds 99999/99998/99997

Eval: all five deployed rules × 6 BA attackers × 500 seeds.
Reuses gridworld_worst_of_16_v3.csv; worst-of-22 = min(worst16, 6 BA returns).

Does NOT overwrite older CSVs / zips. Does NOT touch arxiv_v1 or paper lock.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from gridworld_belief_aware_common_v3 import (
    GridWorld_RL_Attack_BeliefAware,
    eval_belief_aware_mean,
)
from gridworld_final_v3_protocol import DEFENDER_SEEDS, TIMESTEPS

OUT_CSV = "gridworld_belief_aware_attack_v3.csv"
OUT_SUMMARY = "gridworld_belief_aware_attack_v3_summary.csv"
OUT_WIN = "gridworld_belief_aware_attack_v3_winrates.csv"
MAPPING = "attacker_belief_aware_v3_mapping.json"
WORST16 = "gridworld_worst_of_16_v3.csv"
N_SEEDS = 500
MAX_HOURS = 3.0

CELLS = {
    "pes_q_maximin": {
        "prep": "pessimistic",  # maximin Bellman target Q
        "rng": [16001, 16002, 16003],
        "stem": "attacker_pes_q_maximin_belief_aware_final_v3",
        "lab_prefix": "ba_pes",
    },
    "pbgt_q_maximin": {
        "prep": "offpolicy",  # PB-GT Q
        "rng": [17001, 17002, 17003],
        "stem": "attacker_pbgt_q_maximin_belief_aware_final_v3",
        "lab_prefix": "ba_pbgt",
    },
}

ROWS = (
    "vanilla",
    "pes_q_maximin",
    "pes_q_greedy",
    "pbgt_q_greedy",
    "pbgt_q_maximin",
)


def ba_labels() -> List[str]:
    labs = []
    for cell, spec in CELLS.items():
        for i in range(3):
            labs.append(f"{spec['lab_prefix']}{i}")
    return labs


def ba_paths() -> Dict[str, str]:
    paths = {}
    for cell, spec in CELLS.items():
        for i in range(3):
            stem = f"{spec['stem']}_{i}"
            if not os.path.exists(stem + ".zip"):
                raise SystemExit(f"Missing {stem}.zip — train first")
            paths[f"{spec['lab_prefix']}{i}"] = stem
    return paths


def _prep(cell: str, defender_seed: int):
    from main_with_vanilla import (
        GridWorld,
        Q_Agent,
        play,
        play_att_with_belief,
        play_offpolicy_greedy_target,
    )

    env = GridWorld()
    env.seed(defender_seed)
    np.random.seed(defender_seed)
    agent = Q_Agent(env)
    play(env, agent, trials=2000, learn=True)
    if CELLS[cell]["prep"] == "pessimistic":
        play_att_with_belief(env, agent, trials=500, learn=True)
    elif CELLS[cell]["prep"] == "offpolicy":
        play_offpolicy_greedy_target(env, agent, trials=500, learn=True)
    else:
        raise ValueError(cell)
    return agent


def train_one(cell: str, a_idx: int) -> float:
    spec = CELLS[cell]
    stem = f"{spec['stem']}_{a_idx}"
    out_zip = stem + ".zip"
    if os.path.exists(out_zip):
        print(f"SKIP existing {out_zip}", flush=True)
        return 0.0
    defender_seed = DEFENDER_SEEDS[a_idx]
    atk_seed = spec["rng"][a_idx]
    print(
        f"=== Train {stem}: defender_seed={defender_seed} "
        f"attacker_rng={atk_seed} cell={cell} ===",
        flush=True,
    )
    t0 = time.perf_counter()
    agent = _prep(cell, defender_seed)
    ckpt_dir = f"RL_attack/{cell}_belief_aware_final_v3_a{a_idx}"
    os.makedirs(ckpt_dir, exist_ok=True)
    atk_env = GridWorld_RL_Attack_BeliefAware(agent, agent_type="pessimistic")
    callback = CheckpointCallback(save_freq=5000, save_path=ckpt_dir, name_prefix=stem)
    np.random.seed(atk_seed)
    model = DQN(
        "MlpPolicy",
        atk_env,
        learning_rate=1e-3,
        learning_starts=1000,
        batch_size=256,
        exploration_final_eps=0.001,
        gamma=1,
        tau=0.2,
        train_freq=(5, "step"),
        policy_kwargs={"net_arch": [256, 128]},
        tensorboard_log=f"{ckpt_dir}/",
        verbose=0,
        seed=atk_seed,
    )
    print(f"Training {stem} for {TIMESTEPS} steps...", flush=True)
    model.learn(total_timesteps=TIMESTEPS, callback=callback)
    model.save(stem)
    secs = time.perf_counter() - t0
    print(f"Saved {out_zip} in {secs/60:.1f} min", flush=True)
    return secs


def train_all() -> Dict[str, Any]:
    mapping: Dict[str, Any] = {"defenders": {}, "wall_sec_total": 0.0}
    if os.path.exists(MAPPING):
        try:
            with open(MAPPING) as f:
                prev = json.load(f)
            if isinstance(prev.get("defenders"), dict):
                mapping["defenders"] = prev["defenders"]
        except (json.JSONDecodeError, OSError):
            pass
    t_all = time.perf_counter()
    for cell, spec in CELLS.items():
        for a_idx in range(3):
            stem = f"{spec['stem']}_{a_idx}"
            secs = train_one(cell, a_idx)
            mapping["defenders"][stem] = {
                "cell": cell,
                "agent_type_test": "pessimistic",
                "belief_aware": True,
                "attacker_index": a_idx,
                "defender_seed": DEFENDER_SEEDS[a_idx],
                "attacker_rng_seed": spec["rng"][a_idx],
                "zip": stem + ".zip",
                "timesteps": TIMESTEPS,
                "obs": "state2 + belief_mask25",
                "train_wall_sec": secs,
                "skipped_existing": secs == 0.0 and os.path.exists(stem + ".zip"),
            }
            with open(MAPPING, "w") as f:
                json.dump(mapping, f, indent=2)
    mapping["wall_sec_total"] = time.perf_counter() - t_all
    with open(MAPPING, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"BA_TRAIN_DONE wall_min={mapping['wall_sec_total']/60:.1f}", flush=True)
    return mapping


def _completed(path: str) -> set:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get("seed")
            if raw not in (None, ""):
                done.add(int(float(raw)))
    return done


def fieldnames() -> List[str]:
    labs = ba_labels()
    fields = ["seed"]
    for key in ROWS:
        fields += [f"{key}_vs_{lab}" for lab in labs]
        if key in CELLS:
            fields.append(f"{key}_ba_matched")
        fields += [f"{key}_worst16", f"{key}_worst22", f"{key}_ba_sets_worst"]
    fields.append("wall_sec")
    return fields


def _load_worst16() -> Dict[int, Dict[str, float]]:
    table: Dict[int, Dict[str, float]] = {}
    with open(WORST16, newline="") as f:
        for row in csv.DictReader(f):
            seed = int(float(row["seed"]))
            table[seed] = {
                key: float(row[f"{key}_worst16"]) for key in ROWS
            }
    return table


def run_one(seed: int, paths: Dict[str, str], worst16_row: Dict[str, float]) -> Dict[str, Any]:
    from diag_tabular_v3_common import train_defenders

    t0 = time.perf_counter()
    agent_v, agent_p, agent_o, _ = train_defenders(seed, scoring="fixed")
    models = {lab: DQN.load(path) for lab, path in paths.items()}
    labs = ba_labels()
    pol_p = copy.deepcopy(agent_p.defender_policy)
    pol_o = copy.deepcopy(agent_o.defender_policy)

    agents = {
        "vanilla": (agent_v, None),
        "pes_q_maximin": (agent_p, pol_p),
        "pes_q_greedy": (agent_p, None),
        "pbgt_q_maximin": (agent_o, pol_o),
        "pbgt_q_greedy": (agent_o, None),
    }

    row: Dict[str, Any] = {"seed": seed}
    for key in ROWS:
        agent, pol = agents[key]
        vals = []
        for lab in labs:
            mean = eval_belief_aware_mean(key, agent, seed, models[lab], pol_snap=pol)
            row[f"{key}_vs_{lab}"] = mean
            vals.append(mean)
        if key in CELLS:
            prefix = CELLS[key]["lab_prefix"]
            own = [row[f"{key}_vs_{lab}"] for lab in labs if lab.startswith(prefix)]
            row[f"{key}_ba_matched"] = float(np.mean(own))
        w16 = float(worst16_row[key])
        ba_min = float(np.min(vals))
        w22 = float(min(w16, ba_min))
        row[f"{key}_worst16"] = w16
        row[f"{key}_worst22"] = w22
        # BA sets new worst iff min equals a BA return and is strictly worse than old w16
        row[f"{key}_ba_sets_worst"] = int(ba_min < w16 - 1e-12 and abs(w22 - ba_min) < 1e-12)
    row["wall_sec"] = round(time.perf_counter() - t0, 2)
    return row


def estimate_hours(probe_sec: float, n_seeds: int, workers: int) -> float:
    return (n_seeds * probe_sec) / max(workers, 1) / 3600.0


def cmd_probe(args) -> None:
    paths = ba_paths()
    w16 = _load_worst16()
    seed = args.seed
    print(f"PROBE seed={seed} ...", flush=True)
    t0 = time.perf_counter()
    row = run_one(seed, paths, w16[seed])
    sec = time.perf_counter() - t0
    est = estimate_hours(sec, args.n_seeds, args.workers)
    print(
        f"PROBE_DONE seed={seed} wall_sec={sec:.1f} "
        f"est_{args.n_seeds}seeds_{args.workers}workers_h={est:.2f} "
        f"(limit {MAX_HOURS})",
        flush=True,
    )
    for key in ("pes_q_maximin", "pbgt_q_maximin"):
        print(
            f"  {key} ba_matched={row[f'{key}_ba_matched']:.2f} "
            f"w16={row[f'{key}_worst16']:.2f} w22={row[f'{key}_worst22']:.2f}",
            flush=True,
        )
    if est > MAX_HOURS:
        print("ESTIMATE_OVER_LIMIT stop before full eval", flush=True)
        raise SystemExit(2)


def cmd_eval(args) -> None:
    paths = ba_paths()
    w16 = _load_worst16()
    wanted = list(range(args.n_seeds)) if args.only_seeds is None else list(args.only_seeds)
    done = _completed(args.out)
    missing = [s for s in wanted if s not in done]
    print(f"missing {len(missing)} of {len(wanted)} -> {args.out}", flush=True)
    if not missing:
        print("BA_EVAL_DONE nothing to do", flush=True)
        return

    # Optional 1-seed probe gate unless --force
    if not args.force and len(missing) > 5:
        probe_seed = missing[0]
        print(f"Pre-eval probe seed={probe_seed}", flush=True)
        t0 = time.perf_counter()
        probe_row = run_one(probe_seed, paths, w16[probe_seed])
        probe_sec = time.perf_counter() - t0
        est = estimate_hours(probe_sec, len(missing), args.workers)
        print(
            f"PROBE wall_sec={probe_sec:.1f} est_remaining_h={est:.2f}",
            flush=True,
        )
        if est > MAX_HOURS:
            print("ESTIMATE_OVER_LIMIT not starting full eval", flush=True)
            raise SystemExit(2)
        # Write probe row then continue with the rest
        fields = fieldnames()
        file_exists = os.path.exists(args.out) and os.path.getsize(args.out) > 0
        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                w.writeheader()
            w.writerow(probe_row)
            f.flush()
        done.add(probe_seed)
        missing = [s for s in missing if s != probe_seed]
        if not missing:
            print("BA_EVAL_DONE after probe only", flush=True)
            return

    fields = fieldnames()
    file_exists = os.path.exists(args.out) and os.path.getsize(args.out) > 0
    t_all = time.perf_counter()
    times: List[float] = []
    workers = min(args.workers, len(missing))

    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            w.writeheader()

        if workers <= 1:
            for s in missing:
                row = run_one(s, paths, w16[s])
                w.writerow(row)
                f.flush()
                times.append(row["wall_sec"])
                print(f"seed {s} {row['wall_sec']:.1f}s", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {
                    ex.submit(run_one, s, paths, w16[s]): s for s in missing
                }
                finished = 0
                for fut in as_completed(futs):
                    row = fut.result()
                    w.writerow(row)
                    f.flush()
                    finished += 1
                    times.append(row["wall_sec"])
                    avg = sum(times) / len(times)
                    eta = (len(missing) - finished) / workers * avg
                    if finished == 1 or finished % 10 == 0 or finished == len(missing):
                        print(
                            f"progress {finished}/{len(missing)} seed={row['seed']} "
                            f"last={row['wall_sec']:.1f}s mean={avg:.1f}s "
                            f"eta_h={eta/3600:.2f}",
                            flush=True,
                        )

    wall = time.perf_counter() - t_all
    print(
        f"BA_EVAL_DONE wall_min={wall/60:.2f} seeds={len(times)} "
        f"mean_seed_s={float(np.mean(times)) if times else 0:.1f}",
        flush=True,
    )


def _stats(arr: np.ndarray) -> Tuple[float, float, float]:
    return float(arr.mean()), float(arr.std(ddof=1)), float(np.median(arr))


def cmd_summarize(args) -> None:
    rows = list(csv.DictReader(open(args.out, newline="")))
    if len(rows) < 2:
        raise SystemExit(f"need >=2 seeds in {args.out}, got {len(rows)}")

    summary_fields = [
        "row",
        "ba_matched_mean",
        "ba_matched_std",
        "ba_matched_median",
        "worst16_mean",
        "worst16_std",
        "worst16_median",
        "worst22_mean",
        "worst22_std",
        "worst22_median",
        "n_ba_sets_worst",
        "frac_ba_sets_worst",
        "n",
    ]
    out_sum = []
    n = len(rows)
    for key in ROWS:
        w16 = np.asarray([float(r[f"{key}_worst16"]) for r in rows])
        w22 = np.asarray([float(r[f"{key}_worst22"]) for r in rows])
        ba_set = np.asarray([int(float(r[f"{key}_ba_sets_worst"])) for r in rows])
        rec = {
            "row": key,
            "ba_matched_mean": "",
            "ba_matched_std": "",
            "ba_matched_median": "",
            "worst16_mean": _stats(w16)[0],
            "worst16_std": _stats(w16)[1],
            "worst16_median": _stats(w16)[2],
            "worst22_mean": _stats(w22)[0],
            "worst22_std": _stats(w22)[1],
            "worst22_median": _stats(w22)[2],
            "n_ba_sets_worst": int(ba_set.sum()),
            "frac_ba_sets_worst": float(ba_set.mean()),
            "n": n,
        }
        if key in CELLS:
            m = np.asarray([float(r[f"{key}_ba_matched"]) for r in rows])
            mu, sd, med = _stats(m)
            rec["ba_matched_mean"] = mu
            rec["ba_matched_std"] = sd
            rec["ba_matched_median"] = med
        out_sum.append(rec)
        print(
            f"{key}: w16={rec['worst16_mean']:.2f}±{rec['worst16_std']:.2f} "
            f"med={rec['worst16_median']:.2f} | "
            f"w22={rec['worst22_mean']:.2f}±{rec['worst22_std']:.2f} "
            f"med={rec['worst22_median']:.2f} | "
            f"ba_sets_worst={rec['n_ba_sets_worst']}/{n}",
            flush=True,
        )
        if key in CELLS:
            print(
                f"  ba_matched={rec['ba_matched_mean']:.2f}±{rec['ba_matched_std']:.2f} "
                f"med={rec['ba_matched_median']:.2f}",
                flush=True,
            )

    with open(OUT_SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for rec in out_sum:
            w.writerow(rec)

    # W/T/L under worst-of-22
    def wtl(a, b):
        wins = ties = losses = 0
        for x, y in zip(a, b):
            if x > y + 1e-12:
                wins += 1
            elif abs(x - y) <= 1e-12:
                ties += 1
            else:
                losses += 1
        return wins, ties, losses

    pes_m = np.asarray([float(r["pes_q_maximin_worst22"]) for r in rows])
    pes_g = np.asarray([float(r["pes_q_greedy_worst22"]) for r in rows])
    pb_m = np.asarray([float(r["pbgt_q_maximin_worst22"]) for r in rows])
    pb_g = np.asarray([float(r["pbgt_q_greedy_worst22"]) for r in rows])

    comparisons = [
        ("Maximin vs greedy on pes Q (worst-of-22)", *wtl(pes_m, pes_g)),
        ("Maximin vs greedy on PB-GT Q (worst-of-22)", *wtl(pb_m, pb_g)),
        ("Pes-maximin vs PB-GT-maximin (worst-of-22)", *wtl(pes_m, pb_m)),
    ]
    win_fields = [
        "comparison",
        "wins",
        "ties",
        "losses",
        "win_rate",
        "tie_rate",
        "loss_rate",
        "metric",
    ]
    with open(OUT_WIN, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=win_fields)
        w.writeheader()
        for name, wins, ties, losses in comparisons:
            rec = {
                "comparison": name,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "win_rate": wins / n,
                "tie_rate": ties / n,
                "loss_rate": losses / n,
                "metric": "worst-of-22",
            }
            w.writerow(rec)
            print(
                f"{name}: W/T/L={wins}/{ties}/{losses} "
                f"({wins/n:.3f}/{ties/n:.3f}/{losses/n:.3f})",
                flush=True,
            )

    # Overall: any row where BA sets worst
    any_ba = 0
    for r in rows:
        if any(int(float(r[f"{k}_ba_sets_worst"])) for k in ROWS):
            any_ba += 1
    print(f"seeds_any_row_ba_sets_worst={any_ba}/{n}", flush=True)
    print(f"Wrote {OUT_SUMMARY} and {OUT_WIN}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "cmd",
        choices=["estimate", "train", "probe", "eval", "summarize", "all"],
    )
    p.add_argument("--n-seeds", type=int, default=N_SEEDS)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=OUT_CSV)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only-seeds", type=int, nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.out != OUT_CSV and not args.out.endswith("_v3.csv"):
        raise SystemExit(f"refusing non-_v3 path {args.out}")

    if args.cmd == "estimate":
        # Prior rates (no probe yet)
        train_min = 6 * 2.1
        # matched15: ~299s/seed for ~32 maximin + 32 greedy @ serial
        # our job: 12 maximin + 18 greedy ≈ 0.47× attack load → ~140s + regen share
        # van_vs_swap adds context: 6 greedy ~46s incl train
        # Conservative serial ~160s/seed; wall @8 workers:
        eval_h = estimate_hours(160.0, args.n_seeds, args.workers)
        total_h = train_min / 60.0 + eval_h
        print(
            f"ESTIMATE train≈{train_min:.0f}min eval≈{eval_h:.2f}h "
            f"total≈{total_h:.2f}h at {args.workers} workers "
            f"(reuse worst-of-16; 5×6 BA evals only). "
            f"limit={MAX_HOURS}h → "
            f"{'PROCEED' if total_h <= MAX_HOURS else 'STOP_BEFORE_EVAL'}",
            flush=True,
        )
        return

    if args.cmd in ("train", "all"):
        train_all()
    if args.cmd == "probe":
        cmd_probe(args)
    if args.cmd in ("eval", "all"):
        cmd_eval(args)
    if args.cmd in ("summarize", "all"):
        if _completed(args.out):
            cmd_summarize(args)
        else:
            print("summarize skipped: no eval CSV yet", flush=True)


if __name__ == "__main__":
    main()

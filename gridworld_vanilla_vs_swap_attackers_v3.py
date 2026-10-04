#!/usr/bin/env python3
"""Score regenerated vanilla Q-tables against the six swapped-rule attackers.

Does not train attackers. Does not modify gridworld_experiments_500runs_final_v3.csv.
Vanilla worst-of-15 is the per-seed min of the original nine columns in that file
and the six scores written here.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List

# One thread per worker so a single-seed timing scales with the pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from stable_baselines3 import DQN

OUT = "gridworld_vanilla_vs_swap_attackers_v3.csv"
FINAL_CSV = "gridworld_experiments_500runs_final_v3.csv"
ORIG9 = [f"{p}{i}" for p in ("van", "pes", "pbgt") for i in range(3)]
STEMS = {
    "g": [f"attacker_pes_q_greedy_final_v3_{i}" for i in range(3)],
    "m": [f"attacker_pbgt_q_maximin_final_v3_{i}" for i in range(3)],
}


def labels() -> List[str]:
    return [f"g{i}" for i in range(3)] + [f"m{i}" for i in range(3)]


def paths() -> Dict[str, str]:
    out = {}
    for prefix, stems in STEMS.items():
        for i, stem in enumerate(stems):
            zip_path = stem + ".zip"
            if not os.path.exists(zip_path):
                raise SystemExit(f"Missing {zip_path}")
            out[f"{prefix}{i}"] = stem
    return out


def fieldnames() -> List[str]:
    labs = labels()
    return (
        ["seed", "vanilla_clean"]
        + [f"van_vs_{lab}" for lab in labs]
        + ["van_min_of_6", "van_worst15", "wall_sec"]
    )


def _load_final() -> Dict[int, Dict[str, float]]:
    table: Dict[int, Dict[str, float]] = {}
    with open(FINAL_CSV, newline="") as f:
        for row in csv.DictReader(f):
            seed = int(float(row["seed"]))
            table[seed] = {lab: float(row[f"van_vs_{lab}"]) for lab in ORIG9}
            table[seed]["vanilla_clean"] = float(row["vanilla_clean"])
    return table


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


def run_one(seed: int, attacker_paths: Dict[str, str]) -> Dict[str, Any]:
    from diag_tabular_v3_common import apply_scoring_mode, eval_attack
    from main_with_vanilla import GridWorld, Q_Agent, play, play_vanilla_test

    apply_scoring_mode("fixed")
    t0 = time.perf_counter()
    env = GridWorld()
    env.seed(seed)
    agent = Q_Agent(env)
    np.random.seed(seed + 1000)
    play(env, agent, trials=2000, learn=True)
    np.random.seed(seed + 1000)
    clean_rewards = play_vanilla_test(env, agent, trials=100)
    models = {lab: DQN.load(path) for lab, path in attacker_paths.items()}
    row: Dict[str, Any] = {
        "seed": seed,
        "vanilla_clean": float(np.mean(clean_rewards)),
    }
    scores = []
    for lab in labels():
        mean, _, _ = eval_attack("vanilla", agent, seed, models[lab], seed)
        row[f"van_vs_{lab}"] = mean
        scores.append(mean)
    row["van_min_of_6"] = float(np.min(scores))
    row["wall_sec"] = round(time.perf_counter() - t0, 2)
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=500)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=OUT)
    p.add_argument("--only-seeds", type=int, nargs="*", default=None)
    args = p.parse_args()
    if args.out == FINAL_CSV or not args.out.endswith("_v3.csv"):
        raise SystemExit(f"refusing path {args.out}")
    attacker_paths = paths()
    final = _load_final()
    wanted = list(range(args.n_seeds)) if args.only_seeds is None else list(args.only_seeds)
    done = _completed(args.out)
    missing = [s for s in wanted if s not in done]
    print(f"missing {len(missing)} of {len(wanted)} -> {args.out}", flush=True)
    if not missing:
        print("VANILLA_SWAP_DONE nothing to do", flush=True)
        return
    fields = fieldnames()
    file_exists = os.path.exists(args.out) and os.path.getsize(args.out) > 0
    t0 = time.perf_counter()
    times: List[float] = []
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        if not file_exists:
            w.writeheader()

        def _write(row: Dict[str, Any]) -> None:
            seed = int(row["seed"])
            base = final[seed]
            if abs(float(row["vanilla_clean"]) - base["vanilla_clean"]) > 1e-6:
                raise SystemExit(
                    f"seed {seed} vanilla_clean {row['vanilla_clean']} != "
                    f"final_v3 {base['vanilla_clean']}"
                )
            six = [float(row[f"van_vs_{lab}"]) for lab in labels()]
            nine = [base[lab] for lab in ORIG9]
            row["van_worst15"] = float(np.min(nine + six))
            w.writerow(row)
            f.flush()

        workers = min(args.workers, len(missing))
        if workers <= 1:
            for s in missing:
                row = run_one(s, attacker_paths)
                _write(row)
                times.append(float(row["wall_sec"]))
                avg = sum(times) / len(times)
                print(
                    f"seed {row['seed']} clean={row['vanilla_clean']:.2f} "
                    f"{row['wall_sec']:.1f}s mean={avg:.1f}s "
                    f"est_500_at_{args.workers}workers_h="
                    f"{(500 / max(args.workers, 1)) * avg / 3600:.2f}",
                    flush=True,
                )
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(run_one, s, attacker_paths): s for s in missing}
                finished = 0
                for fut in as_completed(futs):
                    row = fut.result()
                    _write(row)
                    finished += 1
                    times.append(float(row["wall_sec"]))
                    avg = sum(times) / len(times)
                    eta = (len(missing) - finished) / workers * avg
                    print(
                        f"progress {finished}/{len(missing)} seed={row['seed']} "
                        f"clean={row['vanilla_clean']:.2f} last={row['wall_sec']:.1f}s "
                        f"mean={avg:.1f}s eta_h={eta/3600:.2f}",
                        flush=True,
                    )
    wall = time.perf_counter() - t0
    print(f"VANILLA_SWAP_PHASE wall_min={wall/60:.2f} seeds={len(missing)}", flush=True)


if __name__ == "__main__":
    main()

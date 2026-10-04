#!/usr/bin/env python3
"""Resume the 500-seed tabular v3 experiment after sleep/crash.

Reads existing gridworld_experiments_500runs_min_v3.csv, finds missing seeds in
0..499, and APPENDS only those rows. Never truncates/overwrites the CSV.
Uses the same v3 attackers and 8 workers as gridworld_500runs_min_v3.py.
Does not touch non-_v3 attacker zip files.
"""
from __future__ import annotations

import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Set, Tuple

from stable_baselines3 import DQN

OUT_CSV = "gridworld_experiments_500runs_min_v3.csv"
N_SEEDS = 500
N_WORKERS = 8
PES_ATK = "attacker_pessimistic_v3"
OFF_ATK = "attacker_offpolicy_v3"


def _completed_seeds(path: str) -> Set[int]:
    if not os.path.exists(path):
        return set()
    done: Set[int] = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Prefer seed; fall back to experiment_number (they match in v3 runs).
            raw = row.get("seed") or row.get("experiment_number")
            if raw is None or raw == "":
                continue
            done.add(int(float(raw)))
    return done


def _run_one(seed: int) -> Tuple[int, Dict[str, Any], float]:
    from main_with_vanilla import run_single_experiment_with_seed

    attack_models = {
        "pessimistic": DQN.load(PES_ATK),
        "offpolicy": DQN.load(OFF_ATK),
    }
    t0 = time.perf_counter()
    results, *_ = run_single_experiment_with_seed(
        experiment_num=seed, seed=seed, attack_models=attack_models
    )
    return seed, results, time.perf_counter() - t0


def main() -> None:
    for stem in (PES_ATK, OFF_ATK):
        if not os.path.exists(stem + ".zip"):
            raise SystemExit(f"Missing {stem}.zip — retrain attackers first")

    done = _completed_seeds(OUT_CSV)
    missing = [s for s in range(N_SEEDS) if s not in done]
    print(
        f"Found {len(done)} completed seed(s) in {OUT_CSV}; "
        f"{len(missing)} missing of {N_SEEDS}",
        flush=True,
    )
    if not missing:
        print("Nothing to do — all seeds 0..499 already present.", flush=True)
        return

    # Append-only via log_experiment_to_csv (open mode 'a'). Never truncate.
    from main_with_vanilla import log_experiment_to_csv

    t_all = time.perf_counter()
    finished = 0
    times = []
    print(
        f"Resuming {len(missing)} seeds with {N_WORKERS} workers → append {OUT_CSV}",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_run_one, s): s for s in missing}
        for fut in as_completed(futs):
            seed, results, secs = fut.result()
            log_experiment_to_csv(seed, results, filename=OUT_CSV)
            finished += 1
            times.append(secs)
            if finished % 10 == 0 or finished == len(missing):
                avg = sum(times) / len(times)
                remain = (len(missing) - finished) / N_WORKERS * avg
                print(
                    f"resume progress {finished}/{len(missing)} last_seed={seed} "
                    f"last_sec={secs:.1f} mean_sec={avg:.1f} eta_min={remain/60:.1f}",
                    flush=True,
                )

    print(
        f"GRIDWORLD_500_V3_RESUME_DONE wall_min={(time.perf_counter()-t_all)/60:.1f} "
        f"appended={finished}",
        flush=True,
    )


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()

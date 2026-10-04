#!/usr/bin/env python3
"""Parallel 500-seed tabular eval with fixed get_att_state + v3 attackers.

Uses 8 workers, seed=exp_num per process. Writes gridworld_experiments_500runs_min_v3.csv
only (never overwrites older CSVs). Vanilla faces the pessimistic attacker (paper protocol).
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Optional, Tuple

from stable_baselines3 import DQN

OUT_CSV = "gridworld_experiments_500runs_min_v3.csv"
N_SEEDS = 500
N_WORKERS = 8
PES_ATK = "attacker_pessimistic_v3"
OFF_ATK = "attacker_offpolicy_v3"


def _run_one(seed: int) -> Tuple[int, Dict[str, Any], float]:
    # Import inside worker so each process gets a fresh module / RNG isolation.
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
    if os.path.exists(OUT_CSV):
        raise SystemExit(f"Refusing to overwrite existing {OUT_CSV}")
    for stem in (PES_ATK, OFF_ATK):
        if not os.path.exists(stem + ".zip"):
            raise SystemExit(f"Missing {stem}.zip — retrain attackers first")

    # Parent writes CSV serially as workers finish (thread-safe append via lock of GIL + one writer).
    from main_with_vanilla import log_experiment_to_csv

    t_all = time.perf_counter()
    done = 0
    times = []
    print(f"Starting {N_SEEDS} seeds with {N_WORKERS} workers → {OUT_CSV}", flush=True)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_run_one, s): s for s in range(N_SEEDS)}
        for fut in as_completed(futs):
            seed, results, secs = fut.result()
            log_experiment_to_csv(seed, results, filename=OUT_CSV)
            done += 1
            times.append(secs)
            if done % 10 == 0 or done == N_SEEDS:
                avg = sum(times) / len(times)
                remain = (N_SEEDS - done) / N_WORKERS * avg
                print(
                    f"progress {done}/{N_SEEDS} last_seed={seed} "
                    f"last_sec={secs:.1f} mean_sec={avg:.1f} eta_min={remain/60:.1f}",
                    flush=True,
                )

    print(f"GRIDWORLD_500_V3_DONE wall_min={(time.perf_counter()-t_all)/60:.1f}", flush=True)


if __name__ == "__main__":
    # Required on macOS for SB3 + multiprocessing.
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()

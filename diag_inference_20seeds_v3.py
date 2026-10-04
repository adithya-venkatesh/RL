#!/usr/bin/env python3
"""Single-process attack-path inference timing, seeds 0..19, v3 attackers.

Vanilla uses pes attacker (paper protocol). Writes diag_inference_20seeds_v3.csv.
"""
from __future__ import annotations

import csv
import os
import time
from typing import Any, Dict, List

import numpy as np
from stable_baselines3 import DQN

from diag_tabular_v3_common import apply_scoring_mode
from main_with_vanilla import run_single_experiment_with_seed

OUT_CSV = "diag_inference_20seeds_v3.csv"
N_SEEDS = 20
PES_ATK = "attacker_pessimistic_v3"
OFF_ATK = "attacker_offpolicy_v3"


def main() -> None:
    if os.path.exists(OUT_CSV):
        raise SystemExit(f"Refusing to overwrite existing {OUT_CSV}")
    for p in (PES_ATK, OFF_ATK):
        if not os.path.exists(p + ".zip"):
            raise SystemExit(f"Missing {p}.zip")

    print(
        "=== ESTIMATE: inference 20 seeds single-process ≈ 8–15 min; proceeding ===",
        flush=True,
    )
    apply_scoring_mode("fixed")
    attack_models = {
        "pessimistic": DQN.load(PES_ATK),
        "offpolicy": DQN.load(OFF_ATK),
    }

    t_all = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    fieldnames = [
        "seed",
        "vanilla_attack_inference_time_ms",
        "pessimistic_attack_inference_time_ms",
        "offpolicy_attack_inference_time_ms",
        "wall_sec",
    ]

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for seed in range(N_SEEDS):
            t0 = time.perf_counter()
            results, *_ = run_single_experiment_with_seed(
                experiment_num=seed, seed=seed, attack_models=attack_models
            )
            row = {
                "seed": seed,
                "vanilla_attack_inference_time_ms": results.get(
                    "vanilla_attack_inference_time_ms", np.nan
                ),
                "pessimistic_attack_inference_time_ms": results.get(
                    "pessimistic_attack_inference_time_ms", np.nan
                ),
                "offpolicy_attack_inference_time_ms": results.get(
                    "offpolicy_attack_inference_time_ms", np.nan
                ),
                "wall_sec": time.perf_counter() - t0,
            }
            w.writerow(row)
            f.flush()
            rows.append(row)
            print(
                f"seed={seed} van={row['vanilla_attack_inference_time_ms']:.4f} "
                f"pes={row['pessimistic_attack_inference_time_ms']:.4f} "
                f"pbgt={row['offpolicy_attack_inference_time_ms']:.4f} "
                f"sec={row['wall_sec']:.1f}",
                flush=True,
            )

    print("\n=== Attack inference ms mean±std (n=20, single process) ===", flush=True)
    for col, name in [
        ("vanilla_attack_inference_time_ms", "vanilla"),
        ("pessimistic_attack_inference_time_ms", "pes"),
        ("offpolicy_attack_inference_time_ms", "PB-GT"),
    ]:
        a = np.array([r[col] for r in rows], dtype=float)
        print(f"{name}: {a.mean():.4f}±{a.std(ddof=1):.4f}", flush=True)

    print(
        f"DIAG_INFERENCE_DONE wall_min={(time.perf_counter()-t_all)/60:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

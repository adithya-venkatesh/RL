#!/usr/bin/env python3
"""Summarize gridworld_experiments_500runs_min_v3.csv for the paper update list."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "gridworld_experiments_500runs_min_v3.csv"


def ms(a: np.ndarray) -> str:
    return f"{a.mean():.2f}±{a.std(ddof=1):.2f}"


def main() -> None:
    df = pd.read_csv(CSV)
    assert len(df) == 500, len(df)

    clean_v = df["vanilla_test_avg_reward"].to_numpy()
    clean_p = df["pessimistic_test_avg_reward"].to_numpy()
    clean_o = df["offpolicy_greedy_test_avg_reward"].to_numpy()
    atk_v = df["vanilla_vs_attacker_reward"].to_numpy()
    atk_p = df["pessimistic_vs_attacker_reward"].to_numpy()
    atk_o = df["offpolicy_greedy_vs_attacker_reward"].to_numpy()

    print("=== CLEAN mean±std (sample) ===")
    print(f"vanilla {ms(clean_v)}")
    print(f"pes     {ms(clean_p)}")
    print(f"PB-GT   {ms(clean_o)}")

    print("=== ATTACK mean±std (sample) ===")
    print(f"vanilla {ms(atk_v)}")
    print(f"pes     {ms(atk_p)}")
    print(f"PB-GT   {ms(atk_o)}")

    print("=== ATTACK median / IQR ===")
    for name, a in [("vanilla", atk_v), ("pes", atk_p), ("PB-GT", atk_o)]:
        q1, med, q3 = np.percentile(a, [25, 50, 75])
        print(f"{name} median={med:.2f} IQR=[{q1:.2f},{q3:.2f}]")

    print("=== per-seed win rates (attack return) ===")
    print(f"pes > PB-GT:  {(atk_p > atk_o).mean()*100:.1f}%")
    print(f"pes > van:    {(atk_p > atk_v).mean()*100:.1f}%")
    print(f"PB-GT > van:  {(atk_o > atk_v).mean()*100:.1f}%")

    print("=== % seeds attack return > -100 ===")
    for name, a in [("vanilla", atk_v), ("pes", atk_p), ("PB-GT", atk_o)]:
        print(f"{name}: {(a > -100).mean()*100:.1f}%")

    print("=== inference ms (attack columns when present, else clean) ===")
    for label, col in [
        ("vanilla_attack", "vanilla_attack_inference_time_ms"),
        ("pes_attack", "pessimistic_attack_inference_time_ms"),
        ("PB-GT_attack", "offpolicy_attack_inference_time_ms"),
        ("vanilla_clean", "vanilla_inference_time_ms"),
        ("pes_clean", "pessimistic_inference_time_ms"),
        ("PB-GT_clean", "offpolicy_greedy_inference_time_ms"),
    ]:
        if col in df.columns:
            a = df[col].to_numpy(dtype=float)
            a = a[np.isfinite(a)]
            if len(a):
                print(f"{label}: {a.mean():.4f}±{a.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()

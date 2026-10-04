#!/usr/bin/env python3
"""Print Bellman-target x test-rule 2x2 tables from the swap CSVs.

Tabular numbers are mean ± sample std (ddof=1) of per-seed means, same units
as gridworld_experiments_500runs_final_v3.csv.
"""
import csv
import sys

import numpy as np


def ms(a):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return "n/a"
    if len(a) == 1:
        return f"{a[0]:.2f}"
    return f"{a.mean():.2f}±{a.std(ddof=1):.2f}"


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    tab_path = sys.argv[1] if len(sys.argv) > 1 else "gridworld_target_test_swap_tabular_v3.csv"
    rows = load(tab_path)
    print(f"tabular rows={len(rows)} from {tab_path}")
    # Published diagonal (final v3 CSV) for reference, not a substitute for the paired cells.
    pub = {}
    with open("gridworld_experiments_500runs_final_v3.csv", newline="") as f:
        for r in csv.DictReader(f):
            pub[int(float(r["seed"]))] = r
    match = 0
    for r in rows:
        s = int(float(r["seed"]))
        if s not in pub:
            continue
        ok = (
            abs(float(r["regen_vanilla_clean"]) - float(pub[s]["vanilla_clean"])) < 1e-6
            and abs(float(r["regen_pes_clean"]) - float(pub[s]["pes_clean"])) < 1e-6
            and abs(float(r["regen_pbgt_clean"]) - float(pub[s]["pbgt_clean"])) < 1e-6
        )
        match += int(ok)
    print(f"regen clean matches final_v3 CSV on {match}/{len(rows)} seeds")

    cells = [
        ("maximin target (pes Q)", "maximin test", "pes_q_maximin"),
        ("maximin target (pes Q)", "greedy test", "pes_q_greedy"),
        ("greedy target (PB-GT Q)", "maximin test", "pbgt_q_maximin"),
        ("greedy target (PB-GT Q)", "greedy test", "pbgt_q_greedy"),
    ]
    print("\nTabular 2x2  (mean±std of per-seed means, ddof=1)")
    print(f"{'Bellman':<28} {'test':<14} {'clean':>16} {'matched':>16} {'worst-of-9':>16}")
    for bell, test, key in cells:
        print(f"{bell:<28} {test:<14} {ms([r[key+'_clean'] for r in rows]):>16} "
              f"{ms([r[key+'_matched'] for r in rows]):>16} "
              f"{ms([r[key+'_worst9'] for r in rows]):>16}")


if __name__ == "__main__":
    main()

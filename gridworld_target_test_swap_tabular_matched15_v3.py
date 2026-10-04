#!/usr/bin/env python3
"""2x2 tabular target x test with true matched attackers (15 total).

Original nine final_v3 attackers plus six swap-matched attackers.
Each seed regenerates both Q-tables once, then evaluates all four cells
against all 15 attackers.

matched = mean of that cell's own three attackers:
  pes Q + maximin test  -> original pes0,1,2
  pes Q + greedy test   -> attacker_pes_q_greedy_final_v3_{0,1,2}
  PB-GT Q + maximin test -> attacker_pbgt_q_maximin_final_v3_{0,1,2}
  PB-GT Q + greedy test  -> original pbgt0,1,2
worst-of-15 = per-seed min over all 15.

Append/resume. Does not touch the nine-attacker swap CSV.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np
from stable_baselines3 import DQN

from gridworld_final_v3_protocol import attacker_labels, _attacker_paths_from_disk
from gridworld_target_test_swap_tabular_v3 import _StayAttacker, _greedy_mean, _maximin_mean

OUT = "gridworld_target_test_swap_tabular_matched15_v3.csv"
NEW_STEMS = {
    "g": [f"attacker_pes_q_greedy_final_v3_{i}" for i in range(3)],
    "m": [f"attacker_pbgt_q_maximin_final_v3_{i}" for i in range(3)],
}


def new_labels() -> List[str]:
    return [f"g{i}" for i in range(3)] + [f"m{i}" for i in range(3)]


def all_labels() -> List[str]:
    return attacker_labels() + new_labels()


def _paths() -> Dict[str, str]:
    paths = _attacker_paths_from_disk()
    for i, stem in enumerate(NEW_STEMS["g"]):
        if not os.path.exists(stem + ".zip"):
            raise SystemExit(f"Missing {stem}.zip — train swap attackers first")
        paths[f"g{i}"] = stem
    for i, stem in enumerate(NEW_STEMS["m"]):
        if not os.path.exists(stem + ".zip"):
            raise SystemExit(f"Missing {stem}.zip — train swap attackers first")
        paths[f"m{i}"] = stem
    return paths


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
    labs = all_labels()
    fields = ["seed", "regen_vanilla_clean", "regen_pes_clean", "regen_pbgt_clean"]
    for key in ("pes_q_maximin", "pes_q_greedy", "pbgt_q_maximin", "pbgt_q_greedy"):
        fields.append(f"{key}_clean")
        fields += [f"{key}_vs_{lab}" for lab in labs]
        fields += [f"{key}_matched", f"{key}_worst15"]
    fields.append("wall_sec")
    return fields


OWN = {
    "pes_q_maximin": ("pes0", "pes1", "pes2"),
    "pes_q_greedy": ("g0", "g1", "g2"),
    "pbgt_q_maximin": ("m0", "m1", "m2"),
    "pbgt_q_greedy": ("pbgt0", "pbgt1", "pbgt2"),
}


def _fill(row, agent, seed, rule, key, labs, models, pol_snap, stay) -> None:
    if rule == "greedy":
        clean = _greedy_mean(agent, seed, None)
        vals = {lab: _greedy_mean(agent, seed, models[lab]) for lab in labs}
    else:
        clean = _maximin_mean(agent, seed, stay, pol_snap)
        vals = {lab: _maximin_mean(agent, seed, models[lab], pol_snap) for lab in labs}
    row[f"{key}_clean"] = clean
    for lab, mean in vals.items():
        row[f"{key}_vs_{lab}"] = mean
    own = [vals[lab] for lab in OWN[key]]
    row[f"{key}_matched"] = float(np.mean(own))
    row[f"{key}_worst15"] = float(np.min(list(vals.values())))


def run_one(seed: int, paths: Dict[str, str]) -> Dict[str, Any]:
    from diag_tabular_v3_common import train_defenders

    t0 = time.perf_counter()
    _, agent_p, agent_o, cleans = train_defenders(seed, scoring="fixed")
    models = {lab: DQN.load(path) for lab, path in paths.items()}
    labs = all_labels()
    stay = _StayAttacker()
    pol_p = copy.deepcopy(agent_p.defender_policy)
    pol_o = copy.deepcopy(agent_o.defender_policy)
    row: Dict[str, Any] = {
        "seed": seed,
        "regen_vanilla_clean": cleans["vanilla_clean"],
        "regen_pes_clean": cleans["pes_clean"],
        "regen_pbgt_clean": cleans["pbgt_clean"],
    }
    _fill(row, agent_p, seed, "maximin", "pes_q_maximin", labs, models, pol_p, stay)
    _fill(row, agent_p, seed, "greedy", "pes_q_greedy", labs, models, pol_p, stay)
    _fill(row, agent_o, seed, "maximin", "pbgt_q_maximin", labs, models, pol_o, stay)
    _fill(row, agent_o, seed, "greedy", "pbgt_q_greedy", labs, models, pol_o, stay)
    row["wall_sec"] = round(time.perf_counter() - t0, 2)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=500)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=OUT)
    p.add_argument("--only-seeds", type=int, nargs="*", default=None)
    args = p.parse_args()
    if not args.out.endswith("_v3.csv"):
        raise SystemExit(f"refusing non-_v3 path {args.out}")
    paths = _paths()
    wanted = list(range(args.n_seeds)) if args.only_seeds is None else list(args.only_seeds)
    done = _completed(args.out)
    missing = [s for s in wanted if s not in done]
    print(f"missing {len(missing)} of {len(wanted)} -> {args.out}", flush=True)
    if not missing:
        print("MATCHED15_DONE nothing to do", flush=True)
        return
    fields = fieldnames()
    file_exists = os.path.exists(args.out) and os.path.getsize(args.out) > 0
    t0 = time.perf_counter()
    times = []
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        if not file_exists:
            w.writeheader()
        workers = min(args.workers, len(missing))
        if workers <= 1:
            for s in missing:
                row = run_one(s, paths)
                w.writerow(row)
                f.flush()
                times.append(row["wall_sec"])
                print(f"seed {s} {row['wall_sec']:.1f}s", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(run_one, s, paths): s for s in missing}
                finished = 0
                for fut in as_completed(futs):
                    row = fut.result()
                    w.writerow(row)
                    f.flush()
                    finished += 1
                    times.append(row["wall_sec"])
                    avg = sum(times) / len(times)
                    eta = (len(missing) - finished) / workers * avg
                    print(
                        f"progress {finished}/{len(missing)} seed={row['seed']} "
                        f"last={row['wall_sec']:.1f}s mean={avg:.1f}s eta_h={eta/3600:.2f}",
                        flush=True,
                    )
    wall = time.perf_counter() - t0
    print(f"MATCHED15_PHASE wall_min={wall/60:.2f} seeds={len(missing)}", flush=True)


if __name__ == "__main__":
    main()

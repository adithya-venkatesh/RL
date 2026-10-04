#!/usr/bin/env python3
"""Bellman-target × test-rule swap for the final v3 tabular protocol.

No new Q-learning protocol. Q-tables were not saved, so each seed regenerates
them with diag_tabular_v3_common.train_defenders (same seeds, same code path).

New cells only:
  (i)  pessimistic Q, GREEDY test on s_tilde
  (ii) PB-GT Q, belief MAXIMIN test
Attackers are the existing nine final_v3 SB3 zips (not retrained).
Matched = mean of the Q-table's own three attackers.
worst-of-9 = per-seed minimum over all nine.

Writes gridworld_target_test_swap_tabular_v3.csv (append/resume, never truncates).
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

OUT_CSV = "gridworld_target_test_swap_tabular_v3.csv"


class _StayAttacker:
    """Identity observation: s_tilde = s, so the test rule runs on the clean state."""

    def predict(self, obs, deterministic=True):
        return (4, None)


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


def _own(labs: List[str], prefix: str, row: Dict[str, Any], key: str) -> List[float]:
    return [row[f"{key}_vs_{lab}"] for lab in labs if lab.startswith(prefix)]


def _greedy_mean(agent, seed: int, attack_model) -> float:
    from main_with_vanilla import GridWorld, play_offpolicy_greedy_target

    env = GridWorld()
    env.seed(seed)
    np.random.seed(seed + 1000)
    rew = play_offpolicy_greedy_target(
        env, agent, trials=100, learn=False, attack_model=attack_model
    )
    return float(np.mean(rew))


def _maximin_mean(agent, seed: int, attack_model, pol_snap) -> float:
    from main_with_vanilla import GridWorld, play_att_with_belief

    agent.defender_policy = copy.deepcopy(pol_snap)
    env = GridWorld()
    env.seed(seed)
    np.random.seed(seed + 1000)
    rew = play_att_with_belief(
        env, agent, trials=100, learn=False, attack_model=attack_model
    )
    return float(np.mean(rew))


def _fill_cell(row, agent, seed, rule, key, labs, models, pol_snap, stay) -> None:
    """Clean + nine attackers + matched (own method) + worst-of-9, one test rule."""
    if rule == "greedy":
        clean = _greedy_mean(agent, seed, None)
        vals = [_greedy_mean(agent, seed, models[lab]) for lab in labs]
    else:
        clean = _maximin_mean(agent, seed, stay, pol_snap)
        vals = [_maximin_mean(agent, seed, models[lab], pol_snap) for lab in labs]
    row[f"{key}_clean"] = clean
    for lab, mean in zip(labs, vals):
        row[f"{key}_vs_{lab}"] = mean
    prefix = "pes" if key.startswith("pes") else "pbgt"
    own = [row[f"{key}_vs_{lab}"] for lab in labs if lab.startswith(prefix)]
    row[f"{key}_matched"] = float(np.mean(own))
    row[f"{key}_worst9"] = float(np.min(vals))


def run_one(seed: int, paths: Dict[str, str]) -> Dict[str, Any]:
    """All four target x test cells on ONE regenerated Q pair.

    Pessimistic tabular training breaks ties via set iteration, which depends on
    PYTHONHASHSEED, so a later process does not recover the CSV's Q-table.
    The published diagonal is therefore not paired with a fresh swap. This worker
    evaluates both test rules for both Q-tables so the 2x2 shares the realization.
    """
    from diag_tabular_v3_common import train_defenders

    t0 = time.perf_counter()
    _, agent_p, agent_o, cleans = train_defenders(seed, scoring="fixed")
    models = {lab: DQN.load(path) for lab, path in paths.items()}
    labs = attacker_labels()
    stay = _StayAttacker()
    pol_p = copy.deepcopy(agent_p.defender_policy)
    pol_o = copy.deepcopy(agent_o.defender_policy)

    row: Dict[str, Any] = {
        "seed": seed,
        "regen_vanilla_clean": cleans["vanilla_clean"],
        "regen_pes_clean": cleans["pes_clean"],
        "regen_pbgt_clean": cleans["pbgt_clean"],
    }
    # pes Q was trained with the maximin Bellman target; pbgt with the greedy target.
    _fill_cell(row, agent_p, seed, "maximin", "pes_q_maximin", labs, models, pol_p, stay)
    _fill_cell(row, agent_p, seed, "greedy", "pes_q_greedy", labs, models, pol_p, stay)
    _fill_cell(row, agent_o, seed, "maximin", "pbgt_q_maximin", labs, models, pol_o, stay)
    _fill_cell(row, agent_o, seed, "greedy", "pbgt_q_greedy", labs, models, pol_o, stay)
    row["wall_sec"] = round(time.perf_counter() - t0, 2)
    return row


def fieldnames() -> List[str]:
    labs = attacker_labels()
    fields = ["seed", "regen_vanilla_clean", "regen_pes_clean", "regen_pbgt_clean"]
    for key in ("pes_q_maximin", "pes_q_greedy", "pbgt_q_maximin", "pbgt_q_greedy"):
        fields.append(f"{key}_clean")
        fields += [f"{key}_vs_{lab}" for lab in labs]
        fields += [f"{key}_matched", f"{key}_worst9"]
    fields.append("wall_sec")
    return fields


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=500)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=str, default=OUT_CSV)
    args = p.parse_args()
    if os.path.exists(args.out) and not args.out.endswith("_v3.csv"):
        raise SystemExit(f"refusing to touch non-_v3 path {args.out}")

    paths = _attacker_paths_from_disk()
    fields = fieldnames()
    already = _completed(args.out)
    missing = [s for s in range(args.n_seeds) if s not in already]
    print(f"CSV={args.out} done={len(already)} missing={len(missing)} workers={args.workers}", flush=True)
    if not missing:
        print("NOTHING_TO_DO", flush=True)
        return

    t_all = time.perf_counter()
    file_exists = os.path.exists(args.out) and os.path.getsize(args.out) > 0
    times: List[float] = []
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            w.writeheader()
        if args.workers <= 1 or len(missing) == 1:
            iterator = ((s, run_one(s, paths)) for s in missing)
            for seed, row in iterator:
                w.writerow(row)
                f.flush()
                times.append(row["wall_sec"])
                print(f"seed={seed} wall={row['wall_sec']:.1f}s "
                      f"pes_max={row['pes_q_maximin_clean']:.2f} pes_g={row['pes_q_greedy_clean']:.2f} "
                      f"pbgt_max={row['pbgt_q_maximin_clean']:.2f} pbgt_g={row['pbgt_q_greedy_clean']:.2f}",
                      flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(run_one, s, paths): s for s in missing}
                done = 0
                for fut in as_completed(futs):
                    row = fut.result()
                    w.writerow(row)
                    f.flush()
                    done += 1
                    times.append(row["wall_sec"])
                    avg = sum(times) / len(times)
                    eta = (len(missing) - done) / args.workers * avg
                    if done == 1 or done % 10 == 0 or done == len(missing):
                        print(f"progress {done}/{len(missing)} seed={row['seed']} "
                              f"last={row['wall_sec']:.1f}s mean={avg:.1f}s eta_min={eta/60:.1f} "
                              f"pes_max={row['pes_q_maximin_clean']:.2f} "
                              f"pes_g={row['pes_q_greedy_clean']:.2f} "
                              f"pbgt_max={row['pbgt_q_maximin_clean']:.2f} "
                              f"pbgt_g={row['pbgt_q_greedy_clean']:.2f}", flush=True)
    wall = time.perf_counter() - t_all
    print(f"SWAP_TABULAR_DONE wall_min={wall/60:.2f} n={len(times)} mean_seed_s={np.mean(times):.1f}",
          flush=True)


if __name__ == "__main__":
    main()

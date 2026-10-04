#!/usr/bin/env python3
"""PGD-10 on the continuous swap off-diagonals, seeds 0-4 only.

Loads the already trained nets. Does not retrain attackers or overwrite
gridworld_target_test_swap_cgw_v3.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import torch

import gridworld_continuous_dqn as G
import gridworld_continuous_prod as P
from gridworld_target_test_swap_cgw_v3 import make_args

OUT = "gridworld_target_test_swap_cgw_pgd_offdiag_v3.csv"
FIELDS = ["seed", "eps", "cell", "condition", "mean", "std", "ms_per_action",
          "box_width", "ball_width", "episodes"]


def _done(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            done.add((int(r["seed"]), r["cell"]))
    return done


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--out", default=OUT)
    ns = p.parse_args()
    if not ns.out.endswith("_v3.csv"):
        raise SystemExit("refusing non-_v3 path")
    args = make_args(ns.seeds)
    args.episodes = ns.episodes
    done = _done(ns.out)
    file_exists = os.path.exists(ns.out) and os.path.getsize(ns.out) > 0
    t_all = time.perf_counter()
    n = 0
    with open(ns.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        if not file_exists:
            w.writeheader()
        for seed in ns.seeds:
            cells = {}
            pes_path = os.path.join(args.ckpt_dir, f"pessimistic_box_s{seed}.pt")
            pb_path = os.path.join(args.ckpt_dir, f"pbgt_box_s{seed}.pt")
            if not (os.path.exists(pes_path) and os.path.exists(pb_path)):
                raise SystemExit(f"missing ckpt for seed {seed}")
            pes, pbgt = G.QNet(), G.QNet()
            pes.load_state_dict(torch.load(pes_path, weights_only=True))
            pbgt.load_state_dict(torch.load(pb_path, weights_only=True))
            cells["pes_q_greedy_test"] = (pes, ("pessimistic", "box", False))
            cells["pbgt_q_maximin_test"] = (pbgt, ("pbgt", "box", True))
            for key, (net, spec) in cells.items():
                if (seed, key) in done:
                    print(f"skip seed {seed} {key}", flush=True)
                    continue
                t0 = time.perf_counter()
                r = P.evaluate(net, spec, "pgd", args, seed, ns.episodes)
                row = {
                    "seed": seed, "eps": args.eps, "cell": key, "condition": "pgd",
                    "mean": r["mean"], "std": r["std"], "ms_per_action": r["ms_per_action"],
                    "box_width": r["box_width"], "ball_width": r["ball_width"],
                    "episodes": ns.episodes,
                }
                w.writerow(row)
                f.flush()
                n += 1
                print(
                    f"seed {seed} {key} pgd {r['mean']:.2f} in {time.perf_counter()-t0:.1f}s",
                    flush=True,
                )
    print(f"PGD_OFFDIAG_DONE n={n} wall_min={(time.perf_counter()-t_all)/60:.2f}", flush=True)


if __name__ == "__main__":
    main()

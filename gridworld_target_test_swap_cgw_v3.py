#!/usr/bin/env python3
"""Bellman-target x test-rule swap on continuous GridWorld, eps=0.25, seeds 0-4.

Regenerates vanilla / pessimistic(box) / PB-GT with the production budget
(deterministic; seed 0 matches gw_prod_eps025_5seeds.csv). Does not retrain the
old matched SB3 attackers. Trains NEW attackers against the swapped test rules,
same budget as the production attackers (40_000 steps, L=5).

Cells:
  pessimistic weights + greedy test
  PB-GT weights + belief maximin test
PGD-10 is omitted: the existing MinBest attack was reviewed and not changed.

Does not overwrite production CSVs or attacker zips.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch

import gridworld_continuous_dqn as G
import gridworld_continuous_prod as P

CONDITIONS = ["clean", "shared_random", "matched_search", "matched_sb3"]


def make_args(seed_list):
    return argparse.Namespace(
        vanilla_steps=80_000,
        finetune_steps=25_000,
        attacker_steps=40_000,
        episodes=50,
        eps=0.25,
        particles=3,
        attack_levels=5,
        pes_nesting="maxmin",
        lr=1e-3,
        ft_lr=1e-4,
        batch=64,
        gamma=0.99,
        buffer=100_000,
        learn_start=1_000,
        target_update=500,
        train_cand=16,
        eval_cand=32,
        pgd_iters=10,
        verbose=True,
        belief="box",
        threads=1,
        seeds=seed_list,
        attacker_dir="gw_swap_attackers_eps025_v3",
        ckpt_dir="gw_regen_ckpts_v3",
    )


def _load_reference():
    ref = {}
    with open("gw_prod_eps025_5seeds.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["condition"] != "clean":
                continue
            if r["variant"] not in ("vanilla", "pessimistic_box", "pbgt_box",
                                    "pessimistic_box_greedyreadout"):
                continue
            ref[(int(r["seed"]), r["variant"])] = float(r["mean"])
    return ref


def run_seed(seed, args, ref):
    t0 = time.time()
    base = argparse.Namespace(**vars(args))
    base.belief = "box"
    ckpt_candidates = [
        os.path.join(args.ckpt_dir, f"vanilla_s{seed}.pt"),
        os.path.join(args.ckpt_dir, f"s{seed}_vanilla.pt"),
    ]
    ckpt = next((p for p in ckpt_candidates if os.path.exists(p)), "")
    if ckpt:
        van = G.QNet()
        van.load_state_dict(torch.load(ckpt, weights_only=True))
        print(f"  [seed {seed}] loaded saved vanilla {ckpt}", flush=True)
    else:
        van, _ = G.train("vanilla", args.vanilla_steps, seed, base)
    pes_ckpt = next((p for p in (
        os.path.join(args.ckpt_dir, f"pessimistic_box_s{seed}.pt"),
        os.path.join(args.ckpt_dir, f"s{seed}_pes.pt"),
    ) if os.path.exists(p)), "")
    pb_ckpt = next((p for p in (
        os.path.join(args.ckpt_dir, f"pbgt_box_s{seed}.pt"),
        os.path.join(args.ckpt_dir, f"s{seed}_pbgt.pt"),
    ) if os.path.exists(p)), "")
    if pes_ckpt and pb_ckpt:
        pes, pbgt = G.QNet(), G.QNet()
        pes.load_state_dict(torch.load(pes_ckpt, weights_only=True))
        pbgt.load_state_dict(torch.load(pb_ckpt, weights_only=True))
        print(f"  [seed {seed}] loaded saved pes/pbgt", flush=True)
    else:
        init = van.state_dict()
        pes, _ = G.train("pessimistic", args.finetune_steps, seed + 1, base, init_state_dict=init,
                         log_prefix=f"[s{seed}] ")
        pbgt, _ = G.train("pbgt", args.finetune_steps, seed + 2, base, init_state_dict=init,
                          log_prefix=f"[s{seed}] ")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.save(van.state_dict(), os.path.join(args.ckpt_dir, f"vanilla_s{seed}.pt"))
    torch.save(pes.state_dict(), os.path.join(args.ckpt_dir, f"pessimistic_box_s{seed}.pt"))
    torch.save(pbgt.state_dict(), os.path.join(args.ckpt_dir, f"pbgt_box_s{seed}.pt"))

    # Confirm this realization is the production one before swapping the test rule.
    checks = [
        (van, P.VARIANT_SPECS["vanilla"], "vanilla", False),
        (pes, P.VARIANT_SPECS["pessimistic_box"], "pessimistic_box", False),
        (pbgt, P.VARIANT_SPECS["pbgt_box"], "pbgt_box", False),
        (pes, P.VARIANT_SPECS["pessimistic_box"], "pessimistic_box_greedyreadout", True),
    ]
    for net, spec, name, force_g in checks:
        r = P.evaluate(net, spec, "clean", args, seed, args.episodes, force_greedy=force_g)
        exp = ref[(seed, name)]
        ok = abs(r["mean"] - exp) < 1e-3
        print(f"  [seed {seed}] regen {name} clean {r['mean']:.4f} expected {exp:.4f} match={ok}",
              flush=True)
        if not ok:
            raise SystemExit(f"regeneration mismatch seed {seed} {name}")

    # Swapped test rules. New attackers see that rule.
    cells = {
        "pes_q_greedy_test": (pes, ("pessimistic", "box", False)),
        "pbgt_q_maximin_test": (pbgt, ("pbgt", "box", True)),
    }
    attackers = {}
    for key, (net, spec) in cells.items():
        tag = f"gw_att_{key}_eps0.25_s{seed}"
        attackers[key] = P.train_matched_attacker(net, spec, args, seed, tag)

    rows = []
    for key, (net, spec) in cells.items():
        for cond in CONDITIONS:
            r = P.evaluate(net, spec, cond, args, seed, args.episodes,
                           attacker=attackers[key])
            rows.append({
                "seed": seed, "eps": args.eps, "cell": key, "condition": cond,
                "mean": r["mean"], "std": r["std"], "ms_per_action": r["ms_per_action"],
                "box_width": r["box_width"], "ball_width": r["ball_width"],
                "episodes": args.episodes,
            })
            print(f"  [seed {seed}] {key:<22} {cond:<16} {r['mean']:8.2f} +/- {r['std']:6.2f}",
                  flush=True)
    print(f"  [seed {seed}] swap done in {time.time()-t0:.0f}s", flush=True)
    return rows


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out", default="gridworld_target_test_swap_cgw_v3.csv")
    ns = p.parse_args()
    if os.path.exists(ns.out):
        raise SystemExit(f"refusing to overwrite {ns.out}")
    args = make_args(ns.seeds)
    ref = _load_reference()
    rows = []
    fields = ["seed", "eps", "cell", "condition", "mean", "std", "ms_per_action",
              "box_width", "ball_width", "episodes"]
    for s in ns.seeds:
        rows.extend(run_seed(s, args, ref))
        with open(ns.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {ns.out}", flush=True)


if __name__ == "__main__":
    main()

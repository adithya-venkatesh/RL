#!/usr/bin/env python3
"""Single-process inference retime. torch threads = 1.

CGW: clean-episode decision time for vanilla, pessimistic(box), PB-GT, and
pessimistic(ball), using saved production weights. Ball checkpoints exist for
seeds 5-9 only.

Tabular: clean-episode greedy read-out (the 0.0056 ms figure) for vanilla,
pessimistic, and PB-GT, 20 seeds, 100 episodes, same measurement functions as
main_with_vanilla.run_single_experiment_with_seed.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"


def time_cgw(episodes, out_rows):
    specs = {
        "vanilla": ("vanilla", "box", False),
        "pessimistic_box": ("pessimistic", "box", True),
        "pbgt_box": ("pbgt", "box", False),
        "pessimistic_ball": ("pessimistic", "ball", True),
    }
    args = argparse.Namespace(eps=0.25, particles=3, eval_cand=32, pgd_iters=10, attack_levels=5)
    ckpt = "gw_regen_ckpts_v3"
    for seed in range(10):
        for label, spec in specs.items():
            path = os.path.join(ckpt, f"{label}_s{seed}.pt")
            if not os.path.exists(path):
                continue
            net = __import__("gridworld_continuous_dqn").QNet()
            net.load_state_dict(torch.load(path, weights_only=True))
            net.eval()
            # Reimplement the clean decision timer so we also count actions.
            import gridworld_continuous_dqn as G
            _, belief_mode, use_belief = spec
            m = args.particles
            times = []
            for ep in range(episodes):
                env = G.ContinuousGridWorld(np.random.default_rng(seed * 10_000 + ep))
                s = env.reset()
                box, last_a = None, 0
                for _ in range(G.MAX_STEPS):
                    obs = s
                    t0 = time.perf_counter()
                    if use_belief:
                        particles, box = G.belief_step(belief_mode, box, last_a, obs, args.eps, m)
                        act = G.maximin_action(net, obs, args.eps, m, particles)
                    else:
                        act = G.greedy_action(net, obs)
                    times.append(time.perf_counter() - t0)
                    last_a = act
                    s, _, done, trunc = env.step(act)
                    if done or trunc:
                        break
            out_rows.append({
                "domain": "cgw", "method": label, "seed": seed,
                "episodes": episodes, "actions": len(times),
                "ms_per_action": float(np.mean(times) * 1000.0),
            })
            print(f"cgw seed={seed} {label} ms={out_rows[-1]['ms_per_action']:.4f} "
                  f"actions={len(times)}", flush=True)


def time_tabular(n_seeds, episodes, out_rows):
    from diag_tabular_v3_common import train_defenders
    from main_with_vanilla import (
        GridWorld, play_offpolicy_greedy_target, play_pessimistic_clean, play_vanilla_test,
    )

    for seed in range(n_seeds):
        agent_v, agent_p, agent_o, _ = train_defenders(seed, scoring="fixed")
        env = GridWorld(); env.seed(seed); np.random.seed(seed + 1000)
        _, t_v = play_vanilla_test(env, agent_v, trials=episodes, measure_time=True)
        env = GridWorld(); env.seed(seed); np.random.seed(seed + 1000)
        _, t_p = play_pessimistic_clean(env, agent_p, trials=episodes, measure_time=True)
        env = GridWorld(); env.seed(seed); np.random.seed(seed + 1000)
        _, t_o = play_offpolicy_greedy_target(env, agent_o, trials=episodes, learn=False, measure_time=True)
        for method, t in (("vanilla", t_v), ("pessimistic_greedy_clean", t_p), ("pbgt_greedy_clean", t_o)):
            out_rows.append({
                "domain": "tabular", "method": method, "seed": seed,
                "episodes": episodes, "actions": "",
                "ms_per_action": float(t * 1000.0),
            })
        print(f"tab seed={seed} van={t_v*1000:.4f} pes={t_p*1000:.4f} pbgt={t_o*1000:.4f}", flush=True)


def summarize(rows):
    print("\n=== single-process inference (torch threads=1) mean±std of per-seed ms/action, ddof=1 ===")
    keys = sorted({(r["domain"], r["method"]) for r in rows})
    for domain, method in keys:
        sub = [r for r in rows if r["domain"] == domain and r["method"] == method]
        vals = np.array([r["ms_per_action"] for r in sub], float)
        eps = sum(int(r["episodes"]) for r in sub)
        acts = [r["actions"] for r in sub if r["actions"] != ""]
        nact = sum(int(a) for a in acts) if acts else "n/a"
        sd = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        print(f"{domain:8} {method:28} {vals.mean():.4f}±{sd:.4f} ms  "
              f"seeds={len(sub)} episodes={eps} actions={nact}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", choices=["cgw", "tabular", "both"], default="both")
    p.add_argument("--out", default="inference_retime_singleproc_v3.csv")
    p.add_argument("--cgw-episodes", type=int, default=50)
    p.add_argument("--tab-seeds", type=int, default=20)
    p.add_argument("--tab-episodes", type=int, default=100)
    args = p.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"refusing to overwrite {args.out}")
    rows = []
    if args.which in ("cgw", "both"):
        time_cgw(args.cgw_episodes, rows)
    if args.which in ("tabular", "both"):
        time_tabular(args.tab_seeds, args.tab_episodes, rows)
    fields = ["domain", "method", "seed", "episodes", "actions", "ms_per_action"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    summarize(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

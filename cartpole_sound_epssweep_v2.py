#!/usr/bin/env python3
"""Epsilon sweep v2: where does each decision rule break? (publication density)

Supersedes cartpole_sound_epssweep.csv, which used 3 episodes per cell on one
seed and is too thin to publish. Here: 5 epsilons x 2 seeds x 20 episodes,
greedy / uniform-ball maximin / sound-box maximin, under two white-box attacks
on the pole sensors:

    worst_of_9_const_bias  min over the 9 constant biases in {-e,0,e}^2 on
                           (theta, theta_dot); reported value is the worst one
    exhaustive_greedy_9    delta_t = argmin_delta Q(s_true, pi(s_tilde)) chosen
                           fresh each step from the same 9-element set

All rules share the same untouched per-seed vanilla Q-network; no fine-tuning.
Artifacts: cartpole_sound_epssweep_v2.csv (new name; the old csv is untouched).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import torch

from cartpole_belief import UniformBallBelief
from cartpole_sound_belief import SoundFilteredBelief
from cartpole_sound_eval_harness import (
    const_bias_attack,
    exhaustive_greedy_attack,
    load_result_rows,
    pole_deltas,
    sweep_rollout,
    write_result_rows,
)
from main_cartpole_dqn import DEFAULT_BELIEF_KAPPA, DQNAgent, DQNConfig

torch.set_num_threads(1)

OUT_CSV = "cartpole_sound_epssweep_v2.csv"
EPS_GRID = [0.025, 0.05, 0.10, 0.15, 0.20]
SEEDS = [0, 1]
N_EP = 20
KAPPA = DEFAULT_BELIEF_KAPPA
KINDS = ["greedy", "uniform", "sound"]


def make_policy(agent: DQNAgent, kind: str, eps: float):
    if kind == "greedy":
        return (lambda o: agent.greedy_action(o)), None
    prov = UniformBallBelief(eps, KAPPA) if kind == "uniform" else SoundFilteredBelief(eps, KAPPA)
    return (lambda o: agent.maximin_action(o, eps, KAPPA, belief_provider=prov)), prov


def main() -> None:
    rows: List[Dict[str, Any]] = load_result_rows(OUT_CSV)
    done = {(int(r["seed"]), float(r["eps"]), r["attack"]) for r in rows}
    if done:
        print(f"resuming {OUT_CSV}: {len(done)} cells already on disk", flush=True)
    t_all = time.time()
    print(f"eps sweep v2: seeds={SEEDS}, {N_EP} ep/cell, kappa={KAPPA}, no fine-tuning", flush=True)
    print(f"{'seed':>4} {'eps':>6} {'attack':<22} " + " ".join(f"{k:>8}" for k in KINDS), flush=True)

    for seed in SEEDS:
        agent = DQNAgent(config=DQNConfig())
        agent.load(f"checkpoints/seed_{seed}/vanilla.pt")
        agent.config.box_min = "particles"
        for eps in EPS_GRID:
            deltas = pole_deltas(eps)
            for label in ("worst_of_9_const_bias", "exhaustive_greedy_9"):
                if (seed, float(eps), label) in done:
                    rec = next(r for r in rows if int(r["seed"]) == seed
                               and float(r["eps"]) == float(eps) and r["attack"] == label)
                    print(f"{seed:>4} {eps:>6.3f} {label:<22} "
                          + " ".join(f"{float(rec[k]):>8.1f}" for k in KINDS)
                          + "   SKIP", flush=True)
                    continue
                rec: Dict[str, Any] = {"seed": seed, "eps": eps, "attack": label, "n_ep": N_EP}
                t0 = time.time()
                for kind in KINDS:
                    if label == "worst_of_9_const_bias":
                        # fresh policy/belief per delta; report the worst delta
                        per_delta = []
                        for d in deltas:
                            act, prov = make_policy(agent, kind, eps)
                            per_delta.append(float(np.mean(
                                sweep_rollout(agent, act, prov, const_bias_attack(d), N_EP, 31000)
                            )))
                        rec[kind] = float(np.min(per_delta))
                        rec[f"{kind}_best_delta"] = float(np.max(per_delta))
                    else:
                        act, prov = make_policy(agent, kind, eps)
                        rec[kind] = float(np.mean(
                            sweep_rollout(agent, act, prov, exhaustive_greedy_attack(deltas),
                                          N_EP, 32000)
                        ))
                rows.append(rec)
                done.add((seed, float(eps), label))
                print(f"{seed:>4} {eps:>6.3f} {label:<22} "
                      + " ".join(f"{rec[k]:>8.1f}" for k in KINDS)
                      + f"   ({time.time()-t0:.0f}s)", flush=True)
                write_result_rows(OUT_CSV, rows)

    print(f"\n===== SEED-AVERAGED (mean over {len(SEEDS)} seeds) =====")
    print(f"{'eps':>6} {'attack':<22} " + " ".join(f"{k:>8}" for k in KINDS))
    for eps in EPS_GRID:
        for label in ("worst_of_9_const_bias", "exhaustive_greedy_9"):
            rs = [r for r in rows if r["eps"] == eps and r["attack"] == label]
            if not rs:
                continue
            print(f"{eps:>6.3f} {label:<22} "
                  + " ".join(f"{np.mean([r[k] for r in rs]):>8.1f}" for k in KINDS), flush=True)

    print(f"\nwrote {OUT_CSV}  total wall time {(time.time()-t_all)/60:.1f} min", flush=True)
    print("EPS_SWEEP_V2_DONE", flush=True)


if __name__ == "__main__":
    main()

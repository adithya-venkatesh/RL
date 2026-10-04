#!/usr/bin/env python3
"""5-seed constant-bias / exhaustive pole-sensor eval at eps=0.05.

Does not overwrite cartpole_sound_epssweep_v2.csv. Frozen per-seed vanilla Q,
no fine-tuning, greedy vs uniform-ball vs sound box. Attacks:

    worst_of_9_const_bias  min over the 9 constant biases in {-e,0,e}^2 on
                           (theta, theta_dot)
    exhaustive_greedy_9    delta_t = argmin_delta Q(s_true, pi(s_tilde)) chosen
                           fresh each step from the same 9-element set

Seeds 0-1 at this (eps, n_ep, seed offsets) already exist in the v2 sweep and
are copied into the new CSV so those cells are not recomputed.
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

OUT_CSV = "cartpole_sound_bias5seeds.csv"
SRC_V2 = "cartpole_sound_epssweep_v2.csv"
EPS = 0.05
SEEDS = [0, 1, 2, 3, 4]
N_EP = 20
KAPPA = DEFAULT_BELIEF_KAPPA
KINDS = ["greedy", "uniform", "sound"]
ATTACKS = ("worst_of_9_const_bias", "exhaustive_greedy_9")


def make_policy(agent: DQNAgent, kind: str, eps: float):
    if kind == "greedy":
        return (lambda o: agent.greedy_action(o)), None
    prov = UniformBallBelief(eps, KAPPA) if kind == "uniform" else SoundFilteredBelief(eps, KAPPA)
    return (lambda o: agent.maximin_action(o, eps, KAPPA, belief_provider=prov)), prov


def seed_from_v2() -> List[Dict[str, Any]]:
    """Copy the already-run (seed in {0,1}, eps=0.05) cells so we don't recompute."""
    out: List[Dict[str, Any]] = []
    for r in load_result_rows(SRC_V2):
        if float(r["eps"]) != EPS:
            continue
        if int(r["seed"]) not in (0, 1):
            continue
        if r["attack"] not in ATTACKS:
            continue
        rec = {
            "seed": int(r["seed"]),
            "eps": EPS,
            "attack": r["attack"],
            "n_ep": int(r.get("n_ep", N_EP)),
            "source": "epssweep_v2",
        }
        for k in KINDS:
            rec[k] = float(r[k])
            bk = f"{k}_best_delta"
            if bk in r and r[bk] not in ("", None):
                rec[bk] = float(r[bk])
        out.append(rec)
    return out


def main() -> None:
    rows: List[Dict[str, Any]] = load_result_rows(OUT_CSV)
    if not rows:
        rows = seed_from_v2()
        if rows:
            write_result_rows(OUT_CSV, rows)
            print(f"seeded {OUT_CSV} with {len(rows)} cells from {SRC_V2}", flush=True)
    done = {(int(r["seed"]), r["attack"]) for r in rows}
    print(
        f"bias 5-seed: seeds={SEEDS}, eps={EPS}, {N_EP} ep/cell, kappa={KAPPA}, "
        f"{len(done)} cells already on disk",
        flush=True,
    )
    print(f"{'seed':>4} {'attack':<22} " + " ".join(f"{k:>8}" for k in KINDS), flush=True)
    t_all = time.time()
    deltas = pole_deltas(EPS)

    for seed in SEEDS:
        agent = None
        for label in ATTACKS:
            if (seed, label) in done:
                rec = next(r for r in rows if int(r["seed"]) == seed and r["attack"] == label)
                print(
                    f"{seed:>4} {label:<22} "
                    + " ".join(f"{float(rec[k]):>8.1f}" for k in KINDS)
                    + "   SKIP",
                    flush=True,
                )
                continue
            if agent is None:
                agent = DQNAgent(config=DQNConfig())
                agent.load(f"checkpoints/seed_{seed}/vanilla.pt")
                agent.config.box_min = "particles"
            rec: Dict[str, Any] = {
                "seed": seed,
                "eps": EPS,
                "attack": label,
                "n_ep": N_EP,
                "source": "this_run",
            }
            t0 = time.time()
            for kind in KINDS:
                if label == "worst_of_9_const_bias":
                    per_delta = []
                    for d in deltas:
                        act, prov = make_policy(agent, kind, EPS)
                        per_delta.append(
                            float(np.mean(sweep_rollout(agent, act, prov, const_bias_attack(d), N_EP, 31000)))
                        )
                    rec[kind] = float(np.min(per_delta))
                    rec[f"{kind}_best_delta"] = float(np.max(per_delta))
                else:
                    act, prov = make_policy(agent, kind, EPS)
                    rec[kind] = float(
                        np.mean(
                            sweep_rollout(agent, act, prov, exhaustive_greedy_attack(deltas), N_EP, 32000)
                        )
                    )
            rows.append(rec)
            done.add((seed, label))
            print(
                f"{seed:>4} {label:<22} "
                + " ".join(f"{rec[k]:>8.1f}" for k in KINDS)
                + f"   ({time.time()-t0:.0f}s)",
                flush=True,
            )
            write_result_rows(OUT_CSV, rows)

    print(f"\n===== SEED-AVERAGED (mean +/- std over {len(SEEDS)} seeds) =====")
    print(f"{'attack':<22} " + " ".join(f"{k:>18}" for k in KINDS))
    for label in ATTACKS:
        rs = [r for r in rows if r["attack"] == label]
        if len(rs) < len(SEEDS):
            print(f"{label:<22} incomplete ({len(rs)}/{len(SEEDS)})", flush=True)
            continue
        parts = []
        for k in KINDS:
            vals = np.array([float(r[k]) for r in rs], dtype=np.float64)
            parts.append(f"{vals.mean():8.1f}+/-{vals.std(ddof=1):6.1f}")
        print(f"{label:<22} " + " ".join(f"{p:>18}" for p in parts), flush=True)

    print(f"\nwrote {OUT_CSV}  total wall time {(time.time()-t_all)/60:.1f} min", flush=True)
    print("BIAS_5SEEDS_DONE", flush=True)


if __name__ == "__main__":
    main()

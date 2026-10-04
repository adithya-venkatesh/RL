#!/usr/bin/env python3
"""Epsilon sweep: where does each decision rule break?

At eps=0.05 the sound-box maximin is unbreakable by every attacker in the repo,
so the interesting question is how far the budget has to grow before it degrades,
and whether it dominates greedy / uniform-ball maximin throughout.

All three rules share the *same* seed-0 vanilla Q-network (no fine-tuning).
Artifacts: cartpole_sound_epssweep.csv (new name).
"""
from __future__ import annotations

import csv
import itertools
import time
from typing import Callable, Dict, List

import numpy as np

from cartpole_belief import UniformBallBelief
from cartpole_sound_belief import SoundFilteredBelief
from main_cartpole_dqn import (
    CartPoleDefenderEnv,
    DEFAULT_BELIEF_KAPPA,
    DQNAgent,
    DQNConfig,
    apply_perturbation,
    set_seed,
)

KAPPA = DEFAULT_BELIEF_KAPPA
EPS_GRID = [0.05, 0.10, 0.20]
N_EP = 3
OUT_CSV = "cartpole_sound_epssweep.csv"


def pole_deltas(eps: float) -> List[np.ndarray]:
    lv = (-eps, 0.0, eps)
    return [np.array([0.0, 0.0, a, b], dtype=np.float32) for a, b in itertools.product(lv, lv)]


def make_policy(agent: DQNAgent, kind: str, eps: float):
    if kind == "greedy":
        return (lambda o: agent.greedy_action(o)), None
    prov = UniformBallBelief(eps, KAPPA) if kind == "uniform" else SoundFilteredBelief(eps, KAPPA)
    return (lambda o: agent.maximin_action(o, eps, KAPPA, belief_provider=prov)), prov


def rollout(agent, kind, eps, delta_fn: Callable, n_ep: int, seed: int) -> float:
    set_seed(seed)
    env = CartPoleDefenderEnv()
    act, prov = make_policy(agent, kind, eps)
    rets = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=seed + ep)
        if prov is not None:
            prov.reset_episode()
        done, ret = False, 0.0
        while not done:
            a = act(delta_fn(obs, agent, act, prov))
            if prov is not None and hasattr(prov, "note_action"):
                prov.note_action(a)
            obs, r, term, trunc, _ = env.step(a)
            ret += r
            done = term or trunc
        rets.append(ret)
    env.close()
    return float(np.mean(rets))


def const(delta):
    return lambda obs, agent, act, prov: apply_perturbation(obs, delta, normalized=True)


def exhaustive(deltas):
    """delta_t = argmin_delta Q(s_true, pi(s_tilde)) over the discrete attack set.

    Probing pi() advances a history-conditioned belief, so the filter state is
    snapshotted and restored around the candidate loop.
    """
    def fn(obs, agent, act, prov):
        q_true = agent.q_values(obs)
        snap = prov.snapshot() if hasattr(prov, "snapshot") else None
        best, best_score = obs, float("inf")
        for d in deltas:
            s_t = apply_perturbation(obs, d, normalized=True)
            sc = float(q_true[act(s_t)])
            if snap is not None:
                prov.restore(snap)
            if sc < best_score:
                best_score, best = sc, s_t
        return best
    return fn


def main() -> None:
    agent = DQNAgent(config=DQNConfig())
    agent.load("checkpoints/seed_0/vanilla.pt")
    agent.config.box_min = "particles"
    kinds = ["greedy", "uniform", "sound"]
    rows: List[Dict] = []

    print(f"seed-0 vanilla Q-net, {N_EP} eps/cell, kappa={KAPPA}")
    print(f"{'eps':>5} {'attack':<22} " + " ".join(f"{k:>8}" for k in kinds))
    for eps in EPS_GRID:
        ds = pole_deltas(eps)
        for label, kind_of_attack in (("worst_of_9_const_bias", "bias"), ("exhaustive_greedy_9", "exh")):
            rec: Dict = {"eps": eps, "attack": label}
            t0 = time.time()
            for k in kinds:
                if kind_of_attack == "bias":
                    rec[k] = float(np.min([rollout(agent, k, eps, const(d), N_EP, 31000) for d in ds]))
                else:
                    rec[k] = rollout(agent, k, eps, exhaustive(ds), N_EP, 32000)
            rows.append(rec)
            print(f"{eps:>5.2f} {label:<22} " + " ".join(f"{rec[k]:>8.1f}" for k in kinds)
                  + f"   ({time.time()-t0:.0f}s)", flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["eps", "attack"] + kinds)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_CSV}")
    print("EPS_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()

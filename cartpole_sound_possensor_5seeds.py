#!/usr/bin/env python3
"""Position-sensor threat model: eps_vec = [eps, 0, eps, 0].

Only the two position sensors (cart x, pole theta) are attacked; the velocity
sensors are observed cleanly. This is the regime where the filter-then-propagate
recursion has real work to do: propagating the previous box through the exact
dynamics produces tight predictions for x and theta (they are integrals of the
clean velocities), so intersecting with B_eps(s_tilde) genuinely shrinks the box
instead of just reproducing it.

Same protocol as cartpole_sound_5seeds.py: 5 seeds, the untouched per-seed
vanilla Q-network, no fine-tuning, four decision rules, four conditions.
Artifacts: cartpole_sound_possensor_5seeds.csv (new name).
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
from stable_baselines3 import DQN as SB3_DQN

from cartpole_belief import UniformBallBelief
from cartpole_sound_belief import SoundFilteredBelief
from cartpole_sound_eval_harness import (
    attacker_is_vacuous,
    load_result_rows,
    masked_delta,
    perturb_from_masked_attacker,
    train_attacker_with_diag,
    write_result_rows,
)
from main_cartpole_dqn import (
    CartPoleDefenderEnv,
    DEFAULT_ATTACK_EPS,
    DEFAULT_BELIEF_KAPPA,
    DQNAgent,
    DQNConfig,
    _note_belief_action,
    apply_perturbation,
    set_seed,
)

torch.set_num_threads(1)

OUT_CSV = "cartpole_sound_possensor_5seeds.csv"
SEEDS = [0, 1, 2, 3, 4]
EVAL_EP = 50
ATK_STEPS = 40_000
EPS = DEFAULT_ATTACK_EPS
KAPPA = DEFAULT_BELIEF_KAPPA
ATK_DIMS = (0, 2)
EPS_VEC = np.array([EPS, 0.0, EPS, 0.0], dtype=np.float64)
PGD_CANDIDATES = 32

RULES = [
    ("greedy", "greedy", lambda: None, "particles"),
    ("uniform", "maximin", lambda: UniformBallBelief(EPS_VEC, KAPPA), "particles"),
    ("sound", "maximin", lambda: SoundFilteredBelief(EPS_VEC, KAPPA), "particles"),
    ("sound_ibp", "maximin", lambda: SoundFilteredBelief(EPS_VEC, KAPPA), "ibp"),
]


def _load(seed: int, box_min: str) -> DQNAgent:
    cfg = DQNConfig()
    cfg.box_min = box_min
    agent = DQNAgent(config=cfg)
    agent.load(f"checkpoints/seed_{seed}/vanilla.pt")
    for p in agent.q_net.parameters():
        p.requires_grad = False
    return agent


def _act(agent: DQNAgent, rule: str, centre: np.ndarray, prov) -> int:
    if rule == "greedy":
        return agent.greedy_action(centre)
    return agent.maximin_action(centre, EPS_VEC, KAPPA, belief_provider=prov)


def _rollout(agent, rule, prov, centre_fn, n_ep, seed) -> List[float]:
    """centre_fn(true_obs, rng) -> observation the defender's rule is centred on."""
    set_seed(seed)
    env = CartPoleDefenderEnv()
    rets: List[float] = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=seed + ep)
        rng = np.random.RandomState(seed + ep)
        if prov is not None:
            prov.reset_episode()
        done, ret = False, 0.0
        while not done:
            centre = centre_fn(obs, rng, agent, rule, prov)
            action = _act(agent, rule, centre, prov)
            _note_belief_action(prov, action)
            obs, r, term, trunc, _ = env.step(action)
            ret += r
            done = term or trunc
        rets.append(ret)
    env.close()
    return rets


def _clean_centre(obs, rng, agent, rule, prov):
    return obs


def _shared_random_centre(obs, rng, agent, rule, prov):
    delta = rng.uniform(-EPS_VEC, EPS_VEC, size=4).astype(np.float32)
    return apply_perturbation(obs, delta, normalized=True)


def _pgd_centre(obs, rng, agent, rule, prov):
    """Masked random-search PGD*: argmin over 32 sampled masked deltas of
    Q(s_true, pi(s_tilde)). Snapshots the filter so probes do not advance it."""
    snap = prov.snapshot() if (prov is not None and hasattr(prov, "snapshot")) else None
    q_true = agent.q_values(obs)
    best, best_score = obs, float("inf")
    for _ in range(PGD_CANDIDATES):
        delta = np.random.uniform(-EPS_VEC, EPS_VEC, size=4).astype(np.float32)
        s_t = apply_perturbation(obs, delta, normalized=True)
        score = float(q_true[_act(agent, rule, s_t, prov)])
        if snap is not None:
            prov.restore(snap)
        if score < best_score:
            best_score, best = score, s_t
    return best


def _make_attacker_centre(attacker):
    def fn(obs, rng, agent, rule, prov):
        return perturb_from_masked_attacker(obs, attacker, EPS_VEC, ATK_DIMS)
    return fn


def box_width_diag(seed: int, n_ep: int = 5) -> Dict[str, float]:
    """Mean filtered-box width per dimension, to show the filter really shrinks it."""
    agent = _load(seed, "particles")
    out = {}
    for tag, prov in (("sound", SoundFilteredBelief(EPS_VEC, KAPPA)),):
        _rollout(agent, "maximin", prov, _shared_random_centre, n_ep, seed + 12000)
        w = prov.mean_width()
        for i, nm in enumerate(("x", "xdot", "th", "thdot")):
            out[f"boxw_{nm}"] = float(w[i])
        out["n_empty_intersections"] = float(prov.n_empty)
    return out


def run_cell(seed: int, name: str, rule: str, factory, box_min: str) -> Dict[str, Any]:
    agent = _load(seed, box_min)
    agent.config.attack_epsilon = EPS_VEC
    row: Dict[str, Any] = {"seed": seed, "rule": name}
    t0 = time.time()

    row["clean"] = float(np.mean(_rollout(agent, rule, factory(), _clean_centre, EVAL_EP, seed + 1000)))
    row["shared"] = float(np.mean(_rollout(agent, rule, factory(), _shared_random_centre, EVAL_EP, seed + 6000)))
    row["pgd"] = float(np.mean(_rollout(agent, rule, factory(), _pgd_centre, EVAL_EP, seed + 7000)))

    path = f"checkpoints/seed_{seed}/attacker_possensor_{name}"
    diag = train_attacker_with_diag(
        agent, "vanilla" if rule == "greedy" else "pessimistic",
        ATK_STEPS, path, belief_provider=factory(),
        epsilon=EPS, eps_vec=EPS_VEC, dims=ATK_DIMS, seed=seed + 9000,
    )
    attacker = SB3_DQN.load(path + ".zip")
    row["matched"] = float(np.mean(
        _rollout(agent, rule, factory(), _make_attacker_centre(attacker), EVAL_EP, seed + 8000)
    ))
    row.update(diag)
    row["matched_vacuous"] = int(attacker_is_vacuous(row["matched"], diag))
    row["secs"] = round(time.time() - t0, 1)

    print(
        f"  {name:<10} clean={row['clean']:6.1f} shared={row['shared']:6.1f} "
        f"pgd={row['pgd']:6.1f} matched={row['matched']:6.1f}"
        f"{' [VACUOUS]' if row['matched_vacuous'] else ''}  "
        f"atk_len min/last10={diag['atk_len_min']:.0f}/{diag['atk_len_last10']:.0f}  "
        f"({row['secs']:.0f}s)",
        flush=True,
    )
    return row


def summarize(rows: List[Dict[str, Any]]) -> None:
    print(f"\n===== POSITION-SENSOR eps_vec={list(EPS_VEC)}, {EVAL_EP} ep/cell, no fine-tuning =====")
    print(f"{'rule':<10} {'clean':>14} {'shared':>14} {'pgd':>14} {'matched':>14}  vacuous")
    for name, _, _, _ in RULES:
        rs = [r for r in rows if r["rule"] == name]
        if not rs:
            continue
        cells = []
        for cond in ("clean", "shared", "pgd", "matched"):
            a = np.asarray([r[cond] for r in rs], dtype=np.float64)
            cells.append(f"{a.mean():7.1f}+/-{a.std():5.1f}")
        nvac = sum(r["matched_vacuous"] for r in rs)
        print(f"{name:<10} " + " ".join(f"{c:>14}" for c in cells)
              + f"  {nvac}/{len(rs)} seeds", flush=True)


def main() -> None:
    missing = [s for s in SEEDS if not os.path.exists(f"checkpoints/seed_{s}/vanilla.pt")]
    if missing:
        sys.exit(f"missing vanilla checkpoints for seeds {missing}")

    rows: List[Dict[str, Any]] = load_result_rows(OUT_CSV)
    done = {(int(r["seed"]), r["rule"]) for r in rows}
    if done:
        print(f"resuming {OUT_CSV}: {len(done)} cells already on disk", flush=True)
    t_all = time.time()
    for seed in SEEDS:
        print(f"\n======== SEED {seed} (position-sensor threat model) ========", flush=True)
        seed_todo = [t for t in RULES if (seed, t[0]) not in done]
        if not seed_todo:
            print("  all rules already in csv", flush=True)
            continue
        wd = box_width_diag(seed)
        print(f"  filtered box mean width: x={wd['boxw_x']:.4f} xdot={wd['boxw_xdot']:.4f} "
              f"th={wd['boxw_th']:.4f} thdot={wd['boxw_thdot']:.4f} "
              f"(unfiltered would be {2*EPS:.3f}/0/{2*EPS:.3f}/0); "
              f"empty intersections={wd['n_empty_intersections']:.0f}", flush=True)
        for name, rule, factory, box_min in seed_todo:
            r = run_cell(seed, name, rule, factory, box_min)
            r.update(wd)
            rows.append(r)
            done.add((seed, name))
            write_result_rows(OUT_CSV, rows)
        summarize(rows)

    summarize(rows)
    print(f"\nwrote {OUT_CSV}  total wall time {(time.time()-t_all)/60:.1f} min", flush=True)
    print("POSSENSOR_DONE", flush=True)


if __name__ == "__main__":
    main()

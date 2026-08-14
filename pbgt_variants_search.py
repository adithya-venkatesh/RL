# -*- coding: utf-8 -*-
"""PB-GT robustness search on tabular GridWorld.

Iteratively evaluates train/test variants against a strong heuristic attacker
(ω_π) and optional DQN attackers. Logs CSV; designed for rapid tweak → test loops.

Target (default): attack mean >= -100 and >=5x faster than full belief maximin.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

# Import GridWorld stack without running interactive CLI
import main_with_vanilla as gw

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
State = Tuple[int, int]
PolicyFn = Callable  # (agent, perturbed_state, belief, env) -> (action, new_belief)


# ---------------------------------------------------------------------------
# Action selectors (test-time policies)
# ---------------------------------------------------------------------------

def _greedy_action(agent, state: State) -> str:
    q = agent.q_table_withatt[state]
    m = max(q.values())
    return np.random.choice([k for k, v in q.items() if v == m])


def _ball_maximin_action(agent, state: State) -> str:
    """a = argmax_a min_{m in B_ε(s̃)} Q(m,a)  — no belief history."""
    ball = agent.get_ball(state)
    scores = {}
    for a in ACTIONS:
        scores[a] = min(agent.q_table_withatt[m][a] for m in ball)
    m = max(scores.values())
    return np.random.choice([k for k, v in scores.items() if v == m])


def _ball_mean_action(agent, state: State) -> str:
    """a = argmax_a mean_{m in B_ε(s̃)} Q(m,a)."""
    ball = agent.get_ball(state)
    scores = {}
    for a in ACTIONS:
        scores[a] = float(np.mean([agent.q_table_withatt[m][a] for m in ball]))
    m = max(scores.values())
    return np.random.choice([k for k, v in scores.items() if v == m])


def _belief_aggregate_action(
    agent, state: State, belief: Set[State], mode: str = "mean"
) -> Tuple[str, Dict]:
    """Eq. 3.4-style policy over M ∩ B_ε(s̃).

    mode='mean' matches main_with_vanilla.get_policy_with_belief (tabular approx).
    mode='min' is strict maximin.
    """
    ball = agent.get_ball(state)
    possible = gw.states_intersection(belief, ball)
    values = {}
    if len(possible) > 1:
        for a in ACTIONS:
            qs = [agent.q_table_withatt[m][a] for m in possible]
            values[a] = float(min(qs) if mode == "min" else np.mean(qs))
    elif len(possible) == 1:
        values = dict(agent.q_table_withatt[list(possible)[0]])
    else:
        values = {a: agent.q_table_withatt[state][a] for a in ACTIONS}
    m = max(values.values())
    return np.random.choice([k for k, v in values.items() if v == m]), values


def _belief_maximin_action(agent, state: State, belief: Set[State]) -> Tuple[str, Dict]:
    """Alias: mean aggregation (matches existing GW defender policy)."""
    return _belief_aggregate_action(agent, state, belief, mode="mean")


def _q_gap(agent, state: State) -> float:
    q = agent.q_table_withatt[state]
    vals = list(q.values())
    return max(vals) - sorted(vals)[-2] if len(vals) >= 2 else 0.0


def _ball_disagreement(agent, state: State) -> bool:
    """True if greedy actions disagree across the ε-ball."""
    ball = agent.get_ball(state)
    acts = set()
    for m in ball:
        acts.add(_greedy_action(agent, m))
        if len(acts) > 1:
            return True
    return False


@dataclass
class PolicyState:
    belief: Optional[Set[State]] = None
    escalations: int = 0
    steps: int = 0
    gap_threshold: float = 0.5
    escalate_frac: float = 1.0  # for hybrid: always escalate if disagree, else use gap


def make_policy(name: str, cfg: Optional[dict] = None):
    """Return (name, select_fn, needs_belief, probe_fn|None)."""
    cfg = cfg or {}

    if name == "greedy":
        def select(agent, s, pst, env):
            pst.steps += 1
            return _greedy_action(agent, s)
        return name, select, False, (lambda agent: (lambda s: _greedy_action(agent, s)))

    if name == "ball_maximin":
        def select(agent, s, pst, env):
            pst.steps += 1
            return _ball_maximin_action(agent, s)
        return name, select, False, (lambda agent: (lambda s: _ball_maximin_action(agent, s)))

    if name == "ball_mean":
        def select(agent, s, pst, env):
            pst.steps += 1
            return _ball_mean_action(agent, s)
        return name, select, False, (lambda agent: (lambda s: _ball_mean_action(agent, s)))

    if name in ("belief_maximin", "belief_mean", "belief_min"):
        mode = "min" if name == "belief_min" else "mean"
        label = "belief_min" if mode == "min" else "belief_maximin"

        def select(agent, s, pst, env, _mode=mode):
            pst.steps += 1
            action, _ = _belief_aggregate_action(agent, s, pst.belief, mode=_mode)
            possible = gw.states_intersection(pst.belief, agent.get_ball(s))
            pst.belief = {env.take_action(st, action) for st in possible}
            return action
        return label, select, True, None

    if name == "hybrid_gap":
        thr = cfg.get("gap_threshold", 0.5)

        def select(agent, s, pst, env):
            pst.steps += 1
            disagree = _ball_disagreement(agent, s)
            gap = _q_gap(agent, s)
            if disagree or gap < thr:
                pst.escalations += 1
                if pst.belief is None:
                    return _ball_maximin_action(agent, s)
                action, _ = _belief_maximin_action(agent, s, pst.belief)
                possible = gw.states_intersection(pst.belief, agent.get_ball(s))
                pst.belief = {env.take_action(st, action) for st in possible}
                return action
            # greedy path — still lightly track belief if present
            action = _greedy_action(agent, s)
            if pst.belief is not None:
                possible = gw.states_intersection(pst.belief, agent.get_ball(s))
                pst.belief = {env.take_action(st, action) for st in possible}
            return action

        return f"hybrid_gap_{thr}", select, True, None

    if name == "hybrid_ball":
        # escalate only to local ball maximin (never full belief) when uncertain
        thr = cfg.get("gap_threshold", 0.5)

        def select(agent, s, pst, env):
            pst.steps += 1
            if _ball_disagreement(agent, s) or _q_gap(agent, s) < thr:
                pst.escalations += 1
                return _ball_maximin_action(agent, s)
            return _greedy_action(agent, s)

        def probe_factory(agent):
            def probe(s):
                if _ball_disagreement(agent, s) or _q_gap(agent, s) < thr:
                    return _ball_maximin_action(agent, s)
                return _greedy_action(agent, s)
            return probe

        return f"hybrid_ball_{thr}", select, False, probe_factory

    if name == "hybrid_disagree":
        # escalate to belief maximin ONLY when ball actions disagree (rarer)
        def select(agent, s, pst, env):
            pst.steps += 1
            if _ball_disagreement(agent, s):
                pst.escalations += 1
                action, _ = _belief_aggregate_action(agent, s, pst.belief, mode="mean")
                possible = gw.states_intersection(pst.belief, agent.get_ball(s))
                pst.belief = {env.take_action(st, action) for st in possible}
                return action
            action = _greedy_action(agent, s)
            if pst.belief is not None:
                possible = gw.states_intersection(pst.belief, agent.get_ball(s))
                pst.belief = {env.take_action(st, action) for st in possible}
            return action

        return "hybrid_disagree", select, True, None

    if name == "softmin_ball":
        # softmin over ball then argmax_a
        temp = cfg.get("temp", 1.0)

        def _softmin_action(agent, s):
            ball = agent.get_ball(s)
            scores = {}
            for a in ACTIONS:
                qs = np.array([agent.q_table_withatt[m][a] for m in ball], dtype=float)
                weights = np.exp(-(qs - qs.min()) / max(temp, 1e-8))
                weights /= weights.sum()
                scores[a] = float(weights @ qs)
            m = max(scores.values())
            return np.random.choice([k for k, v in scores.items() if v == m])

        def select(agent, s, pst, env):
            pst.steps += 1
            return _softmin_action(agent, s)

        return f"softmin_ball_t{temp}", select, False, (lambda agent: (lambda s: _softmin_action(agent, s)))

    raise ValueError(f"Unknown policy {name}")


# ---------------------------------------------------------------------------
# Heuristic attacker ω_π
# ---------------------------------------------------------------------------

def heuristic_attack_state(
    agent,
    true_state: State,
    select: PolicyFn,
    pst: PolicyState,
    env,
    action_for_probe: Optional[Callable[[State], str]] = None,
    mode: str = "code",
) -> State:
    """Worst-case state in B_ε(s).

    mode='code': match main_with_vanilla.get_att_state — score Q(s̃, π(s̃))
    mode='eq22': paper Eq. 2.2 — score Q(s, π(s̃)) at the true state
    """
    ball = agent.get_ball(true_state)
    best_s, best_v = true_state, float("inf")
    for s_tilde in ball:
        if action_for_probe is not None:
            action = action_for_probe(s_tilde)
        else:
            pst_probe = PolicyState(
                belief=set(pst.belief) if pst.belief is not None else None,
                gap_threshold=pst.gap_threshold,
            )
            action = select(agent, s_tilde, pst_probe, env)
        if mode == "eq22":
            v = agent.q_table_withatt[true_state][action]
        else:
            v = agent.q_table_withatt[s_tilde][action]
        if v < best_v:
            best_v = v
            best_s = s_tilde
    return best_s


def apply_dqn_attack(attack_model, true_state: State) -> State:
    attack_obs = np.array(true_state)
    attack_action, _ = attack_model.predict(attack_obs, deterministic=True)
    attack_action = int(attack_action.item() if hasattr(attack_action, "item") else attack_action)
    direction = ACTIONS + ["STAY"]
    if attack_action in range(5):
        pert = direction[attack_action]
    else:
        pert = attack_action
    return gw.apply_perturbation(true_state, pert)


# ---------------------------------------------------------------------------
# Evaluation / training helpers
# ---------------------------------------------------------------------------

def eval_policy_fixed(
    environment,
    agent,
    select: PolicyFn,
    needs_belief: bool,
    trials: int = 100,
    max_steps: int = 1000,
    attack: str = "none",
    attack_model=None,
    measure_time: bool = True,
    gap_threshold: float = 0.5,
    probe_factory=None,
):
    rewards = []
    times = []
    total_esc = 0
    total_steps = 0
    action_for_probe = probe_factory(agent) if probe_factory is not None else None

    for _ in range(trials):
        cumulative = 0.0
        step = 0
        game_over = False
        environment.reset()
        pst = PolicyState(gap_threshold=gap_threshold)
        if needs_belief:
            pst.belief = set(agent.all_state)

        while step < max_steps and not game_over:
            old_state = tuple(environment.current_location)

            if attack in ("heuristic", "heuristic_eq22"):
                s_tilde = heuristic_attack_state(
                    agent, old_state, select, pst, environment,
                    action_for_probe=action_for_probe,
                    mode="eq22" if attack == "heuristic_eq22" else "code",
                )
            elif attack == "dqn" and attack_model is not None:
                s_tilde = apply_dqn_attack(attack_model, old_state)
            else:
                s_tilde = old_state

            t0 = time.perf_counter() if measure_time else None
            action = select(agent, s_tilde, pst, environment)
            if measure_time:
                times.append(time.perf_counter() - t0)

            _, reward, _, _, _ = environment.step(action)
            cumulative += reward
            step += 1

            if environment.check_state() == "TERMINAL":
                environment.__init__()
                game_over = True

        rewards.append(cumulative)
        total_esc += pst.escalations
        total_steps += pst.steps

    avg_ms = (np.mean(times) * 1000) if times else 0.0
    esc_rate = total_esc / max(total_steps, 1)
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "inference_ms": float(avg_ms),
        "escalate_rate": float(esc_rate),
    }


def train_agents(seed: int, baseline_trials: int = 2000, fine_trials: int = 500):
    """Train vanilla baseline, then pes and PB-GT from same Q init."""
    env_b = gw.GridWorld()
    env_b.seed(seed)
    agent_b = gw.Q_Agent(env_b)
    np.random.seed(seed + 1000)
    gw.play(env_b, agent_b, trials=baseline_trials, learn=True)
    baseline_q = copy.deepcopy(agent_b.q_table)

    # Pessimistic
    env_p = gw.GridWorld()
    env_p.seed(seed)
    agent_p = gw.Q_Agent(env_p)
    agent_p.q_table = copy.deepcopy(baseline_q)
    np.random.seed(seed + 1000)
    gw.play_att_with_belief(env_p, agent_p, trials=fine_trials, learn=True)

    # PB-GT
    env_o = gw.GridWorld()
    env_o.seed(seed)
    agent_o = gw.Q_Agent(env_o)
    agent_o.q_table = copy.deepcopy(baseline_q)
    np.random.seed(seed + 1000)
    gw.play_offpolicy_greedy_target(env_o, agent_o, trials=fine_trials, learn=True)

    return agent_b, agent_p, agent_o


def distill_belief_to_greedy(teacher, student, env, episodes: int = 300):
    """Fit student Q so greedy(s̃) matches teacher's belief policy under ω_π.

    Starts from teacher Q; tabular BC along attacked trajectories.
    Test-time student uses plain greedy (fast).
    """
    student.q_table_withatt = copy.deepcopy(teacher.q_table_withatt)
    # freeze a copy of teacher for rollouts
    t_q = copy.deepcopy(teacher.q_table_withatt)
    alpha = 0.25
    for _ep in range(episodes):
        belief = set(teacher.all_state)
        env.reset()
        # build initial belief-enriched policy from teacher Q
        teacher.q_table_withatt = t_q
        pol = teacher.pareto_update(teacher.get_policy(init=True))
        pol = teacher.get_policy_with_belief(pol, belief)
        for _step in range(200):
            true_s = tuple(env.current_location)
            s_tilde = teacher.get_att_state(true_s, pol)
            a_star = pol[s_tilde]
            q = student.q_table_withatt[s_tilde]
            base = max(q.values())
            for a in ACTIONS:
                target = base + (3.0 if a == a_star else -1.0)
                q[a] = (1 - alpha) * q[a] + alpha * target
            possible = gw.states_intersection(belief, teacher.get_ball(s_tilde))
            belief = {env.take_action(st, a_star) for st in possible}
            pol = teacher.get_policy_with_belief(pol, belief)
            env.step(a_star)
            if env.check_state() == "TERMINAL":
                env.__init__()
                break
    teacher.q_table_withatt = t_q
    return student


def distill_static_belief_policy(teacher, student, margin: float = 5.0):
    """One-shot: reshape student Q so greedy matches teacher π with M=all.

    Memoryless approximation of belief policy (no online belief). Fast at test.
    Also blends 50% teacher Q to preserve clean-task values.
    """
    t_q = copy.deepcopy(teacher.q_table_withatt)
    teacher.q_table_withatt = t_q
    belief = set(teacher.all_state)
    pol = teacher.pareto_update(teacher.get_policy(init=True))
    pol = teacher.get_policy_with_belief(pol, belief)
    student.q_table_withatt = copy.deepcopy(t_q)
    for s in teacher.all_state:
        a_star = pol[s]
        q = student.q_table_withatt[s]
        # preserve relative teacher values, enforce greedy(a_star)
        tvals = t_q[s]
        tmax = max(tvals.values())
        for a in ACTIONS:
            q[a] = 0.5 * tvals[a] + 0.5 * (tmax + (margin if a == a_star else -margin))
    teacher.q_table_withatt = t_q
    return student


def distill_rollout_belief(teacher, student, env, episodes: int = 400, alpha: float = 0.2):
    """BC along belief-tracking rollouts + keep teacher Q soft-anchor for clean perf."""
    t_q = copy.deepcopy(teacher.q_table_withatt)
    student.q_table_withatt = copy.deepcopy(t_q)
    for _ep in range(episodes):
        belief = set(teacher.all_state)
        env.reset()
        teacher.q_table_withatt = t_q
        pol = teacher.pareto_update(teacher.get_policy(init=True))
        pol = teacher.get_policy_with_belief(pol, belief)
        for _step in range(200):
            true_s = tuple(env.current_location)
            s_tilde = teacher.get_att_state(true_s, pol)
            a_star = pol[s_tilde]
            q = student.q_table_withatt[s_tilde]
            # soft anchor to teacher Q
            for a in ACTIONS:
                q[a] = 0.85 * q[a] + 0.15 * t_q[s_tilde][a]
            base = max(q.values())
            for a in ACTIONS:
                target = base + (4.0 if a == a_star else -2.0)
                q[a] = (1 - alpha) * q[a] + alpha * target
            possible = gw.states_intersection(belief, teacher.get_ball(s_tilde))
            belief = {env.take_action(st, a_star) for st in possible}
            pol = teacher.get_policy_with_belief(pol, belief)
            env.step(a_star)
            if env.check_state() == "TERMINAL":
                env.__init__()
                break
    teacher.q_table_withatt = t_q
    return student


def learn_mixed_target(environment, agent, trials=500, maximin_prob=0.5):
    """PB-GT behavior; with prob p use maximin target else greedy target."""
    agent.q_table_withatt = copy.deepcopy(agent.q_table)
    agent.defender_policy = agent.pareto_update(agent.get_policy(init=True))
    rewards = []
    for trial in range(trials):
        belief = set(agent.all_state)
        cumulative = 0.0
        step = 0
        game_over = False
        environment.reset()
        agent.defender_policy = agent.get_policy_with_belief(agent.defender_policy, belief)
        while step < 1000 and not game_over:
            old_state = tuple(environment.current_location)
            att_state = agent.get_att_state(old_state, agent.defender_policy)
            eps = gw.linear_schedule_exp(0.01, 0.2, trials, trial)
            action = agent.choose_action_att(environment.actions, att_state, eps)
            possible = gw.states_intersection(belief, agent.get_ball(att_state))
            belief = {environment.take_action(s, action) for s in possible}
            new_policy = agent.get_policy_with_belief(agent.defender_policy, belief)
            _, reward, _, _, _ = environment.step(action)
            new_state = environment.current_location
            alpha = gw.linear_schedule_exp(0.01, 0.1, trials, trial)
            if np.random.rand() < maximin_prob:
                agent.learn_with_att_with_belief(
                    old_state, reward, new_state, action,
                    agent.defender_policy, new_policy, belief, alpha,
                )
            else:
                agent.learn_with_greedy_target(old_state, reward, new_state, action, alpha)
            agent.defender_policy = new_policy
            cumulative += reward
            step += 1
            if environment.check_state() == "TERMINAL":
                environment.__init__()
                game_over = True
        rewards.append(cumulative)
    return rewards


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

POLICY_SPECS = [
    ("greedy", {}),
    ("ball_maximin", {}),
    ("ball_mean", {}),
    ("softmin_ball", {"temp": 0.5}),
    ("hybrid_ball", {"gap_threshold": 0.5}),
    ("hybrid_gap", {"gap_threshold": 0.5}),
    ("belief_maximin", {}),  # mean over belief (matches main GW)
    ("belief_min", {}),      # strict min over belief
]


FOCUS_EVALS = [
    # (train_key, test_policy_name, cfg)
    ("pessimistic", "belief_maximin", {}),
    ("pessimistic", "greedy", {}),
    ("pessimistic", "hybrid_disagree", {}),
    ("pessimistic", "ball_mean", {}),
    ("pbgt", "greedy", {}),
    ("pbgt", "belief_maximin", {}),
    ("pbgt", "hybrid_disagree", {}),
    ("pbgt", "ball_mean", {}),
    ("distill_static", "greedy", {}),
    ("distill_rollout", "greedy", {}),
    ("distill_static", "hybrid_disagree", {}),
]


def run_screen(
    n_seeds: int = 5,
    baseline_trials: int = 2000,
    fine_trials: int = 500,
    eval_trials: int = 100,
    csv_path: str = "pbgt_variant_screen.csv",
    use_dqn_attackers: bool = True,
    focus: bool = False,
):
    dqn_off = dqn_pes = None
    if use_dqn_attackers:
        if os.path.exists("attacker_offpolicy.zip"):
            dqn_off = gw.DQN.load("attacker_offpolicy")
            print("Loaded attacker_offpolicy")
        if os.path.exists("attacker_pessimistic.zip"):
            dqn_pes = gw.DQN.load("attacker_pessimistic")
            print("Loaded attacker_pessimistic")

    fieldnames = [
        "timestamp", "seed", "train", "test_policy",
        "attack", "mean_reward", "std_reward", "inference_ms",
        "escalate_rate", "speedup_vs_belief",
    ]
    write_header = not os.path.exists(csv_path)
    all_rows = []

    for seed in range(n_seeds):
        print(f"\n{'='*70}\nSEED {seed}/{n_seeds-1}: training agents...\n{'='*70}", flush=True)
        agent_b, agent_p, agent_o = train_agents(seed, baseline_trials, fine_trials)
        print("  trained baseline + pes + pbgt", flush=True)

        env_m = gw.GridWorld()
        env_m.seed(seed)
        agent_m = gw.Q_Agent(env_m)
        agent_m.q_table = copy.deepcopy(agent_b.q_table)
        np.random.seed(seed + 1000)
        learn_mixed_target(env_m, agent_m, trials=fine_trials, maximin_prob=0.5)
        print("  trained mixed50", flush=True)

        env_ds = gw.GridWorld()
        env_ds.seed(seed)
        agent_ds = gw.Q_Agent(env_ds)
        distill_static_belief_policy(agent_p, agent_ds)
        print("  distilled static belief->greedy", flush=True)

        env_dr = gw.GridWorld()
        env_dr.seed(seed)
        agent_dr = gw.Q_Agent(env_dr)
        np.random.seed(seed + 3000)
        distill_rollout_belief(agent_p, agent_dr, env_dr, episodes=400)
        print("  distilled rollout belief->greedy", flush=True)

        agents = {
            "pbgt": agent_o,
            "pessimistic": agent_p,
            "mixed50": agent_m,
            "distill_static": agent_ds,
            "distill_rollout": agent_dr,
        }

        env = gw.GridWorld()
        name, select, needs, probe_fac = make_policy("belief_maximin")
        print("  timing reference (pes+belief)...", flush=True)
        ref = eval_policy_fixed(
            env, agent_p, select, needs, trials=eval_trials,
            attack="heuristic", measure_time=True, probe_factory=probe_fac,
        )
        ref_ms = max(ref["inference_ms"], 1e-9)
        print(f"  ref belief ms={ref_ms:.4f} R={ref['mean_reward']:.1f}", flush=True)

        if focus:
            eval_list = FOCUS_EVALS
        else:
            eval_list = [
                (t, p, c) for t in agents for p, c in POLICY_SPECS
            ]

        seed_rows = []
        for train_name, pname, cfg in eval_list:
            if train_name not in agents:
                continue
            agent = agents[train_name]
            pname_full, select, needs, probe_fac = make_policy(pname, cfg)
            attacks = [("heuristic", None), ("none", None)]
            if dqn_off is not None:
                attacks.append(("dqn_offpolicy", dqn_off))
            if dqn_pes is not None and train_name in (
                "pessimistic", "distill", "distill_static", "distill_rollout"
            ):
                attacks.append(("dqn_pessimistic", dqn_pes))

            for attack_name, amodel in attacks:
                env = gw.GridWorld()
                np.random.seed(seed + 2000)
                atk = "dqn" if attack_name.startswith("dqn") else attack_name
                res = eval_policy_fixed(
                    env, agent, select, needs,
                    trials=eval_trials,
                    attack=atk,
                    attack_model=amodel,
                    measure_time=True,
                    gap_threshold=cfg.get("gap_threshold", 0.5),
                    probe_factory=probe_fac,
                )
                row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "seed": seed,
                    "train": train_name,
                    "test_policy": pname_full,
                    "attack": attack_name,
                    "mean_reward": res["mean_reward"],
                    "std_reward": res["std_reward"],
                    "inference_ms": res["inference_ms"],
                    "escalate_rate": res["escalate_rate"],
                    "speedup_vs_belief": ref_ms / max(res["inference_ms"], 1e-12),
                }
                seed_rows.append(row)
                print(
                    f"  [{train_name:11s}|{pname_full:18s}|{attack_name:16s}] "
                    f"R={res['mean_reward']:7.1f}  ms={res['inference_ms']:.4f}  "
                    f"x{row['speedup_vs_belief']:.1f}  esc={res['escalate_rate']:.2f}",
                    flush=True,
                )

        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
                write_header = False
            for r in seed_rows:
                w.writerow(r)
        all_rows.extend(seed_rows)
        print(f"  ✓ seed {seed} flushed to {csv_path}", flush=True)

    for atk in ["dqn_pessimistic", "dqn_offpolicy", "heuristic"]:
        summarize(csv_path, attack_filter=atk)
    return all_rows


def summarize(csv_path: str, attack_filter: str = "heuristic"):
    import pandas as pd

    df = pd.read_csv(csv_path)
    for c in ["mean_reward", "std_reward", "inference_ms", "escalate_rate", "speedup_vs_belief", "seed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["mean_reward", "train", "test_policy", "attack"])
    sub = df[df["attack"] == attack_filter]
    if sub.empty:
        print(f"\nNo rows for {attack_filter}")
        return
    g = (
        sub.groupby(["train", "test_policy"])
        .agg(
            mean_R=("mean_reward", "mean"),
            std_R=("mean_reward", "std"),
            ms=("inference_ms", "mean"),
            speedup=("speedup_vs_belief", "mean"),
            esc=("escalate_rate", "mean"),
            n=("mean_reward", "count"),
        )
        .reset_index()
        .sort_values("mean_R", ascending=False)
    )
    print(f"\n{'='*70}")
    print(f"SUMMARY under {attack_filter} attack (higher R better)")
    print(f"{'='*70}")
    print(g.to_string(index=False))

    # Belief path here is ~0.02ms; greedy ~0.006ms ⇒ ~3–3.5x is the practical ceiling
    # vs this timer (paper's ~11x used a heavier belief timing path).
    for target_R, target_x, label in [(-100, 3.0, "practical"), (-100, 5.0, "stretch")]:
        winners = g[(g["mean_R"] >= target_R) & (g["speedup"] >= target_x)]
        print(f"\nTarget[{label}]: R>={target_R} and speedup>={target_x}x")
        if len(winners):
            print("WINNERS:")
            print(winners.to_string(index=False))
        else:
            print("No config hit this target.")
    fast = g[g["speedup"] >= 2.5].copy()
    if len(fast):
        print("\nBest among speedup>=2.5x:")
        print(fast.head(10).to_string(index=False))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--baseline-trials", type=int, default=2000)
    ap.add_argument("--fine-trials", type=int, default=500)
    ap.add_argument("--eval-trials", type=int, default=100)
    ap.add_argument("--csv", type=str, default="pbgt_variant_screen.csv")
    ap.add_argument("--no-dqn", action="store_true")
    ap.add_argument("--focus", action="store_true", help="Only key configs + distill")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    if args.summarize_only:
        summarize(args.csv)
        summarize(args.csv, "dqn_offpolicy")
        return

    run_screen(
        n_seeds=args.seeds,
        baseline_trials=args.baseline_trials,
        fine_trials=args.fine_trials,
        eval_trials=args.eval_trials,
        csv_path=args.csv,
        use_dqn_attackers=not args.no_dqn,
        focus=args.focus,
    )


if __name__ == "__main__":
    main()

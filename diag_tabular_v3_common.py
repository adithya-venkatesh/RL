#!/usr/bin/env python3
"""Shared helpers for tabular v3 diagnosis (no edits to main_with_vanilla.py)."""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from main_with_vanilla import (
    GridWorld,
    GridWorld_RL_Attack,
    Q_Agent,
    play,
    play_att_with_belief,
    play_offpolicy_greedy_target,
    play_pessimistic_clean,
    play_vanilla_test,
)


def old_get_att_state(self, true_state, policy):
    """Pre-fix scoring: values[i] = Q(i, π(i)) at the candidate observation."""
    states = self.get_ball(true_state)
    values = dict()
    for i in states:
        values[i] = self.q_table_withatt[i][policy[i]]
    return min(values, key=values.get)


def fixed_get_att_state(self, true_state, policy):
    """Fixed scoring: values[i] = Q(true_state, π(i))."""
    states = self.get_ball(true_state)
    values = dict()
    for i in states:
        values[i] = self.q_table_withatt[true_state][policy[i]]
    return min(values, key=values.get)


def apply_scoring_mode(mode: str) -> None:
    """Monkeypatch Q_Agent.get_att_state in this process. mode in {old, fixed}."""
    if mode == "old":
        Q_Agent.get_att_state = old_get_att_state  # type: ignore[method-assign]
    elif mode == "fixed":
        Q_Agent.get_att_state = fixed_get_att_state  # type: ignore[method-assign]
    else:
        raise ValueError(mode)


def patch_vanilla_attacker_env() -> None:
    """Extend GridWorld_RL_Attack.step to support agent_type='vanilla' (q_table greedy)."""
    _orig_step = GridWorld_RL_Attack.step

    def step_with_vanilla(self, action):
        if self.agent_type != "vanilla":
            return _orig_step(self, action)

        last_location = self.current_location
        if action in [0, 1, 2, 3, 4]:
            true_action = self.actions[action]
        else:
            true_action = action

        if true_action == "UP":
            if last_location[0] == 0:
                self.purturb_state = self.current_location
            else:
                self.purturb_state = [self.current_location[0] - 1, self.current_location[1]]
        elif true_action == "DOWN":
            if last_location[0] == self.height - 1:
                self.purturb_state = self.current_location
            else:
                self.purturb_state = [self.current_location[0] + 1, self.current_location[1]]
        elif true_action == "LEFT":
            if last_location[1] == 0:
                self.purturb_state = self.current_location
            else:
                self.purturb_state = [self.current_location[0], self.current_location[1] - 1]
        elif true_action == "RIGHT":
            if last_location[1] == self.width - 1:
                self.purturb_state = self.current_location
            else:
                self.purturb_state = [self.current_location[0], self.current_location[1] + 1]
        elif true_action == "STAY":
            self.purturb_state = self.current_location
        else:
            raise ValueError(true_action)

        state = tuple(self.purturb_state)
        q_values = self.agentQ.q_table[state]
        max_value = max(q_values.values())
        agent_action = np.random.choice([k for k, v in q_values.items() if v == max_value])
        self.current_location = self.take_action(self.current_location, agent_action)

        terminated = list(self.current_location) in self.terminal_states
        reward = -self.get_reward(self.current_location)
        self.total_step += 1
        truncated = self.total_step >= 1000
        return np.asarray(self.current_location), reward, terminated, truncated, {}

    GridWorld_RL_Attack.step = step_with_vanilla  # type: ignore[method-assign]


def train_defenders(
    seed: int,
    scoring: str = "fixed",
) -> Tuple[Any, Any, Any, Dict[str, float]]:
    """Train vanilla / pes / PB-GT from the same baseline Q. Returns agents + clean means."""
    apply_scoring_mode(scoring)

    env_b = GridWorld()
    env_b.seed(seed)
    agent_v = Q_Agent(env_b)
    np.random.seed(seed + 1000)
    play(env_b, agent_v, trials=2000, learn=True)
    baseline_q = copy.deepcopy(agent_v.q_table)

    np.random.seed(seed + 1000)
    van_clean = play_vanilla_test(env_b, agent_v, trials=100)

    env_p = GridWorld()
    env_p.seed(seed)
    agent_p = Q_Agent(env_p)
    agent_p.q_table = copy.deepcopy(baseline_q)
    np.random.seed(seed + 1000)
    play_att_with_belief(env_p, agent_p, trials=500, learn=True)
    np.random.seed(seed + 1000)
    pes_clean = play_pessimistic_clean(env_p, agent_p, trials=100)

    env_o = GridWorld()
    env_o.seed(seed)
    agent_o = Q_Agent(env_o)
    agent_o.q_table = copy.deepcopy(baseline_q)
    np.random.seed(seed + 1000)
    play_offpolicy_greedy_target(env_o, agent_o, trials=500, learn=True)
    np.random.seed(seed + 1000)
    pbgt_clean = play_offpolicy_greedy_target(env_o, agent_o, trials=100, learn=False)

    cleans = {
        "vanilla_clean": float(np.mean(van_clean)),
        "pes_clean": float(np.mean(pes_clean)),
        "pbgt_clean": float(np.mean(pbgt_clean)),
    }
    return agent_v, agent_p, agent_o, cleans


def eval_attack(
    defender: str,
    agent,
    env_seed: int,
    attack_model,
    seed: int,
) -> Tuple[float, float, Optional[float]]:
    """Return (mean, std, inference_ms_or_None)."""
    env = GridWorld()
    env.seed(env_seed)
    np.random.seed(seed + 1000)

    if defender == "vanilla":
        rewards, t = play_vanilla_test(
            env, agent, trials=100, attack_model=attack_model, measure_time=True
        )
    elif defender == "pes":
        # Need agent's env consistent; rebuild with same Q
        rewards, t = play_att_with_belief(
            env, agent, trials=100, learn=False, attack_model=attack_model, measure_time=True
        )
    elif defender == "pbgt":
        rewards, t = play_offpolicy_greedy_target(
            env, agent, trials=100, learn=False, attack_model=attack_model, measure_time=True
        )
    else:
        raise ValueError(defender)

    return float(np.mean(rewards)), float(np.std(rewards)), float(t) * 1000.0


def q_table_stats(q_table_withatt: Dict) -> Tuple[float, float, float]:
    vals = []
    for d in q_table_withatt.values():
        vals.extend(d.values())
    a = np.asarray(vals, dtype=float)
    return float(a.min()), float(a.max()), float(a.mean())


def ms(a) -> str:
    a = np.asarray(a, dtype=float)
    return f"{a.mean():.2f}±{a.std(ddof=1):.2f}"


def med_iqr(a) -> str:
    a = np.asarray(a, dtype=float)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return f"median={med:.2f} IQR=[{q1:.2f},{q3:.2f}]"

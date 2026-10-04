#!/usr/bin/env python3
"""Belief-aware tabular attacker: obs = (r, c) + 25-dim 0/1 belief mask.

The mask is the defender belief M_t that maximin uses for the current step
(after reset / previous belief update, before this step's action).
"""
from __future__ import annotations

import copy
from typing import Set, Tuple

import numpy as np
from gymnasium import spaces

from main_with_vanilla import (
    GridWorld,
    GridWorld_RL_Attack,
    apply_perturbation,
    states_intersection,
)

HEIGHT = 5
WIDTH = 5
N_STATES = HEIGHT * WIDTH
OBS_DIM = 2 + N_STATES  # (r, c) + belief mask


def state_index(state) -> int:
    r, c = int(state[0]), int(state[1])
    return r * WIDTH + c


def belief_to_mask(belief: Set[Tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(N_STATES, dtype=np.float32)
    for s in belief:
        mask[state_index(s)] = 1.0
    return mask


def make_belief_obs(state, belief: Set[Tuple[int, int]]) -> np.ndarray:
    """Float32 vector: [row, col, mask_0, ..., mask_24]."""
    s = (int(state[0]), int(state[1]))
    return np.concatenate(
        [np.asarray([s[0], s[1]], dtype=np.float32), belief_to_mask(belief)]
    )


def mask_to_belief(mask: np.ndarray) -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    for i, v in enumerate(np.asarray(mask).reshape(-1)):
        if float(v) >= 0.5:
            out.add((i // WIDTH, i % WIDTH))
    return out


def init_full_belief(agent) -> Set[Tuple[int, int]]:
    return set(agent.all_state)


def update_belief_after_action(
    belief: Set[Tuple[int, int]],
    agent,
    environment,
    att_state,
    action,
) -> Set[Tuple[int, int]]:
    """Same M update as play_att_with_belief / GridWorld_RL_Attack.step."""
    possible = agent.get_ball(att_state)
    before = states_intersection(belief, possible)
    nxt: Set[Tuple[int, int]] = set()
    for state in before:
        nxt.add(tuple(environment.take_action(state, action)))
    return nxt


class GridWorld_RL_Attack_BeliefAware(GridWorld_RL_Attack):
    """Same attacker MDP as GridWorld_RL_Attack, but obs = state + belief mask."""

    def __init__(self, agentQ, agent_type: str = "pessimistic"):
        if agent_type != "pessimistic":
            raise ValueError(
                "Belief-aware attackers are trained against maximin "
                f"(agent_type='pessimistic'); got {agent_type!r}"
            )
        super().__init__(agentQ, agent_type=agent_type)
        self.observation_space = spaces.Box(
            low=0.0, high=float(max(HEIGHT, WIDTH) - 1), shape=(OBS_DIM,), dtype=np.float32
        )

    def _obs(self) -> np.ndarray:
        return make_belief_obs(self.current_location, self.belief)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Gymnasium API: DummyVecEnv unpacks (obs, info). Returning a bare
        # length-27 vector would be mis-unpacked (the length-2 parent obs
        # only works by accident).
        return self._obs(), {}

    def step(self, action):
        # Parent returns (location, reward, terminated, truncated, info).
        # Belief is updated inside the pessimistic branch before return.
        _, reward, terminated, truncated, info = super().step(action)
        return self._obs(), reward, terminated, truncated, info


def _predict_action(attack_model, obs: np.ndarray) -> int:
    out = attack_model.predict(obs, deterministic=True)
    att = out[0] if isinstance(out, tuple) else out
    if hasattr(att, "item"):
        return int(att.item())
    if isinstance(att, (list, tuple, np.ndarray)):
        return int(np.asarray(att).reshape(-1)[0])
    return int(att)


def _dir_from_action(att_action: int):
    return ["UP", "DOWN", "LEFT", "RIGHT", "STAY"][int(att_action)]


def play_maximin_belief_aware(
    environment,
    agent,
    trials: int = 100,
    max_steps_per_episode: int = 1000,
    attack_model=None,
    pol_snap=None,
    record_masks: bool = False,
):
    """Maximin test with belief-aware attacker observations.

    If record_masks, returns (rewards, list of (obs_mask, defender_belief_set)
    pairs from every attacker decision) for unit tests.
    """
    if pol_snap is not None:
        agent.defender_policy = copy.deepcopy(pol_snap)

    reward_per_episode = []
    recorded = []

    for _ in range(trials):
        belief = init_full_belief(agent)
        cumulative_reward = 0.0
        step = 0
        game_over = False
        environment.reset()
        agent.defender_policy = agent.get_policy_with_belief(
            agent.defender_policy, belief
        )

        while step < max_steps_per_episode and not game_over:
            old_state = tuple(environment.current_location)
            obs = make_belief_obs(old_state, belief)
            if record_masks:
                recorded.append((obs[2:].copy(), set(belief)))

            if attack_model is not None:
                att_action = _predict_action(attack_model, obs)
                true_action = _dir_from_action(att_action)
                att_state = tuple(apply_perturbation(old_state, true_action))
            else:
                att_state = agent.get_att_state(old_state, agent.defender_policy)

            action = agent.choose_action_att(environment.actions, att_state, 0)
            belief = update_belief_after_action(
                belief, agent, environment, att_state, action
            )
            new_policy = agent.get_policy_with_belief(agent.defender_policy, belief)
            _, reward, _, _, _ = environment.step(action)
            agent.defender_policy = new_policy
            cumulative_reward += reward
            step += 1
            if environment.check_state() == "TERMINAL":
                environment.__init__()
                game_over = True

        reward_per_episode.append(cumulative_reward)

    if record_masks:
        return reward_per_episode, recorded
    return reward_per_episode


def play_greedy_shadow_belief_aware(
    environment,
    agent,
    trials: int = 100,
    max_steps_per_episode: int = 1000,
    attack_model=None,
    q_source: str = "withatt",
):
    """Greedy test (vanilla / pes-greedy / PB-GT-greedy) with shadow belief for obs.

    Defender acts greedily on s̃; attacker still sees a belief mask updated with
    the same M-update as maximin (shadow of what maximin would track).
    """
    reward_per_episode = []
    for _ in range(trials):
        belief = init_full_belief(agent)
        cumulative_reward = 0.0
        step = 0
        game_over = False
        environment.reset()

        while step < max_steps_per_episode and not game_over:
            old_state = tuple(environment.current_location)
            obs = make_belief_obs(old_state, belief)

            if attack_model is not None:
                att_action = _predict_action(attack_model, obs)
                true_action = _dir_from_action(att_action)
                att_state = tuple(apply_perturbation(old_state, true_action))
            else:
                att_state = old_state

            q_table = agent.q_table if q_source == "vanilla" else agent.q_table_withatt
            q_values = q_table[att_state]
            max_value = max(q_values.values())
            action = np.random.choice(
                [k for k, v in q_values.items() if v == max_value]
            )

            belief = update_belief_after_action(
                belief, agent, environment, att_state, action
            )
            _, reward, _, _, _ = environment.step(action)
            cumulative_reward += reward
            step += 1
            if environment.check_state() == "TERMINAL":
                environment.__init__()
                game_over = True

        reward_per_episode.append(cumulative_reward)
    return reward_per_episode


def eval_belief_aware_mean(
    rule: str,
    agent,
    seed: int,
    attack_model,
    pol_snap=None,
) -> float:
    """rule in {vanilla, pes_q_maximin, pes_q_greedy, pbgt_q_maximin, pbgt_q_greedy}."""
    env = GridWorld()
    env.seed(seed)
    np.random.seed(seed + 1000)

    if rule in ("pes_q_maximin", "pbgt_q_maximin"):
        rew = play_maximin_belief_aware(
            env, agent, trials=100, attack_model=attack_model, pol_snap=pol_snap
        )
    elif rule == "vanilla":
        rew = play_greedy_shadow_belief_aware(
            env, agent, trials=100, attack_model=attack_model, q_source="vanilla"
        )
    elif rule in ("pes_q_greedy", "pbgt_q_greedy"):
        rew = play_greedy_shadow_belief_aware(
            env, agent, trials=100, attack_model=attack_model, q_source="withatt"
        )
    else:
        raise ValueError(rule)
    return float(np.mean(rew))

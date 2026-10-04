# -*- coding: utf-8 -*-
"""Shared eval helpers for the sound-belief confirmation runs.

Harness only: nothing here changes a decision rule or a learning update. It
adds three things the repo's helpers do not provide.

1. ``truly_clean_maximin`` -- the repo's ``evaluate_agent(mode="pessimistic")``
   clean path runs a 32-candidate random-search PGD before acting, so its
   "clean" column is not clean. Greedy's clean path does use the true obs, so
   the two columns were not comparable.
2. ``train_attacker_with_diag`` -- mirrors ``train_cartpole_attacker`` but wraps
   the attack env in a Monitor so the attacker's own episode-length curve can be
   inspected. A matched-attacker score of 500 is only a defender win if the
   attacker actually found shorter episodes during training; otherwise its
   reward is a constant -1/step and it had no learning signal at all.
3. ``MaskedAttackEnv`` / ``masked_delta`` -- restricted threat models where only
   a subset of sensors is attacked (position sensors: eps_vec = [e, 0, e, 0]).
"""

from __future__ import annotations

import csv
import itertools
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from stable_baselines3 import DQN as SB3_DQN
from stable_baselines3.common.monitor import Monitor

from main_cartpole_dqn import (
    CartPoleAttackEnv,
    CartPoleDefenderEnv,
    DQNAgent,
    _note_belief_action,
    apply_perturbation,
    normalize_obs,
    set_seed,
)

VACUOUS_EP_LEN = 490.0
N_ATT_LEVELS = 3


def load_result_rows(path: str) -> List[Dict[str, Any]]:
    """Reload a partially written results CSV so a killed run can resume."""
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        raw = list(csv.DictReader(f))
    out: List[Dict[str, Any]] = []
    for r in raw:
        nr: Dict[str, Any] = {}
        for k, v in r.items():
            if v is None or v == "":
                nr[k] = v
                continue
            try:
                nr[k] = int(v)
            except ValueError:
                try:
                    nr[k] = float(v)
                except ValueError:
                    nr[k] = v
        out.append(nr)
    return out


def write_result_rows(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    extra = [k for r in rows for k in r if k not in fields]
    # preserve first-row order, then any keys that appeared later
    for k in extra:
        if k not in fields:
            fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# clean eval that is actually clean
# --------------------------------------------------------------------------- #

def truly_clean_maximin(
    agent: DQNAgent,
    provider,
    n_ep: int,
    seed: int,
) -> List[float]:
    """Maximin with the belief centred on the *unperturbed* observation."""
    set_seed(seed)
    env = CartPoleDefenderEnv()
    out: List[float] = []
    eps = agent.config.attack_epsilon
    kappa = agent.config.belief_kappa
    for ep in range(n_ep):
        obs, _ = env.reset(seed=seed + ep)
        if provider is not None:
            provider.reset_episode()
        done, ret = False, 0.0
        while not done:
            action = agent.maximin_action(obs, eps, kappa, belief_provider=provider)
            _note_belief_action(provider, action)
            obs, r, term, trunc, _ = env.step(action)
            ret += r
            done = term or trunc
        out.append(ret)
    env.close()
    return out


# --------------------------------------------------------------------------- #
# restricted threat models
# --------------------------------------------------------------------------- #

def masked_delta(action: int, eps_vec: np.ndarray, dims: Sequence[int]) -> np.ndarray:
    """Discrete delta over ``dims`` only; each dim takes {-eps, 0, +eps}."""
    delta = np.zeros(4, dtype=np.float32)
    rem = int(action)
    for d in dims:
        delta[d] = (-eps_vec[d], 0.0, eps_vec[d])[rem % N_ATT_LEVELS]
        rem //= N_ATT_LEVELS
    return delta


class MaskedAttackEnv(CartPoleAttackEnv):
    """``CartPoleAttackEnv`` restricted to perturbing ``dims`` with ``eps_vec``.

    Step body mirrors the parent exactly; only the action -> delta map changes.
    """

    def __init__(self, *args, eps_vec: np.ndarray, dims: Sequence[int], **kwargs):
        self.eps_vec = np.asarray(eps_vec, dtype=np.float64)
        self.dims = tuple(dims)
        super().__init__(*args, **kwargs)
        # Same space class the parent already constructed (gymnasium Discrete).
        n = N_ATT_LEVELS ** len(self.dims)
        self.action_space = type(self.action_space)(n)

    def step(self, action):
        true_norm = normalize_obs(self.inner.get_true_state())
        delta = masked_delta(int(action), self.eps_vec, self.dims)
        self.perturbed_state = apply_perturbation(true_norm, delta, normalized=True)
        if self.belief_provider is not None:
            self.belief_provider.observe(self.perturbed_state)
        defender_action = self._defender_action(self.perturbed_state)
        _note_belief_action(self.belief_provider, defender_action)
        obs, reward, terminated, truncated, info = self.inner.step(defender_action)
        if self.belief_provider is not None:
            self.belief = self.belief_provider.observe(obs)
        return obs, -float(reward), terminated, truncated, info


def perturb_from_masked_attacker(
    true_obs: np.ndarray, attacker, eps_vec: np.ndarray, dims: Sequence[int]
) -> np.ndarray:
    raw, _ = attacker.predict(true_obs, deterministic=True)
    action = int(np.asarray(raw).reshape(-1)[0])
    return apply_perturbation(true_obs, masked_delta(action, eps_vec, dims), normalized=True)


# --------------------------------------------------------------------------- #
# attacker training with a learning-signal diagnostic
# --------------------------------------------------------------------------- #

def train_attacker_with_diag(
    defender: DQNAgent,
    agent_type: str,
    total_timesteps: int,
    save_path: str,
    belief_provider=None,
    epsilon: Optional[float] = None,
    eps_vec: Optional[np.ndarray] = None,
    dims: Optional[Sequence[int]] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """Same hyperparameters as ``train_cartpole_attacker``, plus Monitor stats."""
    eps = defender.config.attack_epsilon if epsilon is None else epsilon
    if eps_vec is not None:
        env = MaskedAttackEnv(
            defender,
            agent_type=agent_type,
            epsilon=eps,
            belief_provider=belief_provider,
            eps_vec=eps_vec,
            dims=dims or (0, 2),
        )
    else:
        env = CartPoleAttackEnv(
            defender,
            agent_type=agent_type,
            epsilon=eps,
            belief_provider=belief_provider,
        )
    mon = Monitor(env)
    model = SB3_DQN(
        "MlpPolicy",
        mon,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=128,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=0,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    lens = np.asarray(mon.get_episode_lengths(), dtype=np.float64)
    mon.close()
    if lens.size == 0:
        return {
            "atk_n_train_eps": 0.0,
            "atk_len_mean": float("nan"),
            "atk_len_min": float("nan"),
            "atk_len_last10": float("nan"),
        }
    return {
        "atk_n_train_eps": float(lens.size),
        "atk_len_mean": float(lens.mean()),
        "atk_len_min": float(lens.min()),
        "atk_len_last10": float(lens[-10:].mean()),
    }


def attacker_is_vacuous(defender_score: float, diag: Dict[str, float]) -> bool:
    """True when 500 means 'no attacker was found', not 'attacker was beaten'.

    The attacker's reward is -1 per surviving step, so if it never once ended an
    episode early during training its return is a constant and it has zero
    gradient signal. Such a cell carries no robustness evidence.
    """
    if defender_score < VACUOUS_EP_LEN:
        return False
    lmin = diag.get("atk_len_min", float("nan"))
    if not np.isfinite(lmin):
        return True
    return bool(lmin >= VACUOUS_EP_LEN)


# --------------------------------------------------------------------------- #
# epsilon-sweep attacks (same definitions as cartpole_sound_epssweep.py)
# --------------------------------------------------------------------------- #

def pole_deltas(eps: float) -> List[np.ndarray]:
    """9 constant biases on (theta, theta_dot)."""
    lv = (-eps, 0.0, eps)
    return [np.array([0.0, 0.0, a, b], dtype=np.float32) for a, b in itertools.product(lv, lv)]


def const_bias_attack(delta: np.ndarray) -> Callable:
    return lambda obs, agent, act, prov: apply_perturbation(obs, delta, normalized=True)


def exhaustive_greedy_attack(deltas: List[np.ndarray]) -> Callable:
    """delta_t = argmin_delta Q(s_true, pi(s_tilde)) over the discrete set.

    Probing pi() advances a history-conditioned belief, so the filter state is
    snapshotted and restored around the candidate loop.
    """

    def fn(obs, agent, act, prov):
        q_true = agent.q_values(obs)
        snap = prov.snapshot() if hasattr(prov, "snapshot") else None
        best, best_score = obs, float("inf")
        for d in deltas:
            s_t = apply_perturbation(obs, d, normalized=True)
            score = float(q_true[act(s_t)])
            if snap is not None:
                prov.restore(snap)
            if score < best_score:
                best_score, best = score, s_t
        return best
    return fn


def sweep_rollout(agent, act, prov, delta_fn: Callable, n_ep: int, seed: int) -> List[float]:
    set_seed(seed)
    env = CartPoleDefenderEnv()
    rets: List[float] = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=seed + ep)
        if prov is not None:
            prov.reset_episode()
        done, ret = False, 0.0
        while not done:
            a = act(delta_fn(obs, agent, act, prov))
            _note_belief_action(prov, a)
            obs, r, term, trunc, _ = env.step(a)
            ret += r
            done = term or trunc
        rets.append(ret)
    env.close()
    return rets

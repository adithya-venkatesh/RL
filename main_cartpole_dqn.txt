# -*- coding: utf-8 -*-
"""main_cartpole_dqn.py

CartPole DQN: Vanilla vs Pessimistic vs Off-Policy Greedy (Steps 1–6).

Reference: Sun & Zheng (ICLR 2024) — Belief-Enriched Pessimistic Q-Learning
Against Adversarial State Perturbations.

Step 1: CartPole-v1 environment wrappers + vanilla DQN baseline.
Steps 2–5: Attack eval, pessimistic DQN, off-policy DQN, attacker training, experiment pipeline.
"""

import csv
import math
import os
import pickle
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import DQN as SB3_DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from cartpole_belief import (
    BeliefRNN,
    BeliefProvider,
    UniformBallBelief,
    load_belief_rnn,
    make_belief_provider,
    pretrain_belief_rnn_on_agent,
    save_belief_rnn,
)

# CartPole-v1 observation limits used for clipping perturbed states (Eq. 1.4).
CARTPOLE_LOW = np.array([-4.8, -5.0, -0.418, -5.0], dtype=np.float32)
CARTPOLE_HIGH = np.array([4.8, 5.0, 0.418, 5.0], dtype=np.float32)
DEFAULT_ATTACK_EPS = 0.05
DEFAULT_BELIEF_KAPPA = 20
PERTURB_LEVELS = 3  # per-dimension: {-ε, 0, +ε}
N_DISCRETE_ATTACK_ACTIONS = PERTURB_LEVELS ** 4


def delta_from_discrete_action(action: int, epsilon: float = DEFAULT_ATTACK_EPS) -> np.ndarray:
    """Map discrete attacker action to δ ∈ [-ε, ε]^4 (SB3 DQN-compatible action space)."""
    levels = np.array([-epsilon, 0.0, epsilon], dtype=np.float32)
    rem = int(action)
    delta = np.zeros(4, dtype=np.float32)
    for i in range(4):
        delta[i] = levels[rem % PERTURB_LEVELS]
        rem //= PERTURB_LEVELS
    return delta


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quick_eval_mean(
    agent: "DQNAgent",
    n_episodes: int,
    seed: int,
    mode: str = "vanilla",
) -> float:
    """Fast greedy/maximin eval during pessimistic fine-tune (no timing)."""
    returns, _ = evaluate_agent(agent, n_episodes, seed, mode, measure_time=False)
    return float(np.mean(returns))


def linear_schedule_exp(min_exp: float, ini_exp: float, total: int, current: int) -> float:
    """**Eq. 6.3:** ε_t = max(ε_min, (T − t) · ε_init / T)"""
    return max(min_exp, ((total - current) * ini_exp) / total)


def normalize_obs(
    obs: np.ndarray,
    low: np.ndarray = CARTPOLE_LOW,
    high: np.ndarray = CARTPOLE_HIGH,
) -> np.ndarray:
    """Scale observations to approximately [-1, 1] for stable MLP training."""
    return (2.0 * (obs - low) / (high - low) - 1.0).astype(np.float32)


def denormalize_obs(
    obs_norm: np.ndarray,
    low: np.ndarray = CARTPOLE_LOW,
    high: np.ndarray = CARTPOLE_HIGH,
) -> np.ndarray:
    return ((obs_norm + 1.0) * 0.5 * (high - low) + low).astype(np.float32)


def apply_perturbation(
    state: np.ndarray,
    delta: np.ndarray,
    low: np.ndarray = CARTPOLE_LOW,
    high: np.ndarray = CARTPOLE_HIGH,
    normalized: bool = True,
) -> np.ndarray:
    """**Eq. 1.4:** s̃ = clip(s + δ, obs_low, obs_high)  with δ ∈ B_ε(s)"""
    if normalized:
        raw = denormalize_obs(state, low, high)
        perturbed = np.clip(raw + delta, low, high).astype(np.float32)
        return normalize_obs(perturbed, low, high)
    return np.clip(state + delta, low, high).astype(np.float32)


def sample_ball(
    state: np.ndarray,
    epsilon: float,
    n_samples: int,
    low: np.ndarray = CARTPOLE_LOW,
    high: np.ndarray = CARTPOLE_HIGH,
    normalized: bool = True,
) -> List[np.ndarray]:
    """**Eq. 1.4:** Sample κ points uniformly from B_ε(s) (L∞ ball, clipped to bounds)."""
    raw = denormalize_obs(state, low, high) if normalized else state
    samples = []
    for _ in range(n_samples):
        delta = np.random.uniform(-epsilon, epsilon, size=raw.shape).astype(np.float32)
        if normalized:
            samples.append(apply_perturbation(state, delta, low, high, normalized=True))
        else:
            samples.append(np.clip(raw + delta, low, high).astype(np.float32))
    return samples


class CartPoleDefenderEnv(gym.Wrapper):
    """**Eq. 1.1:** MDP M = ⟨S, A, P, R, γ⟩ on CartPole-v1.

    Thin wrapper exposing observation bounds and true state for defender training.
    Perturbations are applied in the training loop (not inside step).
    """

    def __init__(self, render_mode: Optional[str] = None):
        env = gym.make("CartPole-v1", render_mode=render_mode)
        super().__init__(env)
        self.obs_low = CARTPOLE_LOW.copy()
        self.obs_high = CARTPOLE_HIGH.copy()
        self.true_state: Optional[np.ndarray] = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.true_state = np.asarray(obs, dtype=np.float32)
        return normalize_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.true_state = np.asarray(obs, dtype=np.float32)
        return normalize_obs(obs), reward, terminated, truncated, info

    def get_true_state(self) -> np.ndarray:
        if self.true_state is None:
            raise RuntimeError("Environment must be reset before reading true state.")
        return self.true_state.copy()


class CartPoleAttackEnv(gym.Env):
    """**Eq. 1.5–1.6:** Attacker MDP ⟨S, A_att, P̃, R̃, γ⟩

    Attacker chooses δ; defender observes s̃ = clip(s + δ); attacker reward R̃ = −R(s,a).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        defender_agent: "DQNAgent",
        agent_type: str = "vanilla",
        epsilon: float = DEFAULT_ATTACK_EPS,
        belief_provider: Optional[BeliefProvider] = None,
    ):
        self.defender = defender_agent
        self.agent_type = agent_type
        self.epsilon = epsilon
        self.kappa = DEFAULT_BELIEF_KAPPA
        self.belief_provider = belief_provider
        self.inner = CartPoleDefenderEnv()
        self.action_space = gym.spaces.Discrete(N_DISCRETE_ATTACK_ACTIONS)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.perturbed_state: Optional[np.ndarray] = None
        self.belief: List[np.ndarray] = []

    def _defender_action(self, perturbed_norm: np.ndarray) -> int:
        if self.agent_type == "pessimistic":
            return self.defender.maximin_action(
                perturbed_norm,
                self.epsilon,
                self.kappa,
                belief_provider=self.belief_provider,
            )
        return self.defender.greedy_action(perturbed_norm)

    def reset(self, *, seed=None, options=None):
        obs, info = self.inner.reset(seed=seed, options=options)
        self.perturbed_state = obs.copy()
        if self.belief_provider is not None:
            self.belief_provider.reset_episode()
            self.belief = self.belief_provider.observe(obs)
        else:
            self.belief = sample_ball(obs, self.epsilon, self.kappa)
        return obs, info

    def step(self, action):
        """**Eq. 1.4–1.6:** Attacker picks δ; defender acts on s̃; R̃ = −R(s,a)."""
        true_norm = normalize_obs(self.inner.get_true_state())
        delta = delta_from_discrete_action(int(action), self.epsilon)
        self.perturbed_state = apply_perturbation(true_norm, delta, normalized=True)
        if self.belief_provider is not None:
            self.belief_provider.observe(self.perturbed_state)
        defender_action = self._defender_action(self.perturbed_state)
        obs, reward, terminated, truncated, info = self.inner.step(defender_action)
        if self.belief_provider is not None:
            self.belief = self.belief_provider.observe(obs)
        else:
            self.belief = sample_ball(obs, self.epsilon, self.kappa)
        attacker_reward = -float(reward)
        return obs, attacker_reward, terminated, truncated, info

    def close(self):
        self.inner.close()


class ReplayBuffer:
    """Experience replay buffer for vanilla DQN."""

    def __init__(self, capacity: int):
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class BeliefReplayBuffer:
    """Replay buffer storing belief particles M_t for pessimistic DQN (Algorithm 4)."""

    def __init__(self, capacity: int, kappa: int = DEFAULT_BELIEF_KAPPA):
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]] = deque(maxlen=capacity)
        self.kappa = kappa

    def push(self, state, action, reward, next_state, done, next_belief: List[np.ndarray]):
        belief_arr = np.stack(next_belief).astype(np.float32)
        self.buffer.append(
            (np.asarray(state, dtype=np.float32), int(action), float(reward),
             np.asarray(next_state, dtype=np.float32), bool(done), belief_arr)
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, beliefs = zip(*batch)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
            np.stack(beliefs),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """**Eq. 1.3:** Q_θ(s, ·) — MLP approximator for CartPole state vectors."""

    def __init__(self, obs_dim: int = 4, n_actions: int = 2, hidden: Tuple[int, ...] = (128, 128)):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = obs_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 128
    buffer_size: int = 100_000
    learning_starts: int = 1_000
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_episodes: int = 10
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 50_000
    hidden_sizes: Tuple[int, ...] = (128, 128)
    device: str = "cpu"
    attack_epsilon: float = DEFAULT_ATTACK_EPS
    belief_kappa: int = DEFAULT_BELIEF_KAPPA
    pgd_candidates: int = 32


class DQNAgent:
    """DQN agent: vanilla (Step 1), pessimistic (Step 3), off-policy greedy (Step 4)."""

    def __init__(self, n_actions: int = 2, obs_dim: int = 4, config: Optional[DQNConfig] = None):
        self.config = config or DQNConfig()
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.device = torch.device(self.config.device)

        self.q_net = QNetwork(obs_dim, n_actions, self.config.hidden_sizes).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, self.config.hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.lr)
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)
        self.belief_replay_buffer = BeliefReplayBuffer(self.config.buffer_size, self.config.belief_kappa)
        self.total_steps = 0

    def clone_weights_from(self, other: "DQNAgent"):
        """Copy network weights from another agent (for robust fine-tuning from vanilla)."""
        self.q_net.load_state_dict(other.q_net.state_dict())
        self.target_net.load_state_dict(other.target_net.state_dict())

    def prepare_robust_finetune(self, lr: float = 5e-5, freeze_backbone: bool = True):
        """Reset buffers and lower lr/ε for fine-tuning from a trained vanilla policy.

        freeze_backbone: train only the final linear layer (reduces catastrophic forgetting).
        """
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)
        self.belief_replay_buffer = BeliefReplayBuffer(self.config.buffer_size, self.config.belief_kappa)
        self._set_backbone_frozen(freeze_backbone)
        trainable = [p for p in self.q_net.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable, lr=lr)
        self.config.epsilon_start = 0.02
        self.config.epsilon_end = 0.005
        self.config.epsilon_decay_steps = 15_000

    def _set_backbone_frozen(self, freeze: bool) -> None:
        """Freeze all Q-network layers except the final linear head."""
        modules = list(self.q_net.net.children())
        for i, module in enumerate(modules):
            is_head = i == len(modules) - 1
            for param in module.parameters():
                param.requires_grad = (not freeze) or is_head

    def snapshot_weights(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.q_net.state_dict().items()}

    def restore_weights(self, snap: dict) -> None:
        device = self.device
        self.q_net.load_state_dict({k: v.to(device) for k, v in snap.items()})
        self.update_target()

    def prepare_offpolicy_finetune(self, lr: float = 1e-5):
        """Gentler fine-tune for off-policy greedy (Eq. 5.1–5.2): pessimistic behavior, greedy target."""
        self.prepare_robust_finetune(lr=lr)
        self.config.epsilon_start = 0.01
        self.config.epsilon_end = 0.005
        self.config.epsilon_decay_steps = 10_000

    def q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.q_net(s).cpu().numpy()[0]

    def q_value(self, state: np.ndarray, action: int) -> float:
        return float(self.q_values(state)[action])

    def greedy_action(self, state: np.ndarray) -> int:
        """**Eq. 4.2:** a = argmax_a Q(s, a)"""
        return int(np.argmax(self.q_values(state)))

    def maximin_action(
        self,
        center_state: np.ndarray,
        epsilon: float,
        kappa: int,
        particles: Optional[List[np.ndarray]] = None,
        belief_provider: Optional[BeliefProvider] = None,
    ) -> int:
        """**Eq. 3.4 / Algorithm 5, Line 7:** a = argmax_a min_{m∈M∩B_ε(s̃)} Q(m,a)"""
        if particles is None:
            if belief_provider is not None:
                particles = belief_provider.particles_at(center_state)
            else:
                particles = sample_ball(center_state, epsilon, kappa)
        action_mins = []
        for a in range(self.n_actions):
            action_mins.append(min(self.q_value(p, a) for p in particles))
        return int(np.argmax(action_mins))

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """**Eq. 2.3 / Eq. 4.2:** ε-greedy action selection."""
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        return self.greedy_action(state)

    def worst_case_perturbation(self, true_state: np.ndarray, epsilon: float) -> np.ndarray:
        """**Eq. 2.2 / Algorithm 1, Line 6:** s̃* = argmin_{s̃∈B_ε(s)} Q(s, π(s̃))"""
        best_tilde = true_state
        best_score = float("inf")
        raw = denormalize_obs(true_state)
        for _ in range(self.config.pgd_candidates):
            delta = np.random.uniform(-epsilon, epsilon, size=raw.shape).astype(np.float32)
            s_tilde = apply_perturbation(true_state, delta, normalized=True)
            a = self.greedy_action(s_tilde)
            score = self.q_value(true_state, a)
            if score < best_score:
                best_score = score
                best_tilde = s_tilde
        return best_tilde

    def _compute_td_targets(
        self, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray
    ) -> torch.Tensor:
        """**Eq. 4.1 / Eq. 1.3:** Double DQN greedy target."""
        with torch.no_grad():
            next_s = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
            next_actions = self.q_net(next_s).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_s).gather(1, next_actions).squeeze(1)
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
            return rewards_t + self.config.gamma * next_q * (1.0 - dones_t)

    def _compute_pessimistic_td_targets(
        self, rewards: np.ndarray, next_beliefs: np.ndarray, dones: np.ndarray
    ) -> torch.Tensor:
        """**Eq. 2.6 / Algorithm 4, Line 13:** y = r + γ max_{a′} min_{m∈M′} Q_θ′(m,a′)"""
        with torch.no_grad():
            batch, kappa, _ = next_beliefs.shape
            nb = torch.as_tensor(next_beliefs, dtype=torch.float32, device=self.device)
            nb_flat = nb.view(batch * kappa, self.obs_dim)
            q_all = self.target_net(nb_flat).view(batch, kappa, self.n_actions)
            min_per_action = q_all.min(dim=1).values
            max_q = min_per_action.max(dim=1).values
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
            return rewards_t + self.config.gamma * max_q * (1.0 - dones_t)

    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.config.learning_starts:
            return None
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        targets = self._compute_td_targets(rewards, next_states, dones)
        q_sa = self.q_net(states_t).gather(1, actions_t).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q_sa, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def train_step_pessimistic(self) -> Optional[float]:
        if len(self.belief_replay_buffer) < self.config.learning_starts:
            return None
        states, actions, rewards, next_states, dones, beliefs = self.belief_replay_buffer.sample(
            self.config.batch_size
        )
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        targets = self._compute_pessimistic_td_targets(rewards, beliefs, dones)
        q_sa = self.q_net(states_t).gather(1, actions_t).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q_sa, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def train_step_offpolicy_greedy(self) -> Optional[float]:
        """**Eq. 5.2:** Greedy bootstrap target with belief replay (pessimistic behavior at collect)."""
        if len(self.belief_replay_buffer) < self.config.learning_starts:
            return None
        states, actions, rewards, next_states, dones, _beliefs = self.belief_replay_buffer.sample(
            self.config.batch_size
        )
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        targets = self._compute_td_targets(rewards, next_states, dones)
        q_sa = self.q_net(states_t).gather(1, actions_t).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q_sa, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str):
        torch.save(
            {"q_net": self.q_net.state_dict(), "target_net": self.target_net.state_dict(), "config": self.config},
            path,
        )

    def load(self, path: str):
        import sys
        # Checkpoints saved via `python main_cartpole_dqn.py` pickle config as __main__.DQNConfig.
        main_mod = sys.modules.get("__main__")
        if main_mod is not None and not hasattr(main_mod, "DQNConfig"):
            main_mod.DQNConfig = DQNConfig
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        if "config" in checkpoint:
            self.config = checkpoint["config"]


def train_vanilla_dqn(
    total_timesteps: int = 500_000,
    seed: int = 0,
    log_interval: int = 25_000,
    eval_interval: int = 25_000,
    eval_episodes: int = 20,
    eval_seed_base: int = 8000,
) -> Tuple[DQNAgent, List[float]]:
    """**Eq. 4.1:** Train vanilla DQN with mid-training eval + best-checkpoint restore."""
    set_seed(seed)
    env = CartPoleDefenderEnv()
    agent = DQNAgent()
    episode_returns: List[float] = []
    ep_return = 0.0
    episode_count = 0
    best_snap: Optional[dict] = None
    best_eval = -float("inf")
    best_step = 0

    obs, _ = env.reset(seed=seed)

    for step in range(1, total_timesteps + 1):
        cfg = agent.config
        epsilon = cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(
            -step / cfg.epsilon_decay_steps
        )
        action = agent.select_action(obs, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        ep_return += reward
        agent.total_steps = step

        if step % cfg.train_freq == 0:
            for _ in range(cfg.gradient_steps):
                agent.train_step()

        if done:
            episode_returns.append(ep_return)
            episode_count += 1
            if episode_count % cfg.target_update_episodes == 0:
                agent.update_target()
            obs, _ = env.reset()
            ep_return = 0.0

        if step % log_interval == 0 and episode_returns:
            recent = episode_returns[-100:]
            mean100 = float(np.mean(recent))
            print(
                f"  step {step:>7}/{total_timesteps} | "
                f"ε={epsilon:.3f} | episodes={len(episode_returns)} | "
                f"mean_return(last {len(recent)})={mean100:.1f}",
                flush=True,
            )

        if step % eval_interval == 0:
            greedy_mean = quick_eval_mean(
                agent, eval_episodes, eval_seed_base + step, "vanilla"
            )
            print(f"  [eval @ {step}] greedy_clean={greedy_mean:.1f}", flush=True)
            if greedy_mean >= best_eval:
                best_eval = greedy_mean
                best_step = step
                best_snap = agent.snapshot_weights()

    env.close()
    if best_snap is not None:
        agent.restore_weights(best_snap)
        print(
            f"  restored best vanilla checkpoint from step {best_step} "
            f"(greedy_clean={best_eval:.1f})",
            flush=True,
        )
    else:
        agent.update_target()
    return agent, episode_returns


def continue_vanilla_dqn(
    agent: DQNAgent,
    extra_timesteps: int,
    seed: int,
    log_interval: int = 25_000,
    eval_interval: int = 25_000,
    eval_episodes: int = 20,
    eval_seed_base: int = 8500,
) -> Tuple[DQNAgent, List[float]]:
    """Continue vanilla DQN training with mid-eval + best-checkpoint restore."""
    set_seed(seed)
    env = CartPoleDefenderEnv()
    episode_returns: List[float] = []
    ep_return = 0.0
    episode_count = 0
    obs, _ = env.reset(seed=seed)
    start_step = agent.total_steps
    best_snap: Optional[dict] = agent.snapshot_weights()
    best_eval = quick_eval_mean(agent, eval_episodes, eval_seed_base, "vanilla")
    best_step = start_step
    print(
        f"  [continue start] greedy_clean={best_eval:.1f} (baseline checkpoint)",
        flush=True,
    )

    for step in range(1, extra_timesteps + 1):
        global_step = start_step + step
        cfg = agent.config
        epsilon = cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(
            -global_step / cfg.epsilon_decay_steps
        )
        action = agent.select_action(obs, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        ep_return += reward
        agent.total_steps = global_step

        if step % cfg.train_freq == 0:
            for _ in range(cfg.gradient_steps):
                agent.train_step()

        if done:
            episode_returns.append(ep_return)
            episode_count += 1
            if episode_count % cfg.target_update_episodes == 0:
                agent.update_target()
            obs, _ = env.reset()
            ep_return = 0.0

        if step % log_interval == 0 and episode_returns:
            recent = episode_returns[-100:]
            print(
                f"  continue step {step:>7}/{extra_timesteps} | mean_return(last {len(recent)})={np.mean(recent):.1f}",
                flush=True,
            )

        if step % eval_interval == 0:
            greedy_mean = quick_eval_mean(
                agent, eval_episodes, eval_seed_base + global_step, "vanilla"
            )
            print(
                f"  [continue eval @ {global_step}] greedy_clean={greedy_mean:.1f}",
                flush=True,
            )
            if greedy_mean >= best_eval:
                best_eval = greedy_mean
                best_step = global_step
                best_snap = agent.snapshot_weights()

    env.close()
    if best_snap is not None:
        agent.restore_weights(best_snap)
        print(
            f"  restored best continue checkpoint from step {best_step} "
            f"(greedy_clean={best_eval:.1f})",
            flush=True,
        )
    else:
        agent.update_target()
    return agent, episode_returns


def evaluate_vanilla_dqn(
    agent: DQNAgent,
    n_episodes: int = 100,
    seed: int = 0,
    measure_time: bool = False,
) -> Tuple[List[float], Optional[float]]:
    """**Eq. 4.2:** Evaluate greedy policy a = argmax_a Q(s,a) on true observations."""
    set_seed(seed)
    env = CartPoleDefenderEnv()
    returns: List[float] = []
    inference_times: List[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0

        while not done:
            if measure_time:
                t0 = time.perf_counter()
            action = agent.greedy_action(obs)
            if measure_time:
                inference_times.append(time.perf_counter() - t0)

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated

        returns.append(ep_return)

    env.close()
    avg_time = float(np.mean(inference_times) * 1000) if measure_time else None
    return returns, avg_time


def run_step1_vanilla(seed: int = 0, timesteps: int = 500_000):
    """Step 1: Train and evaluate vanilla DQN; CartPole solved if mean ≥ 475 over 100 eps."""
    print("=" * 70)
    print("CARTPOLE DQN — STEP 1: VANILLA BASELINE")
    print("=" * 70)
    print(f"Training timesteps: {timesteps:,}")
    print(f"Seed: {seed}")
    print(f"Target: mean return ≥ 475 over 100 evaluation episodes")
    print("=" * 70)

    agent, train_returns = train_vanilla_dqn(total_timesteps=timesteps, seed=seed)

    if train_returns:
        train_tail = train_returns[-100:]
        print(f"\nTraining summary (last {len(train_tail)} episodes): "
              f"mean={np.mean(train_tail):.1f}, std={np.std(train_tail):.1f}")

    eval_returns, infer_ms = evaluate_vanilla_dqn(agent, n_episodes=100, seed=seed + 1000, measure_time=True)
    mean_eval = float(np.mean(eval_returns))
    std_eval = float(np.std(eval_returns))

    print("\n" + "=" * 70)
    print("EVALUATION (100 episodes, greedy, no attack)")
    print("=" * 70)
    print(f"  Mean return: {mean_eval:.2f} ± {std_eval:.2f}")
    print(f"  Min / Max:   {min(eval_returns):.0f} / {max(eval_returns):.0f}")
    if infer_ms is not None:
        print(f"  Inference:   {infer_ms:.4f} ms per action")
    solved = mean_eval >= 475.0
    print(f"  CartPole solved (≥475): {'YES' if solved else 'NO'}")
    print("=" * 70)

    agent.save("cartpole_vanilla_dqn.pt")
    print("Saved checkpoint: cartpole_vanilla_dqn.pt")
    return agent, eval_returns, solved


def perturb_from_attacker(
    true_obs: np.ndarray,
    attacker,
    epsilon: float = DEFAULT_ATTACK_EPS,
) -> np.ndarray:
    """Apply attacker δ to true observation; fallback to raw numpy action."""
    delta, _ = attacker.predict(true_obs, deterministic=True)
    action = int(np.asarray(delta).reshape(-1)[0])
    delta = delta_from_discrete_action(action, epsilon)
    return apply_perturbation(true_obs, delta, normalized=True)


def _defender_action_under_attack(
    agent: DQNAgent,
    true_obs: np.ndarray,
    perturbed: np.ndarray,
    mode: str,
    epsilon: float,
    kappa: int,
    belief_provider: Optional[BeliefProvider] = None,
    under_attack: bool = True,
) -> int:
    """Select defender action for clean, learned-attacker, or shared-random eval."""
    if mode == "pessimistic" and under_attack:
        if belief_provider is not None:
            belief_provider.observe(perturbed)
        return agent.maximin_action(
            perturbed, epsilon, kappa, belief_provider=belief_provider
        )
    if mode == "pessimistic" and not under_attack:
        s_tilde = agent.worst_case_perturbation(true_obs, epsilon)
        return agent.maximin_action(
            s_tilde, epsilon, kappa, belief_provider=belief_provider
        )
    if mode in ("offpolicy", "vanilla"):
        return agent.greedy_action(perturbed if under_attack else true_obs)
    return agent.greedy_action(true_obs)


def evaluate_agent(
    agent: DQNAgent,
    n_episodes: int = 100,
    seed: int = 0,
    mode: str = "vanilla",
    attacker=None,
    epsilon: float = DEFAULT_ATTACK_EPS,
    measure_time: bool = False,
    belief_provider: Optional[BeliefProvider] = None,
) -> Tuple[List[float], Optional[float]]:
    """Evaluate greedy / maximin policy (clean or vs learned DQN attacker)."""
    set_seed(seed)
    env = CartPoleDefenderEnv()
    returns: List[float] = []
    inference_times: List[float] = []
    kappa = agent.config.belief_kappa
    under_attack = attacker is not None

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        if belief_provider is not None:
            belief_provider.reset_episode()
        done = False
        ep_return = 0.0

        while not done:
            true_obs = obs
            if under_attack:
                perturbed = perturb_from_attacker(true_obs, attacker, epsilon)
            else:
                perturbed = true_obs

            if measure_time:
                t0 = time.perf_counter()

            action = _defender_action_under_attack(
                agent, true_obs, perturbed, mode, epsilon, kappa, belief_provider, under_attack
            )

            if measure_time:
                inference_times.append(time.perf_counter() - t0)

            obs, reward, terminated, truncated, _ = env.step(action)
            if belief_provider is not None:
                belief_provider.observe(obs)
            ep_return += reward
            done = terminated or truncated

        returns.append(ep_return)
    env.close()
    avg_ms = float(np.mean(inference_times) * 1000) if measure_time and inference_times else None
    return returns, avg_ms


def evaluate_shared_random_attack(
    agent: DQNAgent,
    n_episodes: int = 100,
    seed: int = 0,
    mode: str = "vanilla",
    epsilon: float = DEFAULT_ATTACK_EPS,
    measure_time: bool = False,
    belief_provider: Optional[BeliefProvider] = None,
) -> Tuple[List[float], Optional[float]]:
    """Shared threat model: identical δ stream per (seed, episode, step) across defenders.

    Each defender runs its own rollout; at step t the same δ_t ~ Uniform(B_ε) is applied
    to that defender's true state (fair fixed attack distribution).
    """
    set_seed(seed)
    env = CartPoleDefenderEnv()
    returns: List[float] = []
    inference_times: List[float] = []
    kappa = agent.config.belief_kappa

    for ep in range(n_episodes):
        rng = np.random.RandomState(seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        if belief_provider is not None:
            belief_provider.reset_episode()
        done = False
        ep_return = 0.0

        while not done:
            true_obs = obs
            delta = rng.uniform(-epsilon, epsilon, size=4).astype(np.float32)
            perturbed = apply_perturbation(true_obs, delta, normalized=True)

            if measure_time:
                t0 = time.perf_counter()

            action = _defender_action_under_attack(
                agent, true_obs, perturbed, mode, epsilon, kappa, belief_provider, under_attack=True
            )

            if measure_time:
                inference_times.append(time.perf_counter() - t0)

            obs, reward, terminated, truncated, _ = env.step(action)
            if belief_provider is not None:
                belief_provider.observe(obs)
            ep_return += reward
            done = terminated or truncated

        returns.append(ep_return)
    env.close()
    avg_ms = float(np.mean(inference_times) * 1000) if measure_time and inference_times else None
    return returns, avg_ms


def _store_attack_metrics(
    results: Dict,
    prefix: str,
    returns: List[float],
    infer_ms: Optional[float],
    clean_reward: float,
) -> None:
    """Write mean/std/inference and attack-to-clean ratio into results dict."""
    mean_r = float(np.mean(returns))
    results[f"{prefix}_avg_reward"] = mean_r
    results[f"{prefix}_std"] = float(np.std(returns))
    if infer_ms is not None:
        results[f"{prefix}_inference_time_ms"] = infer_ms
    if clean_reward > 0:
        results[f"{prefix}_attack_clean_ratio"] = mean_r / clean_reward
    else:
        results[f"{prefix}_attack_clean_ratio"] = np.nan


def _training_loop(
    agent: DQNAgent,
    total_timesteps: int,
    seed: int,
    mode: str,
    log_interval: int = 25_000,
    belief_provider: Optional[BeliefProvider] = None,
) -> List[float]:
    """Shared training loop for pessimistic and off-policy fine-tuning."""
    set_seed(seed)
    env = CartPoleDefenderEnv()
    episode_returns: List[float] = []
    ep_return = 0.0
    episode_count = 0
    obs, _ = env.reset(seed=seed)
    if belief_provider is not None:
        belief_provider.reset_episode()
    cfg = agent.config

    for step in range(1, total_timesteps + 1):
        epsilon = cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(
            -step / cfg.epsilon_decay_steps
        )
        s_tilde = agent.worst_case_perturbation(obs, cfg.attack_epsilon)
        explore = np.random.random() < epsilon
        if mode in ("pessimistic", "offpolicy"):
            action = (
                np.random.randint(agent.n_actions)
                if explore
                else agent.maximin_action(
                    s_tilde, cfg.attack_epsilon, cfg.belief_kappa, belief_provider=belief_provider
                )
            )
        else:
            action = agent.select_action(obs, epsilon)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if belief_provider is not None:
            next_belief = belief_provider.observe(next_obs)
        else:
            next_belief = sample_ball(next_obs, cfg.attack_epsilon, cfg.belief_kappa)

        if mode in ("pessimistic", "offpolicy"):
            agent.belief_replay_buffer.push(obs, action, reward, next_obs, done, next_belief)
        if mode == "pessimistic":
            agent.replay_buffer.push(obs, action, reward, next_obs, done)

        obs = next_obs
        ep_return += reward
        agent.total_steps = step

        if step % cfg.train_freq == 0:
            for _ in range(cfg.gradient_steps):
                if mode == "pessimistic":
                    agent.train_step_pessimistic()
                elif mode == "offpolicy":
                    agent.train_step_offpolicy_greedy()
                else:
                    agent.train_step()

        if done:
            episode_returns.append(ep_return)
            episode_count += 1
            if episode_count % cfg.target_update_episodes == 0:
                agent.update_target()
            obs, _ = env.reset()
            if belief_provider is not None:
                belief_provider.reset_episode()
            ep_return = 0.0

        if step % log_interval == 0 and episode_returns:
            recent = episode_returns[-100:]
            print(f"  step {step:>7}/{total_timesteps} | mean_return(last {len(recent)})={np.mean(recent):.1f}", flush=True)

    env.close()
    agent.update_target()
    return episode_returns


def _training_loop_pessimistic(
    agent: DQNAgent,
    total_timesteps: int,
    seed: int,
    log_interval: int = 25_000,
    eval_interval: int = 25_000,
    eval_episodes: int = 20,
    eval_seed_base: int = 9000,
    early_stop: bool = True,
    solve_threshold: float = 475.0,
    collapse_floor: float = 400.0,
    belief_provider: Optional[BeliefProvider] = None,
) -> Tuple[List[float], Dict]:
    """Pessimistic fine-tune with mid-training eval, best-checkpoint restore, and early stop."""
    set_seed(seed)
    env = CartPoleDefenderEnv()
    episode_returns: List[float] = []
    ep_return = 0.0
    episode_count = 0
    obs, _ = env.reset(seed=seed)
    if belief_provider is not None:
        belief_provider.reset_episode()
    cfg = agent.config

    meta: Dict = {
        "stopped_step": total_timesteps,
        "best_greedy_eval": 0.0,
        "best_maximin_eval": 0.0,
        "best_step": 0,
        "early_stopped": False,
    }
    best_snap: Optional[dict] = None

    for step in range(1, total_timesteps + 1):
        epsilon = cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(
            -step / cfg.epsilon_decay_steps
        )
        s_tilde = agent.worst_case_perturbation(obs, cfg.attack_epsilon)
        explore = np.random.random() < epsilon
        action = (
            np.random.randint(agent.n_actions)
            if explore
            else agent.maximin_action(
                s_tilde, cfg.attack_epsilon, cfg.belief_kappa, belief_provider=belief_provider
            )
        )

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if belief_provider is not None:
            next_belief = belief_provider.observe(next_obs)
        else:
            next_belief = sample_ball(next_obs, cfg.attack_epsilon, cfg.belief_kappa)
        agent.belief_replay_buffer.push(obs, action, reward, next_obs, done, next_belief)
        agent.replay_buffer.push(obs, action, reward, next_obs, done)

        obs = next_obs
        ep_return += reward
        agent.total_steps = step

        if step % cfg.train_freq == 0:
            agent.train_step_pessimistic()

        if done:
            episode_returns.append(ep_return)
            episode_count += 1
            if episode_count % cfg.target_update_episodes == 0:
                agent.update_target()
            obs, _ = env.reset()
            if belief_provider is not None:
                belief_provider.reset_episode()
            ep_return = 0.0

        if step % log_interval == 0 and episode_returns:
            recent = episode_returns[-100:]
            print(
                f"  step {step:>7}/{total_timesteps} | mean_return(last {len(recent)})={np.mean(recent):.1f}",
                flush=True,
            )

        if step % eval_interval == 0:
            greedy_mean = quick_eval_mean(agent, eval_episodes, eval_seed_base + step, "vanilla")
            maximin_mean = quick_eval_mean(agent, eval_episodes, eval_seed_base + step + 1, "pessimistic")
            print(
                f"  [eval @ {step}] greedy_clean={greedy_mean:.1f}  maximin_clean={maximin_mean:.1f}",
                flush=True,
            )
            if greedy_mean >= meta["best_greedy_eval"]:
                meta["best_greedy_eval"] = greedy_mean
                meta["best_maximin_eval"] = maximin_mean
                meta["best_step"] = step
                best_snap = agent.snapshot_weights()

            if early_stop and best_snap is not None and meta["best_greedy_eval"] >= solve_threshold:
                collapsed = greedy_mean < collapse_floor
                sharp_drop = greedy_mean < meta["best_greedy_eval"] - 75.0
                if collapsed or sharp_drop:
                    print(
                        f"  [early stop @ {step}] greedy={greedy_mean:.1f} "
                        f"(best={meta['best_greedy_eval']:.1f} @ step {meta['best_step']})",
                        flush=True,
                    )
                    meta["stopped_step"] = step
                    meta["early_stopped"] = True
                    break

    env.close()
    if best_snap is not None:
        agent.restore_weights(best_snap)
        print(
            f"  restored best checkpoint from step {meta['best_step']} "
            f"(greedy={meta['best_greedy_eval']:.1f}, maximin={meta['best_maximin_eval']:.1f})",
            flush=True,
        )
    else:
        agent.update_target()
    return episode_returns, meta


def train_pessimistic_dqn(
    agent: DQNAgent,
    total_timesteps: int = 100_000,
    seed: int = 0,
    eval_interval: int = 25_000,
    eval_episodes: int = 20,
    early_stop: bool = True,
    belief_provider: Optional[BeliefProvider] = None,
) -> Tuple[List[float], Dict]:
    """**Algorithm 1 + 2:** Fine-tune DQN with pessimistic targets (Eq. 2.6)."""
    return _training_loop_pessimistic(
        agent,
        total_timesteps,
        seed,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        early_stop=early_stop,
        belief_provider=belief_provider,
    )


def train_offpolicy_greedy_dqn(
    agent: DQNAgent,
    total_timesteps: int = 100_000,
    seed: int = 0,
    belief_provider: Optional[BeliefProvider] = None,
) -> List[float]:
    """**Eq. 5.1–5.2:** Pessimistic behavior, greedy bootstrap target."""
    return _training_loop(agent, total_timesteps, seed, mode="offpolicy", belief_provider=belief_provider)


def train_cartpole_attacker(
    defender: DQNAgent,
    agent_type: str = "vanilla",
    total_timesteps: int = 100_000,
    save_path: Optional[str] = None,
    belief_provider: Optional[BeliefProvider] = None,
) -> SB3_DQN:
    """**Eq. 6.4:** Train DQN attacker ω against fixed defender."""
    save_path = save_path or f"cartpole_attacker_{agent_type}"
    env = CartPoleAttackEnv(
        defender,
        agent_type=agent_type,
        epsilon=defender.config.attack_epsilon,
        belief_provider=belief_provider,
    )
    model = SB3_DQN(
        "MlpPolicy",
        env,
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
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    env.close()
    print(f"Saved attacker: {save_path}.zip")
    return model


def load_or_train_vanilla(path: str = "cartpole_vanilla_dqn.pt", timesteps: int = 500_000, seed: int = 0) -> DQNAgent:
    if os.path.exists(path):
        agent = DQNAgent()
        agent.load(path)
        print(f"Loaded vanilla checkpoint: {path}")
        return agent
    agent, _ = train_vanilla_dqn(total_timesteps=timesteps, seed=seed)
    agent.save(path)
    return agent


def run_step2_attack_eval(seed: int = 0, load_attacker: bool = True):
    """**Step 2:** Evaluate vanilla DQN clean vs under PGD-style worst-case perturbation."""
    print("=" * 70)
    print("CARTPOLE DQN — STEP 2: VANILLA UNDER ATTACK")
    print("=" * 70)
    agent = load_or_train_vanilla(seed=seed)

    clean, clean_ms = evaluate_agent(agent, n_episodes=50, seed=seed + 1000, mode="vanilla", measure_time=True)
    print(f"Clean:      mean={np.mean(clean):.1f} ± {np.std(clean):.1f}  inference={clean_ms:.4f} ms")

    # Worst-case perturbation attack (no trained attacker yet)
    set_seed(seed + 2000)
    env = CartPoleDefenderEnv()
    wc_returns = []
    for ep in range(50):
        obs, _ = env.reset(seed=seed + 2000 + ep)
        done = False
        ep_ret = 0.0
        while not done:
            s_tilde = agent.worst_case_perturbation(obs, agent.config.attack_epsilon)
            action = agent.greedy_action(s_tilde)
            obs, r, term, trunc, _ = env.step(action)
            ep_ret += r
            done = term or trunc
        wc_returns.append(ep_ret)
    env.close()
    print(f"Worst-case: mean={np.mean(wc_returns):.1f} ± {np.std(wc_returns):.1f}")

    attacker_path = "cartpole_attacker_vanilla"
    if load_attacker and os.path.exists(attacker_path + ".zip"):
        attacker = SB3_DQN.load(attacker_path)
        attacked, att_ms = evaluate_agent(
            agent, n_episodes=50, seed=seed + 3000, mode="vanilla",
            attacker=attacker, measure_time=True,
        )
        print(f"DQN attack: mean={np.mean(attacked):.1f} ± {np.std(attacked):.1f}  inference={att_ms:.4f} ms")
    else:
        print("No trained attacker found — run step 5 to train attackers.")
    print("=" * 70)


def run_step3_pessimistic(seed: int = 0, finetune_steps: int = 100_000):
    """**Step 3:** Fine-tune pessimistic DQN from vanilla baseline."""
    print("=" * 70)
    print("CARTPOLE DQN — STEP 3: PESSIMISTIC DQN")
    print("=" * 70)
    vanilla = load_or_train_vanilla(seed=seed)
    agent = DQNAgent(config=vanilla.config)
    agent.clone_weights_from(vanilla)
    agent.prepare_robust_finetune()

    _, pes_meta = train_pessimistic_dqn(agent, total_timesteps=finetune_steps, seed=seed + 1000)
    agent.save("cartpole_pessimistic_dqn.pt")

    clean, _ = evaluate_agent(agent, n_episodes=50, seed=seed + 2000, mode="vanilla")
    clean_mm, _ = evaluate_agent(agent, n_episodes=50, seed=seed + 2100, mode="pessimistic")
    print(f"Clean (greedy):     mean={np.mean(clean):.1f} ± {np.std(clean):.1f}")
    print(f"Clean (maximin):    mean={np.mean(clean_mm):.1f} ± {np.std(clean_mm):.1f}")
    print(f"Finetune:           stopped={pes_meta['stopped_step']}  best_greedy={pes_meta['best_greedy_eval']:.1f}  "
          f"early_stop={pes_meta['early_stopped']}")

    attacker = None
    if os.path.exists("cartpole_attacker_pessimistic.zip"):
        attacker = SB3_DQN.load("cartpole_attacker_pessimistic")
    elif os.path.exists("cartpole_attacker_vanilla.zip"):
        attacker = SB3_DQN.load("cartpole_attacker_vanilla")

    if attacker:
        attacked, att_ms = evaluate_agent(
            agent, n_episodes=50, seed=seed + 3000, mode="pessimistic",
            attacker=attacker, measure_time=True,
        )
        print(f"Under attack:       mean={np.mean(attacked):.1f} ± {np.std(attacked):.1f}  inference={att_ms:.4f} ms")
    print("Saved: cartpole_pessimistic_dqn.pt")
    print("=" * 70)
    return agent


def run_step4_offpolicy(seed: int = 0, finetune_steps: int = 40_000):
    """**Step 4:** Fine-tune off-policy greedy DQN from vanilla baseline."""
    print("=" * 70)
    print("CARTPOLE DQN — STEP 4: OFF-POLICY GREEDY DQN")
    print("=" * 70)
    vanilla = load_or_train_vanilla(seed=seed)
    agent = DQNAgent(config=vanilla.config)
    agent.clone_weights_from(vanilla)
    agent.prepare_offpolicy_finetune()

    train_offpolicy_greedy_dqn(agent, total_timesteps=finetune_steps, seed=seed + 1000)
    agent.save("cartpole_offpolicy_dqn.pt")

    clean, _ = evaluate_agent(agent, n_episodes=50, seed=seed + 2000, mode="vanilla")
    print(f"Clean (greedy):     mean={np.mean(clean):.1f} ± {np.std(clean):.1f}")

    attacker = None
    if os.path.exists("cartpole_attacker_offpolicy.zip"):
        attacker = SB3_DQN.load("cartpole_attacker_offpolicy")
    elif os.path.exists("cartpole_attacker_vanilla.zip"):
        attacker = SB3_DQN.load("cartpole_attacker_vanilla")

    if attacker:
        attacked, att_ms = evaluate_agent(
            agent, n_episodes=50, seed=seed + 3000, mode="offpolicy",
            attacker=attacker, measure_time=True,
        )
        print(f"Under attack:       mean={np.mean(attacked):.1f} ± {np.std(attacked):.1f}  inference={att_ms:.4f} ms")
    print("Saved: cartpole_offpolicy_dqn.pt")
    print("=" * 70)
    return agent


def run_step5_train_attackers(seed: int = 99999, timesteps: int = 100_000):
    """**Step 5:** Train DQN attackers against vanilla, pessimistic, and off-policy defenders."""
    print("=" * 70)
    print("CARTPOLE DQN — STEP 5: TRAIN ATTACKERS")
    print("=" * 70)
    set_seed(seed)

    vanilla = load_or_train_vanilla(seed=seed)
    train_cartpole_attacker(vanilla, "vanilla", timesteps, "cartpole_attacker_vanilla")

    if os.path.exists("cartpole_pessimistic_dqn.pt"):
        pes = DQNAgent()
        pes.load("cartpole_pessimistic_dqn.pt")
    else:
        print("Training pessimistic defender for attacker...")
        pes = run_step3_pessimistic(seed=seed, finetune_steps=timesteps)
    train_cartpole_attacker(pes, "pessimistic", timesteps, "cartpole_attacker_pessimistic")

    if os.path.exists("cartpole_offpolicy_dqn.pt"):
        off = DQNAgent()
        off.load("cartpole_offpolicy_dqn.pt")
    else:
        print("Training off-policy defender for attacker...")
        off = run_step4_offpolicy(seed=seed, finetune_steps=timesteps)
    train_cartpole_attacker(off, "offpolicy", timesteps, "cartpole_attacker_offpolicy")
    print("=" * 70)


def run_full_comparison(seed: int = 0, vanilla_steps: int = 500_000, finetune_steps: int = 100_000, attack_steps: int = 100_000):
    """Run complete three-algorithm comparison (mirrors GridWorld experiment pipeline)."""
    print("=" * 70)
    print("CARTPOLE DQN — FULL COMPARISON")
    print("=" * 70)

    vanilla = load_or_train_vanilla(timesteps=vanilla_steps, seed=seed)
    v_clean, v_clean_ms = evaluate_agent(vanilla, 50, seed + 1000, "vanilla", measure_time=True)

    run_step5_train_attackers(seed=seed + 5000, timesteps=attack_steps)
    att_v = SB3_DQN.load("cartpole_attacker_vanilla")
    att_p = SB3_DQN.load("cartpole_attacker_pessimistic")
    att_o = SB3_DQN.load("cartpole_attacker_offpolicy")

    v_att, v_att_ms = evaluate_agent(vanilla, 50, seed + 2000, "vanilla", att_v, measure_time=True)

    pes = DQNAgent(config=vanilla.config)
    pes.clone_weights_from(vanilla)
    pes.prepare_robust_finetune()
    _, pes_meta = train_pessimistic_dqn(pes, finetune_steps, seed + 3000)
    p_clean, p_clean_ms = evaluate_agent(pes, 50, seed + 4000, "vanilla", measure_time=True)
    p_att, p_att_ms = evaluate_agent(pes, 50, seed + 5000, "pessimistic", att_p, measure_time=True)

    off = DQNAgent(config=vanilla.config)
    off.clone_weights_from(vanilla)
    off.prepare_offpolicy_finetune()
    train_offpolicy_greedy_dqn(off, finetune_steps, seed + 6000)
    o_clean, o_clean_ms = evaluate_agent(off, 50, seed + 7000, "vanilla", measure_time=True)
    o_att, o_att_ms = evaluate_agent(off, 50, seed + 8000, "offpolicy", att_o, measure_time=True)

    print("\n--- RESULTS ---")
    print(f"Vanilla     clean={np.mean(v_clean):.1f}  vs_attack={np.mean(v_att):.1f}  infer={v_clean_ms:.3f}/{v_att_ms:.3f} ms")
    print(f"Pessimistic clean={np.mean(p_clean):.1f}  vs_attack={np.mean(p_att):.1f}  infer={p_clean_ms:.3f}/{p_att_ms:.3f} ms")
    print(f"Off-Policy  clean={np.mean(o_clean):.1f}  vs_attack={np.mean(o_att):.1f}  infer={o_clean_ms:.3f}/{o_att_ms:.3f} ms")
    print("=" * 70)

    vanilla.save("cartpole_vanilla_dqn.pt")
    pes.save("cartpole_pessimistic_dqn.pt")
    off.save("cartpole_offpolicy_dqn.pt")


def seed_paths(seed: int, base_dir: str = "checkpoints") -> Dict[str, str]:
    """Per-seed artifact paths so independent runs never share weights."""
    d = os.path.join(base_dir, f"seed_{seed}")
    return {
        "dir": d,
        "vanilla": os.path.join(d, "vanilla.pt"),
        "pessimistic": os.path.join(d, "pessimistic.pt"),
        "offpolicy": os.path.join(d, "offpolicy.pt"),
        "attacker_vanilla": os.path.join(d, "attacker_vanilla"),
        "attacker_pessimistic": os.path.join(d, "attacker_pessimistic"),
        "attacker_offpolicy": os.path.join(d, "attacker_offpolicy"),
        "belief_rnn": os.path.join(d, "belief_rnn.pt"),
    }


def ensure_seed_dir(seed: int, base_dir: str = "checkpoints") -> Dict[str, str]:
    paths = seed_paths(seed, base_dir)
    os.makedirs(paths["dir"], exist_ok=True)
    return paths


def evaluate_pgd_worst_case(
    agent: DQNAgent,
    n_episodes: int,
    seed: int,
    mode: str = "vanilla",
    belief_provider: Optional[BeliefProvider] = None,
) -> List[float]:
    """PGD-style worst-case perturbation eval (supplement to DQN attacker)."""
    set_seed(seed)
    env = CartPoleDefenderEnv()
    returns: List[float] = []
    eps = agent.config.attack_epsilon
    kappa = agent.config.belief_kappa

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        if belief_provider is not None:
            belief_provider.reset_episode()
        done = False
        ep_return = 0.0
        while not done:
            s_tilde = agent.worst_case_perturbation(obs, eps)
            if mode == "pessimistic":
                action = agent.maximin_action(
                    s_tilde, eps, kappa, belief_provider=belief_provider
                )
            else:
                action = agent.greedy_action(s_tilde)
            obs, reward, terminated, truncated, _ = env.step(action)
            if belief_provider is not None:
                belief_provider.observe(obs)
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)
    env.close()
    return returns


def train_vanilla_with_solve_gate(
    total_timesteps: int,
    seed: int,
    eval_episodes: int = 100,
    min_reward: float = 475.0,
    max_retries: int = 2,
    extra_steps: int = 200_000,
) -> Tuple[DQNAgent, float, bool]:
    """Train vanilla DQN with best-checkpoint restore; extend if final eval is below min_reward.

    ``solved`` is always computed from the final returned evaluation mean (not a stale mid-gate).
    """
    steps = total_timesteps
    vanilla, _ = train_vanilla_dqn(total_timesteps=steps, seed=seed)
    mean_eval = 0.0
    for attempt in range(max_retries + 1):
        eval_returns, _ = evaluate_vanilla_dqn(vanilla, eval_episodes, seed + 1000)
        mean_eval = float(np.mean(eval_returns))
        solved = mean_eval >= min_reward
        print(
            f"  vanilla eval (attempt {attempt + 1}): mean={mean_eval:.1f} "
            f"(threshold={min_reward:.0f}) -> {'PASS' if solved else 'RETRY'}",
            flush=True,
        )
        if solved or attempt >= max_retries:
            return vanilla, mean_eval, solved
        vanilla, _ = continue_vanilla_dqn(vanilla, extra_steps, seed + 10_000 * (attempt + 1))
    return vanilla, mean_eval, mean_eval >= min_reward


def _make_eval_belief_provider(
    belief_mode: str,
    belief_model: Optional[BeliefRNN],
    epsilon: float,
    kappa: int,
) -> Optional[BeliefProvider]:
    if belief_mode == "uniform":
        return UniformBallBelief(epsilon, kappa)
    if belief_model is not None:
        return make_belief_provider("learned", epsilon, kappa, belief_model)
    return None


CSV_FIELDNAMES = [
    "experiment_number",
    "timestamp",
    "seed",
    "belief_mode",
    "vanilla_solved",
    "belief_pretrain_loss",
    # Clean performance
    "vanilla_test_avg_reward",
    "vanilla_test_std",
    "vanilla_inference_time_ms",
    "pessimistic_test_avg_reward",
    "pessimistic_test_std",
    "pessimistic_test_maximin_avg_reward",
    "pessimistic_test_maximin_std",
    "pessimistic_inference_time_ms",
    "offpolicy_greedy_test_avg_reward",
    "offpolicy_greedy_test_std",
    "offpolicy_greedy_inference_time_ms",
    # Matched attacker (primary fair eval: each vs own attacker)
    "vanilla_vs_matched_attacker_avg_reward",
    "vanilla_vs_matched_attacker_std",
    "vanilla_vs_matched_attacker_inference_time_ms",
    "vanilla_vs_matched_attacker_attack_clean_ratio",
    "pessimistic_vs_matched_attacker_avg_reward",
    "pessimistic_vs_matched_attacker_std",
    "pessimistic_vs_matched_attacker_inference_time_ms",
    "pessimistic_vs_matched_attacker_attack_clean_ratio",
    "offpolicy_vs_matched_attacker_avg_reward",
    "offpolicy_vs_matched_attacker_std",
    "offpolicy_vs_matched_attacker_inference_time_ms",
    "offpolicy_vs_matched_attacker_attack_clean_ratio",
    # Shared random-δ attack (identical threat distribution)
    "vanilla_vs_shared_random_avg_reward",
    "vanilla_vs_shared_random_std",
    "vanilla_vs_shared_random_inference_time_ms",
    "vanilla_vs_shared_random_attack_clean_ratio",
    "pessimistic_vs_shared_random_avg_reward",
    "pessimistic_vs_shared_random_std",
    "pessimistic_vs_shared_random_inference_time_ms",
    "pessimistic_vs_shared_random_attack_clean_ratio",
    "offpolicy_vs_shared_random_avg_reward",
    "offpolicy_vs_shared_random_std",
    "offpolicy_vs_shared_random_inference_time_ms",
    "offpolicy_vs_shared_random_attack_clean_ratio",
    # Transfer eval (GridWorld parity: greedy vanilla vs att_p)
    "vanilla_vs_transfer_attacker_avg_reward",
    "vanilla_vs_transfer_attacker_std",
    "vanilla_vs_transfer_attacker_inference_time_ms",
    "vanilla_vs_transfer_attacker_attack_clean_ratio",
    # PGD worst-case (defender-specific)
    "vanilla_pgd_reward",
    "vanilla_pgd_std",
    "vanilla_pgd_attack_clean_ratio",
    "pessimistic_pgd_reward",
    "pessimistic_pgd_std",
    "pessimistic_pgd_attack_clean_ratio",
    "offpolicy_pgd_reward",
    "offpolicy_pgd_std",
    "offpolicy_pgd_attack_clean_ratio",
    # Training metadata
    "pessimistic_train_avg_reward",
    "pessimistic_train_std",
    "pessimistic_finetune_stopped_step",
    "pessimistic_best_greedy_eval",
    "pessimistic_early_stopped",
    "offpolicy_greedy_train_avg_reward",
    "offpolicy_greedy_train_std",
    # Legacy aliases (backward compat for old CSV readers)
    "vanilla_vs_attacker_reward",
    "vanilla_vs_attacker_std",
    "vanilla_attack_inference_time_ms",
    "pessimistic_vs_attacker_reward",
    "pessimistic_vs_attacker_std",
    "pessimistic_attack_inference_time_ms",
    "offpolicy_greedy_vs_attacker_reward",
    "offpolicy_greedy_vs_attacker_std",
    "offpolicy_attack_inference_time_ms",
    # Summary diffs
    "performance_diff_offpolicy_vs_pessimistic",
    "performance_diff_offpolicy_vs_vanilla",
    "performance_diff_pessimistic_vs_vanilla",
    "speedup_offpolicy_vs_pessimistic",
    "speedup_offpolicy_vs_vanilla",
    "error",
]


def log_experiment_to_csv(
    experiment_number: int,
    results_dict: Dict,
    filename: str = "cartpole_experiments.csv",
) -> None:
    """Log one experiment row (GridWorld-compatible columns + PGD extras)."""
    file_exists = os.path.isfile(filename)
    row = {k: results_dict.get(k, np.nan) for k in CSV_FIELDNAMES}
    row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row["experiment_number"] = experiment_number

    with open(filename, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"  ✓ Experiment {experiment_number + 1} logged to {filename}")


def save_experiment_checkpoint(
    experiment_number: int,
    checkpoint_file: str = "cartpole_experiment_checkpoint.pkl",
    csv_filename: str = "cartpole_experiments.csv",
) -> None:
    with open(checkpoint_file, "wb") as f:
        pickle.dump(
            {
                "last_completed_experiment": experiment_number,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "csv_filename": csv_filename,
            },
            f,
        )


def load_experiment_checkpoint(
    checkpoint_file: str = "cartpole_experiment_checkpoint.pkl",
) -> Tuple[int, str]:
    if not os.path.exists(checkpoint_file):
        return -1, "cartpole_experiments.csv"
    with open(checkpoint_file, "rb") as f:
        data = pickle.load(f)
    return int(data["last_completed_experiment"]), data.get("csv_filename", "cartpole_experiments.csv")


def clear_experiment_checkpoint(checkpoint_file: str = "cartpole_experiment_checkpoint.pkl") -> None:
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)


def run_single_experiment_with_seed(
    experiment_num: int,
    seed: int,
    vanilla_steps: int = 500_000,
    finetune_pessimistic: int = 100_000,
    finetune_offpolicy: int = 40_000,
    attack_steps: int = 80_000,
    eval_episodes: int = 100,
    checkpoint_base_dir: str = "checkpoints",
    belief_mode: str = "learned",
    belief_pretrain_steps: int = 30_000,
    vanilla_min_reward: float = 475.0,
) -> Dict:
    """Full fair pipeline for one seed (mirrors GridWorld run_single_experiment_with_seed)."""
    paths = ensure_seed_dir(seed, checkpoint_base_dir)
    results: Dict = {
        "experiment_number": experiment_num,
        "seed": seed,
        "belief_mode": belief_mode,
        "error": "",
    }
    cfg_eps = DEFAULT_ATTACK_EPS
    cfg_kappa = DEFAULT_BELIEF_KAPPA

    print(f"\n[Exp {experiment_num + 1}] Phase 1: Training vanilla DQN (seed={seed}, solve gate ≥{vanilla_min_reward})...")
    vanilla, vanilla_eval, vanilla_solved = train_vanilla_with_solve_gate(
        vanilla_steps, seed, eval_episodes, vanilla_min_reward
    )
    vanilla.save(paths["vanilla"])
    results["vanilla_solved"] = int(vanilla_solved)

    print(f"[Exp {experiment_num + 1}] Phase 1.5: Evaluating vanilla...")
    v_clean, v_clean_ms = evaluate_agent(
        vanilla, eval_episodes, seed + 1000, "vanilla", measure_time=True
    )
    results["vanilla_test_avg_reward"] = float(np.mean(v_clean))
    results["vanilla_test_std"] = float(np.std(v_clean))
    results["vanilla_inference_time_ms"] = v_clean_ms
    # Keep solved flag consistent with the reported clean eval (not a stale gate pass).
    results["vanilla_solved"] = int(results["vanilla_test_avg_reward"] >= vanilla_min_reward)

    belief_model: Optional[BeliefRNN] = None
    if belief_mode == "learned":
        print(f"[Exp {experiment_num + 1}] Phase 1.75: Pretraining Belief RNN ({belief_pretrain_steps} steps)...")
        belief_model = BeliefRNN(kappa=cfg_kappa).to(torch.device(vanilla.config.device))
        pretrain_loss = pretrain_belief_rnn_on_agent(
            vanilla,
            belief_model,
            CartPoleDefenderEnv,
            timesteps=belief_pretrain_steps,
            seed=seed + 1500,
            epsilon=cfg_eps,
            set_seed_fn=set_seed,
        )
        save_belief_rnn(belief_model, paths["belief_rnn"])
        results["belief_pretrain_loss"] = float(pretrain_loss)
    else:
        results["belief_pretrain_loss"] = np.nan

    train_belief = _make_eval_belief_provider(belief_mode, belief_model, cfg_eps, cfg_kappa)
    eval_belief = _make_eval_belief_provider(belief_mode, belief_model, cfg_eps, cfg_kappa)

    print(f"[Exp {experiment_num + 1}] Phase 2: Training pessimistic DQN (belief={belief_mode})...")
    pes = DQNAgent(config=vanilla.config)
    pes.clone_weights_from(vanilla)
    pes.prepare_robust_finetune(freeze_backbone=True)
    pes_train, pes_meta = train_pessimistic_dqn(
        pes, finetune_pessimistic, seed + 2000, belief_provider=train_belief
    )
    pes.save(paths["pessimistic"])
    results["pessimistic_train_avg_reward"] = float(np.mean(pes_train[-100:])) if pes_train else np.nan
    results["pessimistic_train_std"] = float(np.std(pes_train[-100:])) if pes_train else np.nan
    results["pessimistic_finetune_stopped_step"] = pes_meta["stopped_step"]
    results["pessimistic_best_greedy_eval"] = pes_meta["best_greedy_eval"]
    results["pessimistic_early_stopped"] = int(pes_meta["early_stopped"])

    p_clean, p_clean_ms = evaluate_agent(
        pes, eval_episodes, seed + 3000, "vanilla", measure_time=True
    )
    p_clean_mm, _ = evaluate_agent(
        pes, eval_episodes, seed + 3100, "pessimistic", measure_time=True, belief_provider=eval_belief
    )
    results["pessimistic_test_avg_reward"] = float(np.mean(p_clean))
    results["pessimistic_test_std"] = float(np.std(p_clean))
    results["pessimistic_test_maximin_avg_reward"] = float(np.mean(p_clean_mm))
    results["pessimistic_test_maximin_std"] = float(np.std(p_clean_mm))
    results["pessimistic_inference_time_ms"] = p_clean_ms

    print(f"[Exp {experiment_num + 1}] Phase 3: Training off-policy greedy DQN...")
    off = DQNAgent(config=vanilla.config)
    off.clone_weights_from(vanilla)
    off.prepare_offpolicy_finetune()
    train_belief_off = _make_eval_belief_provider(belief_mode, belief_model, cfg_eps, cfg_kappa)
    off_train = train_offpolicy_greedy_dqn(
        off, finetune_offpolicy, seed + 4000, belief_provider=train_belief_off
    )
    off.save(paths["offpolicy"])
    results["offpolicy_greedy_train_avg_reward"] = float(np.mean(off_train[-100:])) if off_train else np.nan
    results["offpolicy_greedy_train_std"] = float(np.std(off_train[-100:])) if off_train else np.nan

    o_clean, o_clean_ms = evaluate_agent(
        off, eval_episodes, seed + 5000, "vanilla", measure_time=True
    )
    results["offpolicy_greedy_test_avg_reward"] = float(np.mean(o_clean))
    results["offpolicy_greedy_test_std"] = float(np.std(o_clean))
    results["offpolicy_greedy_inference_time_ms"] = o_clean_ms

    print(f"[Exp {experiment_num + 1}] Phase 4: Training per-seed DQN attackers...")
    train_cartpole_attacker(vanilla, "vanilla", attack_steps, paths["attacker_vanilla"])
    train_cartpole_attacker(
        pes, "pessimistic", attack_steps, paths["attacker_pessimistic"], belief_provider=eval_belief
    )
    train_cartpole_attacker(off, "offpolicy", attack_steps, paths["attacker_offpolicy"])

    return _run_fair_eval_suite(
        vanilla, pes, off, results, seed, eval_episodes, eval_belief, paths,
        experiment_num, train_attackers_if_missing=False, attack_steps=attack_steps,
        refresh_clean=False,
    )


def _run_fair_eval_suite(
    vanilla: DQNAgent,
    pes: DQNAgent,
    off: DQNAgent,
    results: Dict,
    seed: int,
    eval_episodes: int,
    eval_belief: Optional[BeliefProvider],
    paths: Dict[str, str],
    experiment_num: int,
    train_attackers_if_missing: bool = True,
    attack_steps: int = 80_000,
    refresh_clean: bool = True,
) -> Dict:
    """Run clean + matched + shared + transfer + PGD eval (Phases 5–6)."""
    exp_label = f"[Exp {experiment_num + 1}]"

    if refresh_clean:
        print(f"{exp_label} Evaluating clean performance...")
        v_clean, v_clean_ms = evaluate_agent(vanilla, eval_episodes, seed + 1000, "vanilla", measure_time=True)
        results["vanilla_test_avg_reward"] = float(np.mean(v_clean))
        results["vanilla_test_std"] = float(np.std(v_clean))
        results["vanilla_inference_time_ms"] = v_clean_ms

        p_clean, p_clean_ms = evaluate_agent(pes, eval_episodes, seed + 3000, "vanilla", measure_time=True)
        p_clean_mm, _ = evaluate_agent(
            pes, eval_episodes, seed + 3100, "pessimistic", measure_time=True, belief_provider=eval_belief
        )
        results["pessimistic_test_avg_reward"] = float(np.mean(p_clean))
        results["pessimistic_test_std"] = float(np.std(p_clean))
        results["pessimistic_test_maximin_avg_reward"] = float(np.mean(p_clean_mm))
        results["pessimistic_test_maximin_std"] = float(np.std(p_clean_mm))
        results["pessimistic_inference_time_ms"] = p_clean_ms

        o_clean, o_clean_ms = evaluate_agent(off, eval_episodes, seed + 5000, "vanilla", measure_time=True)
        results["offpolicy_greedy_test_avg_reward"] = float(np.mean(o_clean))
        results["offpolicy_greedy_test_std"] = float(np.std(o_clean))
        results["offpolicy_greedy_inference_time_ms"] = o_clean_ms

    att_v_path = paths["attacker_vanilla"] + ".zip"
    att_p_path = paths["attacker_pessimistic"] + ".zip"
    att_o_path = paths["attacker_offpolicy"] + ".zip"
    if train_attackers_if_missing and (
        not os.path.exists(att_v_path) or not os.path.exists(att_p_path) or not os.path.exists(att_o_path)
    ):
        print(f"{exp_label} Training attackers (missing checkpoints)...")
        train_cartpole_attacker(vanilla, "vanilla", attack_steps, paths["attacker_vanilla"])
        train_cartpole_attacker(
            pes, "pessimistic", attack_steps, paths["attacker_pessimistic"], belief_provider=eval_belief
        )
        train_cartpole_attacker(off, "offpolicy", attack_steps, paths["attacker_offpolicy"])

    att_v = SB3_DQN.load(paths["attacker_vanilla"])
    att_p = SB3_DQN.load(paths["attacker_pessimistic"])
    att_o = SB3_DQN.load(paths["attacker_offpolicy"])

    v_clean_mean = results["vanilla_test_avg_reward"]
    p_clean_mean = results["pessimistic_test_avg_reward"]
    o_clean_mean = results["offpolicy_greedy_test_avg_reward"]

    print(f"{exp_label} Phase 5a: Matched attacker eval...")
    v_matched, v_matched_ms = evaluate_agent(
        vanilla, eval_episodes, seed + 6000, "vanilla", att_v, measure_time=True
    )
    _store_attack_metrics(results, "vanilla_vs_matched_attacker", v_matched, v_matched_ms, v_clean_mean)

    p_matched, p_matched_ms = evaluate_agent(
        pes, eval_episodes, seed + 7000, "pessimistic", att_p, measure_time=True, belief_provider=eval_belief
    )
    _store_attack_metrics(results, "pessimistic_vs_matched_attacker", p_matched, p_matched_ms, p_clean_mean)

    o_matched, o_matched_ms = evaluate_agent(
        off, eval_episodes, seed + 8000, "offpolicy", att_o, measure_time=True
    )
    _store_attack_metrics(results, "offpolicy_vs_matched_attacker", o_matched, o_matched_ms, o_clean_mean)

    print(f"{exp_label} Phase 5b: Shared random-δ attack...")
    shared_seed = seed + 8500
    v_shared, v_shared_ms = evaluate_shared_random_attack(
        vanilla, eval_episodes, shared_seed, "vanilla", measure_time=True
    )
    _store_attack_metrics(results, "vanilla_vs_shared_random", v_shared, v_shared_ms, v_clean_mean)

    p_shared, p_shared_ms = evaluate_shared_random_attack(
        pes, eval_episodes, shared_seed, "pessimistic", measure_time=True, belief_provider=eval_belief
    )
    _store_attack_metrics(results, "pessimistic_vs_shared_random", p_shared, p_shared_ms, p_clean_mean)

    o_shared, o_shared_ms = evaluate_shared_random_attack(
        off, eval_episodes, shared_seed, "offpolicy", measure_time=True
    )
    _store_attack_metrics(results, "offpolicy_vs_shared_random", o_shared, o_shared_ms, o_clean_mean)

    print(f"{exp_label} Phase 5c: Transfer eval (vanilla vs att_p)...")
    v_transfer, v_transfer_ms = evaluate_agent(
        vanilla, eval_episodes, seed + 6500, "vanilla", att_p, measure_time=True
    )
    _store_attack_metrics(results, "vanilla_vs_transfer_attacker", v_transfer, v_transfer_ms, v_clean_mean)

    results["vanilla_vs_attacker_reward"] = results["vanilla_vs_transfer_attacker_avg_reward"]
    results["vanilla_vs_attacker_std"] = results["vanilla_vs_transfer_attacker_std"]
    results["vanilla_attack_inference_time_ms"] = results["vanilla_vs_transfer_attacker_inference_time_ms"]
    results["pessimistic_vs_attacker_reward"] = results["pessimistic_vs_matched_attacker_avg_reward"]
    results["pessimistic_vs_attacker_std"] = results["pessimistic_vs_matched_attacker_std"]
    results["pessimistic_attack_inference_time_ms"] = results["pessimistic_vs_matched_attacker_inference_time_ms"]
    results["offpolicy_greedy_vs_attacker_reward"] = results["offpolicy_vs_matched_attacker_avg_reward"]
    results["offpolicy_greedy_vs_attacker_std"] = results["offpolicy_vs_matched_attacker_std"]
    results["offpolicy_attack_inference_time_ms"] = results["offpolicy_vs_matched_attacker_inference_time_ms"]

    print(f"{exp_label} Phase 6: PGD worst-case...")
    v_pgd = evaluate_pgd_worst_case(vanilla, eval_episodes, seed + 9000, "vanilla")
    results["vanilla_pgd_reward"] = float(np.mean(v_pgd))
    results["vanilla_pgd_std"] = float(np.std(v_pgd))
    results["vanilla_pgd_attack_clean_ratio"] = results["vanilla_pgd_reward"] / v_clean_mean if v_clean_mean > 0 else np.nan

    p_pgd = evaluate_pgd_worst_case(
        pes, eval_episodes, seed + 9100, "pessimistic", belief_provider=eval_belief
    )
    results["pessimistic_pgd_reward"] = float(np.mean(p_pgd))
    results["pessimistic_pgd_std"] = float(np.std(p_pgd))
    results["pessimistic_pgd_attack_clean_ratio"] = results["pessimistic_pgd_reward"] / p_clean_mean if p_clean_mean > 0 else np.nan

    o_pgd = evaluate_pgd_worst_case(off, eval_episodes, seed + 9200, "vanilla")
    results["offpolicy_pgd_reward"] = float(np.mean(o_pgd))
    results["offpolicy_pgd_std"] = float(np.std(o_pgd))
    results["offpolicy_pgd_attack_clean_ratio"] = results["offpolicy_pgd_reward"] / o_clean_mean if o_clean_mean > 0 else np.nan

    results["performance_diff_offpolicy_vs_pessimistic"] = o_clean_mean - p_clean_mean
    results["performance_diff_offpolicy_vs_vanilla"] = o_clean_mean - v_clean_mean
    results["performance_diff_pessimistic_vs_vanilla"] = p_clean_mean - v_clean_mean
    results["speedup_offpolicy_vs_pessimistic"] = (
        p_matched_ms / o_matched_ms if p_matched_ms and o_matched_ms else np.nan
    )
    results["speedup_offpolicy_vs_vanilla"] = (
        v_matched_ms / o_matched_ms if v_matched_ms and o_matched_ms else np.nan
    )

    print(f"\n{exp_label} Summary:")
    print(f"  Vanilla     clean={v_clean_mean:.1f}  matched={results['vanilla_vs_matched_attacker_avg_reward']:.1f}  "
          f"shared={results['vanilla_vs_shared_random_avg_reward']:.1f}  pgd={results['vanilla_pgd_reward']:.1f}")
    print(f"  Pessimistic clean={p_clean_mean:.1f}  matched={results['pessimistic_vs_matched_attacker_avg_reward']:.1f}  "
          f"shared={results['pessimistic_vs_shared_random_avg_reward']:.1f}  pgd={results['pessimistic_pgd_reward']:.1f}")
    print(f"  Off-Policy  clean={o_clean_mean:.1f}  matched={results['offpolicy_vs_matched_attacker_avg_reward']:.1f}  "
          f"shared={results['offpolicy_vs_shared_random_avg_reward']:.1f}  pgd={results['offpolicy_pgd_reward']:.1f}")
    return results


def run_eval_only_from_seed(
    experiment_num: int,
    seed: int,
    eval_episodes: int = 100,
    checkpoint_base_dir: str = "checkpoints",
    belief_mode: str = "learned",
    attack_steps: int = 80_000,
) -> Dict:
    """Re-run fair eval from saved checkpoints (no retraining)."""
    paths = ensure_seed_dir(seed, checkpoint_base_dir)
    results: Dict = {
        "experiment_number": experiment_num,
        "seed": seed,
        "belief_mode": belief_mode,
        "error": "",
    }
    for key in ("vanilla", "pessimistic", "offpolicy"):
        if not os.path.exists(paths[key]):
            raise FileNotFoundError(f"Missing checkpoint: {paths[key]}")

    vanilla = DQNAgent()
    vanilla.load(paths["vanilla"])
    pes = DQNAgent(config=vanilla.config)
    pes.load(paths["pessimistic"])
    off = DQNAgent(config=vanilla.config)
    off.load(paths["offpolicy"])

    belief_model: Optional[BeliefRNN] = None
    if belief_mode == "learned" and os.path.exists(paths["belief_rnn"]):
        belief_model = load_belief_rnn(paths["belief_rnn"])
        results["belief_pretrain_loss"] = np.nan
    eval_belief = _make_eval_belief_provider(belief_mode, belief_model, DEFAULT_ATTACK_EPS, DEFAULT_BELIEF_KAPPA)

    results["vanilla_solved"] = int(results.get("vanilla_solved", 1))
    return _run_fair_eval_suite(
        vanilla, pes, off, results, seed, eval_episodes, eval_belief, paths,
        experiment_num, train_attackers_if_missing=True, attack_steps=attack_steps,
    )


def run_experiment_loop(
    num_experiments: int = 10,
    base_seed: int = 0,
    vanilla_steps: int = 500_000,
    finetune_pessimistic: int = 100_000,
    finetune_offpolicy: int = 40_000,
    attack_steps: int = 80_000,
    eval_episodes: int = 100,
    csv_filename: str = "cartpole_experiments.csv",
    checkpoint_file: str = "cartpole_experiment_checkpoint.pkl",
    resume: bool = False,
    belief_mode: str = "learned",
    belief_pretrain_steps: int = 30_000,
    vanilla_min_reward: float = 475.0,
    eval_only: bool = False,
) -> None:
    """Run N independent seeds with CSV logging and checkpoint resume."""
    start_exp = 0
    if resume:
        last_done, ckpt_csv = load_experiment_checkpoint(checkpoint_file)
        if last_done >= 0:
            start_exp = last_done + 1
            csv_filename = ckpt_csv
            print(f"Resuming from experiment {start_exp + 1} (csv={csv_filename})")

    mode_label = "EVAL-ONLY (fair re-eval from checkpoints)" if eval_only else "FAIR MULTI-SEED EXPERIMENTS"
    print("=" * 70)
    print(f"CARTPOLE DQN — {mode_label} ({num_experiments} total)")
    print("=" * 70)
    print(f"  Seeds:           {base_seed} .. {base_seed + num_experiments - 1}")
    print(f"  CSV:             {csv_filename}")
    print(f"  Belief:          {belief_mode}")
    print(f"  Vanilla gate:    ≥{vanilla_min_reward}")
    print(f"  Eval episodes:   {eval_episodes}")
    print(f"  Attack eval:     matched + shared-random + transfer + PGD")
    print("=" * 70)

    last_completed = start_exp - 1
    for exp_num in range(start_exp, num_experiments):
        seed = base_seed + exp_num
        print(f"\n{'=' * 70}")
        print(f"EXPERIMENT {exp_num + 1}/{num_experiments}  (seed={seed})")
        print(f"{'=' * 70}")

        try:
            if eval_only:
                results = run_eval_only_from_seed(
                    experiment_num=exp_num,
                    seed=seed,
                    eval_episodes=eval_episodes,
                    belief_mode=belief_mode,
                    attack_steps=attack_steps,
                )
            else:
                results = run_single_experiment_with_seed(
                    experiment_num=exp_num,
                    seed=seed,
                    vanilla_steps=vanilla_steps,
                    finetune_pessimistic=finetune_pessimistic,
                    finetune_offpolicy=finetune_offpolicy,
                    attack_steps=attack_steps,
                    eval_episodes=eval_episodes,
                    belief_mode=belief_mode,
                    belief_pretrain_steps=belief_pretrain_steps,
                    vanilla_min_reward=vanilla_min_reward,
                )
            log_experiment_to_csv(exp_num, results, csv_filename)
            save_experiment_checkpoint(exp_num, checkpoint_file, csv_filename)
            last_completed = exp_num
            print(f"✓ Experiment {exp_num + 1} complete")
        except KeyboardInterrupt:
            print(f"\nPaused after experiment {exp_num} (progress saved through {last_completed + 1})")
            raise
        except Exception as exc:
            print(f"\n✗ Experiment {exp_num + 1} failed: {exc}")
            err_row = {
                "seed": seed,
                "error": str(exc),
            }
            log_experiment_to_csv(exp_num, err_row, csv_filename)
            save_experiment_checkpoint(exp_num, checkpoint_file, csv_filename)
            last_completed = exp_num

    if last_completed == num_experiments - 1:
        clear_experiment_checkpoint(checkpoint_file)
        print(f"\n{'=' * 70}")
        print(f"ALL {num_experiments} EXPERIMENTS COMPLETE")
        print(f"Results: {csv_filename}")
        print(f"{'=' * 70}")
        analyze_cartpole_csv(csv_filename)
        compare_narratives(csv_filename)


GRIDWORLD_REFERENCE = {
    "vanilla_clean": 7.35,
    "vanilla_attack": -682.14,
    "pessimistic_clean": 7.35,
    "pessimistic_attack": -21.76,
    "offpolicy_clean": 7.37,
    "offpolicy_attack": -551.88,
}


def compare_narratives(csv_filename: str = "cartpole_experiments.csv") -> None:
    """Compare CartPole results against GridWorld narrative using fair eval columns."""
    import pandas as pd

    if not os.path.exists(csv_filename):
        print(f"No CSV for narrative comparison: {csv_filename}")
        return

    df = pd.read_csv(csv_filename)
    ok = df[df["error"].fillna("") == ""] if "error" in df.columns else df
    if len(ok) == 0:
        print("No successful CartPole runs to compare.")
        return

    def _col(primary: str, fallback: str) -> str:
        return primary if primary in ok.columns else fallback

    def _mean(col: str) -> float:
        if col not in ok.columns:
            return float("nan")
        return float(ok[col].dropna().mean())

    gw = GRIDWORLD_REFERENCE

    print("\n" + "=" * 70)
    print("NARRATIVE COMPARISON — FAIR EVAL (CartPole vs GridWorld reference)")
    print("=" * 70)

    eval_suites = [
        (
            "MATCHED ATTACKER (primary)",
            {
                "vanilla": _col("vanilla_vs_matched_attacker_avg_reward", "vanilla_vs_attacker_reward"),
                "pessimistic": _col("pessimistic_vs_matched_attacker_avg_reward", "pessimistic_vs_attacker_reward"),
                "offpolicy": _col("offpolicy_vs_matched_attacker_avg_reward", "offpolicy_greedy_vs_attacker_reward"),
            },
        ),
        (
            "SHARED RANDOM-δ",
            {
                "vanilla": "vanilla_vs_shared_random_avg_reward",
                "pessimistic": "pessimistic_vs_shared_random_avg_reward",
                "offpolicy": "offpolicy_vs_shared_random_avg_reward",
            },
        ),
        (
            "PGD WORST-CASE",
            {
                "vanilla": "vanilla_pgd_reward",
                "pessimistic": "pessimistic_pgd_reward",
                "offpolicy": "offpolicy_pgd_reward",
            },
        ),
        (
            "TRANSFER (van greedy vs att_p — supplementary)",
            {
                "vanilla": _col("vanilla_vs_transfer_attacker_avg_reward", "vanilla_vs_attacker_reward"),
                "pessimistic": None,
                "offpolicy": None,
            },
        ),
    ]

    clean = {
        "vanilla": _mean("vanilla_test_avg_reward"),
        "pessimistic": _mean("pessimistic_test_avg_reward"),
        "offpolicy": _mean("offpolicy_greedy_test_avg_reward"),
    }

    print(f"\nClean means: vanilla={clean['vanilla']:.1f}  pes={clean['pessimistic']:.1f}  off={clean['offpolicy']:.1f}")
    print(f"GridWorld attack ref: pes={gw['pessimistic_attack']:.1f}  off={gw['offpolicy_attack']:.1f}  van={gw['vanilla_attack']:.1f}")

    for title, cols in eval_suites:
        print(f"\n--- {title} ---")
        if cols["vanilla"] is None or cols["vanilla"] not in ok.columns:
            print("  (columns not available — re-run with updated pipeline)")
            continue
        v = _mean(cols["vanilla"])
        print(f"  Vanilla:     {v:.1f}")
        if cols["pessimistic"] and cols["pessimistic"] in ok.columns:
            p = _mean(cols["pessimistic"])
            o = _mean(cols["offpolicy"]) if cols["offpolicy"] in ok.columns else float("nan")
            print(f"  Pessimistic: {p:.1f}")
            print(f"  Off-policy:  {o:.1f}")
            n1 = p > o > v
            print(f"  pes > off > van: {'YES' if n1 else 'NO'}  ({p:.1f} > {o:.1f} > {v:.1f})")
        else:
            print(f"  (vanilla-only transfer column)")

    n2 = clean["pessimistic"] >= 400 and clean["offpolicy"] >= 400
    speed_col = "speedup_offpolicy_vs_pessimistic"
    n3 = _mean(speed_col) > 1.0 if speed_col in ok.columns else False

    matched_p = _mean(_col("pessimistic_vs_matched_attacker_avg_reward", "pessimistic_vs_attacker_reward"))
    matched_o = _mean(_col("offpolicy_vs_matched_attacker_avg_reward", "offpolicy_greedy_vs_attacker_reward"))
    matched_v = _mean(_col("vanilla_vs_matched_attacker_avg_reward", "vanilla_vs_attacker_reward"))
    n1_primary = matched_p > matched_o > matched_v

    print("\n" + "-" * 70)
    print("Pre-registered narrative checks (CartPole, matched attacker primary):")
    print(f"  1. Attack robustness pes > off > van   {'YES' if n1_primary else 'NO'}  "
          f"({matched_p:.1f} > {matched_o:.1f} > {matched_v:.1f})")
    print(f"  2. Clean performance preserved (≥400)  {'YES' if n2 else 'NO'}")
    print(f"  3. Off-policy faster than pessimistic    {'YES' if n3 else 'NO'}  "
          f"(mean speedup={_mean(speed_col):.1f}x)" if speed_col in ok.columns else "")
    gw_n1 = gw["pessimistic_attack"] > gw["offpolicy_attack"] > gw["vanilla_attack"]
    print(f"\nGridWorld (500 seeds) pes > off > van:   {'YES' if gw_n1 else 'NO'}")
    print(f"CartPole supports same narrative?         {'YES' if n1_primary and n2 else 'PARTIAL' if n1_primary else 'NO'}")
    print("=" * 70)


def analyze_cartpole_csv(csv_filename: str = "cartpole_experiments.csv") -> None:
    """Print GridWorld-style summary from CartPole experiment CSV."""
    import pandas as pd

    if not os.path.exists(csv_filename):
        print(f"No CSV found: {csv_filename}")
        return

    df = pd.read_csv(csv_filename)
    if "error" in df.columns:
        ok = df[df["error"].fillna("") == ""]
    else:
        ok = df
    n_ok = len(ok)
    n_err = len(df) - n_ok

    print("=" * 70)
    print(f"ANALYSIS OF {len(df)} EXPERIMENTS ({n_ok} succeeded, {n_err} failed)")
    print("=" * 70)

    def _report_block(title: str, clean_col: str, matched_col: str, shared_col: str, pgd_col: str, ratio_col: str) -> None:
        print(f"\n--- {title} ---")
        if clean_col in ok.columns:
            s = ok[clean_col].dropna()
            if len(s):
                print(f"  Clean:           {s.mean():.2f} ± {s.std():.2f}")
        if matched_col in ok.columns:
            s = ok[matched_col].dropna()
            if len(s):
                print(f"  Matched attacker:{s.mean():.2f} ± {s.std():.2f}", end="")
                rcol = matched_col.replace("_avg_reward", "_attack_clean_ratio")
                if rcol in ok.columns:
                    r = ok[rcol].dropna()
                    if len(r):
                        print(f"  (attack/clean={r.mean():.3f})", end="")
                print()
        if shared_col in ok.columns:
            s = ok[shared_col].dropna()
            if len(s):
                print(f"  Shared random-δ: {s.mean():.2f} ± {s.std():.2f}")
        if pgd_col in ok.columns:
            s = ok[pgd_col].dropna()
            if len(s):
                print(f"  PGD worst-case:  {s.mean():.2f} ± {s.std():.2f}")

    _report_block(
        "Vanilla DQN",
        "vanilla_test_avg_reward",
        "vanilla_vs_matched_attacker_avg_reward",
        "vanilla_vs_shared_random_avg_reward",
        "vanilla_pgd_reward",
        "vanilla_vs_matched_attacker_attack_clean_ratio",
    )
    _report_block(
        "Pessimistic DQN",
        "pessimistic_test_avg_reward",
        "pessimistic_vs_matched_attacker_avg_reward",
        "pessimistic_vs_shared_random_avg_reward",
        "pessimistic_pgd_reward",
        "pessimistic_vs_matched_attacker_attack_clean_ratio",
    )
    if "pessimistic_test_maximin_avg_reward" in ok.columns:
        s = ok["pessimistic_test_maximin_avg_reward"].dropna()
        if len(s):
            print(f"  Pessimistic maximin clean: {s.mean():.2f} ± {s.std():.2f}")
    _report_block(
        "Off-Policy Greedy",
        "offpolicy_greedy_test_avg_reward",
        "offpolicy_vs_matched_attacker_avg_reward",
        "offpolicy_vs_shared_random_avg_reward",
        "offpolicy_pgd_reward",
        "offpolicy_vs_matched_attacker_attack_clean_ratio",
    )

    if "vanilla_vs_transfer_attacker_avg_reward" in ok.columns:
        s = ok["vanilla_vs_transfer_attacker_avg_reward"].dropna()
        if len(s):
            print(f"\n--- Transfer (vanilla vs att_p, supplementary) ---")
            print(f"  Mean reward: {s.mean():.2f} ± {s.std():.2f}")

    if "pessimistic_early_stopped" in ok.columns:
        print(f"\nPessimistic early stopped: {ok['pessimistic_early_stopped'].sum():.0f}/{len(ok)} runs")

    for col in (
        "performance_diff_offpolicy_vs_pessimistic",
        "performance_diff_offpolicy_vs_vanilla",
        "performance_diff_pessimistic_vs_vanilla",
    ):
        if col in df.columns:
            s = ok[col].dropna()
            if len(s):
                print(f"\n{col}: {s.mean():.3f} ± {s.std():.3f}")

    print("=" * 70)
    compare_narratives(csv_filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CartPole DQN experiments")
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="1=vanilla, 2=attack eval, 3=pessimistic, 4=off-policy, 5=attackers, "
        "6=single comparison, 7=fair multi-seed loop, 8=analyze CSV, 9=narrative compare, "
        "10=eval-only from checkpoints (fair metrics)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=0, help="First seed for step 7 (seed_i = base_seed + i)")
    parser.add_argument("--num-experiments", type=int, default=10, help="Number of seeds for step 7")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Vanilla training timesteps")
    parser.add_argument("--finetune", type=int, default=100_000, help="Fine-tune timesteps (steps 3/4/6)")
    parser.add_argument("--finetune-pessimistic", type=int, default=100_000, help="Pessimistic fine-tune (step 7)")
    parser.add_argument("--finetune-offpolicy", type=int, default=40_000, help="Off-policy fine-tune (step 7)")
    parser.add_argument("--attack-steps", type=int, default=80_000, help="Attacker training timesteps")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Evaluation episodes per condition")
    parser.add_argument("--csv", type=str, default="cartpole_experiments.csv", help="CSV output for step 7/8")
    parser.add_argument("--resume", action="store_true", help="Resume step 7 from checkpoint")
    parser.add_argument(
        "--belief",
        type=str,
        default="learned",
        choices=["learned", "uniform"],
        help="Belief model for pessimistic/off-policy (step 7)",
    )
    parser.add_argument("--belief-pretrain", type=int, default=30_000, help="Belief RNN pretrain steps")
    parser.add_argument("--vanilla-min-reward", type=float, default=475.0, help="Vanilla solve gate")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Step 7/10: skip training, re-run fair eval from checkpoints/seed_*/",
    )
    args = parser.parse_args()

    if args.step == 1:
        run_step1_vanilla(seed=args.seed, timesteps=args.timesteps)
    elif args.step == 2:
        run_step2_attack_eval(seed=args.seed)
    elif args.step == 3:
        run_step3_pessimistic(seed=args.seed, finetune_steps=args.finetune)
    elif args.step == 4:
        run_step4_offpolicy(seed=args.seed, finetune_steps=args.finetune)
    elif args.step == 5:
        run_step5_train_attackers(seed=args.seed, timesteps=args.attack_steps)
    elif args.step == 6:
        run_full_comparison(
            seed=args.seed,
            vanilla_steps=args.timesteps,
            finetune_steps=args.finetune,
            attack_steps=args.attack_steps,
        )
    elif args.step == 7:
        run_experiment_loop(
            num_experiments=args.num_experiments,
            base_seed=args.base_seed,
            vanilla_steps=args.timesteps,
            finetune_pessimistic=args.finetune_pessimistic,
            finetune_offpolicy=args.finetune_offpolicy,
            attack_steps=args.attack_steps,
            eval_episodes=args.eval_episodes,
            csv_filename=args.csv,
            resume=args.resume,
            belief_mode=args.belief,
            belief_pretrain_steps=args.belief_pretrain,
            vanilla_min_reward=args.vanilla_min_reward,
            eval_only=args.eval_only,
        )
    elif args.step == 10:
        run_experiment_loop(
            num_experiments=args.num_experiments,
            base_seed=args.base_seed,
            eval_episodes=args.eval_episodes,
            attack_steps=args.attack_steps,
            csv_filename=args.csv,
            resume=args.resume,
            belief_mode=args.belief,
            eval_only=True,
        )
    elif args.step == 8:
        analyze_cartpole_csv(args.csv)
    elif args.step == 9:
        compare_narratives(args.csv)

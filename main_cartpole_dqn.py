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

    def __init__(self, defender_agent: "DQNAgent", agent_type: str = "vanilla", epsilon: float = DEFAULT_ATTACK_EPS):
        self.defender = defender_agent
        self.agent_type = agent_type
        self.epsilon = epsilon
        self.kappa = DEFAULT_BELIEF_KAPPA
        self.inner = CartPoleDefenderEnv()
        self.action_space = gym.spaces.Discrete(N_DISCRETE_ATTACK_ACTIONS)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.perturbed_state: Optional[np.ndarray] = None
        self.belief: List[np.ndarray] = []

    def _defender_action(self, perturbed_norm: np.ndarray) -> int:
        if self.agent_type == "pessimistic":
            return self.defender.maximin_action(perturbed_norm, self.epsilon, self.kappa)
        return self.defender.greedy_action(perturbed_norm)

    def reset(self, *, seed=None, options=None):
        obs, info = self.inner.reset(seed=seed, options=options)
        self.perturbed_state = obs.copy()
        self.belief = sample_ball(obs, self.epsilon, self.kappa)
        return obs, info

    def step(self, action):
        """**Eq. 1.4–1.6:** Attacker picks δ; defender acts on s̃; R̃ = −R(s,a)."""
        true_norm = normalize_obs(self.inner.get_true_state())
        delta = delta_from_discrete_action(int(action), self.epsilon)
        self.perturbed_state = apply_perturbation(true_norm, delta, normalized=True)
        defender_action = self._defender_action(self.perturbed_state)
        obs, reward, terminated, truncated, info = self.inner.step(defender_action)
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

    def maximin_action(self, center_state: np.ndarray, epsilon: float, kappa: int) -> int:
        """**Eq. 3.4 / Algorithm 5, Line 7:** a = argmax_a min_{m∈M∩B_ε(s̃)} Q(m,a)"""
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
) -> Tuple[DQNAgent, List[float]]:
    """**Eq. 4.1:** Train vanilla DQN on CartPole-v1 (Phase 1 baseline)."""
    set_seed(seed)
    env = CartPoleDefenderEnv()
    agent = DQNAgent()
    episode_returns: List[float] = []
    ep_return = 0.0
    episode_count = 0

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

    env.close()
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


def evaluate_agent(
    agent: DQNAgent,
    n_episodes: int = 100,
    seed: int = 0,
    mode: str = "vanilla",
    attacker=None,
    epsilon: float = DEFAULT_ATTACK_EPS,
    measure_time: bool = False,
) -> Tuple[List[float], Optional[float]]:
    """**Eq. 4.2 / Eq. 3.4:** Evaluate greedy, pessimistic, or off-policy inference."""
    set_seed(seed)
    env = CartPoleDefenderEnv()
    returns: List[float] = []
    inference_times: List[float] = []
    kappa = agent.config.belief_kappa

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0

        while not done:
            true_obs = obs
            if attacker is not None:
                perturbed = perturb_from_attacker(true_obs, attacker, epsilon)
            else:
                perturbed = true_obs

            if measure_time:
                t0 = time.perf_counter()

            if mode == "pessimistic" and attacker is not None:
                action = agent.maximin_action(perturbed, epsilon, kappa)
            elif mode == "offpolicy" or mode == "vanilla":
                action = agent.greedy_action(perturbed if attacker is not None else true_obs)
            else:
                action = agent.greedy_action(true_obs)

            if measure_time:
                inference_times.append(time.perf_counter() - t0)

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated

        returns.append(ep_return)
    env.close()
    avg_ms = float(np.mean(inference_times) * 1000) if measure_time and inference_times else None
    return returns, avg_ms


def _training_loop(
    agent: DQNAgent,
    total_timesteps: int,
    seed: int,
    mode: str,
    log_interval: int = 25_000,
) -> List[float]:
    """Shared training loop for pessimistic and off-policy fine-tuning."""
    set_seed(seed)
    env = CartPoleDefenderEnv()
    episode_returns: List[float] = []
    ep_return = 0.0
    episode_count = 0
    obs, _ = env.reset(seed=seed)
    cfg = agent.config

    for step in range(1, total_timesteps + 1):
        epsilon = cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(
            -step / cfg.epsilon_decay_steps
        )
        s_tilde = agent.worst_case_perturbation(obs, cfg.attack_epsilon)
        explore = np.random.random() < epsilon
        if mode in ("pessimistic", "offpolicy"):
            action = np.random.randint(agent.n_actions) if explore else agent.maximin_action(
                s_tilde, cfg.attack_epsilon, cfg.belief_kappa
            )
        else:
            action = agent.select_action(obs, epsilon)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
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
) -> Tuple[List[float], Dict]:
    """Pessimistic fine-tune with mid-training eval, best-checkpoint restore, and early stop."""
    set_seed(seed)
    env = CartPoleDefenderEnv()
    episode_returns: List[float] = []
    ep_return = 0.0
    episode_count = 0
    obs, _ = env.reset(seed=seed)
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
            else agent.maximin_action(s_tilde, cfg.attack_epsilon, cfg.belief_kappa)
        )

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
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
) -> Tuple[List[float], Dict]:
    """**Algorithm 1 + 2:** Fine-tune DQN with pessimistic targets (Eq. 2.6)."""
    return _training_loop_pessimistic(
        agent,
        total_timesteps,
        seed,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        early_stop=early_stop,
    )


def train_offpolicy_greedy_dqn(agent: DQNAgent, total_timesteps: int = 100_000, seed: int = 0) -> List[float]:
    """**Eq. 5.1–5.2:** Pessimistic behavior, greedy bootstrap target."""
    return _training_loop(agent, total_timesteps, seed, mode="offpolicy")


def train_cartpole_attacker(
    defender: DQNAgent,
    agent_type: str = "vanilla",
    total_timesteps: int = 100_000,
    save_path: Optional[str] = None,
) -> SB3_DQN:
    """**Eq. 6.4:** Train DQN attacker ω against fixed defender."""
    save_path = save_path or f"cartpole_attacker_{agent_type}"
    env = CartPoleAttackEnv(defender, agent_type=agent_type, epsilon=defender.config.attack_epsilon)
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
) -> List[float]:
    """PGD-style worst-case perturbation eval (supplement to DQN attacker)."""
    set_seed(seed)
    env = CartPoleDefenderEnv()
    returns: List[float] = []
    eps = agent.config.attack_epsilon
    kappa = agent.config.belief_kappa

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            s_tilde = agent.worst_case_perturbation(obs, eps)
            if mode == "pessimistic":
                action = agent.maximin_action(s_tilde, eps, kappa)
            else:
                action = agent.greedy_action(s_tilde)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)
    env.close()
    return returns


CSV_FIELDNAMES = [
    "experiment_number",
    "timestamp",
    "seed",
    "vanilla_test_avg_reward",
    "vanilla_test_std",
    "vanilla_inference_time_ms",
    "vanilla_vs_attacker_reward",
    "vanilla_vs_attacker_std",
    "vanilla_attack_inference_time_ms",
    "vanilla_pgd_reward",
    "vanilla_pgd_std",
    "pessimistic_train_avg_reward",
    "pessimistic_train_std",
    "pessimistic_test_avg_reward",
    "pessimistic_test_std",
    "pessimistic_test_maximin_avg_reward",
    "pessimistic_test_maximin_std",
    "pessimistic_finetune_stopped_step",
    "pessimistic_best_greedy_eval",
    "pessimistic_early_stopped",
    "pessimistic_inference_time_ms",
    "pessimistic_vs_attacker_reward",
    "pessimistic_vs_attacker_std",
    "pessimistic_attack_inference_time_ms",
    "pessimistic_pgd_reward",
    "pessimistic_pgd_std",
    "offpolicy_greedy_train_avg_reward",
    "offpolicy_greedy_train_std",
    "offpolicy_greedy_test_avg_reward",
    "offpolicy_greedy_test_std",
    "offpolicy_greedy_inference_time_ms",
    "offpolicy_greedy_vs_attacker_reward",
    "offpolicy_greedy_vs_attacker_std",
    "offpolicy_attack_inference_time_ms",
    "offpolicy_pgd_reward",
    "offpolicy_pgd_std",
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
) -> Dict:
    """Full fair pipeline for one seed (mirrors GridWorld run_single_experiment_with_seed)."""
    paths = ensure_seed_dir(seed, checkpoint_base_dir)
    results: Dict = {"experiment_number": experiment_num, "seed": seed, "error": ""}

    print(f"\n[Exp {experiment_num + 1}] Phase 1: Training vanilla DQN (seed={seed})...")
    vanilla, _ = train_vanilla_dqn(total_timesteps=vanilla_steps, seed=seed)
    vanilla.save(paths["vanilla"])

    print(f"[Exp {experiment_num + 1}] Phase 1.5: Evaluating vanilla...")
    v_clean, v_clean_ms = evaluate_agent(
        vanilla, eval_episodes, seed + 1000, "vanilla", measure_time=True
    )
    results["vanilla_test_avg_reward"] = float(np.mean(v_clean))
    results["vanilla_test_std"] = float(np.std(v_clean))
    results["vanilla_inference_time_ms"] = v_clean_ms

    print(f"[Exp {experiment_num + 1}] Phase 2: Training pessimistic DQN...")
    pes = DQNAgent(config=vanilla.config)
    pes.clone_weights_from(vanilla)
    pes.prepare_robust_finetune(freeze_backbone=True)
    pes_train, pes_meta = train_pessimistic_dqn(pes, finetune_pessimistic, seed + 2000)
    pes.save(paths["pessimistic"])
    results["pessimistic_train_avg_reward"] = float(np.mean(pes_train[-100:])) if pes_train else np.nan
    results["pessimistic_train_std"] = float(np.std(pes_train[-100:])) if pes_train else np.nan
    results["pessimistic_finetune_stopped_step"] = pes_meta["stopped_step"]
    results["pessimistic_best_greedy_eval"] = pes_meta["best_greedy_eval"]
    results["pessimistic_early_stopped"] = int(pes_meta["early_stopped"])

    p_clean, p_clean_ms = evaluate_agent(
        pes, eval_episodes, seed + 3000, "vanilla", measure_time=True
    )
    p_clean_mm, p_clean_mm_ms = evaluate_agent(
        pes, eval_episodes, seed + 3100, "pessimistic", measure_time=True
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
    off_train = train_offpolicy_greedy_dqn(off, finetune_offpolicy, seed + 4000)
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
    train_cartpole_attacker(pes, "pessimistic", attack_steps, paths["attacker_pessimistic"])
    train_cartpole_attacker(off, "offpolicy", attack_steps, paths["attacker_offpolicy"])

    att_p = SB3_DQN.load(paths["attacker_pessimistic"])
    att_o = SB3_DQN.load(paths["attacker_offpolicy"])

    print(f"[Exp {experiment_num + 1}] Phase 5: Adversarial evaluation...")
    # GridWorld parity: vanilla tested with pessimistic's attacker
    v_att, v_att_ms = evaluate_agent(
        vanilla, eval_episodes, seed + 6000, "vanilla", att_p, measure_time=True
    )
    results["vanilla_vs_attacker_reward"] = float(np.mean(v_att))
    results["vanilla_vs_attacker_std"] = float(np.std(v_att))
    results["vanilla_attack_inference_time_ms"] = v_att_ms

    p_att, p_att_ms = evaluate_agent(
        pes, eval_episodes, seed + 7000, "pessimistic", att_p, measure_time=True
    )
    results["pessimistic_vs_attacker_reward"] = float(np.mean(p_att))
    results["pessimistic_vs_attacker_std"] = float(np.std(p_att))
    results["pessimistic_attack_inference_time_ms"] = p_att_ms

    o_att, o_att_ms = evaluate_agent(
        off, eval_episodes, seed + 8000, "offpolicy", att_o, measure_time=True
    )
    results["offpolicy_greedy_vs_attacker_reward"] = float(np.mean(o_att))
    results["offpolicy_greedy_vs_attacker_std"] = float(np.std(o_att))
    results["offpolicy_attack_inference_time_ms"] = o_att_ms

    print(f"[Exp {experiment_num + 1}] Phase 6: PGD worst-case evaluation...")
    v_pgd = evaluate_pgd_worst_case(vanilla, eval_episodes, seed + 9000, "vanilla")
    results["vanilla_pgd_reward"] = float(np.mean(v_pgd))
    results["vanilla_pgd_std"] = float(np.std(v_pgd))

    p_pgd = evaluate_pgd_worst_case(pes, eval_episodes, seed + 9100, "pessimistic")
    results["pessimistic_pgd_reward"] = float(np.mean(p_pgd))
    results["pessimistic_pgd_std"] = float(np.std(p_pgd))

    o_pgd = evaluate_pgd_worst_case(off, eval_episodes, seed + 9200, "vanilla")
    results["offpolicy_pgd_reward"] = float(np.mean(o_pgd))
    results["offpolicy_pgd_std"] = float(np.std(o_pgd))

    results["performance_diff_offpolicy_vs_pessimistic"] = (
        results["offpolicy_greedy_test_avg_reward"] - results["pessimistic_test_avg_reward"]
    )
    results["performance_diff_offpolicy_vs_vanilla"] = (
        results["offpolicy_greedy_test_avg_reward"] - results["vanilla_test_avg_reward"]
    )
    results["performance_diff_pessimistic_vs_vanilla"] = (
        results["pessimistic_test_avg_reward"] - results["vanilla_test_avg_reward"]
    )
    if p_att_ms and p_att_ms > 0:
        results["speedup_offpolicy_vs_pessimistic"] = p_att_ms / o_att_ms if o_att_ms else np.nan
    else:
        results["speedup_offpolicy_vs_pessimistic"] = np.nan
    if v_att_ms and v_att_ms > 0:
        results["speedup_offpolicy_vs_vanilla"] = v_att_ms / o_att_ms if o_att_ms else np.nan
    else:
        results["speedup_offpolicy_vs_vanilla"] = np.nan

    print(f"\n[Exp {experiment_num + 1}] Summary:")
    print(f"  Vanilla     clean={results['vanilla_test_avg_reward']:.1f}  "
          f"attack={results['vanilla_vs_attacker_reward']:.1f}  pgd={results['vanilla_pgd_reward']:.1f}")
    print(f"  Pessimistic clean={results['pessimistic_test_avg_reward']:.1f} (greedy) / "
          f"{results['pessimistic_test_maximin_avg_reward']:.1f} (maximin)  "
          f"attack={results['pessimistic_vs_attacker_reward']:.1f}  pgd={results['pessimistic_pgd_reward']:.1f}")
    print(f"  Off-Policy  clean={results['offpolicy_greedy_test_avg_reward']:.1f}  "
          f"attack={results['offpolicy_greedy_vs_attacker_reward']:.1f}  pgd={results['offpolicy_pgd_reward']:.1f}")

    return results


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
) -> None:
    """Run N independent seeds with CSV logging and checkpoint resume."""
    start_exp = 0
    if resume:
        last_done, ckpt_csv = load_experiment_checkpoint(checkpoint_file)
        if last_done >= 0:
            start_exp = last_done + 1
            csv_filename = ckpt_csv
            print(f"Resuming from experiment {start_exp + 1} (csv={csv_filename})")

    print("=" * 70)
    print(f"CARTPOLE DQN — MULTI-SEED EXPERIMENTS ({num_experiments} total)")
    print("=" * 70)
    print(f"  Seeds:        {base_seed} .. {base_seed + num_experiments - 1}")
    print(f"  CSV:          {csv_filename}")
    print(f"  Eval episodes:{eval_episodes}")
    print("=" * 70)

    last_completed = start_exp - 1
    for exp_num in range(start_exp, num_experiments):
        seed = base_seed + exp_num
        print(f"\n{'=' * 70}")
        print(f"EXPERIMENT {exp_num + 1}/{num_experiments}  (seed={seed})")
        print(f"{'=' * 70}")

        try:
            results = run_single_experiment_with_seed(
                experiment_num=exp_num,
                seed=seed,
                vanilla_steps=vanilla_steps,
                finetune_pessimistic=finetune_pessimistic,
                finetune_offpolicy=finetune_offpolicy,
                attack_steps=attack_steps,
                eval_episodes=eval_episodes,
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

    def _report(label: str, clean_col: str, att_col: str, pgd_col: Optional[str] = None) -> None:
        print(f"\n--- {label} ---")
        if clean_col in df.columns:
            s = ok[clean_col].dropna()
            if len(s):
                print(f"Mean test reward: {s.mean():.2f} ± {s.std():.2f}")
        if att_col in df.columns:
            s = ok[att_col].dropna()
            if len(s):
                print(f"Mean vs attacker: {s.mean():.2f} ± {s.std():.2f}")
        if pgd_col and pgd_col in df.columns:
            s = ok[pgd_col].dropna()
            if len(s):
                print(f"Mean PGD worst-case: {s.mean():.2f} ± {s.std():.2f}")

    _report("Vanilla Q-Learning (Baseline)", "vanilla_test_avg_reward", "vanilla_vs_attacker_reward", "vanilla_pgd_reward")
    _report("Pessimistic Q-Learning", "pessimistic_test_avg_reward", "pessimistic_vs_attacker_reward", "pessimistic_pgd_reward")
    if "pessimistic_test_maximin_avg_reward" in df.columns:
        s = ok["pessimistic_test_maximin_avg_reward"].dropna()
        if len(s):
            print(f"Mean test reward (maximin): {s.mean():.2f} ± {s.std():.2f}")
    if "pessimistic_best_greedy_eval" in df.columns:
        s = ok["pessimistic_best_greedy_eval"].dropna()
        if len(s):
            print(f"Mean best greedy during finetune: {s.mean():.2f} ± {s.std():.2f}")
    if "pessimistic_early_stopped" in df.columns:
        print(f"Early stopped: {ok['pessimistic_early_stopped'].sum():.0f}/{len(ok)} runs")
    _report("Off-Policy Greedy Target", "offpolicy_greedy_test_avg_reward", "offpolicy_greedy_vs_attacker_reward", "offpolicy_pgd_reward")

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CartPole DQN experiments")
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="1=vanilla, 2=attack eval, 3=pessimistic, 4=off-policy, 5=attackers, "
        "6=single comparison, 7=multi-seed loop, 8=analyze CSV",
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
        )
    elif args.step == 8:
        analyze_cartpole_csv(args.csv)

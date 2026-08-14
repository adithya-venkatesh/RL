# -*- coding: utf-8 -*-
"""Belief RNN (PF-RNN-style) for CartPole BP-DQN."""

from __future__ import annotations

from typing import List, Optional, Protocol, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CARTPOLE_LOW = np.array([-4.8, -5.0, -0.418, -5.0], dtype=np.float32)
CARTPOLE_HIGH = np.array([4.8, 5.0, 0.418, 5.0], dtype=np.float32)
DEFAULT_ATTACK_EPS = 0.05
DEFAULT_BELIEF_KAPPA = 20


def denormalize_obs(
    obs_norm: np.ndarray,
    low: np.ndarray = CARTPOLE_LOW,
    high: np.ndarray = CARTPOLE_HIGH,
) -> np.ndarray:
    return ((obs_norm + 1.0) * 0.5 * (high - low) + low).astype(np.float32)


def normalize_obs(
    obs: np.ndarray,
    low: np.ndarray = CARTPOLE_LOW,
    high: np.ndarray = CARTPOLE_HIGH,
) -> np.ndarray:
    return (2.0 * (obs - low) / (high - low) - 1.0).astype(np.float32)


def apply_perturbation(
    state: np.ndarray,
    delta: np.ndarray,
    low: np.ndarray = CARTPOLE_LOW,
    high: np.ndarray = CARTPOLE_HIGH,
    normalized: bool = True,
) -> np.ndarray:
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
    raw = denormalize_obs(state, low, high) if normalized else state
    samples = []
    for _ in range(n_samples):
        delta = np.random.uniform(-epsilon, epsilon, size=raw.shape).astype(np.float32)
        if normalized:
            samples.append(apply_perturbation(state, delta, low, high, normalized=True))
        else:
            samples.append(np.clip(raw + delta, low, high).astype(np.float32))
    return samples


class BeliefProvider(Protocol):
    def reset_episode(self) -> None: ...

    def observe(self, obs: np.ndarray) -> List[np.ndarray]: ...

    def particles_at(self, center: np.ndarray) -> List[np.ndarray]: ...


class UniformBallBelief:
    """Uniform i.i.d. sampling in B_ε(s) — legacy baseline."""

    def __init__(self, epsilon: float = DEFAULT_ATTACK_EPS, kappa: int = DEFAULT_BELIEF_KAPPA):
        self.epsilon = epsilon
        self.kappa = kappa

    def reset_episode(self) -> None:
        pass

    def observe(self, obs: np.ndarray) -> List[np.ndarray]:
        return sample_ball(obs, self.epsilon, self.kappa)

    def particles_at(self, center: np.ndarray) -> List[np.ndarray]:
        return sample_ball(center, self.epsilon, self.kappa)


class BeliefRNN(nn.Module):
    """GRU belief encoder + particle head (κ points in B_ε(s))."""

    def __init__(
        self,
        obs_dim: int = 4,
        hidden: int = 64,
        kappa: int = DEFAULT_BELIEF_KAPPA,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_size = hidden
        self.kappa = kappa
        self.gru = nn.GRU(obs_dim, hidden, batch_first=True)
        self.particle_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, kappa * obs_dim),
        )
        self.next_obs_head = nn.Linear(hidden, obs_dim)

    def init_hidden(self, batch: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        dev = device or next(self.parameters()).device
        return torch.zeros(1, batch, self.hidden_size, device=dev)

    def step(self, obs_seq: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, hidden = self.gru(obs_seq, hidden)
        return out[:, -1, :], hidden

    def predict_next(self, hidden_vec: torch.Tensor) -> torch.Tensor:
        return self.next_obs_head(hidden_vec)

    def particles_from_center(
        self,
        center: np.ndarray,
        hidden: torch.Tensor,
        epsilon: float,
        normalized: bool = True,
    ) -> List[np.ndarray]:
        h = hidden.squeeze(0)[0]
        raw = self.particle_head(h)
        deltas = torch.tanh(raw.view(self.kappa, self.obs_dim)) * float(epsilon)
        center_np = np.asarray(center, dtype=np.float32)
        particles: List[np.ndarray] = []
        for k in range(self.kappa):
            delta = deltas[k].detach().cpu().numpy().astype(np.float32)
            particles.append(
                apply_perturbation(center_np, delta, CARTPOLE_LOW, CARTPOLE_HIGH, normalized=normalized)
            )
        return particles


class LearnedBeliefProvider:
    """Online belief propagation with a trained BeliefRNN."""

    def __init__(
        self,
        model: BeliefRNN,
        epsilon: float = DEFAULT_ATTACK_EPS,
        kappa: int = DEFAULT_BELIEF_KAPPA,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.epsilon = epsilon
        self.kappa = kappa
        self.device = device or next(model.parameters()).device
        self.hidden: Optional[torch.Tensor] = None

    def reset_episode(self) -> None:
        self.hidden = self.model.init_hidden(1, self.device)

    def observe(self, obs: np.ndarray) -> List[np.ndarray]:
        self.model.eval()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, 1, -1)
        with torch.no_grad():
            _, self.hidden = self.model.step(obs_t, self.hidden)
            return self.model.particles_from_center(obs, self.hidden, self.epsilon)

    def particles_at(self, center: np.ndarray) -> List[np.ndarray]:
        if self.hidden is None:
            self.reset_episode()
        self.model.eval()
        with torch.no_grad():
            return self.model.particles_from_center(center, self.hidden, self.epsilon)


def make_belief_provider(
    mode: str,
    epsilon: float = DEFAULT_ATTACK_EPS,
    kappa: int = DEFAULT_BELIEF_KAPPA,
    model: Optional[BeliefRNN] = None,
    device: Optional[torch.device] = None,
) -> BeliefProvider:
    if mode == "uniform":
        return UniformBallBelief(epsilon, kappa)
    if mode == "learned":
        if model is None:
            raise ValueError("learned belief requires a BeliefRNN model")
        return LearnedBeliefProvider(model, epsilon, kappa, device)
    raise ValueError(f"Unknown belief mode: {mode}")


def pretrain_belief_rnn_on_agent(
    vanilla_agent,
    belief: BeliefRNN,
    defender_env_factory,
    timesteps: int = 30_000,
    seed: int = 0,
    epsilon: float = DEFAULT_ATTACK_EPS,
    log_interval: int = 5_000,
    set_seed_fn=None,
) -> float:
    """Pretrain on clean vanilla rollouts with synthetic L∞ perturbations."""
    if set_seed_fn is not None:
        set_seed_fn(seed)
    env = defender_env_factory()
    device = next(belief.parameters()).device
    opt = optim.Adam(belief.parameters(), lr=1e-3)
    belief.train()

    obs, _ = env.reset(seed=seed)
    hidden = belief.init_hidden(1, device)
    total_loss = 0.0
    n_steps = 0

    for step in range(1, timesteps + 1):
        action = vanilla_agent.greedy_action(obs)
        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        true_obs = obs

        if np.random.random() < 0.5:
            delta = np.random.uniform(-epsilon, epsilon, size=4).astype(np.float32)
            defender_obs = apply_perturbation(true_obs, delta, normalized=True)
        else:
            defender_obs = true_obs

        obs_t = torch.as_tensor(defender_obs, dtype=torch.float32, device=device).view(1, 1, -1)
        h_out, hidden = belief.step(obs_t, hidden)
        particles = belief.particles_from_center(defender_obs, hidden, epsilon)
        pt = torch.as_tensor(np.stack(particles), dtype=torch.float32, device=device)
        true_t = torch.as_tensor(true_obs, dtype=torch.float32, device=device)
        cover_loss = torch.min(torch.norm(pt - true_t, dim=1))

        pred = belief.predict_next(h_out.squeeze(0))
        next_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
        dyn_loss = F.mse_loss(pred, next_t)
        loss = cover_loss + dyn_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(belief.parameters(), max_norm=5.0)
        opt.step()
        hidden = hidden.detach()

        total_loss += float(loss.item())
        n_steps += 1
        obs = next_obs
        if done:
            obs, _ = env.reset()
            hidden = belief.init_hidden(1, device)

        if step % log_interval == 0:
            print(f"  belief pretrain step {step}/{timesteps} | loss={total_loss / n_steps:.4f}", flush=True)

    env.close()
    return total_loss / max(n_steps, 1)


def save_belief_rnn(model: BeliefRNN, path: str) -> None:
    torch.save({"state_dict": model.state_dict(), "kappa": model.kappa, "hidden_size": model.hidden_size}, path)


def load_belief_rnn(path: str, device: Optional[torch.device] = None) -> BeliefRNN:
    dev = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    model = BeliefRNN(kappa=int(ckpt.get("kappa", DEFAULT_BELIEF_KAPPA)), hidden=int(ckpt.get("hidden_size", 64)))
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev)
    model.eval()
    return model

# -*- coding: utf-8 -*-
"""Sound (provably true-state-containing) belief set for CartPole BP-DQN.

The tabular GridWorld belief in ``main_with_vanilla_gridworldoriginalwithformulas.py``
is sound because of the filter-then-propagate recursion (Eq. 3.2-3.3):

    M_{t+1} = { T(m, a_t) : m in M_t ∩ B_ε(s̃_t) }

Both operations preserve the invariant ``s_t ∈ M_t``: the true state is in
``B_ε(s̃_t)`` by the threat model, and ``T`` is applied to every member.

The deep port (``cartpole_belief.UniformBallBelief`` / ``LearnedBeliefProvider``)
drops the recursion entirely: each call re-samples κ i.i.d. points from the full
ε-ball around the current centre. Nothing is propagated, nothing is intersected,
and κ=20 i.i.d. samples of a 4-D box essentially never contain the true state.

This module restores the recursion in continuous space with interval (box)
arithmetic over the exact CartPole-v1 dynamics, so the maintained box provably
contains the true state at every step.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

CARTPOLE_LOW = np.array([-4.8, -5.0, -0.418, -5.0], dtype=np.float64)
CARTPOLE_HIGH = np.array([4.8, 5.0, 0.418, 5.0], dtype=np.float64)

# gymnasium.envs.classic_control.cartpole.CartPoleEnv constants
GRAVITY = 9.8
MASSPOLE = 0.1
TOTAL_MASS = 1.1
LENGTH = 0.5
POLEMASS_LENGTH = MASSPOLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02


# --------------------------------------------------------------------------- #
# scalar interval arithmetic
# --------------------------------------------------------------------------- #

def _mul(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    c = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return min(c), max(c)


def _div_pos_den(a: Tuple[float, float], d: Tuple[float, float]) -> Tuple[float, float]:
    """a / d where d is strictly positive."""
    c = (a[0] / d[0], a[0] / d[1], a[1] / d[0], a[1] / d[1])
    return min(c), max(c)


def _sq(a: Tuple[float, float]) -> Tuple[float, float]:
    if a[0] <= 0.0 <= a[1]:
        return 0.0, max(a[0] * a[0], a[1] * a[1])
    lo, hi = a[0] * a[0], a[1] * a[1]
    return min(lo, hi), max(lo, hi)


def _sin(a: Tuple[float, float]) -> Tuple[float, float]:
    """Monotone on the CartPole angle range |θ| ≤ 0.418 < π/2."""
    return math.sin(a[0]), math.sin(a[1])


def _cos(a: Tuple[float, float]) -> Tuple[float, float]:
    m = max(abs(a[0]), abs(a[1]))
    lo = math.cos(m)
    hi = 1.0 if a[0] <= 0.0 <= a[1] else math.cos(min(abs(a[0]), abs(a[1])))
    return lo, hi


def cartpole_interval_step(
    lo: np.ndarray, hi: np.ndarray, action: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Sound interval image of the exact CartPole Euler dynamics (raw units).

    Returns a box guaranteed to contain ``T(s, action)`` for every ``s`` in
    ``[lo, hi]``.
    """
    x = (float(lo[0]), float(hi[0]))
    xd = (float(lo[1]), float(hi[1]))
    th = (float(lo[2]), float(hi[2]))
    thd = (float(lo[3]), float(hi[3]))

    force = FORCE_MAG if int(action) == 1 else -FORCE_MAG
    sin_th = _sin(th)
    cos_th = _cos(th)

    pm = _mul(_sq(thd), sin_th)
    temp = (
        (force + POLEMASS_LENGTH * pm[0]) / TOTAL_MASS,
        (force + POLEMASS_LENGTH * pm[1]) / TOTAL_MASS,
    )

    ct = _mul(cos_th, temp)
    num = (GRAVITY * sin_th[0] - ct[1], GRAVITY * sin_th[1] - ct[0])
    # cos_th is positive on the angle range, so cos² is monotone in cos.
    cos2 = (cos_th[0] * cos_th[0], cos_th[1] * cos_th[1])
    den = (
        LENGTH * (4.0 / 3.0 - MASSPOLE * cos2[1] / TOTAL_MASS),
        LENGTH * (4.0 / 3.0 - MASSPOLE * cos2[0] / TOTAL_MASS),
    )
    thacc = _div_pos_den(num, den)

    tc = _mul(thacc, cos_th)
    k = POLEMASS_LENGTH / TOTAL_MASS
    xacc = (temp[0] - k * tc[1], temp[1] - k * tc[0])

    nlo = np.array(
        [x[0] + TAU * xd[0], xd[0] + TAU * xacc[0], th[0] + TAU * thd[0], thd[0] + TAU * thacc[0]],
        dtype=np.float64,
    )
    nhi = np.array(
        [x[1] + TAU * xd[1], xd[1] + TAU * xacc[1], th[1] + TAU * thd[1], thd[1] + TAU * thacc[1]],
        dtype=np.float64,
    )
    return nlo, nhi


# --------------------------------------------------------------------------- #
# IBP lower bound on min_{m in box} Q(m, a)
# --------------------------------------------------------------------------- #

def ibp_q_bounds(
    q_net: torch.nn.Module, lo_norm: np.ndarray, hi_norm: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Interval bound propagation through a Linear/ReLU MLP.

    ``lo[a]`` is a certified lower bound on ``min_{m ∈ box} Q(m, a)``.
    """
    lo = torch.as_tensor(lo_norm, dtype=torch.float32).view(1, -1)
    hi = torch.as_tensor(hi_norm, dtype=torch.float32).view(1, -1)
    with torch.no_grad():
        for layer in q_net.net:
            if isinstance(layer, torch.nn.Linear):
                w = layer.weight
                c = 0.5 * (lo + hi)
                r = 0.5 * (hi - lo)
                c = torch.nn.functional.linear(c, w, layer.bias)
                r = torch.nn.functional.linear(r, w.abs(), None)
                lo, hi = c - r, c + r
            elif isinstance(layer, torch.nn.ReLU):
                lo, hi = lo.clamp_min(0.0), hi.clamp_min(0.0)
            else:
                raise TypeError(f"ibp_q_bounds supports Linear/ReLU only, got {type(layer)}")
    return lo.view(-1).numpy(), hi.view(-1).numpy()


# --------------------------------------------------------------------------- #
# belief provider
# --------------------------------------------------------------------------- #

def _norm(raw: np.ndarray) -> np.ndarray:
    return (2.0 * (raw - CARTPOLE_LOW) / (CARTPOLE_HIGH - CARTPOLE_LOW) - 1.0).astype(np.float32)


def _denorm(obs_norm: np.ndarray) -> np.ndarray:
    return (
        (np.asarray(obs_norm, dtype=np.float64) + 1.0) * 0.5 * (CARTPOLE_HIGH - CARTPOLE_LOW)
        + CARTPOLE_LOW
    )


_CORNERS = np.array(
    [[(i >> d) & 1 for d in range(4)] for i in range(16)], dtype=np.float64
)

# Continuous GridWorld uses an m×m lattice over the box; CartPole is 4-D so the
# analogous hard-min set is an m^4 axis-aligned lattice (m=2 → 16 = all corners).
DEFAULT_LATTICE_M = 2


def box_lattice_raw(lo: np.ndarray, hi: np.ndarray, m: int) -> np.ndarray:
    """Deterministic m^4 lattice over the raw-space box ``[lo, hi]``."""
    m = int(m)
    assert m >= 2
    axes = [np.linspace(float(lo[d]), float(hi[d]), m, dtype=np.float64) for d in range(4)]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1)


class SoundFilteredBelief:
    """Box belief with the tabular filter-then-propagate recursion (Eq. 3.2-3.3).

    Invariant: the maintained raw-space box ``[lo, hi]`` contains the true state.

    * ``particles_at(center)`` / ``box_at(center)`` perform the *stateful* update
      (propagate previous box through the taken action, intersect with
      ``B_ε(center)``) and are the single mutation point per decision.
    * ``observe(obs)`` is deliberately **non-mutating**: existing call sites feed
      it the *true* post-step observation, which would both leak information and
      double-apply the recursion. It returns an unfiltered ``B_ε(obs)`` particle
      set, matching ``UniformBallBelief`` semantics for the replay buffer.
    * ``propagated_lattice(action)`` returns the lattice over ``T(M_t, a)`` for
      pessimistic TD targets (continuous-GridWorld pattern) without mutating
      the filter; the next ``particles_at`` still applies the full recursion.
    * ``note_action(a)`` must be called after each action so the propagation can
      use the action actually taken; without it the provider falls back to the
      (sound but much looser) union over all actions.
    """

    uses_true_obs = False
    is_sound_filter = True

    def __init__(
        self,
        epsilon: float = 0.05,
        kappa: int = 20,
        n_actions: int = 2,
        slack: float = 1e-4,
        corner_particles: bool = True,
        lattice_m: int = DEFAULT_LATTICE_M,
    ):
        # Scalar, or a per-dimension vector for restricted threat models
        # (e.g. [ε, 0, ε, 0] = position sensors attacked, velocities clean).
        self.epsilon = np.asarray(epsilon, dtype=np.float64)
        self.kappa = int(kappa)
        self.n_actions = int(n_actions)
        # Attackers use corner perturbations δ ∈ {-ε, 0, +ε}, so the true state
        # lies exactly on the boundary of B_ε(s̃). float32 round-tripping through
        # the normalised observation then pushes it a few ULPs outside, and the
        # error compounds through the recursion. Widen both boxes by `slack`
        # (≪ ε) to keep the containment invariant under floating point.
        self.slack = float(slack)
        self.corner_particles = bool(corner_particles)
        self.lattice_m = int(lattice_m)
        self.lo: Optional[np.ndarray] = None
        self.hi: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self.width_log: List[np.ndarray] = []
        self.n_empty = 0

    # -- bookkeeping ------------------------------------------------------- #

    def reset_episode(self) -> None:
        self.lo = None
        self.hi = None
        self.last_action = None

    def note_action(self, action: int) -> None:
        self.last_action = int(action)

    def snapshot(self):
        """Save filter state. An adaptive attacker that probes the defender's
        policy on candidate observations must restore it, otherwise every probe
        advances the recursion and corrupts the belief."""
        return (
            None if self.lo is None else self.lo.copy(),
            None if self.hi is None else self.hi.copy(),
            self.last_action,
            len(self.width_log),
            self.n_empty,
        )

    def restore(self, snap) -> None:
        lo, hi, last_action, n_width, n_empty = snap
        self.lo = None if lo is None else lo.copy()
        self.hi = None if hi is None else hi.copy()
        self.last_action = last_action
        del self.width_log[n_width:]
        self.n_empty = n_empty

    # -- recursion --------------------------------------------------------- #

    def _obs_box(self, center_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raw = _denorm(center_norm)
        lo = np.maximum(raw - self.epsilon - self.slack, CARTPOLE_LOW - self.slack)
        hi = np.minimum(raw + self.epsilon + self.slack, CARTPOLE_HIGH + self.slack)
        return lo, hi

    def _propagate(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.lo is not None and self.hi is not None
        if self.last_action is not None:
            return cartpole_interval_step(self.lo, self.hi, self.last_action)
        lows, highs = [], []
        for a in range(self.n_actions):
            l, h = cartpole_interval_step(self.lo, self.hi, a)
            lows.append(l)
            highs.append(h)
        return np.min(np.stack(lows), axis=0), np.max(np.stack(highs), axis=0)

    def _update(self, center_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        o_lo, o_hi = self._obs_box(center_norm)
        if self.lo is None:
            lo, hi = o_lo, o_hi
        else:
            p_lo, p_hi = self._propagate()
            p_lo = p_lo - self.slack
            p_hi = p_hi + self.slack
            lo = np.maximum(p_lo, o_lo)
            hi = np.minimum(p_hi, o_hi)
            if np.any(lo > hi):
                # Should not happen under the threat model; if numerics or the
                # env's own clipping break the intersection, fall back to the
                # always-sound observation ball.
                self.n_empty += 1
                lo, hi = o_lo, o_hi
        self.lo, self.hi = lo, hi
        self.width_log.append((hi - lo).copy())
        return lo, hi

    # -- BeliefProvider surface ------------------------------------------- #

    def _particles(self, lo: np.ndarray, hi: np.ndarray) -> List[np.ndarray]:
        """Hard-min support: lattice on the box (GridWorld-style), else corners."""
        if self.lattice_m >= 2:
            raw = box_lattice_raw(lo, hi, self.lattice_m)
            pts = [_norm(p) for p in raw]
            if len(pts) >= self.kappa:
                return pts[: self.kappa]
            # Pad with box centre then uniform draws if kappa > m^4.
            pts.append(_norm(0.5 * (lo + hi)))
            while len(pts) < self.kappa:
                pts.append(_norm(np.random.uniform(lo, hi)))
            return pts[: self.kappa]
        pts: List[np.ndarray] = []
        if self.corner_particles:
            corners = lo + _CORNERS * (hi - lo)
            pts.extend(list(corners[: self.kappa]))
        if len(pts) < self.kappa:
            pts.append(0.5 * (lo + hi))
        while len(pts) < self.kappa:
            pts.append(np.random.uniform(lo, hi))
        return [_norm(p) for p in pts[: self.kappa]]

    def particles_at(self, center: np.ndarray) -> List[np.ndarray]:
        lo, hi = self._update(center)
        return self._particles(lo, hi)

    def box_at(self, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filtered box in *normalised* coordinates (for IBP)."""
        lo, hi = self._update(center)
        return _norm(lo), _norm(hi)

    def propagated_lattice(self, action: int) -> List[np.ndarray]:
        """Lattice over ``T(M_t, a)`` for maximin TD (non-mutating).

        Mirrors continuous GridWorld: store / minimise over the sound image of
        the current box under the taken action. The true ``s'`` is in this set
        whenever ``s ∈ M_t``. Does not advance the filter; call ``note_action``
        then ``particles_at`` on the next observation as usual.
        """
        if self.lo is None or self.hi is None:
            raise RuntimeError("propagated_lattice requires a filtered box; call particles_at first")
        nlo, nhi = cartpole_interval_step(self.lo, self.hi, int(action))
        nlo = nlo - self.slack
        nhi = nhi + self.slack
        return self._particles(nlo, nhi)

    def observe(self, obs: np.ndarray) -> List[np.ndarray]:
        """Non-mutating: unfiltered B_ε(obs) particle set (legacy replay-buffer use).

        Sound fine-tuning should use ``propagated_lattice`` for TD targets
        instead of this method.
        """
        lo, hi = self._obs_box(obs)
        return self._particles(lo, hi)

    # -- diagnostics ------------------------------------------------------- #

    def mean_width(self) -> np.ndarray:
        if not self.width_log:
            return np.zeros(4)
        return np.mean(np.stack(self.width_log), axis=0)

    def clear_width_log(self) -> None:
        self.width_log = []


def contains(belief: SoundFilteredBelief, raw_state: Sequence[float]) -> bool:
    """Soundness check: is the true raw state inside the maintained box?"""
    if belief.lo is None:
        return True
    s = np.asarray(raw_state, dtype=np.float64)
    tol = 1e-6
    return bool(np.all(s >= belief.lo - tol) and np.all(s <= belief.hi + tol))

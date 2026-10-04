"""Continuous-state GridWorld: vanilla / pessimistic / PB-GT Double DQN.

Feasibility prototype for replacing CartPole as the deep-RL testbed. The point of
this environment is that the observation perturbation is L_inf in the *state
space itself*, so the set of candidate true states given an observation s~ is
exactly B_eps(s~) intersected with the domain. That set is sound by construction
-- it always contains the true state -- so no learned particle filter (GRU /
PF-RNN) is needed, and the inner min can never be taken over a set that excludes
the truth. This is the same property that makes the tabular 5x5 result work
(main_with_vanilla.py: get_ball + states_intersection), and it is exactly what
the CartPole GRU belief set fails to provide.

Layout is a continuous lift of the tabular 5x5 grid used in the paper
(gold/bomb/-1 step cost), plus a wall and a bomb adjacent to the gold so that an
eps-perturbation can flip the semantics of a state -- the feature that makes
Sun & Zheng's ICLR 2024 continuous GridWorld (gold at cell [9,4], bomb at [9,5],
eps=0.5=one step) adversarially interesting.

Read-only w.r.t. every existing experiment file; writes only to the CSV named by
--out.

Usage:
    python3 gridworld_continuous_dqn.py --smoke
    python3 gridworld_continuous_dqn.py --seeds 0 1 2 --vanilla-steps 150000
"""

import argparse
import copy
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

SIZE = 5.0
STEP = 0.5
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
DELTAS = np.array([[0.0, STEP], [0.0, -STEP], [-STEP, 0.0], [STEP, 0.0]], dtype=np.float32)
N_ACTIONS = 4

# Rectangles as (x0, x1, y0, y1). Tabular grid maps (row, col) -> (x=col, y=4-row),
# so gold [0,3] -> x in [3,4], y in [4,5] and bomb [4,3] -> x in [3,4], y in [0,1].
GOLD = (3.0, 4.0, 4.0, 5.0)
BOMBS = [
    (3.0, 4.0, 0.0, 1.0),   # tabular bomb, opposite end of the gold column
    (4.0, 5.0, 4.0, 5.0),   # adjacent to the gold: one step of overshoot is fatal
]
WALLS = [(0.0, 3.0, 2.0, 2.5)]  # barrier with a gap at x > 3, forcing the risky corridor

R_GOLD, R_BOMB, R_STEP = 10.0, -10.0, -1.0
MAX_STEPS = 100


def _in_rect(p, rect):
    return rect[0] <= p[0] <= rect[1] and rect[2] <= p[1] <= rect[3]


def _in_any(p, rects):
    return any(_in_rect(p, r) for r in rects)


class ContinuousGridWorld:
    """2-D continuous coordinates, 4 discrete moves, deterministic dynamics."""

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.state = np.zeros(2, dtype=np.float32)
        self.t = 0

    def reset(self):
        while True:
            p = self.rng.uniform(0.0, SIZE, size=2).astype(np.float32)
            if not _in_any(p, WALLS) and not _in_rect(p, GOLD) and not _in_any(p, BOMBS):
                break
        self.state = p
        self.t = 0
        return self.state.copy()

    @staticmethod
    def dynamics(state, action):
        """s' = T(s, a). Wall collisions leave the agent in place (as in Sun's Gridmaze)."""
        nxt = np.clip(state + DELTAS[action], 0.0, SIZE).astype(np.float32)
        if _in_any(nxt, WALLS):
            return state.astype(np.float32).copy()
        return nxt

    def step(self, action):
        self.state = self.dynamics(self.state, action)
        self.t += 1
        if _in_rect(self.state, GOLD):
            return self.state.copy(), R_GOLD, True, False
        if _in_any(self.state, BOMBS):
            return self.state.copy(), R_BOMB, True, False
        return self.state.copy(), R_STEP, False, self.t >= MAX_STEPS


def batch_dynamics(states, action):
    """Vectorised T(s, a) for propagating a whole particle set."""
    nxt = np.clip(states + DELTAS[action], 0.0, SIZE).astype(np.float32)
    blocked = np.zeros(len(states), dtype=bool)
    for x0, x1, y0, y1 in WALLS:
        blocked |= (nxt[:, 0] >= x0) & (nxt[:, 0] <= x1) & (nxt[:, 1] >= y0) & (nxt[:, 1] <= y1)
    nxt[blocked] = states[blocked]
    return nxt


# ---------------------------------------------------------------------------
# Sound belief sets
# ---------------------------------------------------------------------------

def ball_lattice(obs, eps, m):
    """Deterministic m x m lattice over B_eps(obs) intersected with the domain.

    Every point is a feasible true state given observation obs under the L_inf
    model, and the lattice includes the observation itself (m odd) and the ball
    corners, so the min is taken over the extremes of the sound candidate set.
    """
    lo = np.maximum(obs - eps, 0.0)
    hi = np.minimum(obs + eps, SIZE)
    xs = np.linspace(lo[0], hi[0], m)
    ys = np.linspace(lo[1], hi[1], m)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)


def batch_box_lattice(lo, hi, m):
    """(B, 2), (B, 2) -> (B, m*m, 2)."""
    w = np.linspace(0.0, 1.0, m, dtype=np.float32)
    ax = lo[:, None, 0] + (hi[:, None, 0] - lo[:, None, 0]) * w[None, :]
    ay = lo[:, None, 1] + (hi[:, None, 1] - lo[:, None, 1]) * w[None, :]
    gx = np.repeat(ax, m, axis=1)
    gy = np.tile(ay, (1, m))
    return np.stack([gx, gy], axis=2).astype(np.float32)


def batch_ball_lattice(obs_batch, eps, m):
    return batch_box_lattice(np.maximum(obs_batch - eps, 0.0),
                             np.minimum(obs_batch + eps, SIZE), m)


def obs_box(obs, eps):
    return np.maximum(obs - eps, 0.0).astype(np.float32), np.minimum(obs + eps, SIZE).astype(np.float32)


def _propagate_box(lo, hi, action):
    """Sound over-approximation of T(box, a).

    Wall collisions make T non-linear: a point either translates or stays put. If
    the translated box touches a wall, both outcomes are possible for some member,
    so the hull of {box, translated box} is returned. Otherwise the box translates
    rigidly and no over-approximation is needed.
    """
    t_lo = np.clip(lo + DELTAS[action], 0.0, SIZE)
    t_hi = np.clip(hi + DELTAS[action], 0.0, SIZE)
    touches_wall = any(t_lo[0] <= x1 and t_hi[0] >= x0 and t_lo[1] <= y1 and t_hi[1] >= y0
                       for x0, x1, y0, y1 in WALLS)
    if touches_wall:
        return np.minimum(lo, t_lo).astype(np.float32), np.maximum(hi, t_hi).astype(np.float32)
    return t_lo.astype(np.float32), t_hi.astype(np.float32)


def box_belief_update(box, action, obs, eps):
    """Tabular Eq. 3.2-3.3 lifted to continuous state as a set-membership filter.

    M'_t = T(M_t, a) then M_{t+1} = M'_t intersect B_eps(s~_{t+1}). Because the
    observation boxes at consecutive steps are centred on s+delta_t and s'+delta_{t+1},
    the intersection contracts by |delta_t - delta_{t+1}| per axis, so the candidate
    set gets strictly tighter than the raw eps-ball without any learned filter.
    Sound throughout: the true state is in every set produced here.
    """
    o_lo, o_hi = obs_box(obs, eps)
    if box is None:
        return o_lo, o_hi
    p_lo, p_hi = _propagate_box(box[0], box[1], action)
    lo = np.maximum(p_lo, o_lo)
    hi = np.minimum(p_hi, o_hi)
    if np.any(lo > hi):  # inconsistent (over-approximation was too tight somewhere)
        return o_lo, o_hi
    return lo.astype(np.float32), hi.astype(np.float32)


def box_lattice(box, m):
    lo, hi = box
    xs = np.linspace(lo[0], hi[0], m)
    ys = np.linspace(lo[1], hi[1], m)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Q-network (matches Sun & Zheng's continuous-Grid MLP: 2 -> 128 -> 128 -> |A|)
# ---------------------------------------------------------------------------

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                               nn.Linear(128, N_ACTIONS))

    def forward(self, x):
        return self.f(x)


def q_of(net, states):
    with torch.no_grad():
        return net(torch.as_tensor(np.atleast_2d(states), dtype=torch.float32)).numpy()


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------

def greedy_action(net, obs):
    return int(np.argmax(q_of(net, obs)[0]))


def maximin_action(net, obs, eps, m, particles=None):
    """pi(s~) = argmax_a min_{m in B} Q(m, a) -- hard min, as in the tabular code."""
    pts = ball_lattice(obs, eps, m) if particles is None else particles
    return int(np.argmax(q_of(net, pts).min(axis=0)))


def belief_step(mode, box, last_a, obs, eps, m):
    """Return (particles, new_box) for the current observation under the chosen mode.

    mode='ball' is the memoryless sound eps-ball; mode='box' additionally carries the
    set-membership filter forward in time (the continuous analogue of the tabular
    belief intersection). Both are sound; only 'box' contracts.
    """
    if mode == 'box':
        box = box_belief_update(box, last_a, obs, eps)
        return box_lattice(box, m), box
    return ball_lattice(obs, eps, m), None


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------

def clip_ball(cand, state, eps):
    return np.clip(np.clip(cand, state - eps, state + eps), 0.0, SIZE).astype(np.float32)


def attack_shared_random(state, eps, rng):
    return clip_ball(state + rng.uniform(-eps, eps, size=2).astype(np.float32), state, eps)


def attack_pgd(net, state, eps, iters=10):
    """10-step L_inf PGD MinBest: push Q(s~, a*) down, a* greedy at the true state.

    Genuinely iterative gradient PGD, which the tabular/CartPole PGD* (a 32-sample
    random search) could not do -- the continuous state makes the attack surface
    differentiable.
    """
    a_star = int(np.argmax(q_of(net, state)[0]))
    x = torch.as_tensor(state[None, :], dtype=torch.float32)
    adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clone()
    step = eps / iters
    for _ in range(iters):
        adv.requires_grad_(True)
        loss = net(adv)[0, a_star]
        grad, = torch.autograd.grad(loss, adv)
        adv = (adv.detach() - step * grad.sign())
        adv = torch.clamp(torch.clamp(adv, x - eps, x + eps), 0.0, SIZE)
    return adv.detach().numpy()[0].astype(np.float32)


def attack_policy_aware(net, state, eps, policy_rule, rng, n_cand=32, m_lattice=3):
    """omega_pi(s) = argmin_{s~ in B_eps(s)} Q(s, pi(s~)) -- tabular Eq. 2.2.

    Defender-specific and rule-aware: it searches over observations and scores
    them by the true-state value of whatever action the defender's *actual* rule
    (greedy or maximin-over-belief) would take, so it adapts to the maximin
    defender rather than to a surrogate.
    """
    cands = [state.astype(np.float32)]
    cands.append(clip_ball(state + np.full(2, eps, np.float32), state, eps))
    lat = ball_lattice(state, eps, m_lattice)
    cands.extend(list(lat))
    extra = max(0, n_cand - len(cands))
    if extra:
        cands.extend(list(clip_ball(state[None, :] + rng.uniform(-eps, eps, size=(extra, 2)), state, eps)))
    cands = np.asarray(cands, dtype=np.float32)
    q_true = q_of(net, state)[0]
    best, best_v = cands[0], np.inf
    for c in cands:
        v = q_true[policy_rule(c)]
        if v < best_v:
            best_v, best = v, c
    return best


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class Buffer:
    """Stores the sound candidate set for s' alongside the transition, so the
    pessimistic target mins over the same set the behaviour policy actually had."""

    def __init__(self, cap):
        self.cap = cap
        self.s = np.zeros((cap, 2), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.s2 = np.zeros((cap, 2), np.float32)
        self.d = np.zeros(cap, np.float32)
        self.blo = np.zeros((cap, 2), np.float32)
        self.bhi = np.zeros((cap, 2), np.float32)
        self.n, self.i = 0, 0

    def add(self, s, a, r, s2, d, blo, bhi):
        self.s[self.i], self.a[self.i], self.r[self.i] = s, a, r
        self.s2[self.i], self.d[self.i] = s2, float(d)
        self.blo[self.i], self.bhi[self.i] = blo, bhi
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, bs, rng):
        idx = rng.integers(0, self.n, size=bs)
        return (self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx],
                self.blo[idx], self.bhi[idx])


def linear_schedule(final, initial, total, current):
    frac = min(1.0, current / max(1, total))
    return initial + frac * (final - initial)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def td_update(online, target, opt, batch, gamma, pessimistic, eps, m, pes_nesting):
    s, a, r, s2, d, blo, bhi = batch
    st = torch.as_tensor(s)
    q = online(st).gather(1, torch.as_tensor(a)[:, None]).squeeze(1)

    with torch.no_grad():
        if pessimistic:
            # Eq. 2.6: y = r + gamma * max_a min_{m in M'_t} Q_target(m, a)
            pts = batch_box_lattice(blo, bhi, m)               # (B, P, 2)
            B, P, _ = pts.shape
            qt = target(torch.as_tensor(pts.reshape(B * P, 2))).reshape(B, P, N_ACTIONS)
            if pes_nesting == 'maxmin':
                boot = qt.min(dim=1).values.max(dim=1).values
            else:  # 'minmax': the nesting actually shipped in Sun & Zheng's train.py
                boot = qt.max(dim=2).values.min(dim=1).values
        else:
            # Double DQN greedy target on the true next state
            a2 = online(torch.as_tensor(s2)).argmax(dim=1, keepdim=True)
            boot = target(torch.as_tensor(s2)).gather(1, a2).squeeze(1)
        y = torch.as_tensor(r) + gamma * (1.0 - torch.as_tensor(d)) * boot

    loss = F.smooth_l1_loss(q, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.item())


def train(variant, steps, seed, args, init_state_dict=None, log_prefix=''):
    """variant in {vanilla, pessimistic, pbgt}.

    vanilla     : clean observations, Double-DQN greedy target.
    pessimistic : maximin behaviour under worst-case observations, pessimistic target.
    pbgt        : maximin behaviour under worst-case observations, greedy target.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 7919)
    env = ContinuousGridWorld(np.random.default_rng(seed + 101))

    online, target = QNet(), QNet()
    if init_state_dict is not None:
        online.load_state_dict(copy.deepcopy(init_state_dict))
    target.load_state_dict(online.state_dict())
    lr = args.lr if variant == 'vanilla' else args.ft_lr
    opt = torch.optim.Adam(online.parameters(), lr=lr, eps=1.5e-4)
    buf = Buffer(args.buffer)

    eps_ini = 0.2 if variant == 'vanilla' else 0.05
    m = args.particles
    s = env.reset()
    box, last_a = None, 0
    ep_ret, returns = 0.0, []

    for t in range(steps):
        explore = linear_schedule(0.01, eps_ini, steps, t)
        if variant == 'vanilla':
            act = rng.integers(N_ACTIONS) if rng.random() < explore else greedy_action(online, s)
            act = int(act)
            # Vanilla trains on clean states; its target only needs the true s'.
            nlo = nhi = ContinuousGridWorld.dynamics(s, act)
        else:
            rule = lambda c: maximin_action(online, c, args.eps, m)
            obs = attack_policy_aware(online, s, args.eps, rule, rng, n_cand=args.train_cand)
            particles, box = belief_step(args.belief, box, last_a, obs, args.eps, m)
            act = rng.integers(N_ACTIONS) if rng.random() < explore else \
                maximin_action(online, obs, args.eps, m, particles)
            act = int(act)
            # M'_t: sound candidate set for s', used by the pessimistic target.
            if args.belief == 'box':
                nlo, nhi = _propagate_box(box[0], box[1], act)
            else:
                nlo, nhi = obs_box(ContinuousGridWorld.dynamics(s, act), args.eps)
        last_a = act

        s2, r, done, trunc = env.step(act)
        buf.add(s, act, r, s2, done, nlo, nhi)
        ep_ret += r
        s = s2
        if done or trunc:
            returns.append(ep_ret)
            ep_ret, box, last_a = 0.0, None, 0
            s = env.reset()

        if buf.n >= args.learn_start:
            td_update(online, target, opt, buf.sample(args.batch, rng), args.gamma,
                      pessimistic=(variant == 'pessimistic'), eps=args.eps, m=m,
                      pes_nesting=args.pes_nesting)
        if (t + 1) % args.target_update == 0:
            target.load_state_dict(online.state_dict())
        if args.verbose and (t + 1) % max(1, steps // 5) == 0:
            recent = np.mean(returns[-20:]) if returns else float('nan')
            print(f'    {log_prefix}{variant:<12} step {t+1:>7}/{steps}  train_ret(last20)={recent:7.2f}',
                  flush=True)

    return online, returns


# ---------------------------------------------------------------------------
# Evaluation: clean / shared random / PGD / matched, mirroring the tabular suite
# ---------------------------------------------------------------------------

def evaluate(net, variant, condition, args, seed, episodes):
    m = args.particles
    use_belief = (variant == 'pessimistic')
    rule = (lambda c: maximin_action(net, c, args.eps, m)) if use_belief else \
           (lambda c: greedy_action(net, c))
    rets, times, widths = [], [], []

    for ep in range(episodes):
        env = ContinuousGridWorld(np.random.default_rng(seed * 10_000 + ep))
        # Shared stream: same seed for every defender, so all see identical deltas.
        noise_rng = np.random.default_rng(777_000 + ep)
        atk_rng = np.random.default_rng(seed * 31 + ep)
        s = env.reset()
        box, total, last_a = None, 0.0, 0

        for _ in range(MAX_STEPS):
            if condition == 'clean':
                obs = s
            elif condition == 'shared_random':
                obs = attack_shared_random(s, args.eps, noise_rng)
            elif condition == 'pgd':
                obs = attack_pgd(net, s, args.eps, iters=args.pgd_iters)
            elif condition == 'matched':
                obs = attack_policy_aware(net, s, args.eps, rule, atk_rng, n_cand=args.eval_cand)
            else:
                raise ValueError(condition)

            t0 = time.perf_counter()
            if use_belief:
                particles, box = belief_step(args.belief, box, last_a, obs, args.eps, m)
                act = maximin_action(net, obs, args.eps, m, particles)
                if box is not None:
                    widths.append(float(np.mean(box[1] - box[0])))
            else:
                act = greedy_action(net, obs)
            times.append(time.perf_counter() - t0)

            last_a = act
            s, r, done, trunc = env.step(act)
            total += r
            if done or trunc:
                break
        rets.append(total)

    width = float(np.mean(widths)) if widths else float('nan')
    return float(np.mean(rets)), float(np.std(rets)), float(np.mean(times) * 1000.0), width


CONDITIONS = ['clean', 'shared_random', 'pgd', 'matched']
VARIANTS = ['vanilla', 'pessimistic', 'pbgt']


def run_seed(seed, args):
    print(f'\n=== seed {seed} ===', flush=True)
    t0 = time.time()
    van, van_ret = train('vanilla', args.vanilla_steps, seed, args)
    print(f'  vanilla trained ({time.time()-t0:.0f}s), last-20 train return '
          f'{np.mean(van_ret[-20:]) if van_ret else float("nan"):.2f}', flush=True)

    init = van.state_dict()
    pes, _ = train('pessimistic', args.finetune_steps, seed + 1, args, init_state_dict=init)
    pbgt, _ = train('pbgt', args.finetune_steps, seed + 2, args, init_state_dict=init)
    print(f'  fine-tunes done ({time.time()-t0:.0f}s)', flush=True)

    nets = {'vanilla': van, 'pessimistic': pes, 'pbgt': pbgt}
    row = {'seed': seed}
    for v in VARIANTS:
        for c in CONDITIONS:
            mu, sd, ms, width = evaluate(nets[v], v, c, args, seed, args.episodes)
            row[f'{v}_{c}'] = mu
            row[f'{v}_{c}_std'] = sd
            row[f'{v}_{c}_ms'] = ms
            if v == 'pessimistic':
                row[f'belief_width_{c}'] = width
            extra = '' if np.isnan(width) else f'  belief width {width:.3f} (ball {2*args.eps:.2f})'
            print(f'  {v:<12} {c:<14} {mu:8.2f} +/- {sd:6.2f}   ({ms:.3f} ms/action){extra}',
                  flush=True)
    # Pessimistic weights read out greedily, to detect the CartPole pathology where
    # maximin scores below its own greedy read-out.
    mu, sd, _, _ = evaluate(pes, 'pessimistic_greedy', 'clean', args, seed, args.episodes)
    row['pessimistic_clean_greedy'] = mu
    print(f'  {"pessimistic":<12} {"clean(greedy)":<14} {mu:8.2f} +/- {sd:6.2f}', flush=True)
    row['wall_clock_s'] = round(time.time() - t0, 1)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--vanilla-steps', type=int, default=120_000)
    p.add_argument('--finetune-steps', type=int, default=40_000)
    p.add_argument('--episodes', type=int, default=30)
    p.add_argument('--eps', type=float, default=0.25, help='L_inf observation budget')
    p.add_argument('--particles', type=int, default=3, help='m; belief set is an m x m lattice')
    p.add_argument('--belief', choices=['ball', 'box'], default='box',
                   help="'ball': memoryless sound eps-ball. 'box': eps-ball plus the "
                        "temporal set-membership intersection, which contracts.")
    p.add_argument('--pes-nesting', choices=['maxmin', 'minmax'], default='maxmin')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--ft-lr', type=float, default=1e-4)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--buffer', type=int, default=100_000)
    p.add_argument('--learn-start', type=int, default=1_000)
    p.add_argument('--target-update', type=int, default=500)
    p.add_argument('--train-cand', type=int, default=16)
    p.add_argument('--eval-cand', type=int, default=32)
    p.add_argument('--pgd-iters', type=int, default=10)
    p.add_argument('--out', type=str, default='gridworld_continuous_smoke.csv')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--smoke', action='store_true', help='tiny budget, 2 seeds')
    args = p.parse_args()

    if args.smoke:
        args.seeds = [0, 1]
        args.vanilla_steps = 40_000
        args.finetune_steps = 12_000
        args.episodes = 20
        args.verbose = True

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    print(f'eps={args.eps}  belief={args.belief} ({args.particles}x{args.particles} lattice)  '
          f'nesting={args.pes_nesting}  vanilla={args.vanilla_steps} ft={args.finetune_steps} '
          f'episodes={args.episodes}', flush=True)

    rows = [run_seed(s, args) for s in args.seeds]

    if os.path.exists(args.out):
        raise SystemExit(f'refusing to overwrite {args.out}')
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print('\n=== summary (mean over seeds of per-seed mean return) ===')
    hdr = f'{"condition":<15}' + ''.join(f'{v:>16}' for v in VARIANTS)
    print(hdr)
    for c in CONDITIONS:
        line = f'{c:<15}'
        for v in VARIANTS:
            vals = [r[f'{v}_{c}'] for r in rows]
            line += f'{np.mean(vals):>10.1f}+/-{np.std(vals):<5.1f}'
        print(line)
    print(f'\npessimistic clean greedy read-out: '
          f'{np.mean([r["pessimistic_clean_greedy"] for r in rows]):.1f}')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

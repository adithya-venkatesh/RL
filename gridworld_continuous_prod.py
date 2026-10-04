"""Production driver for the continuous GridWorld experiment.

Read-only w.r.t. ``gridworld_continuous_dqn.py`` (imported as a library) and
every CartPole file. Adds what the smoke prototype was missing:

* a genuine **trained SB3 DQN matched attacker**, the same protocol the tabular
  code (main_with_vanilla_gridworldoriginalwithformulas.py) and the CartPole code
  (main_cartpole_dqn.train_cartpole_attacker) use: an attacker MDP that observes
  the true state, acts by choosing a perturbation from a discrete grid over
  B_eps, and is rewarded with the negative of the defender's reward. Trained per
  (defender, seed). The cheap 32-candidate one-step search is kept as a second
  matched column.
* the ``pessimistic(ball)`` ablation as a first-class variant, so the "memoryless
  sound ball is not enough" result is measured under the same protocol.
* the contraction measurement (filtered box width vs raw observation ball width)
  and per-action inference cost on every cell.

Long-format CSV: one row per (seed, variant, condition).

Usage:
    python3 gridworld_continuous_prod.py --eps 0.35 --seeds 0 --out foo.csv
    python3 gridworld_continuous_prod.py --summarize foo.csv bar.csv
"""

import argparse
import copy
import csv
import os
import time

import numpy as np
import torch

import gridworld_continuous_dqn as G

# label -> (train variant, belief mode, uses belief at test time)
VARIANT_SPECS = {
    'vanilla': ('vanilla', 'box', False),
    'pessimistic_box': ('pessimistic', 'box', True),
    'pbgt_box': ('pbgt', 'box', False),
    'pessimistic_ball': ('pessimistic', 'ball', True),
}
ALL_VARIANTS = list(VARIANT_SPECS)
ALL_CONDITIONS = ['clean', 'shared_random', 'pgd', 'matched_search', 'matched_sb3']


# ---------------------------------------------------------------------------
# Trained matched attacker (SB3 DQN over the attacker MDP)
# ---------------------------------------------------------------------------

def _make_attack_env(net, belief_mode, use_belief, args, seed):
    import gymnasium as gym

    eps, m, L = args.eps, args.particles, args.attack_levels
    levels = np.linspace(-eps, eps, L).astype(np.float32)

    class GridAttackEnv(gym.Env):
        """Attacker MDP <S, A_att, P~, R~, gamma>.

        Observes the true state s (full information, as in the CartPole attacker
        env), picks delta on an L x L grid over B_eps, the defender sees
        s~ = clip(s + delta, 0, SIZE) and acts with its *actual* decision rule
        (greedy, or maximin over the belief set it is carrying), and the attacker
        is paid -R(s, a). The defender's belief filter advances inside the env,
        so the attacker is playing against the real, history-dependent defender.
        """

        metadata = {'render_modes': []}

        def __init__(self):
            self.action_space = gym.spaces.Discrete(L * L)
            self.observation_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
            self.inner = G.ContinuousGridWorld(np.random.default_rng(seed))
            self.box, self.last_a = None, 0

        @staticmethod
        def _enc(s):
            return (2.0 * np.asarray(s, np.float32) / G.SIZE - 1.0).astype(np.float32)

        def reset(self, *, seed=None, options=None):
            s = self.inner.reset()
            self.box, self.last_a = None, 0
            return self._enc(s), {}

        def step(self, action):
            i, j = divmod(int(action), L)
            s = self.inner.state.copy()
            obs = np.clip(s + np.array([levels[i], levels[j]], np.float32), 0.0, G.SIZE).astype(np.float32)
            if use_belief:
                particles, self.box = G.belief_step(belief_mode, self.box, self.last_a, obs, eps, m)
                a = G.maximin_action(net, obs, eps, m, particles)
            else:
                a = G.greedy_action(net, obs)
            self.last_a = a
            s2, r, done, trunc = self.inner.step(a)
            if done or trunc:
                self.box, self.last_a = None, 0
            return self._enc(s2), -float(r), bool(done), bool(trunc), {}

    return GridAttackEnv()


def train_matched_attacker(net, spec, args, seed, tag):
    """Train an SB3 DQN attacker against this exact (defender, seed)."""
    from stable_baselines3 import DQN as SB3_DQN

    _, belief_mode, use_belief = spec
    env = _make_attack_env(net, belief_mode, use_belief, args, seed + 5_000)
    model = SB3_DQN(
        'MlpPolicy', env,
        learning_rate=1e-3, buffer_size=50_000, learning_starts=1_000, batch_size=128,
        gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=500,
        exploration_fraction=0.3, exploration_final_eps=0.05,
        policy_kwargs={'net_arch': [128, 128]}, verbose=0, seed=seed + 991,
    )
    t0 = time.time()
    model.learn(total_timesteps=args.attacker_steps)
    if args.attacker_dir:
        os.makedirs(args.attacker_dir, exist_ok=True)
        model.save(os.path.join(args.attacker_dir, tag))
    print(f'    attacker[{tag}] trained in {time.time()-t0:.0f}s', flush=True)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(net, spec, condition, args, seed, episodes, attacker=None, force_greedy=False):
    """Returns dict with mean/std return, ms per action, filtered vs ball width."""
    _, belief_mode, use_belief = spec
    if force_greedy:
        use_belief = False
    eps, m, L = args.eps, args.particles, args.attack_levels
    levels = np.linspace(-eps, eps, L).astype(np.float32)
    rule = (lambda c: G.maximin_action(net, c, eps, m)) if use_belief else \
           (lambda c: G.greedy_action(net, c))

    rets, times, widths, ball_widths = [], [], [], []
    for ep in range(episodes):
        env = G.ContinuousGridWorld(np.random.default_rng(seed * 10_000 + ep))
        noise_rng = np.random.default_rng(777_000 + ep)   # shared across defenders
        atk_rng = np.random.default_rng(seed * 31 + ep)
        s = env.reset()
        box, total, last_a = None, 0.0, 0

        for _ in range(G.MAX_STEPS):
            if condition == 'clean':
                obs = s
            elif condition == 'shared_random':
                obs = G.attack_shared_random(s, eps, noise_rng)
            elif condition == 'pgd':
                obs = G.attack_pgd(net, s, eps, iters=args.pgd_iters)
            elif condition == 'matched_search':
                obs = G.attack_policy_aware(net, s, eps, rule, atk_rng, n_cand=args.eval_cand)
            elif condition == 'matched_sb3':
                enc = (2.0 * s / G.SIZE - 1.0).astype(np.float32)
                a_att, _ = attacker.predict(enc, deterministic=True)
                i, j = divmod(int(a_att), L)
                obs = np.clip(s + np.array([levels[i], levels[j]], np.float32), 0.0, G.SIZE).astype(np.float32)
            else:
                raise ValueError(condition)

            t0 = time.perf_counter()
            if use_belief:
                particles, box = G.belief_step(belief_mode, box, last_a, obs, eps, m)
                act = G.maximin_action(net, obs, eps, m, particles)
            else:
                act = G.greedy_action(net, obs)
            times.append(time.perf_counter() - t0)

            if use_belief:
                o_lo, o_hi = G.obs_box(obs, eps)
                ball_widths.append(float(np.mean(o_hi - o_lo)))
                if box is not None:
                    widths.append(float(np.mean(box[1] - box[0])))

            last_a = act
            s, r, done, trunc = env.step(act)
            total += r
            if done or trunc:
                break
        rets.append(total)

    return {
        'mean': float(np.mean(rets)), 'std': float(np.std(rets)),
        'ms_per_action': float(np.mean(times) * 1000.0),
        'box_width': float(np.mean(widths)) if widths else float('nan'),
        'ball_width': float(np.mean(ball_widths)) if ball_widths else float('nan'),
        'returns': rets,
    }


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------

def run_seed(seed, args):
    rows = []
    t_start = time.time()
    base = copy.copy(args)
    base.belief = 'box'
    van, van_ret = G.train('vanilla', args.vanilla_steps, seed, base)
    print(f'  [seed {seed}] vanilla trained ({time.time()-t_start:.0f}s), '
          f'train_ret(last20)={np.mean(van_ret[-20:]) if van_ret else float("nan"):.2f}', flush=True)
    init = van.state_dict()

    nets = {'vanilla': van}
    for label in args.variants:
        if label == 'vanilla':
            continue
        tvar, bmode, _ = VARIANT_SPECS[label]
        a = copy.copy(args)
        a.belief = bmode
        off = {'pessimistic_box': 1, 'pbgt_box': 2, 'pessimistic_ball': 3}[label]
        nets[label], _ = G.train(tvar, args.finetune_steps, seed + off, a, init_state_dict=init,
                                 log_prefix=f'[s{seed}] ')
        print(f'  [seed {seed}] {label} fine-tuned ({time.time()-t_start:.0f}s)', flush=True)

    attackers = {}
    if 'matched_sb3' in args.conditions:
        for label in args.variants:
            attackers[label] = train_matched_attacker(
                nets[label], VARIANT_SPECS[label], args, seed, f'gw_att_{label}_eps{args.eps}_s{seed}')

    for label in args.variants:
        spec = VARIANT_SPECS[label]
        for cond in args.conditions:
            r = evaluate(nets[label], spec, cond, args, seed, args.episodes,
                         attacker=attackers.get(label))
            rows.append({'seed': seed, 'eps': args.eps, 'variant': label, 'condition': cond,
                         'mean': r['mean'], 'std': r['std'], 'ms_per_action': r['ms_per_action'],
                         'box_width': r['box_width'], 'ball_width': r['ball_width'],
                         'episodes': args.episodes})
            w = '' if np.isnan(r['box_width']) else \
                f"  box {r['box_width']:.3f} / ball {r['ball_width']:.3f}"
            print(f"  [seed {seed}] {label:<17} {cond:<15} {r['mean']:8.2f} +/- {r['std']:6.2f}"
                  f"   ({r['ms_per_action']:.3f} ms){w}", flush=True)

    # Pessimistic weights read out greedily on clean states: detects the CartPole
    # pathology where maximin scores below its own greedy read-out.
    for label in ('pessimistic_box', 'pessimistic_ball'):
        if label in args.variants:
            r = evaluate(nets[label], VARIANT_SPECS[label], 'clean', args, seed, args.episodes,
                         force_greedy=True)
            rows.append({'seed': seed, 'eps': args.eps, 'variant': label + '_greedyreadout',
                         'condition': 'clean', 'mean': r['mean'], 'std': r['std'],
                         'ms_per_action': r['ms_per_action'], 'box_width': float('nan'),
                         'ball_width': float('nan'), 'episodes': args.episodes})
            print(f"  [seed {seed}] {label+'(greedy readout)':<17} {'clean':<15} "
                  f"{r['mean']:8.2f} +/- {r['std']:6.2f}", flush=True)

    if getattr(args, 'ckpt_dir', ''):
        os.makedirs(args.ckpt_dir, exist_ok=True)
        for label, net in nets.items():
            torch.save(net.state_dict(), os.path.join(args.ckpt_dir, f'{label}_s{seed}.pt'))
    print(f'  [seed {seed}] done in {time.time()-t_start:.0f}s', flush=True)
    return rows


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarize(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                for k in ('mean', 'std', 'ms_per_action', 'box_width', 'ball_width', 'eps'):
                    r[k] = float(r[k])
                r['seed'] = int(r['seed'])
                rows.append(r)
    if not rows:
        print('no rows')
        return

    for eps in sorted({r['eps'] for r in rows}):
        sub = [r for r in rows if r['eps'] == eps]
        variants = [v for v in ALL_VARIANTS + ['pessimistic_box_greedyreadout',
                                               'pessimistic_ball_greedyreadout']
                    if any(r['variant'] == v for r in sub)]
        conds = [c for c in ALL_CONDITIONS if any(r['condition'] == c for r in sub)]
        seeds = sorted({r['seed'] for r in sub})
        print(f'\n=== eps={eps}  seeds={seeds} ===')
        print(f'{"condition":<16}' + ''.join(f'{v:>22}' for v in variants))
        for c in conds:
            line = f'{c:<16}'
            for v in variants:
                vals = [r['mean'] for r in sub if r['variant'] == v and r['condition'] == c]
                line += f'{np.mean(vals):>14.2f}+/-{np.std(vals):<7.2f}' if vals else f'{"-":>22}'
            print(line)
        print(f'{"ms/action":<16}' + ''.join(
            f'{np.mean([r["ms_per_action"] for r in sub if r["variant"] == v]):>22.3f}'
            for v in variants))
        for v in variants:
            bw = [r['box_width'] for r in sub if r['variant'] == v and not np.isnan(r['box_width'])]
            ba = [r['ball_width'] for r in sub if r['variant'] == v and not np.isnan(r['ball_width'])]
            if bw:
                print(f'  belief width {v:<20} {np.mean(bw):.4f}  (raw ball {np.mean(ba):.4f}, '
                      f'{100*np.mean(bw)/np.mean(ba):.1f}% of ball)')
        print('\n  per-seed:')
        for c in conds:
            for s in seeds:
                vals = {v: [r['mean'] for r in sub
                            if r['variant'] == v and r['condition'] == c and r['seed'] == s]
                        for v in variants}
                txt = '  '.join(f'{v}={vals[v][0]:.2f}' for v in variants if vals[v])
                print(f'    {c:<15} seed {s}: {txt}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--vanilla-steps', type=int, default=120_000)
    p.add_argument('--finetune-steps', type=int, default=40_000)
    p.add_argument('--attacker-steps', type=int, default=40_000)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--eps', type=float, default=0.35)
    p.add_argument('--particles', type=int, default=3)
    p.add_argument('--attack-levels', type=int, default=5,
                   help='per-axis perturbation levels for the trained attacker (>=3, odd)')
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
    p.add_argument('--variants', nargs='+', default=ALL_VARIANTS, choices=ALL_VARIANTS)
    p.add_argument('--conditions', nargs='+', default=ALL_CONDITIONS, choices=ALL_CONDITIONS)
    p.add_argument('--attacker-dir', type=str, default='')
    p.add_argument('--ckpt-dir', type=str, default='',
                   help='if set, save defender state_dicts after training (does not change the run)')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--out', type=str, default='')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--summarize', nargs='+', default=None)
    args = p.parse_args()

    if args.summarize:
        summarize(args.summarize)
        return
    if not args.out:
        raise SystemExit('--out is required')
    if os.path.exists(args.out):
        raise SystemExit(f'refusing to overwrite {args.out}')

    torch.set_num_threads(max(1, args.threads))
    print(f'eps={args.eps} variants={args.variants} conditions={args.conditions} '
          f'vanilla={args.vanilla_steps} ft={args.finetune_steps} att={args.attacker_steps} '
          f'episodes={args.episodes} seeds={args.seeds}', flush=True)

    rows = []
    for s in args.seeds:
        rows.extend(run_seed(s, args))
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f'wrote {args.out}', flush=True)
    summarize([args.out])
    print('PROD_RUN_DONE', flush=True)


if __name__ == '__main__':
    main()

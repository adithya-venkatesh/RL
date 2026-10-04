#!/usr/bin/env python3
"""Clean vs attacked belief size under tabular maximin test.

Hypothesis: without an attacker the belief does not contract, so maximin
episodes run to the 1000-step cap.

Per episode we record |M ∩ B_eps(s_tilde)| (the intersection used before the
belief is propagated), plus checks for a reset bug, an empty intersection, and
whether the true state sits in that intersection.

CSV: gridworld_clean_maximin_belief_v3.csv (append/resume by seed×condition).
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

OUT = "gridworld_clean_maximin_belief_v3.csv"
FIELDS = [
    "seed",
    "q_table",
    "condition",
    "episode",
    "mean_intersect_size",
    "final_intersect_size",
    "mean_belief_size",
    "final_belief_size",
    "belief_at_start",
    "first_intersect_size",
    "episode_length",
    "truncated",
    "return",
    "frac_empty_intersect",
    "frac_true_in_intersect",
    "frac_stilde_equals_true",
    "mean_intersect_around_true",
    "final_propagated_size",
]


def _completed(path: str) -> set:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            done.add((int(float(row["seed"])), row["q_table"], row["condition"]))
    return done


def _rollout(agent, seed: int, attack_model, trials: int = 100):
    """Same loop as play_att_with_belief(learn=False), with per-step sizes."""
    from main_with_vanilla import GridWorld, states_intersection
    import copy

    env = GridWorld()
    env.seed(seed)
    np.random.seed(seed + 1000)
    pol0 = copy.deepcopy(agent.defender_policy)
    episodes = []
    for ep in range(trials):
        belief = set(agent.all_state)
        belief_at_start = len(belief)
        agent.defender_policy = agent.get_policy_with_belief(pol0, belief)
        env.reset()
        step = 0
        ret = 0.0
        game_over = False
        sizes = []
        true_hits = []
        empty = []
        same = []
        true_ball_hits = []
        first = None
        while step < 1000 and not game_over:
            true_state = tuple(env.current_location)
            if attack_model is not None:
                att_action = attack_model.predict(np.asarray(true_state))
                idx = att_action[0]
                if idx in (0, 1, 2, 3, 4):
                    true_action = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"][int(idx)]
                else:
                    true_action = idx
                att_state = env.take_action(list(true_state), true_action)
            else:
                att_state = agent.get_att_state(true_state, agent.defender_policy)
            att_state = tuple(att_state)
            action = agent.choose_action_att(env.actions, att_state, 0)
            ball = agent.get_ball(att_state)
            inter = states_intersection(belief, ball)
            inter_true = states_intersection(belief, agent.get_ball(true_state))
            if first is None:
                first = len(inter)
            sizes.append(len(inter))
            true_ball_hits.append(len(inter_true))
            empty.append(len(inter) == 0)
            true_hits.append(true_state in inter)
            same.append(att_state == true_state)
            belief_before = len(belief)
            # mirror the implemented update: propagate the intersection
            nxt = set()
            for st in inter:
                nxt.add(env.take_action(st, action))
            belief = nxt
            agent.defender_policy = agent.get_policy_with_belief(agent.defender_policy, belief)
            _, reward, terminated, truncated, _ = env.step(action)
            ret += float(reward)
            step += 1
            if env.check_state() == "TERMINAL":
                env.__init__()
                game_over = True
            _ = belief_before  # kept so a reset-vs-carry bug is visible via belief_at_start
        episodes.append({
            "mean_intersect_size": float(np.mean(sizes)) if sizes else 0.0,
            "final_intersect_size": float(sizes[-1]) if sizes else 0.0,
            "mean_belief_size": float(np.mean(sizes)) if sizes else 0.0,
            "final_belief_size": float(sizes[-1]) if sizes else 0.0,
            "final_propagated_size": float(len(belief)),
            "belief_at_start": belief_at_start,
            "first_intersect_size": float(first if first is not None else 0),
            "episode_length": step,
            "truncated": int(step >= 1000 and not game_over),
            "return": ret,
            "frac_empty_intersect": float(np.mean(empty)) if empty else 0.0,
            "frac_true_in_intersect": float(np.mean(true_hits)) if true_hits else 0.0,
            "frac_stilde_equals_true": float(np.mean(same)) if same else 0.0,
            "mean_intersect_around_true": float(np.mean(true_ball_hits)) if true_ball_hits else 0.0,
        })
    return episodes


def run_seed(seed: int, trials: int, attacker_stem: str) -> list:
    import copy
    from diag_tabular_v3_common import train_defenders
    from stable_baselines3 import DQN

    t0 = time.perf_counter()
    _, agent_p, agent_o, _ = train_defenders(seed, scoring="fixed")
    atk = DQN.load(attacker_stem)
    from gridworld_target_test_swap_tabular_v3 import _StayAttacker

    jobs = [
        ("pes", "clean", agent_p, _StayAttacker()),
        ("pbgt", "clean", agent_o, _StayAttacker()),
        ("pes", "attacked", agent_p, atk),
    ]
    rows = []
    for qname, cond, agent, model in jobs:
        # policy snapshot is inside _rollout via the agent's current policy
        agent.defender_policy = copy.deepcopy(agent.defender_policy)
        eps = _rollout(agent, seed, model, trials=trials)
        for i, ep in enumerate(eps):
            rows.append({"seed": seed, "q_table": qname, "condition": cond, "episode": i, **ep})
    print(
        f"seed {seed} wall {time.perf_counter()-t0:.1f}s "
        f"trunc_pes_clean {np.mean([r['truncated'] for r in rows if r['q_table']=='pes' and r['condition']=='clean']):.2f}",
        flush=True,
    )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=50)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--attacker", default="attacker_pessimistic_final_v3_0")
    p.add_argument("--out", default=OUT)
    p.add_argument("--probe-only", action="store_true")
    args = p.parse_args()
    if os.path.exists(args.out) and not args.out.endswith("_v3.csv"):
        raise SystemExit(f"refusing non-_v3 path {args.out}")

    done = _completed(args.out)
    missing = [s for s in range(args.n_seeds) if ("pes", "clean") not in {(b, c) for (ss, b, c) in done if ss == s}]
    # simpler missing test
    missing = []
    for s in range(args.n_seeds):
        need = {(s, "pes", "clean"), (s, "pbgt", "clean"), (s, "pes", "attacked")}
        if not need.issubset(done):
            missing.append(s)
    print(f"missing {len(missing)} / {args.n_seeds}", flush=True)
    if not missing:
        print("BELIEF_DIAG_DONE nothing to do", flush=True)
        return
    if args.probe_only:
        missing = missing[:1]

    file_exists = os.path.exists(args.out) and os.path.getsize(args.out) > 0
    t0 = time.perf_counter()
    workers = 1 if args.probe_only else args.workers
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        if not file_exists:
            w.writeheader()
        if workers == 1:
            for s in missing:
                for row in run_seed(s, args.trials, args.attacker):
                    w.writerow(row)
                f.flush()
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(run_seed, s, args.trials, args.attacker): s for s in missing}
                for fut in as_completed(futs):
                    for row in fut.result():
                        w.writerow(row)
                    f.flush()
    wall = time.perf_counter() - t0
    print(f"BELIEF_DIAG_PHASE wall_min={wall/60:.2f} seeds={len(missing)} workers={workers}", flush=True)
    if args.probe_only and missing:
        est = wall / len(missing) * args.n_seeds / max(args.workers, 1)
        print(f"PROBE_EST_FULL_SEC {est:.0f} ({est/3600:.2f} h at {args.workers} workers)", flush=True)


if __name__ == "__main__":
    main()

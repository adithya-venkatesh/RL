#!/usr/bin/env python3
"""SB3 attackers matched to the swapped tabular test rules.

Same budget and DQN path as gridworld_final_v3_protocol.train_one_attacker
(200_000 steps). Each of the three attackers for a cell is trained against a
different defender seed (99999, 99998, 99997) and its own attacker RNG.

Cells:
  pes Q (maximin Bellman target) + GREEDY test  -> agent_type offpolicy
  PB-GT Q (greedy Bellman target) + MAXIMIN test -> agent_type pessimistic

Does not overwrite the original nine final_v3 zips.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from gridworld_final_v3_protocol import DEFENDER_SEEDS, TIMESTEPS

MAPPING = "attacker_swap_matched_v3_mapping.json"

CELLS = {
    # greedy test on the pessimistic Q-table
    "pes_q_greedy": {
        "agent_type": "offpolicy",
        "rng": [14001, 14002, 14003],
        "stem": "attacker_pes_q_greedy_final_v3",
    },
    # maximin test on the PB-GT Q-table
    "pbgt_q_maximin": {
        "agent_type": "pessimistic",
        "rng": [15001, 15002, 15003],
        "stem": "attacker_pbgt_q_maximin_final_v3",
    },
}


def _prep(cell: str, defender_seed: int):
    from main_with_vanilla import (
        GridWorld,
        Q_Agent,
        play,
        play_att_with_belief,
        play_offpolicy_greedy_target,
    )

    env = GridWorld()
    env.seed(defender_seed)
    np.random.seed(defender_seed)
    agent = Q_Agent(env)
    play(env, agent, trials=2000, learn=True)
    if cell == "pes_q_greedy":
        play_att_with_belief(env, agent, trials=500, learn=True)
    elif cell == "pbgt_q_maximin":
        play_offpolicy_greedy_target(env, agent, trials=500, learn=True)
    else:
        raise ValueError(cell)
    return agent


def train_one(cell: str, a_idx: int) -> float:
    from main_with_vanilla import GridWorld_RL_Attack

    spec = CELLS[cell]
    stem = f"{spec['stem']}_{a_idx}"
    out_zip = stem + ".zip"
    if os.path.exists(out_zip):
        print(f"SKIP existing {out_zip}", flush=True)
        return 0.0
    defender_seed = DEFENDER_SEEDS[a_idx]
    atk_seed = spec["rng"][a_idx]
    print(
        f"=== Train {stem}: defender_seed={defender_seed} "
        f"attacker_rng={atk_seed} test={spec['agent_type']} ===",
        flush=True,
    )
    t0 = time.perf_counter()
    agent = _prep(cell, defender_seed)
    ckpt_dir = f"RL_attack/{cell}_final_v3_a{a_idx}"
    os.makedirs(ckpt_dir, exist_ok=True)
    atk_env = GridWorld_RL_Attack(agent, agent_type=spec["agent_type"])
    callback = CheckpointCallback(save_freq=5000, save_path=ckpt_dir, name_prefix=stem)
    np.random.seed(atk_seed)
    model = DQN(
        "MlpPolicy",
        atk_env,
        learning_rate=1e-3,
        learning_starts=1000,
        batch_size=256,
        exploration_final_eps=0.001,
        gamma=1,
        tau=0.2,
        train_freq=(5, "step"),
        policy_kwargs={"net_arch": [256, 128]},
        tensorboard_log=f"{ckpt_dir}/",
        verbose=0,
        seed=atk_seed,
    )
    print(f"Training {stem} for {TIMESTEPS} steps...", flush=True)
    model.learn(total_timesteps=TIMESTEPS, callback=callback)
    model.save(stem)
    secs = time.perf_counter() - t0
    print(f"Saved {out_zip} in {secs/60:.1f} min", flush=True)
    return secs


def main() -> None:
    mapping = {"defenders": {}, "wall_sec_total": 0.0}
    if os.path.exists(MAPPING):
        try:
            with open(MAPPING) as f:
                prev = json.load(f)
            if isinstance(prev.get("defenders"), dict):
                mapping["defenders"] = prev["defenders"]
        except (json.JSONDecodeError, OSError):
            pass
    t_all = time.perf_counter()
    for cell, spec in CELLS.items():
        for a_idx in range(3):
            stem = f"{spec['stem']}_{a_idx}"
            secs = train_one(cell, a_idx)
            mapping["defenders"][stem] = {
                "cell": cell,
                "agent_type_test": spec["agent_type"],
                "attacker_index": a_idx,
                "defender_seed": DEFENDER_SEEDS[a_idx],
                "attacker_rng_seed": spec["rng"][a_idx],
                "zip": stem + ".zip",
                "timesteps": TIMESTEPS,
                "train_wall_sec": secs,
                "skipped_existing": secs == 0.0 and os.path.exists(stem + ".zip"),
            }
            with open(MAPPING, "w") as f:
                json.dump(mapping, f, indent=2)
    mapping["wall_sec_total"] = time.perf_counter() - t_all
    with open(MAPPING, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"SWAP_ATTACKERS_DONE wall_min={mapping['wall_sec_total']/60:.1f}", flush=True)


if __name__ == "__main__":
    main()

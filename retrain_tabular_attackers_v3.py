#!/usr/bin/env python3
"""Retrain tabular SB3 attackers against fixed get_att_state defenders.

Matches Option 4 in main_with_vanilla.py:
  ATTACKER_TRAINING_SEED = 99999
  pes defender: seed 99999+1; PB-GT defender: seed 99999
  DQN: 2e5 steps, net_arch [256,128], lr 1e-3, gamma=1

Saves attacker_pessimistic_v3.zip and attacker_offpolicy_v3.zip.
Does NOT overwrite attacker_pessimistic.zip / attacker_offpolicy.zip.
"""
from __future__ import annotations

import os
import time

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from main_with_vanilla import (
    GridWorld,
    GridWorld_RL_Attack,
    Q_Agent,
    play,
    play_att_with_belief,
    play_offpolicy_greedy_target,
)

ATTACKER_TRAINING_SEED = 99999
TIMESTEPS = 200_000


def _train_attacker(agentQ, agent_type: str, out_stem: str) -> None:
    out_zip = out_stem + ".zip"
    if os.path.exists(out_zip):
        raise SystemExit(f"Refusing to overwrite existing {out_zip}")

    ckpt_dir = f"RL_attack/{agent_type}_v3"
    os.makedirs(ckpt_dir, exist_ok=True)
    env = GridWorld_RL_Attack(agentQ, agent_type=agent_type)
    callback = CheckpointCallback(
        save_freq=5000,
        save_path=ckpt_dir,
        name_prefix=out_stem,
    )
    model = DQN(
        "MlpPolicy",
        env,
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
    )
    t0 = time.perf_counter()
    print(f"Training {out_stem} for {TIMESTEPS} steps...", flush=True)
    model.learn(total_timesteps=TIMESTEPS, callback=callback)
    model.save(out_stem)
    print(f"Saved {out_zip} in {(time.perf_counter()-t0)/60:.1f} min", flush=True)


def main() -> None:
    for stem in ("attacker_pessimistic_v3", "attacker_offpolicy_v3"):
        if os.path.exists(stem + ".zip"):
            raise SystemExit(f"Refusing to overwrite existing {stem}.zip")

    t_all = time.perf_counter()

    print("=== Train pessimistic defender (seed 99999+1) ===", flush=True)
    env2 = GridWorld()
    env2.seed(ATTACKER_TRAINING_SEED + 1)
    np.random.seed(ATTACKER_TRAINING_SEED + 1)
    agent2 = Q_Agent(env2)
    play(env2, agent2, trials=2000, learn=True)
    play_att_with_belief(env2, agent2, trials=500, learn=True)
    print("Pessimistic defender ready.", flush=True)

    print("=== Train PB-GT / offpolicy defender (seed 99999) ===", flush=True)
    env1 = GridWorld()
    env1.seed(ATTACKER_TRAINING_SEED)
    np.random.seed(ATTACKER_TRAINING_SEED)
    agent1 = Q_Agent(env1)
    play(env1, agent1, trials=2000, learn=True)
    play_offpolicy_greedy_target(env1, agent1, trials=500, learn=True)
    print("PB-GT defender ready.", flush=True)

    _train_attacker(agent2, "pessimistic", "attacker_pessimistic_v3")
    _train_attacker(agent1, "offpolicy", "attacker_offpolicy_v3")

    print(f"ALL_ATTACKERS_V3_DONE wall_min={(time.perf_counter()-t_all)/60:.1f}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""5-seed confirmation of the sound filtered-box belief. NO fine-tuning.

For each seed the *same* untouched ``checkpoints/seed_{s}/vanilla.pt`` Q-network
is evaluated under four decision rules, so every difference is attributable to
the belief set / inner-min alone:

    greedy      argmax_a Q(s_tilde, a)                        [vanilla baseline]
    uniform     maximin over UniformBallBelief particles      [current deep rule]
    sound       maximin over the filter-then-propagate box    [the fix]
    sound_ibp   maximin over a certified IBP inner min        [certified variant]

Conditions: clean, shared random delta, PGD*(32 candidates), matched SB3 attacker
trained per (rule, seed). Each defender/attacker gets its own belief instance so
filter state cannot leak between training and evaluation.

Artifacts: cartpole_sound_5seeds.csv (new name).
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
from stable_baselines3 import DQN as SB3_DQN

from cartpole_belief import UniformBallBelief
from cartpole_sound_belief import SoundFilteredBelief
from cartpole_sound_eval_harness import (
    attacker_is_vacuous,
    load_result_rows,
    train_attacker_with_diag,
    truly_clean_maximin,
    write_result_rows,
)
from main_cartpole_dqn import (
    DEFAULT_ATTACK_EPS,
    DEFAULT_BELIEF_KAPPA,
    DQNAgent,
    DQNConfig,
    evaluate_agent,
    evaluate_pgd_worst_case,
    evaluate_shared_random_attack,
)

torch.set_num_threads(1)

OUT_CSV = "cartpole_sound_5seeds.csv"
SEEDS = [0, 1, 2, 3, 4]
EVAL_EP = 50
ATK_STEPS = 40_000
EPS = DEFAULT_ATTACK_EPS
KAPPA = DEFAULT_BELIEF_KAPPA

RULES = [
    # name, eval mode, provider factory, box_min
    ("greedy", "vanilla", lambda: None, "particles"),
    ("uniform", "pessimistic", lambda: UniformBallBelief(EPS, KAPPA), "particles"),
    ("sound", "pessimistic", lambda: SoundFilteredBelief(EPS, KAPPA), "particles"),
    ("sound_ibp", "pessimistic", lambda: SoundFilteredBelief(EPS, KAPPA), "ibp"),
]


def _load(seed: int, box_min: str) -> DQNAgent:
    cfg = DQNConfig()
    cfg.box_min = box_min
    agent = DQNAgent(config=cfg)
    agent.load(f"checkpoints/seed_{seed}/vanilla.pt")
    for p in agent.q_net.parameters():
        p.requires_grad = False
    return agent


def run_cell(seed: int, name: str, mode: str, factory, box_min: str) -> Dict[str, Any]:
    agent = _load(seed, box_min)
    row: Dict[str, Any] = {"seed": seed, "rule": name}
    t0 = time.time()

    if mode == "vanilla":
        row["clean"] = float(np.mean(evaluate_agent(agent, EVAL_EP, seed + 1000, "vanilla")[0]))
    else:
        row["clean"] = float(np.mean(truly_clean_maximin(agent, factory(), EVAL_EP, seed + 1000)))

    row["shared"] = float(np.mean(
        evaluate_shared_random_attack(agent, EVAL_EP, seed + 6000, mode, belief_provider=factory())[0]
    ))
    row["pgd"] = float(np.mean(
        evaluate_pgd_worst_case(agent, EVAL_EP, seed + 7000, mode, belief_provider=factory())
    ))

    path = f"checkpoints/seed_{seed}/attacker_sound5s_{name}"
    diag = train_attacker_with_diag(
        agent, mode, ATK_STEPS, path,
        belief_provider=factory(), seed=seed + 9000,
    )
    attacker = SB3_DQN.load(path + ".zip")
    row["matched"] = float(np.mean(
        evaluate_agent(agent, EVAL_EP, seed + 8000, mode, attacker=attacker,
                       belief_provider=factory())[0]
    ))
    row.update(diag)
    row["matched_vacuous"] = int(attacker_is_vacuous(row["matched"], diag))
    row["secs"] = round(time.time() - t0, 1)

    print(
        f"  {name:<10} clean={row['clean']:6.1f} shared={row['shared']:6.1f} "
        f"pgd={row['pgd']:6.1f} matched={row['matched']:6.1f}"
        f"{' [VACUOUS]' if row['matched_vacuous'] else ''}  "
        f"atk_len min/last10={diag['atk_len_min']:.0f}/{diag['atk_len_last10']:.0f}  "
        f"({row['secs']:.0f}s)",
        flush=True,
    )
    return row


def summarize(rows: List[Dict[str, Any]]) -> None:
    print("\n===== 5-SEED MEAN +/- STD (eps=%.3f, %d ep/cell, no fine-tuning) =====" % (EPS, EVAL_EP))
    print(f"{'rule':<10} {'clean':>14} {'shared':>14} {'pgd':>14} {'matched':>14}  vacuous")
    for name, _, _, _ in RULES:
        rs = [r for r in rows if r["rule"] == name]
        if not rs:
            continue
        cells = []
        for cond in ("clean", "shared", "pgd", "matched"):
            a = np.asarray([r[cond] for r in rs], dtype=np.float64)
            cells.append(f"{a.mean():7.1f}+/-{a.std():5.1f}")
        nvac = sum(r["matched_vacuous"] for r in rs)
        print(f"{name:<10} " + " ".join(f"{c:>14}" for c in cells)
              + f"  {nvac}/{len(rs)} seeds", flush=True)


def main() -> None:
    missing = [s for s in SEEDS if not os.path.exists(f"checkpoints/seed_{s}/vanilla.pt")]
    if missing:
        sys.exit(f"missing vanilla checkpoints for seeds {missing}")
    print(f"all {len(SEEDS)} vanilla checkpoints present", flush=True)

    rows: List[Dict[str, Any]] = load_result_rows(OUT_CSV)
    done = {(int(r["seed"]), r["rule"]) for r in rows}
    if done:
        print(f"resuming {OUT_CSV}: {len(done)} cells already on disk", flush=True)
    t_all = time.time()
    for seed in SEEDS:
        print(f"\n======== SEED {seed} ========", flush=True)
        for name, mode, factory, box_min in RULES:
            if (seed, name) in done:
                print(f"  {name:<10} SKIP (already in csv)", flush=True)
                continue
            rows.append(run_cell(seed, name, mode, factory, box_min))
            done.add((seed, name))
            write_result_rows(OUT_CSV, rows)
        summarize(rows)

    summarize(rows)
    print(f"\nwrote {OUT_CSV}  total wall time {(time.time()-t_all)/60:.1f} min", flush=True)
    print("SOUND_5SEEDS_DONE", flush=True)


if __name__ == "__main__":
    main()

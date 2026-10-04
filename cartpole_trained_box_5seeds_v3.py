#!/usr/bin/env python3
"""5-seed trained sound-box CartPole (pes + PB-GT fine-tune). NEW files only.

Reuses checkpoints/seed_{0,1,2}/{pessimistic,offpolicy}_sound_tune_v3.pt when the
tune-round settings are unchanged (pes LR 1e-5, 5e4 cap, deployed-maximin
checkpoint including step 0, early stop). Trains seeds 3–4 under that same
recipe. Does not overwrite tune CSVs or those checkpoints.

Eval matches the frozen-Q harness (cartpole_sound_5seeds.py / Table 6):
50 episodes, eps=0.05, clean / shared-random / PGD*, then an SB3 DQN attacker
per (seed, method) at 40k steps. Shaped attackers (reward = |theta| + 0.1*|x|)
are a separate column, trained for every method only if a pessimistic matched
cell is vacuous (attacker never produced a terminal during training).

Artifacts:
  cartpole_trained_box_5seeds_v3.csv
  cartpole_trained_box_5seeds_v3.log          (stdout redirect)
  cartpole_trained_box_5seeds_curves_v3.csv
  checkpoints/seed_{3,4}/pessimistic_trained_box_v3.pt
  checkpoints/seed_{3,4}/offpolicy_trained_box_v3.pt
  checkpoints/seed_{s}/attacker_trained_box_v3_{method}[.zip]
  checkpoints/seed_{s}/attacker_trained_box_v3_{method}_shaped[.zip]
  checkpoints/seed_{s}/trained_box_v3_meta.json
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from stable_baselines3 import DQN as SB3_DQN
from stable_baselines3.common.monitor import Monitor

from cartpole_sound_belief import DEFAULT_LATTICE_M, SoundFilteredBelief
from cartpole_sound_eval_harness import (
    VACUOUS_EP_LEN,
    load_result_rows,
    train_attacker_with_diag,
    truly_clean_maximin,
    write_result_rows,
)
from main_cartpole_dqn import (
    DEFAULT_ATTACK_EPS,
    CartPoleAttackEnv,
    CartPoleDefenderEnv,
    DQNAgent,
    DQNConfig,
    _note_belief_action,
    evaluate_agent,
    evaluate_pgd_worst_case,
    evaluate_shared_random_attack,
    set_seed,
    train_pbgt_sound_dqn,
    train_pessimistic_sound_dqn,
)

torch.set_num_threads(1)

OUT_CSV = "cartpole_trained_box_5seeds_v3.csv"
CURVE_CSV = "cartpole_trained_box_5seeds_curves_v3.csv"
TUNE_CSV = "cartpole_sound_train_tune_v3.csv"
SEEDS = [0, 1, 2, 3, 4]
REUSE_SEEDS = (0, 1, 2)
EVAL_EP = 50
INFER_EP = 20
ATK_STEPS = 40_000
PES_STEPS = 50_000
OFF_STEPS = 40_000
EVAL_INTERVAL = 5_000
PES_LR = 1e-5
OFF_LR = 1e-5
LATTICE_M = DEFAULT_LATTICE_M
KAPPA = LATTICE_M ** 4  # 16
EPS = DEFAULT_ATTACK_EPS
METHODS = ("vanilla", "pessimistic", "pbgt")


def _provider() -> SoundFilteredBelief:
    return SoundFilteredBelief(EPS, KAPPA, lattice_m=LATTICE_M)


def _pes_path(seed: int) -> str:
    if seed in REUSE_SEEDS:
        return f"checkpoints/seed_{seed}/pessimistic_sound_tune_v3.pt"
    return f"checkpoints/seed_{seed}/pessimistic_trained_box_v3.pt"


def _pbgt_path(seed: int) -> str:
    if seed in REUSE_SEEDS:
        return f"checkpoints/seed_{seed}/offpolicy_sound_tune_v3.pt"
    return f"checkpoints/seed_{seed}/offpolicy_trained_box_v3.pt"


def _meta_path(seed: int) -> str:
    return f"checkpoints/seed_{seed}/trained_box_v3_meta.json"


def _atk_path(seed: int, method: str, shaped: bool) -> str:
    tag = "shaped" if shaped else "unshaped"
    return f"checkpoints/seed_{seed}/attacker_trained_box_v3_{method}_{tag}"


def _load_agent(path: str) -> DQNAgent:
    cfg = DQNConfig(attack_epsilon=EPS, belief_kappa=KAPPA)
    cfg.box_min = "particles"
    agent = DQNAgent(config=cfg)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    agent.load(path)
    return agent


def _load_vanilla(seed: int) -> DQNAgent:
    return _load_agent(f"checkpoints/seed_{seed}/vanilla.pt")


def _append_curves(seed: int, curve: List[Dict[str, float]]) -> None:
    if not curve:
        return
    write_header = not os.path.exists(CURVE_CSV)
    with open(CURVE_CSV, "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["seed", "step", "greedy_clean", "deployed_maximin_clean"],
        )
        if write_header:
            w.writeheader()
        for pt in curve:
            w.writerow(
                {
                    "seed": seed,
                    "step": int(pt["step"]),
                    "greedy_clean": float(pt["greedy_clean"]),
                    "deployed_maximin_clean": float(pt["maximin_clean"]),
                }
            )


def _seed_curves_present(seed: int) -> bool:
    if not os.path.exists(CURVE_CSV):
        return False
    with open(CURVE_CSV, newline="") as f:
        return any(row.get("seed") == str(seed) for row in csv.DictReader(f))


def _never_terminal(diag: Dict[str, float]) -> bool:
    """Vacuous iff training never produced a terminal (no short episode)."""
    n = float(diag.get("atk_n_train_eps", 0.0) or 0.0)
    lmin = float(diag.get("atk_len_min", float("nan")))
    if n <= 0 or not np.isfinite(lmin):
        return True
    return bool(lmin >= VACUOUS_EP_LEN)


def _tune_meta(seed: int) -> Dict[str, Any]:
    with open(TUNE_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    hit = [r for r in rows if int(r["seed"]) == seed]
    if not hit:
        raise KeyError(f"seed {seed} missing from {TUNE_CSV}")
    r = hit[0]
    return {
        "pes_selected_step": int(float(r["pes_selected_step"])),
        "pes_step0_selected": int(float(r["pes_step0_selected"])),
        "pes_early_stopped": int(float(r["pes_early_stopped"])),
        "pes_stopped_step": int(float(r["pes_stopped_step"])),
        "pes_coverage_min": float(r["pes_coverage_min"]),
        "pbgt_coverage_min": float(r["pbgt_coverage_min"]),
        "pes_ft_min": float(r["pes_ft_min"]),
        "pbgt_ft_min": float(r["pbgt_ft_min"]),
        "pes_best_maximin": float(r["pes_best_maximin_meta"]),
        "curve": json.loads(r["curve_json"]),
        "ckpt_reused": 1,
        "settings_note": "reused sound_tune_v3 (LR 1e-5, cap 5e4, deployed-maximin incl step0, early stop)",
    }


def _save_meta(seed: int, meta: Dict[str, Any]) -> None:
    payload = dict(meta)
    with open(_meta_path(seed), "w") as f:
        json.dump(payload, f)


def _load_saved_meta(seed: int) -> Optional[Dict[str, Any]]:
    path = _meta_path(seed)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def ensure_finetune(seed: int) -> Dict[str, Any]:
    """Train seeds 3–4, or load identical tune checkpoints for seeds 0–2."""
    saved = _load_saved_meta(seed)
    if (
        saved is not None
        and os.path.exists(_pes_path(seed))
        and os.path.exists(_pbgt_path(seed))
    ):
        print(f"  finetune SKIP (checkpoints + meta present) reused={saved.get('ckpt_reused')}", flush=True)
        if not _seed_curves_present(seed) and saved.get("curve"):
            _append_curves(seed, saved["curve"])
        return saved

    if seed in REUSE_SEEDS:
        if not (os.path.exists(_pes_path(seed)) and os.path.exists(_pbgt_path(seed))):
            raise FileNotFoundError(f"expected tune checkpoints for seed {seed}")
        meta = _tune_meta(seed)
        _save_meta(seed, meta)
        if not _seed_curves_present(seed):
            _append_curves(seed, meta["curve"])
        s0 = meta["pes_step0_selected"]
        print(
            f"  REUSE tune ckpts step={meta['pes_selected_step']}"
            f"{' *** STEP-0 ***' if s0 else ''}  "
            f"cover pes/pbgt={meta['pes_coverage_min']:.3f}/{meta['pbgt_coverage_min']:.3f}",
            flush=True,
        )
        return meta

    vanilla = _load_vanilla(seed)
    # Force the tune recipe onto the cloned nets (saved vanilla config is irrelevant).
    set_seed(seed)
    pes_partial = saved if (saved and os.path.exists(_pes_path(seed)) and saved.get("pes_done")) else None
    if pes_partial is None:
        pes = DQNAgent(config=DQNConfig(attack_epsilon=EPS, belief_kappa=KAPPA))
        pes.clone_weights_from(vanilla)
        pes.prepare_robust_finetune(lr=PES_LR, freeze_backbone=True)
        t0 = time.time()
        _, pes_run = train_pessimistic_sound_dqn(
            pes,
            _provider(),
            total_timesteps=PES_STEPS,
            seed=seed + 2000,
            eval_interval=EVAL_INTERVAL,
            eval_episodes=15,
            assert_coverage=True,
        )
        pes_mins = (time.time() - t0) / 60.0
        pes.save(_pes_path(seed))
        curve = pes_run.get("deployed_clean_curve", [])
        _append_curves(seed, curve)
        ckpt_step = int(pes_run["best_step"])
        step0 = int(bool(pes_run.get("step0_selected", ckpt_step == 0)))
        pes_partial = {
            "pes_done": 1,
            "pes_selected_step": ckpt_step,
            "pes_step0_selected": step0,
            "pes_early_stopped": int(bool(pes_run.get("early_stopped", False))),
            "pes_stopped_step": int(pes_run.get("stopped_step", PES_STEPS)),
            "pes_coverage_min": float(pes_run["coverage_min"]),
            "pes_ft_min": round(pes_mins, 2),
            "pes_best_maximin": float(pes_run["best_maximin_eval"]),
            "curve": curve,
            "ckpt_reused": 0,
        }
        _save_meta(seed, pes_partial)
        print(
            f"  pes FT {pes_mins:.1f}m selected_step={ckpt_step}"
            f"{' *** STEP-0 ***' if step0 else ''}  "
            f"best_deployed={pes_run['best_maximin_eval']:.1f}  "
            f"cover_min={pes_run['coverage_min']:.3f} -> {_pes_path(seed)}",
            flush=True,
        )
    else:
        print(
            f"  pes FT SKIP (checkpoint present, step={pes_partial['pes_selected_step']})",
            flush=True,
        )
        if not _seed_curves_present(seed) and pes_partial.get("curve"):
            _append_curves(seed, pes_partial["curve"])

    off = DQNAgent(config=DQNConfig(attack_epsilon=EPS, belief_kappa=KAPPA))
    off.clone_weights_from(vanilla)
    off.prepare_offpolicy_finetune(lr=OFF_LR, freeze_backbone=True)
    t0 = time.time()
    _, off_meta = train_pbgt_sound_dqn(
        off,
        _provider(),
        total_timesteps=OFF_STEPS,
        seed=seed + 4000,
        assert_coverage=True,
    )
    off_mins = (time.time() - t0) / 60.0
    off.save(_pbgt_path(seed))
    print(
        f"  pbgt FT {off_mins:.1f}m cover_min={off_meta['coverage_min']:.3f} -> {_pbgt_path(seed)}",
        flush=True,
    )
    meta = {
        "pes_done": 1,
        "pbgt_done": 1,
        "pes_selected_step": int(pes_partial["pes_selected_step"]),
        "pes_step0_selected": int(pes_partial["pes_step0_selected"]),
        "pes_early_stopped": int(pes_partial["pes_early_stopped"]),
        "pes_stopped_step": int(pes_partial["pes_stopped_step"]),
        "pes_coverage_min": float(pes_partial["pes_coverage_min"]),
        "pbgt_coverage_min": float(off_meta["coverage_min"]),
        "pes_ft_min": pes_partial["pes_ft_min"],
        "pbgt_ft_min": round(off_mins, 2),
        "pes_best_maximin": float(pes_partial["pes_best_maximin"]),
        "curve": pes_partial.get("curve", []),
        "ckpt_reused": 0,
        "settings_note": "trained with tune recipe (LR 1e-5, cap 5e4, deployed-maximin incl step0, early stop)",
    }
    _save_meta(seed, meta)
    return meta


def _agents(seed: int) -> Dict[str, DQNAgent]:
    return {
        "vanilla": _load_vanilla(seed),
        "pessimistic": _load_agent(_pes_path(seed)),
        "pbgt": _load_agent(_pbgt_path(seed)),
    }


def _belief_for(method: str):
    if method == "pessimistic":
        return _provider()
    return None


def _mode(method: str) -> str:
    return "pessimistic" if method == "pessimistic" else "vanilla"


def eval_table6(seed: int) -> Dict[str, float]:
    agents = _agents(seed)
    out: Dict[str, float] = {}
    print("  eval clean / shared-random / PGD* ...", flush=True)
    for method, agent in agents.items():
        mode = _mode(method)
        if method == "pessimistic":
            clean = truly_clean_maximin(agent, _provider(), EVAL_EP, seed + 1000)
        else:
            clean = evaluate_agent(agent, EVAL_EP, seed + 1000, "vanilla")[0]
        shared = evaluate_shared_random_attack(
            agent, EVAL_EP, seed + 6000, mode, belief_provider=_belief_for(method)
        )[0]
        pgd = evaluate_pgd_worst_case(
            agent, EVAL_EP, seed + 7000, mode, belief_provider=_belief_for(method)
        )
        out[f"{method}_clean"] = float(np.mean(clean))
        out[f"{method}_shared"] = float(np.mean(shared))
        out[f"{method}_pgd"] = float(np.mean(pgd))
        print(
            f"    {method:<12} clean={out[f'{method}_clean']:.1f}  "
            f"shared={out[f'{method}_shared']:.1f}  pgd={out[f'{method}_pgd']:.1f}",
            flush=True,
        )
    return out


def _diag_path(stem: str) -> str:
    return stem + "_diag.json"


def _write_diag(stem: str, diag: Dict[str, float]) -> None:
    with open(_diag_path(stem), "w") as f:
        json.dump({k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in diag.items()}, f)


def _read_diag(stem: str) -> Optional[Dict[str, float]]:
    path = _diag_path(stem)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    out: Dict[str, float] = {}
    for k, v in raw.items():
        out[k] = float("nan") if v is None else float(v)
    return out


def _diag_from_row(row: Dict[str, Any], prefix: str) -> Optional[Dict[str, float]]:
    lmin = row.get(f"{prefix}_len_min", row.get(f"{prefix}_atk_len_min"))
    try:
        lmin_f = float(lmin)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(lmin_f):
        return None
    def _f(key: str) -> float:
        try:
            return float(row.get(key))
        except (TypeError, ValueError):
            return float("nan")
    return {
        "atk_n_train_eps": _f(f"{prefix}_n_eps") if f"{prefix}_n_eps" in row else _f(f"{prefix}_atk_n_eps"),
        "atk_len_min": lmin_f,
        "atk_len_mean": _f(f"{prefix}_len_mean") if f"{prefix}_len_mean" in row else _f(f"{prefix}_atk_len_mean"),
        "atk_len_last10": _f(f"{prefix}_len_last10") if f"{prefix}_len_last10" in row else _f(f"{prefix}_atk_len_last10"),
    }


def _matched_eval(agent: DQNAgent, method: str, attacker, seed: int) -> float:
    rets, _ = evaluate_agent(
        agent,
        EVAL_EP,
        seed + 8000,
        _mode(method),
        attacker=attacker,
        belief_provider=_belief_for(method),
    )
    return float(np.mean(rets))


def train_unshaped(seed: int, row: Dict[str, Any], on_progress=None) -> None:
    agents = _agents(seed)
    for method, agent in agents.items():
        key = f"{method}_matched"
        if row.get(f"{method}_matched_done"):
            print(f"  matched {method} SKIP", flush=True)
            continue
        path = _atk_path(seed, method, shaped=False)
        zip_path = path + ".zip"
        diag = _read_diag(path) if os.path.exists(zip_path) else None
        if diag is None and os.path.exists(zip_path):
            diag = _diag_from_row(row, f"{method}_atk")
        if os.path.exists(zip_path) and diag is None:
            print(f"  matched {method} zip has no diag; retraining", flush=True)
            os.remove(zip_path)
        elif diag is not None:
            print(f"  matched {method} attacker exists, eval only", flush=True)
        if diag is None:
            t0 = time.time()
            diag = train_attacker_with_diag(
                agent,
                _mode(method),
                ATK_STEPS,
                path,
                belief_provider=_belief_for(method),
                epsilon=EPS,
                seed=seed + 9000,
            )
            _write_diag(path, diag)
            print(
                f"  matched {method} train {(time.time()-t0)/60:.1f}m  "
                f"n_eps={diag['atk_n_train_eps']:.0f} len_min={diag['atk_len_min']}",
                flush=True,
            )
        attacker = SB3_DQN.load(zip_path)
        score = _matched_eval(agent, method, attacker, seed)
        vac = int(_never_terminal(diag))
        row[key] = score
        row[f"{method}_matched_vacuous"] = vac
        row[f"{method}_atk_n_eps"] = diag["atk_n_train_eps"]
        row[f"{method}_atk_len_min"] = diag["atk_len_min"]
        row[f"{method}_atk_len_mean"] = diag["atk_len_mean"]
        row[f"{method}_atk_len_last10"] = diag["atk_len_last10"]
        row[f"{method}_matched_done"] = 1
        flag = " [VACUOUS]" if vac else ""
        print(f"    {method:<12} matched={score:.1f}{flag}", flush=True)
        if on_progress is not None:
            on_progress()


class ShapedCartPoleAttackEnv(CartPoleAttackEnv):
    """Same attack MDP, but attacker reward is |theta| + 0.1*|x| (raw state)."""

    def step(self, action):
        obs, _reward, terminated, truncated, info = super().step(action)
        raw = self.inner.get_true_state()
        shaped = abs(float(raw[2])) + 0.1 * abs(float(raw[0]))
        return obs, float(shaped), terminated, truncated, info


def train_shaped_attacker(
    defender: DQNAgent,
    method: str,
    save_path: str,
    seed: int,
) -> Dict[str, float]:
    env = ShapedCartPoleAttackEnv(
        defender,
        agent_type=_mode(method),
        epsilon=EPS,
        belief_provider=_belief_for(method),
    )
    mon = Monitor(env)
    model = SB3_DQN(
        "MlpPolicy",
        mon,
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
        seed=seed,
    )
    model.learn(total_timesteps=ATK_STEPS)
    model.save(save_path)
    lens = np.asarray(mon.get_episode_lengths(), dtype=np.float64)
    mon.close()
    if lens.size == 0:
        return {
            "atk_n_train_eps": 0.0,
            "atk_len_mean": float("nan"),
            "atk_len_min": float("nan"),
            "atk_len_last10": float("nan"),
        }
    return {
        "atk_n_train_eps": float(lens.size),
        "atk_len_mean": float(lens.mean()),
        "atk_len_min": float(lens.min()),
        "atk_len_last10": float(lens[-10:].mean()),
    }


def train_shaped_all(seed: int, row: Dict[str, Any], on_progress=None) -> None:
    agents = _agents(seed)
    for method, agent in agents.items():
        if row.get(f"{method}_shaped_done"):
            print(f"  shaped {method} SKIP", flush=True)
            continue
        path = _atk_path(seed, method, shaped=True)
        zip_path = path + ".zip"
        diag = _read_diag(path) if os.path.exists(zip_path) else None
        if diag is None and os.path.exists(zip_path):
            diag = _diag_from_row(row, f"{method}_shaped_atk")
        if os.path.exists(zip_path) and diag is None:
            print(f"  shaped {method} zip has no diag; retraining", flush=True)
            os.remove(zip_path)
        elif diag is not None:
            print(f"  shaped {method} attacker exists, eval only", flush=True)
        if diag is None:
            t0 = time.time()
            diag = train_shaped_attacker(agent, method, path, seed=seed + 19000)
            _write_diag(path, diag)
            print(
                f"  shaped {method} train {(time.time()-t0)/60:.1f}m  "
                f"n_eps={diag['atk_n_train_eps']:.0f} len_min={diag['atk_len_min']}",
                flush=True,
            )
        attacker = SB3_DQN.load(zip_path)
        score = _matched_eval(agent, method, attacker, seed)
        vac = int(_never_terminal(diag))
        row[f"{method}_shaped"] = score
        row[f"{method}_shaped_vacuous"] = vac
        row[f"{method}_shaped_atk_n_eps"] = diag["atk_n_train_eps"]
        row[f"{method}_shaped_atk_len_min"] = diag["atk_len_min"]
        row[f"{method}_shaped_atk_len_mean"] = diag["atk_len_mean"]
        row[f"{method}_shaped_atk_len_last10"] = diag["atk_len_last10"]
        row[f"{method}_shaped_done"] = 1
        flag = " [VACUOUS]" if vac else ""
        print(f"    {method:<12} shaped={score:.1f}{flag}", flush=True)
        if on_progress is not None:
            on_progress()


def time_inference(seed: int) -> Dict[str, float]:
    """Single-process per-action latency, 20 clean episodes, threads=1."""
    agents = _agents(seed)
    out: Dict[str, float] = {}
    for method, agent in agents.items():
        set_seed(seed + 11000)
        env = CartPoleDefenderEnv()
        times: List[float] = []
        prov = _provider() if method == "pessimistic" else None
        for ep in range(INFER_EP):
            obs, _ = env.reset(seed=seed + 11000 + ep)
            if prov is not None:
                prov.reset_episode()
            done = False
            while not done:
                t0 = time.perf_counter()
                if method == "pessimistic":
                    action = agent.maximin_action(obs, EPS, KAPPA, belief_provider=prov)
                else:
                    action = agent.greedy_action(obs)
                times.append(time.perf_counter() - t0)
                _note_belief_action(prov, action)
                obs, _r, term, trunc, _ = env.step(action)
                done = term or trunc
        env.close()
        ms = np.asarray(times, dtype=np.float64) * 1000.0
        out[f"{method}_infer_ms_mean"] = float(ms.mean())
        out[f"{method}_infer_ms_std"] = float(ms.std(ddof=1)) if ms.size > 1 else 0.0
        out[f"{method}_infer_n"] = float(ms.size)
        print(
            f"  infer {method:<12} {out[f'{method}_infer_ms_mean']:.4f} ms "
            f"± {out[f'{method}_infer_ms_std']:.4f}  n={int(ms.size)}",
            flush=True,
        )
    return out


def _blank_row(seed: int, meta: Dict[str, Any]) -> Dict[str, Any]:
    cover_ok = int(
        float(meta["pes_coverage_min"]) >= 1.0 - 1e-12
        and float(meta["pbgt_coverage_min"]) >= 1.0 - 1e-12
    )
    return {
        "row_kind": "seed",
        "seed": seed,
        "ckpt_reused": int(meta["ckpt_reused"]),
        "pes_selected_step": int(meta["pes_selected_step"]),
        "pes_step0_selected": int(meta["pes_step0_selected"]),
        "pes_early_stopped": int(meta["pes_early_stopped"]),
        "pes_stopped_step": int(meta["pes_stopped_step"]),
        "pes_coverage_min": float(meta["pes_coverage_min"]),
        "pbgt_coverage_min": float(meta["pbgt_coverage_min"]),
        "coverage_assert_ok": cover_ok,
        "pes_ft_min": meta.get("pes_ft_min", ""),
        "pbgt_ft_min": meta.get("pbgt_ft_min", ""),
        "settings_note": meta.get("settings_note", ""),
        "shaped_ran": 0,
        "eps": EPS,
        "eval_episodes": EVAL_EP,
        "atk_steps": ATK_STEPS,
        "pes_lr": PES_LR,
        "lattice": KAPPA,
    }


def _is_seed_row(row: Dict[str, Any]) -> bool:
    if str(row.get("row_kind", "seed")) != "seed":
        return False
    try:
        int(row["seed"])
        return True
    except (TypeError, ValueError):
        return False


def _upsert(rows: List[Dict[str, Any]], row: Dict[str, Any]) -> List[Dict[str, Any]]:
    kept = [r for r in rows if not (_is_seed_row(r) and int(r["seed"]) == int(row["seed"]))]
    kept.append(row)
    kept.sort(key=lambda r: int(r["seed"]) if _is_seed_row(r) else 10_000)
    return kept


def _persist(rows: List[Dict[str, Any]]) -> None:
    write_result_rows(OUT_CSV, rows)


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.1f} ± {std:.1f}"


def summarize(rows: List[Dict[str, Any]]) -> None:
    seeds = [r for r in rows if _is_seed_row(r)]
    print("\n===== TRAINED BOX 5-SEED (ddof=1) =====", flush=True)
    for r in seeds:
        s0 = int(float(r["pes_step0_selected"]))
        print(
            f"seed {r['seed']}: ckpt={r['pes_selected_step']}"
            f"{' STEP-0' if s0 else ''} reused={r['ckpt_reused']}  "
            f"cover pes/pbgt={float(r['pes_coverage_min']):.3f}/{float(r['pbgt_coverage_min']):.3f}",
            flush=True,
        )

    def col(name: str) -> np.ndarray:
        return np.asarray([float(r[name]) for r in seeds], dtype=np.float64)

    blocks = [
        ("clean", "{m}_clean"),
        ("shared", "{m}_shared"),
        ("pgd", "{m}_pgd"),
        ("matched", "{m}_matched"),
    ]
    if seeds and int(float(seeds[0].get("shaped_ran", 0) or 0)) == 1:
        blocks.append(("shaped", "{m}_shaped"))
    print(f"{'cond':<10} {'vanilla':>18} {'pessimistic':>18} {'pbgt':>18}", flush=True)
    summary_rows = []
    for label, pattern in blocks:
        cells = {}
        for method in METHODS:
            a = col(pattern.format(m=method))
            cells[method] = (float(a.mean()), float(a.std(ddof=1)))
        print(
            f"{label:<10} {_fmt(*cells['vanilla']):>18} {_fmt(*cells['pessimistic']):>18} "
            f"{_fmt(*cells['pbgt']):>18}",
            flush=True,
        )
        summary_rows.append((label, cells))

    print("\nwin rate vs vanilla (strict >), n=%d" % len(seeds), flush=True)
    for label, pattern in blocks:
        for challenger in ("pessimistic", "pbgt"):
            wins = int(np.sum(col(pattern.format(m=challenger)) > col(pattern.format(m="vanilla"))))
            print(f"  {challenger}>van {label}: {wins}/{len(seeds)}", flush=True)

    print("\nvacuous (never a terminal in attacker training)", flush=True)
    for kind, pattern in (("matched", "{m}_matched_vacuous"), ("shaped", "{m}_shaped_vacuous")):
        if kind == "shaped" and not (seeds and int(float(seeds[0].get("shaped_ran", 0) or 0)) == 1):
            print("  shaped: not run", flush=True)
            continue
        if kind == "shaped" and f"{METHODS[0]}_shaped_vacuous" not in seeds[0]:
            print("  shaped: not run", flush=True)
            continue
        for method in METHODS:
            n = int(np.sum(col(pattern.format(m=method))))
            print(f"  {kind} {method}: {n}/{len(seeds)}", flush=True)

    print("\ninference ms, single-process, mean of per-seed means ± std (ddof=1)", flush=True)
    for method in METHODS:
        a = col(f"{method}_infer_ms_mean")
        print(f"  {method:<12} {a.mean():.4f} ± {a.std(ddof=1):.4f} ms", flush=True)

    pes_shared = col("pessimistic_shared").mean() > col("vanilla_shared").mean()
    pes_pgd = col("pessimistic_pgd").mean() > col("vanilla_pgd").mean()
    pes_matched = col("pessimistic_matched").mean() > col("vanilla_matched").mean()
    print(
        f"\npes mean > van mean: shared={pes_shared}  pgd={pes_pgd}  matched={pes_matched}",
        flush=True,
    )
    cover_ok = all(int(float(r["coverage_assert_ok"])) == 1 for r in seeds)
    print(f"coverage assertion 100% all seeds: {cover_ok}", flush=True)


def main() -> None:
    missing = [s for s in SEEDS if not os.path.exists(f"checkpoints/seed_{s}/vanilla.pt")]
    if missing:
        sys.exit(f"missing vanilla checkpoints for seeds {missing}")

    print(
        "RUNTIME ESTIMATE (before launch)\n"
        "  Anchors: tune FT+eval seeds 0–2 was 30.1 min wall (no attackers); "
        "pes FT ~1–5 min, PB-GT ~2.5 min, Table-6 eval ~4 min/seed.\n"
        "  Frozen-Q 5-seed harness (4 rules) was ~75 min wall; a sound maximin "
        "cell (eval + 40k SB3) was ~5 min, a greedy cell ~1 min.\n"
        "  This job reuses seeds 0–2 fine-tunes (settings identical). "
        "Seeds 3–4 FT ~15–20 min. Eval 5 seeds ~20 min.\n"
        "  15 unshaped attackers ~35–50 min. Shaped (only if pes matched is "
        "vacuous) another ~35–50 min. Inference ~5–10 min.\n"
        "  Expected ~1.5 h, ~2.5 h if shaped runs. Upper bound ~4 h. Under 12 h, so RUN.",
        flush=True,
    )
    print(
        f"SEEDS={SEEDS} reuse={list(REUSE_SEEDS)} pes_lr={PES_LR} pes_cap={PES_STEPS} "
        f"pbgt_steps={OFF_STEPS} eval_ep={EVAL_EP} atk_steps={ATK_STEPS} eps={EPS} lattice={KAPPA}",
        flush=True,
    )

    rows = [r for r in load_result_rows(OUT_CSV) if _is_seed_row(r)]
    by_seed = {int(r["seed"]): r for r in rows}
    t_all = time.time()

    for seed in SEEDS:
        print(f"\n======== SEED {seed} ========", flush=True)
        t_seed = time.time()
        meta = ensure_finetune(seed)
        row = by_seed.get(seed) or _blank_row(seed, meta)
        # Refresh ckpt fields from meta (resume may have a partial row).
        row.update({k: v for k, v in _blank_row(seed, meta).items() if k in (
            "ckpt_reused", "pes_selected_step", "pes_step0_selected", "pes_early_stopped",
            "pes_stopped_step", "pes_coverage_min", "pbgt_coverage_min", "coverage_assert_ok",
            "pes_ft_min", "pbgt_ft_min", "settings_note", "eps", "eval_episodes",
            "atk_steps", "pes_lr", "lattice", "row_kind", "seed",
        )})
        def _save_seed() -> None:
            nonlocal rows
            rows = _upsert(rows, row)
            by_seed[seed] = row
            _persist(rows)

        if not row.get("eval_done"):
            row.update(eval_table6(seed))
            row["eval_done"] = 1
            _save_seed()
        else:
            print("  eval SKIP", flush=True)
        train_unshaped(seed, row, on_progress=_save_seed)
        row["matched_done"] = 1
        _save_seed()
        if not row.get("infer_done"):
            row.update(time_inference(seed))
            row["infer_done"] = 1
        else:
            print("  infer SKIP", flush=True)
        row["seed_wall_min"] = round(float(row.get("seed_wall_min") or 0) + (time.time() - t_seed) / 60.0, 2)
        _save_seed()
        print(f"  seed {seed} checkpointed to {OUT_CSV}", flush=True)

    pes_vac = [int(float(by_seed[s]["pessimistic_matched_vacuous"])) for s in SEEDS]
    print(f"\npes matched vacuous flags: {pes_vac}  count={sum(pes_vac)}/{len(SEEDS)}", flush=True)
    if sum(pes_vac) > 0:
        print(
            "Pessimistic matched cells are vacuous on at least one seed; "
            "training shaped-reward attackers (|theta|+0.1*|x|) for ALL methods "
            "as a SEPARATE column.",
            flush=True,
        )
        for seed in SEEDS:
            print(f"\n======== SHAPED SEED {seed} ========", flush=True)
            t_seed = time.time()
            row = by_seed[seed]

            def _save_shaped(row=row, seed=seed) -> None:
                nonlocal rows
                rows = _upsert(rows, row)
                by_seed[seed] = row
                _persist(rows)

            train_shaped_all(seed, row, on_progress=_save_shaped)
            row["shaped_ran"] = 1
            row["shaped_reward"] = "abs(theta)+0.1*abs(x)"
            row["seed_wall_min"] = round(
                float(row.get("seed_wall_min") or 0) + (time.time() - t_seed) / 60.0, 2
            )
            rows = _upsert(rows, row)
            _persist(rows)
    else:
        print("No pessimistic matched cell is vacuous; shaped attackers NOT run.", flush=True)
        for seed in SEEDS:
            by_seed[seed]["shaped_ran"] = 0
            by_seed[seed]["shaped_reward"] = ""
            rows = _upsert(rows, by_seed[seed])
        _persist(rows)

    summarize(rows)
    print(f"\nwrote {OUT_CSV}  total wall {(time.time()-t_all)/60:.1f} min", flush=True)
    print("TRAINED_BOX_5SEEDS_V3_DONE", flush=True)


if __name__ == "__main__":
    main()

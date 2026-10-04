#!/usr/bin/env python3
"""Final tabular evaluation protocol (9 attackers × 500 seeds).

Phases:
  probe        — 1-seed full train + 9 attacker evals/defender; print wall estimate; stop if >2h
  train_atk    — train 9 SB3 attackers (new final_v3 names; skip if zip exists; never overwrite)
  eval500      — seeds × 8 workers → CSV; append per seed; resume skips completed seeds
  inference    — attack-path timing; append/resume same as eval
  summarize    — print matched / worst-case / win rates / clean / spread tables
  resume_smoke — 2-seed CSV append/skip self-test (temp CSV; no attackers)
  all          — probe → (if ≤2h or --force-proceed) train_atk → eval500 → inference → summarize

Eval/inference CSVs: append-only, never truncate. Resume skips unique seed ids already present.
Uses FIXED get_att_state from main_with_vanilla.py (no monkeypatch needed).
Does NOT edit main.tex. Does NOT overwrite older CSVs / non-final attacker zips.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

# ---------------------------------------------------------------------------
# Constants / naming
# ---------------------------------------------------------------------------

OUT_CSV = "gridworld_experiments_500runs_final_v3.csv"
OUT_INFER_CSV = "gridworld_inference_20seeds_final_v3.csv"
MAPPING_JSON = "attacker_final_v3_mapping.json"
N_SEEDS = 500
N_WORKERS = 8
N_INFER_SEEDS = 20
TIMESTEPS = 200_000
MAX_TOTAL_HOURS = 2.0

# Each of the 3 attackers for a defender type is trained against a DIFFERENT
# defender seed. Attacker DQN RNG seeds are also distinct.
DEFENDER_SEEDS = [99999, 99998, 99997]  # index a ∈ {0,1,2}
ATTACKER_RNG = {
    "vanilla": [11001, 11002, 11003],
    "pessimistic": [12001, 12002, 12003],
    "offpolicy": [13001, 13002, 13003],
}
DEF_TYPES = ("vanilla", "pessimistic", "offpolicy")
DEF_SHORT = {"vanilla": "van", "pessimistic": "pes", "offpolicy": "pbgt"}


def attacker_stem(def_type: str, a_idx: int) -> str:
    return f"attacker_{def_type}_final_v3_{a_idx}"


def all_attacker_stems() -> List[Tuple[str, int, str]]:
    """List of (def_type, a_idx, stem)."""
    out = []
    for dt in DEF_TYPES:
        for a in range(3):
            out.append((dt, a, attacker_stem(dt, a)))
    return out


def attacker_labels() -> List[str]:
    """Column suffixes: van0, van1, van2, pes0, pes1, pes2, pbgt0, pbgt1, pbgt2."""
    labs = []
    for dt in DEF_TYPES:
        for a in range(3):
            labs.append(f"{DEF_SHORT[dt]}{a}")
    return labs


# ---------------------------------------------------------------------------
# Defender prep / attacker training
# ---------------------------------------------------------------------------

def _prep_defender(def_type: str, defender_seed: int):
    """Train a frozen defender for attacker training (same as retrain_v3 / Option 4)."""
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
    if def_type == "vanilla":
        agent.q_table_withatt = {k: dict(v) for k, v in agent.q_table.items()}
    elif def_type == "pessimistic":
        play_att_with_belief(env, agent, trials=500, learn=True)
    elif def_type == "offpolicy":
        play_offpolicy_greedy_target(env, agent, trials=500, learn=True)
    else:
        raise ValueError(def_type)
    return agent


def train_one_attacker(def_type: str, a_idx: int) -> float:
    """Train one SB3 attacker. Returns wall seconds. Skips if zip already exists."""
    from diag_tabular_v3_common import patch_vanilla_attacker_env
    from main_with_vanilla import GridWorld_RL_Attack

    stem = attacker_stem(def_type, a_idx)
    out_zip = stem + ".zip"

    # Safety: never touch legacy / single-v3 diagnosis artifacts
    forbidden = [
        "attacker_pessimistic.zip",
        "attacker_offpolicy.zip",
        "attacker_pessimistic_v3.zip",
        "attacker_offpolicy_v3.zip",
        "attacker_vanilla_v3.zip",
    ]
    assert out_zip not in forbidden

    if os.path.exists(out_zip):
        print(f"SKIP existing {out_zip} (no overwrite)", flush=True)
        return 0.0

    defender_seed = DEFENDER_SEEDS[a_idx]
    atk_seed = ATTACKER_RNG[def_type][a_idx]
    print(
        f"=== Train {stem}: defender_seed={defender_seed} "
        f"attacker_rng={atk_seed} type={def_type} ===",
        flush=True,
    )

    if def_type == "vanilla":
        patch_vanilla_attacker_env()

    t0 = time.perf_counter()
    agent = _prep_defender(def_type, defender_seed)

    ckpt_dir = f"RL_attack/{def_type}_final_v3_a{a_idx}"
    os.makedirs(ckpt_dir, exist_ok=True)
    atk_env = GridWorld_RL_Attack(agent, agent_type=def_type)
    callback = CheckpointCallback(
        save_freq=5000, save_path=ckpt_dir, name_prefix=stem
    )
    # Seed SB3 + numpy for attacker training reproducibility
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


def train_all_attackers() -> Dict[str, Any]:
    mapping: Dict[str, Any] = {"defenders": {}, "wall_sec_total": 0.0}
    # Preserve prior mapping entries when resuming mid-train
    if os.path.exists(MAPPING_JSON):
        try:
            with open(MAPPING_JSON) as f:
                prev = json.load(f)
            if isinstance(prev.get("defenders"), dict):
                mapping["defenders"] = prev["defenders"]
        except (json.JSONDecodeError, OSError):
            pass
    t_all = time.perf_counter()
    times = []
    for def_type, a_idx, stem in all_attacker_stems():
        secs = train_one_attacker(def_type, a_idx)
        times.append(secs)
        mapping["defenders"][stem] = {
            "def_type": def_type,
            "attacker_index": a_idx,
            "defender_seed": DEFENDER_SEEDS[a_idx],
            "attacker_rng_seed": ATTACKER_RNG[def_type][a_idx],
            "zip": stem + ".zip",
            "train_wall_sec": secs,
            "skipped_existing": secs == 0.0 and os.path.exists(stem + ".zip"),
        }
    mapping["wall_sec_total"] = time.perf_counter() - t_all
    trained = [t for t in times if t > 0]
    mapping["per_attacker_mean_sec"] = float(np.mean(trained)) if trained else 0.0
    with open(MAPPING_JSON, "w") as f:
        json.dump(mapping, f, indent=2)
    print(
        f"ALL_9_ATTACKERS_DONE wall_min={mapping['wall_sec_total']/60:.1f} "
        f"→ {MAPPING_JSON}",
        flush=True,
    )
    return mapping


# ---------------------------------------------------------------------------
# Per-seed evaluation (3 defenders × 9 attackers)
# ---------------------------------------------------------------------------

def _fieldnames() -> List[str]:
    labs = attacker_labels()
    fields = ["seed", "vanilla_clean", "pes_clean", "pbgt_clean"]
    for d in ("van", "pes", "pbgt"):
        for lab in labs:
            fields.append(f"{d}_vs_{lab}")
        fields.append(f"{d}_matched_mean_of_3")
        fields.append(f"{d}_worst_of_own3")
        fields.append(f"{d}_worst_of_all9")
    return fields


def _run_one_seed(seed: int, attacker_paths: Dict[str, str]) -> Tuple[int, Dict[str, Any], float]:
    """Worker: train 3 defenders, eval vs all 9 attackers."""
    from diag_tabular_v3_common import eval_attack, train_defenders

    t0 = time.perf_counter()
    agent_v, agent_p, agent_o, cleans = train_defenders(seed, scoring="fixed")
    agents = {"van": agent_v, "pes": agent_p, "pbgt": agent_o}
    def_name = {"van": "vanilla", "pes": "pes", "pbgt": "pbgt"}

    models = {lab: DQN.load(path) for lab, path in attacker_paths.items()}
    labs = attacker_labels()

    row: Dict[str, Any] = {"seed": seed, **cleans}
    for d in ("van", "pes", "pbgt"):
        agent = agents[d]
        pol_snap = copy.deepcopy(getattr(agent, "defender_policy", None))
        vals_all = []
        vals_own = []
        for lab in labs:
            agent.defender_policy = copy.deepcopy(pol_snap)
            mean, _, _ = eval_attack(def_name[d], agent, seed, models[lab], seed)
            row[f"{d}_vs_{lab}"] = mean
            vals_all.append(mean)
            if lab.startswith(d):
                vals_own.append(mean)
        row[f"{d}_matched_mean_of_3"] = float(np.mean(vals_own))
        row[f"{d}_worst_of_own3"] = float(np.min(vals_own))
        row[f"{d}_worst_of_all9"] = float(np.min(vals_all))

    return seed, row, time.perf_counter() - t0


def _attacker_paths_from_disk() -> Dict[str, str]:
    paths = {}
    for dt in DEF_TYPES:
        for a in range(3):
            stem = attacker_stem(dt, a)
            if not os.path.exists(stem + ".zip"):
                raise SystemExit(f"Missing {stem}.zip — run train_atk first")
            paths[f"{DEF_SHORT[dt]}{a}"] = stem
    return paths


def _completed_seeds(path: str) -> Set[int]:
    """Unique seed ids already present in CSV (append/resume helper)."""
    if not os.path.exists(path):
        return set()
    done: Set[int] = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("seed")
            if raw is None or raw == "":
                continue
            done.add(int(float(raw)))
    return done


def run_resume_smoke(csv_path: str = "gridworld_final_v3_resume_smoke.csv") -> None:
    """Prove append + resume skip without training attackers (synthetic 2-seed rows)."""
    if os.path.exists(csv_path):
        os.remove(csv_path)
    fields = _fieldnames()

    def _fake_row(seed: int) -> Dict[str, Any]:
        row: Dict[str, Any] = {k: 0.0 for k in fields}
        row["seed"] = seed
        row["vanilla_clean"] = 100.0 + seed
        row["pes_clean"] = 90.0 + seed
        row["pbgt_clean"] = 80.0 + seed
        return row

    def _append_missing(n_seeds: int) -> List[int]:
        already = _completed_seeds(csv_path)
        missing = [s for s in range(n_seeds) if s not in already]
        if not missing:
            return []
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            for s in missing:
                writer.writerow(_fake_row(s))
                f.flush()
        return missing

    # Pass 1: write seed 0 only
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(_fake_row(0))
    assert _completed_seeds(csv_path) == {0}, _completed_seeds(csv_path)

    # Pass 2: resume n=2 → should append only seed 1
    appended = _append_missing(2)
    assert appended == [1], appended
    done = _completed_seeds(csv_path)
    assert done == {0, 1}, done

    # Pass 3: resume again → nothing to do, no duplicate rows
    appended2 = _append_missing(2)
    assert appended2 == [], appended2
    with open(csv_path, newline="") as f:
        nrows = sum(1 for _ in csv.DictReader(f))
    assert nrows == 2, nrows
    print(
        f"RESUME_SMOKE_OK csv={csv_path} seeds={sorted(done)} nrows={nrows} "
        "(append + skip verified; no duplicates)",
        flush=True,
    )


def run_eval500(csv_path: str = OUT_CSV, n_seeds: int = N_SEEDS) -> float:
    """Eval seeds 0..n_seeds-1; append per seed; resume skips completed seeds.

    Never truncates/overwrites an existing CSV — only appends missing seeds.
    """
    paths = _attacker_paths_from_disk()
    fields = _fieldnames()
    already = _completed_seeds(csv_path)
    missing = [s for s in range(n_seeds) if s not in already]
    print(
        f"Eval CSV={csv_path}: {len(already)} seed(s) done; "
        f"{len(missing)} missing of 0..{n_seeds - 1}",
        flush=True,
    )
    if not missing:
        print("EVAL500_DONE nothing to do — all seeds already present.", flush=True)
        return 0.0

    t_all = time.perf_counter()
    done = 0
    times: List[float] = []
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    print(
        f"Starting {len(missing)} seeds × {N_WORKERS} workers → append {csv_path}",
        flush=True,
    )

    # Append-only: never open with "w" on an existing file.
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(_run_one_seed, s, paths): s for s in missing}
            for fut in as_completed(futs):
                seed, row, secs = fut.result()
                writer.writerow(row)
                f.flush()
                done += 1
                times.append(secs)
                if done % 10 == 0 or done == len(missing):
                    avg = sum(times) / len(times)
                    eta = (len(missing) - done) / N_WORKERS * avg
                    print(
                        f"progress {done}/{len(missing)} "
                        f"(total_done≈{len(already) + done}/{n_seeds}) "
                        f"seed={seed} last={secs:.1f}s mean={avg:.1f}s "
                        f"eta_min={eta/60:.1f}",
                        flush=True,
                    )

    wall = time.perf_counter() - t_all
    print(f"EVAL500_DONE wall_min={wall/60:.1f} appended={done}", flush=True)
    return wall


# ---------------------------------------------------------------------------
# Probe + estimate
# ---------------------------------------------------------------------------

def run_probe(use_existing_placeholders: bool = True) -> Dict[str, float]:
    """1-seed full train + 9 attacker evals per defender; print estimate.

    If final_v3 attackers exist, use them. Else reuse existing single v3 attackers
    as timing placeholders (each loaded 3×) — weights don't affect wall time.
    """
    from diag_tabular_v3_common import eval_attack, train_defenders

    print("=== PROBE: 1 seed, full train + 9 attack evals × 3 defenders ===", flush=True)

    # Resolve 9 attacker paths for timing
    paths: Dict[str, str] = {}
    have_final = all(
        os.path.exists(attacker_stem(dt, a) + ".zip")
        for dt in DEF_TYPES
        for a in range(3)
    )
    if have_final:
        paths = _attacker_paths_from_disk()
        print("Using final_v3 attackers for probe.", flush=True)
        atk_train_est = None
        if os.path.exists(MAPPING_JSON):
            with open(MAPPING_JSON) as f:
                atk_train_est = json.load(f).get("wall_sec_total")
    else:
        placeholders = {
            "van": "attacker_vanilla_v3",
            "pes": "attacker_pessimistic_v3",
            "pbgt": "attacker_offpolicy_v3",
        }
        for short, stem in placeholders.items():
            if not os.path.exists(stem + ".zip"):
                raise SystemExit(f"Need {stem}.zip for probe placeholders")
            for a in range(3):
                paths[f"{short}{a}"] = stem
        print(
            "Using existing single v3 attackers ×3 as TIMING placeholders "
            "(not for final results).",
            flush=True,
        )
        # Historical: ~1.4 min/attacker × 9
        atk_train_est = 9 * 1.5 * 60.0

    t0 = time.perf_counter()
    seed = 0
    agent_v, agent_p, agent_o, cleans = train_defenders(seed, scoring="fixed")
    agents = {"van": agent_v, "pes": agent_p, "pbgt": agent_o}
    def_name = {"van": "vanilla", "pes": "pes", "pbgt": "pbgt"}
    models = {lab: DQN.load(p) for lab, p in paths.items()}
    labs = attacker_labels()

    for d in ("van", "pes", "pbgt"):
        agent = agents[d]
        pol_snap = copy.deepcopy(getattr(agent, "defender_policy", None))
        for lab in labs:
            agent.defender_policy = copy.deepcopy(pol_snap)
            eval_attack(def_name[d], agent, seed, models[lab], seed)

    sec_per_seed = time.perf_counter() - t0
    print(f"PROBE seed=0 wall_sec={sec_per_seed:.1f} cleans={cleans}", flush=True)

    # Attacker train estimate
    if atk_train_est is None:
        atk_train_est = 9 * 1.5 * 60.0

    eval_wall = (N_SEEDS / N_WORKERS) * sec_per_seed
    # Inference: ~same train+matched-attack as old 20-seed run; use ~0.55× seed time
    # (matched 3 attacks vs 27) ≈ train + 3/27 of attack portion. Conservative: 20 * sec_per_seed * 0.4
    infer_est = N_INFER_SEEDS * sec_per_seed * 0.45
    total = atk_train_est + eval_wall + infer_est

    print("\n=== WALL-TIME ESTIMATE ===", flush=True)
    print(f"  attackers_train: {atk_train_est/60:.1f} min", flush=True)
    print(f"  sec_per_seed:    {sec_per_seed:.1f} s", flush=True)
    print(f"  eval_500@8:      {eval_wall/60:.1f} min  (= 500/8 * {sec_per_seed:.1f})", flush=True)
    print(f"  inference_20:    {infer_est/60:.1f} min (est)", flush=True)
    print(f"  TOTAL:           {total/60:.1f} min ({total/3600:.2f} h)", flush=True)
    print(f"  threshold:       {MAX_TOTAL_HOURS:.1f} h", flush=True)

    return {
        "sec_per_seed": sec_per_seed,
        "atk_train_est": float(atk_train_est),
        "eval_wall_est": eval_wall,
        "infer_est": infer_est,
        "total_est": total,
        "proceed": total <= MAX_TOTAL_HOURS * 3600,
    }


# ---------------------------------------------------------------------------
# Inference (20 seeds, single process, attack-path timing)
# ---------------------------------------------------------------------------

def run_inference(csv_path: str = OUT_INFER_CSV, n_seeds: int = N_INFER_SEEDS) -> float:
    """Single-process attack-path timing for van/pes/PB-GT (matched attacker_0).

    Appends per seed; resumes by skipping seeds already in csv_path. Never truncates.
    """
    from main_with_vanilla import (
        play_att_with_belief,
        play_offpolicy_greedy_target,
        play_vanilla_test,
    )
    from diag_tabular_v3_common import train_defenders

    # Matched attacker index 0 for each method
    atk = {
        "van": DQN.load(attacker_stem("vanilla", 0)),
        "pes": DQN.load(attacker_stem("pessimistic", 0)),
        "pbgt": DQN.load(attacker_stem("offpolicy", 0)),
    }

    fields = [
        "seed",
        "vanilla_attack_inference_time_ms",
        "pessimistic_attack_inference_time_ms",
        "offpolicy_attack_inference_time_ms",
        "wall_sec",
    ]
    already = _completed_seeds(csv_path)
    missing = [s for s in range(n_seeds) if s not in already]
    print(
        f"Inference CSV={csv_path}: {len(already)} seed(s) done; "
        f"{len(missing)} missing of 0..{n_seeds - 1}",
        flush=True,
    )
    if not missing:
        print("INFERENCE_DONE nothing to do — all seeds already present.", flush=True)
        # Still print summary from full CSV
        _print_inference_summary(csv_path)
        return 0.0

    t_all = time.perf_counter()
    rows = []
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    print(
        f"Inference: {len(missing)} seeds single-process → append {csv_path}",
        flush=True,
    )

    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            w.writeheader()
        for seed in missing:
            t0 = time.perf_counter()
            agent_v, agent_p, agent_o, _ = train_defenders(seed, scoring="fixed")

            from main_with_vanilla import GridWorld

            env_v = GridWorld()
            env_v.seed(seed)
            np.random.seed(seed + 1000)
            _, t_van = play_vanilla_test(
                env_v, agent_v, trials=100, attack_model=atk["van"], measure_time=True
            )

            env_p = GridWorld()
            env_p.seed(seed)
            np.random.seed(seed + 1000)
            _, t_pes = play_att_with_belief(
                env_p, agent_p, trials=100, learn=False,
                attack_model=atk["pes"], measure_time=True,
            )

            env_o = GridWorld()
            env_o.seed(seed)
            np.random.seed(seed + 1000)
            _, t_pbgt = play_offpolicy_greedy_target(
                env_o, agent_o, trials=100, learn=False,
                attack_model=atk["pbgt"], measure_time=True,
            )

            row = {
                "seed": seed,
                "vanilla_attack_inference_time_ms": t_van * 1000,
                "pessimistic_attack_inference_time_ms": t_pes * 1000,
                "offpolicy_attack_inference_time_ms": t_pbgt * 1000,
                "wall_sec": time.perf_counter() - t0,
            }
            w.writerow(row)
            f.flush()
            rows.append(row)
            print(
                f"seed={seed} van={row['vanilla_attack_inference_time_ms']:.4f} "
                f"pes={row['pessimistic_attack_inference_time_ms']:.4f} "
                f"pbgt={row['offpolicy_attack_inference_time_ms']:.4f} "
                f"sec={row['wall_sec']:.1f}",
                flush=True,
            )

    wall = time.perf_counter() - t_all
    _print_inference_summary(csv_path)
    print(f"INFERENCE_DONE wall_min={wall/60:.1f} appended={len(rows)}", flush=True)
    return wall


def _print_inference_summary(csv_path: str) -> None:
    if not os.path.exists(csv_path):
        return
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    print(
        f"\n=== Attack inference ms mean±std (n={len(rows)}, single process) ===",
        flush=True,
    )
    for col, name in [
        ("vanilla_attack_inference_time_ms", "vanilla"),
        ("pessimistic_attack_inference_time_ms", "pes"),
        ("offpolicy_attack_inference_time_ms", "PB-GT"),
    ]:
        a = np.array([float(r[col]) for r in rows], dtype=float)
        std = a.std(ddof=1) if len(a) > 1 else 0.0
        print(f"{name}: {a.mean():.4f}±{std:.4f}", flush=True)


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------

def _ms(a: np.ndarray) -> str:
    return f"{a.mean():.2f}±{a.std(ddof=1):.2f}"


def _med_iqr(a: np.ndarray) -> str:
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return f"median={med:.2f} IQR=[{q1:.2f},{q3:.2f}]"


def summarize(csv_path: str = OUT_CSV) -> None:
    import pandas as pd

    df = pd.read_csv(csv_path)
    n = len(df)
    print(f"\n{'='*70}\nFINAL V3 SUMMARY (n={n})  file={csv_path}\n{'='*70}", flush=True)

    methods = [
        ("van", "vanilla", "vanilla_clean"),
        ("pes", "pes", "pes_clean"),
        ("pbgt", "PB-GT", "pbgt_clean"),
    ]

    # (a) Matched
    print("\n=== (a) MATCHED (own 3 attackers) ===", flush=True)
    print(
        "  mean_of_3_attacker_means: mean over seeds of (mean of returns vs own 3)",
        flush=True,
    )
    print(
        "  mean_of_perseed_min:      mean over seeds of (min return among own 3)",
        flush=True,
    )
    for d, name, _ in methods:
        own_cols = [f"{d}_vs_{d}{a}" for a in range(3)]
        # Per-attacker mean across seeds
        atk_means = [df[c].mean() for c in own_cols]
        mean_of_atk_means = float(np.mean(atk_means))
        # Per-seed mean of own 3, then mean±std
        perseed_mean3 = df[[f"{d}_matched_mean_of_3"]].to_numpy().ravel()
        perseed_min3 = df[[f"{d}_worst_of_own3"]].to_numpy().ravel()
        print(f"\n  {name}:", flush=True)
        print(
            f"    per-attacker means: "
            + ", ".join(f"a{i}={atk_means[i]:.2f}" for i in range(3)),
            flush=True,
        )
        print(
            f"    mean of 3 attacker-means: {mean_of_atk_means:.2f}",
            flush=True,
        )
        print(
            f"    mean±std of per-seed mean-of-own-3: {_ms(perseed_mean3)}",
            flush=True,
        )
        print(
            f"    mean±std of per-seed min-of-own-3 (worst-of-own-3): {_ms(perseed_min3)}",
            flush=True,
        )
        print(f"    {_med_iqr(perseed_min3)} (worst-of-own-3)", flush=True)

    # (b) Worst-case over all 9
    print("\n=== (b) WORST-CASE (per-seed min over all 9 attackers) ===", flush=True)
    worst = {}
    for d, name, _ in methods:
        a = df[f"{d}_worst_of_all9"].to_numpy(dtype=float)
        worst[d] = a
        print(
            f"  {name}: {_ms(a)}  {_med_iqr(a)}  "
            f"%seeds>-100={(a > -100).mean()*100:.1f}%",
            flush=True,
        )

    # (c) Win rates under (b)
    print("\n=== (c) WIN RATES under worst-of-9 ===", flush=True)
    print(
        f"  pes > PB-GT:  {(worst['pes'] > worst['pbgt']).mean()*100:.1f}%",
        flush=True,
    )
    print(
        f"  pes > van:    {(worst['pes'] > worst['van']).mean()*100:.1f}%",
        flush=True,
    )
    print(
        f"  PB-GT > van:  {(worst['pbgt'] > worst['van']).mean()*100:.1f}%",
        flush=True,
    )

    # (d) Clean
    print("\n=== (d) CLEAN ===", flush=True)
    for d, name, col in methods:
        a = df[col].to_numpy(dtype=float)
        print(
            f"  {name}: {_ms(a)}  median={np.median(a):.2f}  "
            f"%seeds<0={(a < 0).mean()*100:.1f}%",
            flush=True,
        )

    # (e) Attacker-to-attacker spread
    print(
        "\n=== (e) ATTACKER-TO-ATTACKER SPREAD "
        "(std across its 3 attackers' mean returns) ===",
        flush=True,
    )
    for d, name, _ in methods:
        own_cols = [f"{d}_vs_{d}{a}" for a in range(3)]
        atk_means = np.array([df[c].mean() for c in own_cols])
        print(
            f"  {name}: std_across_3={atk_means.std(ddof=1):.2f}  "
            f"means={atk_means.round(2).tolist()}",
            flush=True,
        )

    # Full 3×9 mean table
    print("\n=== FULL CROSS mean returns (defender × attacker) ===", flush=True)
    labs = attacker_labels()
    col0 = "def\\atk"
    hdr = f"{col0:<8}" + "".join(f"{lab:>10}" for lab in labs)
    print(hdr, flush=True)
    for d, name, _ in methods:
        cells = []
        for lab in labs:
            vals = df[f"{d}_vs_{lab}"].to_numpy(dtype=float)
            cells.append(f"{vals.mean():.1f}")
        print(f"{name:<8}" + "".join(f"{c:>10}" for c in cells), flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=[
            "probe",
            "train_atk",
            "eval500",
            "inference",
            "summarize",
            "resume_smoke",
            "all",
        ],
        help="Which phase to run",
    )
    parser.add_argument(
        "--force-proceed",
        action="store_true",
        help="Ignore 2h gate (not used unless Adi re-authorizes)",
    )
    parser.add_argument(
        "--csv",
        default=OUT_CSV,
        help=f"Eval CSV path (default {OUT_CSV}); append/resume never truncates",
    )
    parser.add_argument(
        "--infer-csv",
        default=OUT_INFER_CSV,
        help=f"Inference CSV path (default {OUT_INFER_CSV})",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=N_SEEDS,
        help=f"Number of eval seeds 0..N-1 (default {N_SEEDS})",
    )
    parser.add_argument(
        "--n-infer-seeds",
        type=int,
        default=N_INFER_SEEDS,
        help=f"Number of inference seeds 0..N-1 (default {N_INFER_SEEDS})",
    )
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="Skip probe in 'all' (still respects --force-proceed / prior auth)",
    )
    args = parser.parse_args()

    if args.phase == "probe":
        run_probe()
        return

    if args.phase == "resume_smoke":
        smoke_csv = (
            args.csv
            if args.csv != OUT_CSV
            else "gridworld_final_v3_resume_smoke.csv"
        )
        run_resume_smoke(csv_path=smoke_csv)
        return

    if args.phase == "train_atk":
        train_all_attackers()
        return

    if args.phase == "eval500":
        run_eval500(csv_path=args.csv, n_seeds=args.n_seeds)
        return

    if args.phase == "inference":
        run_inference(csv_path=args.infer_csv, n_seeds=args.n_infer_seeds)
        return

    if args.phase == "summarize":
        summarize(csv_path=args.csv)
        return

    # phase == all
    print("=" * 70, flush=True)
    print("FINAL V3 PROTOCOL — ALL", flush=True)
    print("=" * 70, flush=True)
    print("\nAttacker ↔ defender-seed mapping:", flush=True)
    for dt in DEF_TYPES:
        for a in range(3):
            print(
                f"  {attacker_stem(dt, a)}.zip  ← defender_seed={DEFENDER_SEEDS[a]}  "
                f"atk_rng={ATTACKER_RNG[dt][a]}",
                flush=True,
            )

    if args.skip_probe:
        print("Skipping probe (--skip-probe).", flush=True)
        if not args.force_proceed:
            print(
                "STOP: --skip-probe requires --force-proceed (no fresh estimate).",
                flush=True,
            )
            return
    else:
        est = run_probe(use_existing_placeholders=True)
        if not est["proceed"] and not args.force_proceed:
            print(
                f"\nSTOP: estimated TOTAL {est['total_est']/3600:.2f} h > "
                f"{MAX_TOTAL_HOURS:.1f} h. Not launching 500-seed eval.",
                flush=True,
            )
            return
        print(
            f"\nPROCEED: estimate {est['total_est']/3600:.2f} h "
            f"(force={args.force_proceed}). "
            "Training 9 attackers then eval500 + inference.",
            flush=True,
        )

    t_pipeline = time.perf_counter()
    mapping = train_all_attackers()
    # Re-probe with real attackers for logging (optional quick check skipped)
    eval_wall = run_eval500(csv_path=args.csv, n_seeds=args.n_seeds)
    infer_wall = run_inference(csv_path=args.infer_csv, n_seeds=args.n_infer_seeds)
    summarize(csv_path=args.csv)
    total_wall = time.perf_counter() - t_pipeline
    print(
        f"\nPIPELINE_DONE train={mapping['wall_sec_total']/60:.1f}min "
        f"eval={eval_wall/60:.1f}min infer={infer_wall/60:.1f}min "
        f"total={total_wall/60:.1f}min",
        flush=True,
    )


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()

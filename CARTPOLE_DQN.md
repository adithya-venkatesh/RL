# CartPole DQN: Project Documentation

This document records the full history, design, experiments, and open questions for the **CartPole-v1 DQN** port of our GridWorld comparison pipeline. It is intended for team members reviewing the project in Cursor.

**Reference paper:** Sun & Zheng (ICLR 2024) — *Belief-Enriched Pessimistic Q-Learning Against Adversarial State Perturbations*

**Primary code:** [`main_cartpole_dqn.py`](main_cartpole_dqn.py)  
**GridWorld counterpart:** [`main_with_vanilla.py`](main_with_vanilla.py)  
**GridWorld docs:** [`readme.md`](readme.md)

---

## Table of Contents

1. [Project Goal](#project-goal)
2. [Repository Layout](#repository-layout)
3. [Three Algorithms Compared](#three-algorithms-compared)
4. [Implementation Timeline](#implementation-timeline)
5. [Architecture & Key Components](#architecture--key-components)
6. [CLI Reference](#cli-reference)
7. [Experiment Pipeline (Step 7)](#experiment-pipeline-step-7)
8. [Results Summary](#results-summary)
9. [Comparison with GridWorld](#comparison-with-gridworld)
10. [Comparison with the Original Paper (BP-DQN)](#comparison-with-the-original-paper-bp-dqn)
11. [Bugs Found and Fixes Applied](#bugs-found-and-fixes-applied)
12. [The Seed 0 Mystery (Resolved)](#the-seed-0-mystery-resolved)
13. [Pessimistic Stability Fixes (v2)](#pessimistic-stability-fixes-v2)
14. [Known Limitations & Open Questions](#known-limitations--open-questions)
15. [How to Contribute](#how-to-contribute)
16. [Suggested Next Steps](#suggested-next-steps)

---

## Project Goal

Replicate the GridWorld experiment design in **CartPole-v1 with deep Q-networks**, comparing:

| Method | Role |
|--------|------|
| **Vanilla DQN** | Baseline (standard Double DQN) |
| **Pessimistic DQN** | Belief-enriched maximin targets (Eq. 2.6) |
| **Off-Policy Greedy Target** | Pessimistic behavior at train, greedy bootstrap + greedy inference at test (Eq. 5.1–5.2) |

We evaluate each method **clean**, under a **trained DQN attacker**, and under **PGD-style worst-case perturbation** (CartPole-only extra metric).

---

## Repository Layout

```
RL/
├── main_cartpole_dqn.py              # Main CartPole pipeline (Steps 1–8)
├── main_with_vanilla.py              # GridWorld tabular baseline (500-run CSV)
├── readme.md                         # GridWorld logic documentation
├── requirements.txt                  # gymnasium, torch, stable_baselines3, ...
├── CARTPOLE_DQN.md                   # This file
│
├── cartpole_experiments_10runs.csv   # First 10-seed batch (pre-stability-fix)
├── cartpole_experiments_3runs_v2.csv # 3-seed validation (post-stability-fix)
├── cartpole_vanilla_dqn.pt           # Legacy global vanilla checkpoint (Steps 1–6)
├── cartpole_pessimistic_dqn.pt       # Legacy global pessimistic checkpoint
├── cartpole_offpolicy_dqn.pt         # Legacy global off-policy checkpoint
│
├── logs/
│   ├── cartpole_10runs.log
│   └── cartpole_3runs_v2.log
│
└── checkpoints/                      # Per-seed artifacts (Step 7 fair testing)
    └── seed_{N}/
        ├── vanilla.pt
        ├── pessimistic.pt
        ├── offpolicy.pt
        ├── attacker_vanilla.zip
        ├── attacker_pessimistic.zip
        └── attacker_offpolicy.zip
```

---

## Three Algorithms Compared

### Vanilla DQN (baseline)

- **Train:** ε-greedy exploration on true observations; standard Double DQN targets (Eq. 4.1).
- **Test:** Greedy action on true (or attacker-perturbed) state (Eq. 4.2).

### Pessimistic DQN

- **Train:** Worst-case perturbation δ during rollouts; maximin action over κ belief particles; pessimistic Bellman target:
  ```
  y = r + γ max_{a'} min_{m ∈ M} Q_target(m, a')
  ```
- **Test (clean):** Greedy on true state (matches GridWorld CSV column `pessimistic_test_avg_reward`).
- **Test (under attack):** Maximin over belief ball (Eq. 3.4).

### Off-Policy Greedy Target

- **Train:** Same pessimistic **behavior** as above, but **greedy bootstrap** target on true next-states (Eq. 5.2).
- **Test:** Greedy on perturbed state — no belief tracking (fast inference).

### Attack model

- Attacker chooses perturbation δ ∈ B_ε(s) (L∞ ball, ε = 0.05).
- Defender observes s̃ = clip(s + δ); attacker reward = −defender reward.
- **SB3 DQN attacker** uses **81 discrete actions** (3 levels per obs dimension: {−ε, 0, +ε}) because SB3 DQN requires discrete action spaces.

---

## Implementation Timeline

Chronological summary of work done in Cursor (June 2025–2026).

### Phase 1: Vanilla baseline (Step 1)

- Wrapped `CartPole-v1` with observation normalization to ~[−1, 1].
- Implemented PyTorch MLP Double DQN [128, 128].
- Trained 500k steps; achieved **500.0 ± 0.0** over 100 eval episodes (solved).
- Saved `cartpole_vanilla_dqn.pt`.

### Phase 2: Attack evaluation (Step 2)

- Added PGD-style `worst_case_perturbation` (32 random δ candidates).
- Vanilla under PGD worst-case: ~487 mean return (ε = 0.05).

### Phase 3: Pessimistic & off-policy (Steps 3–4)

- Extended `DQNAgent` with belief replay buffer, maximin action, pessimistic TD targets.
- **First Step 3 run failed:** fine-tune with ε starting at 1.0 destroyed policy (clean ~10.7).
- **Fix:** `prepare_robust_finetune()` — reset buffers, lr=1e-4, ε start=0.05.
- Re-run Step 3: clean **500.0** (single seed).
- Off-policy: 100k finetune destroyed policy; **40k finetune** preserves 500 clean.

### Phase 4: Attacker training (Step 5)

- **First attempt failed:** SB3 DQN does not support continuous Box action space for δ.
- **Fix:** Discrete 81-action perturbation grid via `delta_from_discrete_action()`.

### Phase 5: Full comparison & dtype bug (Step 6)

- Fixed `torch.as_tensor(..., dtype=np.float32)` → `torch.float32` in off-policy training.
- Single-seed comparison showed pessimistic robust to PGD (500) but slow under attack (~1.5 ms vs ~0.04 ms).

### Phase 6: Fair multi-seed pipeline (Step 7)

Addressed caveats for publication-quality testing:

| Caveat | Fix |
|--------|-----|
| Shared global checkpoint across seeds | Per-seed dirs under `checkpoints/seed_{N}/` |
| No CSV logging | `log_experiment_to_csv()` with GridWorld-matched columns + PGD extras |
| No resume | `cartpole_experiment_checkpoint.pkl` + `--resume` |
| Unfair eval (50 ep) | 100 eval episodes |
| Off-policy finetune too long | Split `--finetune-pessimistic` (100k) vs `--finetune-offpolicy` (40k) |
| GridWorld attack pairing | Vanilla tested with **pessimistic's attacker** |

**10-seed run (`cartpole_experiments_10runs.csv`):** ~3.6 hours. Pessimistic **failed badly** (mean clean 27, attack ~9).

### Phase 7: Pessimistic stability fixes (v2)

After debugging with Claude, identified **training collapse** (not eval bug):

- Mid-training eval every 25k steps (greedy + maximin clean).
- **Best-checkpoint restore** at end of pessimistic fine-tune.
- **Early stop** if greedy clean collapses after reaching ≥475.
- **Frozen backbone** — only final linear layer trains during pessimistic fine-tune.
- Gentler hyperparams: lr `5e-5`, ε start `0.02`.
- New CSV columns: `pessimistic_test_maximin_*`, `pessimistic_best_greedy_eval`, etc.

**3-seed validation (`cartpole_experiments_3runs_v2.csv`):** ~70 min. Seed 0 pessimistic restored to **500/500/500** (clean/attack/PGD).

---

## Architecture & Key Components

### Environment wrappers

| Class | Purpose |
|-------|---------|
| `CartPoleDefenderEnv` | Normalized CartPole-v1 for defender training/eval |
| `CartPoleAttackEnv` | Attacker MDP: chooses δ, defender acts on s̃ |

### Core agent

| Component | Purpose |
|-----------|---------|
| `QNetwork` | MLP Q_θ(s, a) |
| `DQNAgent` | Vanilla + pessimistic + off-policy methods |
| `ReplayBuffer` | Standard experience replay |
| `BeliefReplayBuffer` | Stores κ belief particles for pessimistic targets |
| `sample_ball()` | Random particles in L∞ ε-ball (simplified belief; **not** PF-RNN) |

### Key functions

| Function | Description |
|----------|-------------|
| `train_vanilla_dqn()` | 500k-step vanilla training |
| `train_pessimistic_dqn()` | Pessimistic fine-tune with eval + early stop + restore |
| `train_offpolicy_greedy_dqn()` | Off-policy fine-tune |
| `train_cartpole_attacker()` | SB3 DQN attacker vs fixed defender |
| `evaluate_agent()` | Unified eval (clean / attack / mode) |
| `evaluate_pgd_worst_case()` | PGD-style stress test |
| `run_single_experiment_with_seed()` | Full fair pipeline for one seed |
| `run_experiment_loop()` | Multi-seed loop with CSV + checkpoint |
| `analyze_cartpole_csv()` | GridWorld-style summary printout |

### Hyperparameters (defaults)

| Parameter | Vanilla | Pessimistic fine-tune | Off-policy fine-tune |
|-----------|---------|----------------------|----------------------|
| Timesteps | 500,000 | 100,000 (max) | 40,000 |
| Learning rate | 1e-3 | 5e-5 | 1e-5 |
| ε start | 1.0 | 0.02 | 0.01 |
| Batch size | 128 | 128 | 128 |
| Attack ε | 0.05 | 0.05 | 0.05 |
| Belief κ | — | 20 | 20 |
| PGD candidates | — | 32 | 32 |
| Attacker train | — | 80,000 steps | 80,000 steps |

---

## CLI Reference

```bash
# Step 1: Train vanilla DQN
python3 main_cartpole_dqn.py --step 1 --timesteps 500000 --seed 0

# Step 2: Vanilla attack eval (PGD + optional trained attacker)
python3 main_cartpole_dqn.py --step 2 --seed 0

# Step 3: Pessimistic fine-tune from vanilla checkpoint
python3 main_cartpole_dqn.py --step 3 --seed 0 --finetune 100000

# Step 4: Off-policy fine-tune (use 40000, not 100000)
python3 main_cartpole_dqn.py --step 4 --seed 0 --finetune 40000

# Step 5: Train three DQN attackers
python3 main_cartpole_dqn.py --step 5 --seed 0 --attack-steps 80000

# Step 6: Single-seed full comparison (uses global .pt files)
python3 main_cartpole_dqn.py --step 6 --seed 0 --finetune 40000

# Step 7: Multi-seed fair experiment loop (RECOMMENDED)
PYTHONUNBUFFERED=1 python3 main_cartpole_dqn.py --step 7 \
  --num-experiments 10 \
  --base-seed 0 \
  --timesteps 500000 \
  --finetune-pessimistic 100000 \
  --finetune-offpolicy 40000 \
  --attack-steps 80000 \
  --eval-episodes 100 \
  --csv cartpole_experiments_10runs_v2.csv \
  2>&1 | tee logs/cartpole_10runs_v2.log

# Resume interrupted Step 7 run
python3 main_cartpole_dqn.py --step 7 --num-experiments 10 --resume \
  --csv cartpole_experiments_10runs_v2.csv

# Step 8: Analyze CSV (GridWorld-style summary)
python3 main_cartpole_dqn.py --step 8 --csv cartpole_experiments_3runs_v2.csv
```

**Tip:** Set `MPLCONFIGDIR=/tmp/mpl` to avoid matplotlib cache warnings from SB3 imports.

---

## Experiment Pipeline (Step 7)

Each seed runs an **isolated** full pipeline:

```
Phase 1   Train vanilla DQN from scratch          → checkpoints/seed_N/vanilla.pt
Phase 1.5 Eval vanilla clean (100 ep)
Phase 2   Clone → pessimistic fine-tune (v2)      → pessimistic.pt
          Eval pessimistic clean (greedy + maximin)
Phase 3   Clone vanilla → off-policy fine-tune    → offpolicy.pt
          Eval off-policy clean
Phase 4   Train 3 attackers (per seed)            → attacker_*.zip
Phase 5   Adversarial eval:
            - Vanilla vs pessimistic's attacker   (GridWorld parity)
            - Pessimistic vs pessimistic's attacker (maximin defense)
            - Off-policy vs off-policy's attacker (greedy defense)
Phase 6   PGD worst-case eval (all three)
          Log row to CSV + save checkpoint
```

---

## Results Summary

### Run A: 10 seeds, pre-stability-fix

**File:** `cartpole_experiments_10runs.csv`  
**Log:** `logs/cartpole_10runs.log`  
**Runtime:** ~3.6 hours

| Algorithm | Mean clean | Mean vs attacker | Mean PGD |
|-----------|------------|------------------|----------|
| Vanilla | 290.3 ± 221.8 | 204.7 ± 227.5 | 205.8 ± 201.1 |
| Pessimistic | **27.4 ± 49.7** | **9.3 ± 0.2** | 23.2 ± 43.6 |
| Off-policy | 390.7 ± 179.2 | 143.5 ± 196.3 | 270.2 ± 171.5 |

**Solved clean (≥475):** vanilla 5/10, pessimistic **0/10**, off-policy 7/10.

This run showed pessimistic DQN **collapsing** during fine-tune on most seeds.

---

### Run B: 3 seeds, post-stability-fix (v2)

**File:** `cartpole_experiments_3runs_v2.csv`  
**Log:** `logs/cartpole_3runs_v2.log`  
**Runtime:** ~70 minutes

| Seed | Vanilla clean | Pes clean (greedy / maximin) | Pes attack | Pes PGD | Off clean |
|------|---------------|------------------------------|------------|---------|-----------|
| 0 | 500 | 500 / 500 | 500 | 500 | 500 |
| 1 | 500 | 500 / 500 | 182 | 253 | 278 |
| 2 | 56 | 240 / 264 | 18 | 497 | 489 |

| Algorithm | Mean clean | Mean vs attacker | Mean PGD |
|-----------|------------|------------------|----------|
| Vanilla | 352.1 ± 256.1 | 338.5 ± 279.8 | 291.5 ± 217.9 |
| Pessimistic | **413.2 ± 150.3** | 233.3 ± 245.2 | **416.7 ± 141.4** |
| Off-policy | 422.4 ± 125.1 | 177.2 ± 279.6 | 242.4 ± 174.8 |

**Solved clean (≥475):** vanilla 2/3, pessimistic 2/3, off-policy 2/3 (seed 2 hurts all).

---

## Comparison with GridWorld

GridWorld results from `gridworld_experiments_500runs.csv` (500 seeds):

| Algorithm | Mean clean | Mean vs attacker |
|-----------|------------|------------------|
| Vanilla | 7.35 ± 0.24 | **−682.14 ± 328.97** |
| Pessimistic | 7.35 ± 0.27 | **−21.76 ± 50.03** |
| Off-policy | 7.37 ± 0.23 | −552.44 ± 335.17 |

### What matches

- Off-policy has **~40× faster inference** than pessimistic maximin under attack.
- All three can solve the clean task when training succeeds.

### What differs

| Aspect | GridWorld | CartPole (v2) |
|--------|-----------|---------------|
| Representation | Tabular Q | Neural DQN |
| Seeds run | 500 | 3 (v2) / 10 (v1) |
| Pessimistic robustness | Strong (−22 vs −682) | Improved after v2 fix; still seed-dependent |
| Clean variance | Very low (σ ≈ 0.2) | High (σ > 100 on some batches) |
| Belief model | Exact discrete states | Random κ-ball (simplified) |
| Attacker | Shared pre-trained | Per-seed retrained |

---

## Comparison with the Original Paper (BP-DQN)

The paper's **BP-DQN** succeeded in continuous DQN settings (Table 1a: ~176 clean, ~165–171 under attack vs vanilla ~156 / negative).

Our simplified port **does not fully replicate BP-DQN**:

| Paper (BP-DQN) | Our CartPole pipeline |
|----------------|----------------------|
| PF-RNN belief model, pretrained on clean trajectories | Random κ-ball particles |
| Belief trained **before** robust policy | End-to-end fine-tune from vanilla |
| Continuous Gridworld environment | CartPole-v1 |
| Full Algorithm 4 training schedule | Bolt-on pessimistic fine-tune |

We **do** use Huber loss (`smooth_l1_loss`) as in Algorithm 4.

---

## Bugs Found and Fixes Applied

| Issue | Symptom | Fix |
|-------|---------|-----|
| Duplicate `prepare_robust_finetune()` in Step 3 | Redundant buffer reset | Removed duplicate call |
| `dtype=np.float32` in `torch.as_tensor` | Step 4 crash | Changed to `torch.float32` |
| SB3 DQN + continuous attacker actions | Step 5 `AssertionError` | 81 discrete perturbation actions |
| Shared `cartpole_vanilla_dqn.pt` across seeds | Seeds 1–9 reused seed 0 weights | Per-seed dirs in Step 7 |
| `__main__.DQNConfig` pickle error | Could not load checkpoints from script | `load()` registers `DQNConfig` on `__main__` |
| Pessimistic fine-tune collapse | Clean drops to ~9 mid-training | v2 stability fixes (see below) |
| Off-policy 100k finetune | Clean policy destroyed | Default 40k for off-policy |

---

## The Seed 0 Mystery (Resolved)

**Observation:** Pilot Step 3 on seed 0 reported pessimistic clean **500**; the 10-run batch reported **9.5** for the same seed.

**Root cause:** Pessimistic fine-tune **collapsed between 50k–75k steps** (not an eval-mode bug):

```
step  25000 | training return ≈ 500
step  50000 | training return ≈ 295
step  75000 | training return ≈ 10   ← collapse
step 100000 | training return ≈ 9.6
→ eval clean = 9.5
```

Greedy vs maximin eval both returned ~9 on the collapsed checkpoint — eval mode was not the issue.

The pilot happened to preserve a good Q-function despite low rollout returns; the 10-run batch fully collapsed.

---

## Pessimistic Stability Fixes (v2)

Implemented in `_training_loop_pessimistic()` and `prepare_robust_finetune()`:

1. **Mid-training eval** every 25k steps — logs `greedy_clean` and `maximin_clean`.
2. **Best-checkpoint restore** — always restore best weights at end of fine-tune.
3. **Early stop** — if greedy clean ≥ 475 then drops below 400 or falls >75 from best.
4. **Frozen backbone** — only the final linear layer updates during pessimistic fine-tune.
5. **Gentler hyperparams** — lr `5e-5`, ε start `0.02` (was `1e-4` / `0.05`).
6. **Determinism flags** — `cudnn.deterministic = True`.

**Result on seed 0 (v2):**

```
[eval @ 25000] greedy_clean=500.0  maximin_clean=500.0
[eval @ 50000] greedy_clean=500.0  maximin_clean=500.0
[eval @ 75000] greedy_clean=500.0  maximin_clean=500.0
[eval @100000] greedy_clean=500.0  maximin_clean=500.0
Final: clean=500, attack=500, PGD=500
```

---

## Known Limitations & Open Questions

1. **Vanilla DQN seed instability** — only 5/10 (v1) or 2/3 (v2) always solve from 500k steps with default hyperparameters.

2. **Simplified belief** — random ball sampling vs paper's PF-RNN; likely a major gap for true BP-DQN replication.

3. **Weak discrete attacker** — when vanilla solves, DQN attacker often returns 500; PGD is the stronger stress test.

4. **Statistical power** — 3–10 seeds vs GridWorld's 500; need 30+ for paper-quality claims.

5. **Pessimistic attack eval uses maximin** — very slow (~1.4 ms/action vs ~0.035 ms greedy).

6. **Off-policy still sensitive to finetune length** — must use 40k, not 100k.

---

## How to Contribute

### Setup

```bash
cd /path/to/RL
pip install -r requirements.txt
```

Requires: Python 3.9+, `gymnasium`, `torch`, `stable_baselines3`, `pandas` (for Step 8 analysis).

### Recommended workflow

1. Read this doc and skim [`main_cartpole_dqn.py`](main_cartpole_dqn.py) (search for `run_single_experiment_with_seed`).
2. Run Step 8 on existing CSVs to see output format.
3. Pick a task from [Suggested Next Steps](#suggested-next-steps).
4. Use **Step 7** for any new experiments (not Steps 1–6 alone) to ensure fair per-seed isolation.
5. Never commit large `.pt` / `.zip` checkpoints unless the team agrees — they live in `checkpoints/`.

### Code conventions

- Formula references use `**Eq. X.X:**` docstrings (matching GridWorld style).
- Keep GridWorld CSV column names compatible where possible.
- Off-policy default finetune: **40k**; pessimistic: **100k max** with v2 early stop/restore.

### Areas needing help

| Area | Description |
|------|-------------|
| **PF-RNN belief** | Implement paper's pretrained belief model |
| **Hyperparameter sweep** | Vanilla solve rate, pessimistic finetune length |
| **Stronger attacker** | SAC/TD3 continuous δ, higher ε |
| **Scale experiments** | 30–50 seed Step 7 batch |
| **Analysis / plots** | CSV → figures for paper |
| **CartPole continuous grid** | Environment closer to paper's Table 1a setting |

---

## Suggested Next Steps

### Immediate (before professor meeting)

- [ ] Run **10-seed Step 7** with v2 fixes: `--csv cartpole_experiments_10runs_v2.csv`
- [ ] Compare v2 CSV against GridWorld 500-run summary side-by-side
- [ ] Prepare 1-slide "implementation gap vs BP-DQN" (PF-RNN, belief pretraining)

### Medium term

- [ ] Investigate vanilla solve rate (seeds 2, 6, 8 fail in v1 batch)
- [ ] Try pessimistic eval reporting **both** greedy and maximin clean in all tables
- [ ] Add learning-curve plots from mid-training eval logs

### Long term (paper direction)

**Option A — Replication:** Closer BP-DQN (PF-RNN + pretraining) on CartPole or continuous grid.

**Option B — Ablation study:** Tabular GridWorld success vs simplified deep port failure modes; role of belief quality.

**Option C — Practical tradeoff:** Off-policy greedy target as fast approximate robustness in DQN settings.

---

## CSV Column Reference (Step 7)

Key columns in experiment CSVs:

| Column | Meaning |
|--------|---------|
| `vanilla_test_avg_reward` | Vanilla clean eval (greedy, 100 ep) |
| `vanilla_vs_attacker_reward` | Vanilla under **pessimistic's** attacker |
| `pessimistic_test_avg_reward` | Pessimistic clean (greedy) |
| `pessimistic_test_maximin_avg_reward` | Pessimistic clean (maximin) — v2 only |
| `pessimistic_vs_attacker_reward` | Pessimistic under attack (maximin defense) |
| `pessimistic_best_greedy_eval` | Best greedy score during fine-tune — v2 only |
| `pessimistic_early_stopped` | 1 if early stop triggered — v2 only |
| `offpolicy_greedy_test_avg_reward` | Off-policy clean (greedy) |
| `*_pgd_reward` | PGD worst-case eval |
| `speedup_offpolicy_vs_pessimistic` | Inference time ratio under attack |

---

## Contact & Handoff Notes

- **Global checkpoints** (`cartpole_vanilla_dqn.pt`, etc.) are from early single-seed Steps 1–6; Step 7 uses `checkpoints/seed_N/` exclusively.
- **Authoritative recent results:** `cartpole_experiments_3runs_v2.csv` (post-fix).
- **Superseded results:** `cartpole_experiments_10runs.csv` (pre-fix pessimistic collapse — do not use for pessimistic claims).

When in doubt, run:

```bash
python3 main_cartpole_dqn.py --step 8 --csv cartpole_experiments_3runs_v2.csv
```

---

*Last updated: June 2026 — reflects work through 3-seed v2 validation run.*

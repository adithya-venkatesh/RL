# Gridworld RL: Vanilla vs Pessimistic vs Off-Policy Greedy

This document describes the logic of `main_with_vanilla.py` (note: the filename uses **vanilla**, not "vanila"). The script compares three Q-learning approaches in a 5×5 gridworld under optional adversarial state perturbations.

---

## Table of Contents

1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [Environment: `GridWorld`](#environment-gridworld)
4. [Attack Environment: `GridWorld_RL_Attack`](#attack-environment-gridworld_rl_attack)
5. [Agent: `Q_Agent`](#agent-q_agent)
6. [Three Algorithms Compared](#three-algorithms-compared)
7. [Helper Functions](#helper-functions)
8. [Training & Evaluation Routines](#training--evaluation-routines)
9. [Adversarial Attacker Training](#adversarial-attacker-training)
10. [Experiment Pipeline](#experiment-pipeline)
11. [Logging, Checkpoints & Analysis](#logging-checkpoints--analysis)
12. [Main Menu & Execution Flow](#main-menu--execution-flow)
13. [Output Files](#output-files)

---

## Overview

The script implements a reinforcement learning research pipeline that:

1. Trains a **baseline vanilla Q-learning** agent in a gridworld.
2. Fine-tunes two **robust** variants starting from the same baseline Q-table:
   - **Pessimistic Q-Learning** — uses belief tracking and max-min Bellman targets to defend against worst-case state perturbations.
   - **Off-Policy Greedy Target** — uses pessimistic behavior during training but a standard greedy Q-learning target for updates; at test time it acts greedily without belief tracking.
3. Optionally trains **DQN attackers** (via Stable-Baselines3) that perturb the agent's perceived state.
4. Runs repeatable experiments across many random seeds, logging rewards, inference times, and comparative metrics to CSV.

The core research question: can off-policy greedy target achieve similar robustness to full pessimistic Q-learning with lower inference cost?

---

## Dependencies

| Library | Role |
|---------|------|
| `numpy` | Arrays, random seeds, statistics |
| `gymnasium` | Environment API (`gym.Env`, action/observation spaces) |
| `stable_baselines3` | DQN for training adversarial attackers |
| `copy`, `time`, `datetime`, `csv`, `pickle`, `pandas` | Deep copies, timing, logging, checkpoints, analysis |

---

## Environment: `GridWorld`

A custom Gymnasium environment representing a **5×5 grid**.

### Grid layout

- **Size:** 5 rows × 5 columns (indices 0–4).
- **Default cell reward:** `-1` (every cell initialized to `-1`).
- **Bomb** at `[4, 3]` → reward `-10` (terminal).
- **Gold** at `[0, 3]` → reward `+10` (terminal).
- **Actions:** `UP`, `DOWN`, `LEFT`, `RIGHT` (discrete 0–3).
- **Observation:** agent position as `[row, col]` (`MultiDiscrete([5, 5])`).
- **Episode limit:** 1000 steps (`truncated`).

### Key methods

| Method | Logic |
|--------|-------|
| `step(action)` | Moves agent; if move hits a border, position stays the same but reward is still read from the current cell. Increments `total_step`. Returns `(obs, reward, terminated, truncated, info)`. |
| `take_action(state, action)` | Pure transition function: given a state tuple and action, returns the next state tuple (no side effects). Used for belief updates. |
| `reset(seed)` | Resets step count; samples a random non-terminal start position anywhere on the grid. |
| `get_reward(location)` | Looks up reward from `self.grid`. |
| `check_state()` | Returns `'TERMINAL'` if agent is on bomb or gold. |

### Movement rules

- **UP:** row decreases (unless row == 0).
- **DOWN:** row increases (unless row == height − 1).
- **LEFT:** column decreases (unless column == 0).
- **RIGHT:** column increases (unless column == width − 1).

---

## Attack Environment: `GridWorld_RL_Attack`

A variant of the gridworld used to **train DQN attackers**. The attacker's action chooses a **perturbed state** the defender *believes* it occupies; the defender then responds; the attacker is rewarded for making the defender reach bad outcomes.

### Differences from `GridWorld`

- **5 actions:** `UP`, `DOWN`, `LEFT`, `RIGHT`, **`STAY`**.
- Holds a reference to a trained `Q_Agent` (`agentQ`).
- `agent_type` parameter: `'pessimistic'` or `'offpolicy'` — controls how the defender selects actions during attacker training.

### `step(action)` flow

1. Attacker picks a perturbation direction → computes `purturb_state` (possible false belief state).
2. **Defender responds** based on `agent_type`:
   - **`pessimistic`:** calls `agentQ.choose_action_att()` using belief-based `defender_policy`; updates belief set and policy via `get_policy_with_belief`.
   - **`offpolicy`:** greedy action from `agentQ.q_table_withatt` at the perturbed state (no belief).
3. Defender's action moves the **true** `current_location`.
4. **Attacker reward:** `-get_reward(current_location)` (negative of defender's reward — attacker wants defender to fail).
5. Episode ends at terminal states or 1000 steps.

### `reset(seed)`

- Random non-terminal start.
- Initializes `belief` to all states in `agentQ.all_state`.
- Recomputes `defender_policy` from belief.

---

## Agent: `Q_Agent`

Tabular Q-learning agent with two Q-tables:

| Table | Purpose |
|-------|---------|
| `q_table` | Standard vanilla Q-values |
| `q_table_withatt` | Q-values used for attack-aware / robust learning |

Both tables are initialized for all 25 states with random Q-values in `[0, 1]` for each of the four movement actions.

### Hyperparameters

- `epsilon` (default 0.05) — exploration rate
- `alpha` (default 0.1) — learning rate
- `gamma` (default 1) — discount factor (undiscounted)

### Core action selection

- **`choose_action(available_actions, eps)`** — ε-greedy on `q_table` at the environment's current location. Used for vanilla training/testing.
- **`choose_action_att(available_actions, state, eps)`** — ε-greedy but exploits via `defender_policy[state]` instead of raw Q-values. Used during pessimistic training and testing with belief.

### Learning updates

| Method | Update rule |
|--------|-------------|
| `learn(...)` | Standard Q-learning: `Q(s,a) ← (1−α)Q(s,a) + α(r + γ max_a' Q(s',a'))` on `q_table`. |
| `learn_with_att_with_belief(...)` | Pessimistic update on `q_table_withatt`: target uses `find_maxmin_with_belief` instead of simple max. |
| `learn_with_greedy_target(...)` | Off-policy update on `q_table_withatt`: target is `max_a' Q(s',a')` (same as vanilla), even though behavior may be pessimistic. |

### Belief & perturbation model

- **`get_ball(state)`** — returns the **L1-neighborhood** of a state: the state itself plus all orthogonally adjacent cells (respecting grid borders). This models possible true states given a perturbed observation.
- **`get_att_state(true_state, policy)`** — finds the **worst-case perturbation**: among all states in the ball around `true_state`, picks the one with the **lowest** Q-value under the current policy (adversarial state for training).
- **`states_intersection(belief, states)`** — filters belief set to states also in `states`.

### Policy computation

- **`get_policy(init=False)`** — if `init=True`, random actions; else for each state, average Q-values over its ball and pick the maximizing action.
- **`get_policy_with_belief(policy, belief)`** — belief-restricted policy: for each state, average Q-values only over states in `belief ∩ ball(state)`, then argmax. This is the **defender policy** used during pessimistic inference.
- **`get_policy_naive()`** — greedy per-state without ball averaging.
- **`pareto_update(policy)`** — iterative policy improvement: repeatedly updates each state's action to maximize mean Q over its ball until convergence. Used to initialize `defender_policy` before robust training.

### Pessimistic Bellman target: `find_maxmin_with_belief`

Given next state `s'` and current belief:

1. Compute `possible_states = belief ∩ ball(s')`.
2. If **multiple** possible states: for each action, take the **minimum** Q across possible states, then take the **maximum** over actions (max-min).
3. If **one** possible state: standard `max_a Q(s', a)`.
4. If **empty** belief intersection: fallback to max over `q_table_withatt[s']`.

This implements a robust Bellman backup under state uncertainty.

---

## Three Algorithms Compared

### 1. Vanilla Q-Learning (baseline)

| Phase | Behavior |
|-------|----------|
| **Training** | `play(..., learn=True)` — 2000 episodes, ε and α decay via `linear_schedule_exp`. Standard ε-greedy on `q_table`. |
| **Testing** | `play_vanilla_test(...)` — greedy on `q_table` at (possibly perturbed) state. No belief tracking. |

**Inference cost:** O(1) — one Q-table lookup and argmax.

### 2. Pessimistic Q-Learning

| Phase | Behavior |
|-------|----------|
| **Training** | `play_att_with_belief(..., learn=True)` — 500 episodes. Copies baseline into `q_table_withatt`, initializes policy via `pareto_update`. Each step: worst-case perturbation → belief update → policy update → max-min Q-learning update. |
| **Testing (clean)** | `play_pessimistic_clean(...)` — greedy on `q_table_withatt` at true state (no attacker, no belief). |
| **Testing (under attack)** | `play_att_with_belief(..., learn=False)` — full belief tracking + `choose_action_att` at perturbed state. Inference time includes belief intersection, policy recomputation. |

**Inference cost (under attack):** O(states × actions) per step — belief update and `get_policy_with_belief`.

### 3. Off-Policy Greedy Target

| Phase | Behavior |
|-------|----------|
| **Training** | `play_offpolicy_greedy_target(..., learn=True)` — 500 episodes. **Behavior policy:** same pessimistic flow as above (worst-case state, belief, defender policy). **Target policy:** `learn_with_greedy_target` uses standard max Q backup. |
| **Testing** | `play_offpolicy_greedy_target(..., learn=False)` — greedy on `q_table_withatt` at perturbed state. **No belief tracking at test time.** |

**Key insight:** Training sees adversarial perturbations and uses pessimistic exploration, but the Q-function is shaped by greedy targets. At test time, inference is as cheap as vanilla (single argmax) while hopefully retaining some robustness from training.

---

## Helper Functions

### `linear_schedule_exp(min_exp, ini_exp, total, current)`

Decaying schedule for ε or α:

```
value = max(min_exp, (total - current) * ini_exp / total)
```

Used to anneal exploration and learning rate over training trials.

### `apply_perturbation(state, perturbation_direction)`

Maps a perturbation direction (`UP`/`DOWN`/`LEFT`/`RIGHT`/`STAY`) to a new state tuple. If the move would leave the grid, returns the original state. Used when a trained DQN attacker is applied during testing.

### `states_intersection(current_belief, states)`

Set intersection helper for belief tracking.

---

## Training & Evaluation Routines

### `play(environment, agent, trials, max_steps_per_episode, learn)`

Vanilla training/evaluation loop:

1. Reset environment.
2. ε-greedy action from `q_table`.
3. Step environment, optionally Q-update.
4. On terminal: re-initialize environment, end episode.
5. Return list of per-episode cumulative rewards.

### `play_vanilla_test(environment, agent, trials, measure_time, attack_model)`

Vanilla **evaluation** with optional DQN attacker:

1. If `attack_model` provided: attacker predicts perturbation → `apply_perturbation`.
2. Greedy action from `q_table` at perturbed state.
3. Optionally times inference (argmax only).

### `play_att_with_belief(environment, agent, trials, learn, attack_model, measure_time)`

Pessimistic training/evaluation loop:

1. If learning: deep-copy `q_table` → `q_table_withatt`, initialize `defender_policy`.
2. Each episode: reset belief to all states.
3. Each step:
   - Determine `att_state` (worst-case or from `attack_model`).
   - `choose_action_att` at `att_state`.
   - Update belief: intersect with ball, propagate through transition.
   - Recompute `defender_policy` via `get_policy_with_belief`.
   - Step environment; if learning, `learn_with_att_with_belief`.
4. When `measure_time` and not learning: times belief + policy update + action selection.

### `play_offpolicy_greedy_target(environment, agent, trials, learn, measure_time, attack_model)`

Dual-mode loop:

- **`learn=True`:** pessimistic behavior (worst-case state, belief, defender policy) + `learn_with_greedy_target` update.
- **`learn=False`:** attacker perturbation (if any) → greedy argmax on `q_table_withatt` → time inference.

### `play_pessimistic_clean(environment, agent, trials, measure_time)`

Pessimistic agent tested **without** attacker or belief — greedy on true state using `q_table_withatt`.

---

## Adversarial Attacker Training

### `train_RL_attack_with_belief(agentQ, agent_type, model_name)`

Trains a **DQN attacker** (Stable-Baselines3) against a fixed defender:

1. Wraps defender in `GridWorld_RL_Attack(agentQ, agent_type)`.
2. DQN config: `MlpPolicy`, lr=1e-3, batch=256, γ=1, 200k timesteps, network `[256, 128]`.
3. Saves checkpoints every 5000 steps to `RL_attack/{agent_type}/`.
4. Saves final model as `{model_name}_{agent_type}.zip` (e.g. `attacker_pessimistic.zip`).

The attacker learns which perturbation direction (including STAY) causes the defender to perform worst.

---

## Experiment Pipeline

### `run_single_experiment_with_seed(experiment_num, seed, attack_models)`

Runs one full comparison with a fixed seed. Phases:

```
Phase 1:    Baseline training (2000 trials, vanilla Q-learning)
            → save baseline_avg_reward (mean of last 100 episodes)

Phase 1.5:  Test vanilla (100 trials, measure inference time)
            → optional: test vs pessimistic attacker

Phase 2:    Pessimistic training (500 trials, from baseline Q-table)
            → test clean (100 trials)
            → optional: test vs pessimistic attacker

Phase 3:    Off-policy greedy training (500 trials, from baseline Q-table)
            → test clean (100 trials)
            → optional: test vs offpolicy attacker

Phase 4:    Compute diffs and speedups
```

**Seed convention:** environment seed = `experiment_num`; numpy seed = `seed + 1000` for reproducibility within an experiment.

**Returned metrics include:**

- Average/std rewards for each algorithm (train, test, vs attacker)
- Inference times (ms) for each algorithm
- Performance differences (off-policy vs pessimistic, vs vanilla)
- Speedup ratios (inference time comparisons)

---

## Logging, Checkpoints & Analysis

### `log_experiment_to_csv(experiment_number, results_dict, filename)`

Appends one row per experiment to a CSV with ~30 columns covering all reward, timing, diff, and speedup fields. Writes header only if file is new.

### Checkpoint functions

| Function | Purpose |
|----------|---------|
| `save_checkpoint(exp_num, file)` | Pickle `last_completed_experiment` + timestamp |
| `load_checkpoint(file)` | Resume from last completed experiment (returns 0 if none) |
| `clear_checkpoint(file)` | Delete checkpoint when all experiments finish |

Enables **pause/resume** via Ctrl+C during the experiment loop.

### `check_experiment_progress(csv_filename, checkpoint_file, total_experiments)`

Prints completion percentage, average rewards so far, next experiment number, and rough time estimate.

### `analyze_csv_results(csv_filename)`

Reads CSV with pandas; prints mean ± std for vanilla, pessimistic, and off-policy (test and vs-attacker columns).

---

## Main Menu & Execution Flow

When run as `python main_with_vanilla.py`, an interactive menu appears:

| Option | Description |
|--------|-------------|
| **1** | Quick test — 5 experiments → `gridworld_test_5runs.csv` |
| **2** | Medium test — 50 experiments → `gridworld_medium_50runs.csv` |
| **3** | Full test — 500 experiments → `gridworld_experiments_500runs.csv` |
| **4** | Train attackers (one-time): baseline agents → DQN attackers saved as `attacker_pessimistic.zip` and `attacker_offpolicy.zip` |
| **5** | Analyze existing CSV results |
| **6** | Check experiment progress |

### Options 1–3 flow

1. Optionally load trained attackers (`attacker_*.zip`) for adversarial testing.
2. Optionally resume from checkpoint.
3. For each experiment `exp_num` in range:
   - Call `run_single_experiment_with_seed(exp_num, seed=exp_num, attack_models)`.
   - Log to CSV, save checkpoint.
   - Handle `KeyboardInterrupt` for graceful pause.
4. On completion: clear checkpoint, offer analysis.

### Option 4 flow (train attackers)

1. Train pessimistic agent: vanilla 2000 trials → pessimistic 500 trials.
2. Train off-policy agent: vanilla 2000 trials → off-policy 500 trials.
3. Train DQN attacker against each → save `.zip` models.

---

## Output Files

| File / Directory | Contents |
|------------------|----------|
| `gridworld_test_5runs.csv` | Results from 5-experiment run |
| `gridworld_medium_50runs.csv` | Results from 50-experiment run |
| `gridworld_experiments_500runs.csv` | Results from 500-experiment run |
| `test_checkpoint.pkl` / `medium_checkpoint.pkl` / `experiment_checkpoint.pkl` | Resume state |
| `attacker_pessimistic.zip` | DQN attacker trained vs pessimistic defender |
| `attacker_offpolicy.zip` | DQN attacker trained vs off-policy defender |
| `RL_attack/{agent_type}/` | DQN training checkpoints and TensorBoard logs |

---

## High-Level Architecture Diagram

```
                    ┌─────────────────┐
                    │   GridWorld     │
                    │  (5×5, bomb,    │
                    │   gold, -1)     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌────────────┐  ┌──────────────┐
        │  Vanilla │  │ Pessimistic│  │ Off-Policy   │
        │ Q-Learn  │  │ Q-Learn    │  │ Greedy Target│
        │ q_table  │  │ q_table_   │  │ q_table_     │
        │          │  │ withatt +  │  │ withatt      │
        │          │  │ belief     │  │ (greedy test)│
        └────┬─────┘  └─────┬──────┘  └──────┬───────┘
             │              │                 │
             └──────────────┼─────────────────┘
                            ▼
                 ┌─────────────────────┐
                 │ GridWorld_RL_Attack │
                 │ (DQN attacker train)│
                 └─────────────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  CSV + Checkpoints  │
                 │  Rewards, timing,   │
                 │  diffs, speedups    │
                 └─────────────────────┘
```

---

## Summary of Algorithmic Differences at Test Time

| | Vanilla | Pessimistic (under attack) | Off-Policy Greedy |
|---|---------|---------------------------|-------------------|
| Q-table used | `q_table` | `q_table_withatt` | `q_table_withatt` |
| Belief tracking | No | Yes | No |
| Policy computation | Argmax Q | `get_policy_with_belief` + `choose_action_att` | Argmax Q |
| Perturbation source | DQN attacker or none | DQN attacker or `get_att_state` | DQN attacker or none |
| Inference complexity | Low | High | Low |

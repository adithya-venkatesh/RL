# Pessimistic-Behavior / Greedy-Target Q-Learning

This repo studies whether robustness to adversarial *observation* perturbations in belief-enriched pessimistic Q-learning comes mostly from the learned \(Q\) or from the test-time maximin. It compares vanilla Q-learning, belief-enriched pessimistic Q-learning, and PB-GT (pessimistic behavior during data collection, greedy backup and greedy test), with target/test swaps that hold training fixed. Paper source: `arxiv_v1/main.tex`. Final CSVs and the figure script are listed in Appendix C of that file.

## Environments

1. **Tabular \(5\times5\) GridWorld** — discrete cells, belief over cells, 500 seeds for the main tables.
2. **Continuous GridWorld** — 2-D lift of the same layout, sound box belief, \(\varepsilon=0.25\) production runs (10 seeds) plus budget sweeps.
3. **CartPole-v1** — sound set-membership box (and an older GRU-particle port kept for the negative result).

## Reproduce tables and figures

Times below are rough wall-clock on an Apple M1 CPU from the logs that produced these CSVs. Most table numbers can also be recomputed from the committed CSVs without re-training. Figures read those CSVs directly.

### Main tabular results (Table `tab:gw`, Figs. 1–2)

| Artifact | Script / command | CSV read or written | ~M1 time |
|---|---|---|---|
| Belief-blind 500-seed returns | `python3 gridworld_final_v3_protocol.py all --skip-probe` | writes `gridworld_experiments_500runs_final_v3.csv` | ~2.5 h (train attackers + eval + short inference) |
| Fifteen belief-blind swap attackers | `python3 gridworld_target_test_swap_tabular_matched15_v3.py` | writes `gridworld_target_test_swap_tabular_matched15_v3.csv` | ~5 h (500 seeds) |
| Worst-of-16 sidecars | built with the matched-15 / final-v3 pipeline; see `gridworld_worst_of_16_v3*.csv` | `gridworld_worst_of_16_v3.csv` (+ summary/winrates) | included above |
| Belief-aware attackers + worst-of-22 | `python3 gridworld_belief_aware_attack_v3.py all` | writes `gridworld_belief_aware_attack_v3.csv` | ~10 min train + ~3.3 h eval |
| Vanilla vs swapped attackers | `python3 gridworld_vanilla_vs_swap_attackers_v3.py` | writes `gridworld_vanilla_vs_swap_attackers_v3.csv` | ~hours at 500 seeds |
| Clean belief / truncation diag | `python3 diag_clean_maximin_belief_v3.py` | writes `gridworld_clean_maximin_belief_v3.csv` | ~minutes |
| Attack-path timing (20 seeds) | `python3 gridworld_final_v3_protocol.py inference` or `python3 diag_inference_20seeds_v3.py` | `gridworld_inference_20seeds_final_v3.csv` | ~5–10 min |
| Figures 1–2 (and others) | `python3 arxiv_v1/build_figures.py` | reads the tabular CSVs above | under a minute |

### Continuous GridWorld (Table `tab:cgw`, Fig. 6; swap Table `tab:cgw-swap`)

| Artifact | Script / command | CSV | ~M1 time |
|---|---|---|---|
| Production 10 seeds, \(\varepsilon=0.25\) | `python3 gridworld_continuous_prod.py --eps 0.25 --vanilla-steps 80000 --finetune-steps 25000 --episodes 50 --seeds 0 1 2 3 4 5 6 7 8 9 --out gw_prod_eps025_10seeds_v3.csv` | `gw_prod_eps025_10seeds_v3.csv` | ~1 h (≈6 min/seed) |
| Target/test swap, seeds 0–4 | `python3 gridworld_target_test_swap_cgw_v3.py` | `gridworld_target_test_swap_cgw_v3.csv` | ~tens of minutes |
| PGD-10 under swapped rule | `python3 cgw_swap_pgd_offdiag_v3.py` | `gridworld_target_test_swap_cgw_pgd_offdiag_v3.csv` | minutes |
| Budget sweeps (appendix) | `python3 gridworld_continuous_prod.py --eps <ε> --vanilla-steps 100000 --finetune-steps 40000 --episodes 30 --seeds 0 1 2 --out gw_epssweep_eps0XX.csv` (and the `ft25k` variants with 40k/25k steps) | `gw_epssweep_*.csv` | ~12–20 min per ε setting |

### CartPole (Table `tab:cp`, seed table; appendix sound / GRU tables)

| Artifact | Script / command | CSV | ~M1 time |
|---|---|---|---|
| Trained sound box, 5 seeds | `python3 cartpole_trained_box_5seeds_v3.py` | `cartpole_trained_box_5seeds_v3.csv`, curves `cartpole_trained_box_5seeds_curves_v3.csv` | ~1.5–2.5 h |
| Frozen-Q sound probes | `python3 cartpole_sound_5seeds.py` | `cartpole_sound_5seeds.csv` | ~75 min |
| Pole-sensor bias / ε / position probes | `python3 cartpole_sound_bias5seeds.py`, `cartpole_sound_epssweep_v2.py`, `cartpole_sound_possensor_5seeds.py`, `cartpole_sound_epssweep.py` | matching `cartpole_sound_*.csv` | ~tens of minutes each |
| GRU-particle port (did not hold) | `main_cartpole_dqn.py` (historical) | `cartpole_experiments_restore_5runs.csv` | hours; kept as the negative result |

### Inference timing (Table `tab:speed`, Fig. 3)

```bash
python3 retime_inference_v3.py
```

Writes/updates `inference_retime_singleproc_v3.csv`. Continuous rows use the deployed box; tabular rows in that CSV are greedy clean readouts and are not the maximin cost in the speed table. Regenerating from scratch is on the order of tens of minutes.

### Build all paper figures from CSVs

```bash
python3 arxiv_v1/build_figures.py
```

Reads (among others) `gridworld_experiments_500runs_final_v3.csv`, `gridworld_target_test_swap_tabular_matched15_v3.csv`, `gridworld_worst_of_16_v3.csv`, `gridworld_belief_aware_attack_v3.csv`, `gridworld_inference_20seeds_final_v3.csv`, `inference_retime_singleproc_v3.csv`, `cartpole_experiments_restore_5runs.csv`, and `gw_prod_eps025_10seeds_v3.csv`. Writes PDFs under `arxiv_v1/figures/`.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

Python 3.9 was used for the reported runs. Model checkpoints and SB3 attacker zips are not committed; re-running train phases recreates them under `checkpoints/`, `RL_attack/`, and related directories.

## Layout

- `arxiv_v1/` — paper (`main.tex`), `build_figures.py`, and figure PDFs.
- Root `*_v3.csv` / matching scripts — current paper artifacts (Appendix C).
- `old_results/` — superseded CSVs, smoke tests, scratch scripts, and old logs (not deleted).

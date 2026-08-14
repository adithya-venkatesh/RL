# Next experiments: Continuous GridWorld + CAR-DQN + PB-GT

**Status:** Draft tables/figures regenerated in `paper.pdf` (GridWorld 500 + CartPole restore-5).  
**Advisor priority:** Continuous GridWorld vs Xiaolin BP-DQN **first**; CAR-DQN baseline; Atari later.

---

## 1. Continuous GridWorld (Xiaolin / ICLR 2024)

### Official code
- Repo: https://github.com/SliencerX/Belief-enriched-robust-Q-learning
- Train BP-DQN: `python3 train.py --config config/Grid_continous_ours.json`
- Test: `python3 test_gridmaze.py --config config/Grid_continous_ours.json`
- Env: `gridworld.py` + `maze.csv` (10×10 continuous coordinates, 8 actions; see Appendix F.2)

### Target paper numbers (Table 1a / Appendix F, continuous Gridworld)
Reproduce **vanilla DQN** and **BP-DQN** first under Natural / PGD / MinBest / PA-AD at ε ∈ {0.1, 0.5}, then add **PB-GT**.

Example reference points from the paper (means ± std over 10 episodes):
- DQN natural ~156; fails under strong attack
- BP-DQN natural ~163; stays high under PGD/MinBest/PA-AD

### Our additions on this env
| Method | Role |
|--------|------|
| Vanilla DQN | Reproduce paper baseline |
| BP-DQN (paper) | Reproduce Algorithm 4 |
| **PB-GT DQN** | Our extension: pessimistic behavior, greedy target/test |
| **CAR-DQN** | Li et al. 2024 baseline (see §2) |

### Practical setup (needs GPU workstation)
1. Clone Xiaolin repo on workstation; install `requirements.txt`.
2. Train / load PF-RNN: `python3 pfrnns/main.py` (or use `pretrained/`).
3. Reproduce BP-DQN with `Grid_continous_ours.json`.
4. Fork training loop for PB-GT: same collection as BP-DQN; change TD target to Double-DQN greedy on true `s'`; test with greedy (no maximin).
5. Match eval scripts (`test_gridmaze.py`) for fair comparison.
6. Scale seeds (paper uses 10 test envs; aim for more training seeds with GPU).

---

## 2. CAR-DQN baseline (arXiv:2402.02165)

**Paper:** Li et al., *Towards Optimal Adversarial Robust Q-learning with Bellman Infinity-error* (ICML / arXiv:2402.02165).

**Method:** Consistent Adversarial Robust DQN (**CAR-DQN**). Under a Consistency Assumption of Policy (CAP), an optimal robust policy aligns with the Bellman optimal policy; robustness requires minimizing a **Bellman ∞-error** surrogate (not standard L1 TD). Implementation uses Huber + inner max over `B_ε(s)` via **PGD** or **IBP**.

**Code:** https://github.com/leoranlmia/CAR-DQN

**Integration plan:**
1. Clone CAR-DQN on workstation; run their continuous-control / maze settings if overlapping, or port their loss into Xiaolin’s DQN stack on continuous GridWorld.
2. Report Natural + PGD (and MinBest/PA-AD if feasible) next to BP-DQN and PB-GT.
3. Cite as additional state-of-the-art robust DQN baseline (alongside SA-DQN / WocaR where available in Xiaolin’s repo).

**Contrast vs our methods:**
| | BP-DQN / pessimistic | PB-GT | CAR-DQN |
|--|----------------------|-------|---------|
| Core idea | Maximin over belief | Pessimistic behavior, greedy target/test | Bellman ∞-error / CAP |
| Belief / history | Yes (PF-RNN) | Optional at collect | No belief tracking |
| Test latency | High (maximin) | Low (greedy) | Low–medium (depends on defense) |

---

## 3. PB-GT motivation (for paper §)

**Why:** Maximin inference is robust but slow; many settings need fast test-time policies.

**What:** Train with maximin behavior under perturbed observations; learn with greedy Double-DQN targets on true next states; test with greedy on `s̃`.

**GridWorld evidence:** PB-GT > vanilla under attack (~61% seed win rate) at ~same speed as vanilla; pessimistic still best but ~11× slower.

**Improvements to try (after continuous GridWorld baseline works):**
1. Hybrid inference (greedy ↔ maximin on uncertainty)
2. Softmin instead of hard min over belief
3. Dual heads (robust + fast) / distillation
4. Retune fine-tune schedule after strong vanilla
5. Align κ / PGD with Algorithm 4

---

## 4. Atari (later)

Pong / Freeway + DP-DQN after continuous GridWorld + CAR-DQN. Use Xiaolin configs `Pong_ours.json` / Freeway; GPU required.

---

## 5. Deliverables already refreshed (this pass)

| File | Content |
|------|---------|
| `paper.pdf` | Tables + figures + interpretation (GW + CartPole restore) + PB-GT motivation + roadmap |
| `paper_figures/fig1_gridworld.png` | Clean vs attack |
| `paper_figures/fig2_cartpole.png` | 4-metric fair eval |
| `paper_figures/fig3_inference.png` | Speed comparison |
| `paper_figures/fig4_cartpole_per_seed.png` | Per-seed clean |
| `arxiv_prep_algorithms_metrics.pdf` | Algorithms / metrics / hyperparams (earlier) |
| `cartpole_experiments_restore_5runs.csv` | Full CartPole numbers |

## 6. Local clones started (for reference)

Under `third_party/`:
- `Belief-enriched-robust-Q-learning/` — includes `config/Grid_continous_ours.json`, `gridworld.py`, `train.py`, `test_gridmaze.py`
- `CAR-DQN/` — README + requirements (full PGD/IBP trees: clone on workstation with `git clone https://github.com/leoranlmia/CAR-DQN.git`)

## 7. Blocked on workstation (Yibin)

Cannot complete continuous GridWorld / CAR-DQN / Atari **training** on M1 CPU in reasonable time. Next: get SSH access, install deps, smoke-test:
```bash
cd third_party/Belief-enriched-robust-Q-learning
python3 train.py --config config/Grid_continous_ours.json   # BP-DQN continuous GridWorld
python3 test_gridmaze.py --config config/Grid_continous_ours.json
```

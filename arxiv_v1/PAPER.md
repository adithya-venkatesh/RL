# Pessimistic-Behavior / Greedy-Target Q-Learning against Adversarial State Perturbations

**Adithya Venkatesh** (independent researcher) · **Zizhan Zheng** (Department of Computer Science, Tulane University)  
*ArXiv v1 draft — for advisor review before posting. Do not claim PB-GT ≈ BP-DQN.*

---

## Abstract

A well-trained RL agent can be defeated by strategically perturbing its state observations at test time. Sun & Zheng (ICLR 2024) address this with *belief-enriched pessimistic Q-learning*: the agent chooses a maximin action over a belief about possible true states, both when collecting data and when forming Bellman targets. The defense is strong in discrete domains but costly at test time.

We study a cheaper variant, **PB-GT** (Pessimistic-Behavior / Greedy-Target) Q-learning: training still uses pessimistic (maximin) *behavior* under perturbed observations, but the *target* is ordinary greedy Q-learning on the true next state, and test-time actions are greedy on the observed state—no belief, no maximin.

On a tabular 5×5 GridWorld with **500** random seeds, full pessimistic Q-learning remains far more robust under trained DQN attackers (**−21.8 ± 50.0** vs **−552.4 ± 335.2** for PB-GT and **−682.1 ± 329.0** for vanilla), while PB-GT matches vanilla inference speed and is **≈11.6×** faster than maximin. A Double-DQN port to CartPole-v1 (**5** seeds) is a preliminary deep-RL check: vanilla converges reliably after checkpoint restore, but our BP-style pessimistic and PB-GT agents do *not* reproduce the GridWorld robustness ordering. We present PB-GT as a fast approximation with a documented robustness gap, not as a substitute for BP-DQN.

---

## 1. Introduction

RL agents that act on learned value networks are brittle under *adversarial state perturbations*: a test-time attacker, subject to budget ε, replaces the true state \(s\) by a nearby observation \(\tilde s \in B_\varepsilon(s)\) to minimize return (Huang et al., 2017; Zhang et al., 2020). The attacker does not change transition dynamics—only what the agent sees.

Defenses include SA-MDP / SA-DQN (Zhang et al., 2020) and WocaR-RL (Liang et al., 2022), which regularize smoothness, and ATLA (Zhang et al., 2021), which co-trains an attacker (more robust, more expensive).

Sun & Zheng (ICLR 2024) view the problem as a Stackelberg Markov game and propose pessimistic Q-learning: \(\pi(\tilde s) = \arg\max_a \min_{\bar s \in B_\varepsilon(\tilde s)} Q(\bar s, a)\), used in both behavior and Bellman targets. Belief \(M_t\) (tabular history, or PF-RNN particles in BP-DQN) shrinks uncertainty; DP-DQN uses diffusion on pixels.

Maximin-over-belief is the source of robustness *and* cost. In our tabular GridWorld it is ~11× slower than greedy; on CartPole DQN ~50×. We ask:

> If we keep pessimistic **behavior** while collecting data, but revert to vanilla greedy **targets** and greedy **test** actions, how much robustness do we retain, and how much inference cost do we save?

**PB-GT** is that pairing—not “off-policy DQN” (all three methods are off-policy). It is a deliberate train/test mismatch, opposite to Sun & Zheng’s consistency argument. We compare vanilla, belief-enriched pessimistic, and PB-GT throughout.

**Contributions**

1. Tabular reproduction at scale: vanilla vs Algorithms 1–2 on 5×5 GridWorld, **500 seeds**, trained DQN attackers.
2. PB-GT: pessimistic behavior, greedy target, greedy test. On GridWorld it sits between vanilla and full pessimism and matches vanilla speed.
3. Preliminary CartPole Double-DQN port (BP-*style* GRU belief, not full PF-RNN). Mixed/negative; we do not claim GridWorld-style deep robustness.
4. Scoped claim: v1 documents the robustness–compute tradeoff. Closing the gap to BP-DQN is future work.

---

## 2. Background

**MDPs / Q-learning.** Vanilla Q-learning: \(Q(s,a) \leftarrow (1-\alpha)Q(s,a)+\alpha(R+\gamma\max_{a'}Q(s',a'))\), ε-greedy on true \(s\). DQN / Double DQN replace the table with \(Q_\theta\).

**Attacks.** \(\tilde s_t \in B_\varepsilon(s_t)\); agent acts on \(\tilde s_t\); environment transitions from true \(s_t\). For fixed \(\pi\), the attacker is itself an MDP, so a DQN can attack a frozen defender. Training assumes an intact MDP and known ε.

**Pessimistic Q (Sun & Zheng).** Maximin policy + attacker \(\omega_\pi(s)=\arg\min_{\tilde s} Q(s,\pi(\tilde s))\). Belief: \(M_{t+1} = \{s': \exists s\in M_t, P(s'|s,a_t)>0\} \cap B_\varepsilon(\tilde s_{t+1})\). BP-DQN = deep maximin + PF-RNN; DP-DQN = diffusion. Our deep pessimistic agent is **BP-style** (GRU particles), not full BP-DQN. We do not implement DP-DQN.

---

## 3. Methods

| | Vanilla | Pessimistic | **PB-GT** |
|--|---------|-------------|-----------|
| Train behavior | ε-greedy on \(s\) | ε-greedy maximin on \(\tilde s\) | same as pes. |
| TD target | greedy on \(s'\) | max min over \(M'\) | greedy on \(s'\) |
| Test under attack | greedy on \(\tilde s\) | maximin over \(M\) | greedy on \(\tilde s\) |
| Belief | none | yes | train only |
| Counterpart | DQN | BP-style | this work |

**Vanilla.** Tabular Q or Double DQN; test greedy on the observation.

**Pessimistic (tabular).** Algorithms 1–2. Discrete ball = cell + 1-step neighbors. When \(|M\cap B|>1\), our *policy* uses *mean* Q over the belief (softer approximation); TD target still uses maximin. Fine-tune 500 episodes from a vanilla Q-table. Attack test: belief + maximin. Clean test: greedy on true \(s\).

**PB-GT (tabular).** Same collection as pessimistic (ω_π, ε-greedy maximin, belief update) but backup is vanilla:
\(Q(s,a)\leftarrow (1-\alpha)Q(s,a)+\alpha(R+\gamma\max_{a'}Q(s',a'))\).
Test: \(a=\arg\max_a Q(\tilde s,a)\) — no belief. Inference = vanilla.

**Why it might help.** Pessimistic rollouts expose \(Q\) to adversarial observations; maximin cost is training-only. **Risk (confirmed):** greedy test is still exploitable; do not expect BP-DQN-level robustness.

**Deep PB-GT DQN.** Fine-tune from vanilla (often freeze backbone). Collection: random-search PGD \(\tilde s\), ε-greedy maximin over GRU particles. Loss: Double-DQN greedy targets on *true* \(s'\), not \(\max\min\) over \(M'\). Test: greedy on \(\tilde s\). Full pseudocode is in the LaTeX appendix.

---

## 4. Experimental setup

**GridWorld.** 5×5; step −1; gold (0,3) +10; bomb (4,3) −10; γ=1; vanilla 2000 episodes; pes/PB-GT 500 fine-tune from same Q; eval 100 episodes; cap 1000 steps. **n = 500 seeds.** Attackers: SB3 DQN, frozen; vanilla vs pessimistic attacker; pes/PB-GT vs own attackers (matched). Not MinBest / PA-AD / iterative PGD.

**CartPole-v1 (preliminary).** ε=0.05 \(L_\infty\) raw; vanilla Double DQN [128,128], 500k steps, restore-best (solve ≥475). Belief: GRU h=64, κ=20, 30k pretrain (not κ_p=30 PF-LSTM). Pes fine-tune: freeze backbone, LR 5e-5, ≤100k, maximin target. PB-GT: 40k steps, LR 1e-5, greedy target. **n = 5.** Hardware: M1 Pro CPU (Sun & Zheng used RTX 3090).

Metrics: (1) clean; (2) matched SB3 attacker (81 actions, 80k steps); (3) shared random δ; (4) PGD* = 32-candidate random search (not 10-step PGD).

---

## 5. Results

### 5.1 GridWorld (main result)

| Method | Clean | Under attack | Attack infer. (ms) |
|--------|-------|--------------|--------------------|
| Vanilla | 7.35 ± 0.24 | −682.1 ± 329.0 | 0.008 |
| Pessimistic | 7.35 ± 0.27 | **−21.8 ± 50.0** | 0.093 |
| PB-GT | 7.37 ± 0.23 | −552.4 ± 335.2 | 0.008 |

Clean ≈ identical. Under attack: **pes ≫ PB-GT > vanilla**. Pes beats PB-GT in 96.4% of seeds, vanilla in 96.0%; PB-GT beats vanilla in 60.6%. PB-GT is a *modest* vanilla improvement, not a near-match to maximin. Inference: **11.6×** slower for pessimistic vs PB-GT/vanilla.

**Allowed claims:** (1) belief-maximin is highly effective here; (2) pessimistic *behavior* alone does not capture most of that robustness if test is greedy; (3) we do **not** claim PB-GT approaches BP-DQN robustness (~530 return gap).

### 5.2 Inference

| Environment | Pes (ms) | PB-GT (ms) | Slowdown |
|-------------|----------|------------|----------|
| GridWorld | 0.093 | 0.008 | 11.6× |
| CartPole (matched) | 1.88 | 0.037 | 51× |

### 5.3 CartPole (preliminary — do not oversell)

| Metric | Vanilla | Pes (BP-style) | PB-GT |
|--------|---------|----------------|-------|
| Clean | **499 ± 2** | 334 ± 226 | 403 ± 214 |
| Matched attacker | 113 ± 216 | 11 ± 3 | 109 ± 219 |
| Shared random δ | 284 ± 176 | 125 ± 185 | 150 ± 209 |
| PGD* | 319 ± 154 | 131 ± 126 | 189 ± 148 |

Vanilla: 5/5 seeds solved after restore-best. Pes/PB-GT unstable on clean; pes ~11 on every seed under matched attack. **No deep robustness claim.** Causes: hyperparams pre-restore; simplified GRU ≠ PF-RNN; different attacks; n=5; not continuous GridWorld.

Per-seed clean (V / P / PB-GT): 0: 500/500/500 · 1: 500/133/500 · 2: 500/500/500 · 3: 496/44/20 · 4: 500/490/495.

---

## 6. Related work

State attacks: Huang 2017; SA-MDP/SA-DQN, ATLA, WocaR; survey Ilahi 2022. Direct predecessor: Sun & Zheng ICLR 2024. CAR-DQN (Li et al., 2024) listed as *future* baseline. PF-RNN (Ma et al., 2020) is BP-DQN’s belief module.

---

## 7. Limitations and future work

**Not shown:** PB-GT ≈ pes robustness; CartPole confirmation; Xiaolin Table 1a; CAR-DQN / SA-DQN / WocaR / Atari; iterative PGD; vanilla-matched GW attacker.

**Next:** (1) continuous GridWorld + official BP-DQN, then PB-GT; (2) CAR-DQN baseline; (3) PB-GT refinements (hybrid inference, distillation, mixed targets); (4) Atari + DP-DQN on GPU.

---

## 8. Conclusion

Vanilla vs pessimistic vs PB-GT on 500-seed GridWorld and 5-seed CartPole DQN: full pessimism is far more robust under learned attacks; PB-GT is far cheaper at test time and only slightly better than vanilla. The deep port is not yet a robustness result. **v1 claim:** the robustness–compute tradeoff of belief-maximin Q-learning is real; PB-GT is a well-defined fast baseline that does not close the gap. Closing that gap on Sun & Zheng’s environments is the next paper.

---

## References

Huang et al., ICLR Workshop 2017 · Ilahi et al., IEEE TAI 2022 · Li et al., arXiv:2402.02165, 2024 · Liang et al., NeurIPS 2022 · Ma et al., AAAI 2020 · Mnih et al., Nature 2015 · Sun & Zheng, ICLR 2024 · van Hasselt et al., AAAI 2016 · Watkins & Dayan, 1992 · Zhang et al., NeurIPS 2020 · Zhang et al., ICLR 2021.

## Code / data

`main_with_vanilla.py`, `gridworld_experiments_500runs.csv`, `main_cartpole_dqn.py`, `cartpole_belief.py`, `cartpole_experiments_restore_5runs.csv`.  
BP-DQN: https://github.com/SliencerX/Belief-enriched-robust-Q-learning · CAR-DQN: https://github.com/leoranlmia/CAR-DQN

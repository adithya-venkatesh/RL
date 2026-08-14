# CartPole DQN Extension — One-Page Summary

**Adithya Venkatesh · Professor Zheng · July 2026**

**Context:** GridWorld comparison (500 seeds) approved. Professor instructed extension to a different environment → **CartPole-v1 + Double DQN** (`main_cartpole_dqn.py`).

---

## GridWorld (approved baseline, n = 500)

| | Clean | Under attack* |
|--|-------|---------------|
| Vanilla | 7.35 ± 0.24 | −682 ± 329 |
| Pessimistic | 7.35 ± 0.27 | **−22 ± 50** |
| Off-policy | 7.37 ± 0.23 | −552 ± 335 |

*Vanilla vs pessimistic's attacker; pes/off vs own attackers. Fixed attackers across seeds. Pessimistic inference ~11× slower under attack.

---

## CartPole fair eval (extension, n = 5)

| Metric | Vanilla | Pessimistic | Off-policy |
|--------|---------|-------------|------------|
| **Clean** | 421 ± 176 | **485 ± 33** | **485 ± 20** |
| **Matched attacker** | 239 ± 247 | **246 ± 238** | 153 ± 136 |
| **Shared random δ** | 344 ± 224 | 329 ± 197 | **365 ± 206** |
| **Worst-case (32-δ)*** | 232 ± 182 | **280 ± 141** | 243 ± 124 |

*Defender-specific worst case, not shared PGD. Pessimistic inference ~52× slower under matched attack (2.0 vs 0.04 ms/step).

**Per-seed clean (V/P/O):** 0: 500/500/500 · 1: 500/500/500 · 2: **107**/500/500 · 3: 500/500/464 · 4: 500/426/461

---

## Takeaways (descriptive only)

1. Extension **runs end-to-end**; pessimistic fine-tune stable after v2 fixes (checkpoint restore, frozen backbone, early stop).
2. **No single ordering** across CartPole metrics (unlike GridWorld's one attack column).
3. Pessimistic **highest mean** on worst-case eval; off-policy **highest mean** on shared-random; off-policy **lowest** on matched attacker.
4. **Robustness–speed tradeoff** persists (maximin vs greedy inference).
5. **Not** full Sun & Zheng BP-DQN (simplified belief, CartPole not continuous Gridworld, n = 5).

---

## Files & reproduce

| File | Role |
|------|------|
| `cartpole_experiments_fair_v2.csv` | Results (dedupe by seed) |
| `main_cartpole_dqn.txt` → rename to `.py` | Code |
| `cartpole_belief.py` | Belief RNN |
| `gridworld_experiments_500runs.csv` | Approved baseline |

```bash
pip install -r requirements.txt
cp main_cartpole_dqn.txt main_cartpole_dqn.py
python3 main_cartpole_dqn.py --step 8 --csv cartpole_experiments_fair_v2.csv
```

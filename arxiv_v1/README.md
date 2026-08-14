# ArXiv v1 manuscript

**Pessimistic-Behavior / Greedy-Target Q-Learning against Adversarial State Perturbations**  
Adithya Venkatesh & Zizhan Zheng (Tulane)

Draft for advisor review before arXiv (`cs.LG`). Claims are scoped: GridWorld n=500 is the main result; CartPole n=5 is preliminary; PB-GT is **not** claimed to match BP-DQN robustness.

## Compile (Overleaf — recommended)

1. Create a new Overleaf project.
2. Upload this folder (`main.tex` + `figures/`).
3. Optional: switch `\documentclass` to the [ICLR 2024 style](https://github.com/ICLR/Master-Template) if you want camera-ready look (`iclr2024_conference.sty`).
4. Compile with **pdfLaTeX**.

Locally (if TeX is installed):

```bash
cd arxiv_v1
python3 build_figures.py
pdflatex main.tex && pdflatex main.tex
```

## Regenerate figures from CSVs

```bash
MPLCONFIGDIR=/tmp/mpl python3 arxiv_v1/build_figures.py
```

Uses:
- `../gridworld_experiments_500runs.csv` (n=500)
- `../cartpole_experiments_restore_5runs.csv` (n=5, restore-best vanilla)

## Before sending to Prof. Zheng

- [ ] Fill Adithya affiliation / email on the title page
- [ ] Confirm author order
- [ ] He reviews claims in §5–6 (limitations)
- [ ] ArXiv submit only after his OK (he should submit or endorse `cs.LG`)

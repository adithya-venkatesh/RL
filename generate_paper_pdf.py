#!/usr/bin/env python3
"""Generate arXiv-style draft PDF: GridWorld + CartPole tables/figures + PB-GT motivation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
OUT_PDF = ROOT / "paper.pdf"
FIG_DIR = ROOT / "paper_figures"
GW_CSV = ROOT / "gridworld_experiments_500runs.csv"
CP_CSV = ROOT / "cartpole_experiments_restore_5runs.csv"


def load_results():
    gw = pd.read_csv(GW_CSV)
    cp = pd.read_csv(CP_CSV).drop_duplicates("seed", keep="last")
    return gw, cp


def make_figures(gw: pd.DataFrame, cp: pd.DataFrame) -> dict[str, Path]:
    FIG_DIR.mkdir(exist_ok=True)
    paths: dict[str, Path] = {}
    colors = {"vanilla": "#4C72B0", "pessimistic": "#C44E52", "offpolicy": "#55A868"}

    # Fig 1: GridWorld clean vs attack
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    methods = ["Vanilla", "Pessimistic", "PB-GT"]
    clean = [
        gw["vanilla_test_avg_reward"].mean(),
        gw["pessimistic_test_avg_reward"].mean(),
        gw["offpolicy_greedy_test_avg_reward"].mean(),
    ]
    clean_std = [
        gw["vanilla_test_avg_reward"].std(),
        gw["pessimistic_test_avg_reward"].std(),
        gw["offpolicy_greedy_test_avg_reward"].std(),
    ]
    attack = [
        gw["vanilla_vs_attacker_reward"].mean(),
        gw["pessimistic_vs_attacker_reward"].mean(),
        gw["offpolicy_greedy_vs_attacker_reward"].mean(),
    ]
    attack_std = [
        gw["vanilla_vs_attacker_reward"].std(),
        gw["pessimistic_vs_attacker_reward"].std(),
        gw["offpolicy_greedy_vs_attacker_reward"].std(),
    ]
    x = np.arange(3)
    c = [colors["vanilla"], colors["pessimistic"], colors["offpolicy"]]
    axes[0].bar(x, clean, yerr=clean_std, capsize=3, color=c, edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylabel("Mean episode return")
    axes[0].set_title("Clean (n=500)")
    axes[1].bar(x, attack, yerr=attack_std, capsize=3, color=c, edgecolor="black", linewidth=0.5)
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_title("Under trained DQN attacker")
    fig.suptitle("Tabular GridWorld (500 seeds)", fontsize=12)
    fig.tight_layout()
    p1 = FIG_DIR / "fig1_gridworld.png"
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["gridworld"] = p1

    # Fig 2: CartPole multi-metric
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.6), sharey=True)
    suites = [
        ("Clean", "vanilla_test_avg_reward", "pessimistic_test_avg_reward", "offpolicy_greedy_test_avg_reward"),
        ("Matched\nattacker", "vanilla_vs_matched_attacker_avg_reward",
         "pessimistic_vs_matched_attacker_avg_reward", "offpolicy_vs_matched_attacker_avg_reward"),
        ("Shared\nrandom delta", "vanilla_vs_shared_random_avg_reward",
         "pessimistic_vs_shared_random_avg_reward", "offpolicy_vs_shared_random_avg_reward"),
        ("PGD*", "vanilla_pgd_reward", "pessimistic_pgd_reward", "offpolicy_pgd_reward"),
    ]
    method_labels = ["Van", "Pes", "PB-GT"]
    for ax, (title, vc, pc, oc) in zip(axes, suites):
        vals = [cp[vc].mean(), cp[pc].mean(), cp[oc].mean()]
        errs = [cp[vc].std(), cp[pc].std(), cp[oc].std()]
        ax.bar(method_labels, vals, yerr=errs, capsize=2.5, color=c, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 550)
    axes[0].set_ylabel("Mean return (100 eval ep)")
    fig.suptitle("CartPole-v1 DQN (5 seeds, restore-best vanilla)", fontsize=11, y=1.03)
    fig.tight_layout()
    p2 = FIG_DIR / "fig2_cartpole.png"
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["cartpole"] = p2

    # Fig 3: inference speed
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    envs = ["GridWorld\n(tabular)", "CartPole\n(DQN)"]
    pes = [
        gw["pessimistic_attack_inference_time_ms"].mean(),
        cp["pessimistic_vs_matched_attacker_inference_time_ms"].mean(),
    ]
    off = [
        gw["offpolicy_attack_inference_time_ms"].mean(),
        cp["offpolicy_vs_matched_attacker_inference_time_ms"].mean(),
    ]
    w = 0.35
    x = np.arange(len(envs))
    ax.bar(x - w / 2, pes, w, label="Pessimistic (maximin)", color=colors["pessimistic"], edgecolor="black")
    ax.bar(x + w / 2, off, w, label="PB-GT (greedy)", color=colors["offpolicy"], edgecolor="black")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.set_ylabel("Inference time per step (ms, log)")
    ax.set_title("Attack-time inference cost")
    ax.legend(fontsize=8)
    for i, (p, o) in enumerate(zip(pes, off)):
        ax.text(i, max(p, o) * 2.2, f"{p / o:.0f}x", ha="center", fontsize=9)
    fig.tight_layout()
    p3 = FIG_DIR / "fig3_inference.png"
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    paths["speed"] = p3

    # Fig 4: CartPole per-seed clean
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    seeds = cp["seed"].astype(int).tolist()
    width = 0.25
    xs = np.arange(len(seeds))
    ax.bar(xs - width, cp["vanilla_test_avg_reward"], width, label="Vanilla", color=colors["vanilla"], edgecolor="black")
    ax.bar(xs, cp["pessimistic_test_avg_reward"], width, label="Pessimistic", color=colors["pessimistic"], edgecolor="black")
    ax.bar(xs + width, cp["offpolicy_greedy_test_avg_reward"], width, label="PB-GT", color=colors["offpolicy"], edgecolor="black")
    ax.axhline(475, color="gray", linestyle="--", linewidth=0.8, label="Solve (>=475)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Clean return")
    ax.set_ylim(0, 520)
    ax.set_title("CartPole clean return by seed (post vanilla restore-best)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p4 = FIG_DIR / "fig4_cartpole_per_seed.png"
    fig.savefig(p4, dpi=200)
    plt.close(fig)
    paths["cartpole_seeds"] = p4

    return paths


class PaperPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(22, 16, 22)

    def section_title(self, title: str):
        self.ln(3)
        self.set_font("Helvetica", "B", 12)
        self.multi_cell(0, 6, title)
        self.ln(1)

    def body(self, text: str):
        self.set_font("Helvetica", "", 9.5)
        self.multi_cell(0, 4.8, text)
        self.ln(1.5)

    def add_table(self, headers: list[str], rows: list[list[str]], col_widths: list[float] | None = None):
        self.set_font("Helvetica", "B", 8)
        if col_widths is None:
            col_widths = [160 / len(headers)] * len(headers)
        for h, w in zip(headers, col_widths):
            self.cell(w, 6, h, border=1, align="C")
        self.ln()
        self.set_font("Helvetica", "", 8)
        for row in rows:
            for cell, w in zip(row, col_widths):
                self.cell(w, 5.5, cell, border=1, align="C")
            self.ln()
        self.ln(2)


def fmt(mean: float, std: float, nd: int = 1) -> str:
    return f"{mean:.{nd}f} +/- {std:.{nd}f}"


def build_pdf(gw: pd.DataFrame, cp: pd.DataFrame, figures: dict[str, Path]) -> None:
    pdf = PaperPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(
        0,
        7,
        "Draft: Pessimistic and PB-GT Q-Learning under\nAdversarial State Perturbations",
    )
    pdf.ln(1)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 4.5, "Adithya Venkatesh - Prof. Zizhan Zheng - July 2026 (draft for feedback)")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 5, "Abstract")
    pdf.ln(5)
    pdf.body(
        "We study belief-enriched pessimistic Q-learning (Sun & Zheng, ICLR 2024) and a "
        "Pessimistic-Behavior / Greedy-Target (PB-GT) variant that uses maximin behavior during "
        "training but greedy Double-DQN targets and greedy inference at test time. On tabular "
        "GridWorld (500 seeds), pessimistic Q-learning dominates under trained attackers "
        f"({fmt(gw['pessimistic_vs_attacker_reward'].mean(), gw['pessimistic_vs_attacker_reward'].std())} "
        f"vs {fmt(gw['offpolicy_greedy_vs_attacker_reward'].mean(), gw['offpolicy_greedy_vs_attacker_reward'].std())} "
        f"PB-GT and {fmt(gw['vanilla_vs_attacker_reward'].mean(), gw['vanilla_vs_attacker_reward'].std())} vanilla), "
        "while PB-GT is ~11x faster than maximin at attack-time inference. On CartPole-v1 with "
        "Double DQN (5 seeds, vanilla best-checkpoint restore), vanilla converges reliably "
        f"({fmt(cp['vanilla_test_avg_reward'].mean(), cp['vanilla_test_avg_reward'].std())}, 5/5 solved), "
        "but BP-style pessimistic and PB-GT show high seed variance and do not yet reproduce the "
        "GridWorld robustness ordering. We treat CartPole as preliminary and outline next steps: "
        "continuous GridWorld comparison to BP-DQN, CAR-DQN (Li et al., 2024) as a baseline, "
        "then Atari (Pong/Freeway)."
    )

    # ---- Tables & interpretation ----
    pdf.section_title("1. GridWorld results (500 seeds) and interpretation")
    pdf.add_table(
        ["Method", "Clean", "Under trained attacker", "Attack infer. (ms)"],
        [
            [
                "Vanilla",
                fmt(gw["vanilla_test_avg_reward"].mean(), gw["vanilla_test_avg_reward"].std(), 2),
                fmt(gw["vanilla_vs_attacker_reward"].mean(), gw["vanilla_vs_attacker_reward"].std(), 1),
                f"{gw['vanilla_attack_inference_time_ms'].mean():.3f}",
            ],
            [
                "Pessimistic",
                fmt(gw["pessimistic_test_avg_reward"].mean(), gw["pessimistic_test_avg_reward"].std(), 2),
                fmt(gw["pessimistic_vs_attacker_reward"].mean(), gw["pessimistic_vs_attacker_reward"].std(), 1),
                f"{gw['pessimistic_attack_inference_time_ms'].mean():.3f}",
            ],
            [
                "PB-GT",
                fmt(gw["offpolicy_greedy_test_avg_reward"].mean(), gw["offpolicy_greedy_test_avg_reward"].std(), 2),
                fmt(gw["offpolicy_greedy_vs_attacker_reward"].mean(), gw["offpolicy_greedy_vs_attacker_reward"].std(), 1),
                f"{gw['offpolicy_attack_inference_time_ms'].mean():.3f}",
            ],
        ],
        col_widths=[32, 40, 55, 40],
    )
    pdf.body(
        "Interpretation. Clean returns are essentially identical (~7.35). Under attack, the "
        "ordering is pessimistic >> PB-GT > vanilla. Pessimistic wins over PB-GT and vanilla in "
        ">96% of seeds; PB-GT beats vanilla in ~61% of seeds. This supports the paper's claim "
        "that maximin belief-restricted policies are highly robust in discrete settings, while "
        "PB-GT is a faster approximate alternative that still improves over vanilla on average. "
        "The main cost is inference: pessimistic maximin is ~11x slower than greedy under attack."
    )
    pdf.image(str(figures["gridworld"]), w=165)
    pdf.ln(2)

    pdf.section_title("2. CartPole DQN results (5 seeds, restore-best vanilla)")
    suites = [
        ("Clean", "vanilla_test_avg_reward", "pessimistic_test_avg_reward", "offpolicy_greedy_test_avg_reward"),
        ("Matched attacker", "vanilla_vs_matched_attacker_avg_reward",
         "pessimistic_vs_matched_attacker_avg_reward", "offpolicy_vs_matched_attacker_avg_reward"),
        ("Shared random delta", "vanilla_vs_shared_random_avg_reward",
         "pessimistic_vs_shared_random_avg_reward", "offpolicy_vs_shared_random_avg_reward"),
        ("PGD* (32-cand.)", "vanilla_pgd_reward", "pessimistic_pgd_reward", "offpolicy_pgd_reward"),
    ]
    rows = []
    for name, vc, pc, oc in suites:
        rows.append([
            name,
            fmt(cp[vc].mean(), cp[vc].std(), 0),
            fmt(cp[pc].mean(), cp[pc].std(), 0),
            fmt(cp[oc].mean(), cp[oc].std(), 0),
        ])
    pdf.add_table(["Metric", "Vanilla", "Pessimistic (BP-style)", "PB-GT"], rows, [40, 40, 48, 40])
    pdf.body(
        "Interpretation. After adding mid-training eval + best-checkpoint restore for vanilla "
        "(same idea as our pessimistic v2 stabilizers), vanilla clean converges on all 5 seeds "
        f"(mean {cp['vanilla_test_avg_reward'].mean():.1f}). This fixes the earlier seed-2 collapse "
        "(clean 107). However, BP-style pessimistic and PB-GT are no longer stable on clean "
        f"(pes {fmt(cp['pessimistic_test_avg_reward'].mean(), cp['pessimistic_test_avg_reward'].std(), 0)}; "
        f"PB-GT {fmt(cp['offpolicy_greedy_test_avg_reward'].mean(), cp['offpolicy_greedy_test_avg_reward'].std(), 0)}), "
        "and under matched attackers pessimistic returns ~11 on every seed. We do NOT claim "
        "GridWorld-style robustness on CartPole. Likely causes: (i) fine-tune hyperparams were "
        "tuned before vanilla restore and need retuning; (ii) simplified GRU belief != PF-RNN "
        "BP-DQN; (iii) matched attackers are retrained vs each defender; (iv) n=5 is small. "
        "We keep CartPole as a preliminary DQN port and move the main deep comparison to "
        "continuous GridWorld (Xiaolin's Table 1a setting)."
    )
    pdf.image(str(figures["cartpole"]), w=170)
    pdf.ln(1)
    pdf.image(str(figures["cartpole_seeds"]), w=150)
    pdf.ln(2)

    pdf.section_title("3. Inference efficiency (motivation for PB-GT)")
    pdf.add_table(
        ["Environment", "Pessimistic (ms)", "PB-GT (ms)", "Slowdown"],
        [
            [
                "GridWorld attack",
                f"{gw['pessimistic_attack_inference_time_ms'].mean():.3f}",
                f"{gw['offpolicy_attack_inference_time_ms'].mean():.3f}",
                f"{gw['pessimistic_attack_inference_time_ms'].mean() / gw['offpolicy_attack_inference_time_ms'].mean():.1f}x",
            ],
            [
                "CartPole matched",
                f"{cp['pessimistic_vs_matched_attacker_inference_time_ms'].mean():.2f}",
                f"{cp['offpolicy_vs_matched_attacker_inference_time_ms'].mean():.3f}",
                f"{cp['pessimistic_vs_matched_attacker_inference_time_ms'].mean() / cp['offpolicy_vs_matched_attacker_inference_time_ms'].mean():.0f}x",
            ],
        ],
        col_widths=[42, 42, 38, 35],
    )
    pdf.image(str(figures["speed"]), w=145)
    pdf.ln(2)

    # ---- PB-GT motivation ----
    pdf.section_title("4. Motivating PB-GT DQN (and how to improve it)")
    pdf.body(
        "Problem. Full pessimistic / BP-DQN defense uses maximin over a belief set at every "
        "decision. That is robust in GridWorld but slow (~11-50x vs greedy) and hard to "
        "stabilize in deep RL. Many deployments need low-latency inference under moderate "
        "perturbations, not only worst-case maximin at every step.\n\n"
        "PB-GT idea. Keep pessimistic (maximin) BEHAVIOR while collecting data under "
        "perturbed observations, so the replay buffer reflects robust exploration, but use a "
        "standard greedy Double-DQN TARGET on the true next state and a greedy TEST policy. "
        "This is a deliberate behavior-target mismatch (our GridWorld Eq. 5.1-5.2), not "
        "'off-policy DQN' in the classical sense (all three methods are off-policy).\n\n"
        "Why it can help. (1) Inference cost matches vanilla. (2) Training still sees "
        "adversarial observations and maximin actions, which can improve robustness over "
        "vanilla without paying maximin cost at test time. (3) On GridWorld, PB-GT sits "
        "between vanilla and pessimistic under attack while being as fast as vanilla.\n\n"
        "Opportunities to improve PB-GT (next experiments). "
        "(A) Hybrid inference: use greedy by default, switch to maximin when belief entropy "
        "or Q-gap indicates high uncertainty. "
        "(B) Soft maximin / temperature: replace hard min_m with softmin to reduce "
        "over-pessimism during fine-tune. "
        "(C) Separate Q-heads: train a robust head with pessimistic targets and a fast head "
        "with greedy targets; distill or ensemble. "
        "(D) Retune after strong vanilla: shorter/longer fine-tune, unfreeze more layers, "
        "align kappa and PGD with Algorithm 4. "
        "(E) Evaluate on continuous GridWorld against BP-DQN and CAR-DQN under identical "
        "PGD / MinBest / PA-AD protocols before claiming deep-RL gains."
    )

    # ---- Next steps ----
    pdf.section_title("5. Next experimental steps (advisor roadmap)")
    pdf.body(
        "1. Continuous GridWorld (priority). Use Xiaolin's environment and BP-DQN pipeline "
        "(github.com/SliencerX/Belief-enriched-robust-Q-learning; config Grid_continous_ours.json; "
        "train.py / test_gridmaze.py). First reproduce vanilla DQN and BP-DQN Table 1a numbers, "
        "then add PB-GT, then increase seeds for a rigorous comparison.\n\n"
        "2. CAR-DQN baseline (Li et al., arXiv:2402.02165 / ICML). Consistent Adversarial "
        "Robust DQN minimizes a surrogate of Bellman Infinity-error under the Consistency "
        "Assumption of Policy. Code: github.com/leoranlmia/CAR-DQN. Include as an additional "
        "robust DQN baseline on continuous GridWorld (and later Atari) alongside SA-DQN / "
        "WocaR-DQN where feasible.\n\n"
        "3. Then Pong / Freeway + DP-DQN, with GPU workstation support (Yibin).\n\n"
        "4. CartPole: optional retune of pessimistic / PB-GT after vanilla restore; treat as "
        "appendix / preliminary unless continuous GridWorld confirms the narrative."
    )

    pdf.section_title("6. Limitations (current draft)")
    pdf.body(
        "CartPole pessimistic is BP-style (simplified GRU particles, kappa=20), not full "
        "PF-RNN Algorithm 4. PGD* is defender-specific 32-candidate search, not paper PGD "
        "(eta=10). Matched attackers are not a shared threat. n=5 on CartPole. Continuous "
        "GridWorld and CAR-DQN results are not yet included."
    )

    pdf.section_title("References")
    for r in [
        "[1] X. Sun and Z. Zheng. Belief-Enriched Pessimistic Q-Learning Against Adversarial State Perturbations. ICLR 2024.",
        "[2] H. Li et al. Towards Optimal Adversarial Robust Q-learning with Bellman Infinity-error (CAR-DQN). arXiv:2402.02165, 2024.",
        "[3] V. Mnih et al. Human-level control through deep reinforcement learning. Nature, 2015.",
        "[4] Code: SliencerX/Belief-enriched-robust-Q-learning; leoranlmia/CAR-DQN.",
    ]:
        pdf.body(r)

    pdf.section_title("Appendix: reproducibility")
    pdf.body(
        "GridWorld: main_with_vanilla.py, gridworld_experiments_500runs.csv. "
        "CartPole: main_cartpole_dqn.py, cartpole_belief.py, cartpole_experiments_restore_5runs.csv. "
        "Methods writeup: arxiv_prep_algorithms_metrics.pdf. "
        "Regenerate this draft: python3 generate_paper_pdf.py"
    )

    pdf.output(str(OUT_PDF))


def main():
    gw, cp = load_results()
    figures = make_figures(gw, cp)
    build_pdf(gw, cp, figures)
    print(f"Wrote {OUT_PDF}")
    for k, p in figures.items():
        print(f"  figure {k}: {p}")


if __name__ == "__main__":
    main()

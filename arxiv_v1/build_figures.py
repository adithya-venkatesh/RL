#!/usr/bin/env python3
"""Build arXiv v1 figures from GridWorld (n=500) and CartPole restore (n=5) CSVs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
FIG = ROOT / "figures"
GW_CSV = REPO / "gridworld_experiments_500runs.csv"
CP_CSV = REPO / "cartpole_experiments_restore_5runs.csv"

COLORS = {"vanilla": "#4C72B0", "pessimistic": "#C44E52", "pbgt": "#55A868"}


def main() -> None:
    FIG.mkdir(exist_ok=True)
    gw = pd.read_csv(GW_CSV)
    cp = pd.read_csv(CP_CSV).drop_duplicates("seed", keep="last")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 200,
        }
    )

    methods = ["Vanilla", "Pessimistic", "PB-GT"]
    c = [COLORS["vanilla"], COLORS["pessimistic"], COLORS["pbgt"]]

    # Fig 1: GridWorld clean vs attack
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
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
    axes[0].bar(x, clean, yerr=clean_std, capsize=3, color=c, edgecolor="black", linewidth=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylabel("Mean episode return")
    axes[0].set_title("Clean")
    axes[1].bar(x, attack, yerr=attack_std, capsize=3, color=c, edgecolor="black", linewidth=0.6)
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_title("Under trained DQN attacker")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_gridworld_bars.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig1_gridworld_bars.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 2: GridWorld attack boxplots
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    data = [
        gw["vanilla_vs_attacker_reward"].values,
        gw["pessimistic_vs_attacker_reward"].values,
        gw["offpolicy_greedy_vs_attacker_reward"].values,
    ]
    bp = ax.boxplot(data, tick_labels=methods, patch_artist=True, showfliers=False, widths=0.55)
    for patch, col in zip(bp["boxes"], c):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)
        patch.set_edgecolor("black")
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.2)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Episode return under attack")
    ax.set_title("Tabular GridWorld attack returns (n=500 seeds)")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_gridworld_box.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig2_gridworld_box.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 3: inference
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
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
    xx = np.arange(len(envs))
    ax.bar(xx - w / 2, pes, w, label="Pessimistic (maximin)", color=COLORS["pessimistic"], edgecolor="black")
    ax.bar(xx + w / 2, off, w, label="PB-GT (greedy)", color=COLORS["pbgt"], edgecolor="black")
    ax.set_yscale("log")
    ax.set_xticks(xx)
    ax.set_xticklabels(envs)
    ax.set_ylabel("Inference time per step (ms, log)")
    ax.set_title("Attack-time inference cost")
    ax.legend(frameon=False)
    for i, (p, o) in enumerate(zip(pes, off)):
        ax.text(i, max(p, o) * 2.4, f"{p / o:.0f}$\\times$", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_inference.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig3_inference.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 4: CartPole multi-metric
    fig, axes = plt.subplots(1, 4, figsize=(8.8, 3.1), sharey=True)
    suites = [
        ("Clean", "vanilla_test_avg_reward", "pessimistic_test_avg_reward", "offpolicy_greedy_test_avg_reward"),
        ("Matched\nattacker", "vanilla_vs_matched_attacker_avg_reward",
         "pessimistic_vs_matched_attacker_avg_reward", "offpolicy_vs_matched_attacker_avg_reward"),
        ("Shared\nrandom $\\delta$", "vanilla_vs_shared_random_avg_reward",
         "pessimistic_vs_shared_random_avg_reward", "offpolicy_vs_shared_random_avg_reward"),
        ("PGD*", "vanilla_pgd_reward", "pessimistic_pgd_reward", "offpolicy_pgd_reward"),
    ]
    labels = ["Van", "Pes", "PB-GT"]
    for ax, (title, vc, pc, oc) in zip(axes, suites):
        vals = [cp[vc].mean(), cp[pc].mean(), cp[oc].mean()]
        errs = [cp[vc].std(), cp[pc].std(), cp[oc].std()]
        ax.bar(labels, vals, yerr=errs, capsize=2.5, color=c, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 550)
    axes[0].set_ylabel("Mean return (100 eval ep.)")
    fig.suptitle("CartPole-v1 Double DQN (5 seeds; preliminary)", fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_cartpole.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig4_cartpole.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 5: CartPole per-seed clean
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    seeds = cp["seed"].astype(int).tolist()
    width = 0.25
    xs = np.arange(len(seeds))
    ax.bar(xs - width, cp["vanilla_test_avg_reward"], width, label="Vanilla", color=COLORS["vanilla"], edgecolor="black")
    ax.bar(xs, cp["pessimistic_test_avg_reward"], width, label="Pessimistic", color=COLORS["pessimistic"], edgecolor="black")
    ax.bar(xs + width, cp["offpolicy_greedy_test_avg_reward"], width, label="PB-GT", color=COLORS["pbgt"], edgecolor="black")
    ax.axhline(475, color="gray", linestyle="--", linewidth=0.8, label="Solve ($\\geq$475)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Clean return")
    ax.set_ylim(0, 530)
    ax.set_title("CartPole clean return by seed")
    ax.legend(fontsize=7.5, loc="lower right", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_cartpole_seeds.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig5_cartpole_seeds.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote figures to {FIG}")


if __name__ == "__main__":
    main()

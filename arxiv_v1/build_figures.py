#!/usr/bin/env python3
"""Build arXiv v1 figures from tabular GridWorld, continuous GridWorld, and CartPole CSVs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
FIG = ROOT / "figures"
GW_CSV = REPO / "gridworld_experiments_500runs_final_v3.csv"
SWAP_CSV = REPO / "gridworld_target_test_swap_tabular_matched15_v3.csv"
W16_CSV = REPO / "gridworld_worst_of_16_v3.csv"
BA_CSV = REPO / "gridworld_belief_aware_attack_v3.csv"
GW_INF_CSV = REPO / "gridworld_inference_20seeds_final_v3.csv"
RETIME_CSV = REPO / "inference_retime_singleproc_v3.csv"
CP_CSV = REPO / "cartpole_experiments_restore_5runs.csv"
CGW_CSV = REPO / "gw_prod_eps025_10seeds_v3.csv"

COLORS = {
    "vanilla": "#4C72B0",
    "pessimistic": "#C44E52",
    "pes_greedy": "#E39C3B",
    "pbgt": "#55A868",
    "pbgt_maximin": "#2A706C",
    "ball": "#8172B3",
}
# Method order matches Table 3. Fig1 uses single-line rotated labels; fig2 keeps stacked ticks.
FIG1_BAR_LABELS = ["Vanilla", "Pes. maximin", "Pes. greedy", "PB-GT greedy", "PB-GT maximin"]
BAR_LABELS = ["Vanilla", "Pes.\nmaximin", "Pes.\ngreedy", "PB-GT\ngreedy", "PB-GT\nmaximin"]
ORIG9 = [f"{p}{i}" for p in ("van", "pes", "pbgt") for i in range(3)]
METHOD_COLORS = [
    COLORS["vanilla"],
    COLORS["pessimistic"],
    COLORS["pes_greedy"],
    COLORS["pbgt"],
    COLORS["pbgt_maximin"],
]
CGW_VARIANTS = ["vanilla", "pessimistic_box", "pbgt_box", "pessimistic_ball"]
CGW_LABELS = ["Vanilla", "Pes. (box)", "PB-GT", "Pes. (ball)"]
CGW_COLORS = [COLORS["vanilla"], COLORS["pessimistic"], COLORS["pbgt"], COLORS["ball"]]
CGW_CONDS = [
    ("clean", "Clean"),
    ("shared_random", "Shared rand."),
    ("pgd", "PGD-10"),
    ("matched_search", "Matched search"),
    ("matched_sb3", "Matched SB3"),
]


def method_handles() -> list[Patch]:
    return [
        Patch(facecolor=COLORS["vanilla"], edgecolor="black", linewidth=0.6, label="Vanilla"),
        Patch(facecolor=COLORS["pessimistic"], edgecolor="black", linewidth=0.6, label="Pessimistic"),
        Patch(facecolor=COLORS["pbgt"], edgecolor="black", linewidth=0.6, label="PB-GT"),
    ]


def cgw_handles() -> list[Patch]:
    return [
        Patch(facecolor=c, edgecolor="black", linewidth=0.6, label=lab)
        for c, lab in zip(CGW_COLORS, CGW_LABELS)
    ]


def seed_mean_std(df: pd.DataFrame, variant: str, condition: str) -> tuple[float, float]:
    vals = df[(df["variant"] == variant) & (df["condition"] == condition)]["mean"].to_numpy()
    return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)


def save(fig: plt.Figure, name: str, pdf_path: Path | None = None) -> None:
    if pdf_path is not None:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        return
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def main(fig1_only: Path | None = None) -> None:
    FIG.mkdir(exist_ok=True)
    gw = pd.read_csv(GW_CSV).sort_values("seed").reset_index(drop=True)
    swap = pd.read_csv(SWAP_CSV).sort_values("seed").reset_index(drop=True)
    w16 = pd.read_csv(W16_CSV).sort_values("seed").reset_index(drop=True)
    ba = pd.read_csv(BA_CSV).sort_values("seed").reset_index(drop=True)
    gw_inf = pd.read_csv(GW_INF_CSV)
    retime = pd.read_csv(RETIME_CSV)
    cp = pd.read_csv(CP_CSV).drop_duplicates("seed", keep="last")
    cgw = pd.read_csv(CGW_CSV)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )

    if not np.array_equal(gw["seed"].to_numpy(), swap["seed"].to_numpy()):
        raise SystemExit("tabular seed mismatch between final_v3 and matched15")
    if not np.array_equal(gw["seed"].to_numpy(), w16["seed"].to_numpy()):
        raise SystemExit("tabular seed mismatch between final_v3 and worst_of_16")
    if not np.array_equal(gw["seed"].to_numpy(), ba["seed"].to_numpy()):
        raise SystemExit("tabular seed mismatch between final_v3 and belief_aware")

    # Clean uses the deployed rule. Matched: BA own-3 for maximin rows, belief-blind otherwise.
    series = {
        "clean": [
            gw["vanilla_clean"],
            swap["pes_q_maximin_clean"],
            swap["pes_q_greedy_clean"],
            swap["pbgt_q_greedy_clean"],
            swap["pbgt_q_maximin_clean"],
        ],
        "matched": [
            gw["van_matched_mean_of_3"],
            ba["pes_q_maximin_ba_matched"],
            swap["pes_q_greedy_matched"],
            swap["pbgt_q_greedy_matched"],
            ba["pbgt_q_maximin_ba_matched"],
        ],
        "worst22": [
            ba["vanilla_worst22"],
            ba["pes_q_maximin_worst22"],
            ba["pes_q_greedy_worst22"],
            ba["pbgt_q_greedy_worst22"],
            ba["pbgt_q_maximin_worst22"],
        ],
    }

    def _ms(col):
        if col is None:
            return np.nan, np.nan
        return float(col.mean()), float(col.std(ddof=1))

    def _draw_bars(ax, stats, title: str) -> None:
        keep = [i for i, (m, _) in enumerate(stats) if np.isfinite(m)]
        means = [stats[i][0] for i in keep]
        stds = [stats[i][1] for i in keep]
        xpos = np.arange(len(keep))
        ax.bar(
            xpos,
            means,
            width=0.72,
            yerr=stds,
            capsize=2.5,
            color=[METHOD_COLORS[i] for i in keep],
            edgecolor="black",
            linewidth=0.6,
            error_kw={"elinewidth": 0.8, "capthick": 0.8, "clip_on": False},
            clip_on=False,
        )
        ax.set_xticks(xpos)
        ax.set_xticklabels(
            [FIG1_BAR_LABELS[i] for i in keep],
            fontsize=7,
            rotation=35,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_title(title, fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        lo = min(m - s for m, s in zip(means, stds))
        hi = max(m + s for m, s in zip(means, stds))
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        pad = 0.08 * (hi - lo if hi > lo else 1.0)
        ax.set_ylim(min(lo - pad, -1000.0), hi + pad)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=False)
    panels = [
        ("Clean (deployed)", [_ms(c) for c in series["clean"]]),
        ("Matched (own 3)", [_ms(c) for c in series["matched"]]),
        ("Worst of 22", [_ms(c) for c in series["worst22"]]),
    ]
    for ax, (title, stats) in zip(axes, panels):
        _draw_bars(ax, stats, title)
    axes[0].set_ylabel("Mean episode return")
    fig.tight_layout(w_pad=0.8)
    if fig1_only is not None:
        # Print Table-3 stats for verification, then write only fig1 to the given path.
        for title, stats in panels:
            print(title + ":")
            for lab, (m, s) in zip(FIG1_BAR_LABELS, stats):
                print(f"  {lab}: {m:.2f}±{s:.2f}")
        save(fig, "fig1_gridworld_bars", pdf_path=fig1_only)
        print(f"Wrote fig1 to {fig1_only}")
        return
    save(fig, "fig1_gridworld_bars")

    # Fig 2: median/IQR boxplots. Fliers kept so the tails are not clipped.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), sharey=True)
    box_panels = [
        ("Matched (own 3)", series["matched"]),
        ("Worst of 22", series["worst22"]),
    ]
    pooled = []
    for ax, (title, cols) in zip(axes, box_panels):
        data = [c.to_numpy(dtype=float) for c in cols if c is not None]
        labels = [lab for c, lab in zip(cols, BAR_LABELS) if c is not None]
        colors = [col for c, col in zip(cols, METHOD_COLORS) if c is not None]
        pooled.extend(data)
        bp = ax.boxplot(
            data,
            tick_labels=labels,
            patch_artist=True,
            showfliers=True,
            widths=0.58,
            flierprops={"marker": "o", "markersize": 2.2, "alpha": 0.35},
        )
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.9)
            patch.set_edgecolor("black")
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.2)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelsize=7.0, pad=3)
        for lab in ax.get_xticklabels():
            lab.set_rotation(0)
            lab.set_ha("center")
            lab.set_linespacing(0.9)
    ys = np.concatenate(pooled)
    pad = 0.05 * (float(ys.max()) - float(ys.min()))
    for ax in axes:
        ax.set_ylim(float(ys.min()) - pad, float(ys.max()) + pad)
    axes[0].set_ylabel("Return under attack")
    fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=1.4)
    save(fig, "fig2_gridworld_box")

    # Fig 3: single-process timings of the deployed rules.
    cp_box = pd.read_csv(REPO / "cartpole_trained_box_5seeds_v3.csv")
    cp_box = cp_box.loc[cp_box["row_kind"] == "seed"]
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    envs = ["Tabular GW", "Cont. GW", "CartPole"]

    def _rt(method: str) -> float:
        vals = retime[(retime["domain"] == "cgw") & (retime["method"] == method)]["ms_per_action"]
        return float(vals.mean())

    van = [
        gw_inf["vanilla_attack_inference_time_ms"].mean(),
        _rt("vanilla"),
        cp_box["vanilla_infer_ms_mean"].astype(float).mean(),
    ]
    pes = [
        gw_inf["pessimistic_attack_inference_time_ms"].mean(),
        _rt("pessimistic_box"),
        cp_box["pessimistic_infer_ms_mean"].astype(float).mean(),
    ]
    pbgt = [
        gw_inf["offpolicy_attack_inference_time_ms"].mean(),
        _rt("pbgt_box"),
        cp_box["pbgt_infer_ms_mean"].astype(float).mean(),
    ]
    w = 0.25
    xx = np.arange(len(envs))
    ax.bar(xx - w, van, w, color=COLORS["vanilla"], edgecolor="black", linewidth=0.6)
    ax.bar(xx, pes, w, color=COLORS["pessimistic"], edgecolor="black", linewidth=0.6)
    ax.bar(xx + w, pbgt, w, color=COLORS["pbgt"], edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xticks(xx)
    ax.set_xticklabels(envs)
    ax.set_ylabel("Inference time (ms)")
    ax.legend(handles=method_handles(), frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig3_inference")

    # Fig 6: continuous GridWorld, 10 seeds, eps=0.25. Clean is the deployed rule.
    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    n_cond = len(CGW_CONDS)
    n_var = len(CGW_VARIANTS)
    width = 0.18
    xs = np.arange(n_cond)
    for i, variant in enumerate(CGW_VARIANTS):
        means, stds = zip(*[seed_mean_std(cgw, variant, cond) for cond, _ in CGW_CONDS])
        ax.bar(
            xs + (i - (n_var - 1) / 2) * width,
            means,
            width,
            yerr=stds,
            capsize=2.0,
            color=CGW_COLORS[i],
            edgecolor="black",
            linewidth=0.5,
            label=CGW_LABELS[i],
        )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in CGW_CONDS])
    ax.set_ylabel("Mean episode return")
    ax.legend(handles=cgw_handles(), loc="lower left", frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    save(fig, "fig6_gridworld_cont")

    # Fig 4: frozen-Q sound box (supporting CartPole result)
    cps = pd.read_csv(REPO / "cartpole_sound_5seeds.csv")
    sound_rules = ["greedy", "uniform", "sound", "sound_ibp"]
    sound_labels = ["Greedy", "Uniform", "Sound box", "Sound+IBP"]
    sound_colors = [COLORS["vanilla"], COLORS["ball"], COLORS["pessimistic"], COLORS["pbgt"]]
    sound_conds = [("clean", "Clean"), ("shared", "Shared"), ("pgd", "PGD*")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    xs = np.arange(len(sound_rules))
    for ax, (col, title) in zip(axes, sound_conds):
        means, stds = [], []
        for rule in sound_rules:
            vals = cps.loc[cps["rule"] == rule, col].to_numpy(dtype=float)
            means.append(float(vals.mean()))
            stds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
        ax.bar(xs, means, yerr=stds, capsize=2.5, color=sound_colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_ylim(0, 620)
    axes[0].set_ylabel("Mean return")
    fig.legend(
        handles=[
            Patch(facecolor=c, edgecolor="black", linewidth=0.6, label=lab)
            for c, lab in zip(sound_colors, sound_labels)
        ],
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.10),
    )
    fig.tight_layout()
    save(fig, "fig4_cartpole")

    # Fig 5: per-seed clean — legend above the plot, methods then solve line
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    seeds = cp["seed"].astype(int).tolist()
    width = 0.25
    xs = np.arange(len(seeds))
    ax.bar(xs - width, cp["vanilla_test_avg_reward"], width, color=COLORS["vanilla"], edgecolor="black", linewidth=0.6)
    ax.bar(xs, cp["pessimistic_test_avg_reward"], width, color=COLORS["pessimistic"], edgecolor="black", linewidth=0.6)
    ax.bar(xs + width, cp["offpolicy_greedy_test_avg_reward"], width, color=COLORS["pbgt"], edgecolor="black", linewidth=0.6)
    ax.axhline(475, color="gray", linestyle="--", linewidth=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Seed {s}" for s in seeds])
    ax.set_ylabel("Clean return")
    ax.set_ylim(0, 530)
    solve = Line2D([0], [0], color="gray", linestyle="--", linewidth=0.9, label=r"Solve ($\geq$475)")
    ax.legend(
        handles=method_handles() + [solve],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.tight_layout()
    save(fig, "fig5_cartpole_seeds")

    print(f"Wrote figures to {FIG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fig1-only",
        type=Path,
        default=None,
        help="Build only fig1 and write the PDF to this path (no other figures).",
    )
    args = parser.parse_args()
    # Optional override used by verification scripts.
    fig1_only = args.fig1_only or (Path(os.environ["FIG1_ONLY"]) if os.environ.get("FIG1_ONLY") else None)
    main(fig1_only=fig1_only)

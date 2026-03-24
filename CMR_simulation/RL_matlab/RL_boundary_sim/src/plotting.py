"""
Produce both individual plots and a compact comparison panel.

All public functions return a matplotlib Figure so callers can
``fig.savefig(...)`` or display inline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# House style
_PALETTE = {"baseline": "#4C72B0", "H1_smaller_boundary": "#DD8452", "H2_global_reduction": "#55A868"}
_HYP_ORDER = ["baseline", "H1_smaller_boundary", "H2_global_reduction"]
_HYP_LABELS = {"baseline": "H0 Baseline", "H1_smaller_boundary": "H1 Smaller Δ", "H2_global_reduction": "H2 Global ↓"}


def _style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})


def _hyp_label(name: str) -> str:
    return _HYP_LABELS.get(name, name)


# ===================================================================
# Individual plots
# ===================================================================

def plot_overall_recall(subject_df: pd.DataFrame) -> plt.Figure:
    """Bar plot of mean overall recall ± SE by hypothesis."""
    _style()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(
        data=subject_df, x="hypothesis", y="overall_recall",
        order=_HYP_ORDER, palette=_PALETTE, errorbar="se",
        capsize=0.15, ax=ax,
    )
    ax.set_xticklabels([_hyp_label(h) for h in _HYP_ORDER])
    ax.set_ylabel("Overall Recall")
    ax.set_xlabel("")
    ax.set_title("Overall Recall by Hypothesis")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def plot_boundary_recall(subject_df: pd.DataFrame) -> plt.Figure:
    """Grouped bar: boundary vs non‑boundary recall by hypothesis."""
    _style()
    melted = subject_df.melt(
        id_vars=["hypothesis", "subject_id"],
        value_vars=["boundary_recall", "nonboundary_recall"],
        var_name="item_type", value_name="recall",
    )
    melted["item_type"] = melted["item_type"].map(
        {"boundary_recall": "Boundary", "nonboundary_recall": "Non-boundary"}
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=melted, x="hypothesis", y="recall", hue="item_type",
        order=_HYP_ORDER, errorbar="se", capsize=0.1, ax=ax,
    )
    ax.set_xticklabels([_hyp_label(h) for h in _HYP_ORDER])
    ax.set_ylabel("Recall")
    ax.set_xlabel("")
    ax.set_title("Boundary vs Non-Boundary Recall")
    ax.set_ylim(0, 1)
    ax.legend(title="")
    fig.tight_layout()
    return fig


def plot_boundary_advantage(subject_df: pd.DataFrame) -> plt.Figure:
    """Bar plot of boundary advantage by hypothesis."""
    _style()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(
        data=subject_df, x="hypothesis", y="boundary_advantage",
        order=_HYP_ORDER, palette=_PALETTE, errorbar="se",
        capsize=0.15, ax=ax,
    )
    ax.set_xticklabels([_hyp_label(h) for h in _HYP_ORDER])
    ax.set_ylabel("Boundary Advantage (Δ recall)")
    ax.set_xlabel("")
    ax.set_title("Boundary Advantage")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    return fig


def plot_serial_position(spc_df: pd.DataFrame) -> plt.Figure:
    """Line plot of serial‑position curves."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))
    for hyp in _HYP_ORDER:
        sub = spc_df[spc_df["hypothesis"] == hyp]
        ax.plot(sub["within_list_pos"], sub["recall"],
                label=_hyp_label(hyp), color=_PALETTE[hyp], linewidth=1.5)
    ax.set_xlabel("Within‑List Position")
    ax.set_ylabel("P(Recall)")
    ax.set_title("Serial Position Curves")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_drift_distributions(trial_df: pd.DataFrame) -> plt.Figure:
    """Violin / box plots of encoding drift by boundary flag and hypothesis."""
    _style()
    trial_df = trial_df.copy()
    trial_df["Item Type"] = trial_df["is_boundary"].map({0: "Non-boundary", 1: "Boundary"})
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(
        data=trial_df, x="hypothesis", y="encoding_drift",
        hue="Item Type", order=_HYP_ORDER,
        showfliers=False, ax=ax,
    )
    ax.set_xticklabels([_hyp_label(h) for h in _HYP_ORDER])
    ax.set_ylabel("Encoding Drift")
    ax.set_xlabel("")
    ax.set_title("Encoding Drift Distributions")
    ax.legend(title="")
    fig.tight_layout()
    return fig


def plot_rpe_distributions(trial_df: pd.DataFrame) -> plt.Figure:
    """Histogram / KDE of |outcome RPE| by hypothesis (sanity check)."""
    _style()
    fig, ax = plt.subplots(figsize=(6, 4))
    for hyp in _HYP_ORDER:
        sub = trial_df[trial_df["hypothesis"] == hyp]
        sns.kdeplot(sub["abs_outcome_rpe"], label=_hyp_label(hyp),
                    color=_PALETTE[hyp], ax=ax, linewidth=1.5)
    ax.set_xlabel("|Outcome RPE|")
    ax.set_title("RPE Distributions (should overlap)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_recall_by_train(train_df: pd.DataFrame) -> plt.Figure:
    """Mean within‑train recall across subjects, by hypothesis."""
    _style()
    agg = (
        train_df.groupby(["hypothesis", "train_id"])["train_recall"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    for hyp in _HYP_ORDER:
        sub = agg[agg["hypothesis"] == hyp]
        ax.plot(sub["train_id"], sub["train_recall"],
                label=_hyp_label(hyp), color=_PALETTE[hyp], linewidth=1.3)
    ax.set_xlabel("Train ID")
    ax.set_ylabel("P(Recall)")
    ax.set_title("Recall by Train")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_mean_drift_by_item_type(subject_df: pd.DataFrame) -> plt.Figure:
    """Bar plot: mean encoding drift by item type and hypothesis."""
    _style()
    melted = subject_df.melt(
        id_vars=["hypothesis", "subject_id"],
        value_vars=["mean_boundary_drift", "mean_nonboundary_drift"],
        var_name="item_type", value_name="drift",
    )
    melted["item_type"] = melted["item_type"].map(
        {"mean_boundary_drift": "Boundary", "mean_nonboundary_drift": "Non-boundary"}
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=melted, x="hypothesis", y="drift", hue="item_type",
        order=_HYP_ORDER, errorbar="se", capsize=0.1, ax=ax,
    )
    ax.set_xticklabels([_hyp_label(h) for h in _HYP_ORDER])
    ax.set_ylabel("Mean Encoding Drift")
    ax.set_xlabel("")
    ax.set_title("Drift by Item Type & Hypothesis")
    ax.legend(title="")
    fig.tight_layout()
    return fig


# ===================================================================
# Compact 2×3 comparison panel
# ===================================================================

def make_compact_comparison_panel(
    subject_df: pd.DataFrame,
    spc_df: pd.DataFrame,
    train_df: pd.DataFrame,
    trial_df: pd.DataFrame,
) -> plt.Figure:
    """2×3 grid combining the six key diagnostic views."""
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # --- Panel A: overall recall ---
    ax = axes[0, 0]
    sns.barplot(
        data=subject_df, x="hypothesis", y="overall_recall",
        order=_HYP_ORDER, palette=_PALETTE, errorbar="se",
        capsize=0.12, ax=ax,
    )
    ax.set_xticklabels([_hyp_label(h) for h in _HYP_ORDER], fontsize=8)
    ax.set_ylabel("Overall Recall")
    ax.set_xlabel("")
    ax.set_title("A. Overall Recall", fontweight="bold")
    ax.set_ylim(0, 1)

    # --- Panel B: boundary vs non‑boundary recall ---
    ax = axes[0, 1]
    melted = subject_df.melt(
        id_vars=["hypothesis", "subject_id"],
        value_vars=["boundary_recall", "nonboundary_recall"],
        var_name="item_type", value_name="recall",
    )
    melted["item_type"] = melted["item_type"].map(
        {"boundary_recall": "Boundary", "nonboundary_recall": "Non-boundary"}
    )
    sns.barplot(
        data=melted, x="hypothesis", y="recall", hue="item_type",
        order=_HYP_ORDER, errorbar="se", capsize=0.08, ax=ax,
    )
    ax.set_xticklabels([_hyp_label(h) for h in _HYP_ORDER], fontsize=8)
    ax.set_ylabel("Recall")
    ax.set_xlabel("")
    ax.set_title("B. Boundary vs Non-Boundary", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(title="", fontsize=7)

    # --- Panel C: boundary advantage ---
    ax = axes[0, 2]
    sns.barplot(
        data=subject_df, x="hypothesis", y="boundary_advantage",
        order=_HYP_ORDER, palette=_PALETTE, errorbar="se",
        capsize=0.12, ax=ax,
    )
    ax.set_xticklabels([_hyp_label(h) for h in _HYP_ORDER], fontsize=8)
    ax.set_ylabel("Δ Recall")
    ax.set_xlabel("")
    ax.set_title("C. Boundary Advantage", fontweight="bold")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")

    # --- Panel D: mean drift by item type ---
    ax = axes[1, 0]
    drift_m = subject_df.melt(
        id_vars=["hypothesis", "subject_id"],
        value_vars=["mean_boundary_drift", "mean_nonboundary_drift"],
        var_name="item_type", value_name="drift",
    )
    drift_m["item_type"] = drift_m["item_type"].map(
        {"mean_boundary_drift": "Boundary", "mean_nonboundary_drift": "Non-boundary"}
    )
    sns.barplot(
        data=drift_m, x="hypothesis", y="drift", hue="item_type",
        order=_HYP_ORDER, errorbar="se", capsize=0.08, ax=ax,
    )
    ax.set_xticklabels([_hyp_label(h) for h in _HYP_ORDER], fontsize=8)
    ax.set_ylabel("Mean Drift")
    ax.set_xlabel("")
    ax.set_title("D. Drift by Item Type", fontweight="bold")
    ax.legend(title="", fontsize=7)

    # --- Panel E: serial position ---
    ax = axes[1, 1]
    for hyp in _HYP_ORDER:
        sub = spc_df[spc_df["hypothesis"] == hyp]
        ax.plot(sub["within_list_pos"], sub["recall"],
                label=_hyp_label(hyp), color=_PALETTE[hyp], linewidth=1.3)
    ax.set_xlabel("Within‑List Position")
    ax.set_ylabel("P(Recall)")
    ax.set_title("E. Serial Position Curve", fontweight="bold")
    ax.legend(fontsize=7)

    # --- Panel F: recall by train ---
    ax = axes[1, 2]
    train_agg = (
        train_df.groupby(["hypothesis", "train_id"])["train_recall"]
        .mean()
        .reset_index()
    )
    for hyp in _HYP_ORDER:
        sub = train_agg[train_agg["hypothesis"] == hyp]
        ax.plot(sub["train_id"], sub["train_recall"],
                label=_hyp_label(hyp), color=_PALETTE[hyp], linewidth=1.3)
    ax.set_xlabel("Train ID")
    ax.set_ylabel("P(Recall)")
    ax.set_title("F. Recall by Train", fontweight="bold")
    ax.legend(fontsize=7)

    fig.suptitle("Hypothesis Comparison Panel", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ===================================================================
# Save helper
# ===================================================================

def save_all_figures(
    figures: dict[str, plt.Figure],
    out_dir: str | Path = "results/figures",
) -> None:
    """Save a ``{name: fig}`` dict to *out_dir* as PNGs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        fig.savefig(out_dir / f"{name}.png")
    print(f"  Saved {len(figures)} figures to {out_dir}")

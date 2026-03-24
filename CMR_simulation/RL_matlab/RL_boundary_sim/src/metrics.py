"""
Compute all summary metrics for comparing hypotheses.

Three granularities:
  1. Trial‑level (already in the raw simulation DataFrame).
  2. Subject‑level summaries.
  3. Hypothesis‑level aggregates (means, SEs, CIs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ===================================================================
# Trial → Subject aggregation
# ===================================================================

def compute_recall_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Overall recall proportion per subject × hypothesis."""
    return (
        df.groupby(["hypothesis", "subject_id"])["recalled"]
        .mean()
        .reset_index()
        .rename(columns={"recalled": "overall_recall"})
    )


def compute_boundary_vs_nonboundary_recall(df: pd.DataFrame) -> pd.DataFrame:
    """Recall split by boundary flag, per subject × hypothesis."""
    out = (
        df.groupby(["hypothesis", "subject_id", "is_boundary"])["recalled"]
        .mean()
        .reset_index()
        .rename(columns={"recalled": "recall"})
    )
    # Pivot to wide form
    wide = out.pivot_table(
        index=["hypothesis", "subject_id"],
        columns="is_boundary",
        values="recall",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={0: "nonboundary_recall", 1: "boundary_recall"})
    wide["boundary_advantage"] = wide["boundary_recall"] - wide["nonboundary_recall"]
    return wide


def compute_serial_position_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Mean recall by within‑list position, per hypothesis."""
    return (
        df.groupby(["hypothesis", "within_list_pos"])["recalled"]
        .mean()
        .reset_index()
        .rename(columns={"recalled": "recall"})
    )


def compute_recall_by_train(df: pd.DataFrame) -> pd.DataFrame:
    """Mean recall per train, per subject × hypothesis."""
    return (
        df.groupby(["hypothesis", "subject_id", "train_id"])["recalled"]
        .mean()
        .reset_index()
        .rename(columns={"recalled": "train_recall"})
    )


def compute_drift_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean encoding drift by boundary flag, per subject × hypothesis."""
    out = (
        df.groupby(["hypothesis", "subject_id", "is_boundary"])["encoding_drift"]
        .mean()
        .reset_index()
        .rename(columns={"encoding_drift": "mean_drift"})
    )
    wide = out.pivot_table(
        index=["hypothesis", "subject_id"],
        columns="is_boundary",
        values="mean_drift",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={0: "mean_nonboundary_drift", 1: "mean_boundary_drift"}
    )
    wide["drift_boundary_contrast"] = (
        wide["mean_boundary_drift"] - wide["mean_nonboundary_drift"]
    )
    return wide


def compute_rpe_recall_coupling(df: pd.DataFrame) -> pd.DataFrame:
    """Per‑subject correlation between |RPE| and recall."""
    def _corr(g):
        if g["abs_outcome_rpe"].std() == 0:
            return np.nan
        return g["abs_outcome_rpe"].corr(g["recalled"])

    return (
        df.groupby(["hypothesis", "subject_id"])
        .apply(_corr, include_groups=False)
        .reset_index()
        .rename(columns={0: "rpe_recall_r"})
    )


# ===================================================================
# Subject‑level summary table
# ===================================================================

def compute_subject_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per subject × hypothesis with all key metrics."""
    acc = compute_recall_accuracy(df)
    bnd = compute_boundary_vs_nonboundary_recall(df)
    dft = compute_drift_summary(df)
    rrc = compute_rpe_recall_coupling(df)

    # Mean |RPE| per subject
    mean_rpe = (
        df.groupby(["hypothesis", "subject_id"])["abs_outcome_rpe"]
        .mean()
        .reset_index()
        .rename(columns={"abs_outcome_rpe": "mean_abs_rpe"})
    )

    # Mean drift overall
    mean_drift_all = (
        df.groupby(["hypothesis", "subject_id"])["encoding_drift"]
        .mean()
        .reset_index()
        .rename(columns={"encoding_drift": "mean_drift"})
    )

    out = acc
    for right in [bnd, dft, mean_rpe, mean_drift_all, rrc]:
        out = out.merge(right, on=["hypothesis", "subject_id"], how="left")
    return out


# ===================================================================
# Hypothesis‑level aggregate
# ===================================================================

def compute_hypothesis_summary(subject_df: pd.DataFrame) -> pd.DataFrame:
    """Means and SEs across subjects, per hypothesis."""
    numeric_cols = subject_df.select_dtypes(include="number").columns.tolist()
    # Remove subject_id from aggregation
    numeric_cols = [c for c in numeric_cols if c != "subject_id"]

    agg = subject_df.groupby("hypothesis")[numeric_cols].agg(["mean", "sem"])
    # Flatten multi‑level columns
    agg.columns = ["_".join(col) for col in agg.columns]
    return agg.reset_index()


# ===================================================================
# All‑in‑one
# ===================================================================

def summarize_all_metrics(trial_df: pd.DataFrame):
    """Return (subject_summary, hypothesis_summary, serial_position, train_recall)."""
    subj = compute_subject_summary(trial_df)
    hyp = compute_hypothesis_summary(subj)
    spc = compute_serial_position_curve(trial_df)
    trn = compute_recall_by_train(trial_df)
    return subj, hyp, spc, trn

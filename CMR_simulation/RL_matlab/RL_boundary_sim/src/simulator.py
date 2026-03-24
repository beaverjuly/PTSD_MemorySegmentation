"""
Main simulation engine.

Combines the task schedule, RPE computation, hypothesis‑specific
drift rules, and a recall‑generation mechanism.  This is the Python
analogue of ``exp1_models.m`` plus the looping logic from
``exp1_driver.m`` and ``sims_driver.m``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TaskConfig, ModelConfig, HypothesisParams, SimConfig
from .task_design import build_trial_dataframe
from .hypotheses import compute_encoding_drift, list_hypotheses
from .utils import sigmoid


# ---------------------------------------------------------------------------
# RL helpers
# ---------------------------------------------------------------------------

def compute_outcome_rpe(reward: float, expected_value: float) -> float:
    """Signed prediction error."""
    return reward - expected_value


def update_expected_value(
    prev_value: float, reward: float, alpha: float
) -> float:
    """Rescorla–Wagner value update."""
    return prev_value + alpha * (reward - prev_value)


# ---------------------------------------------------------------------------
# Single‑subject simulation
# ---------------------------------------------------------------------------

def simulate_subject(
    trial_df: pd.DataFrame,
    hypothesis_name: str,
    model_cfg: ModelConfig,
    hyp: HypothesisParams,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run one simulated subject.

    Parameters
    ----------
    trial_df : DataFrame
        Trial schedule (shared across hypotheses / subjects).
        Must already contain ``reward``, ``is_boundary``,
        ``abs_outcome_rpe``.  A *copy* is used internally.
    hypothesis_name : str
    model_cfg : ModelConfig
    hyp : HypothesisParams
    rng : numpy Generator

    Returns
    -------
    DataFrame with added columns:
        encoding_drift, recall_prob, recalled
    """
    df = trial_df.copy()

    # --- encoding drift (hypothesis‑specific) ---
    drift = compute_encoding_drift(df, hypothesis_name, model_cfg, hyp)
    df["encoding_drift"] = drift

    # --- recall probability via logistic transform ---
    logit = model_cfg.recall_intercept + model_cfg.recall_drift_weight * drift
    noise = rng.normal(0, model_cfg.recall_noise_std, size=len(df))
    recall_prob = sigmoid(logit + noise)
    df["recall_prob"] = np.round(recall_prob, 6)

    # --- binary recall ---
    df["recalled"] = (rng.random(len(df)) < recall_prob).astype(int)

    df["hypothesis"] = hypothesis_name
    return df


# ---------------------------------------------------------------------------
# Group simulation
# ---------------------------------------------------------------------------

def simulate_group(
    task_cfg: TaskConfig,
    hypothesis_name: str,
    model_cfg: ModelConfig,
    hyp: HypothesisParams,
    n_subjects: int,
    rng_seed: int,
) -> pd.DataFrame:
    """Simulate *n_subjects* under one hypothesis.

    Each subject receives the same trial schedule geometry but an
    independently‑sampled reward sequence.

    Returns
    -------
    DataFrame with a ``subject_id`` column added.
    """
    master_rng = np.random.default_rng(rng_seed)
    frames = []

    for s in range(n_subjects):
        # Independent reward draws per subject
        subj_rng = np.random.default_rng(master_rng.integers(0, 2**31))
        trial_df = build_trial_dataframe(task_cfg, rng=subj_rng, alpha=model_cfg.alpha)

        # Recall simulation uses its own stream
        recall_rng = np.random.default_rng(master_rng.integers(0, 2**31))
        subj_df = simulate_subject(
            trial_df, hypothesis_name, model_cfg, hyp, recall_rng
        )
        subj_df["subject_id"] = s
        frames.append(subj_df)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Run all hypotheses
# ---------------------------------------------------------------------------

def run_all_hypotheses(
    task_cfg: TaskConfig | None = None,
    model_cfg: ModelConfig | None = None,
    hyp: HypothesisParams | None = None,
    sim_cfg: SimConfig | None = None,
) -> pd.DataFrame:
    """Convenience: simulate every hypothesis and concatenate.

    Returns a single long‑form DataFrame.
    """
    from .config import (
        get_default_task_config,
        get_default_model_config,
        get_hypothesis_params,
        get_default_sim_config,
    )

    task_cfg = task_cfg or get_default_task_config()
    model_cfg = model_cfg or get_default_model_config()
    hyp = hyp or get_hypothesis_params()
    sim_cfg = sim_cfg or get_default_sim_config()

    frames = []
    for h_name in sim_cfg.hypothesis_names:
        print(f"  Simulating {h_name} ({sim_cfg.n_subjects} subjects) …")
        df = simulate_group(
            task_cfg, h_name, model_cfg, hyp,
            n_subjects=sim_cfg.n_subjects,
            rng_seed=sim_cfg.seed,
        )
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    print("  Done.")
    return all_df

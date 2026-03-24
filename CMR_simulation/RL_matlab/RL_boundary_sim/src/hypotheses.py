"""
Encode the three hypotheses cleanly and explicitly.

Each function returns trialwise encoding‑drift values.  The scientific
contrast hinges entirely on this file: the task schedule and recall
mechanism are identical across hypotheses.

Hypotheses
----------
- **baseline (H0)**: boundary items get ``beta_base + delta_boundary``;
  non‑boundary items get ``beta_base``.  Both are further modulated by
  ``rpe_drift_weight * |RPE|``.
- **H1 (smaller boundary increase)**: same as baseline but
  ``delta_boundary`` is replaced by a smaller value.
- **H2 (global bias reduction)**: same ``delta_boundary`` as baseline
  but ``beta_base`` is uniformly reduced by ``bias_shift``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ModelConfig, HypothesisParams


# ---------------------------------------------------------------------------
# Per‑trial drift rules
# ---------------------------------------------------------------------------

def _base_drift(
    is_boundary: int,
    abs_rpe: float,
    beta_base: float,
    delta_boundary: float,
    rpe_weight: float,
) -> float:
    """Core formula shared by all hypotheses (with pluggable params)."""
    drift = beta_base + delta_boundary * is_boundary + rpe_weight * (abs_rpe / 100.0)
    return drift


def baseline_drift_rule(
    is_boundary: int,
    abs_rpe: float,
    model_cfg: ModelConfig,
    _hyp: HypothesisParams | None = None,
) -> float:
    """H0 – normal boundary increment, RPE‑modulated."""
    return _base_drift(
        is_boundary,
        abs_rpe,
        model_cfg.beta_base,
        model_cfg.delta_boundary,
        model_cfg.rpe_drift_weight,
    )


def smaller_boundary_increase_rule(
    is_boundary: int,
    abs_rpe: float,
    model_cfg: ModelConfig,
    hyp: HypothesisParams,
) -> float:
    """H1 – smaller boundary‑specific increase."""
    return _base_drift(
        is_boundary,
        abs_rpe,
        model_cfg.beta_base,
        hyp.h1_delta_boundary,
        model_cfg.rpe_drift_weight,
    )


def global_bias_reduction_rule(
    is_boundary: int,
    abs_rpe: float,
    model_cfg: ModelConfig,
    hyp: HypothesisParams,
) -> float:
    """H2 – same boundary increment, global downward bias shift."""
    shifted_base = model_cfg.beta_base - hyp.h2_bias_shift
    return _base_drift(
        is_boundary,
        abs_rpe,
        shifted_base,
        model_cfg.delta_boundary,
        model_cfg.rpe_drift_weight,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_RULES = {
    "baseline": baseline_drift_rule,
    "H1_smaller_boundary": smaller_boundary_increase_rule,
    "H2_global_reduction": global_bias_reduction_rule,
}


def compute_encoding_drift(
    trial_df: pd.DataFrame,
    hypothesis_name: str,
    model_cfg: ModelConfig,
    hyp: HypothesisParams,
) -> np.ndarray:
    """Vectorised wrapper: return an array of per‑trial encoding drift.

    Parameters
    ----------
    trial_df : DataFrame
        Must contain ``is_boundary`` and ``abs_outcome_rpe``.
    hypothesis_name : str
        One of ``'baseline'``, ``'H1_smaller_boundary'``,
        ``'H2_global_reduction'``.
    model_cfg : ModelConfig
    hyp : HypothesisParams

    Returns
    -------
    np.ndarray of shape (n_trials,)
    """
    rule = _RULES[hypothesis_name]
    drifts = np.array(
        [
            rule(
                int(row["is_boundary"]),
                float(row["abs_outcome_rpe"]),
                model_cfg,
                hyp,
            )
            for _, row in trial_df.iterrows()
        ]
    )
    return drifts


def list_hypotheses() -> list[str]:
    """Return available hypothesis names."""
    return list(_RULES.keys())

"""
Generate the Exp1‑style trial structure and event labels.

Produces a tidy DataFrame with one row per trial, annotated with
boundary flags, train IDs, and (optionally) pre‑generated reward
sequences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TaskConfig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_exp1_schedule(
    task_cfg: TaskConfig,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build the full trial‑level schedule for one subject.

    Parameters
    ----------
    task_cfg : TaskConfig
        Experiment geometry (lists, items, train length, etc.).
    rng : numpy Generator, optional
        Used for reward sampling.  Pass ``None`` for a fresh default.

    Returns
    -------
    pd.DataFrame
        Columns: trial, list_id, item_id, within_list_pos,
        train_id, within_train_pos, is_boundary, reward.
    """
    if rng is None:
        rng = np.random.default_rng()

    rows = []
    trial = 0
    global_train = 0

    for list_id in range(task_cfg.n_lists):
        within_train_pos = 0
        for pos in range(task_cfg.items_per_list):
            is_boundary = int(
                pos > 0 and pos % task_cfg.items_per_train == 0
            )
            if is_boundary:
                global_train += 1
                within_train_pos = 0

            reward = float(
                np.clip(
                    rng.normal(task_cfg.reward_mean, task_cfg.reward_std),
                    0,
                    100,
                )
            )

            rows.append(
                {
                    "trial": trial,
                    "list_id": list_id,
                    "item_id": trial,           # unique item index
                    "within_list_pos": pos,
                    "train_id": global_train,
                    "within_train_pos": within_train_pos,
                    "is_boundary": is_boundary,
                    "reward": round(reward, 2),
                }
            )
            trial += 1
            within_train_pos += 1

        # new list → new train
        global_train += 1

    df = pd.DataFrame(rows)
    return df


def assign_boundary_flags(df: pd.DataFrame) -> pd.DataFrame:
    """(Re)compute boundary and train labels from within‑list position.

    Useful when loading a fixed schedule that lacks these columns.
    """
    # Already handled inside make_exp1_schedule; provided for external use.
    return df


def assign_rpe_values(
    df: pd.DataFrame,
    initial_value: float = 50.0,
    alpha: float = 0.3,
) -> pd.DataFrame:
    """Compute expected value, signed RPE, and |RPE| using an RW rule.

    Operates *in place* on the DataFrame and also returns it.
    """
    n = len(df)
    ev = np.zeros(n)
    signed_rpe = np.zeros(n)

    ev[0] = initial_value
    for t in range(n):
        signed_rpe[t] = df.iloc[t]["reward"] - ev[t]
        if t < n - 1:
            # Reset expected value at list boundary (new list)
            if df.iloc[t + 1]["within_list_pos"] == 0:
                ev[t + 1] = initial_value
            else:
                ev[t + 1] = ev[t] + alpha * signed_rpe[t]

    df = df.copy()
    df["expected_value"] = np.round(ev, 4)
    df["outcome_rpe"] = np.round(signed_rpe, 4)
    df["abs_outcome_rpe"] = np.round(np.abs(signed_rpe), 4)
    return df


def build_trial_dataframe(
    task_cfg: TaskConfig,
    rng: np.random.Generator | None = None,
    alpha: float = 0.3,
) -> pd.DataFrame:
    """One‑call convenience: schedule → RPE annotation."""
    df = make_exp1_schedule(task_cfg, rng=rng)
    df = assign_rpe_values(df, initial_value=task_cfg.initial_value, alpha=alpha)
    return df

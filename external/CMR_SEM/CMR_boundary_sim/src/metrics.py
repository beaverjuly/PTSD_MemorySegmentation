"""
CMR Boundary-Signal Simulation — Behavioral Metrics

Whole-list metrics
------------------
compute_spc, compute_pfr, compute_lag_crp, recall_accuracy

Boundary-local metrics
----------------------
boundary_transition   — the single workhorse for any transition
                        defined relative to a boundary position.
                        Pools across all boundaries in the list.

boundary_local_spc    — SPC in a window around each boundary,
                        averaged across boundaries.

How boundary_transition works
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Every transition is specified by two offsets relative to boundary
position j:

    boundary_transition(rs, N, start_offset, target_offset)

The function loops over every boundary j, computes the
opportunity-corrected CRP for (j + start_offset) → (j + target_offset),
and pools the numerators and denominators across boundaries.

Examples (lag-1 checks):
    boundary_transition(rs, N, -1,  0)   # (j-1) → j    pre→boundary
    boundary_transition(rs, N,  0, -1)   # j → (j-1)    boundary→pre
    boundary_transition(rs, N,  0,  1)   # j → (j+1)    boundary→post

To add lag-2 checks later, just change the offsets:
    boundary_transition(rs, N, -2,  0)   # (j-2) → j
    boundary_transition(rs, N,  0, -2)   # j → (j-2)
    boundary_transition(rs, N,  0,  2)   # j → (j+2)
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .config import BOUNDARY_POSITIONS


# =====================================================================
# Whole-list metrics
# =====================================================================

def compute_spc(recall_sims, N):
    """Serial Position Curve: P(item at position j is recalled)."""
    spc = np.zeros(N)
    for j in range(1, N + 1):
        spc[j - 1] = np.mean(np.any(recall_sims == j, axis=0))
    return spc


def compute_pfr(recall_sims, N):
    """Probability of First Recall for each serial position."""
    first = recall_sims[0, :]
    first = first[first > 0]
    pfr = np.zeros(N)
    if len(first) > 0:
        for j in range(1, N + 1):
            pfr[j - 1] = np.mean(first == j)
    return pfr


def compute_lag_crp(recall_sims, N):
    """Lag-CRP with opportunity correction. Returns (lag_vals, crp)."""
    max_lag = N - 1
    lag_vals = np.arange(-max_lag, max_lag + 1)
    numer = np.zeros(len(lag_vals), dtype=float)
    denom = np.zeros(len(lag_vals), dtype=float)
    lag_to_idx = {L: i for i, L in enumerate(lag_vals)}

    for s in range(recall_sims.shape[1]):
        seq = recall_sims[:, s]
        seq = seq[seq > 0].astype(int)
        if len(seq) < 2:
            continue
        recalled = set()
        for t in range(len(seq) - 1):
            cur, nxt = seq[t], seq[t + 1]
            recalled.add(cur)
            remaining = [j for j in range(1, N + 1) if j not in recalled]
            for j in remaining:
                denom[lag_to_idx[j - cur]] += 1
            numer[lag_to_idx[nxt - cur]] += 1

    crp = np.zeros_like(numer)
    valid = denom > 0
    crp[valid] = numer[valid] / denom[valid]
    return lag_vals, crp


def recall_accuracy(recall_sims, N, unique=True):
    """E[#unique_recalled / N] across simulations."""
    if recall_sims is None or recall_sims.size == 0:
        return np.nan
    acc = []
    for s in range(recall_sims.shape[1]):
        seq = recall_sims[:, s]
        seq = seq[seq > 0].astype(int)
        if seq.size == 0:
            acc.append(0.0)
            continue
        if unique:
            seq = np.unique(seq)
        acc.append(len(seq) / float(N))
    return float(np.mean(acc)) if acc else np.nan


# =====================================================================
# Position-conditional transition probability (low-level engine)
# =====================================================================

class TransitionResult(NamedTuple):
    """Holds a transition probability together with its raw counts."""
    prob: float
    num: int
    den: int


def position_conditional_crp(
    recall_sims, N, start_pos, target_lag, *, return_counts=False
):
    """
    Opportunity-corrected probability that, given the current recall
    is at serial position ``start_pos``, the next recall is at
    ``start_pos + target_lag``.

    Parameters
    ----------
    recall_sims : (N, n_sims) int array
        Matrix of recalled serial positions (1-based, 0 = not recalled).
    N : int
        List length.
    start_pos : int  (1-based)
        Serial position of the "current" recall.
    target_lag : int
        Signed lag to the "next" recall.
    return_counts : bool
        If True, return a TransitionResult(prob, num, den).

    Returns
    -------
    float or TransitionResult
    """
    target_pos = start_pos + target_lag
    if target_pos < 1 or target_pos > N:
        if return_counts:
            return TransitionResult(np.nan, 0, 0)
        return np.nan

    num = 0
    den = 0

    for s in range(recall_sims.shape[1]):
        seq = recall_sims[:, s]
        seq = seq[seq > 0].astype(int)
        if len(seq) < 2:
            continue

        recalled = set()
        for t in range(len(seq) - 1):
            cur = seq[t]
            nxt = seq[t + 1]
            recalled.add(cur)

            if cur == start_pos:
                if target_pos not in recalled:
                    den += 1
                    if nxt == target_pos:
                        num += 1

    prob = (num / den) if den > 0 else np.nan
    if return_counts:
        return TransitionResult(prob, num, den)
    return prob


# =====================================================================
# Boundary transition — the single workhorse
# =====================================================================

def boundary_transition(
    recall_sims,
    N: int,
    start_offset: int,
    target_offset: int,
    boundary_positions: list[int] = BOUNDARY_POSITIONS,
    *,
    return_counts: bool = False,
):
    """
    Transition probability pooled across all boundary positions.

    For every boundary position j in ``boundary_positions``, this
    computes the opportunity-corrected CRP for the transition

        (j + start_offset)  →  (j + target_offset)

    and pools the raw numerators and denominators across boundaries.

    Parameters
    ----------
    recall_sims : (N, n_sims) int array
        Matrix of recalled serial positions (1-based).
    N : int
        List length.
    start_offset : int
        Offset of the start position relative to boundary j.
        Example: -1 means one position before the boundary.
    target_offset : int
        Offset of the target position relative to boundary j.
        Example: 0 means the boundary item itself.
    boundary_positions : list[int]
        1-based serial positions of boundary items.
    return_counts : bool
        If True, return a TransitionResult(prob, num, den).

    Returns
    -------
    float or TransitionResult

    Examples
    --------
    # Lag-1 checks
    boundary_transition(rs, N, -1,  0)   # (j-1) → j   pre→boundary
    boundary_transition(rs, N,  0, -1)   # j → (j-1)   boundary→pre
    boundary_transition(rs, N,  0,  1)   # j → (j+1)   boundary→post

    # Lag-2 checks (easy to add later)
    boundary_transition(rs, N, -2,  0)   # (j-2) → j
    boundary_transition(rs, N,  0, -2)   # j → (j-2)
    boundary_transition(rs, N,  0,  2)   # j → (j+2)
    """
    total_num = 0
    total_den = 0

    for j in boundary_positions:
        start = j + start_offset    # absolute serial position
        target = j + target_offset  # absolute serial position

        # skip if either position falls outside the list
        if start < 1 or start > N:
            continue
        if target < 1 or target > N:
            continue

        lag = target - start
        tr = position_conditional_crp(
            recall_sims, N, start, lag, return_counts=True
        )
        total_num += tr.num
        total_den += tr.den

    prob = (total_num / total_den) if total_den > 0 else np.nan
    if return_counts:
        return TransitionResult(prob, total_num, total_den)
    return prob


# =====================================================================
# Local SPC helpers
# =====================================================================

def local_spc(recall_sims, N, center, half_window=4):
    """
    SPC restricted to a window around a given position.

    Parameters
    ----------
    center : int  (1-based)
    half_window : int

    Returns
    -------
    positions : (window_size,) int array (1-based)
    spc : (window_size,) float array
    """
    lo = max(1, center - half_window)
    hi = min(N, center + half_window)
    pos_range = np.arange(lo, hi + 1)
    spc_full = compute_spc(recall_sims, N)
    return pos_range, spc_full[pos_range - 1]


def boundary_local_spc(
    recall_sims,
    N: int,
    boundary_positions: list[int] = BOUNDARY_POSITIONS,
    half_window: int = 4,
):
    """
    SPC in a local window around each boundary, averaged across
    boundaries.  Positions are returned as offsets relative to the
    boundary (0 = boundary item).

    Returns
    -------
    rel_positions : (window_size,) int array
    mean_spc : (window_size,) float array
    se_spc : (window_size,) float array  (SE across boundaries)
    """
    spc_full = compute_spc(recall_sims, N)
    rel = np.arange(-half_window, half_window + 1)

    curves = []
    for j in boundary_positions:
        abs_pos = j + rel  # 1-based
        valid = (abs_pos >= 1) & (abs_pos <= N)
        curve = np.full(len(rel), np.nan)
        curve[valid] = spc_full[abs_pos[valid] - 1]
        curves.append(curve)

    curves = np.array(curves)
    mean_spc = np.nanmean(curves, axis=0)
    se_spc = np.nanstd(curves, axis=0) / np.sqrt(
        np.sum(~np.isnan(curves), axis=0)
    )
    return rel, mean_spc, se_spc


# =====================================================================
# Compact summary
# =====================================================================

def summarize_condition_metrics(
    recall_sims,
    N: int,
    boundary_positions: list[int] = BOUNDARY_POSITIONS,
    half_window: int = 4,
) -> dict:
    """
    Compute all primary and secondary metrics for one condition.

    Returns
    -------
    dict with keys:
        recall_accuracy, whole_spc, whole_pfr, whole_lag_crp,
        pre_to_boundary, boundary_backward, boundary_forward,
        boundary_local_spc
    """
    lag_vals, crp = compute_lag_crp(recall_sims, N)
    rel, mean_spc_local, se_spc_local = boundary_local_spc(
        recall_sims, N, boundary_positions, half_window
    )

    return {
        "recall_accuracy": recall_accuracy(recall_sims, N),
        "whole_spc": compute_spc(recall_sims, N),
        "whole_pfr": compute_pfr(recall_sims, N),
        "whole_lag_crp": (lag_vals, crp),
        # lag-1 boundary transitions
        "pre_to_boundary":    boundary_transition(
            recall_sims, N, -1, 0, boundary_positions),
        "boundary_backward":  boundary_transition(
            recall_sims, N, 0, -1, boundary_positions),
        "boundary_forward":   boundary_transition(
            recall_sims, N, 0, 1, boundary_positions),
        "boundary_local_spc": (rel, mean_spc_local, se_spc_local),
    }

"""
CMR Boundary-Signal Simulation — Behavioral Metrics

Whole-list metrics
------------------
compute_spc, compute_pfr, compute_lag_crp, recall_accuracy

Train-level metrics (Polyn 2009-style)
---------------------------------------
compute_train_recall  — proportion of items recalled from each train
mean_train_recall     — unweighted mean across trains (convenience scalar)

    Trains are segments of the list defined by boundary positions.
    For the default 32-item design with boundaries at [9, 17, 25]:
        Train 1 = positions 1–8
        Train 2 = positions 9–16
        Train 3 = positions 17–24
        Train 4 = positions 25–32

    Interpretive caveats
    ~~~~~~~~~~~~~~~~~~~~
    - Train 1 may partly reflect primacy-related dynamics rather than
      pure boundary structure.  Differences among Trains 2–4 are more
      directly informative about the manipulated boundary/baseline
      drift schedule.
    - mean_train_recall equals whole-list recall_accuracy only when
      all trains have the same length.  This holds for the default
      32-item design but is NOT guaranteed in general.
    - Zero-padding in recall_sims is explicitly excluded so that
      unused recall slots are never counted as recalled items.

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
# Train-level recall (Polyn 2009-style)
# =====================================================================

def _train_bounds_from_boundaries(N, boundary_positions):
    """
    Derive 1-based inclusive (start, end) tuples for each train.

    Trains are the segments between successive boundary positions.
    For N=32 and boundary_positions=[9, 17, 25] the result is:
        [(1, 8), (9, 16), (17, 24), (25, 32)]

    Parameters
    ----------
    N : int
        List length.
    boundary_positions : list[int]
        1-based serial positions where boundaries occur.

    Returns
    -------
    bounds : list of (int, int)
        Each tuple is (start, end) inclusive, in 1-based indexing.
    """
    bounds = []
    starts = [1] + sorted(boundary_positions)

    for i in range(len(starts)):
        start = starts[i]
        end = starts[i + 1] - 1 if i + 1 < len(starts) else N
        if start <= end:
            bounds.append((start, end))

    return bounds


def compute_train_recall(recall_sims, N, boundary_positions, unique=True):
    """
    Proportion of items recalled from each train, per simulation.

    For each simulation s and train t:

        TrainRecall[s, t] = #(unique recalled items in train t)
                            / #(items in train t)

    Parameters
    ----------
    recall_sims : (N, n_sims) int array
        Recalled serial positions (1-based, 0-padded).
    N : int
        List length.
    boundary_positions : list[int]
        1-based serial positions of boundary items.
    unique : bool
        If True (default), count each recalled item at most once.

    Returns
    -------
    train_ids : (n_trains,) int array
        Train numbers (1-based).
    bounds : list of (int, int)
        Inclusive (start, end) for each train.
    mean_tr : (n_trains,) float array
        Mean train recall across simulations.
    se_tr : (n_trains,) float array
        Standard error of train recall across simulations.
    per_sim_train_recall : (n_sims, n_trains) float array
        Full per-simulation × per-train recall matrix.
    """
    bounds = _train_bounds_from_boundaries(N, boundary_positions)
    n_sims = recall_sims.shape[1]
    n_trains = len(bounds)

    per_sim_train_recall = np.zeros((n_sims, n_trains))

    for s in range(n_sims):
        recalls = recall_sims[:, s]
        # Remove zero-padding, then optionally unique
        recalls = recalls[recalls > 0]
        if unique:
            recalls = np.unique(recalls)

        for t, (start, end) in enumerate(bounds):
            train_length = end - start + 1
            hits = np.sum((recalls >= start) & (recalls <= end))
            per_sim_train_recall[s, t] = hits / train_length

    train_ids = np.arange(1, n_trains + 1)
    mean_tr = np.mean(per_sim_train_recall, axis=0)
    se_tr = np.std(per_sim_train_recall, axis=0, ddof=1) / np.sqrt(n_sims)

    return train_ids, bounds, mean_tr, se_tr, per_sim_train_recall


def mean_train_recall(recall_sims, N, boundary_positions, unique=True):
    """
    Unweighted mean of train-level recall proportions.

    This is a convenience scalar: the average of per-train recall
    proportions, first averaged across simulations, then across trains.

    Note: this equals whole-list recall_accuracy ONLY when all trains
    have the same length.  For the default 32-item design that is true,
    but it is not guaranteed in general.

    Parameters
    ----------
    recall_sims : (N, n_sims) int array
    N : int
    boundary_positions : list[int]
    unique : bool

    Returns
    -------
    float
        Scalar mean across trains.
    """
    _, _, mean_tr, _, _ = compute_train_recall(
        recall_sims, N, boundary_positions, unique=unique
    )
    return float(np.mean(mean_tr))


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
# Boundary transition (wrapper)
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
    Compute all primary, secondary, and train-level metrics for one
    condition.

    Returns
    -------
    dict with keys:
        recall_accuracy, whole_spc, whole_pfr, whole_lag_crp,
        pre_to_boundary, boundary_backward, boundary_forward,
        boundary_local_spc,
        train_recall, train_bounds, mean_train_recall
    """
    lag_vals, crp = compute_lag_crp(recall_sims, N)
    rel, mean_spc_local, se_spc_local = boundary_local_spc(
        recall_sims, N, boundary_positions, half_window
    )

    # Train-level recall
    train_ids, bounds, mean_tr, se_tr, _ = compute_train_recall(
        recall_sims, N, boundary_positions
    )
    mtr = float(np.mean(mean_tr))

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
        # train-level recall (Polyn 2009-style)
        "train_recall": (train_ids, mean_tr, se_tr),
        "train_bounds": bounds,
        "mean_train_recall": mtr,
    }

## ------------- Segment-specific metrics -------------

def extract_segment_recall_sims(recall_sims, start, end):
    """
    Restrict each recall sequence to items from [start, end] (inclusive),
    preserve recall order, and remap serial positions to 1..segment_length.

    Parameters
    ----------
    recall_sims : np.ndarray
        Array of shape (max_recalls, n_sims) containing recalled serial positions,
        with 0 used as padding for unrecalled slots.
    start : int
        Inclusive start serial position of the segment in original list coordinates.
    end : int
        Inclusive end serial position of the segment in original list coordinates.

    Returns
    -------
    np.ndarray
        Array of shape (segment_length, n_sims) containing only recalls from the
        requested segment, remapped to local positions 1..segment_length, with 0
        padding below the realized recalls for each simulation.
    """
    seg_len = end - start + 1
    n_sims = recall_sims.shape[1]
    out = np.zeros((seg_len, n_sims), dtype=int)

    for s in range(n_sims):
        seq = recall_sims[:, s]
        seq = seq[seq > 0].astype(int)          # drop padding
        seq = seq[(seq >= start) & (seq <= end)]  # keep only this segment
        seq = seq - start + 1                   # remap to local 1..seg_len
        out[:len(seq), s] = seq

    return out


def summarize_segment_metrics(results, start, end, N, boundary_pos=None, half_window=4):
    """
    Build segment-conditional summaries for each condition.

    Within-segment metrics:
        Restrict recall sequences to items from the chosen segment only, remap them
        to local positions 1..segment_length, and recompute SPC, PFR, lag-CRP,
        and proportion recalled.

    Boundary-local metrics:
        If boundary_pos is provided, compute boundary-local transition and local-SPC
        summaries on the ORIGINAL full recall sequences, but only for that one
        boundary position.

    Parameters
    ----------
    results : dict
        Mapping from condition label -> result dict, where each result dict must
        contain 'recall_sims'.
    start : int
        Inclusive start serial position of the focal segment in original coordinates.
    end : int
        Inclusive end serial position of the focal segment in original coordinates.
    N : int
        Full list length in original coordinates.
    boundary_pos : int or None, optional
        Original-coordinate boundary position to use for boundary-local summaries.
        If None, no boundary-local metrics are added.
    half_window : int, optional
        Window size for boundary_local_spc.

    Returns
    -------
    dict
        Nested mapping:
            condition_label -> summary dict
    """
    seg_summaries = {}
    seg_len = end - start + 1

    for label, res in results.items():
        recall_sims_full = res['recall_sims']
        recall_sims_seg = extract_segment_recall_sims(recall_sims_full, start, end)

        lag_vals, crp = compute_lag_crp(recall_sims_seg, seg_len)

        out = {
            'segment_recall_sims': recall_sims_seg,
            'segment_spc': compute_spc(recall_sims_seg, seg_len),
            'segment_pfr': compute_pfr(recall_sims_seg, seg_len),
            'segment_lag_crp': (lag_vals, crp),
            'segment_accuracy': recall_accuracy(recall_sims_seg, seg_len),
        }

        if boundary_pos is not None:
            rel, mspc, se = boundary_local_spc(
                recall_sims_full,
                N,
                boundary_positions=[boundary_pos],
                half_window=half_window,
            )

            out['boundary_pre_to_boundary'] = boundary_transition(
                recall_sims_full,
                N,
                -1,
                0,
                boundary_positions=[boundary_pos],
            )
            out['boundary_backward'] = boundary_transition(
                recall_sims_full,
                N,
                0,
                -1,
                boundary_positions=[boundary_pos],
            )
            out['boundary_forward'] = boundary_transition(
                recall_sims_full,
                N,
                0,
                1,
                boundary_positions=[boundary_pos],
            )
            out['boundary_local_spc'] = (rel, mspc, se)

        seg_summaries[label] = out

    return seg_summaries
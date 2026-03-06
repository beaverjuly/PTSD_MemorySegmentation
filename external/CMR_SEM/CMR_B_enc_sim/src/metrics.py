"""
CMR Behavioral Analysis Metrics
=================================
Pure behavioral readouts from recall sequences: SPC, PFR, lag-CRP,
conditional forward & backward lag rates, unconditional transition
summaries, and recall accuracy.

Everything here depends only on recall_sims — no model-internal quantities.
"""

import numpy as np


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

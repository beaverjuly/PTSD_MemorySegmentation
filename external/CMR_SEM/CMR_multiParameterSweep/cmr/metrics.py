"""
CMR Multi-Parameter Sweep — Behavioral Analysis Metrics
========================================================
Pure behavioral readouts from recall sequences: SPC, PFR, lag-CRP (with
counts), conditional forward & backward lag rates, unconditional transition
summaries, and recall accuracy.

Everything here depends only on ``recall_sims`` / ``times_sims`` 
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# SPC, PFR, lag-CRP (basic)
# ─────────────────────────────────────────────────────────────────────

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
    """Lag-CRP with opportunity correction.  Returns (lag_vals, crp)."""
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


# ─────────────────────────────────────────────────────────────────────
# lag-CRP with numerator / denominator counts (excluding lag 0)
# ─────────────────────────────────────────────────────────────────────

def lag_crp_with_counts(recall_sims, N):
    """
    Returns
    -------
    lags : array  — lags -(N-1)…(N-1) excluding 0
    crp  : array  — conditional transition probabilities
    num  : array  — numerator (observed transitions)
    den  : array  — denominator (available opportunities)
    """
    lag_list = [l for l in range(-(N - 1), N) if l != 0]
    idx = {l: i for i, l in enumerate(lag_list)}
    num = np.zeros(len(lag_list), dtype=float)
    den = np.zeros(len(lag_list), dtype=float)

    for s in range(recall_sims.shape[1]):
        r = recall_sims[:, s]
        r = r[r > 0]
        if len(r) < 2:
            continue
        recalled = set()
        for t in range(len(r) - 1):
            i, j = int(r[t]), int(r[t + 1])
            recalled.add(i)
            remaining = set(range(1, N + 1)) - recalled
            for l in lag_list:
                k = i + l
                if 1 <= k <= N and k in remaining:
                    den[idx[l]] += 1
            l_real = j - i
            if l_real != 0 and l_real in idx:
                num[idx[l_real]] += 1

    crp = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return np.array(lag_list), crp, num, den


# ─────────────────────────────────────────────────────────────────────
# Conditional FORWARD lag scalar summaries  (opportunity-corrected)
# ─────────────────────────────────────────────────────────────────────

def conditional_forward_lag_rates(lags, num, den, large_lag_thresh=4):
    """
    From lag_crp_with_counts output:

    Returns
    -------
    p_plus_one    : P(ℓ = +1 | opportunity)
    p_large_fwd   : P(ℓ ≥ k  | opportunity)
    """
    lags = np.asarray(lags, dtype=int)
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    valid = (lags != 0) & (den > 0)

    m = valid & (lags == 1)
    p_plus_one = (num[m].sum() / den[m].sum()) if den[m].sum() > 0 else np.nan

    m = valid & (lags >= large_lag_thresh)
    p_large_fwd = (num[m].sum() / den[m].sum()) if den[m].sum() > 0 else np.nan

    return p_plus_one, p_large_fwd


# ─────────────────────────────────────────────────────────────────────
# Conditional BACKWARD lag scalar summaries  (opportunity-corrected)
# ─────────────────────────────────────────────────────────────────────

def conditional_backward_lag_rates(lags, num, den, large_lag_thresh=4):
    """
    From lag_crp_with_counts output:

    Returns
    -------
    p_minus_one   : P(ℓ = −1 | opportunity)
    p_large_bwd   : P(ℓ ≤ −k | opportunity)
    """
    lags = np.asarray(lags, dtype=int)
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    valid = (lags != 0) & (den > 0)

    m = valid & (lags == -1)
    p_minus_one = (num[m].sum() / den[m].sum()) if den[m].sum() > 0 else np.nan

    m = valid & (lags <= -large_lag_thresh)
    p_large_bwd = (num[m].sum() / den[m].sum()) if den[m].sum() > 0 else np.nan

    return p_minus_one, p_large_bwd


# ─────────────────────────────────────────────────────────────────────
# Unconditional (observed) transition helpers
# ─────────────────────────────────────────────────────────────────────

def get_lags_from_recall_sims(recall_sims):
    """All observed lags (next − current), pooled over trials, excluding 0."""
    lags = []
    for s in range(recall_sims.shape[1]):
        seq = recall_sims[:, s]
        seq = seq[seq > 0].astype(int)
        if len(seq) < 2:
            continue
        lags.extend(np.diff(seq))
    return np.asarray(lags, dtype=int)


def unconditional_transition_summaries(recall_sims, large_lag_thresh=4):
    """
    Absolute-value unconditional summaries (cheap sanity check).

    Returns
    -------
    p_abs1      : P(|ℓ| = 1)
    p_abs_ge_k  : P(|ℓ| ≥ k)
    mean_abs    : E[|ℓ|]
    """
    lags = get_lags_from_recall_sims(recall_sims)
    lags = lags[lags != 0]
    if lags.size == 0:
        return np.nan, np.nan, np.nan
    al = np.abs(lags)
    return float(np.mean(al == 1)), float(np.mean(al >= large_lag_thresh)), float(np.mean(al))


# ─────────────────────────────────────────────────────────────────────
# Recall accuracy
# ─────────────────────────────────────────────────────────────────────

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

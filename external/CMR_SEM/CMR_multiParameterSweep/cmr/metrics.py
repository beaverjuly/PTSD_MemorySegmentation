"""
CMR Multi-Parameter Sweep — Behavioral Analysis Metrics
========================================================
SPC, PFR, lag-CRP (with counts, conditional & unconditional forward lags),
recall accuracy, and sweep-aligned scalar metric extraction.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# SPC, PFR, lag-CRP
# ─────────────────────────────────────────────────────────────────────

def compute_spc(recall_sims, N):
    """Serial Position Curve."""
    spc = np.zeros(N)
    for j in range(1, N + 1):
        spc[j - 1] = np.mean(np.any(recall_sims == j, axis=0))
    return spc


def compute_pfr(recall_sims, N):
    """Probability of First Recall."""
    first = recall_sims[0, :]
    first = first[first > 0]

    pfr = np.zeros(N)
    if len(first) > 0:
        for j in range(1, N + 1):
            pfr[j - 1] = np.mean(first == j)
    return pfr


def compute_lag_crp(recall_sims, N):
    """Lag-CRP with opportunity correction."""
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
            cur = seq[t]
            nxt = seq[t + 1]

            recalled.add(cur)
            remaining = [j for j in range(1, N + 1) if j not in recalled]

            for j in remaining:
                L = j - cur
                denom[lag_to_idx[L]] += 1

            L_obs = nxt - cur
            numer[lag_to_idx[L_obs]] += 1

    crp = np.zeros_like(numer)
    valid = denom > 0
    crp[valid] = numer[valid] / denom[valid]

    return lag_vals, crp


# ─────────────────────────────────────────────────────────────────────
# lag-CRP with numerator / denominator counts
# ─────────────────────────────────────────────────────────────────────

def lag_crp_with_counts(recall_sims, N):
    """
    Returns:
      lags: array of lags from -(N-1) .. (N-1), excluding 0
      crp: conditional probs
      num: numerator counts (# realized transitions)
      den: denominator opportunity counts (# possible transitions given what's left)
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
            i = int(r[t])
            j = int(r[t + 1])
            recalled.add(i)

            remaining = set(range(1, N + 1)) - recalled
            for l in lag_list:
                k = i + l
                if 1 <= k <= N and (k in remaining):
                    den[idx[l]] += 1

            l_real = j - i
            if l_real != 0 and l_real in idx:
                num[idx[l_real]] += 1

    crp = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return np.array(lag_list), crp, num, den


# ─────────────────────────────────────────────────────────────────────
# Conditional forward lag scalar summaries
# ─────────────────────────────────────────────────────────────────────

def conditional_forward_lag_rates_from_counts(lags, num, den, large_lag_thresh=4):
    """
    Opportunity-corrected FORWARD scalar summaries consistent with lag-CRP.

    Returns:
      lag_plus_one   = sum_num(lag=+1) / sum_den(lag=+1)
      large_forward  = sum_num(lag>=k) / sum_den(lag>=k)
    """
    lags = np.asarray(lags, dtype=int)
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)

    valid = (lags != 0) & (den > 0)

    # forward lag +1
    lagp1 = valid & (lags == 1)
    den_lagp1 = den[lagp1].sum()
    lag_plus_one = (num[lagp1].sum() / den_lagp1) if den_lagp1 > 0 else np.nan

    # forward large lag >= k
    large = valid & (lags >= large_lag_thresh)
    den_large = den[large].sum()
    large_forward = (num[large].sum() / den_large) if den_large > 0 else np.nan

    return lag_plus_one, large_forward


# ─────────────────────────────────────────────────────────────────────
# Unconditional forward lag summaries
# ─────────────────────────────────────────────────────────────────────

def get_lags_from_recall_sims(recall_sims):
    """Unconditional observed transition lags: lag = next - current, pooled over trials."""
    lags = []
    for s in range(recall_sims.shape[1]):
        seq = recall_sims[:, s]
        seq = seq[seq > 0].astype(int)
        if len(seq) < 2:
            continue
        lags.extend(np.diff(seq))
    return np.asarray(lags, dtype=int)


def lag_stats_unconditional_forward(recall_sims, large_lag_thresh=4):
    """
    Unconditional summaries over OBSERVED transitions (FORWARD-only for rates):
      lag_plus_one_rate   = P(lag=+1)
      large_forward_rate  = P(lag>=k)
      mean_abs_lag        = E[|lag|]
    """
    lags = get_lags_from_recall_sims(recall_sims)
    lags = lags[lags != 0]
    if lags.size == 0:
        return np.nan, np.nan, np.nan

    lag_plus_one_rate = np.mean(lags == 1)
    large_forward_rate = np.mean(lags >= large_lag_thresh)
    mean_abs_lag = np.mean(np.abs(lags))
    return lag_plus_one_rate, large_forward_rate, mean_abs_lag


# ─────────────────────────────────────────────────────────────────────
# Recall accuracy
# ─────────────────────────────────────────────────────────────────────

def recall_accuracy(recall_sims, N, unique=True):
    """
    Scalar accuracy for free recall = mean proportion of list items recalled.
    For each sim: acc = |unique recalled| / N. Returns E[acc].
    """
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
    return float(np.mean(acc)) if len(acc) else np.nan


def sweep_scalar_metric_array(sweep_results, param_grid, metric_fn):
    """
    Returns y[i] = metric_fn(recall_sims) aligned with param_grid.
    metric_fn should be a closure accepting recall_sims only.
    """
    from .diagnostics import _get_sweep_entry  # avoid circular at module level

    param_grid = np.asarray(param_grid, dtype=float)
    y = np.full(len(param_grid), np.nan, dtype=float)
    for i, v in enumerate(param_grid):
        entry = _get_sweep_entry(sweep_results, v)
        y[i] = metric_fn(entry["recall_sims"])
    return y

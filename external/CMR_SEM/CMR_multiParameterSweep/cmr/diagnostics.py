"""
CMR Multi-Parameter Sweep — Model Internal Diagnostics
=======================================================
Cue advantage computation, trace extraction helpers, and
learned-association asymmetry analysis.
"""

import numpy as np
from collections import defaultdict

from .config import N, BASE_PARAMS
from .simulation import simulate_single_trial


# ─────────────────────────────────────────────────────────────────────
# Robust sweep-entry accessor
# ─────────────────────────────────────────────────────────────────────

def _get_sweep_entry(sweep_results, v):
    """Robustly fetch sweep_results[v] when keys might be float-ish."""
    if v in sweep_results:
        return sweep_results[v]
    if float(v) in sweep_results:
        return sweep_results[float(v)]
    for k in sweep_results:
        try:
            if np.isclose(float(k), float(v), atol=1e-12, rtol=0):
                return sweep_results[k]
        except Exception:
            pass
    raise KeyError(f"Missing sweep entry for {v}")


# ─────────────────────────────────────────────────────────────────────
# Cue advantage diagnostics
# ─────────────────────────────────────────────────────────────────────

def compute_cue_diagnostics(B_rec, n_sims=500, seed=2026,
                            gamma_fc_val=None, eta_val=None, B_encD_scale=1.0):
    """Compute neighbor cue advantage diagnostics."""
    rng = np.random.default_rng(seed)

    deltas_all = []
    deltas_by_pos = defaultdict(list)
    delta_forward = []
    delta_backward = []

    for s in range(n_sims):
        _, _, _, _, diag = simulate_single_trial(
            B_rec=B_rec,
            rng=rng,
            gamma_fc_val=gamma_fc_val,
            eta_val=eta_val,
            B_encD_scale=B_encD_scale,
            record_diagnostics=True
        )

        if diag is not None:
            cue = diag.get("cue", diag)

            deltas_all.extend(cue.get("deltas_all", []))
            delta_forward.extend(cue.get("delta_forward", []))
            delta_backward.extend(cue.get("delta_backward", []))

            for pos, vals in cue.get("deltas_by_pos", {}).items():
                deltas_by_pos[pos].extend(vals)

    return {
        "deltas_all": deltas_all,
        "deltas_by_pos": dict(deltas_by_pos),
        "delta_forward": delta_forward,
        "delta_backward": delta_backward,
    }


# ─────────────────────────────────────────────────────────────────────
# Trace extraction helpers
# ─────────────────────────────────────────────────────────────────────

def get_trace_sims(sweep_results, val):
    """Robustly retrieve trace_sims for a parameter value."""
    if val in sweep_results:
        return sweep_results[val]["trace_sims"]
    if float(val) in sweep_results:
        return sweep_results[float(val)]["trace_sims"]
    for k in sweep_results.keys():
        try:
            if np.isclose(float(k), float(val), atol=1e-12, rtol=0):
                return sweep_results[k]["trace_sims"]
        except Exception:
            pass
    raise KeyError(f"Could not find traces for val={val}")


def mean_curve_over_sims(trace_sims, field):
    """
    field in {"rad","c_norm","cos_after"} etc.
    Returns:
      x: step indices starting at 1
      mean: nanmean across sims at each step
    """
    ts = [t for t in trace_sims if (t is not None and field in t)]
    if len(ts) == 0:
        return np.array([]), np.array([])
    lens = [len(t[field]) for t in ts]
    T = max(lens)
    M = np.full((len(ts), T), np.nan, dtype=float)
    for i, t in enumerate(ts):
        arr = np.asarray(t[field], dtype=float)
        M[i, :len(arr)] = arr
    mean = np.nanmean(M, axis=0)
    x = np.arange(1, T + 1)
    return x, mean


def mean_scalar_over_sims(trace_sims, field):
    """
    field in {"f_recency_mass","f_entropy"} (logged per retrieval attempt).
    Returns: mean over sims of per-sim mean (robust to variable length).
    """
    vals = []
    for t in trace_sims:
        if t is None or field not in t:
            continue
        arr = np.asarray(t[field], dtype=float)
        if arr.size:
            vals.append(np.nanmean(arr))
    return np.nanmean(vals) if len(vals) else np.nan


def mean_vector_over_sims(trace_sims, field):
    """Mean of a length-N vector field across sims (e.g. f_mean_by_pos)."""
    vecs = []
    for t in trace_sims:
        if t is None or field not in t:
            continue
        v = np.asarray(t[field], dtype=float)
        if v.size == N:
            vecs.append(v)
    if len(vecs) == 0:
        return np.full(N, np.nan)
    return np.nanmean(np.vstack(vecs), axis=0)


# ─────────────────────────────────────────────────────────────────────
# Asymmetry in learned associations
# ─────────────────────────────────────────────────────────────────────

def neighbor_fc_asymmetry_means_from_sweep(sweep_results, param_grid):
    """
    Returns arrays aligned with param_grid for M_FC-based neighbor
    context-input asymmetry:

      mean_all[i] = E[Δ_FC]
      mean_fwd[i] = E[Δ_FC | ℓ=+1]
      mean_bwd[i] = E[Δ_FC | ℓ=-1]

    Falls back to old Δf keys (deltas_all / delta_forward / delta_backward)
    if the newer keys are absent.
    """
    param_grid = np.asarray(param_grid, dtype=float)

    mean_all = np.full(len(param_grid), np.nan, dtype=float)
    mean_fwd = np.full(len(param_grid), np.nan, dtype=float)
    mean_bwd = np.full(len(param_grid), np.nan, dtype=float)

    for i, v in enumerate(param_grid):
        entry = _get_sweep_entry(sweep_results, v)
        diag = entry.get("cue_diag", None)
        if diag is None:
            continue

        cue = diag.get("cue", diag)

        da = cue.get("deltas_fc_all", cue.get("deltas_all", []))
        df = cue.get("delta_fc_forward", cue.get("delta_forward", []))
        db = cue.get("delta_fc_backward", cue.get("delta_backward", []))

        da = np.asarray(da, dtype=float)
        df = np.asarray(df, dtype=float)
        db = np.asarray(db, dtype=float)

        mean_all[i] = np.nanmean(da) if da.size else np.nan
        mean_fwd[i] = np.nanmean(df) if df.size else np.nan
        mean_bwd[i] = np.nanmean(db) if db.size else np.nan

    return mean_all, mean_fwd, mean_bwd

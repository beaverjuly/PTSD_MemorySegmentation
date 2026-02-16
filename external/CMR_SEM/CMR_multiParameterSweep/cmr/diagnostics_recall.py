"""
CMR Multi-Parameter Sweep — Recall-Stage Diagnostics
=====================================================
Everything computed *during retrieval attempts / after each recall step*,
or derived from per-trial retrieval traces:

* Context-update sanity traces: radicand, dot, rho, context norm, cos(c, c_in).
* Evidence-shape traces: f_mean_by_pos, f_mean_centered_by_pos, f_entropy,
  f_recency_mass.
* **Neighbor item-evidence asymmetry during selection** (Δf):
  Δf = f_in(i+1) − f_in(i−1)  where  f_in = M_CF · c.
  Measures whether the combined retrieval cue (episodic + semantic)
  provides more evidence for the forward vs. backward serial-position
  neighbor when the model is about to select its next recall.
* **Neighbor context-to-item-input alignment (Δ_FC) asymmetry during
  updating**:
  Δ_FC = cos(c, c_in(i+1)) − cos(c, c_in(i−1))
  where c_in(j) = M_FC · f_j / ‖M_FC · f_j‖.
  Measures whether the current context is more aligned with the
  context representation that the forward neighbor would reactivate
  through M_FC.  This reflects the asymmetry in learned feature→context
  associations and predicts which direction context will drift upon the
  next retrieval update.
"""

import numpy as np
from collections import defaultdict

from .config import N
from .utils import _get_sweep_entry


# ─────────────────────────────────────────────────────────────────────
# Trace extraction helpers
# ─────────────────────────────────────────────────────────────────────

def get_trace_sims(sweep_results, val):
    """Return per-trial trace list for a given parameter value."""
    entry = _get_sweep_entry(sweep_results, val)
    return entry.get("trace_sims", [])


def mean_curve_over_sims(trace_sims, field):
    """
    Compute the nanmean curve of a variable-length per-step field
    (e.g. ``"rad"``, ``"c_norm"``, ``"cos_after"``, ``"dot"``).

    Returns (x, mean) where x starts at 1.
    """
    ts = [t for t in trace_sims if t is not None and field in t]
    if not ts:
        return np.array([]), np.array([])
    lens = [len(t[field]) for t in ts]
    T = max(lens)
    M = np.full((len(ts), T), np.nan, dtype=float)
    for i, t in enumerate(ts):
        arr = np.asarray(t[field], dtype=float)
        M[i, :len(arr)] = arr
    return np.arange(1, T + 1), np.nanmean(M, axis=0)


def mean_scalar_over_sims(trace_sims, field):
    """Mean over sims of per-sim mean for a per-attempt scalar field
    (e.g. ``"f_entropy"``, ``"f_recency_mass"``)."""
    vals = []
    for t in trace_sims:
        if t is None or field not in t:
            continue
        arr = np.asarray(t[field], dtype=float)
        if arr.size:
            vals.append(np.nanmean(arr))
    return np.nanmean(vals) if vals else np.nan


def mean_vector_over_sims(trace_sims, field):
    """Mean of a length-N vector field across sims
    (e.g. ``"f_mean_by_pos"``, ``"f_mean_centered_by_pos"``)."""
    vecs = []
    for t in trace_sims:
        if t is None or field not in t:
            continue
        v = np.asarray(t[field], dtype=float)
        if v.size == N:
            vecs.append(v)
    if not vecs:
        return np.full(N, np.nan)
    return np.nanmean(np.vstack(vecs), axis=0)


# ─────────────────────────────────────────────────────────────────────
# Aggregators: item-evidence asymmetry & FC alignment from diagnostics
# ─────────────────────────────────────────────────────────────────────

def _aggregate_neighbor_asymmetry(all_diagnostics, key):
    """
    Pool a neighbor-asymmetry diagnostic (``"item_evidence"`` or
    ``"fc_alignment"``) across all per-trial diagnostic dicts.

    Parameters
    ----------
    all_diagnostics : list[dict | None]
        Per-trial diagnostic dicts from ``run_simulation_with_diagnostics``.
    key : str
        ``"item_evidence"`` or ``"fc_alignment"``.

    Returns
    -------
    dict with ``deltas_all``, ``deltas_by_pos``, ``delta_forward``,
    ``delta_backward``.
    """
    deltas_all = []
    deltas_by_pos = defaultdict(list)
    delta_forward = []
    delta_backward = []

    for diag in all_diagnostics:
        if diag is None or key not in diag:
            continue
        sub = diag[key]
        deltas_all.extend(sub.get("deltas_all", []))
        delta_forward.extend(sub.get("delta_forward", []))
        delta_backward.extend(sub.get("delta_backward", []))
        for pos, vals in sub.get("deltas_by_pos", {}).items():
            deltas_by_pos[pos].extend(vals)

    return {
        "deltas_all": deltas_all,
        "deltas_by_pos": dict(deltas_by_pos),
        "delta_forward": delta_forward,
        "delta_backward": delta_backward,
    }


def aggregate_item_evidence_asymmetry(all_diagnostics):
    """Aggregate neighbor item-evidence asymmetry (Δf) across trials."""
    return _aggregate_neighbor_asymmetry(all_diagnostics, "item_evidence")


def aggregate_fc_alignment_asymmetry(all_diagnostics):
    """Aggregate neighbor FC-alignment asymmetry (Δ_FC) across trials."""
    return _aggregate_neighbor_asymmetry(all_diagnostics, "fc_alignment")


# ─────────────────────────────────────────────────────────────────────
# Standalone diagnostic runner (when traces are NOT collected)
# ─────────────────────────────────────────────────────────────────────

def compute_item_evidence_asymmetry(B_rec, n_sims=500, seed=2026,
                                    gamma_fc_val=None, eta_val=None,
                                    B_encD_scale=1.0):
    """
    Run a small batch of trials with diagnostics enabled and return the
    aggregated **item-evidence asymmetry** dict.  Use when the main
    simulation batch was run without diagnostics.
    """
    from .simulation import simulate_single_trial  # local to avoid circular

    rng = np.random.default_rng(seed)
    diag_list = []
    for _ in range(n_sims):
        _, _, _, _, diag = simulate_single_trial(
            B_rec=B_rec, rng=rng,
            gamma_fc_val=gamma_fc_val, eta_val=eta_val,
            B_encD_scale=B_encD_scale, record_diagnostics=True,
        )
        diag_list.append(diag)
    return _aggregate_neighbor_asymmetry(diag_list, "item_evidence")


# ─────────────────────────────────────────────────────────────────────
# Sweep-level convenience extractors
# ─────────────────────────────────────────────────────────────────────


def item_evidence_asymmetry_means(sweep_results, param_grid):
    """E[Δf], E[Δf | ℓ=+1], E[Δf | ℓ=−1] aligned with param_grid."""
    param_grid = np.asarray(param_grid, dtype=float)
    mean_all = np.full(len(param_grid), np.nan)
    mean_fwd = np.full(len(param_grid), np.nan)
    mean_bwd = np.full(len(param_grid), np.nan)
    for i, v in enumerate(param_grid):
        entry = _get_sweep_entry(sweep_results, v)
        d = entry.get("item_evidence_diag")
        if d is None:
            continue
        da = np.asarray(d.get("deltas_all", []), dtype=float)
        df = np.asarray(d.get("delta_forward", []), dtype=float)
        db = np.asarray(d.get("delta_backward", []), dtype=float)
        mean_all[i] = np.nanmean(da) if da.size else np.nan
        mean_fwd[i] = np.nanmean(df) if df.size else np.nan
        mean_bwd[i] = np.nanmean(db) if db.size else np.nan
    return mean_all, mean_fwd, mean_bwd


def fc_alignment_asymmetry_means(sweep_results, param_grid):
    """E[Δ_FC], E[Δ_FC | ℓ=+1], E[Δ_FC | ℓ=−1] aligned with param_grid."""
    param_grid = np.asarray(param_grid, dtype=float)
    mean_all = np.full(len(param_grid), np.nan)
    mean_fwd = np.full(len(param_grid), np.nan)
    mean_bwd = np.full(len(param_grid), np.nan)
    for i, v in enumerate(param_grid):
        entry = _get_sweep_entry(sweep_results, v)
        d = entry.get("fc_alignment_diag")
        if d is None:
            continue
        da = np.asarray(d.get("deltas_all", []), dtype=float)
        df = np.asarray(d.get("delta_forward", []), dtype=float)
        db = np.asarray(d.get("delta_backward", []), dtype=float)
        mean_all[i] = np.nanmean(da) if da.size else np.nan
        mean_fwd[i] = np.nanmean(df) if df.size else np.nan
        mean_bwd[i] = np.nanmean(db) if db.size else np.nan
    return mean_all, mean_fwd, mean_bwd

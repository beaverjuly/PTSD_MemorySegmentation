"""
CMR Encoding-Stage Diagnostics
================================
Core functions for inspecting M_FC and M_CF after encoding.

Provides:
1. Remap weight matrices from item-index space → serial-position space
2. Lag-strength profiles (mean |weight| at each serial-position lag)
3. Forward / backward asymmetry (lag +1 vs lag −1)

The sweep-level convenience functions are retained for compatibility
but are not used in the targeted-simulation notebook.
"""

from __future__ import annotations

import numpy as np

from .config import N, pres_indices
from .utils import _get_sweep_entry


# =====================================================================
# 1. Remap to serial-position space
# =====================================================================

def remap_to_serial_position_space(
    W: np.ndarray,
    pres_indices=pres_indices,
    N: int = N,
) -> np.ndarray:
    """
    Re-index a weight matrix so rows/columns correspond to serial
    position (study order) instead of raw item indices.

    ``W_sp[sp_i, sp_j] = W[item(sp_i), item(sp_j)]``
    """
    W_sp = np.zeros((N, N), dtype=float)
    for sp_i in range(N):
        for sp_j in range(N):
            row = int(pres_indices[sp_i]) - 1
            col = int(pres_indices[sp_j]) - 1
            W_sp[sp_i, sp_j] = W[row, col]
    return W_sp


# =====================================================================
# 2. Lag-strength profile
# =====================================================================

def lag_strength_profile(
    W_sp: np.ndarray,
    N: int = N,
    absolute: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean weight at each serial-position lag δ ∈ {−(N−1), …, +(N−1)}.

    δ = +1 → forward neighbour, δ = −1 → backward neighbour.
    """
    lags_out, means_out = [], []
    for delta in range(-(N - 1), N):
        vals = []
        for row in range(N):
            col = row + delta
            if 0 <= col < N:
                w = abs(W_sp[row, col]) if absolute else W_sp[row, col]
                vals.append(w)
        lags_out.append(delta)
        means_out.append(float(np.mean(vals)) if vals else 0.0)
    return np.asarray(lags_out, dtype=int), np.asarray(means_out, dtype=float)


# =====================================================================
# 3. Forward / backward asymmetry
# =====================================================================

def compute_forward_backward_asymmetry(
    W_sp: np.ndarray,
    N: int = N,
    absolute: bool = True,
) -> tuple[float, float, float]:
    """
    Compare mean weight at lag +1 (forward) vs lag −1 (backward).

    Returns (fwd_mean, bwd_mean, asymmetry = fwd − bwd).
    """
    def _lag_mean(delta: int) -> float:
        vals = []
        for row in range(N):
            col = row + delta
            if 0 <= col < N:
                w = abs(W_sp[row, col]) if absolute else W_sp[row, col]
                vals.append(w)
        return float(np.mean(vals)) if vals else 0.0

    fwd = _lag_mean(+1)
    bwd = _lag_mean(-1)
    return fwd, bwd, fwd - bwd


# =====================================================================
# 4. Sweep-level convenience functions (retained for compatibility)
# =====================================================================

def sweep_lag_strength_profile(sweep_results, param_grid,
                               matrix_key="net_w_fc", absolute=True):
    """Lag-strength profile for every parameter-grid value."""
    param_grid = np.asarray(param_grid, dtype=float)
    lags = None
    profiles = []
    for v in param_grid:
        entry = _get_sweep_entry(sweep_results, v)
        W_sp = remap_to_serial_position_space(entry[matrix_key])
        l, m = lag_strength_profile(W_sp, absolute=absolute)
        if lags is None:
            lags = l
        profiles.append(m)
    return lags, np.vstack(profiles)


def sweep_forward_backward_asymmetry(sweep_results, param_grid,
                                     matrix_key="net_w_fc", absolute=True):
    """Forward/backward asymmetry for every parameter-grid value."""
    param_grid = np.asarray(param_grid, dtype=float)
    n = len(param_grid)
    fwd_arr = np.full(n, np.nan)
    bwd_arr = np.full(n, np.nan)
    asym_arr = np.full(n, np.nan)
    for i, v in enumerate(param_grid):
        entry = _get_sweep_entry(sweep_results, v)
        W_sp = remap_to_serial_position_space(entry[matrix_key])
        fwd_arr[i], bwd_arr[i], asym_arr[i] = compute_forward_backward_asymmetry(
            W_sp, absolute=absolute)
    return fwd_arr, bwd_arr, asym_arr

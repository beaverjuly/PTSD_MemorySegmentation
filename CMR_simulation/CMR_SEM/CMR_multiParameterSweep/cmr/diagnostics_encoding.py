"""
CMR Encoding-Stage Diagnostics (γ_fc & B_enc scale only)

Helper functions that inspect the association matrices M_FC and M_CF
**after** the study list has been encoded but **before** retrieval begins.

What this module provides
-------------------------
1) **Lag-strength profile** – For every possible serial-position lag
   δ ∈ {−(N−1), …, +(N−1)}, compute the average absolute weight in the
   matrix. Lag 0 is the diagonal (item associated with its own context),
   lag +1 is the forward neighbour, lag −1 is the backward neighbour, etc.

2) **Forward / backward asymmetry** – The difference between the mean
   weight at lag +1 and the mean weight at lag −1. A positive value
   means the matrix encodes a stronger *forward* association than a
   *backward* one.

Why only these readouts?
------------------------
- Lag profiles and asymmetry of M_FC reveal the temporal-contiguity
  structure that drives CRP curves during recall.
- Lag profiles of M_CF reveal global accessibility shifts (SPC / PFR).
- Matrix norms, Δ_FC alignment, η effects, and β_rec dynamics are
  omitted here because they reflect retrieval-stage mechanics rather
  than the structure laid down during encoding (see notebook for details).

Terminology
-----------
- **item-index space** – rows/columns correspond to item feature indices
  (the order in which items appear in the feature vector, which is
  randomised across trials).
- **serial-position space** – rows/columns correspond to presentation
  order (position 0 = first studied item, position N−1 = last).
"""

from __future__ import annotations

import numpy as np

from .config import N, pres_indices
from .utils import _get_sweep_entry


# =====================================================================
# 1. Remap a weight matrix from item-index space → serial-position space
# =====================================================================

def remap_to_serial_position_space(
    W: np.ndarray,
    pres_indices=pres_indices,
    N: int = N
) -> np.ndarray:
    """
    Re-index a weight matrix so that rows and columns correspond to
    **serial position** (study order) instead of raw item indices.

    Parameters
    ----------
    W : (n_items, n_items) ndarray
        Weight matrix in item-index space (e.g. the raw M_FC or M_CF
        stored by the model after encoding).
    pres_indices : (N,) array-like
        ``pres_indices[sp]`` gives the 1-based feature index of the item
        presented at serial position ``sp``.
    N : int
        Number of items on the study list.

    Returns
    -------
    W_sp : (N, N) ndarray
        The same weights, but now ``W_sp[sp_i, sp_j]`` holds the weight
        that originally sat at ``W[item(sp_i), item(sp_j)]``.

    Why we need this
    ----------------
    The model stores weights using item feature indices, which are
    randomly assigned. To compare across conditions we need a common
    coordinate system — serial position — so that "lag +1" always means
    "one study position forward."
    """
    W_sp = np.zeros((N, N), dtype=float)
    for sp_i in range(N):
        for sp_j in range(N):
            # pres_indices is 1-based, so subtract 1 for 0-based indexing
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
    absolute: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean weight at every serial-position lag δ.

    "Lag" here means the column position minus the row position:
        δ  =  sp_col − sp_row

    So δ = +1 captures forward-neighbour associations and δ = −1
    captures backward-neighbour associations.

    Parameters
    ----------
    W_sp : (N, N) ndarray
        Weight matrix in serial-position space (output of
        ``remap_to_serial_position_space``).
    N : int
        Number of items on the study list.
    absolute : bool, default True
        If True, take the absolute value of each weight before
        averaging. This is the standard choice because we care about
        association *strength* regardless of sign.

    Returns
    -------
    lags : (2N−1,) int array
        The lag values from −(N−1) to +(N−1).
    means : (2N−1,) float array
        The mean (absolute) weight at each lag.
    """
    lags_out = []
    means_out = []

    for delta in range(-(N - 1), N):  # δ from −(N−1) to +(N−1)
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
    absolute: bool = True
) -> tuple[float, float, float]:
    """
    Compare the mean weight at lag +1 (forward neighbour) with the
    mean weight at lag −1 (backward neighbour).

    Parameters
    ----------
    W_sp : (N, N) ndarray
        Serial-position-space weight matrix.
    N : int
        Number of items on the study list.
    absolute : bool
        Take absolute values before averaging.

    Returns
    -------
    fwd_mean  : float
        Mean weight at lag δ = +1
    bwd_mean  : float
        Mean weight at lag δ = −1
    asymmetry : float
        fwd_mean − bwd_mean  (positive ⇒ forward bias)
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
# 4. Sweep-level convenience functions
# =====================================================================

def sweep_lag_strength_profile(
    sweep_results,
    param_grid,
    matrix_key: str = "net_w_fc",
    absolute: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the lag-strength profile for every parameter-grid value.

    Parameters
    ----------
    sweep_results : list[dict] | dict
        Output of ``sweep_one_param``. Each entry must contain a key
        ``matrix_key`` holding the (n_items, n_items) weight matrix.
        (This function uses ``_get_sweep_entry`` so it supports either
        dict- or list-style sweep containers, as defined in your utils.)
    param_grid : array-like of float
        The swept parameter values (e.g. ``gamma_fc_grid``).
    matrix_key : str
        ``"net_w_fc"`` for M_FC or ``"net_w_cf"`` for M_CF.
    absolute : bool
        Passed through to ``lag_strength_profile``.

    Returns
    -------
    lags : (2N−1,) int array
        Shared lag axis.
    profiles : (len(param_grid), 2N−1) float array
        One row per grid value; each row is the lag-strength profile.
    """
    param_grid = np.asarray(param_grid, dtype=float)
    lags = None
    profiles = []

    for v in param_grid:
        entry = _get_sweep_entry(sweep_results, v)
        W = entry[matrix_key]
        W_sp = remap_to_serial_position_space(W)
        l, m = lag_strength_profile(W_sp, absolute=absolute)
        if lags is None:
            lags = l
        profiles.append(m)

    return lags, np.vstack(profiles)


def sweep_forward_backward_asymmetry(
    sweep_results,
    param_grid,
    matrix_key: str = "net_w_fc",
    absolute: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward/backward asymmetry for every parameter-grid value.

    Returns
    -------
    fwd_arr  : (len(param_grid),) float array
        Mean weight at lag +1
    bwd_arr  : (len(param_grid),) float array
        Mean weight at lag −1
    asym_arr : (len(param_grid),) float array
        fwd − bwd
    """
    param_grid = np.asarray(param_grid, dtype=float)
    n = len(param_grid)
    fwd_arr = np.full(n, np.nan, dtype=float)
    bwd_arr = np.full(n, np.nan, dtype=float)
    asym_arr = np.full(n, np.nan, dtype=float)

    for i, v in enumerate(param_grid):
        entry = _get_sweep_entry(sweep_results, v)
        W = entry[matrix_key]
        W_sp = remap_to_serial_position_space(W)
        fwd_arr[i], bwd_arr[i], asym_arr[i] = compute_forward_backward_asymmetry(
            W_sp, absolute=absolute
        )

    return fwd_arr, bwd_arr, asym_arr


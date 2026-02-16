"""
CMR Multi-Parameter Sweep — Encoding-Stage Diagnostics
=======================================================
Summaries of learned association matrices (M_FC, M_CF) that reflect
the structure written during the encoding phase.

Because retrieval learning rates are zero in the default configuration,
the weight matrices are deterministic within a parameter condition
(identical across trials), so the stored ``net_w_fc`` / ``net_w_cf``
from any single trial is representative.

Key quantities
--------------
* **Band-strength profile** — mean absolute weight at each
  serial-position lag δ ∈ {−(N−1), …, N−1}.  The lag-0 band is the
  diagonal (self-association), lag +1 is the forward-neighbor band, etc.
* **Forward / backward neighbor-band asymmetry** —
  mean(|W_sp[i, i+1]|) − mean(|W_sp[i, i−1]|) for the remapped matrix.
* **Matrix norms** — Frobenius norm and mean absolute weight,
  summarising overall association strength.
"""

import numpy as np

from .config import N, pres_indices
from .utils import _get_sweep_entry


# ─────────────────────────────────────────────────────────────────────
# Remap to serial-position space
# ─────────────────────────────────────────────────────────────────────

def remap_to_serial_position_space(W, pres_indices=pres_indices, N=N):
    """
    Remap weight matrix from item-index space to serial-position space.

    ``W_sp[sp_i, sp_j]`` = ``W[item(sp_i), item(sp_j)]``
    where ``item(sp)`` is the feature index of the item presented at
    serial position ``sp``.
    """
    W_sp = np.zeros((N, N))
    for sp_i in range(N):
        for sp_j in range(N):
            W_sp[sp_i, sp_j] = W[int(pres_indices[sp_i] - 1),
                                  int(pres_indices[sp_j] - 1)]
    return W_sp


# ─────────────────────────────────────────────────────────────────────
# Band-strength profile
# ─────────────────────────────────────────────────────────────────────

def band_strength_profile(W_sp, N=N, absolute=True):
    """
    Mean weight at each serial-position lag δ.

    For M_FC (feature→context): ``W_sp[sp_c, sp_f]`` — the lag is
    ``δ = sp_f − sp_c`` (how far the feature's serial position is from
    the context row's serial position).

    Parameters
    ----------
    W_sp : (N, N) array in serial-position space.
    absolute : bool
        If True, take absolute values before averaging.

    Returns
    -------
    lags   : array of ints −(N−1) … (N−1)
    means  : corresponding mean weight per band
    """
    lags_out = []
    means_out = []
    for delta in range(-(N - 1), N):
        vals = []
        for i in range(N):
            j = i + delta          # sp_f = sp_c + delta
            if 0 <= j < N:
                vals.append(abs(W_sp[i, j]) if absolute else W_sp[i, j])
        lags_out.append(delta)
        means_out.append(np.mean(vals) if vals else 0.0)
    return np.array(lags_out), np.array(means_out)


# ─────────────────────────────────────────────────────────────────────
# Forward / backward neighbor-band asymmetry
# ─────────────────────────────────────────────────────────────────────

def neighbor_band_asymmetry(W_sp, N=N, absolute=True):
    """
    Forward-minus-backward neighbor-band mean.

    Returns
    -------
    fwd_mean : float — mean of band at δ = +1
    bwd_mean : float — mean of band at δ = −1
    asymmetry : float — fwd_mean − bwd_mean
    """
    def _band_mean(delta):
        vals = []
        for i in range(N):
            j = i + delta
            if 0 <= j < N:
                vals.append(abs(W_sp[i, j]) if absolute else W_sp[i, j])
        return np.mean(vals) if vals else 0.0

    fwd = _band_mean(+1)
    bwd = _band_mean(-1)
    return fwd, bwd, fwd - bwd


# ─────────────────────────────────────────────────────────────────────
# Matrix norms
# ─────────────────────────────────────────────────────────────────────

def matrix_norms(W):
    """
    Returns
    -------
    frob  : float — Frobenius norm ‖W‖_F
    mean_abs : float — mean |W_ij|
    """
    return float(np.linalg.norm(W, 'fro')), float(np.mean(np.abs(W)))


# ─────────────────────────────────────────────────────────────────────
# Sweep-level convenience extractors
# ─────────────────────────────────────────────────────────────────────

def sweep_matrix_norms(sweep_results, param_grid, matrix_key="net_w_fc"):
    """
    Frobenius norm and mean-abs weight across a sweep.

    Parameters
    ----------
    matrix_key : ``"net_w_fc"`` or ``"net_w_cf"``

    Returns
    -------
    frob_arr, mean_abs_arr : arrays aligned with param_grid
    """
    param_grid = np.asarray(param_grid, dtype=float)
    frob_arr = np.full(len(param_grid), np.nan)
    mabs_arr = np.full(len(param_grid), np.nan)
    for i, v in enumerate(param_grid):
        entry = _get_sweep_entry(sweep_results, v)
        W = entry.get(matrix_key)
        if W is not None:
            frob_arr[i], mabs_arr[i] = matrix_norms(W)
    return frob_arr, mabs_arr


def sweep_band_profiles(sweep_results, param_grid, matrix_key="net_w_fc",
                        absolute=True):
    """
    Band-strength profiles for every grid value.

    Returns
    -------
    lags : (2N−1,) int array
    profiles : (len(param_grid), 2N−1) float array
    """
    param_grid = np.asarray(param_grid, dtype=float)
    lags = None
    profiles = []
    for v in param_grid:
        entry = _get_sweep_entry(sweep_results, v)
        W = entry.get(matrix_key)
        W_sp = remap_to_serial_position_space(W)
        l, m = band_strength_profile(W_sp, absolute=absolute)
        if lags is None:
            lags = l
        profiles.append(m)
    return lags, np.vstack(profiles)


def sweep_neighbor_band_asymmetry(sweep_results, param_grid,
                                  matrix_key="net_w_fc", absolute=True):
    """
    Forward & backward neighbor-band means and asymmetry across a sweep.

    Returns
    -------
    fwd_arr, bwd_arr, asym_arr : arrays aligned with param_grid
    """
    param_grid = np.asarray(param_grid, dtype=float)
    fwd_arr = np.full(len(param_grid), np.nan)
    bwd_arr = np.full(len(param_grid), np.nan)
    asym_arr = np.full(len(param_grid), np.nan)
    for i, v in enumerate(param_grid):
        entry = _get_sweep_entry(sweep_results, v)
        W = entry.get(matrix_key)
        W_sp = remap_to_serial_position_space(W)
        f, b, a = neighbor_band_asymmetry(W_sp, absolute=absolute)
        fwd_arr[i], bwd_arr[i], asym_arr[i] = f, b, a
    return fwd_arr, bwd_arr, asym_arr


def sweep_recall_counts(sweep_results, param_grid):
    """
    Mean number of items recalled per trial (from recall_sims).

    Returns
    -------
    mean_counts : array aligned with param_grid
    """
    param_grid = np.asarray(param_grid, dtype=float)
    out = np.full(len(param_grid), np.nan)
    for i, v in enumerate(param_grid):
        entry = _get_sweep_entry(sweep_results, v)
        rs = entry.get("recall_sims")
        if rs is not None:
            out[i] = float(np.mean(np.sum(rs > 0, axis=0)))
    return out

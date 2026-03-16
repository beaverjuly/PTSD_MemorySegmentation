"""
CMR Boundary-Signal Simulation — Simulation Functions

Drift-schedule builders for the two hypotheses and CMR trial runner.

Schedule constructors produce a flat (N,) array of per-position
encoding drift rates.  All builders now accept a *list* of boundary
positions so multi-boundary designs are first-class citizens.

    build_boundary_schedule   — H1: fixed baseline + variable Δ at boundaries
    build_global_schedule     — H2: globally shifted baseline + fixed boundary value
    build_baseline_schedule   — convenience wrapper using config defaults
    describe_schedule         — human-readable summary of a drift array
"""

from __future__ import annotations

import numpy as np

from .config import (
    N, pres_indices, BASE_PARAMS, sem_mat,
    BOUNDARY_POSITIONS,
    B_NON_BOUNDARY_BASE, B_BOUNDARY_DELTA_BASE,
)


# =====================================================================
# Drift schedule builders
# =====================================================================

def build_boundary_schedule(
    B_non: float,
    delta: float,
    boundary_positions: list[int] | np.ndarray = BOUNDARY_POSITIONS,
) -> np.ndarray:
    """
    Flat drift with an additive boundary boost at each boundary.

        B[i] = B_non              for all non-boundary positions
        B[j] = B_non + delta      for every j in boundary_positions

    Parameters
    ----------
    B_non : float
        Baseline drift for non-boundary items.
    delta : float
        Additive boost at each boundary position.
    boundary_positions : list[int]
        1-based serial positions of boundary items.

    Returns
    -------
    B_encD : (N,) float array, clipped to [0, 1].
    """
    B = np.full(N, B_non, dtype=float)
    for j in boundary_positions:
        B[j - 1] = B_non + delta
    return np.clip(B, 0.0, 1.0)


def build_global_schedule(
    B_non: float,
    B_boundary: float,
    boundary_positions: list[int] | np.ndarray = BOUNDARY_POSITIONS,
) -> np.ndarray:
    """
    Globally shifted drift with a fixed boundary value.

        B[i] = B_non          for non-boundary positions
        B[j] = B_boundary     for every j in boundary_positions

    Parameters
    ----------
    B_non : float
        Drift for non-boundary items.
    B_boundary : float
        Drift at each boundary position.
    boundary_positions : list[int]
        1-based serial positions of boundary items.

    Returns
    -------
    B_encD : (N,) float array, clipped to [0, 1].
    """
    B = np.full(N, B_non, dtype=float)
    for j in boundary_positions:
        B[j - 1] = B_boundary
    return np.clip(B, 0.0, 1.0)


def build_baseline_schedule(
    boundary_positions: list[int] | np.ndarray = BOUNDARY_POSITIONS,
) -> np.ndarray:
    """Convenience: healthy baseline using config defaults."""
    return build_boundary_schedule(
        B_NON_BOUNDARY_BASE, B_BOUNDARY_DELTA_BASE, boundary_positions
    )


def describe_schedule(B_encD: np.ndarray, boundary_positions: list[int] | np.ndarray = BOUNDARY_POSITIONS) -> str:
    """Human-readable summary of a drift schedule."""
    bps = set(boundary_positions)
    non_vals = [B_encD[i] for i in range(len(B_encD)) if (i + 1) not in bps]
    bdy_vals = [B_encD[j - 1] for j in boundary_positions]
    parts = [
        f"N = {len(B_encD)}, boundaries at {sorted(boundary_positions)}",
        f"non-boundary drift: {np.mean(non_vals):.3f} (range {min(non_vals):.3f}\u2013{max(non_vals):.3f})",
        f"boundary drift:     {np.mean(bdy_vals):.3f} (range {min(bdy_vals):.3f}\u2013{max(bdy_vals):.3f})",
    ]
    return "\n".join(parts)


# =====================================================================
# Vector helpers
# =====================================================================

def _dot(a, b) -> float:
    return float((np.asarray(a).T @ np.asarray(b)).ravel()[0])


def _norm(v) -> float:
    return float(np.linalg.norm(v))


# =====================================================================
# Single-trial CMR
# =====================================================================

def simulate_single_trial(
    B_encD: np.ndarray,
    rng: np.random.Generator,
    gamma_fc: float | None = None,
    eta: float | None = None,
    B_rec: float | None = None,
):
    """
    One encode -> retrieve CMR trial.

    Parameters
    ----------
    B_encD : (N,) array
        Per-position encoding drift rates.
    rng : numpy RNG.
    gamma_fc, eta, B_rec : optional overrides.

    Returns
    -------
    recalls : (N,) int
        1-based serial positions (0-padded).
    times : (N,) float
        Cumulative recall times (ms).
    net_w_fc, net_w_cf : (N, N)
        Final weight matrices.
    """
    p = BASE_PARAMS
    gamma_fc = gamma_fc if gamma_fc is not None else p["gamma_fc"]
    eta = eta if eta is not None else p["eta"]
    B_rec = B_rec if B_rec is not None else p["B_rec"]

    eye_fc = 1.0 - gamma_fc
    lrate_fc_enc = gamma_fc
    lrate_cf_enc = p["lrate_cf_enc"]
    lrate_fc_rec = p["lrate_fc_rec"]
    lrate_cf_rec = p["lrate_cf_rec"]
    thresh = p["thresh"]
    rec_time = p["rec_time"]
    dt = p["dt"]
    tau = p["tau"]
    K = p["K"]
    L = p["L"]
    episodic_w = p["episodic_weight"]
    sem_w = p["sem_weight"]
    B_encD = np.asarray(B_encD, dtype=float)

    net_f = np.zeros((N, 1))
    net_c = np.zeros((N, 1))
    net_w_fc = np.eye(N) * eye_fc
    net_w_cf = np.zeros((N, N))

    # ── Encoding ──
    for pos in range(N):
        feat = int(pres_indices[pos]) - 1
        net_f[:] = 0.0
        net_f[feat] = 1.0

        net_c_in = net_w_fc @ net_f
        net_c_in /= _norm(net_c_in)

        B = float(B_encD[pos])
        dot = _dot(net_c, net_c_in)
        rho = np.sqrt(1.0 + B**2 * (dot**2 - 1.0)) - B * dot
        net_c = rho * net_c + B * net_c_in

        net_w_fc += (net_c @ net_f.T) * lrate_fc_enc
        net_w_cf += (net_f @ net_c.T) * lrate_cf_enc

    # ── Retrieval ──
    recalls = np.zeros(N, dtype=int)
    times = np.zeros(N, dtype=float)
    retrieved = np.zeros((N, 1), dtype=bool)
    thresholds = np.ones((N, 1))
    net_weights = episodic_w * net_w_cf + sem_w * sem_mat

    time_passed = 0.0
    recall_count = 0

    while time_passed < rec_time:
        f_in = net_weights @ net_c
        max_cycles = int((rec_time - time_passed) / dt)
        dt_tau = dt / tau
        sq_dt_tau = np.sqrt(dt_tau)
        noise = rng.normal(0, eta * sq_dt_tau, size=(N, max_cycles))
        lmat = (~np.eye(N, dtype=bool)).astype(float) * L

        x = np.zeros((N, 1))
        K_arr = np.ones((N, 1)) * K
        inds = np.arange(N)
        crossed = False
        winners = None
        i = 0

        while i < max_cycles and not crossed:
            x = x + (f_in - K_arr * x - lmat @ x) * dt_tau + noise[:, i:i + 1]
            x[x < 0] = 0.0
            reset = retrieved & (x >= thresholds)
            x[reset] = 0.95 * thresholds[reset]
            retrievable = ~retrieved
            if np.any(x[retrievable] >= thresholds[retrievable]):
                crossed = True
                mask = (x[retrievable] >= thresholds[retrievable]).flatten()
                cands = inds[retrievable.flatten()][mask]
                winners = np.array([rng.choice(cands)]) if len(cands) > 1 else cands
            i += 1

        time_passed += i * dt

        if crossed and winners is not None:
            winner = int(winners[0])
            sp1 = int(np.where(pres_indices - 1 == winner)[0][0]) + 1

            net_f[:] = 0.0
            net_f[winner] = 1.0
            net_c_in = net_w_fc @ net_f
            net_c_in /= _norm(net_c_in)

            dot = _dot(net_c, net_c_in)
            rho = np.sqrt(1.0 + B_rec**2 * (dot**2 - 1.0)) - B_rec * dot
            net_c = rho * net_c + B_rec * net_c_in

            net_w_fc += (net_c @ net_f.T) * lrate_fc_rec
            net_w_cf += (net_f @ net_c.T) * lrate_cf_rec

            recall_count += 1
            recalls[recall_count - 1] = sp1
            times[recall_count - 1] = time_passed
            retrieved[winner] = True

    return recalls, times, net_w_fc, net_w_cf


# =====================================================================
# Batch runner
# =====================================================================

def run_batch(
    B_encD: np.ndarray,
    n_sims: int = 1000,
    seed: int = 2026,
    label: str = "",
    **sim_kwargs,
) -> dict:
    """
    Run n_sims CMR trials with a fixed drift schedule.

    Returns
    -------
    dict with: label, B_encD, recall_sims, times_sims, net_w_fc, net_w_cf.
    """
    rng = np.random.default_rng(seed)
    B = np.asarray(B_encD, dtype=float)
    recall_sims = np.zeros((N, n_sims), dtype=int)
    times_sims = np.zeros((N, n_sims), dtype=float)
    wfc_last = None
    wcf_last = None

    for s in range(n_sims):
        rec, t, wfc, wcf = simulate_single_trial(B, rng, **sim_kwargs)
        recall_sims[:, s] = rec
        times_sims[:, s] = t
        wfc_last, wcf_last = wfc, wcf

    return {
        "label": label,
        "B_encD": B,
        "recall_sims": recall_sims,
        "times_sims": times_sims,
        "net_w_fc": wfc_last,
        "net_w_cf": wcf_last,
    }

"""
CMR B_enc Targeted Simulation — Simulation Functions

RPE computation, drift-schedule construction, per-trial noise
injection, and single-trial CMR runner.

Design
------
Three independent manipulations, composed *before* simulation:

1) **RPE gain** (additive, loss items only):
   B[i] = B_baseline[i] + g × RPE[i]
   where g is the gain parameter. Applied only at positions where
   `loss_mask` is True. (Brown et al., 2018)

2) **Per-trial drift noise** (loss items only):
   On each trial, Gaussian noise N(0, σ²) is added to positions
   where `noise_mask` is True, then clipped to [0, 1].
   (Pitts et al., 2022; Granger et al., 2025)

3) **Tonic reward-context scaling**:
   B[i] = s × B_baseline[i]     for reward positions
   where s < 1 represents a pessimistic/blunted reward encoding.
   (Nawijn et al., 2015)

All three are applied to the drift schedule *outside* the trial
runner, so `simulate_single_trial` stays general-purpose.
"""

from __future__ import annotations

import numpy as np

from .config import N, pres_indices, BASE_PARAMS, sem_mat


# =====================================================================
# RPE computation
# =====================================================================

def compute_rpe(reward_sequence: np.ndarray) -> np.ndarray:
    """
    Absolute reward prediction error at each study position.

        RPE[i] = |outcome[i+1] − outcome[i]|

    Parameters
    ----------
    reward_sequence : (N+1,) array
        Element 0 = prior expectation; elements 1..N = outcomes.

    Returns
    -------
    rpe : (N,) float array
    """
    return np.abs(np.diff(np.asarray(reward_sequence, dtype=float)))


# =====================================================================
# Drift schedule builders
# =====================================================================

def apply_rpe_gain(
    B_base: np.ndarray,
    rpe: np.ndarray,
    gain: float,
    loss_mask: np.ndarray,
) -> np.ndarray:
    """
    Additive RPE-gain modulation on loss-context items.

        B_out[i] = B_base[i] + gain × RPE[i]   (where loss_mask[i])
        B_out[i] = B_base[i]                   (elsewhere)

    Result is clipped to [0, 1].

    Parameters
    ----------
    B_base : (N,) array
        Baseline drift schedule.
    rpe : (N,) array
        Absolute prediction errors.
    gain : float
        Gain parameter g.
    loss_mask : (N,) bool array
        True at loss-context positions.

    Returns
    -------
    B_out : (N,) float array, clipped to [0, 1].
    """
    B = np.asarray(B_base, dtype=float).copy()
    rpe = np.asarray(rpe, dtype=float)
    B[loss_mask] += gain * rpe[loss_mask]
    return np.clip(B, 0.0, 1.0)


def apply_reward_tonic_scale(
    B_base: np.ndarray,
    scale: float,
    reward_mask: np.ndarray,
) -> np.ndarray:
    """
    Tonic multiplicative scaling on reward-context items.

        B_out[i] = scale × B_base[i]    (where reward_mask[i])
        B_out[i] = B_base[i]            (elsewhere)

    Result is clipped to [0, 1].

    Parameters
    ----------
    B_base : (N,) array
        Baseline drift schedule.
    scale : float
        Tonic scale factor s (< 1 = reduced reward drift).
    reward_mask : (N,) bool array
        True at reward-context positions.

    Returns
    -------
    B_out : (N,) float array, clipped to [0, 1].
    """
    B = np.asarray(B_base, dtype=float).copy()
    B[reward_mask] *= scale
    return np.clip(B, 0.0, 1.0)


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
    Run one encode → retrieve CMR trial with the given drift schedule.

    Parameters
    ----------
    B_encD : (N,) array
        Per-position encoding drift rates.
    rng : numpy RNG.
    gamma_fc, eta, B_rec : optional parameter overrides.

    Returns
    -------
    recalls : (N,) int array
        1-based serial positions (0-padded).
    times : (N,) float array
        Cumulative recall times (ms).
    net_w_fc : (N, N)
        Final M_FC.
    net_w_cf : (N, N)
        Final M_CF.
    """
    p = BASE_PARAMS
    gamma_fc = gamma_fc if gamma_fc is not None else p["gamma_fc"]
    eta      = eta      if eta      is not None else p["eta"]
    B_rec    = B_rec    if B_rec    is not None else p["B_rec"]

    eye_fc       = 1.0 - gamma_fc
    lrate_fc_enc = gamma_fc
    lrate_cf_enc = p["lrate_cf_enc"]
    lrate_fc_rec = p["lrate_fc_rec"]
    lrate_cf_rec = p["lrate_cf_rec"]
    thresh       = p["thresh"]
    rec_time     = p["rec_time"]
    dt           = p["dt"]
    tau          = p["tau"]
    K            = p["K"]
    L            = p["L"]
    episodic_w   = p["episodic_weight"]
    sem_w        = p["sem_weight"]

    B_encD = np.asarray(B_encD, dtype=float)

    # ── Initialisation ──
    net_f    = np.zeros((N, 1))
    net_c    = np.zeros((N, 1))
    net_w_fc = np.eye(N) * eye_fc
    net_w_cf = np.zeros((N, N))

    # ── Encoding ──
    for pos in range(N):
        feat = int(pres_indices[pos]) - 1
        net_f[:] = 0.0
        net_f[feat] = 1.0

        net_c_in = net_w_fc @ net_f
        net_c_in /= _norm(net_c_in)

        B   = float(B_encD[pos])
        dot = _dot(net_c, net_c_in)
        rho = np.sqrt(1.0 + B**2 * (dot**2 - 1.0)) - B * dot
        net_c = rho * net_c + B * net_c_in

        net_w_fc += (net_c @ net_f.T) * lrate_fc_enc
        net_w_cf += (net_f @ net_c.T) * lrate_cf_enc

    # ── Retrieval setup ──
    recalls    = np.zeros(N, dtype=int)
    times      = np.zeros(N, dtype=float)
    retrieved  = np.zeros((N, 1), dtype=bool)
    thresholds = np.ones((N, 1))
    net_weights = episodic_w * net_w_cf + sem_w * sem_mat

    time_passed  = 0.0
    recall_count = 0

    # ── Retrieval loop ──
    while time_passed < rec_time:
        f_in = net_weights @ net_c
        max_cycles = int((rec_time - time_passed) / dt)
        dt_tau     = dt / tau
        sq_dt_tau  = np.sqrt(dt_tau)
        noise      = rng.normal(0, eta * sq_dt_tau, size=(N, max_cycles))
        lmat       = (~np.eye(N, dtype=bool)).astype(float) * L

        x      = np.zeros((N, 1))
        K_arr  = np.ones((N, 1)) * K
        inds   = np.arange(N)
        crossed = False
        winners = None
        i = 0

        while i < max_cycles and not crossed:
            x = x + (f_in - K_arr * x - lmat @ x) * dt_tau + noise[:, i:i+1]
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

            # Map winner feature index -> serial position (1-based)
            sp1 = int(np.where(pres_indices - 1 == winner)[0][0]) + 1

            net_f[:] = 0.0
            net_f[winner] = 1.0

            net_c_in  = net_w_fc @ net_f
            net_c_in /= _norm(net_c_in)

            dot = _dot(net_c, net_c_in)
            rho = np.sqrt(1.0 + B_rec**2 * (dot**2 - 1.0)) - B_rec * dot
            net_c = rho * net_c + B_rec * net_c_in

            net_w_fc += (net_c @ net_f.T) * lrate_fc_rec
            net_w_cf += (net_f @ net_c.T) * lrate_cf_rec

            recall_count += 1
            recalls[recall_count - 1] = sp1
            times[recall_count - 1]   = time_passed
            retrieved[winner] = True

    return recalls, times, net_w_fc, net_w_cf


# =====================================================================
# Batch runner with per-trial noise injection
# =====================================================================

def run_batch(
    B_encD_base: np.ndarray,
    n_sims: int = 1000,
    seed: int = 2026,
    drift_noise_std: float = 0.0,
    noise_mask: np.ndarray | None = None,
    label: str = "",
    **sim_kwargs,
) -> dict:
    """
    Run `n_sims` CMR trials. On each trial, optionally injects
    fresh Gaussian noise into positions specified by `noise_mask`.

    Parameters
    ----------
    B_encD_base : (N,) array
        Deterministic drift schedule.
    n_sims : int
        Number of trials.
    seed : int
        RNG seed.
    drift_noise_std : float
        SD of per-trial noise (0 = deterministic).
    noise_mask : (N,) bool array
        Which positions receive noise. None → no noise regardless of drift_noise_std.
    label : str
        Condition name stored in output dict.
    **sim_kwargs
        Forwarded to simulate_single_trial.

    Returns
    -------
    dict with keys: label, B_encD_base, B_encD_trials, recall_sims,
    times_sims, net_w_fc, net_w_cf.
    """
    rng = np.random.default_rng(seed)
    B_base = np.asarray(B_encD_base, dtype=float)

    recall_sims = np.zeros((N, n_sims), dtype=int)
    times_sims  = np.zeros((N, n_sims), dtype=float)
    B_trials    = np.zeros((n_sims, N), dtype=float)

    wfc_last = None
    wcf_last = None

    for s in range(n_sims):
        B_trial = B_base.copy()

        if drift_noise_std > 0 and noise_mask is not None:
            perturbation = rng.normal(0, drift_noise_std, size=N)
            B_trial[noise_mask] += perturbation[noise_mask]
            B_trial = np.clip(B_trial, 0.0, 1.0)

        B_trials[s] = B_trial

        rec, t, wfc, wcf = simulate_single_trial(B_trial, rng, **sim_kwargs)
        recall_sims[:, s] = rec
        times_sims[:, s]  = t
        wfc_last, wcf_last = wfc, wcf

    return {
        "label":         label,
        "B_encD_base":   B_base,
        "B_encD_trials": B_trials,
        "recall_sims":   recall_sims,
        "times_sims":    times_sims,
        "net_w_fc":      wfc_last,
        "net_w_cf":      wcf_last,
    }

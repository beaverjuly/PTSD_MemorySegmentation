"""
CMR B_enc Simulation — Simulation Functions
=============================================
Single-trial CMR runner, RPE computation, and drift-schedule modulation.

Overview
--------
The core function ``simulate_single_trial`` runs one full
encode → retrieve cycle of the CMR model.  It accepts a **pre-computed**
``B_encD`` drift schedule, so the caller controls whether encoding is
baseline, globally scaled, or RPE-modulated.

Two helper functions build the RPE-modulated schedule:

    compute_rpe(reward_sequence)
        → absolute prediction errors, one per study position

    modulate_drift_by_rpe(B_encD_base, rpe, rpe_gain, method)
        → new B_encD array with position-specific RPE scaling

The notebook uses these helpers to construct the drift schedule
*before* calling the simulation, keeping the simulation itself
simple and general.
"""

from __future__ import annotations

import numpy as np

from .config import (
    N, pres_indices, BASE_PARAMS, sem_mat,
)


# =====================================================================
# RPE computation
# =====================================================================

def compute_rpe(reward_sequence: np.ndarray) -> np.ndarray:
    """
    Compute the absolute reward prediction error (RPE) at each study
    position from a sequence of observed outcomes.

    The RPE at position *i* is defined as the unsigned difference
    between consecutive outcomes::

        RPE[i] = |outcome[i+1] − outcome[i]|

    The first entry ``outcome[0]`` is the learner's prior expectation
    *before* any item is studied, so ``RPE[0]`` captures the surprise
    of the very first item.

    Parameters
    ----------
    reward_sequence : (N+1,) array
        ``reward_sequence[0]`` = prior expectation;
        ``reward_sequence[1..N]`` = outcomes after each study item.

    Returns
    -------
    rpe : (N,) float array
        Absolute prediction errors, one per study position.
    """
    reward_sequence = np.asarray(reward_sequence, dtype=float)
    rpe = np.abs(np.diff(reward_sequence))           # length = N
    return rpe


def modulate_drift_by_rpe(
    B_encD_base: np.ndarray,
    rpe: np.ndarray,
    rpe_gain: float = 0.4,
    method: str = "multiplicative",
) -> np.ndarray:
    """
    Build an RPE-modulated encoding drift schedule.

    Starts from a baseline schedule ``B_encD_base`` and scales each
    position's drift rate according to the local RPE.

    Parameters
    ----------
    B_encD_base : (N,) array
        Baseline per-position drift rates (e.g. ``B_encD_baseline``
        from config).
    rpe : (N,) array
        Absolute prediction errors from ``compute_rpe``.
    rpe_gain : float
        Scaling constant that controls how strongly RPE modulates
        the drift rate.  Larger values → stronger modulation.
    method : ``"multiplicative"`` | ``"additive"``
        How RPE enters the drift computation:

        * **multiplicative** (default):
          ``B_eff[i] = B_base[i] × (1 + rpe_gain × |RPE_z[i]|)``
          where ``RPE_z`` is the z-scored RPE vector.
          Positions with above-average surprise get a *proportional*
          boost.

        * **additive**:
          ``B_eff[i] = B_base[i] + rpe_gain × |RPE_z[i]|``
          Adds a flat increment regardless of the base rate.

    Returns
    -------
    B_encD_mod : (N,) float array
        Modulated drift schedule, clipped to [0, 1].
    """
    rpe = np.asarray(rpe, dtype=float)
    B_base = np.asarray(B_encD_base, dtype=float).copy()

    # z-score the RPE so the gain parameter has a consistent scale
    rpe_std = np.std(rpe)
    if rpe_std > 1e-12:
        rpe_z = np.abs((rpe - np.mean(rpe)) / rpe_std)
    else:
        # all RPEs identical → no modulation
        rpe_z = np.zeros_like(rpe)

    if method == "multiplicative":
        B_mod = B_base * (1.0 + rpe_gain * rpe_z)
    elif method == "additive":
        B_mod = B_base + rpe_gain * rpe_z
    else:
        raise ValueError(f"Unknown method {method!r}; use 'multiplicative' or 'additive'.")

    return np.clip(B_mod, 0.0, 1.0)


# =====================================================================
# Vectorial helpers
# =====================================================================

def _dot(a, b):
    """Scalar dot product for (N,1) column vectors."""
    return float(np.asarray(a.T @ b).ravel()[0])


def _norm(v):
    """L2 norm for (N,1) column vectors."""
    return float(np.linalg.norm(v))


# =====================================================================
# Single-trial CMR simulation
# =====================================================================

def simulate_single_trial(
    B_encD: np.ndarray,
    rng: np.random.Generator,
    gamma_fc: float | None = None,
    eta: float | None = None,
    B_rec: float | None = None,
):
    """
    Run one encoding → retrieval trial of the CMR model.

    This function is intentionally **parameter-explicit**: the caller
    passes in the full ``B_encD`` drift schedule, so it works
    identically for baseline, globally-scaled, and RPE-modulated
    conditions — no internal branching.

    Parameters
    ----------
    B_encD : (N,) array
        Per-position encoding drift rates.  This is the *only*
        encoding-related input; all modulation (global scaling,
        RPE, etc.) must be applied before calling this function.
    rng : numpy.random.Generator
        Random number generator (for accumulator noise).
    gamma_fc : float, optional
        Pre-experimental feature→context weight.
        Default: ``BASE_PARAMS["gamma_fc"]``.
    eta : float, optional
        Accumulator noise SD.
        Default: ``BASE_PARAMS["eta"]``.
    B_rec : float, optional
        Context drift rate during retrieval.
        Default: ``BASE_PARAMS["B_rec"]``.

    Returns
    -------
    recalls : (N,) int array
        1-based serial positions of recalled items (0-padded).
    times : (N,) float array
        Cumulative recall times in ms.
    net_w_fc : (N, N) array
        Final feature→context weight matrix after encoding.
    net_w_cf : (N, N) array
        Final context→feature weight matrix after encoding.
    """
    # ── Resolve defaults ──
    p = BASE_PARAMS
    if gamma_fc is None:
        gamma_fc = p["gamma_fc"]
    if eta is None:
        eta = p["eta"]
    if B_rec is None:
        B_rec = p["B_rec"]

    eye_fc = 1.0 - gamma_fc          # pre-experimental M_FC diagonal
    lrate_fc_enc = gamma_fc           # Hebbian learning rate for M_FC
    lrate_cf_enc = p["lrate_cf_enc"]  # learning rate for M_CF
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

    # ════════════════════ INITIALISATION ════════════════════
    net_f = np.zeros((N, 1))
    net_c = np.zeros((N, 1))
    net_w_fc = np.eye(N) * eye_fc
    net_w_cf = np.eye(N) * 0.0       # eye_cf = 0 in original

    # ════════════════════ ENCODING ════════════════════
    for pos in range(N):
        feature_idx = int(pres_indices[pos]) - 1   # 1-based → 0-based

        # activate the item's feature unit
        net_f[:] = 0.0
        net_f[feature_idx] = 1.0

        # context input via M_FC
        net_c_in = net_w_fc @ net_f
        net_c_in = net_c_in / _norm(net_c_in)

        # context drift (Equation 3 in Polyn et al., 2009)
        B = float(B_encD[pos])
        dot = _dot(net_c, net_c_in)
        rho = np.sqrt(1.0 + B**2 * (dot**2 - 1.0)) - B * dot
        net_c = rho * net_c + B * net_c_in

        # Hebbian weight updates
        net_w_fc += (net_c @ net_f.T) * lrate_fc_enc
        net_w_cf += (net_f @ net_c.T) * lrate_cf_enc

    # ════════════════════ RETRIEVAL SETUP ════════════════════
    recalls = np.zeros(N, dtype=int)
    times = np.zeros(N, dtype=float)
    retrieved = np.zeros((N, 1), dtype=bool)
    thresholds = np.ones((N, 1))
    net_weights = episodic_w * net_w_cf + sem_w * sem_mat

    time_passed = 0.0
    recall_count = 0

    # ════════════════════ RETRIEVAL LOOP ════════════════════
    while time_passed < rec_time:

        # evidence vector: M_CF (combined) × current context
        f_in = net_weights @ net_c

        max_cycles = int((rec_time - time_passed) / dt)
        dt_tau = dt / tau
        sq_dt_tau = np.sqrt(dt_tau)

        # accumulator noise matrix (pre-drawn for speed)
        noise = rng.normal(0, eta * sq_dt_tau, size=(N, max_cycles))
        lmat = (~np.eye(N, dtype=bool)).astype(float) * L

        x = np.zeros((N, 1))
        K_arr = np.ones((N, 1)) * K
        inds = np.arange(N)

        crossed = False
        winners = None
        i = 0

        while i < max_cycles and not crossed:
            lx = lmat @ x               # lateral inhibition
            kx = K_arr * x              # self-decay
            x = x + (f_in - kx - lx) * dt_tau + noise[:, i:i+1]
            x[x < 0] = 0.0

            # prevent already-retrieved items from re-accumulating
            reset = retrieved & (x >= thresholds)
            x[reset] = 0.95 * thresholds[reset]

            # check for threshold crossing among retrievable items
            retrievable = ~retrieved
            if np.any(x[retrievable] >= thresholds[retrievable]):
                crossed = True
                mask = (x[retrievable] >= thresholds[retrievable]).flatten()
                candidates = inds[retrievable.flatten()][mask]
                winners = np.array([rng.choice(candidates)]) if len(candidates) > 1 else candidates
            i += 1

        time_passed += i * dt

        # ── process successful recall ──
        if crossed and winners is not None:
            winner = int(winners[0])
            serial_pos1 = int(np.where(pres_indices - 1 == winner)[0][0]) + 1

            # reactivate recalled feature
            net_f[:] = 0.0
            net_f[winner] = 1.0

            # context reinstatement via M_FC
            net_c_in = net_w_fc @ net_f
            net_c_in = net_c_in / _norm(net_c_in)

            # retrieval context update
            dot = _dot(net_c, net_c_in)
            rho = np.sqrt(1.0 + B_rec**2 * (dot**2 - 1.0)) - B_rec * dot
            net_c = rho * net_c + B_rec * net_c_in

            # retrieval-phase weight updates (both zero by default)
            net_w_fc += (net_c @ net_f.T) * lrate_fc_rec
            net_w_cf += (net_f @ net_c.T) * lrate_cf_rec

            # record
            recall_count += 1
            recalls[recall_count - 1] = serial_pos1
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
    gamma_fc: float | None = None,
    eta: float | None = None,
    B_rec: float | None = None,
    label: str = "",
) -> dict:
    """
    Run ``n_sims`` independent CMR trials with a given drift schedule
    and collect the results into a single dictionary.

    Parameters
    ----------
    B_encD : (N,) array
        Per-position encoding drift rates.
    n_sims : int
        Number of Monte-Carlo simulated trials.
    seed : int
        Random seed for reproducibility.
    gamma_fc, eta, B_rec : float, optional
        Override base parameters (see ``simulate_single_trial``).
    label : str
        Human-readable condition label (stored in output dict).

    Returns
    -------
    result : dict with keys
        ``"label"``         — condition name
        ``"B_encD"``        — the drift schedule used
        ``"recall_sims"``   — (N, n_sims) int array of recalled positions
        ``"times_sims"``    — (N, n_sims) float array of recall times
        ``"net_w_fc"``      — (N, N) final M_FC (same across trials)
        ``"net_w_cf"``      — (N, N) final M_CF (same across trials)
    """
    rng = np.random.default_rng(seed)
    recall_sims = np.zeros((N, n_sims), dtype=int)
    times_sims = np.zeros((N, n_sims), dtype=float)
    net_w_fc_last = net_w_cf_last = None

    for s in range(n_sims):
        recalls, times, wfc, wcf = simulate_single_trial(
            B_encD, rng,
            gamma_fc=gamma_fc, eta=eta, B_rec=B_rec,
        )
        recall_sims[:, s] = recalls
        times_sims[:, s] = times
        net_w_fc_last, net_w_cf_last = wfc, wcf

    return {
        "label": label,
        "B_encD": np.asarray(B_encD, dtype=float),
        "recall_sims": recall_sims,
        "times_sims": times_sims,
        "net_w_fc": net_w_fc_last,
        "net_w_cf": net_w_cf_last,
    }

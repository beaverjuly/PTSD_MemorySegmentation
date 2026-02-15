"""
CMR Multi-Parameter Sweep — Simulation Functions
=================================================
Single-trial CMR runner and batch wrappers (with/without diagnostic traces).
"""

import numpy as np
from collections import defaultdict

from .config import (
    N, pres_indices, BASE_PARAMS, B_encD,
    episodic_weight, sem_weight, sem_mat,
    lrate_cf_enc, lrate_fc_rec, lrate_cf_rec,
    thresh, rec_time, dt, tau, K, L,
)


# ─────────────────────────────────────────────────────────────────────
# Single-trial runner
# ─────────────────────────────────────────────────────────────────────

def simulate_single_trial(
    B_rec,
    rng,
    gamma_fc_val=None,
    eta_val=None,
    B_encD_scale=1.0,
    record_diagnostics=False,
    recency_k=3,
):
    """
    Simulate one encoding–retrieval trial (CMR-style).

    Parameters
    ----------
    B_rec : float
        Context drift rate during retrieval.
    rng : numpy.random.Generator
        Random number generator.
    gamma_fc_val : float, optional
        Pre-existing association strength (if None, uses BASE_PARAMS["gamma_fc"]).
    eta_val : float, optional
        Accumulator noise (if None, uses BASE_PARAMS["eta"]).
    B_encD_scale : float, optional
        Multiplier for encoding drift schedule B_encD (per-serial-position drift).
    record_diagnostics : bool
        If True, record cue advantage diagnostics AND internal retrieval traces.
    recency_k : int
        How many final serial positions count as "recency" for evidence mass diagnostic.

    Returns
    -------
    recalls : (N,) array
        Serial positions recalled in order (0-filled past recall_count).
    times : (N,) array
        Recall times for each output position (0-filled past recall_count).
    net_w_fc : (N,N) array
        Final item->context weights.
    net_w_cf : (N,N) array
        Final context->item weights.
    diagnostics : dict or None
        If record_diagnostics=True, returns dict with:
          - "cue": neighbor cue advantage diagnostics
          - "trace": internal traces (rad, dot, rho, c_norm, cos_after, evidence summaries,
                     and mean evidence by serial position: f_mean_by_pos, f_mean_centered_by_pos)
        Otherwise None.
    """

    # ---- params ----
    if gamma_fc_val is None:
        gamma_fc_val = BASE_PARAMS["gamma_fc"]
    if eta_val is None:
        eta_val = BASE_PARAMS["eta"]

    # derived params (recomputed if gamma_fc changes)
    eye_fc_local = 1.0 - gamma_fc_val
    eye_cf_local = 0.0
    lrate_fc_enc_local = gamma_fc_val

    # scale encoding drift schedule
    B_encD_local = B_encD * B_encD_scale

    # ================== INITIALIZATION ==================
    net_f = np.zeros((N, 1))
    net_c = np.zeros((N, 1))

    # initialize weights with pre-existing associations
    net_w_fc = np.eye(N) * eye_fc_local
    net_w_cf = np.eye(N) * eye_cf_local

    # ================== ENCODING PHASE ==================
    for item_idx in range(N):
        feature_idx = int(pres_indices[item_idx] - 1)

        net_f[:] = 0
        net_f[feature_idx] = 1

        # Context input through M_FC
        net_c_in = net_w_fc @ net_f
        net_c_in = net_c_in / float(np.sqrt(net_c_in.T @ net_c_in))

        # Context update (encoding drift schedule)
        c_in, c = net_c_in, net_c
        B = float(B_encD_local[item_idx])
        dot = float(c.T @ c_in)
        rad = 1.0 + (B**2) * ((dot**2) - 1.0)
        rho = np.sqrt(rad) - B * dot

        net_c = rho * c + B * c_in

        # Weight updates
        net_w_fc += (net_c @ net_f.T) * lrate_fc_enc_local
        net_w_cf += (net_f @ net_c.T) * lrate_cf_enc

    # ================== RETRIEVAL SETUP ==================
    recalls = np.zeros((N, 1))
    times = np.zeros((N, 1))

    retrieved = np.zeros((N, 1), dtype=bool)
    thresholds = np.ones((N, 1))

    net_weights = episodic_weight * net_w_cf + sem_weight * sem_mat

    time_passed = 0.0
    recall_count = 0

    # ---- diagnostics containers ----
    diagnostics = None
    cue_diag = None
    trace = None

    if record_diagnostics:
        # cue advantage diagnostics
        deltas_all = []
        deltas_by_pos = defaultdict(list)
        delta_forward = []
        delta_backward = []
        pending = None

        # internal traces (per successful recall update + per retrieval attempt)
        trace = {
            # per successful recall update
            "dot": [],
            "rad": [],
            "rho": [],
            "c_norm": [],
            "cos_after": [],

            # per retrieval attempt (before accumulator)
            "f_max": [],
            "f_entropy": [],
            "f_recency_mass": [],

            # mean evidence by SERIAL POSITION across retrieval attempts (per trial)
            "f_mean_by_pos": None,
            "f_mean_centered_by_pos": None,
        }

        # accumulate evidence-by-position across retrieval attempts in this trial
        f_sum_by_pos = np.zeros(N, dtype=float)
        f_count = 0

    # ================== RETRIEVAL LOOP ==================
    while time_passed < rec_time:

        # evidence vector for accumulator
        f_in = net_weights @ net_c  # (N,1)

        # ---- log evidence summaries BEFORE accumulator ----
        if record_diagnostics:
            f = f_in.flatten().astype(float)

            # stable softmax
            f_shift = f - np.nanmax(f)
            expf = np.exp(f_shift)
            Z = np.nansum(expf)
            if not np.isfinite(Z) or Z <= 0:
                p = np.full_like(f, np.nan)
            else:
                p = expf / Z

            # recency mass in SERIAL POSITION space (last recency_k serial positions)
            k = int(recency_k)
            k = max(1, min(N, k))
            rec_pos0 = np.arange(N - k, N)  # 0-indexed serial positions

            # map serial position -> item index (feature index)
            rec_item_idx = [int(pres_indices[pos0] - 1) for pos0 in rec_pos0]
            rec_mass = np.nansum(p[rec_item_idx])

            # entropy and max
            ent = -np.nansum(p * np.log(p + 1e-12))
            fmax = np.nanmax(f)

            trace["f_recency_mass"].append(float(rec_mass))
            trace["f_entropy"].append(float(ent))
            trace["f_max"].append(float(fmax))

            # evidence-by-serial-position (raw, not softmax)
            f_pos = np.empty(N, dtype=float)
            for pos0 in range(N):
                item_idx = int(pres_indices[pos0] - 1)
                f_pos[pos0] = f[item_idx]
            f_sum_by_pos += f_pos
            f_count += 1

        # accumulator setup
        max_cycles = int((rec_time - time_passed) / dt)
        dt_tau = dt / tau
        sq_dt_tau = np.sqrt(dt_tau)

        # pre-generate noise
        noise = rng.normal(0, eta_val * sq_dt_tau, size=(N, max_cycles))

        eyeI = ~np.eye(N, dtype=bool)
        lmat = eyeI.astype(float) * L

        x = np.zeros((N, 1))
        K_array = np.ones((N, 1)) * K
        inds = np.arange(N)

        crossed = 0
        i = 0

        while i < max_cycles and crossed == 0:

            lx = lmat @ x
            kx = K_array * x

            x = x + ((f_in - kx - lx) * dt_tau + noise[:, i:i+1])
            x[x < 0] = 0

            reset_these = retrieved & (x >= thresholds)
            x[reset_these] = 0.95 * thresholds[reset_these]

            retrievable = ~retrieved

            if np.any(x[retrievable] >= thresholds[retrievable]):
                crossed = 1
                temp_win = x[retrievable] >= thresholds[retrievable]
                temp_ind = inds[retrievable.flatten()]
                winners = temp_ind[temp_win.flatten()]

                if len(winners) > 1:
                    winners = np.array([rng.choice(winners)])

            i += 1

        time_passed += i * dt

        # ================== ON SUCCESSFUL RECALL ==================
        if crossed == 1:
            winner = int(winners[0])

            # map feature index -> serial position in the presented order
            serial_pos0 = int(np.where(pres_indices - 1 == winner)[0][0])
            serial_pos1 = serial_pos0 + 1  # 1-indexed

            # cue-transition bookkeeping (uses previous step's delta_f)
            if record_diagnostics and pending is not None:
                prev_pos0 = pending["serial_pos0"]
                delta_f_prev = pending["delta_f"]
                transition = serial_pos0 - prev_pos0

                if transition == 1:
                    delta_forward.append(delta_f_prev)
                elif transition == -1:
                    delta_backward.append(delta_f_prev)

                pending = None

            # Reactivate recalled item feature
            net_f[:] = 0
            net_f[winner] = 1

            # Context input
            net_c_in = net_w_fc @ net_f
            net_c_in = net_c_in / float(np.sqrt(net_c_in.T @ net_c_in))

            # Retrieval context update
            c_in, c = net_c_in, net_c
            dot = float(c.T @ c_in)
            rad = 1.0 + (B_rec**2) * ((dot**2) - 1.0)
            rho = np.sqrt(rad) - B_rec * dot  # becomes NaN if rad < 0

            net_c = rho * c + B_rec * c_in

            # log post-update stability quantities
            if record_diagnostics:
                trace["dot"].append(float(dot))
                trace["rad"].append(float(rad))
                trace["rho"].append(float(rho) if np.isfinite(rho) else np.nan)

                c_norm = float(np.sqrt(net_c.T @ net_c)) if np.all(np.isfinite(net_c)) else np.nan
                trace["c_norm"].append(c_norm)

                # cosine similarity after update (how "locked" c is to c_in)
                if np.isfinite(c_norm) and c_norm > 0 and np.all(np.isfinite(c_in)):
                    cin_norm = float(np.sqrt(c_in.T @ c_in))
                    cos_after = float((net_c.T @ c_in) / (c_norm * cin_norm))
                else:
                    cos_after = np.nan
                trace["cos_after"].append(cos_after)

            # weight updates during retrieval
            net_w_fc += (net_c @ net_f.T) * lrate_fc_rec
            net_w_cf += (net_f @ net_c.T) * lrate_cf_rec

            # record recall
            recall_count += 1
            recalls[recall_count - 1, 0] = serial_pos1
            times[recall_count - 1, 0] = time_passed

            # Cue advantage diagnostics (neighbor advantage)
            if record_diagnostics:
                left_pos0 = serial_pos0 - 1
                right_pos0 = serial_pos0 + 1

                if 0 <= left_pos0 < N and 0 <= right_pos0 < N:
                    left_item = int(pres_indices[left_pos0] - 1)
                    right_item = int(pres_indices[right_pos0] - 1)

                    if (not retrieved[left_item]) and (not retrieved[right_item]):
                        f_after = (net_weights @ net_c).flatten()
                        delta_f = float(f_after[right_item] - f_after[left_item])

                        deltas_all.append(delta_f)
                        deltas_by_pos[serial_pos1].append(delta_f)
                        pending = {"serial_pos0": serial_pos0, "delta_f": delta_f}

            retrieved[winner] = True

    # ================== per-trial DIAGNOSTICS ==================
    if record_diagnostics:
        cue_diag = {
            "deltas_all": deltas_all,
            "deltas_by_pos": dict(deltas_by_pos),
            "delta_forward": delta_forward,
            "delta_backward": delta_backward,
        }

        # per-trial mean evidence by serial position across retrieval attempts
        if f_count > 0:
            f_mean_by_pos = f_sum_by_pos / f_count
            f_mean_centered_by_pos = f_mean_by_pos - np.mean(f_mean_by_pos)
        else:
            f_mean_by_pos = np.full(N, np.nan, dtype=float)
            f_mean_centered_by_pos = np.full(N, np.nan, dtype=float)

        trace["f_mean_by_pos"] = f_mean_by_pos
        trace["f_mean_centered_by_pos"] = f_mean_centered_by_pos

        diagnostics = {
            "cue": cue_diag,
            "trace": trace
        }

    return recalls.flatten(), times.flatten(), net_w_fc, net_w_cf, diagnostics


# ─────────────────────────────────────────────────────────────────────
# Batch runner (no traces)
# ─────────────────────────────────────────────────────────────────────

def run_simulation(B_rec, n_sims=500, seed=2026, gamma_fc_val=None, eta_val=None, B_encD_scale=1.0):
    """
    Run multiple independent trials.

    Parameters
    ----------
    B_rec : float
        Retrieval context drift
    n_sims : int
        Number of simulations
    seed : int
        Random seed
    gamma_fc_val : float, optional
        Pre-existing association strength
    eta_val : float, optional
        Accumulator noise
    B_encD_scale : float, optional
        Encoding drift multiplier

    Returns
    -------
    recall_sims, times_sims, net_w_fc, net_w_cf
    """
    rng = np.random.default_rng(seed)

    recall_sims = np.zeros((N, n_sims), dtype=int)
    times_sims = np.zeros((N, n_sims), dtype=float)

    net_w_fc_last = None
    net_w_cf_last = None

    for s in range(n_sims):
        recalls, times, net_w_fc, net_w_cf, _ = simulate_single_trial(
            B_rec=B_rec,
            rng=rng,
            gamma_fc_val=gamma_fc_val,
            eta_val=eta_val,
            B_encD_scale=B_encD_scale,
            record_diagnostics=False
        )

        recall_sims[:, s] = recalls.astype(int)
        times_sims[:, s] = times

        net_w_fc_last = net_w_fc
        net_w_cf_last = net_w_cf

    return recall_sims, times_sims, net_w_fc_last, net_w_cf_last


# ─────────────────────────────────────────────────────────────────────
# Batch runner with per-trial diagnostic traces
# ─────────────────────────────────────────────────────────────────────

def run_simulation_with_traces(
    B_rec, n_sims=500, seed=2026,
    gamma_fc_val=None, eta_val=None, B_encD_scale=1.0,
    recency_k=3,
):
    """
    Run multiple independent trials and store retrieval stability traces.

    Returns
    -------
    recall_sims, times_sims, net_w_fc_last, net_w_cf_last, trace_sims
      trace_sims is a list of length n_sims, each element is diagnostics["trace"].
    """
    rng = np.random.default_rng(seed)

    recall_sims = np.zeros((N, n_sims), dtype=int)
    times_sims = np.zeros((N, n_sims), dtype=float)

    net_w_fc_last = None
    net_w_cf_last = None
    trace_sims = []

    for s in range(n_sims):
        recalls, times, net_w_fc, net_w_cf, diag = simulate_single_trial(
            B_rec=B_rec,
            rng=rng,
            gamma_fc_val=gamma_fc_val,
            eta_val=eta_val,
            B_encD_scale=B_encD_scale,
            record_diagnostics=True,
            recency_k=recency_k
        )

        recall_sims[:, s] = recalls.astype(int)
        times_sims[:, s] = times

        net_w_fc_last = net_w_fc
        net_w_cf_last = net_w_cf

        # store trace (could be None if something unexpected happens)
        trace_sims.append(diag["trace"] if (diag is not None and "trace" in diag) else None)

    return recall_sims, times_sims, net_w_fc_last, net_w_cf_last, trace_sims

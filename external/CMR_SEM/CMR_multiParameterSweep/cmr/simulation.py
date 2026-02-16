"""
CMR Multi-Parameter Sweep — Simulation Functions
=================================================
Single-trial CMR runner and batch wrappers (with/without diagnostic traces).

The simulation is kept "dumb": it generates raw outputs and optional per-trial
diagnostic dicts.  All aggregation / analysis lives in metrics.py and the
diagnostics modules.
"""

import numpy as np
from collections import defaultdict

from .config import (
    N, pres_indices, BASE_PARAMS, B_encD,
    episodic_weight, sem_weight, sem_mat,
    lrate_cf_enc, lrate_fc_rec, lrate_cf_rec,
    thresh, rec_time, dt, tau, K, L,
)


def _dot(a, b):
    """Safe scalar dot product for (N,1) column vectors."""
    return float(np.asarray(a.T @ b).ravel()[0])


def _norm(v):
    """Safe L2 norm for (N,1) column vectors."""
    return float(np.linalg.norm(v))


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
    Simulate one encoding–retrieval trial of the Context Maintenance and
    Retrieval (CMR) model.

    Parameters
    ----------
    B_rec : float
        Context drift rate during retrieval (controls how strongly the
        retrieved item's context input replaces the current context).
    rng : numpy.random.Generator
        Random number generator.
    gamma_fc_val : float, optional
        Pre-experimental feature-to-context association strength.
        If None, falls back to BASE_PARAMS["gamma_fc"].
    eta_val : float, optional
        Accumulator noise standard deviation.
        If None, falls back to BASE_PARAMS["eta"].
    B_encD_scale : float, optional
        Global multiplier applied to the per-position encoding drift
        schedule B_encD.
    record_diagnostics : bool
        If True, collect two families of per-trial diagnostics:

        * **item_evidence** — neighbor item-evidence asymmetry during
          selection.  For each recall where both serial-position neighbors
          are still available, records
          Δf = f_in(i+1) − f_in(i−1)  where  f_in = M_CF · c.

        * **fc_alignment** — neighbor context-to-item-input alignment
          (Δ_FC) asymmetry during updating.  Records
          Δ_FC = cos(c, c_in(i+1)) − cos(c, c_in(i−1))
          where  c_in(j) = M_FC · f_j / ‖M_FC · f_j‖.

        * **trace** — per-recall-step internal quantities: radicand,
          dot product, rho, context norm, cosine similarity, and
          per-retrieval-attempt evidence summaries.
    recency_k : int
        How many final serial positions count as "recency" for the
        evidence-mass diagnostic.

    Returns
    -------
    recalls : (N,) array
        Serial positions recalled in order (0-padded beyond recall_count).
    times : (N,) array
        Cumulative recall times (ms) for each output position.
    net_w_fc : (N, N) array
        Final feature-to-context weight matrix M_FC.
    net_w_cf : (N, N) array
        Final context-to-feature weight matrix M_CF.
    diagnostics : dict or None
        If record_diagnostics is True, a dict with keys
        ``"item_evidence"``, ``"fc_alignment"``, and ``"trace"``.
    """

    # ---- params ----
    if gamma_fc_val is None:
        gamma_fc_val = BASE_PARAMS["gamma_fc"]
    if eta_val is None:
        eta_val = BASE_PARAMS["eta"]

    eye_fc_local = 1.0 - gamma_fc_val
    eye_cf_local = 0.0
    lrate_fc_enc_local = gamma_fc_val

    B_encD_local = B_encD * B_encD_scale

    # ================== INITIALIZATION ==================
    net_f = np.zeros((N, 1))
    net_c = np.zeros((N, 1))

    net_w_fc = np.eye(N) * eye_fc_local
    net_w_cf = np.eye(N) * eye_cf_local

    # ================== ENCODING PHASE ==================
    for item_idx in range(N):
        feature_idx = int(pres_indices[item_idx] - 1)

        net_f[:] = 0
        net_f[feature_idx] = 1

        net_c_in = net_w_fc @ net_f
        net_c_in = net_c_in / _norm(net_c_in)

        c_in, c = net_c_in, net_c
        B = float(B_encD_local[item_idx])
        dot = _dot(c, c_in)
        rad = 1.0 + (B**2) * ((dot**2) - 1.0)
        rho = np.sqrt(rad) - B * dot

        net_c = rho * c + B * c_in

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

    if record_diagnostics:
        # --- item-evidence asymmetry during selection (Δf) ---
        ie_deltas_all = []
        ie_deltas_by_pos = defaultdict(list)
        ie_delta_forward = []
        ie_delta_backward = []
        ie_pending = None

        # --- FC alignment asymmetry during context updating (Δ_FC) ---
        fc_deltas_all = []
        fc_deltas_by_pos = defaultdict(list)
        fc_delta_forward = []
        fc_delta_backward = []
        fc_pending = None

        # --- internal traces ---
        trace = {
            "dot": [], "rad": [], "rho": [],
            "c_norm": [], "cos_after": [],
            "f_max": [], "f_entropy": [], "f_recency_mass": [],
            "f_mean_by_pos": None,
            "f_mean_centered_by_pos": None,
        }
        f_sum_by_pos = np.zeros(N, dtype=float)
        f_count = 0

    # ================== RETRIEVAL LOOP ==================
    while time_passed < rec_time:

        f_in = net_weights @ net_c  # evidence vector (N,1)

        # ---- per-attempt evidence summaries ----
        if record_diagnostics:
            f = f_in.flatten().astype(float)

            f_shift = f - np.nanmax(f)
            expf = np.exp(f_shift)
            Z = np.nansum(expf)
            p = (expf / Z) if (np.isfinite(Z) and Z > 0) else np.full_like(f, np.nan)

            k = max(1, min(N, int(recency_k)))
            rec_pos0 = np.arange(N - k, N)
            rec_item_idx = [int(pres_indices[pos0] - 1) for pos0 in rec_pos0]
            rec_mass = np.nansum(p[rec_item_idx])

            trace["f_recency_mass"].append(float(rec_mass))
            trace["f_entropy"].append(float(-np.nansum(p * np.log(p + 1e-12))))
            trace["f_max"].append(float(np.nanmax(f)))

            f_pos = np.empty(N, dtype=float)
            for pos0 in range(N):
                f_pos[pos0] = f[int(pres_indices[pos0] - 1)]
            f_sum_by_pos += f_pos
            f_count += 1

        # ---- accumulator race ----
        max_cycles = int((rec_time - time_passed) / dt)
        dt_tau = dt / tau
        sq_dt_tau = np.sqrt(dt_tau)

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
            serial_pos0 = int(np.where(pres_indices - 1 == winner)[0][0])
            serial_pos1 = serial_pos0 + 1

            # -- resolve pending transition labels from previous step --
            if record_diagnostics:
                transition = serial_pos0 - (ie_pending["serial_pos0"] if ie_pending else -999)
                if ie_pending is not None:
                    if transition == 1:
                        ie_delta_forward.append(ie_pending["delta"])
                    elif transition == -1:
                        ie_delta_backward.append(ie_pending["delta"])
                    ie_pending = None
                if fc_pending is not None:
                    if transition == 1:
                        fc_delta_forward.append(fc_pending["delta"])
                    elif transition == -1:
                        fc_delta_backward.append(fc_pending["delta"])
                    fc_pending = None

            # reactivate recalled item
            net_f[:] = 0
            net_f[winner] = 1

            # context input through M_FC
            net_c_in = net_w_fc @ net_f
            net_c_in = net_c_in / _norm(net_c_in)

            # retrieval context update
            c_in, c = net_c_in, net_c
            dot_val = _dot(c, c_in)
            rad_val = 1.0 + (B_rec**2) * ((dot_val**2) - 1.0)
            rho_val = np.sqrt(rad_val) - B_rec * dot_val

            net_c = rho_val * c + B_rec * c_in

            # ---- trace logging ----
            if record_diagnostics:
                trace["dot"].append(float(dot_val))
                trace["rad"].append(float(rad_val))
                trace["rho"].append(float(rho_val) if np.isfinite(rho_val) else np.nan)
                c_norm_val = _norm(net_c) if np.all(np.isfinite(net_c)) else np.nan
                trace["c_norm"].append(c_norm_val)
                if np.isfinite(c_norm_val) and c_norm_val > 0 and np.all(np.isfinite(c_in)):
                    cin_norm = _norm(c_in)
                    trace["cos_after"].append(_dot(net_c, c_in) / (c_norm_val * cin_norm))
                else:
                    trace["cos_after"].append(np.nan)

            # weight updates during retrieval (both lr == 0 by default)
            net_w_fc += (net_c @ net_f.T) * lrate_fc_rec
            net_w_cf += (net_f @ net_c.T) * lrate_cf_rec

            # record recall
            recall_count += 1
            recalls[recall_count - 1, 0] = serial_pos1
            times[recall_count - 1, 0] = time_passed

            # ---- neighbor diagnostics (both-neighbors-available) ----
            if record_diagnostics:
                left_pos0 = serial_pos0 - 1
                right_pos0 = serial_pos0 + 1

                if 0 <= left_pos0 < N and 0 <= right_pos0 < N:
                    left_item = int(pres_indices[left_pos0] - 1)
                    right_item = int(pres_indices[right_pos0] - 1)

                    if (not retrieved[left_item]) and (not retrieved[right_item]):

                        # --- Item-evidence asymmetry (Δf = f_in(i+1) − f_in(i−1)) ---
                        f_after = (net_weights @ net_c).flatten()
                        delta_f = float(f_after[right_item] - f_after[left_item])
                        ie_deltas_all.append(delta_f)
                        ie_deltas_by_pos[serial_pos1].append(delta_f)
                        ie_pending = {"serial_pos0": serial_pos0, "delta": delta_f}

                        # --- FC alignment asymmetry ---
                        # Δ_FC = cos(c, c_in(i+1)) − cos(c, c_in(i−1))
                        # where c_in(j) = M_FC · f_j / ‖M_FC · f_j‖
                        def _cos_c_cin(item_idx):
                            fv = np.zeros((N, 1))
                            fv[item_idx] = 1.0
                            ci = net_w_fc @ fv
                            ci_n = _norm(ci)
                            if ci_n < 1e-15:
                                return np.nan
                            ci = ci / ci_n
                            cn = _norm(net_c)
                            if cn < 1e-15 or not np.all(np.isfinite(net_c)):
                                return np.nan
                            return _dot(net_c, ci) / cn

                        cos_fwd = _cos_c_cin(right_item)
                        cos_bwd = _cos_c_cin(left_item)
                        delta_fc = cos_fwd - cos_bwd

                        fc_deltas_all.append(delta_fc)
                        fc_deltas_by_pos[serial_pos1].append(delta_fc)
                        fc_pending = {"serial_pos0": serial_pos0, "delta": delta_fc}

            retrieved[winner] = True

    # ================== assemble diagnostics ==================
    if record_diagnostics:
        if f_count > 0:
            f_mean_by_pos = f_sum_by_pos / f_count
            trace["f_mean_by_pos"] = f_mean_by_pos
            trace["f_mean_centered_by_pos"] = f_mean_by_pos - np.mean(f_mean_by_pos)
        else:
            trace["f_mean_by_pos"] = np.full(N, np.nan)
            trace["f_mean_centered_by_pos"] = np.full(N, np.nan)

        diagnostics = {
            "item_evidence": {
                "deltas_all": ie_deltas_all,
                "deltas_by_pos": dict(ie_deltas_by_pos),
                "delta_forward": ie_delta_forward,
                "delta_backward": ie_delta_backward,
            },
            "fc_alignment": {
                "deltas_all": fc_deltas_all,
                "deltas_by_pos": dict(fc_deltas_by_pos),
                "delta_forward": fc_delta_forward,
                "delta_backward": fc_delta_backward,
            },
            "trace": trace,
        }

    return recalls.flatten(), times.flatten(), net_w_fc, net_w_cf, diagnostics


# ─────────────────────────────────────────────────────────────────────
# Batch runner (no diagnostics)
# ─────────────────────────────────────────────────────────────────────

def run_simulation(B_rec, n_sims=500, seed=2026,
                   gamma_fc_val=None, eta_val=None, B_encD_scale=1.0):
    """
    Run multiple independent CMR trials without diagnostics.

    Returns
    -------
    recall_sims : (N, n_sims) int array
    times_sims  : (N, n_sims) float array
    net_w_fc    : (N, N) — from last trial (deterministic across trials)
    net_w_cf    : (N, N)
    """
    rng = np.random.default_rng(seed)
    recall_sims = np.zeros((N, n_sims), dtype=int)
    times_sims = np.zeros((N, n_sims), dtype=float)

    net_w_fc_last = net_w_cf_last = None

    for s in range(n_sims):
        recalls, times, net_w_fc, net_w_cf, _ = simulate_single_trial(
            B_rec=B_rec, rng=rng,
            gamma_fc_val=gamma_fc_val, eta_val=eta_val,
            B_encD_scale=B_encD_scale, record_diagnostics=False,
        )
        recall_sims[:, s] = recalls.astype(int)
        times_sims[:, s] = times
        net_w_fc_last, net_w_cf_last = net_w_fc, net_w_cf

    return recall_sims, times_sims, net_w_fc_last, net_w_cf_last


# ─────────────────────────────────────────────────────────────────────
# Batch runner with full per-trial diagnostics
# ─────────────────────────────────────────────────────────────────────

def run_simulation_with_diagnostics(
    B_rec, n_sims=500, seed=2026,
    gamma_fc_val=None, eta_val=None, B_encD_scale=1.0,
    recency_k=3,
):
    """
    Run multiple independent CMR trials, collecting full diagnostics.

    Returns
    -------
    recall_sims  : (N, n_sims) int array
    times_sims   : (N, n_sims) float array
    net_w_fc     : (N, N)
    net_w_cf     : (N, N)
    all_diagnostics : list[dict | None]
        Per-trial diagnostic dicts, each with keys
        ``"item_evidence"``, ``"fc_alignment"``, ``"trace"``.
    """
    rng = np.random.default_rng(seed)
    recall_sims = np.zeros((N, n_sims), dtype=int)
    times_sims = np.zeros((N, n_sims), dtype=float)

    net_w_fc_last = net_w_cf_last = None
    all_diagnostics = []

    for s in range(n_sims):
        recalls, times, net_w_fc, net_w_cf, diag = simulate_single_trial(
            B_rec=B_rec, rng=rng,
            gamma_fc_val=gamma_fc_val, eta_val=eta_val,
            B_encD_scale=B_encD_scale,
            record_diagnostics=True, recency_k=recency_k,
        )
        recall_sims[:, s] = recalls.astype(int)
        times_sims[:, s] = times
        net_w_fc_last, net_w_cf_last = net_w_fc, net_w_cf
        all_diagnostics.append(diag)

    return recall_sims, times_sims, net_w_fc_last, net_w_cf_last, all_diagnostics

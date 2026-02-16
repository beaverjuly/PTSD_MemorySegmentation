"""
CMR Multi-Parameter Sweep — Sweep Engine
==========================================
``sweep_one_param`` orchestrates simulation, metric computation, and
diagnostic aggregation across a parameter grid.

When ``collect_diagnostics=True`` the main simulation batch already
records per-trial diagnostics (item-evidence asymmetry, FC-alignment
asymmetry, and internal traces), so no separate diagnostic run is
needed.
"""

import numpy as np

from .config import N
from .simulation import run_simulation, run_simulation_with_diagnostics
from .metrics import compute_spc, compute_pfr, compute_lag_crp
from .diagnostics_recall import (
    aggregate_item_evidence_asymmetry,
    aggregate_fc_alignment_asymmetry,
    compute_item_evidence_asymmetry,
)


def sweep_one_param(param_name, param_grid, base_params, n_sims=500,
                    base_seed=2026, collect_diagnostics=True,
                    recency_k=3, verbose=True):
    """
    Sweep exactly one parameter while holding others at *base_params*.

    Parameters
    ----------
    param_name : str
        ``"B_rec"``, ``"gamma_fc"``, ``"eta"``, or ``"B_encD_scale"``.
    param_grid : list
        Values to sweep.
    base_params : dict
        Default values for all parameters.
    n_sims : int
        Simulations per condition.
    base_seed : int
        Random seed (incremented per condition).
    collect_diagnostics : bool
        If True, run with full per-trial diagnostics and aggregate
        item-evidence asymmetry, FC-alignment asymmetry, and internal
        traces.  If False, run a small separate batch for item-evidence
        diagnostics only (legacy path).
    recency_k : int
        Positions counted as "recency" for evidence-mass trace.
    verbose : bool
        Print progress.

    Returns
    -------
    sweep_results : dict
        Keyed by parameter value.  Each entry contains:
        ``recall_sims``, ``times_sims``, ``SPC``, ``PFR``,
        ``lag_vals``, ``lag_probs``, ``net_w_fc``, ``net_w_cf``,
        ``item_evidence_diag``, ``fc_alignment_diag``, ``trace_sims``,
        ``params``.
    """
    sweep_results = {}

    for idx, val in enumerate(param_grid):
        params = base_params.copy()
        params[param_name] = float(val)
        seed = base_seed + idx

        sim_kw = dict(
            B_rec=params["B_rec"],
            n_sims=n_sims,
            seed=seed,
            gamma_fc_val=params["gamma_fc"],
            eta_val=params["eta"],
            B_encD_scale=params["B_encD_scale"],
        )

        if collect_diagnostics:
            (recall_sims, times_sims, net_w_fc, net_w_cf,
             all_diagnostics) = run_simulation_with_diagnostics(
                 **sim_kw, recency_k=recency_k)

            # aggregate per-trial diagnostics
            ie_diag = aggregate_item_evidence_asymmetry(all_diagnostics)
            fc_diag = aggregate_fc_alignment_asymmetry(all_diagnostics)
            trace_sims = [
                (d["trace"] if d is not None and "trace" in d else None)
                for d in all_diagnostics
            ]
        else:
            recall_sims, times_sims, net_w_fc, net_w_cf = \
                run_simulation(**sim_kw)
            ie_diag = compute_item_evidence_asymmetry(
                n_sims=min(200, n_sims // 5), seed=seed, **sim_kw)
            fc_diag = None
            trace_sims = None

        # behavioral metrics
        spc = compute_spc(recall_sims, N)
        pfr = compute_pfr(recall_sims, N)
        lag_vals, lag_probs = compute_lag_crp(recall_sims, N)

        sweep_results[val] = {
            "params": params,
            "recall_sims": recall_sims,
            "times_sims": times_sims,
            "SPC": spc,
            "PFR": pfr,
            "lag_vals": lag_vals,
            "lag_probs": lag_probs,
            "net_w_fc": net_w_fc,
            "net_w_cf": net_w_cf,
            "item_evidence_diag": ie_diag,
            "fc_alignment_diag": fc_diag,
            "trace_sims": trace_sims,
        }

        if verbose:
            print(f"  {param_name}={val:.4g}  done")

    return sweep_results

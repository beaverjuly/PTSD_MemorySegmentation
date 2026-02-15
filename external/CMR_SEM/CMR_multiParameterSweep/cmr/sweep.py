"""
CMR Multi-Parameter Sweep — Sweep Engine
==========================================
Main sweep_one_param function that orchestrates simulation, metric
computation, and diagnostic collection across a parameter grid.
"""

import numpy as np

from .config import N
from .simulation import run_simulation, run_simulation_with_traces
from .metrics import compute_spc, compute_pfr, compute_lag_crp
from .diagnostics import compute_cue_diagnostics


def sweep_one_param(param_name, param_grid, base_params, n_sims=500, base_seed=2026,
                    collect_traces=False, recency_k=3, verbose=True):
    """
    Sweep exactly one parameter while holding others fixed.

    Parameters
    ----------
    param_name : str
        Name of parameter to sweep ("B_rec", "gamma_fc", "eta", "B_encD_scale")
    param_grid : list
        Values to test
    base_params : dict
        Default values for all parameters
    n_sims : int
        Simulations per condition
    base_seed : int
        Random seed (incremented for each condition)
    collect_traces : bool
        If True, use run_simulation_with_traces and store trace_sims
    recency_k : int
        Number of recent recalls for recency mass (only used when collect_traces=True)
    verbose : bool
        Print progress

    Returns
    -------
    sweep_results : dict
        Results keyed by parameter value
    """
    sweep_results = {}

    for idx, val in enumerate(param_grid):

        # Build params for this condition
        params = base_params.copy()
        params[param_name] = float(val)

        seed = base_seed + idx

        sim_kwargs = dict(
            B_rec=params["B_rec"],
            n_sims=n_sims,
            seed=seed,
            gamma_fc_val=params["gamma_fc"],
            eta_val=params["eta"],
            B_encD_scale=params["B_encD_scale"],
        )

        # Run simulation (with or without traces)
        if collect_traces:
            recall_sims, times_sims, net_w_fc, net_w_cf, trace_sims = \
                run_simulation_with_traces(**sim_kwargs, recency_k=recency_k)
        else:
            recall_sims, times_sims, net_w_fc, net_w_cf = run_simulation(**sim_kwargs)
            trace_sims = None

        # Compute metrics
        spc = compute_spc(recall_sims, N)
        pfr = compute_pfr(recall_sims, N)
        lag_vals, lag_probs = compute_lag_crp(recall_sims, N)

        # Diagnostics (use fewer sims to save time)
        diag = compute_cue_diagnostics(
            B_rec=params["B_rec"],
            n_sims=min(200, n_sims // 5),
            seed=seed,
            gamma_fc_val=params["gamma_fc"],
            eta_val=params["eta"],
            B_encD_scale=params["B_encD_scale"],
        )

        # Store results
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
            "cue_diag": diag,
            "trace_sims": trace_sims,
        }

    return sweep_results

"""
CMR B_enc Simulation — Utility Functions
==========================================
Minimal helpers shared across modules.
"""

import numpy as np


def _get_sweep_entry(sweep_results, param_value):
    """
    Look up a sweep-results entry by parameter value.

    This function exists for compatibility with the diagnostics_encoding
    module (which was written for parameter-sweep workflows).  In this
    project we mostly call the core diagnostic functions directly on
    single weight matrices, so this helper is rarely needed.

    Parameters
    ----------
    sweep_results : list[dict]
        Each dict must have a ``"param_value"`` key.
    param_value : float
        The value to match (compared with a small tolerance).

    Returns
    -------
    entry : dict
        The matching sweep-results dict.
    """
    param_value = float(param_value)
    for entry in sweep_results:
        if np.isclose(entry["param_value"], param_value):
            return entry
    raise KeyError(f"No entry for param_value={param_value}")

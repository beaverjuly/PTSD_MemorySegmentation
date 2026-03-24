"""
CMR Multi-Parameter Sweep — Shared Utilities
=============================================
Small helpers used across diagnostics and visualization modules.
"""

import numpy as np


def _get_sweep_entry(sweep_results, v):
    """Robustly fetch ``sweep_results[v]`` when keys may be float-ish."""
    if v in sweep_results:
        return sweep_results[v]
    try:
        fv = float(v)
        if fv in sweep_results:
            return sweep_results[fv]
    except (TypeError, ValueError):
        pass
    for k in sweep_results:
        try:
            if np.isclose(float(k), float(v), atol=1e-12, rtol=0):
                return sweep_results[k]
        except Exception:
            pass
    raise KeyError(f"Missing sweep entry for {v}")

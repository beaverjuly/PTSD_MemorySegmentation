"""
CMR Multi-Parameter Sweep package.

Modules
-------
config               : Shared constants and parameter grids.
simulation           : Single-trial and batch CMR simulation runners.
metrics              : Behavioral analysis metrics (SPC, PFR, lag-CRP, accuracy).
diagnostics_recall   : Recall-stage diagnostics (traces, item-evidence asymmetry,
                       FC-alignment asymmetry).
diagnostics_encoding : Encoding-stage diagnostics (lag-strength profiles, matrix
                       norms, forward/backward lag asymmetry).
visualization        : All plotting functions (behavioral + diagnostic).
sweep                : Parameter sweep orchestration.
utils                : Shared helpers.
"""

from .config import *
from .sweep import sweep_one_param

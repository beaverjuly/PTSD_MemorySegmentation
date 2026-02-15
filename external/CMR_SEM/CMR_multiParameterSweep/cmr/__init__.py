"""
CMR Multi-Parameter Sweep package.

Modules
-------
config         : Shared constants and parameter grids.
simulation     : Single-trial and batch CMR simulation runners.
metrics        : Behavioral analysis metrics (SPC, PFR, lag-CRP, accuracy).
diagnostics    : Cue advantage, trace extraction, association asymmetry.
visualization  : All plotting functions.
sweep          : Parameter sweep orchestration.
"""

from .config import *
from .sweep import sweep_one_param

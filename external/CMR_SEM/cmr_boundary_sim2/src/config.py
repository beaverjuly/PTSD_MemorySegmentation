"""
CMR Boundary-Signal Simulation — Configuration

Supports single- and multi-boundary list structures.

Default configuration: N = 32, boundaries at positions 9, 17, 25
(four 8-item segments).

Two hypothesis families:
  H1 — Boundary/update-signal impairment  (vary Δ only)
  H2 — Global tonic drift reduction       (lower all drifts)
"""

from __future__ import annotations

import numpy as np


# =====================================================================
# List structure
# =====================================================================

N = 32

# Boundary configurations
BOUNDARY_POSITIONS_SINGLE = [13]           # legacy single-boundary
BOUNDARY_POSITIONS_MULTI  = [9, 17, 25]    # four 8-item segments
BOUNDARY_POSITIONS        = BOUNDARY_POSITIONS_MULTI   # active config

# Fixed item-to-slot mapping (seed-locked for reproducibility)
_rng_pres = np.random.default_rng(seed=42)
pres_indices = _rng_pres.permutation(N) + 1  # 1-based


# =====================================================================
# Drift parameters
# =====================================================================

B_NON_BOUNDARY_BASE  = 0.6      # baseline drift for non-boundary items
B_BOUNDARY_DELTA_BASE = 0.4     # healthy boundary boost Δ
B_BOUNDARY_BASE = B_NON_BOUNDARY_BASE + B_BOUNDARY_DELTA_BASE  


# =====================================================================
# Canonical hypothesis condition values
# =====================================================================

# H1 — Boundary impairment: vary Δ, fix non-boundary at 0.6
BOUNDARY_DELTAS = [0.4, 0.2, 0.0]

# H2 — Global tonic lowering: (non-boundary, boundary) pairs
GLOBAL_LEVELS = [(0.6, 1), (0.5, 0.9), (0.4, 0.8)]


# =====================================================================
# CMR base parameters (Polyn, Norman & Kahana, 2009)
# =====================================================================

BASE_PARAMS = {
    "gamma_fc": 0.581,
    "lrate_fc_enc": 0.581,
    "lrate_cf_enc": 1.0,
    "B_rec": 0.36,
    "lrate_fc_rec": 0.0,
    "lrate_cf_rec": 0.0,
    "eta": 0.3699,
    "thresh": 1.0,
    "K": 0.091,
    "L": 0.375,
    "tau": 413.0,
    "dt": 100.0,
    "rec_time": 90_000.0,
    "sem_weight": 0.5,
    "episodic_weight": 0.5,
}

sem_mat = np.eye(N)


# =====================================================================
# Simulation defaults
# =====================================================================

N_SIMS = 1000
SEED = 2026

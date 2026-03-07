"""
CMR B_enc Targeted Simulation — Configuration

Base CMR parameters (Polyn, Norman & Kahana, 2009 fits), N=24 list
structure with valence-segmented blocks, and reward sequences for
RPE-modulated encoding simulations.

List structure
--------------
The 24-item study list is split into two valence blocks:

    Positions  1–12 : reward-context items  (high outcomes)
    Positions 13–24 : loss-context items    (low outcomes)

The reward sequence starts at a high baseline (90) for the reward
block, then drops to a low baseline (10) for the loss block. This
produces a large RPE at the reward → loss transition (position 13),
modelling the sudden onset of unexpected negative outcomes.
"""

from __future__ import annotations

import numpy as np

# =====================================================================
# List structure
# =====================================================================

N = 24

# Fixed item-to-slot mapping (deterministic across runs)
_rng_pres = np.random.default_rng(seed=42)
pres_indices = _rng_pres.permutation(N) + 1   # 1-based indices

# Valence masks: first half = reward, second half = loss
REWARD_POSITIONS = np.array([True] * 12 + [False] * 12)
LOSS_POSITIONS   = ~REWARD_POSITIONS

# =====================================================================
# Baseline encoding drift (Polyn, Norman & Kahana, 2009)
# =====================================================================

# Position 1 gets full drift (B=1.0); all subsequent positions get
# B=0.65. This primacy gradient is standard in CMR implementations.
B_encD_baseline = np.array([1.0] + [0.65] * (N - 1), dtype=float)

# =====================================================================
# Reward / outcome sequences
# =====================================================================

# Each sequence has N+1 entries:
#   element 0    = prior expectation (before any item is studied)
#   elements 1–N = outcomes observed after studying each item)
#
# RPE[i] = |sequence[i+1] − sequence[i]|   (one per study position)
#
# High-reward block (90) → sudden drop → low/loss block (10).
# The transition at position 13 produces a large RPE, consistent
# with unexpected-loss onset.
SEQUENCE_LOSS_ONSET = np.array(
    [90] + [90] * 12 + [10] * 12,
    dtype=float,
)

# =====================================================================
# CMR base parameters (Polyn, Norman & Kahana, 2009 fits)
# =====================================================================

BASE_PARAMS = {
    # --- association strengths / encoding ---
    "gamma_fc":     0.581,
    "lrate_cf_enc": 1.0,

    # --- retrieval ---
    "B_rec":        0.36,
    "lrate_fc_rec": 0.0,
    "lrate_cf_rec": 0.0,

    # --- accumulator / decision ---
    "eta":          0.3699,
    "thresh":       1.0,
    "K":            0.091,
    "L":            0.375,
    "tau":          413.0,
    "dt":           100.0,
    "rec_time":     90_000.0,

    # --- semantic / episodic mixing ---
    "sem_weight":       0.5,
    "episodic_weight":  0.5,
}

# Semantic similarity matrix (identity = no inter-item similarity)
sem_mat = np.eye(N)

# =====================================================================
# Simulation defaults
# =====================================================================

N_SIMS = 1000
SEED   = 2026

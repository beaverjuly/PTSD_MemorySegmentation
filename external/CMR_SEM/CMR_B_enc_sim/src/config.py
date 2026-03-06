"""
CMR B_enc Simulation — Configuration
======================================
Base CMR parameters (Polyn, Norman & Kahana, 2009 fits) and reward
sequences for RPE-modulated encoding simulations.

All parameter values are taken from the original fitted CMR model.
The reward sequences define the RPE landscape that modulates B_enc
when RPE-driven encoding is enabled.
"""

import numpy as np

# =====================================================================
# List structure
# =====================================================================

N = 10  # number of items on the study list

# Fixed random assignment of items to feature-vector slots.
# Using a fixed seed so that every run produces identical results.
_rng_pres = np.random.default_rng(seed=42)
pres_indices = _rng_pres.permutation(N) + 1  # 1-based indices


# =====================================================================
# Encoding drift schedule (baseline)
# =====================================================================

# "Manual drift" from the original model:
# Position 1 gets full drift (B=1), all subsequent positions get B=0.65.
# This produces a strong primacy bump in the serial-position curve.
B_encD_baseline = np.array([1.0, 0.65, 0.65, 0.65, 0.65,
                            0.65, 0.65, 0.65, 0.65, 0.65])


# =====================================================================
# CMR base parameters (Polyn, Norman & Kahana, 2009)
# =====================================================================

BASE_PARAMS = {
    # --- association strengths ---
    "gamma_fc": 0.581,        # pre-experimental M_FC weight (feature→context)

    # --- encoding ---
    "lrate_fc_enc": 0.581,    # = gamma_fc (Hebbian learning rate, M_FC)
    "lrate_cf_enc": 1.0,      # learning rate for M_CF during encoding

    # --- retrieval ---
    "B_rec": 0.36,            # context drift rate during retrieval
    "lrate_fc_rec": 0.0,      # M_FC learning during retrieval (off)
    "lrate_cf_rec": 0.0,      # M_CF learning during retrieval (off)

    # --- accumulator / decision ---
    "eta": 0.3699,            # noise SD in the leaky-accumulator race
    "thresh": 1.0,            # accumulator threshold
    "K": 0.091,               # self-decay rate
    "L": 0.375,               # lateral inhibition
    "tau": 413.0,             # time constant (ms)
    "dt": 100.0,              # time step (ms)
    "rec_time": 90_000.0,     # total retrieval window (ms) — 90 s

    # --- semantic / episodic mixing ---
    "sem_weight": 0.5,
    "episodic_weight": 0.5,
}

# Semantic similarity matrix (identity = no inter-item similarity)
sem_mat = np.eye(N)


# =====================================================================
# Reward sequences for RPE modulation
# =====================================================================

# Each sequence has N+1 entries: the first element is the "prior
# expectation" (before any item is studied), and entries 1..N are the
# outcomes observed after studying each item.
#
# RPE at position i  =  |sequence[i+1] − sequence[i]|
#   (absolute prediction error, one per study item)

# (A) Primacy sequence: small, gradually declining rewards.
#     → Produces a large RPE at position 1 and small RPEs thereafter.
SEQUENCE_PRIMACY = np.array([50, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45])

# (B) Mid-list surprise: stable rewards with a large RPE mid-list.
#     → Produces a spike in B_enc around positions 5–6.
SEQUENCE_MIDLIST = np.array([0, 54, 57, 56, 53, 55, 7, 5, 4, 6, 3])

# (C) Flat / no-surprise: constant rewards → RPE ≈ 0 everywhere.
SEQUENCE_FLAT = np.array([50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50])


# =====================================================================
# Simulation defaults
# =====================================================================

N_SIMS = 1000          # Monte-Carlo trials per condition
SEED = 2026            # default RNG seed for reproducibility

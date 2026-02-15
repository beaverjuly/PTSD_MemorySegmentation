"""
CMR Multi-Parameter Sweep — Configuration & Constants
======================================================
All shared state: list parameters, base model parameters, sweep grids,
retrieval route weights, encoding drift schedule, and fixed accumulator constants.
"""

import numpy as np

# ── List Configuration ───────────────────────────────────────────────
N = 10
np.random.seed(42)
pres_indices = np.random.permutation(N) + 1  # 1-indexed presentation order
sequence = np.array([50, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45])

# ── Base Parameters (defaults when NOT sweeping) ─────────────────────
BASE_PARAMS = {
    "B_rec": 0.55,
    "gamma_fc": 0.581,
    "eta": 0.3699,
    "B_encD_scale": 1.0,
}

# ── Parameter Grids for Sweeping ─────────────────────────────────────
B_rec_grid = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
gamma_fc_grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
eta_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
B_encD_scale_grid = [0.2, 0.4, 0.6, 0.8, 1.0]

BASE_SEED = 2026
n_sims = 1000

# ── Retrieval Route Weights ──────────────────────────────────────────
sem = 0
episodic = 1
sem_weight = sem / (episodic + sem)
episodic_weight = episodic / (episodic + sem)

# ── Context Drift & Semantic Matrix ──────────────────────────────────
B_encD = np.array([1.0, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65])
sem_mat = np.eye(N)

# ── Fixed Accumulator / Learning-Rate Constants ──────────────────────
lrate_cf_enc = 1.0
lrate_fc_rec = 0.0
lrate_cf_rec = 0.0

thresh = 1.0
rec_time = 90000   # ms
dt = 100           # ms
tau = 413          # ms
K = 0.091
L = 0.375

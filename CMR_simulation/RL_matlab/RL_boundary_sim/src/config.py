"""
Centralized constants and default parameter sets.

All tunable knobs live here so that the notebook stays clean
and changes propagate consistently.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Task design
# ---------------------------------------------------------------------------
@dataclass
class TaskConfig:
    """Parameters that define the Exp1 trial structure."""
    n_lists: int = 4                  # number of study lists
    items_per_list: int = 24          # items per list
    items_per_train: int = 6          # items between successive boundaries
    reward_mean: float = 50.0         # mean reward (range 0–100)
    reward_std: float = 20.0          # reward noise SD
    initial_value: float = 50.0       # starting value estimate


# ---------------------------------------------------------------------------
# Model / simulation
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Parameters shared by the RL + encoding‑drift + recall model."""
    alpha: float = 0.3                # Rescorla–Wagner learning rate
    beta_base: float = 0.5           # baseline encoding drift
    delta_boundary: float = 0.25      # boundary drift increment (H0 & H2)
    rpe_drift_weight: float = 0.15    # |RPE|→drift coupling strength
    recall_intercept: float = -0.5    # logistic recall intercept
    recall_drift_weight: float = 2.0  # drift → recall logistic weight
    recall_noise_std: float = 0.05    # noise on recall probability


# ---------------------------------------------------------------------------
# Hypothesis‑specific overrides
# ---------------------------------------------------------------------------
@dataclass
class HypothesisParams:
    """
    Overrides that distinguish H0, H1, H2.

    Each field is the *value* used in place of the corresponding
    ModelConfig default when that hypothesis is active.
    """
    # H1 uses a smaller boundary increment
    h1_delta_boundary: float = 0.10

    # H2 applies a global bias shift downward
    h2_bias_shift: float = 0.15


# ---------------------------------------------------------------------------
# Simulation run
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    """Top‑level simulation‑run settings."""
    n_subjects: int = 100
    seed: int = 42
    hypothesis_names: tuple = ("baseline", "H1_smaller_boundary", "H2_global_reduction")


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------
def get_default_task_config() -> TaskConfig:
    return TaskConfig()


def get_default_model_config() -> ModelConfig:
    return ModelConfig()


def get_hypothesis_params() -> HypothesisParams:
    return HypothesisParams()


def get_default_sim_config() -> SimConfig:
    return SimConfig()


def get_all_configs() -> Dict[str, Any]:
    """Return a dict of all default config objects (handy for serialisation)."""
    return {
        "task": get_default_task_config(),
        "model": get_default_model_config(),
        "hypothesis": get_hypothesis_params(),
        "sim": get_default_sim_config(),
    }

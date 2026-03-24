"""Small shared helper functions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def set_seed(seed: int) -> np.random.Generator:
    """Return a seeded numpy Generator."""
    return np.random.default_rng(seed)


def sigmoid(x):
    """Numerically stable sigmoid, works on scalars and arrays."""
    x = np.asarray(x, dtype=float)
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if needed; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Save DataFrame to CSV, creating parent dirs."""
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False)
    print(f"  Saved {len(df)} rows → {p}")


def save_json(obj, path: str | Path) -> None:
    """Serialise a JSON‑friendly object to disk."""
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  Saved JSON → {p}")


def zscore_safe(x):
    """Z‑score that returns zeros when std == 0."""
    x = np.asarray(x, dtype=float)
    s = np.std(x, ddof=1)
    if s == 0:
        return np.zeros_like(x)
    return (x - np.mean(x)) / s

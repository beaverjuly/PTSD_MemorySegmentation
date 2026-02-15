"""
CMR Multi-Parameter Sweep — Visualization Functions
=====================================================
All plotting routines: SPC, PFR heatmap, lag-CRP (including diagnostics
and forward-lag panels), cue advantage, cosine similarity, evidence profiles,
association asymmetry, and generic scalar-metric sweeps.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from .config import N
from .metrics import (
    compute_lag_crp,
    lag_crp_with_counts,
    conditional_forward_lag_rates_from_counts,
    lag_stats_unconditional_forward,
    recall_accuracy,
    sweep_scalar_metric_array,
)
from .diagnostics import (
    _get_sweep_entry,
    get_trace_sims,
    mean_curve_over_sims,
    mean_vector_over_sims,
    neighbor_fc_asymmetry_means_from_sweep,
)


# ─────────────────────────────────────────────────────────────────────
# Color palette helpers
# ─────────────────────────────────────────────────────────────────────

def make_sweep_colors(param_grid, cmap_name="viridis"):
    """
    Returns:
      colors: list of RGBA colors, same length as param_grid
      norm: Normalize object (for colorbar)
      cmap: Colormap object
    """
    param_grid = np.asarray(param_grid, dtype=float)
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=np.min(param_grid), vmax=np.max(param_grid))
    colors = [cmap(norm(v)) for v in param_grid]
    return colors, norm, cmap


def line_with_colored_points(ax, x, y, colors):
    ax.plot(x, y, linewidth=1.0, alpha=0.6)
    ax.scatter(x, y, c=colors, s=50, edgecolor="none")


# ─────────────────────────────────────────────────────────────────────
# PFR heatmap
# ─────────────────────────────────────────────────────────────────────

def plot_pfr_heatmap(
    sweep_results,
    param_grid,
    param_name="Parameter",
    cmap_name="viridis",
    show_colorbar=True,
):
    param_grid = np.asarray(param_grid, dtype=float)
    positions = np.arange(1, N + 1)

    M = np.zeros((len(param_grid), N), dtype=float)
    for i, val in enumerate(param_grid):
        M[i, :] = sweep_results[val]["PFR"]

    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    im = ax.imshow(
        M,
        aspect="auto",
        origin="lower",
        cmap=cmap_name,
        interpolation="nearest",
    )

    ax.set_title(f"PFR heatmap across {param_name}")
    ax.set_xlabel("Serial Position")
    ax.set_ylabel(param_name)

    ax.set_xticks(np.arange(N))
    ax.set_xticklabels(positions)

    max_yticks = 8
    if len(param_grid) <= max_yticks:
        ytick_idx = np.arange(len(param_grid))
    else:
        ytick_idx = np.linspace(0, len(param_grid) - 1, max_yticks).astype(int)

    ax.set_yticks(ytick_idx)
    ax.set_yticklabels([f"{param_grid[j]:.2f}" for j in ytick_idx])

    plt.tight_layout()

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("P(First Recall)")

    plt.show()


# ─────────────────────────────────────────────────────────────────────
# SPC sweep
# ─────────────────────────────────────────────────────────────────────

def plot_spc_sweep(sweep_results, param_grid, param_name="Parameter",
                   cmap_name="viridis", show_colorbar=True):
    plt.figure(figsize=(8, 5))
    serial_labels = np.arange(1, N + 1)

    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name=cmap_name)

    for param_val, color in zip(param_grid, colors):
        spc = sweep_results[param_val]["SPC"]
        plt.plot(serial_labels, spc, marker="o", color=color)

    plt.title(f"Serial Position Curve across {param_name}")
    plt.xlabel("Serial Position")
    plt.ylabel("P(Recall)")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if show_colorbar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        ax = plt.gca()
        fig = plt.gcf()
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(param_name)

    plt.show()


# ─────────────────────────────────────────────────────────────────────
# lag-CRP sweep
# ─────────────────────────────────────────────────────────────────────

def plot_split_lags(ax, lags, y, color, marker="o", **kwargs):
    lags = np.asarray(lags)
    y = np.asarray(y)
    neg = lags < 0
    pos = lags > 0
    ax.plot(lags[neg], y[neg], marker=marker, color=color, **kwargs)
    ax.plot(lags[pos], y[pos], marker=marker, color=color, **kwargs)


def plot_lag_crp_sweep(
    sweep_results,
    param_grid,
    param_name="Parameter",
    cmap_name="viridis",
    show_colorbar=True,
):
    plt.figure(figsize=(8, 5))

    param_grid = np.asarray(param_grid, dtype=float)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name=cmap_name)

    ax = plt.gca()

    for param_val, color in zip(param_grid, colors):
        recall_sims = sweep_results[param_val]["recall_sims"]
        lag_vals, crp = compute_lag_crp(recall_sims, N)

        plot_split_lags(ax, lag_vals, crp, color=color, marker="o")

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"Lag-CRP across {param_name}")
    ax.set_xlabel("Lag (next − current)")
    ax.set_ylabel("CRP")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if show_colorbar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig = plt.gcf()
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(param_name)

    plt.show()


# ─────────────────────────────────────────────────────────────────────
# lag-CRP diagnostics (CRP + denominator + numerator)
# ─────────────────────────────────────────────────────────────────────

def plot_lag_crp_diagnostics(
    sweep_results,
    param_grid,
    param_name=r"$B_{rec}$",
    cmap_name="viridis",
):
    param_grid = np.asarray(param_grid, dtype=float)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name=cmap_name)

    first = param_grid[0]
    recall_sims = sweep_results[first]["recall_sims"]
    lags, _, _, _ = lag_crp_with_counts(recall_sims, N)

    fig, axs = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    for val, color in zip(param_grid, colors):
        recall_sims = sweep_results[val]["recall_sims"]
        lags, crp, num, den = lag_crp_with_counts(recall_sims, N)

        plot_split_lags(axs[0], lags, crp, color=color, marker="o")
        plot_split_lags(axs[1], lags, den, color=color, marker="o")
        plot_split_lags(axs[2], lags, num, color=color, marker="o")

    axs[0].axvline(0, color="gray", linestyle="--", alpha=0.5)
    axs[0].set_ylabel("CRP (conditional)")
    axs[0].set_title(f"Lag-CRP across {param_name}")

    axs[1].set_ylabel("denominator (opportunities)")
    axs[1].set_title("Opportunities per lag (denominator)")

    axs[2].set_ylabel("numerator (observed transitions)")
    axs[2].set_title("Observed transitions per lag (numerator)")
    axs[2].set_xlabel("Lag (next − current)")

    for ax in axs:
        ax.grid(alpha=0.3)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(param_name)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# Forward-lag rate arrays + panel plots
# ─────────────────────────────────────────────────────────────────────

def sweep_lag_rate_arrays(
    sweep_results,
    param_grid,
    N,
    large_lag_thresh=4,
    conditional=False,
):
    """
    Returns arrays aligned with param_grid:

    If conditional=False:
      lag_rate[i]   = P(lag=+1)
      large_rate[i] = P(lag>=k)
      mean_abs[i]   = E[|lag|]   (unconditional)

    If conditional=True:
      lag_rate[i]   = sum_num(lag=+1) / sum_den(lag=+1)
      large_rate[i] = sum_num(lag>=k) / sum_den(lag>=k)
      mean_abs[i]   = unconditional E[|lag|]
    """
    param_grid = np.asarray(param_grid, dtype=float)

    lag_rate = np.full(len(param_grid), np.nan, dtype=float)
    large_rate = np.full(len(param_grid), np.nan, dtype=float)
    mean_abs = np.full(len(param_grid), np.nan, dtype=float)

    def _get_recall_sims(v):
        if v in sweep_results:
            return sweep_results[v]["recall_sims"]
        if float(v) in sweep_results:
            return sweep_results[float(v)]["recall_sims"]
        for k in sweep_results:
            try:
                if np.isclose(float(k), float(v), atol=1e-12, rtol=0):
                    return sweep_results[k]["recall_sims"]
            except Exception:
                pass
        raise KeyError(f"Missing recall_sims for param value {v}")

    for i, v in enumerate(param_grid):
        rs = _get_recall_sims(v)

        u_lagp1, u_largef, u_meanabs = lag_stats_unconditional_forward(
            rs, large_lag_thresh=large_lag_thresh
        )
        mean_abs[i] = u_meanabs

        if conditional:
            lags, crp, num, den = lag_crp_with_counts(rs, N)
            c_lagp1, c_largef = conditional_forward_lag_rates_from_counts(
                lags, num, den, large_lag_thresh=large_lag_thresh
            )
            lag_rate[i] = c_lagp1
            large_rate[i] = c_largef
        else:
            lag_rate[i] = u_lagp1
            large_rate[i] = u_largef

    return lag_rate, large_rate, mean_abs


def plot_lagOne_largeLag_rates(
    sweep_results,
    param_grid,
    param_label,
    N,
    large_lag_thresh=4,
    conditional=False,
    cmap_name="viridis",
):
    """
    If conditional=False:
      3 panels: P(lag=+1), P(lag>=k), E[|lag|] (unconditional)

    If conditional=True:
      2 panels: P(lag=+1), P(lag>=k) (opportunity-corrected)
    """
    param_grid = np.asarray(param_grid, dtype=float)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name=cmap_name)

    lag_rate, large_rate, mean_abs = sweep_lag_rate_arrays(
        sweep_results,
        param_grid,
        N=N,
        large_lag_thresh=large_lag_thresh,
        conditional=conditional,
    )

    ncols = 2 if conditional else 3
    fig, axs = plt.subplots(1, ncols, figsize=(12, 3.6), sharex=True)

    if ncols == 2:
        axs0, axs1 = axs
    else:
        axs0, axs1, axs2 = axs

    line_with_colored_points(axs0, param_grid, lag_rate, colors)
    axs0.set_title(
        f"Forward lag-1 rate vs {param_label}"
        + (" (conditional)" if conditional else " (unconditional)")
    )
    axs0.set_ylabel(r"$P(\ell = +1)$")
    axs0.grid(alpha=0.3)

    line_with_colored_points(axs1, param_grid, large_rate, colors)
    axs1.set_title(
        f"Forward large-lag rate vs {param_label}"
        + (" (conditional)" if conditional else " (unconditional)")
    )
    axs1.set_ylabel(rf"$P(\ell \geq {large_lag_thresh})$")
    axs1.grid(alpha=0.3)

    if not conditional:
        line_with_colored_points(axs2, param_grid, mean_abs, colors)
        axs2.set_title(f"Mean |lag| vs {param_label} (unconditional)")
        axs2.set_ylabel(r"$\mathbb{E}[|\ell|]$")
        axs2.grid(alpha=0.3)

    for ax in (axs if hasattr(axs, "__len__") else [axs]):
        ax.set_xlabel(param_label)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig.tight_layout(rect=[0, 0, 0.9, 1])
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(param_label)

    plt.show()


# ─────────────────────────────────────────────────────────────────────
# Cue advantage by serial position
# ─────────────────────────────────────────────────────────────────────

def plot_cue_advantage_by_pos_sweep(
    sweep_results,
    param_grid,
    param_name="Parameter",
    cmap_name="viridis",
    show_colorbar=True,
):
    plt.figure(figsize=(8, 5))
    positions = np.arange(1, N + 1)

    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name=cmap_name)

    for param_val, color in zip(param_grid, colors):
        diag = sweep_results[param_val]["cue_diag"]

        means = []
        for p in positions:
            vals = diag["deltas_by_pos"].get(p, [])
            means.append(np.mean(vals) if len(vals) > 0 else np.nan)

        plt.plot(positions, means, marker="o", color=color)

    plt.axhline(0, linestyle="--", alpha=0.6)
    plt.title(f"Cue advantage by position across {param_name}")
    plt.xlabel(r"current serial position $i$")
    plt.ylabel(r"mean $f(i+1)-f(i-1)$")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if show_colorbar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig = plt.gcf()
        ax = plt.gca() if fig.axes else fig.add_subplot(111)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(param_name)

    plt.show()


# ─────────────────────────────────────────────────────────────────────
# Neighbor cue advantage vs parameter
# ─────────────────────────────────────────────────────────────────────

def plot_neighbor_advantage_vs_param(
    sweep_results,
    param_grid,
    param_name="Parameter",
    cmap_name="viridis",
    show_colorbar=True,
):
    param_grid = np.asarray(param_grid, dtype=float)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name=cmap_name)

    adv = []
    for val in param_grid:
        diag = sweep_results[val]["cue_diag"]
        adv.append(np.mean(diag["deltas_all"]) if len(diag["deltas_all"]) > 0 else np.nan)
    adv = np.asarray(adv, dtype=float)

    plt.figure(figsize=(7, 4))

    plt.plot(param_grid, adv, linewidth=1.0, alpha=0.5)
    plt.scatter(param_grid, adv, c=param_grid, cmap=cmap, norm=norm)

    plt.axhline(0, linestyle="--", alpha=0.6)
    plt.title(f"Neighbor cue advantage vs {param_name}")
    plt.xlabel(param_name)
    plt.ylabel(r"mean cue advantage: $f(i+1) - f(i-1)$")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if show_colorbar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig = plt.gcf()
        ax = plt.gca() if fig.axes else fig.add_subplot(111)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(param_name)

    plt.show()


# ─────────────────────────────────────────────────────────────────────
# Asymmetry three-curve helper
# ─────────────────────────────────────────────────────────────────────

def plot_asymmetry_three_curves(ax, x, d_all, d_fwd, d_bwd, colors, title, xlabel):
    x = np.asarray(x, dtype=float)

    l1, = ax.plot(x, d_all, linewidth=1.5, alpha=0.9, label=r"$\mathbb{E}[\Delta_{FC}]$")
    l2, = ax.plot(x, d_fwd, linewidth=1.5, alpha=0.9, label=r"$\mathbb{E}[\Delta_{FC}\mid \ell=+1]$")
    l3, = ax.plot(x, d_bwd, linewidth=1.5, alpha=0.9, label=r"$\mathbb{E}[\Delta_{FC}\mid \ell=-1]$")

    ax.scatter(x, d_all, c=colors, s=45, edgecolor="none")
    ax.scatter(x, d_fwd, c=colors, s=45, edgecolor="none")
    ax.scatter(x, d_bwd, c=colors, s=45, edgecolor="none")

    ax.axhline(0, linestyle="--", alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"neighbor $M_{FC}$ context-input asymmetry $\Delta_{FC}$")
    ax.grid(alpha=0.3)
    ax.legend(handles=[l1, l2, l3], frameon=False, loc="upper right")


# ─────────────────────────────────────────────────────────────────────
# Scalar metric vs parameter
# ─────────────────────────────────────────────────────────────────────

def plot_scalar_metric_vs_param(
    sweep_results, param_grid, param_label, metric_fn,
    title, y_label, cmap_name="viridis", show_colorbar=True,
):
    """Plot a scalar metric (e.g. recall accuracy) vs swept parameter."""
    param_grid = np.asarray(param_grid, dtype=float)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name=cmap_name)
    y = sweep_scalar_metric_array(sweep_results, param_grid, metric_fn)

    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.plot(param_grid, y, linewidth=1.0, alpha=0.6)
    ax.scatter(param_grid, y, c=colors, s=55, edgecolor="none")
    ax.set_title(title)
    ax.set_xlabel(param_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)

    if show_colorbar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(param_label)
    fig.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# Cue advantage paired (by position + pooled)
# ─────────────────────────────────────────────────────────────────────

def plot_cue_advantage_paired(
    sweep_results, param_grid, param_name="Parameter", cmap_name="viridis",
):
    """Side-by-side: (L) cue advantage by serial position, (R) pooled advantage vs parameter."""
    param_grid = np.asarray(param_grid, dtype=float)
    positions = np.arange(1, N + 1)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name=cmap_name)

    fig, (ax_pos, ax_pool) = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

    for pv, color in zip(param_grid, colors):
        diag = sweep_results[pv]["cue_diag"]
        means = [np.mean(diag["deltas_by_pos"].get(p, [np.nan])) for p in positions]
        ax_pos.plot(positions, means, marker="o", color=color, markersize=4)
    ax_pos.axhline(0, linestyle="--", alpha=0.6)
    ax_pos.set_title(f"Cue advantage by position across {param_name}")
    ax_pos.set_xlabel(r"current serial position $i$")
    ax_pos.set_ylabel(r"mean $f(i+1)-f(i-1)$")
    ax_pos.grid(alpha=0.3)

    adv = np.array([
        np.mean(sweep_results[v]["cue_diag"]["deltas_all"])
        if len(sweep_results[v]["cue_diag"]["deltas_all"]) > 0 else np.nan
        for v in param_grid
    ], dtype=float)
    ax_pool.plot(param_grid, adv, linewidth=1.0, alpha=0.5)
    ax_pool.scatter(param_grid, adv, c=param_grid, cmap=cmap, norm=norm, s=50)
    ax_pool.axhline(0, linestyle="--", alpha=0.6)
    ax_pool.set_title(f"Neighbor cue advantage vs {param_name}")
    ax_pool.set_xlabel(param_name)
    ax_pool.set_ylabel(r"mean cue advantage: $f(i+1) - f(i-1)$")
    ax_pool.grid(alpha=0.3)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_pos, ax_pool], pad=0.015, shrink=0.85)
    cbar.set_label(param_name)
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# Cosine similarity sweep
# ─────────────────────────────────────────────────────────────────────

def plot_cosine_similarity_sweep(
    sweep_results, param_grid, param_name="Parameter", cmap_name="viridis",
):
    """Cosine similarity cos(c, c_in) after each retrieval update across a swept parameter."""
    param_grid = np.asarray(param_grid, dtype=float)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name=cmap_name)

    fig, ax = plt.subplots(figsize=(8, 5))
    for val, color in zip(param_grid, colors):
        ts = get_trace_sims(sweep_results, val)
        x, ca_mean = mean_curve_over_sims(ts, "cos_after")
        if x.size:
            ax.plot(x, ca_mean, marker="o", color=color, markersize=4)

    ax.axhline(1, linestyle="--", alpha=0.6)
    ax.set_title(
        rf"Cosine similarity $\cos(c,c_{{in}})$ after retrieval update across {param_name} (mean over sims)"
    )
    ax.set_xlabel("recall step (successful recalls)")
    ax.set_ylabel(r"$\cos(c, c_{in})$")
    ax.grid(alpha=0.3)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(param_name)
    fig.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# Mean evidence by serial position
# ─────────────────────────────────────────────────────────────────────

def plot_mean_evidence_by_pos(
    sweep_results, param_grid, param_name="Parameter", cmap_name="viridis",
):
    """Side-by-side: (L) raw mean evidence, (R) centered mean evidence by serial position."""
    param_grid = np.asarray(param_grid, dtype=float)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name=cmap_name)
    serial_pos = np.arange(1, N + 1)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for val, color in zip(param_grid, colors):
        ts = get_trace_sims(sweep_results, val)
        f_mean = mean_vector_over_sims(ts, "f_mean_by_pos")
        f_centered = mean_vector_over_sims(ts, "f_mean_centered_by_pos")
        axs[0].plot(serial_pos, f_mean, marker="o", color=color, markersize=4)
        axs[1].plot(serial_pos, f_centered, marker="o", color=color, markersize=4)

    axs[0].set_title(rf"Mean evidence by serial position (raw) across {param_name}")
    axs[0].set_xlabel("Serial position $i$")
    axs[0].set_ylabel(r"$\mathbb{E}[f_{in}(i)]$")
    axs[0].grid(alpha=0.3)

    axs[1].set_title(rf"Mean evidence by serial position (centered) across {param_name}")
    axs[1].set_xlabel("Serial position $i$")
    axs[1].set_ylabel(r"$\mathbb{E}[\tilde f_{in}(i)]$")
    axs[1].grid(alpha=0.3)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(param_name)
    plt.tight_layout(rect=[0, 0, 0.91, 1])
    plt.show()

"""
CMR Multi-Parameter Sweep — Visualization
==========================================
All plotting routines, organized by category:

1. **Mandatory behavioral readouts** — produced for every sweep.
2. **Recall-stage diagnostics** — context-update traces, evidence
   profiles, item-evidence asymmetry, FC-alignment asymmetry.
3. **Encoding-stage diagnostics** — matrix band profiles, norms,
   neighbor-band asymmetry.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from .config import N
from .metrics import (
    compute_lag_crp,
    lag_crp_with_counts,
    conditional_forward_lag_rates,
    conditional_backward_lag_rates,
    unconditional_transition_summaries,
    recall_accuracy,
)
from .utils import _get_sweep_entry
from .diagnostics_recall import (
    get_trace_sims,
    mean_curve_over_sims,
    mean_vector_over_sims,
    item_evidence_asymmetry_means,
    fc_alignment_asymmetry_means,
)
from .diagnostics_encoding import (
    sweep_matrix_norms,
    sweep_band_profiles,
    sweep_neighbor_band_asymmetry,
)


# =====================================================================
#  Color palette helpers
# =====================================================================

def make_sweep_colors(param_grid, cmap_name="viridis"):
    """(colors, norm, cmap) for a parameter grid."""
    param_grid = np.asarray(param_grid, dtype=float)
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=param_grid.min(), vmax=param_grid.max())
    return [cmap(norm(v)) for v in param_grid], norm, cmap


def _add_colorbar(fig, ax_or_axs, norm, cmap, label):
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_or_axs, pad=0.02)
    cbar.set_label(label)
    return cbar


def line_with_colored_points(ax, x, y, colors):
    ax.plot(x, y, linewidth=1.0, alpha=0.6)
    ax.scatter(x, y, c=colors, s=50, edgecolor="none")


# =====================================================================
#  1. BEHAVIORAL READOUTS
# =====================================================================

# -- Recall accuracy --------------------------------------------------

def _sweep_scalar_metric_array(sweep_results, param_grid, metric_fn):
    """y[i] = metric_fn(recall_sims) aligned with param_grid."""
    param_grid = np.asarray(param_grid, dtype=float)
    y = np.full(len(param_grid), np.nan)
    for i, v in enumerate(param_grid):
        entry = _get_sweep_entry(sweep_results, v)
        y[i] = metric_fn(entry["recall_sims"])
    return y


def plot_recall_accuracy(sweep_results, param_grid, param_name,
                         cmap_name="viridis"):
    param_grid = np.asarray(param_grid, dtype=float)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name)
    y = _sweep_scalar_metric_array(
        sweep_results, param_grid,
        lambda rs: recall_accuracy(rs, N=N, unique=True))

    fig, ax = plt.subplots(figsize=(7.2, 4))
    line_with_colored_points(ax, param_grid, y, colors)
    ax.set_title(f"Recall accuracy vs {param_name}")
    ax.set_xlabel(param_name)
    ax.set_ylabel(r"$\mathbb{E}[\#\mathrm{unique\ recalled}/N]$")
    ax.grid(alpha=0.3)
    _add_colorbar(fig, ax, norm, cmap, param_name)
    fig.tight_layout(); plt.show()


# -- PFR heatmap ------------------------------------------------------

def plot_pfr_heatmap(sweep_results, param_grid, param_name="Parameter",
                     cmap_name="viridis", show_colorbar=True):
    param_grid = np.asarray(param_grid, dtype=float)
    M = np.zeros((len(param_grid), N))
    for i, val in enumerate(param_grid):
        M[i, :] = sweep_results[val]["PFR"]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap_name,
                   interpolation="nearest")
    ax.set_title(f"PFR heatmap across {param_name}")
    ax.set_xlabel("Serial Position")
    ax.set_ylabel(param_name)
    ax.set_xticks(np.arange(N)); ax.set_xticklabels(np.arange(1, N + 1))
    nticks = min(8, len(param_grid))
    ytick_idx = np.linspace(0, len(param_grid) - 1, nticks).astype(int)
    ax.set_yticks(ytick_idx)
    ax.set_yticklabels([f"{param_grid[j]:.2f}" for j in ytick_idx])
    if show_colorbar:
        plt.colorbar(im, ax=ax).set_label("P(First Recall)")
    fig.tight_layout(); plt.show()


# -- SPC sweep --------------------------------------------------------

def plot_spc_sweep(sweep_results, param_grid, param_name="Parameter",
                   cmap_name="viridis"):
    param_grid = np.asarray(param_grid, dtype=float)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name)
    sp = np.arange(1, N + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for val, c in zip(param_grid, colors):
        ax.plot(sp, sweep_results[val]["SPC"], marker="o", color=c)
    ax.set_title(f"Serial Position Curve across {param_name}")
    ax.set_xlabel("Serial Position"); ax.set_ylabel("P(Recall)")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    _add_colorbar(fig, ax, norm, cmap, param_name)
    fig.tight_layout(); plt.show()


# -- lag-CRP sweep (split negative / positive) -----------------------

def _plot_split_lags(ax, lags, y, color, marker="o", **kw):
    lags, y = np.asarray(lags), np.asarray(y)
    for mask in [lags < 0, lags > 0]:
        if mask.any():
            ax.plot(lags[mask], y[mask], marker=marker, color=color, **kw)


def plot_lag_crp_sweep(sweep_results, param_grid, param_name="Parameter",
                       cmap_name="viridis"):
    param_grid = np.asarray(param_grid, dtype=float)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name)

    fig, ax = plt.subplots(figsize=(8, 5))
    for val, c in zip(param_grid, colors):
        lv, crp = compute_lag_crp(sweep_results[val]["recall_sims"], N)
        _plot_split_lags(ax, lv, crp, color=c)
    ax.axvline(0, color="gray", ls="--", alpha=0.5)
    ax.set_title(f"Lag-CRP across {param_name}")
    ax.set_xlabel("Lag (next − current)"); ax.set_ylabel("CRP")
    ax.grid(alpha=0.3)
    _add_colorbar(fig, ax, norm, cmap, param_name)
    fig.tight_layout(); plt.show()


# -- Directional conditional lag rates (forward + backward) -----------

def _sweep_directional_rates(sweep_results, param_grid, N,
                             large_lag_thresh=4):
    """Return (fwd_p1, fwd_large, bwd_m1, bwd_large) arrays."""
    param_grid = np.asarray(param_grid, dtype=float)
    n = len(param_grid)
    fp1 = np.full(n, np.nan); fl = np.full(n, np.nan)
    bm1 = np.full(n, np.nan); bl = np.full(n, np.nan)

    for i, v in enumerate(param_grid):
        rs = _get_sweep_entry(sweep_results, v)["recall_sims"]
        lags, crp, num, den = lag_crp_with_counts(rs, N)
        fp1[i], fl[i] = conditional_forward_lag_rates(lags, num, den,
                                                       large_lag_thresh)
        bm1[i], bl[i] = conditional_backward_lag_rates(lags, num, den,
                                                        large_lag_thresh)
    return fp1, fl, bm1, bl


def plot_directional_lag_rates(sweep_results, param_grid, param_name,
                               N=N, large_lag_thresh=4,
                               cmap_name="viridis"):
    """4-panel: forward P(ℓ=+1), P(ℓ≥k) | backward P(ℓ=−1), P(ℓ≤−k)."""
    param_grid = np.asarray(param_grid, dtype=float)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name)
    fp1, fl, bm1, bl = _sweep_directional_rates(
        sweep_results, param_grid, N, large_lag_thresh)

    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    titles = [
        (axs[0, 0], fp1, r"$P(\ell=+1)$ (conditional fwd)"),
        (axs[0, 1], fl,  rf"$P(\ell\geq{large_lag_thresh})$ (conditional fwd)"),
        (axs[1, 0], bm1, r"$P(\ell=-1)$ (conditional bwd)"),
        (axs[1, 1], bl,  rf"$P(\ell\leq-{large_lag_thresh})$ (conditional bwd)"),
    ]
    for ax, y, title in titles:
        line_with_colored_points(ax, param_grid, y, colors)
        ax.set_title(f"{title} vs {param_name}")
        ax.set_xlabel(param_name); ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 0.92, 1])
    cax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax).set_label(param_name)
    plt.show()


# -- Unconditional |lag| summaries ------------------------------------

def _sweep_unconditional_abs(sweep_results, param_grid, large_lag_thresh=4):
    param_grid = np.asarray(param_grid, dtype=float)
    n = len(param_grid)
    pa1 = np.full(n, np.nan); pak = np.full(n, np.nan); ma = np.full(n, np.nan)
    for i, v in enumerate(param_grid):
        rs = _get_sweep_entry(sweep_results, v)["recall_sims"]
        pa1[i], pak[i], ma[i] = unconditional_transition_summaries(
            rs, large_lag_thresh)
    return pa1, pak, ma


def plot_unconditional_lag_summaries(sweep_results, param_grid, param_name,
                                     large_lag_thresh=4, cmap_name="viridis"):
    """3-panel: P(|ℓ|=1), P(|ℓ|≥k), E[|ℓ|]."""
    param_grid = np.asarray(param_grid, dtype=float)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name)
    pa1, pak, ma = _sweep_unconditional_abs(
        sweep_results, param_grid, large_lag_thresh)

    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), sharex=True)
    for ax, y, lab in [
        (axs[0], pa1, r"$P(|\ell|=1)$"),
        (axs[1], pak, rf"$P(|\ell|\geq{large_lag_thresh})$"),
        (axs[2], ma,  r"$\mathbb{E}[|\ell|]$"),
    ]:
        line_with_colored_points(ax, param_grid, y, colors)
        ax.set_title(f"{lab} vs {param_name} (unconditional)")
        ax.set_xlabel(param_name); ax.set_ylabel(lab); ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 0.92, 1])
    cax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax).set_label(param_name)
    plt.show()


# -- all readouts in one call ------------------

def plot_all_behavioral_readouts(sweep_results, param_grid, param_name,
                                  N=N, large_lag_thresh=4,
                                  cmap_name="viridis"):
    """
    Produce the full mandatory behavioral readout suite:

    1. Recall accuracy
    2. SPC
    3. PFR heatmap
    4. Lag-CRP (split ℓ < 0 / ℓ > 0)
    5. Conditional forward & backward lag rates (4-panel)
    6. Unconditional |ℓ| summaries (3-panel)
    """
    plot_recall_accuracy(sweep_results, param_grid, param_name, cmap_name)
    plot_spc_sweep(sweep_results, param_grid, param_name, cmap_name)
    plot_pfr_heatmap(sweep_results, param_grid, param_name, cmap_name)
    plot_lag_crp_sweep(sweep_results, param_grid, param_name, cmap_name)
    plot_directional_lag_rates(sweep_results, param_grid, param_name,
                                N=N, large_lag_thresh=large_lag_thresh,
                                cmap_name=cmap_name)
    plot_unconditional_lag_summaries(sweep_results, param_grid, param_name,
                                     large_lag_thresh=large_lag_thresh,
                                     cmap_name=cmap_name)


# =====================================================================
#  2. RECALL-STAGE DIAGNOSTIC PLOTS
# =====================================================================

# -- lag-CRP diagnostics (CRP + num + den) ----------------------------

def plot_lag_crp_diagnostics(sweep_results, param_grid,
                             param_name=r"$B_{rec}$", cmap_name="viridis"):
    param_grid = np.asarray(param_grid, dtype=float)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name)

    fig, axs = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for val, c in zip(param_grid, colors):
        rs = sweep_results[val]["recall_sims"]
        lags, crp, num, den = lag_crp_with_counts(rs, N)
        _plot_split_lags(axs[0], lags, crp, c)
        _plot_split_lags(axs[1], lags, den, c)
        _plot_split_lags(axs[2], lags, num, c)

    axs[0].axvline(0, color="gray", ls="--", alpha=0.5)
    axs[0].set_ylabel("CRP"); axs[0].set_title(f"Lag-CRP across {param_name}")
    axs[1].set_ylabel("den (opportunities)"); axs[1].set_title("Opportunities per lag")
    axs[2].set_ylabel("num (transitions)"); axs[2].set_title("Observed transitions per lag")
    axs[2].set_xlabel("Lag (next − current)")
    for a in axs: a.grid(alpha=0.3)

    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax).set_label(param_name)
    fig.tight_layout(rect=[0, 0, 0.9, 1]); plt.show()


# -- Neighbor item-evidence asymmetry during selection (Δf) -----------

def plot_item_evidence_asymmetry_paired(
        sweep_results, param_grid, param_name="Parameter",
        cmap_name="viridis"):
    """Side-by-side: (L) asymmetry by serial position, (R) pooled vs parameter."""
    param_grid = np.asarray(param_grid, dtype=float)
    positions = np.arange(1, N + 1)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name)

    fig, (ax_pos, ax_pool) = plt.subplots(
        1, 2, figsize=(13, 4.5), constrained_layout=True)

    for pv, c in zip(param_grid, colors):
        d = sweep_results[pv]["item_evidence_diag"]
        means = [np.mean(d["deltas_by_pos"].get(p, [np.nan])) for p in positions]
        ax_pos.plot(positions, means, marker="o", color=c, ms=4)
    ax_pos.axhline(0, ls="--", alpha=0.6)
    ax_pos.set_title(f"Neighbor item-evidence asymmetry\nduring selection across {param_name}")
    ax_pos.set_xlabel(r"current serial position $i$")
    ax_pos.set_ylabel(r"mean $\Delta f = f_{in}(i\!+\!1) - f_{in}(i\!-\!1)$")
    ax_pos.grid(alpha=0.3)

    adv = np.array([
        np.mean(sweep_results[v]["item_evidence_diag"]["deltas_all"])
        if sweep_results[v]["item_evidence_diag"]["deltas_all"] else np.nan
        for v in param_grid], dtype=float)
    ax_pool.plot(param_grid, adv, lw=1, alpha=0.5)
    ax_pool.scatter(param_grid, adv, c=param_grid, cmap=cmap, norm=norm, s=50)
    ax_pool.axhline(0, ls="--", alpha=0.6)
    ax_pool.set_title(f"Pooled item-evidence asymmetry vs {param_name}")
    ax_pool.set_xlabel(param_name)
    ax_pool.set_ylabel(r"mean $\Delta f$")
    ax_pool.grid(alpha=0.3)

    _add_colorbar(fig, [ax_pos, ax_pool], norm, cmap, param_name)
    plt.show()


# -- FC-alignment asymmetry curves (Δ_FC) ----------------------------

def plot_fc_alignment_asymmetry_curves(ax, x, d_all, d_fwd, d_bwd,
                                       colors, title, xlabel):
    """Three-curve helper: overall / forward / backward Δ_FC."""
    x = np.asarray(x, dtype=float)
    l1, = ax.plot(x, d_all, lw=1.5, alpha=0.9,
                  label=r"$\mathbb{E}[\Delta_{FC}]$")
    l2, = ax.plot(x, d_fwd, lw=1.5, alpha=0.9,
                  label=r"$\mathbb{E}[\Delta_{FC}\mid \ell\!=\!+1]$")
    l3, = ax.plot(x, d_bwd, lw=1.5, alpha=0.9,
                  label=r"$\mathbb{E}[\Delta_{FC}\mid \ell\!=\!-1]$")
    for curve in [d_all, d_fwd, d_bwd]:
        ax.scatter(x, curve, c=colors, s=45, edgecolor="none")
    ax.axhline(0, ls="--", alpha=0.4)
    ax.set_title(title); ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\Delta_{FC}$ (context-to-item-input alignment)")
    ax.grid(alpha=0.3)
    ax.legend(handles=[l1, l2, l3], frameon=False, loc="upper right")


def plot_fc_alignment_asymmetry_sweep(
        sweep_results, param_grid, param_name="Parameter",
        cmap_name="viridis"):
    """Single-panel Δ_FC vs swept parameter."""
    param_grid = np.asarray(param_grid, dtype=float)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name)
    m_all, m_fwd, m_bwd = fc_alignment_asymmetry_means(
        sweep_results, param_grid)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_fc_alignment_asymmetry_curves(
        ax, param_grid, m_all, m_fwd, m_bwd, colors,
        title=rf"Neighbor context-to-item-input alignment ($\Delta_{{FC}}$) vs {param_name}",
        xlabel=param_name)
    _add_colorbar(fig, ax, norm, cmap, param_name)
    fig.tight_layout(); plt.show()


# -- Cosine similarity cos(c, c_in) after retrieval update ------------

def plot_cosine_similarity_sweep(sweep_results, param_grid,
                                  param_name="Parameter",
                                  cmap_name="viridis"):
    param_grid = np.asarray(param_grid, dtype=float)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name)

    fig, ax = plt.subplots(figsize=(8, 5))
    for val, c in zip(param_grid, colors):
        ts = get_trace_sims(sweep_results, val)
        x, ca = mean_curve_over_sims(ts, "cos_after")
        if x.size:
            ax.plot(x, ca, marker="o", color=c, ms=4)
    ax.axhline(1, ls="--", alpha=0.6)
    ax.set_title(rf"$\cos(c,c_{{in}})$ after retrieval update across {param_name}")
    ax.set_xlabel("recall step"); ax.set_ylabel(r"$\cos(c, c_{in})$")
    ax.grid(alpha=0.3)
    _add_colorbar(fig, ax, norm, cmap, param_name)
    fig.tight_layout(); plt.show()


# -- Mean evidence by serial position (raw + centered) ----------------

def plot_mean_evidence_by_pos(sweep_results, param_grid,
                               param_name="Parameter",
                               cmap_name="viridis"):
    param_grid = np.asarray(param_grid, dtype=float)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name)
    sp = np.arange(1, N + 1)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for val, c in zip(param_grid, colors):
        ts = get_trace_sims(sweep_results, val)
        axs[0].plot(sp, mean_vector_over_sims(ts, "f_mean_by_pos"),
                    marker="o", color=c, ms=4)
        axs[1].plot(sp, mean_vector_over_sims(ts, "f_mean_centered_by_pos"),
                    marker="o", color=c, ms=4)
    axs[0].set_title(rf"Mean evidence (raw) across {param_name}")
    axs[0].set_xlabel("Serial position $i$")
    axs[0].set_ylabel(r"$\mathbb{E}[f_{in}(i)]$"); axs[0].grid(alpha=0.3)
    axs[1].set_title(rf"Mean evidence (centered) across {param_name}")
    axs[1].set_xlabel("Serial position $i$")
    axs[1].set_ylabel(r"$\mathbb{E}[\tilde{f}_{in}(i)]$"); axs[1].grid(alpha=0.3)

    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax).set_label(param_name)
    fig.tight_layout(rect=[0, 0, 0.91, 1]); plt.show()


# -- Generic scalar metric vs parameter (reusable) --------------------

def plot_scalar_metric_vs_param(sweep_results, param_grid, param_label,
                                 metric_fn, title, y_label,
                                 cmap_name="viridis"):
    param_grid = np.asarray(param_grid, dtype=float)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name)
    y = _sweep_scalar_metric_array(sweep_results, param_grid, metric_fn)

    fig, ax = plt.subplots(figsize=(7.2, 4))
    line_with_colored_points(ax, param_grid, y, colors)
    ax.set_title(title); ax.set_xlabel(param_label); ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    _add_colorbar(fig, ax, norm, cmap, param_label)
    fig.tight_layout(); plt.show()


# =====================================================================
#  3. ENCODING-STAGE DIAGNOSTIC PLOTS
# =====================================================================

def plot_matrix_band_profiles(sweep_results, param_grid, param_name,
                               matrix_key="net_w_fc", cmap_name="viridis"):
    """Band-strength profile (mean |weight| at each SP lag) across sweep."""
    param_grid = np.asarray(param_grid, dtype=float)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name)
    lags, profiles = sweep_band_profiles(sweep_results, param_grid,
                                          matrix_key=matrix_key)
    matrix_label = r"$M_{FC}$" if "fc" in matrix_key else r"$M_{CF}$"

    fig, ax = plt.subplots(figsize=(8, 5))
    for prof, c in zip(profiles, colors):
        _plot_split_lags(ax, lags, prof, color=c)
    ax.axvline(0, color="gray", ls="--", alpha=0.5)
    ax.set_title(f"{matrix_label} band-strength profile across {param_name}")
    ax.set_xlabel(r"serial-position lag $\delta$")
    ax.set_ylabel("mean |weight|"); ax.grid(alpha=0.3)
    _add_colorbar(fig, ax, norm, cmap, param_name)
    fig.tight_layout(); plt.show()


def plot_matrix_norms_sweep(sweep_results, param_grid, param_name,
                             matrix_key="net_w_fc", cmap_name="viridis"):
    """Frobenius norm and mean |weight| across sweep."""
    param_grid = np.asarray(param_grid, dtype=float)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name)
    frob, mabs = sweep_matrix_norms(sweep_results, param_grid, matrix_key)
    matrix_label = r"$M_{FC}$" if "fc" in matrix_key else r"$M_{CF}$"

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    line_with_colored_points(axs[0], param_grid, frob, colors)
    axs[0].set_title(rf"$\|{matrix_label}\|_F$ vs {param_name}")
    axs[0].set_ylabel("Frobenius norm"); axs[0].grid(alpha=0.3)

    line_with_colored_points(axs[1], param_grid, mabs, colors)
    axs[1].set_title(rf"mean $|w_{{ij}}|$ of {matrix_label} vs {param_name}")
    axs[1].set_ylabel("mean |weight|"); axs[1].grid(alpha=0.3)

    for a in axs: a.set_xlabel(param_name)
    fig.tight_layout()
    _add_colorbar(fig, axs, norm, cmap, param_name)
    plt.show()


def plot_neighbor_band_asymmetry_sweep(
        sweep_results, param_grid, param_name,
        matrix_key="net_w_fc", cmap_name="viridis"):
    """Forward vs backward neighbor-band mean + asymmetry across sweep."""
    param_grid = np.asarray(param_grid, dtype=float)
    colors, norm, cmap = make_sweep_colors(param_grid, cmap_name)
    fwd, bwd, asym = sweep_neighbor_band_asymmetry(
        sweep_results, param_grid, matrix_key=matrix_key)
    matrix_label = r"$M_{FC}$" if "fc" in matrix_key else r"$M_{CF}$"

    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    line_with_colored_points(axs[0], param_grid, fwd, colors)
    axs[0].set_title(rf"Fwd neighbor band ($\delta\!=\!+1$) of {matrix_label}")
    axs[0].set_ylabel("mean |weight|"); axs[0].grid(alpha=0.3)

    line_with_colored_points(axs[1], param_grid, bwd, colors)
    axs[1].set_title(rf"Bwd neighbor band ($\delta\!=\!-1$) of {matrix_label}")
    axs[1].set_ylabel("mean |weight|"); axs[1].grid(alpha=0.3)

    line_with_colored_points(axs[2], param_grid, asym, colors)
    axs[2].axhline(0, ls="--", alpha=0.5)
    axs[2].set_title(rf"Neighbor band asymmetry of {matrix_label}")
    axs[2].set_ylabel("fwd − bwd"); axs[2].grid(alpha=0.3)

    for a in axs: a.set_xlabel(param_name)
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    cax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax).set_label(param_name)
    plt.show()

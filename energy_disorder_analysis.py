#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Optional, used for p value if available
try:
    from scipy import stats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


INPUT_XLSX = "cluster_summary_wide_blends_as_columns.xlsx"
SHEET_NAME = None  # None means first sheet
OUTDIR = "pretty_panels_sci"


# -------------------------
# Scientific color scheme
# -------------------------
SCI_COLORS = [
    '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', 
    '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF',
    '#393B79', '#637939', '#8C6D31', '#843C39', '#7B4173',
    '#5254A3', '#6B6ECF', '#9C9EDE', '#AD494A', '#D6616B'
]

# -------------------------
# Data loading and parsing
# -------------------------
def load_wide_table(xlsx_path: str, sheet_name=None) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name if sheet_name else 0, header=0)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "metric"})
    df["metric"] = df["metric"].astype(str).str.strip()
    df = df.set_index("metric")
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def find_rows(index_list, include_patterns, exclude_patterns=None):
    exclude_patterns = exclude_patterns or []
    out = []
    for name in index_list:
        s = str(name)
        ok = True
        for p in include_patterns:
            if re.search(p, s, flags=re.IGNORECASE) is None:
                ok = False
                break
        if not ok:
            continue
        for p in exclude_patterns:
            if re.search(p, s, flags=re.IGNORECASE) is not None:
                ok = False
                break
        if ok:
            out.append(name)
    return out


def first_match(index_list, include_patterns, exclude_patterns=None):
    rows = find_rows(index_list, include_patterns, exclude_patterns)
    return rows[0] if rows else None


def series_from_row(df_wide: pd.DataFrame, row_name: str) -> pd.Series:
    s = df_wide.loc[row_name].copy()
    s.index = s.index.astype(str)
    return s


def mean_index(df_wide: pd.DataFrame, row_names: list[str]) -> pd.Series | None:
    if not row_names:
        return None
    sub = df_wide.loc[row_names]
    return sub.mean(axis=0, skipna=True)


# -------------------------
# Stats helpers
# -------------------------
def pearson_r_p(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3:
        return np.nan, np.nan, n
    r = float(np.corrcoef(x[m], y[m])[0, 1])
    if SCIPY_OK:
        rp = stats.pearsonr(x[m], y[m])
        return float(rp.statistic), float(rp.pvalue), n
    # Fallback, no p value without scipy
    return r, np.nan, n


def linfit_with_ci(x, y, n_boot=2000, seed=0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = x.size
    if n < 3:
        return None

    X = np.vstack([np.ones(n), x]).T
    beta = np.linalg.lstsq(X, y, rcond=None)[0]  # intercept, slope

    rng = np.random.default_rng(seed)
    betas = np.zeros((n_boot, 2), dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]
        yb = y[idx]
        betas[i] = np.linalg.lstsq(Xb, yb, rcond=None)[0]

    return beta, betas


def format_p(p):
    if not np.isfinite(p):
        return "p = N/A"
    if p < 0.0001:
        return "p < 0.0001"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


# -------------------------
# Plot styling - SCI standard
# -------------------------
def set_global_style():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.weight": "bold",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "axes.linewidth": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.title_fontsize": 10,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
    })


def create_blend_colormap(blends):
    """Create color mapping for each blend"""
    n_blends = len(blends)
    colors = SCI_COLORS * (n_blends // len(SCI_COLORS) + 1)
    return {blend: colors[i] for i, blend in enumerate(blends)}


def annotate_extremes(ax, x, y, labels, k=3, color_dict=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return
    xm = x[m]
    ym = y[m]
    labm = np.asarray(labels)[m]

    # Annotate extremes in y and x directions
    idx_y_hi = np.argsort(ym)[-k:]
    idx_y_lo = np.argsort(ym)[:k]
    idx_x_hi = np.argsort(xm)[-k:]
    idx_x_lo = np.argsort(xm)[:k]
    idx = np.unique(np.concatenate([idx_y_hi, idx_y_lo, idx_x_hi, idx_x_lo]))

    for i in idx:
        blend_name = str(labm[i])
        color = color_dict.get(blend_name, 'black') if color_dict else 'black'
        
        ax.annotate(
            blend_name,
            (xm[i], ym[i]),
            textcoords="offset points",
            xytext=(4, 4),
            ha="left",
            va="bottom",
            fontsize=8,
            fontweight='bold',
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor='none')
        )


def scatter_pretty(ax, x, y, labels, xlab, ylab, title, color_dict, show_ci=True, seed=0, marker_size=70):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    lab = np.asarray(labels)[m]

    # Set color for each point
    point_colors = [color_dict.get(str(l), '#1F77B4') for l in lab]
    
    # Plot scatter points with black borders
    ax.scatter(x, y, s=marker_size, c=point_colors, alpha=0.85, 
               edgecolor='black', linewidth=0.8, zorder=3)

    # Correlation annotation
    r, p, n = pearson_r_p(x, y)
    r_text = f"r = {r:.3f}"
    p_text = format_p(p)
    n_text = f"n = {n}"
    
    # Place statistics in top-left corner
    stats_text = f"{r_text}\n{p_text}\n{n_text}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='gray'))

    # Regression line and confidence interval
    fit = linfit_with_ci(x, y, n_boot=2000, seed=seed)
    if fit is not None:
        beta, betas = fit
        xs = np.linspace(np.min(x), np.max(x), 200)
        yhat = beta[0] + beta[1] * xs
        ax.plot(xs, yhat, color='black', linewidth=2.0, linestyle='-', zorder=2)

        if show_ci:
            yhats = betas[:, 0:1] + betas[:, 1:2] * xs.reshape(1, -1)
            lo = np.percentile(yhats, 2.5, axis=0)
            hi = np.percentile(yhats, 97.5, axis=0)
            ax.fill_between(xs, lo, hi, alpha=0.25, color='gray', zorder=1)

    # Annotate extreme values
    annotate_extremes(ax, x, y, lab, k=2, color_dict=color_dict)

    # Set axis labels (replace sigma with sigma)
    xlab = xlab.replace('sigma_', 'sigma_').replace('sigma ', 'sigma ')
    ylab = ylab.replace('sigma_', 'sigma_').replace('sigma ', 'sigma ')
    
    ax.set_xlabel(xlab, fontweight='bold')
    ax.set_ylabel(ylab, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=12)
    
    # Add grid
    ax.grid(True, linewidth=0.5, alpha=0.3, zorder=0)
    
    # Set border thickness
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


# -------------------------
# Main workflow
# -------------------------
def main():
    set_global_style()
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)

    dfw = load_wide_table(INPUT_XLSX, SHEET_NAME)
    metrics = list(dfw.index)
    blends = dfw.columns.astype(str).tolist()
    
    # Create color mapping for all blends
    color_dict = create_blend_colormap(blends)

    # Core rows
    entropy_row = first_match(metrics, [r"Entropy", r"mean"], [r"std"]) or first_match(metrics, [r"Entropy"], [])
    sigma_etotal_row = first_match(metrics, [r"sigma", r"Etotal"], []) or first_match(metrics, [r"proxy", r"sigma", r"Etotal"], [])
    sigma_pot_row = first_match(metrics, [r"sigma", r"Potential"], []) or first_match(metrics, [r"roughness", r"sigma", r"Potential"], [])

    if entropy_row is None or sigma_etotal_row is None:
        raise ValueError("Could not find Entropy or sigma_Etotal rows. Please check row names in the wide table.")

    entropy = series_from_row(dfw, entropy_row)
    sigma_etotal = series_from_row(dfw, sigma_etotal_row)
    sigma_pot = series_from_row(dfw, sigma_pot_row) if sigma_pot_row else None

    # RDF fluctuation indices
    rdf_all_rows = find_rows(metrics, [r"g_first", r"__relstd"], [])
    rdf_DA_rows = find_rows(metrics, [r"\bDA\b|DA", r"g_first", r"__relstd"], [])
    rdf_AA_rows = find_rows(metrics, [r"A-A", r"g_first", r"__relstd"], [])

    rdf_all = mean_index(dfw, rdf_all_rows)
    rdf_DA = mean_index(dfw, rdf_DA_rows)
    rdf_AA = mean_index(dfw, rdf_AA_rows)

    # Energy term std rows to sigma_Etotal
    term_patterns = {
        "Coulomb-(SR)__std": [r"Coulomb-\(SR\)", r"__std"],
        "Coul.-recip.__std": [r"Coul\.-recip\.", r"__std"],
        "Coulomb-14__std": [r"Coulomb-14", r"__std"],
        "LJ-(SR)__std": [r"LJ-\(SR\)", r"__std"],
        "LJ-14__std": [r"LJ-14", r"__std"],
        "Disper.-corr.__std": [r"Disper\.-corr\.", r"__std"],
    }

    term_series = {}
    for key, pats in term_patterns.items():
        row = first_match(metrics, pats, [])
        if row is not None:
            term_series[key] = series_from_row(dfw, row)

    # -------------------------
    # Figure 1: Entropy chain panels
    # -------------------------
    fig1, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    if rdf_DA is not None:
        scatter_pretty(
            axes[0],
            entropy.values, rdf_DA.values, blends,
            xlab="Entropy (cluster level)",
            ylab="RDF fluctuation index (DA)",
            title="Entropy vs RDF fluctuations (DA)",
            color_dict=color_dict,
            show_ci=True,
            seed=1,
        )
    else:
        axes[0].axis("off")
        axes[0].set_title("DA RDF fluctuation index not found")

    scatter_pretty(
        axes[1],
        entropy.values, sigma_etotal.values, blends,
        xlab="Entropy (cluster level)",
        ylab="sigma_Etotal (energy disorder)",
        title="Entropy vs sigma_Etotal",
        color_dict=color_dict,
        show_ci=True,
        seed=2,
    )

    if sigma_pot is not None:
        scatter_pretty(
            axes[2],
            entropy.values, sigma_pot.values, blends,
            xlab="Entropy (cluster level)",
            ylab="sigma_Potential (surface roughness)",
            title="Entropy vs sigma_Potential",
            color_dict=color_dict,
            show_ci=True,
            seed=3,
        )
    else:
        axes[2].axis("off")
        axes[2].set_title("sigma_Potential not found")

    fig1.savefig(os.path.join(OUTDIR, "panel1_entropy_chain.png"), bbox_inches='tight', dpi=600)
    fig1.savefig(os.path.join(OUTDIR, "panel1_entropy_chain.pdf"), bbox_inches='tight')
    plt.close(fig1)

    # -------------------------
    # Figure 2: Structure to energy panels
    # -------------------------
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    if rdf_all is not None:
        scatter_pretty(
            axes[0],
            rdf_all.values, sigma_etotal.values, blends,
            xlab="RDF fluctuation index (all pairs)",
            ylab="sigma_Etotal (energy disorder)",
            title="RDF fluctuations (all) vs sigma_Etotal",
            color_dict=color_dict,
            show_ci=True,
            seed=4,
        )
    else:
        axes[0].axis("off")
        axes[0].set_title("All RDF fluctuation index not found")

    if rdf_DA is not None:
        scatter_pretty(
            axes[1],
            rdf_DA.values, sigma_etotal.values, blends,
            xlab="RDF fluctuation index (DA)",
            ylab="sigma_Etotal (energy disorder)",
            title="RDF fluctuations (DA) vs sigma_Etotal",
            color_dict=color_dict,
            show_ci=True,
            seed=5,
        )
    else:
        axes[1].axis("off")
        axes[1].set_title("DA RDF fluctuation index not found")

    if sigma_pot is not None:
        scatter_pretty(
            axes[2],
            sigma_pot.values, sigma_etotal.values, blends,
            xlab="sigma_Potential (surface roughness)",
            ylab="sigma_Etotal (energy disorder)",
            title="sigma_Potential vs sigma_Etotal",
            color_dict=color_dict,
            show_ci=True,
            seed=6,
        )
    else:
        axes[2].axis("off")
        axes[2].set_title("sigma_Potential not found")

    fig2.savefig(os.path.join(OUTDIR, "panel2_structure_to_energy.png"), bbox_inches='tight', dpi=600)
    fig2.savefig(os.path.join(OUTDIR, "panel2_structure_to_energy.pdf"), bbox_inches='tight')
    plt.close(fig2)

    # -------------------------
    # Figure 3: Energy decomposition panels
    # -------------------------
    # Select up to 6 terms, layout 2x3
    keys = list(term_series.keys())
    if len(keys) > 0:
        n_show = min(6, len(keys))
        fig3, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        axes = axes.flatten()

        for i in range(6):
            if i >= n_show:
                axes[i].axis("off")
                continue
            k = keys[i]
            s = term_series[k]
            
            # Clean up key name for display
            clean_k = k.replace('__std', ' std')
            if 'Coulomb' in clean_k:
                clean_k = clean_k.replace('Coulomb', 'Coul')
            
            scatter_pretty(
                axes[i],
                s.values, sigma_etotal.values, blends,
                xlab=clean_k,
                ylab="sigma_Etotal (energy disorder)",
                title=f"{clean_k} vs sigma_Etotal",
                color_dict=color_dict,
                show_ci=True,
                seed=10 + i,
                marker_size=60,
            )

        fig3.savefig(os.path.join(OUTDIR, "panel3_energy_decomposition.png"), bbox_inches='tight', dpi=600)
        fig3.savefig(os.path.join(OUTDIR, "panel3_energy_decomposition.pdf"), bbox_inches='tight')
        plt.close(fig3)

    # Optional: compact panel for AA vs DA RDF, if AA exists
    if rdf_AA is not None and rdf_DA is not None:
        fig4, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        scatter_pretty(
            axes[0],
            entropy.values, rdf_DA.values, blends,
            xlab="Entropy (cluster level)",
            ylab="RDF fluctuation index (DA)",
            title="Entropy vs RDF fluctuations (DA)",
            color_dict=color_dict,
            show_ci=True,
            seed=21,
        )
        scatter_pretty(
            axes[1],
            entropy.values, rdf_AA.values, blends,
            xlab="Entropy (cluster level)",
            ylab="RDF fluctuation index (A-A)",
            title="Entropy vs RDF fluctuations (A-A)",
            color_dict=color_dict,
            show_ci=True,
            seed=22,
        )
        fig4.savefig(os.path.join(OUTDIR, "supp_DA_vs_AA_selectivity.png"), bbox_inches='tight', dpi=600)
        fig4.savefig(os.path.join(OUTDIR, "supp_DA_vs_AA_selectivity.pdf"), bbox_inches='tight')
        plt.close(fig4)

    # Create legend figure
    fig_legend, ax_legend = plt.subplots(figsize=(4, 6))
    ax_legend.axis('off')
    
    # Create legend
    legend_elements = []
    for i, blend in enumerate(blends):
        if i < 20:  # Show at most 20 blends
            color = color_dict.get(blend, 'gray')
            legend_elements.append(
                mpl.lines.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=color, markeredgecolor='black',
                                markersize=10, label=blend, markeredgewidth=0.8)
            )
    
    ax_legend.legend(handles=legend_elements, title="Blends", loc='center', 
                    frameon=True, fancybox=True, framealpha=0.9,
                    title_fontsize=11, fontsize=9)
    
    fig_legend.savefig(os.path.join(OUTDIR, "color_legend.png"), bbox_inches='tight', dpi=300)
    fig_legend.savefig(os.path.join(OUTDIR, "color_legend.pdf"), bbox_inches='tight')
    plt.close(fig_legend)

    print("=" * 60)
    print("SCI standard figures generated successfully!")
    print(f"Input file: {INPUT_XLSX}")
    print(f"Output directory: {OUTDIR}")
    print("\nMain output files:")
    print("  panel1_entropy_chain.png/pdf    - Entropy-related relationships")
    print("  panel2_structure_to_energy.png/pdf - Structure-to-energy relationships")
    print("  panel3_energy_decomposition.png/pdf - Energy term decomposition")
    print("  color_legend.png/pdf            - Color legend")
    print("\nEnhancements applied:")
    print("  - Unique color for each cluster")
    print("  - Arial font with bold weight")
    print("  - sigma symbol instead of sigma")
    print("  - Statistical information displayed")
    print("  - Regression line with 95% confidence interval")
    print("  - Extreme value annotation")
    print("=" * 60)


if __name__ == "__main__":
    main()
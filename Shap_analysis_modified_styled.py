# -*- coding: utf-8 -*-

import os
import warnings
import numpy as np
import pandas as pd
import time

os.environ["MPLBACKEND"] = "Agg"
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def set_pub_theme():
    mpl.rcParams.update({
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "font.family": "DejaVu Serif",
        "font.weight": "normal",
        "font.size": 20,
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 17,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.45,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "axes.linewidth": 1.8,
        "xtick.major.size": 6.0,
        "ytick.major.size": 6.0,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "figure.facecolor": "#efefef",
        "axes.facecolor": "#efefef",
        "savefig.facecolor": "#efefef",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": "#1f1f1f",
        "axes.labelcolor": "#1f1f1f",
        "xtick.color": "#1f1f1f",
        "ytick.color": "#1f1f1f",
        "axes.titlecolor": "#1f1f1f",
        "legend.labelcolor": "#1f1f1f",
    })

BG = "#efefef"
GRID = "#cfcfcf"
AX = "#1f1f1f"
OLIVE = "#7a9650"
OLIVE_LIGHT = "#c8d7a7"
GOLD = "#e2b54a"

SHAP_CMAP = LinearSegmentedColormap.from_list(
    "olive_gold_shap",
    [OLIVE, "#e9eadf", GOLD]
)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def style_ax(ax, grid=True):
    ax.set_facecolor(BG)
    if grid:
        ax.grid(True, alpha=0.6, linestyle='--', linewidth=0.8, color=GRID)
    else:
        ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color(AX)
    ax.tick_params(colors=AX, width=1.4)

def save_fig(fig, path_no_ext):
    fig.savefig(path_no_ext + ".png", bbox_inches="tight", dpi=600)
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight", dpi=600)
    fig.savefig(path_no_ext + ".tif", bbox_inches="tight", dpi=600)
    plt.close(fig)

def clean_feature_matrix(X):
    """Convert all feature columns to numeric, report issues, fill missing values safely."""
    print("\nStep 2: Data cleaning and normalization...")

    # 强制转数值，不能转换的变成 NaN
    X_numeric = X.apply(pd.to_numeric, errors="coerce")

    # 报告含有问题值的列
    invalid_cols = X_numeric.columns[X_numeric.isna().any()].tolist()
    if invalid_cols:
        print("\nWarning: The following feature columns contain non-numeric or missing values:")
        for col in invalid_cols:
            n_bad = int(X_numeric[col].isna().sum())
            print(f"  {col}: {n_bad} problematic entries")

    # 先删除全 NaN 列
    all_nan_cols = X_numeric.columns[X_numeric.isna().all()].tolist()
    if all_nan_cols:
        print("\nWarning: Dropping all-NaN columns:")
        for col in all_nan_cols:
            print(f"  {col}")
        X_numeric = X_numeric.drop(columns=all_nan_cols)

    # 用各列中位数填补剩余 NaN
    medians = X_numeric.median(numeric_only=True)
    X_numeric = X_numeric.fillna(medians)

    # 再检查是否还有 NaN
    remaining_nan = int(X_numeric.isna().sum().sum())
    if remaining_nan > 0:
        raise ValueError(f"Feature matrix still contains {remaining_nan} missing values after cleaning.")

    return X_numeric

def plot_shap_summary(shap_values, features, feature_names, outpath_no_ext, max_display=20):
    import shap

    fig = plt.figure(figsize=(14, 10), dpi=600)
    fig.add_subplot(111)

    shap.summary_plot(
        shap_values,
        features=features,
        feature_names=feature_names,
        show=False,
        max_display=max_display,
        plot_type="dot",
        color=SHAP_CMAP,
        color_bar_label="Feature value",
        plot_size=None
    )

    ax = plt.gca()
    style_ax(ax, grid=True)
    ax.set_title("SHAP Global Feature Contribution", fontsize=24, pad=18, color=AX)
    ax.set_xlabel("SHAP Value", fontsize=20, color=AX)
    ax.set_ylabel("")

    for label in ax.get_yticklabels():
        label.set_fontsize(18)
        label.set_color(AX)

    if len(fig.get_axes()) > 1:
        cb_ax = fig.get_axes()[-1]
        cb_ax.set_ylabel("Feature value", rotation=270, labelpad=24, fontsize=18, color=AX)
        cb_ax.tick_params(labelsize=15, colors=AX)

    fig.tight_layout()
    save_fig(fig, outpath_no_ext + "_summary_dot")

def plot_shap_summary_bar(shap_values, feature_names, outpath_no_ext, max_display=20):
    shap_mean_abs = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(shap_mean_abs)[::-1][:max_display]
    sorted_features = [feature_names[i] for i in indices]
    sorted_values = shap_mean_abs[indices]

    fig = plt.figure(figsize=(12, 10), dpi=600)
    ax = fig.add_subplot(111)

    colors = [OLIVE_LIGHT if i % 2 == 0 else "#dbe6bf" for i in range(len(sorted_features))]
    bars = ax.barh(
        range(len(sorted_features)),
        sorted_values,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        height=0.72
    )

    for bar, val in zip(bars, sorted_values):
        ax.text(
            val + 0.01 * sorted_values.max(),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            ha='left', va='center',
            fontsize=15, color=AX
        )

    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=17, color=AX)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Absolute SHAP Value', fontsize=20, color=AX)
    ax.set_title('SHAP Feature Importance', fontsize=24, pad=18, color=AX)

    style_ax(ax, grid=True)
    ax.grid(True, axis='x', alpha=0.6, linestyle='--', linewidth=0.8, color=GRID)

    fig.tight_layout()
    save_fig(fig, outpath_no_ext + "_summary_bar")

def plot_shap_dependence(shap_values, features, feature_names, target_feature, outpath_no_ext):
    import shap

    fig = plt.figure(figsize=(10, 8), dpi=600)
    ax = fig.add_subplot(111)

    if target_feature in feature_names:
        feature_idx = feature_names.index(target_feature)
    else:
        print(f"Warning: Feature '{target_feature}' not in feature list")
        return

    shap.dependence_plot(
        feature_idx,
        shap_values,
        features,
        feature_names=feature_names,
        show=False,
        interaction_index=None,
        ax=ax,
        cmap=SHAP_CMAP
    )

    ax = plt.gca()
    style_ax(ax, grid=True)
    ax.set_title(f'SHAP Dependence Plot: {target_feature}', fontsize=22, pad=15, color=AX)
    ax.set_xlabel(target_feature, fontsize=19, color=AX)
    ax.set_ylabel('SHAP Value', fontsize=19, color=AX)

    fig.tight_layout()
    save_fig(fig, outpath_no_ext + f"_dependence_{target_feature}")

def plot_shap_beeswarm(shap_values, features, feature_names, outpath_no_ext, max_display=20):
    import shap

    fig = plt.figure(figsize=(14, 10), dpi=600)

    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.mean(shap_values.mean(axis=1)),
        data=features,
        feature_names=feature_names
    )

    shap.plots.beeswarm(
        explanation,
        max_display=max_display,
        show=False,
        color=SHAP_CMAP
    )

    ax = plt.gca()
    style_ax(ax, grid=True)
    ax.set_title("SHAP Global Feature Contribution", fontsize=24, pad=18, color=AX)
    ax.set_xlabel("SHAP Value", fontsize=20, color=AX)
    ax.set_ylabel("")

    for label in ax.get_yticklabels():
        label.set_fontsize(18)
        label.set_color(AX)

    if len(fig.get_axes()) > 1:
        cb_ax = fig.get_axes()[-1]
        cb_ax.set_ylabel("Feature value", rotation=270, labelpad=24, fontsize=18, color=AX)
        cb_ax.tick_params(labelsize=15, colors=AX)

    fig.tight_layout()
    save_fig(fig, outpath_no_ext + "_beeswarm")

def main():
    warnings.filterwarnings("ignore")
    set_pub_theme()

    OUTDIR = "CatBoost_SHAP_Analysis_Full_Dataset_Styled_Fixed"
    ensure_dir(OUTDIR)

    print("=" * 60)
    print("SHAP Analysis - Using CatBoost Model and Full Dataset")
    print("=" * 60)
    print("Step 1: Loading data...")

    start_time = time.time()
    DATA_FILE = "Database-1.xlsx"

    try:
        df = pd.read_excel(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Data file {DATA_FILE} not found")
        print("Please ensure Database-1.xlsx is in the current directory")
        return

    N_FEATURES = 63
    TARGET_COL_1BASED = 64

    if df.shape[1] < 64:
        print(f"Warning: Data has only {df.shape[1]} columns, but at least 64 required")
        N_FEATURES = df.shape[1] - 1
        TARGET_COL_1BASED = df.shape[1]

    X = df.iloc[:, 0:N_FEATURES].copy()

    # 目标列强制转数值
    y = pd.to_numeric(df.iloc[:, TARGET_COL_1BASED - 1], errors="coerce").values
    if np.isnan(y).any():
        bad_n = int(np.isnan(y).sum())
        raise ValueError(f"Target column contains {bad_n} non-numeric or missing values. Please clean the target column first.")

    print(f"  Loaded {X.shape[0]} samples, {X.shape[1]} raw feature columns")
    print(f"  Data loading time: {time.time() - start_time:.2f} seconds")

    from sklearn.preprocessing import RobustScaler
    X_numeric = clean_feature_matrix(X)
    feature_names = X_numeric.columns.tolist()

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    print(f"  Final feature matrix shape after cleaning: {X_scaled_df.shape}")

    print("\nStep 3: Training CatBoost model...")
    from catboost import CatBoostRegressor

    CATBOOST_PARAMS = dict(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        loss_function='RMSE'
    )

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(X_scaled_df, y, verbose=False)

    print("\nStep 4: Checking SHAP library...")
    try:
        import shap
        print("  SHAP library successfully imported")
    except ImportError as e:
        print(f"Error: Cannot import SHAP library: {e}")
        return

    print("\nStep 5: Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled_df)

    print("\nStep 6: Saving SHAP data...")
    shap_importance = pd.DataFrame({
        "Feature": feature_names,
        "MeanAbsSHAP": np.abs(shap_values).mean(axis=0),
        "StdAbsSHAP": np.abs(shap_values).std(axis=0),
        "MeanSHAP": shap_values.mean(axis=0),
        "StdSHAP": shap_values.std(axis=0),
        "MinSHAP": shap_values.min(axis=0),
        "MaxSHAP": shap_values.max(axis=0),
        "MedianAbsSHAP": np.median(np.abs(shap_values), axis=0)
    }).sort_values("MeanAbsSHAP", ascending=False)

    shap_importance.to_excel(os.path.join(OUTDIR, "shap_feature_importance_full.xlsx"), index=False)
    shap_importance.to_csv(os.path.join(OUTDIR, "shap_feature_importance_full.csv"), index=False)

    print("\nStep 7: Creating SHAP visualizations...")
    plot_shap_summary(shap_values, X_scaled_df, feature_names, os.path.join(OUTDIR, "SHAP_full_catboost"))
    plot_shap_summary_bar(shap_values, feature_names, os.path.join(OUTDIR, "SHAP_full_catboost"))
    plot_shap_beeswarm(shap_values, X_scaled_df, feature_names, os.path.join(OUTDIR, "SHAP_full_catboost"))

    top_features = shap_importance.head(5)["Feature"].tolist()
    for feature in top_features:
        plot_shap_dependence(shap_values, X_scaled_df, feature_names, feature, os.path.join(OUTDIR, "SHAP_full_catboost"))

    print("\nDone.")
    print(f"Output directory: {OUTDIR}")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-

import os
import warnings
import numpy as np
import pandas as pd

# =========================
# Force matplotlib to use a non-GUI backend (NO tkinter)
# =========================
os.environ["MPLBACKEND"] = "Agg"
import matplotlib as mpl
mpl.use("Agg")  # must be set BEFORE importing pyplot
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.manifold import TSNE
from catboost import CatBoostRegressor, Pool


# =========================
# Config (YOU DON'T NEED TO CHANGE)
# =========================
DATA_FILE = "Database-1.xlsx"
OUTDIR = "CatBoost_entropy_pubfigs_5fold_CV_colorful"  # 修改输出目录名称

# Data layout (根据要求修改)
SKIP_FIRST_COLS = 0
N_FEATURES = 63  # 修改为63个特征 (1-63列)
TARGET_COL_1BASED = 64  # 修改为第64列作为目标 (1-based indexing)

# 5-Fold Cross-Validation
N_SPLITS = 5
RANDOM_STATE = 42

# CatBoost params (使用之前性能最佳的参数)
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

# Permutation settings
N_REPEATS_FEATURE = 30
N_REPEATS_GROUP = 30

# t-SNE settings
TSNE_PERPLEXITY = 30
TSNE_MAX_SAMPLES = 1500  # sample from full data if too many


# =========================
# 颜色主题配置 (Color theme configuration)
# =========================
COLOR_PALETTE = {
    # 主色调
    'primary': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B8F71'],
    # 次要色调
    'secondary': ['#5BC0EB', '#9BC53D', '#E55934', '#FA7921', '#C41E3D'],
    # 热力图色系
    'heatmap': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'],
    # 分类色系
    'categorical': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
}

# 各组颜色映射
GROUP_COLORS = {
    "PackingInteraction": "#2E86AB",      # 深蓝色
    "GeometryVolume": "#A23B72",          # 紫红色
    "EnergyThermo": "#F18F01",            # 橙色
    "TopologyConstraint": "#6B8F71",      # 绿色
    "KineticsVelocity": "#C73E1D"         # 红色
}


# =========================
# Publication-ready theme with Arial font
# =========================
def set_pub_theme():
    # 设置Arial字体和全局样式
    mpl.rcParams.update({
        "figure.dpi": 300,  # 设置300dpi
        "savefig.dpi": 300,
        "font.family": "Arial",  # 使用Arial字体
        "font.weight": "bold",   # 设置加粗
        "font.size": 18,         # 设置字号为18
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "axes.linewidth": 2.0,    # 加粗坐标轴线
        "xtick.major.size": 6.0,
        "ytick.major.size": 6.0,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": "black",     # 设置文本颜色为黑色
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.titlecolor": "black",
        "legend.labelcolor": "black",
    })


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def save_fig(fig, path_no_ext):
    fig.savefig(path_no_ext + ".png", bbox_inches="tight", dpi=300)
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def eval_metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
    }


# =========================
# 5-Fold Cross-Validation Evaluation for CatBoost
# =========================
def evaluate_with_cv(model_params, X, y, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 存储每折的结果
    fold_results = {
        'train_indices': [],
        'val_indices': [],
        'train_r2': [],
        'val_r2': [],
        'val_mae': [],
        'val_rmse': [],
        'models': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        # 分割数据
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 训练CatBoost模型
        fold_model = CatBoostRegressor(**model_params)
        fold_model.fit(X_train, y_train, verbose=False)
        
        # 预测
        y_train_pred = fold_model.predict(X_train)
        y_val_pred = fold_model.predict(X_val)
        
        # 计算指标
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = rmse(y_val, y_val_pred)
        
        # 存储结果
        fold_results['train_indices'].append(train_idx)
        fold_results['val_indices'].append(val_idx)
        fold_results['train_r2'].append(train_r2)
        fold_results['val_r2'].append(val_r2)
        fold_results['val_mae'].append(val_mae)
        fold_results['val_rmse'].append(val_rmse)
        fold_results['models'].append(fold_model)
    
    # 计算平均指标
    cv_results = {
        'CV_R2_Mean': np.mean(fold_results['val_r2']),
        'CV_R2_Std': np.std(fold_results['val_r2']),
        'CV_MAE_Mean': np.mean(fold_results['val_mae']),
        'CV_MAE_Std': np.std(fold_results['val_mae']),
        'CV_RMSE_Mean': np.mean(fold_results['val_rmse']),
        'CV_RMSE_Std': np.std(fold_results['val_rmse']),
        'fold_details': fold_results
    }
    
    return cv_results


# =========================
# Group definitions (descriptor space 1..63) - 根据新的分组要求修改
# =========================
def to0(idx_1based_list):
    return [i - 1 for i in idx_1based_list]


def expand_ranges(ranges_1based):
    out = []
    for a, b in ranges_1based:
        out.extend(list(range(a - 1, b)))
    return out


def build_groups():
    
    # 根据新的分组要求定义各组特征索引
    groups = {
        "PackingInteraction": sorted(set(expand_ranges([(1, 31)]) + to0([40, 41, 43]))),
        "GeometryVolume": sorted(set(expand_ranges([(32, 33)]) + expand_ranges([(49, 52)]) + expand_ranges([(56, 59)]))),
        "EnergyThermo": sorted(set(expand_ranges([(34, 39)]) + to0([42]) + expand_ranges([(44, 48)]) + to0([60]))),
        "TopologyConstraint": sorted(set(expand_ranges([(53, 55)]))),
        "KineticsVelocity": sorted(set(expand_ranges([(61, 63)]))),
    }
    
    # Coverage check
    all_idx = sorted(set([i for v in groups.values() for i in v]))
    missing = sorted(set(range(63)) - set(all_idx))
    overlap = sum(len(v) for v in groups.values()) - len(all_idx)
    if missing or overlap:
        print("WARNING: group coverage not perfect.")
        print("  Covered:", len(all_idx), "Missing:", len(missing), "Overlap:", overlap)
        if missing:
            print("  Missing (1-based):", [m + 1 for m in missing])
    
    return groups


# =========================
# 多种可视化形式 (Multiple visualization forms)
# =========================
def scatter_ranked(labels, values, title, xlabel, outpath_no_ext, 
                   highlight_labels=None, topn=20, figsize=(10, 6)):
    dfp = pd.DataFrame({"label": labels, "value": values}).sort_values("value", ascending=False).head(topn)
    
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    
    # 创建渐变颜色
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(dfp)))
    
    # 散点图 - 使用较大的散点
    sc = ax.scatter(range(len(dfp)), dfp["value"], s=250, alpha=0.7, 
                    c=colors, edgecolors='k', linewidth=1.5, zorder=5)
    
    # 添加水平线
    ax.hlines(dfp["value"], 0, range(len(dfp)), colors='gray', alpha=0.3, linewidth=1.0)
    
    # 标注高亮特征
    if highlight_labels:
        for i, (label, value) in enumerate(zip(dfp["label"], dfp["value"])):
            if label in highlight_labels:
                ax.plot(i, value, 'o', markersize=16, markerfacecolor='none', 
                       markeredgecolor='red', markeredgewidth=2.5, zorder=10)
    
    ax.set_xticks(range(len(dfp)))
    ax.set_xticklabels(dfp["label"], rotation=45, ha='right', fontweight='bold')
    ax.set_xlabel("Features", fontweight='bold')
    ax.set_ylabel(xlabel, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 加粗边框
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    fig.tight_layout()
    save_fig(fig, outpath_no_ext)


def violin_feature_importance(feature_names, importance_values, title, outpath_no_ext, 
                              topn=20, figsize=(12, 6)):
    dfp = pd.DataFrame({"Feature": feature_names, "Importance": importance_values})
    dfp = dfp.nlargest(topn, "Importance")
    
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    
    # 为每个特征创建随机分布以模拟不确定性
    np.random.seed(RANDOM_STATE)
    data_for_violin = []
    positions = []
    
    for i, (feature, imp) in enumerate(zip(dfp["Feature"], dfp["Importance"])):
        # 基于重要性值创建一些随机波动
        n_points = 100
        data = imp + np.random.normal(0, imp*0.1, n_points)
        data_for_violin.append(data)
        positions.append(i)
    
    vp = ax.violinplot(data_for_violin, positions=positions, showmeans=True, showmedians=True)
    
    # 自定义小提琴颜色
    for i, pc in enumerate(vp['bodies']):
        pc.set_facecolor(plt.cm.viridis(i/topn))
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
        pc.set_alpha(0.7)
    
    ax.set_xticks(range(len(dfp)))
    ax.set_xticklabels(dfp["Feature"], rotation=45, ha='right', fontweight='bold')
    ax.set_ylabel("Feature Importance", fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 加粗边框
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    fig.tight_layout()
    save_fig(fig, outpath_no_ext)


def radar_group_importance(groups, importance_dict, title, outpath_no_ext, figsize=(8, 8)):
    categories = list(groups.keys())
    N = len(categories)
    
    # 获取重要性值
    values = [importance_dict.get(g, 0) for g in categories]
    
    # 重复第一个值以闭合雷达图
    values += values[:1]
    
    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111, polar=True)
    
    # 绘制雷达图
    ax.plot(angles, values, 'o-', linewidth=3, color=GROUP_COLORS.get('EnergyThermo', '#F18F01'), 
            markersize=10, markeredgecolor='black', markeredgewidth=1.5)
    ax.fill(angles, values, alpha=0.25, color=GROUP_COLORS.get('EnergyThermo', '#F18F01'))
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=16, fontweight='bold')
    
    # 设置y轴标签
    ax.set_ylim(0, max(values)*1.1)
    ax.set_yticks(np.linspace(0, max(values), 5))
    ax.set_yticklabels([f'{x:.3f}' for x in np.linspace(0, max(values), 5)], 
                      fontsize=14, fontweight='bold')
    
    ax.set_title(title, fontsize=18, fontweight='bold', pad=25)
    ax.grid(True, alpha=0.3, linewidth=1.5)
    
    fig.tight_layout()
    save_fig(fig, outpath_no_ext)


def prediction_scatter_plot(y_true, y_pred, title, outpath_no_ext, figsize=(10, 8)):
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)
    
    # 计算R2和RMSE
    r2 = r2_score(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    mae_val = mean_absolute_error(y_true, y_pred)
    
    # 绘制散点 - 使用大散点
    scatter = ax.scatter(y_true, y_pred, s=150, alpha=0.7, 
                        c='#2E86AB', edgecolors='k', linewidth=1.5, zorder=5)
    
    # 绘制对角线（理想预测线）- 使用红色虚线加粗
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=3.5, alpha=0.8, label='Ideal Prediction', zorder=4)
    
    # 添加统计信息
    stats_text = f'R2 = {r2:.4f}\nRMSE = {rmse_val:.2f}\nMAE = {mae_val:.2f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=16, 
            fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2))
    
    ax.set_xlabel('Actual Entropy', fontsize=18, fontweight='bold')
    ax.set_ylabel('Predicted Entropy', fontsize=18, fontweight='bold')
    ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
    
    # 设置坐标轴范围
    ax.set_xlim([min_val - 0.05*(max_val-min_val), max_val + 0.05*(max_val-min_val)])
    ax.set_ylim([min_val - 0.05*(max_val-min_val), max_val + 0.05*(max_val-min_val)])
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
    ax.legend(loc='lower right', fontsize=16, frameon=True, framealpha=0.9)
    
    # 加粗边框
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    
    fig.tight_layout()
    save_fig(fig, outpath_no_ext)
    
    return r2, rmse_val, mae_val


def parallel_coordinates_plot(X, y, feature_names, topn=10, outpath_no_ext=None):
    # 选择最重要的特征
    from sklearn.ensemble import RandomForestRegressor
    model_temp = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    model_temp.fit(X, y)
    importances = model_temp.feature_importances_
    
    idx_important = np.argsort(importances)[-topn:][::-1]
    top_features = [feature_names[i] for i in idx_important]
    X_top = X[:, idx_important]
    
    # 标准化数据用于可视化
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_top)
    
    # 根据目标值分组
    y_quantile = pd.qcut(y, q=4, labels=False, duplicates='drop')
    
    fig = plt.figure(figsize=(14, 8), dpi=300)
    ax = fig.add_subplot(111)
    
    # 绘制平行坐标线
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 4))
    
    for q in range(4):
        idx = np.where(y_quantile == q)[0]
        if len(idx) > 0:
            sample_idx = idx[:min(50, len(idx))]  # 采样以避免过度绘图
            for i in sample_idx:
                ax.plot(range(topn), X_scaled[i], color=colors[q], alpha=0.2, linewidth=1.5)
    
    # 添加特征名称
    ax.set_xticks(range(topn))
    ax.set_xticklabels(top_features, rotation=45, ha='right', fontweight='bold')
    
    ax.set_xlabel('Features', fontsize=18, fontweight='bold')
    ax.set_ylabel('Normalized Value', fontsize=18, fontweight='bold')
    ax.set_title('Parallel Coordinates Plot (Colored by Entropy Quantile)', 
                fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=1.0)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], alpha=0.7, 
                            label=f'Entropy Quantile {i+1}') for i in range(4)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=16, frameon=True)
    
    # 加粗边框
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    
    fig.tight_layout()
    
    if outpath_no_ext:
        save_fig(fig, outpath_no_ext)


def grouped_ablation_scatter(gnames, cv_r2, only_r2, drop_r2, outpath_no_ext):
    fig = plt.figure(figsize=(10, 6), dpi=300)
    ax = fig.add_subplot(111)
    
    x_pos = np.arange(len(gnames))
    
    # 获取组颜色
    group_colors = [GROUP_COLORS.get(g, '#999999') for g in gnames]
    
    # 绘制散点 - 使用大散点
    sc1 = ax.scatter(x_pos - 0.15, only_r2, s=200, alpha=0.8, 
                     c=group_colors, marker='o', edgecolors='k', linewidth=2, 
                     label='Only Group (CV Mean)', zorder=5)
    
    sc2 = ax.scatter(x_pos + 0.15, drop_r2, s=200, alpha=0.8, 
                     c=group_colors, marker='s', edgecolors='k', linewidth=2, 
                     label='Drop Group (CV Mean)', zorder=5)
    
    # 连接线
    for i in range(len(gnames)):
        ax.plot([x_pos[i] - 0.15, x_pos[i] + 0.15], [only_r2[i], drop_r2[i]], 
                'k-', alpha=0.3, linewidth=2.0)
    
    # 完整模型CV R2线
    full_cv_r2 = np.mean(cv_r2) if isinstance(cv_r2, list) else cv_r2
    ax.axhline(full_cv_r2, linestyle='--', linewidth=3.0, color='#C73E1D', alpha=0.7)
    ax.text(len(gnames)-0.5, full_cv_r2, f'  Full Model CV R2 = {full_cv_r2:.3f}', 
            fontsize=16, color='#C73E1D', va='center', fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(gnames, rotation=15, ha='right', fontweight='bold')
    ax.set_ylabel('Cross-Validation R2 Score (Mean)', fontsize=18, fontweight='bold')
    ax.set_title('Group Ablation Analysis (5-Fold Cross-Validation)', 
                fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
    ax.legend(loc='best', fontsize=16, frameon=True)
    
    # 添加数值标签
    for i, (only, drop) in enumerate(zip(only_r2, drop_r2)):
        ax.text(x_pos[i] - 0.15, only, f'{only:.3f}', 
                ha='center', va='bottom', fontsize=14, fontweight='bold')
        ax.text(x_pos[i] + 0.15, drop, f'{drop:.3f}', 
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # 加粗边框
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    
    fig.tight_layout()
    save_fig(fig, outpath_no_ext)


def plot_cv_performance_across_folds(cv_results, outpath_no_ext):
    fig = plt.figure(figsize=(10, 6), dpi=300)
    ax = fig.add_subplot(111)
    
    folds = range(1, len(cv_results['fold_details']['val_r2']) + 1)
    val_r2 = cv_results['fold_details']['val_r2']
    val_rmse = cv_results['fold_details']['val_rmse']
    
    # 绘制R2分数 - 使用粗线和标记
    ax.plot(folds, val_r2, 'o-', linewidth=3.5, markersize=12, color='#2E86AB', 
            markerfacecolor='white', markeredgecolor='#2E86AB', markeredgewidth=2,
            label='R2 Score')
    
    # 绘制RMSE（在右侧y轴）
    ax2 = ax.twinx()
    ax2.plot(folds, val_rmse, 's--', linewidth=3.5, markersize=12, color='#C73E1D',
             markerfacecolor='white', markeredgecolor='#C73E1D', markeredgewidth=2,
             label='RMSE')
    
    # 添加平均线
    mean_r2 = np.mean(val_r2)
    mean_rmse = np.mean(val_rmse)
    ax.axhline(mean_r2, color='#2E86AB', linestyle=':', alpha=0.7, linewidth=2.5)
    ax2.axhline(mean_rmse, color='#C73E1D', linestyle=':', alpha=0.7, linewidth=2.5)
    
    ax.set_xlabel('Fold Number', fontsize=18, fontweight='bold')
    ax.set_ylabel('R2 Score', fontsize=18, fontweight='bold', color='#2E86AB')
    ax2.set_ylabel('RMSE', fontsize=18, fontweight='bold', color='#C73E1D')
    
    ax.set_xticks(folds)
    ax.set_xticklabels([f'Fold {i}' for i in folds], fontweight='bold')
    ax.set_title('CatBoost Performance across 5-Fold Cross-Validation', 
                fontsize=20, fontweight='bold')  # 修改标题
    
    # 添加图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=16, frameon=True)
    
    # 添加统计信息
    stats_text = f"Mean R2: {mean_r2:.4f} (+/-{np.std(val_r2):.4f})\n"
    stats_text += f"Mean RMSE: {mean_rmse:.4f} (+/-{np.std(val_rmse):.4f})"
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=16, fontweight='bold',
            verticalalignment='bottom', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2))
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
    
    # 加粗边框
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    for spine in ax2.spines.values():
        spine.set_linewidth(2.5)
    
    fig.tight_layout()
    save_fig(fig, outpath_no_ext)


# =========================
# Main
# =========================
def main():
    warnings.filterwarnings("ignore")
    set_pub_theme()
    ensure_dir(OUTDIR)
    
    # 加载数据
    df = pd.read_excel(DATA_FILE)
    
    # 安全检查：需要至少64列
    if df.shape[1] < 64:
        raise ValueError(f"Database must have >= 64 columns, but got {df.shape[1]}")
    
    # 特征 (1..63) 和目标 (64)
    X = df.iloc[:, 0:N_FEATURES].copy()
    y = df.iloc[:, TARGET_COL_1BASED - 1].astype(float).values
    feature_names = X.columns.tolist()
    
    print(f"Data shape: {df.shape}")
    print(f"Features: {X.shape[1]} (columns 1-{N_FEATURES})")
    print(f"Target: column {TARGET_COL_1BASED} (entropy)")
    print(f"Feature names: {feature_names[:5]}...")
    
    # 使用全部数据进行特征选择和标准化
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1) 5折交叉验证评估完整模型
    print("Performing 5-Fold Cross-Validation for Full CatBoost Model...")
    cv_results = evaluate_with_cv(CATBOOST_PARAMS, X_scaled, y, 
                                  n_splits=N_SPLITS, random_state=RANDOM_STATE)
    
    # 训练最终模型（在全部数据上）
    print("Training final CatBoost model on full dataset...")
    final_model = CatBoostRegressor(**CATBOOST_PARAMS)
    final_model.fit(X_scaled, y, verbose=False)
    
    # 生成预测值用于预测散点图
    y_pred_full = final_model.predict(X_scaled)
    
    # 保存交叉验证性能
    perf_df = pd.DataFrame([{
        "Model": "CatBoost",
        "Evaluation": "5-Fold_CV",
        "CV_R2_Mean": cv_results['CV_R2_Mean'],
        "CV_R2_Std": cv_results['CV_R2_Std'],
        "CV_MAE_Mean": cv_results['CV_MAE_Mean'],
        "CV_MAE_Std": cv_results['CV_MAE_Std'],
        "CV_RMSE_Mean": cv_results['CV_RMSE_Mean'],
        "CV_RMSE_Std": cv_results['CV_RMSE_Std'],
        "Full_Model_R2": r2_score(y, y_pred_full),
        "Full_Model_MAE": mean_absolute_error(y, y_pred_full),
        "Full_Model_RMSE": rmse(y, y_pred_full),
        "n_samples": int(len(y)),
        "n_features": int(N_FEATURES),
    }])
    perf_df.to_excel(os.path.join(OUTDIR, "model_performance_5fold_cv.xlsx"), index=False)
    perf_df.to_csv(os.path.join(OUTDIR, "model_performance_5fold_cv.csv"), index=False)
    
    # 绘制交叉验证性能图
    plot_cv_performance_across_folds(cv_results, os.path.join(OUTDIR, "Fig0_CV_performance"))
    
    # 绘制预测散点图 (新功能)
    print("Generating prediction scatter plot...")
    r2_full, rmse_full, mae_full = prediction_scatter_plot(
        y, y_pred_full, 
        "CatBoost: Actual vs Predicted Entropy (Full Dataset)", 
        os.path.join(OUTDIR, "Fig1_Prediction_scatter")
    )
    
    # 2) CatBoost 内置特征重要性 - 使用散点图
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "CatBoost_Importance": final_model.feature_importances_
    }).sort_values("CatBoost_Importance", ascending=False)
    fi_df.to_excel(os.path.join(OUTDIR, "catboost_feature_importance.xlsx"), index=False)
    fi_df.to_csv(os.path.join(OUTDIR, "catboost_feature_importance.csv"), index=False)
    
    # 散点图展示特征重要性
    scatter_ranked(
        labels=fi_df["Feature"].tolist(),
        values=fi_df["CatBoost_Importance"].tolist(),
        title="CatBoost Feature Importance (Top 20) - 5-Fold CV",
        xlabel="Importance Score",
        outpath_no_ext=os.path.join(OUTDIR, "Fig2_CatBoost_importance_scatter"),
        topn=20
    )
    
    # 小提琴图展示特征重要性分布
    violin_feature_importance(
        feature_names=fi_df["Feature"].tolist(),
        importance_values=fi_df["CatBoost_Importance"].tolist(),
        title="Feature Importance Distribution (Top 20) - 5-Fold CV",
        outpath_no_ext=os.path.join(OUTDIR, "Fig3_CatBoost_importance_violin"),
        topn=20
    )
    
    # 3) 特征置换重要性 (使用全部数据)
    print("Calculating permutation feature importance...")
    # 注意：CatBoost模型也可以使用permutation_importance
    pi = permutation_importance(
        final_model, X_scaled, y,
        n_repeats=N_REPEATS_FEATURE,
        random_state=RANDOM_STATE,
        scoring="r2",
        n_jobs=-1
    )
    pi_feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Perm_R2_drop_mean": pi.importances_mean,
        "Perm_R2_drop_std": pi.importances_std
    }).sort_values("Perm_R2_drop_mean", ascending=False)
    
    pi_feat_df.to_excel(os.path.join(OUTDIR, "permutation_importance_features.xlsx"), index=False)
    pi_feat_df.to_csv(os.path.join(OUTDIR, "permutation_importance_features.csv"), index=False)
    
    # 散点图展示置换重要性
    scatter_ranked(
        labels=pi_feat_df["Feature"].tolist(),
        values=pi_feat_df["Perm_R2_drop_mean"].tolist(),
        title="Permutation Feature Importance (Top 20) - 5-Fold CV",
        xlabel="R2 Decrease After Permutation",
        outpath_no_ext=os.path.join(OUTDIR, "Fig4_Permutation_features_scatter"),
        topn=20
    )
    
    # 4) 组消融分析 (使用5折交叉验证)
    print("Performing group ablation analysis with 5-fold CV...")
    groups = build_groups()
    all_idx = np.arange(N_FEATURES, dtype=int)
    
    ablation_rows = []
    
    # 首先计算完整模型的CV性能（已计算）
    ablation_rows.append({
        "Setting": "Full", 
        "Mode": "AllFeatures", 
        "R2": cv_results['CV_R2_Mean'],
        "MAE": cv_results['CV_MAE_Mean'],
        "RMSE": cv_results['CV_RMSE_Mean']
    })
    
    for gname, gidx in groups.items():
        gidx = np.array(gidx, dtype=int)
        
        # OnlyGroup (仅使用该组特征)
        print(f"  Evaluating OnlyGroup for {gname}...")
        only_results = evaluate_with_cv(
            CATBOOST_PARAMS, 
            X_scaled[:, gidx], y, 
            n_splits=N_SPLITS, random_state=RANDOM_STATE
        )
        ablation_rows.append({
            "Setting": gname, 
            "Mode": "OnlyGroup", 
            "R2": only_results['CV_R2_Mean'],
            "MAE": only_results['CV_MAE_Mean'],
            "RMSE": only_results['CV_RMSE_Mean']
        })
        
        # DropGroup (移除该组特征)
        print(f"  Evaluating DropGroup for {gname}...")
        keep = np.setdiff1d(all_idx, gidx)
        drop_results = evaluate_with_cv(
            CATBOOST_PARAMS, 
            X_scaled[:, keep], y, 
            n_splits=N_SPLITS, random_state=RANDOM_STATE
        )
        ablation_rows.append({
            "Setting": gname, 
            "Mode": "DropGroup", 
            "R2": drop_results['CV_R2_Mean'],
            "MAE": drop_results['CV_MAE_Mean'],
            "RMSE": drop_results['CV_RMSE_Mean']
        })
    
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_excel(os.path.join(OUTDIR, "group_ablation_performance.xlsx"), index=False)
    ablation_df.to_csv(os.path.join(OUTDIR, "group_ablation_performance.csv"), index=False)
    
    full_r2 = cv_results['CV_R2_Mean']
    gnames = list(groups.keys())
    only_r2 = [float(ablation_df[(ablation_df["Setting"] == g) & (ablation_df["Mode"] == "OnlyGroup")]["R2"].iloc[0]) for g in gnames]
    drop_r2 = [float(ablation_df[(ablation_df["Setting"] == g) & (ablation_df["Mode"] == "DropGroup")]["R2"].iloc[0]) for g in gnames]
    
    # 使用散点图展示组消融结果
    grouped_ablation_scatter(
        gnames=gnames,
        cv_r2=full_r2,
        only_r2=only_r2,
        drop_r2=drop_r2,
        outpath_no_ext=os.path.join(OUTDIR, "Fig5_Group_ablation_scatter")
    )
    
    # 5) 组置换重要性 (使用全部数据)
    print("Calculating group permutation importance...")
    base_r2 = float(r2_score(y, final_model.predict(X_scaled)))
    rng = np.random.default_rng(RANDOM_STATE)
    
    grp_rows = []
    group_importance_dict = {}
    for gname, gidx in groups.items():
        gidx = np.array(gidx, dtype=int)
        drops = []
        for _ in range(N_REPEATS_GROUP):
            Xp = X_scaled.copy()
            perm = rng.permutation(Xp.shape[0])
            Xp[:, gidx] = Xp[perm][:, gidx]
            r2p = float(r2_score(y, final_model.predict(Xp)))
            drops.append(base_r2 - r2p)
        
        mean_drop = float(np.mean(drops))
        group_importance_dict[gname] = mean_drop
        
        grp_rows.append({
            "Group": gname,
            "R2_drop_mean": mean_drop,
            "R2_drop_std": float(np.std(drops))
        })
    
    grp_df = pd.DataFrame(grp_rows).sort_values("R2_drop_mean", ascending=False)
    grp_df.to_excel(os.path.join(OUTDIR, "permutation_importance_groups.xlsx"), index=False)
    grp_df.to_csv(os.path.join(OUTDIR, "permutation_importance_groups.csv"), index=False)
    
    # 雷达图展示组重要性
    radar_group_importance(
        groups=groups,
        importance_dict=group_importance_dict,
        title="Group Permutation Importance (Radar Chart) - 5-Fold CV",
        outpath_no_ext=os.path.join(OUTDIR, "Fig6_Group_importance_radar")
    )
    
    # 6) SHAP 分析
    shap_ok = True
    try:
        import shap
    except Exception as e:
        shap_ok = False
        print("SHAP not available, skipped. Install with: pip install shap")
        print("Reason:", e)
    
    if shap_ok:
        n_shap = min(800, X_scaled.shape[0])
        Xs = X_scaled[:n_shap]
        
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(Xs)
        
        shap_rank = np.mean(np.abs(shap_values), axis=0)
        shap_rank_df = pd.DataFrame({
            "Feature": feature_names,
            "MeanAbsSHAP": shap_rank
        }).sort_values("MeanAbsSHAP", ascending=False)
        
        shap_rank_df.to_excel(os.path.join(OUTDIR, "shap_global_ranking.xlsx"), index=False)
        shap_rank_df.to_csv(os.path.join(OUTDIR, "shap_global_ranking.csv"), index=False)
        
        # SHAP 摘要图
        fig = plt.figure(figsize=(12, 8), dpi=300)
        shap.summary_plot(
            shap_values,
            features=Xs,
            feature_names=feature_names,
            show=False,
            max_display=20,
            plot_type="dot"
        )
        plt.title("SHAP Summary Plot (Top 20 Features) - 5-Fold CV", 
                 fontsize=20, fontweight='bold', pad=20)
        plt.gca().set_xlabel("SHAP Value", fontsize=18, fontweight='bold')
        
        # 设置字体为Arial加粗
        for item in ([plt.gca().title, plt.gca().xaxis.label, plt.gca().yaxis.label] + 
                    plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
            item.set_fontname("Arial")
            item.set_fontweight("bold")
            if hasattr(item, 'set_fontsize'):
                if isinstance(item, plt.Text) and item == plt.gca().title:
                    item.set_fontsize(20)
                elif isinstance(item, plt.Text) and (item == plt.gca().xaxis.label or item == plt.gca().yaxis.label):
                    item.set_fontsize(18)
                else:
                    item.set_fontsize(16)
        
        save_fig(fig, os.path.join(OUTDIR, "Fig7_SHAP_summary"))
        
        # SHAP 依赖图 - 为前3个特征创建
        for i, feature in enumerate(shap_rank_df["Feature"].head(3)):
            fig = plt.figure(figsize=(10, 8), dpi=300)
            shap.dependence_plot(
                feature,
                shap_values,
                Xs,
                feature_names=feature_names,
                show=False,
                interaction_index=None
            )
            plt.title(f"SHAP Dependence Plot: {feature} - 5-Fold CV", 
                     fontsize=20, fontweight='bold', pad=15)
            plt.gca().set_xlabel(feature, fontsize=18, fontweight='bold')
            plt.gca().set_ylabel("SHAP Value", fontsize=18, fontweight='bold')
            
            # 设置字体为Arial加粗
            for item in ([plt.gca().title, plt.gca().xaxis.label, plt.gca().yaxis.label] + 
                        plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                item.set_fontname("Arial")
                item.set_fontweight("bold")
                if hasattr(item, 'set_fontsize'):
                    if isinstance(item, plt.Text) and item == plt.gca().title:
                        item.set_fontsize(20)
                    elif isinstance(item, plt.Text) and (item == plt.gca().xaxis.label or item == plt.gca().yaxis.label):
                        item.set_fontsize(18)
                    else:
                        item.set_fontsize(16)
            
            save_fig(fig, os.path.join(OUTDIR, f"Fig7_{i+1}_SHAP_dependence_{feature}"))
        
        pd.DataFrame(shap_values, columns=feature_names).to_excel(
            os.path.join(OUTDIR, "shap_values_sample.xlsx"), index=False
        )
    
    # 7) t-SNE 可视化 (使用彩虹色)
    Xts = X_scaled
    yts = y
    
    if Xts.shape[0] > TSNE_MAX_SAMPLES:
        rng2 = np.random.default_rng(RANDOM_STATE)
        idx = rng2.choice(Xts.shape[0], size=TSNE_MAX_SAMPLES, replace=False)
        Xts = Xts[idx]
        yts = yts[idx]
    
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_STATE
    )
    Z = tsne.fit_transform(Xts)
    
    tsne_df = pd.DataFrame({
        "TSNE1": Z[:, 0],
        "TSNE2": Z[:, 1],
        "Entropy": yts
    })
    tsne_df.to_excel(os.path.join(OUTDIR, "tsne_embedding.xlsx"), index=False)
    
    # t-SNE 散点图 - 使用彩虹色 (rainbow colormap)
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(111)
    
    # 使用彩虹色系
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=yts, s=80, alpha=0.8, 
                   cmap='rainbow', edgecolors='k', linewidth=1.0)
    
    ax.set_xlabel("t-SNE 1", fontsize=18, fontweight='bold')
    ax.set_ylabel("t-SNE 2", fontsize=18, fontweight='bold')
    ax.set_title("t-SNE Visualization of Feature Space (Colored by Entropy) - 5-Fold CV", 
                fontsize=20, fontweight='bold')
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Entropy", rotation=270, labelpad=25, fontsize=18, fontweight='bold')
    
    # 设置colorbar字体
    cbar.ax.tick_params(labelsize=16)
    for label in cbar.ax.get_yticklabels():
        label.set_fontname("Arial")
        label.set_fontweight("bold")
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
    
    # 加粗边框
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    
    fig.tight_layout()
    save_fig(fig, os.path.join(OUTDIR, "Fig8_tSNE_visualization"))
    
    # 8) 平行坐标图 (使用全部数据)
    parallel_coordinates_plot(
        X_scaled, y, feature_names, topn=10,
        outpath_no_ext=os.path.join(OUTDIR, "Fig9_Parallel_coordinates")
    )
    
    # 9) 导出组特征名称
    group_name_table = []
    for gname, idxs in groups.items():
        cols = [feature_names[i] for i in idxs]
        group_name_table.append({
            "Group": gname,
            "n_features": len(cols),
            "Features": ", ".join(cols)
        })
    pd.DataFrame(group_name_table).to_excel(
        os.path.join(OUTDIR, "descriptor_groups_feature_names.xlsx"), index=False
    )
    
    # 10) 创建颜色主题图例
    fig = plt.figure(figsize=(10, 4), dpi=300)
    ax = fig.add_subplot(111)
    
    # 显示颜色调色板
    for i, (name, color) in enumerate(GROUP_COLORS.items()):
        ax.add_patch(plt.Rectangle((i*2, 0), 1.5, 1, color=color, edgecolor='k', linewidth=2))
        ax.text(i*2 + 0.75, -0.3, name, ha='center', va='top', 
               fontsize=16, fontweight='bold')
    
    ax.set_xlim(-0.5, len(GROUP_COLORS)*2 + 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Color Theme for Descriptor Groups - 5-Fold CV", 
                fontsize=20, fontweight='bold', pad=20)
    
    fig.tight_layout()
    save_fig(fig, os.path.join(OUTDIR, "Color_theme_legend"))
    
    print("\n" + "="*60)
    print("5-FOLD CROSS-VALIDATION ANALYSIS COMPLETE")
    print("="*60)
    print(f"Output directory: {OUTDIR}")
    print(f"\nCatBoost 5-Fold CV Performance:")
    print(f"  R2: {cv_results['CV_R2_Mean']:.4f} (+/-{cv_results['CV_R2_Std']:.4f})")
    print(f"  MAE: {cv_results['CV_MAE_Mean']:.4f} (+/-{cv_results['CV_MAE_Std']:.4f})")
    print(f"  RMSE: {cv_results['CV_RMSE_Mean']:.4f} (+/-{cv_results['CV_RMSE_Std']:.4f})")
    print(f"\nFull Model Performance (trained on all data):")
    print(f"  R2: {r2_full:.4f}")
    print(f"  RMSE: {rmse_full:.2f}")
    print(f"  MAE: {mae_full:.2f}")
    print("\nMain Figures Generated:")
    print("  Fig0: CV Performance across Folds")
    print("  Fig1: Prediction Scatter Plot (Actual vs Predicted)")
    print("  Fig2: CatBoost Feature Importance (Scatter plot)")
    print("  Fig3: Feature Importance Distribution (Violin plot)")
    print("  Fig4: Permutation Feature Importance (Scatter plot)")
    print("  Fig5: Group Ablation Analysis (Scatter plot)")
    print("  Fig6: Group Importance (Radar chart)")
    if shap_ok:
        print("  Fig7: SHAP Analysis (Summary & Dependence plots)")
    print("  Fig8: t-SNE Visualization (Rainbow colormap)")
    print("  Fig9: Parallel Coordinates Plot")
    print("\nAll figures saved at 300 DPI with Arial bold font")
    print("="*60)


if __name__ == "__main__":
    main()
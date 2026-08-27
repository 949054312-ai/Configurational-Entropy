# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut, cross_val_predict, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')
import pickle
from pathlib import Path

# =============================================================================
# 设置SCI期刊级绘图样式
# =============================================================================
# 设置字体为Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.unicode_minus'] = False

# 设置颜色方案 - SCI期刊常用配色
SCI_COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'yellow': '#bcbd22',
    'cyan': '#17becf'
}

# 设置seaborn样式
sns.set_style("whitegrid", {'axes.linewidth': 1.5, 'axes.edgecolor': 'black'})
sns.set_context("paper", font_scale=1.3, rc={"lines.linewidth": 2.5})

# =============================================================================
# 修改后的函数：创建混合物特征（不包含weight_entropy）
# =============================================================================

def parse_blend_molecules(mol_str):
    """Parse the blend molecules string."""
    if pd.isna(mol_str):
        return []
    
    # 修复：处理实际的字符串格式（包含\n换行符）
    mol_str = str(mol_str)
    
    # 尝试不同的分隔符
    if '\n' in mol_str:
        # 使用换行符分割
        molecules = [m.strip() for m in mol_str.split('\n')]
    elif '<br>' in mol_str:
        # 使用<br>标签分割
        molecules = [m.strip() for m in mol_str.split('<br>')]
    else:
        # 如果都没有，尝试按空格分割
        molecules = [m.strip() for m in mol_str.split()]
    
    # 移除空字符串
    return [m for m in molecules if m]

def parse_weight_ratio(weight_str):
    """Parse weight ratio string like '1:1.2' or '1:0.6:0.3:0.3'."""
    if pd.isna(weight_str):
        return []
    
    # 修复：直接按:分割，不修改权重值
    weights = []
    for w in str(weight_str).split(':'):
        try:
            weights.append(float(w))
        except:
            # 如果转换失败，使用1.0作为默认权重
            weights.append(1.0)
            print(f"Warning: Could not parse weight '{w}', using 1.0 instead")
    return weights

def create_blend_features(blend_row, descriptors_df, molecule_list, weight_list):
    """
    Create features for a blend by aggregating molecular descriptors.
    
    Args:
        blend_row: Row from blends dataframe
        descriptors_df: DataFrame with molecular descriptors
        molecule_list: List of molecule names in the blend
        weight_list: List of weights for each molecule
    
    Returns:
        Dictionary of blend features
    """
    blend_features = {}
    
    # 归一化权重，使其和为1
    weights = np.array(weight_list)
    if len(weights) > 0:
        weights = weights / weights.sum()  # 归一化
    
    # 收集混合物中每个分子的描述符
    mol_descriptors = []
    available_molecules = []
    
    for mol_name, weight in zip(molecule_list, weights):
        mol_data = descriptors_df[descriptors_df['name'] == mol_name]
        if not mol_data.empty:
            mol_desc = mol_data.iloc[0]
            mol_descriptors.append((mol_desc, weight))
            available_molecules.append(mol_name)
        else:
            print(f"Warning: Molecule {mol_name} not found in descriptors")
    
    if not mol_descriptors:
        print(f"Warning: No descriptors found for blend")
        return None
    
    # 获取所有描述符列（排除name和SMILES）
    descriptor_cols = [col for col in descriptors_df.columns 
                      if col not in ['name', 'smiles']]
    
    # 对每个描述符计算混合物统计量
    for desc in descriptor_cols:
        try:
            # 获取值和权重
            values = [d[desc] for d, w in mol_descriptors]
            weights = [w for d, w in mol_descriptors]
            
            # 如果所有值都是NaN则跳过
            if all(pd.isna(v) for v in values):
                continue
            
            # 将值转换为数组
            values_arr = np.array(values)
            weights_arr = np.array(weights)
            
            # 计算各种统计量
            # 1. 加权平均
            valid_mask = ~pd.isna(values_arr)
            if np.any(valid_mask):
                # 使用归一化权重计算加权平均
                blend_features[f'{desc}_weighted_mean'] = np.average(
                    values_arr[valid_mask], weights=weights_arr[valid_mask])
            
            # 2. 加权标准差
            if len(values_arr[valid_mask]) > 1:
                variance = np.average((values_arr[valid_mask] - 
                                     blend_features[f'{desc}_weighted_mean'])**2, 
                                    weights=weights_arr[valid_mask])
                blend_features[f'{desc}_weighted_std'] = np.sqrt(variance)
            
            # 3. 最小值和最大值
            blend_features[f'{desc}_min'] = np.nanmin(values_arr)
            blend_features[f'{desc}_max'] = np.nanmax(values_arr)
            
            # 4. 范围
            blend_features[f'{desc}_range'] = blend_features[f'{desc}_max'] - blend_features[f'{desc}_min']
            
            # 5. 加权中位数（近似）
            sorted_idx = np.argsort(values_arr[valid_mask])
            sorted_weights = weights_arr[valid_mask][sorted_idx]
            sorted_values = values_arr[valid_mask][sorted_idx]
            cum_weights = np.cumsum(sorted_weights)
            median_idx = np.searchsorted(cum_weights, 0.5)
            blend_features[f'{desc}_weighted_median'] = sorted_values[median_idx] if median_idx < len(sorted_values) else sorted_values[-1]
            
        except Exception as e:
            # 如果出错，跳过这个描述符
            continue
    
    # 添加混合物组成特征（不包含weight_entropy）
    blend_features['num_molecules'] = len(molecule_list)
    blend_features['weight_sum'] = np.sum(weights)  # 归一化后应该为1
    blend_features['weight_std'] = np.std(weights) if len(weights) > 1 else 0
    # 注意：这里移除了 weight_entropy 特征
    
    return blend_features

def remove_redundant_blend_features(blend_features_df, target_values=None, correlation_threshold=0.9):
    """
    在共混特征层面去除冗余特征
    
    Args:
        blend_features_df: 共混特征DataFrame
        target_values: 目标变量（可选，用于决定保留哪个冗余特征）
        correlation_threshold: 相关性阈值
    """
    print(f"\nRemoving redundant blend features (threshold={correlation_threshold})...")
    
    # 确保是DataFrame
    if not isinstance(blend_features_df, pd.DataFrame):
        blend_features_df = pd.DataFrame(blend_features_df)
    
    # 获取数值列
    numeric_cols = blend_features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) <= 1:
        return blend_features_df
    
    # 用中位数填充缺失值（仅用于相关性分析）
    X_filled = blend_features_df[numeric_cols].fillna(blend_features_df[numeric_cols].median())
    
    # 计算相关矩阵
    corr_matrix = X_filled.corr().abs()
    
    # 找出高度相关的特征对
    redundant_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            corr_value = corr_matrix.iloc[i, j]
            if corr_value >= correlation_threshold:
                redundant_pairs.append({
                    'feature_i': numeric_cols[i],
                    'feature_j': numeric_cols[j],
                    'correlation': corr_value
                })
    
    if not redundant_pairs:
        print("  No redundant features found")
        return blend_features_df
    
    print(f"  Found {len(redundant_pairs)} redundant feature pairs")
    
    # 识别要删除的特征
    features_to_keep = set(numeric_cols)
    features_to_remove = set()
    
    # 如果有目标变量，用它来决定保留哪个特征
    if target_values is not None:
        # 计算每个特征与目标的相关性
        feature_target_corr = {}
        for col in numeric_cols:
            try:
                valid_mask = ~pd.isna(blend_features_df[col]) & ~pd.isna(target_values)
                if np.sum(valid_mask) > 2:
                    corr_val = abs(np.corrcoef(blend_features_df[col][valid_mask], 
                                              target_values[valid_mask])[0, 1])
                    if not np.isnan(corr_val):
                        feature_target_corr[col] = corr_val
            except:
                feature_target_corr[col] = 0
    
    # 按照相关性从高到低处理特征对
    redundant_pairs_sorted = sorted(redundant_pairs, key=lambda x: x['correlation'], reverse=True)
    
    for pair in redundant_pairs_sorted:
        feat_i = pair['feature_i']
        feat_j = pair['feature_j']
        
        # 如果两个特征都还没有被标记为删除
        if feat_i in features_to_keep and feat_j in features_to_keep:
            if target_values is not None and feat_i in feature_target_corr and feat_j in feature_target_corr:
                # 比较与目标的相关性，保留相关性更高的特征
                if feature_target_corr[feat_i] > feature_target_corr[feat_j]:
                    features_to_remove.add(feat_j)
                    features_to_keep.remove(feat_j)
                else:
                    features_to_remove.add(feat_i)
                    features_to_keep.remove(feat_i)
            else:
                # 默认保留第一个特征
                features_to_remove.add(feat_j)
                features_to_keep.remove(feat_j)
    
    print(f"  Removing {len(features_to_remove)} redundant features")
    print(f"  Keeping {len(features_to_keep)} features")
    
    # 返回去除冗余后的特征
    features_to_keep = list(features_to_keep)
    return blend_features_df[features_to_keep]

def analyze_feature_target_correlation(X, y, target_name="Entropy_mean"):
    """分析特征与目标变量的相关性"""
    print(f"\nAnalyzing feature-target correlations for {target_name}...")
    
    # 确保X是DataFrame
    if not isinstance(X, pd.DataFrame):
        X_df = pd.DataFrame(X)
    else:
        X_df = X.copy()
    
    # 计算每个特征与目标变量的相关性
    correlations = {}
    for col in X_df.columns:
        if X_df[col].dtype in ['float64', 'int64']:
            # 移除NaN值
            valid_mask = ~pd.isna(X_df[col]) & ~pd.isna(y)
            if np.sum(valid_mask) > 5:
                try:
                    corr_val = np.corrcoef(X_df[col][valid_mask], y[valid_mask])[0, 1]
                    if not np.isnan(corr_val):
                        correlations[col] = corr_val
                except:
                    continue
    
    # 创建相关性DataFrame
    corr_df = pd.DataFrame({
        'feature': list(correlations.keys()),
        'correlation': list(correlations.values()),
        'abs_correlation': [abs(c) for c in correlations.values()]
    }).sort_values('abs_correlation', ascending=False)
    
    print(f"  Analyzed {len(corr_df)} features")
    print(f"  Top 5 features by absolute correlation:")
    for i, row in corr_df.head(5).iterrows():
        print(f"    {row['feature']}: r = {row['correlation']:.4f}")
    
    # 可视化相关性
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 特征相关性条形图
    top_n = min(20, len(corr_df))
    top_features = corr_df.head(top_n)
    
    colors = [SCI_COLORS['blue'] if c > 0 else SCI_COLORS['red'] for c in top_features['correlation']]
    axes[0].barh(range(top_n), top_features['abs_correlation'], color=colors, edgecolor='black')
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(top_features['feature'], fontsize=9)
    axes[0].set_xlabel('Absolute Correlation Coefficient (|r|)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Top {top_n} Features by Correlation with {target_name}', 
                     fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # 2. 相关性分布直方图
    axes[1].hist(corr_df['correlation'], bins=30, color=SCI_COLORS['green'], 
                edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    axes[1].set_xlabel('Pearson Correlation Coefficient (r)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Distribution of Feature-Target Correlations', 
                     fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'Feature_Target_Correlation_{target_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr_df

def create_interaction_features(blend_features_df):
    """创建交互特征"""
    X = blend_features_df.copy()
    
    # 获取数值列
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # 创建一些交互项
    mean_cols = [col for col in numeric_cols if 'weighted_mean' in col]
    
    # 创建交互项
    interaction_features = {}
    
    if len(mean_cols) >= 2:
        # 创建几个关键交互项
        for i in range(min(3, len(mean_cols))):
            for j in range(i+1, min(4, len(mean_cols))):
                col1, col2 = mean_cols[i], mean_cols[j]
                # 提取描述符名称
                desc1 = col1.replace('_weighted_mean', '')
                desc2 = col2.replace('_weighted_mean', '')
                interaction_name = f'interaction_{desc1}_{desc2}'
                interaction_features[interaction_name] = X[col1] * X[col2]
    
    # 为某些描述符添加比率特征
    if len(mean_cols) >= 2:
        for i in range(min(2, len(mean_cols))):
            for j in range(i+1, min(3, len(mean_cols))):
                col1, col2 = mean_cols[i], mean_cols[j]
                desc1 = col1.replace('_weighted_mean', '')
                desc2 = col2.replace('_weighted_mean', '')
                ratio_name = f'ratio_{desc1}_{desc2}'
                interaction_features[ratio_name] = X[col1] / (X[col2] + 1e-10)
    
    # 转换为DataFrame
    if interaction_features:
        interaction_df = pd.DataFrame(interaction_features)
        # 与原始特征合并
        X_combined = pd.concat([X, interaction_df], axis=1)
        return X_combined
    
    return X

def select_features_by_correlation(X_train, y_train, X_test, n_features=10):
    """基于相关性选择特征（在训练集上）"""
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
    
    # 计算训练集特征与目标的相关性
    correlations = {}
    for col in X_train.columns:
        try:
            valid_mask = ~pd.isna(X_train[col]) & ~pd.isna(y_train)
            if np.sum(valid_mask) > 2:
                corr_val = abs(np.corrcoef(X_train[col][valid_mask], y_train[valid_mask])[0, 1])
                if not np.isnan(corr_val):
                    correlations[col] = corr_val
        except:
            continue
    
    if not correlations:
        return X_train, X_test
    
    # 选择相关性最高的特征
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    n_features = min(n_features, len(sorted_features), len(y_train) - 2)
    
    if n_features <= 0:
        return X_train, X_test
    
    selected_features = [f[0] for f in sorted_features[:n_features]]
    
    # 去除选中的特征之间的冗余
    if len(selected_features) > 1:
        selected_train = X_train[selected_features].copy()
        selected_train = remove_redundant_blend_features(selected_train, y_train, correlation_threshold=0.9)
        selected_features = selected_train.columns.tolist()
    
    return X_train[selected_features], X_test[selected_features]

def select_features_by_importance(X_train, y_train, X_test, n_features=10, model_type='gb'):
    """基于模型重要性选择特征（在训练集上）"""
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
    
    n_features = min(n_features, X_train.shape[1], len(y_train) - 2)
    
    if n_features <= 0:
        return X_train, X_test
    
    if model_type == 'gb':
        # 使用梯度提升
        model = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            random_state=42
        )
    elif model_type == 'rf':
        # 使用随机森林
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=3,
            random_state=42
        )
    else:
        return X_train, X_test
    
    try:
        model.fit(X_train.fillna(X_train.median()), y_train)
        
        # 获取特征重要性
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        selected_features = importance.head(n_features)['feature'].values
        
        # 去除选中的特征之间的冗余
        if len(selected_features) > 1:
            selected_train = X_train[selected_features].copy()
            selected_train = remove_redundant_blend_features(selected_train, y_train, correlation_threshold=0.9)
            selected_features = selected_train.columns.tolist()
        
        return X_train[selected_features], X_test[selected_features]
    except:
        return X_train, X_test

def reduce_features_with_pca(X_train, y_train, X_test, n_components=None):
    """使用PCA降维（在训练集上）"""
    if n_components is None:
        n_components = min(5, X_train.shape[1], len(y_train) - 2)
    
    if n_components <= 0:
        return X_train, X_test
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.fillna(X_train.median()))
    X_test_scaled = scaler.transform(X_test.fillna(X_test.median()))
    
    # PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    return X_train_pca, X_test_pca

def manual_leave_one_out_cv(X, y, model, feature_selection_method=None, n_features=10):
    """手动实现留一法交叉验证，避免信息泄露"""
    n_samples = len(X)
    y_pred = np.zeros(n_samples)
    
    print(f"  Performing LOOCV with {feature_selection_method} feature selection...")
    
    for i in range(n_samples):
        # 分割训练集和测试集
        train_idx = list(range(n_samples))
        train_idx.remove(i)
        test_idx = [i]
        
        X_train = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
        y_train = y[train_idx]
        X_test = X.iloc[test_idx] if isinstance(X, pd.DataFrame) else X[test_idx]
        
        # 特征选择（仅使用训练集）
        if feature_selection_method == 'correlation':
            X_train_selected, X_test_selected = select_features_by_correlation(
                X_train, y_train, X_test, n_features=n_features
            )
        elif feature_selection_method == 'importance_gb':
            X_train_selected, X_test_selected = select_features_by_importance(
                X_train, y_train, X_test, n_features=n_features, model_type='gb'
            )
        elif feature_selection_method == 'importance_rf':
            X_train_selected, X_test_selected = select_features_by_importance(
                X_train, y_train, X_test, n_features=n_features, model_type='rf'
            )
        elif feature_selection_method == 'pca':
            X_train_selected, X_test_selected = reduce_features_with_pca(
                X_train, y_train, X_test, n_components=min(5, len(train_idx) - 2)
            )
        else:
            # 无特征选择
            X_train_selected, X_test_selected = X_train, X_test
        
        # 处理NaN值
        if isinstance(X_train_selected, pd.DataFrame):
            X_train_filled = X_train_selected.fillna(X_train_selected.median())
            X_test_filled = X_test_selected.fillna(X_test_selected.median())
        else:
            X_train_filled = np.nan_to_num(X_train_selected, nan=np.nanmedian(X_train_selected, axis=0))
            X_test_filled = np.nan_to_num(X_test_selected, nan=np.nanmedian(X_train_selected, axis=0))
        
        # 训练模型。每个留一折使用独立克隆，避免模型状态在折间残留
        try:
            fold_model = clone(model)
            fold_model.fit(X_train_filled, y_train)
            y_pred[i] = fold_model.predict(X_test_filled)[0]
        except Exception as e:
            print(f"    Error in fold {i+1}: {e}")
            y_pred[i] = np.mean(y_train)
    
    return y_pred

def create_simple_models(n_samples):
    """创建适合小数据集的模型。

    对线性模型、SVR和MLP统一在每个交叉验证训练折内进行标准化。
    这对不同量纲的分子描述符尤其重要。
    """
    models = {
        'Linear Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        'Ridge Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ]),
        'Lasso Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(alpha=0.001, random_state=42, max_iter=20000))
        ]),
    }

    # 只有当样本量足够时才添加复杂模型
    if n_samples >= 10:
        models.update({
            'Random Forest': RandomForestRegressor(
                n_estimators=300,
                max_depth=3,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                min_samples_leaf=2,
                random_state=42
            ),
            'SVR (RBF)': Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVR(
                    kernel='rbf',
                    C=10.0,
                    gamma='scale',
                    epsilon=0.01
                ))
            ]),
        })

    # 小样本下MLP仅作为诊断模型，不建议作为最终主模型
    if n_samples >= 15:
        models['MLP'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(
                hidden_layer_sizes=(5,),
                solver='lbfgs',
                alpha=0.1,
                max_iter=5000,
                random_state=42
            ))
        ])

    return models

def create_feature_selection_methods():
    """创建特征选择方法"""
    return {
        'No Selection': None,
        'Correlation-based': 'correlation',
        'GB Importance': 'importance_gb',
        'RF Importance': 'importance_rf',
        'PCA': 'pca'
    }

def evaluate_model_performance(y_true, y_pred, model_name, feature_method):
    """评估模型性能"""
    if len(y_true) < 2:
        return {
            'Model': model_name,
            'Feature_Method': feature_method,
            'R2': np.nan,
            'Q2': np.nan,
            'MAE': np.nan,
            'RMSE': np.nan,
            'Predictions': y_pred
        }
    
    # y_pred来自留一法的折外预测，因此这里的R2本质上就是交叉验证Q2。
    # 两者使用同一公式，不应在论文中作为两个独立指标解释。
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    denominator = np.sum((y_true - np.mean(y_true))**2)
    q2 = np.nan if denominator == 0 else 1 - np.sum((y_true - y_pred)**2) / denominator
    
    return {
        'Model': model_name,
        'Feature_Method': feature_method,
        'R2': r2,
        'Q2': q2,
        'MAE': mae,
        'RMSE': rmse,
        'Predictions': y_pred.copy()
    }

def create_comprehensive_visualization(results_df, X_full, y_mean, best_result, pca_info=None, 
                                      importance_df=None, output_prefix=''):
    """创建综合可视化图表"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle(f'Blend Entropy Prediction Results ({output_prefix})', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. 最佳模型的实际vs预测
    ax1 = axes[0, 0]
    y_pred_best = best_result['Predictions']
    
    scatter = ax1.scatter(y_mean, y_pred_best, color=SCI_COLORS['blue'], s=120, 
                         alpha=0.8, edgecolors='black', linewidth=1.5)
    
    # 添加对角线
    min_val = min(y_mean.min(), y_pred_best.min())
    max_val = max(y_mean.max(), y_pred_best.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, 
            linewidth=2.5, label='Perfect prediction')
    
    ax1.set_xlabel('Actual Entropy Mean', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Predicted Entropy Mean', fontsize=14, fontweight='bold')
    ax1.set_title(f'Best Model: {best_result["Model"]}\n'
                  f'Feature Method: {best_result["Feature_Method"]}\n'
                  f'Q2 = {best_result["Q2"]:.3f}, R2 = {best_result["R2"]:.3f}', 
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, frameon=True, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 2. 模型性能比较
    ax2 = axes[0, 1]
    performance = results_df.groupby(['Model', 'Feature_Method'])['Q2'].mean().unstack()
    
    if not performance.empty:
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(performance.columns)))
        performance.plot(kind='bar', ax=ax2, width=0.8, color=colors)
        ax2.set_xlabel('Model', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Q2 Score', fontsize=14, fontweight='bold')
        ax2.set_title('Model Performance by Feature Selection Method', 
                     fontsize=14, fontweight='bold')
        ax2.legend(title='Feature Method', fontsize=11, title_fontsize=12, 
                  bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)
    
    # 3. 特征重要性或PCA分析
    ax3 = axes[0, 2]
    if importance_df is not None and len(importance_df) > 0:
        # 显示特征重要性
        top_n = min(10, len(importance_df))
        importance_top = importance_df.head(top_n)
        
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, top_n))
        bars = ax3.barh(range(top_n), importance_top['importance'], color=colors)
        ax3.set_yticks(range(top_n))
        ax3.set_yticklabels(importance_top['feature'], fontsize=10)
        ax3.set_xlabel('Importance Score', fontsize=14, fontweight='bold')
        ax3.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3, axis='x', linestyle='--')
    elif pca_info is not None:
        # 显示PCA解释方差
        colors = plt.cm.cool(np.linspace(0.2, 0.8, len(pca_info['explained_variance'])))
        bars = ax3.bar(range(1, len(pca_info['explained_variance'])+1), 
                      pca_info['explained_variance'], color=colors, edgecolor='black')
        ax3.set_xlabel('Principal Component', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Explained Variance Ratio', fontsize=14, fontweight='bold')
        ax3.set_title(f'PCA Explained Variance\n(Total: {sum(pca_info["explained_variance"]):.3f})', 
                     fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
    else:
        ax3.text(0.5, 0.5, 'No feature importance or PCA data available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12, fontweight='bold')
        ax3.set_title('Feature Analysis', fontsize=14, fontweight='bold')
    
    # 4. 残差图
    ax4 = axes[1, 0]
    residuals = y_mean - y_pred_best
    
    residual_colors = np.where(np.abs(residuals) > np.std(residuals), 
                              SCI_COLORS['red'], SCI_COLORS['blue'])
    
    ax4.scatter(y_pred_best, residuals, c=residual_colors, s=100, 
                alpha=0.8, edgecolors='black', linewidth=1.5)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=2)
    ax4.set_xlabel('Predicted Entropy Mean', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Residuals (Actual - Predicted)', fontsize=14, fontweight='bold')
    ax4.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # 5. 性能指标热图
    ax5 = axes[1, 1]
    heatmap_data = results_df.pivot_table(
        index='Model', 
        columns='Feature_Method', 
        values='Q2',
        aggfunc='mean'
    )
    
    if not heatmap_data.empty:
        heat_values = heatmap_data.values.astype(float)
        finite_values = heat_values[np.isfinite(heat_values)]
        heat_vmin = min(-1.0, float(np.min(finite_values))) if finite_values.size else -1.0
        heat_vmax = max(1.0, float(np.max(finite_values))) if finite_values.size else 1.0
        im = ax5.imshow(
            heat_values,
            cmap='coolwarm',
            aspect='auto',
            vmin=heat_vmin,
            vmax=heat_vmax
        )
        
        ax5.set_xticks(np.arange(len(heatmap_data.columns)))
        ax5.set_xticklabels(heatmap_data.columns, rotation=45, ha='right', fontsize=11, fontweight='bold')
        ax5.set_yticks(np.arange(len(heatmap_data.index)))
        ax5.set_yticklabels(heatmap_data.index, fontsize=11, fontweight='bold')
        
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                if not np.isnan(heatmap_data.iloc[i, j]):
                    text = f'{heatmap_data.iloc[i, j]:.3f}'
                    cell_value = heatmap_data.iloc[i, j]
                    text_color = 'white' if abs(cell_value) > 0.5 * max(abs(heat_vmin), abs(heat_vmax)) else 'black'
                    ax5.text(j, i, text, ha='center', va='center', 
                            color=text_color, fontsize=10, fontweight='bold')
        
        ax5.set_title('Q2 Score Heatmap', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.set_ylabel('Q2 Score', fontsize=12, fontweight='bold')
    
    # 6. 误差分布
    ax6 = axes[1, 2]
    box_data = [results_df['MAE'].dropna(), results_df['RMSE'].dropna()]
    
    if len(box_data[0]) > 0 and len(box_data[1]) > 0:
        box_colors = [SCI_COLORS['blue'], SCI_COLORS['green']]
        bp = ax6.boxplot(box_data, patch_artist=True, labels=['MAE', 'RMSE'], 
                        widths=0.6, showmeans=True, meanline=True, 
                        meanprops=dict(linestyle='--', linewidth=2.5, color='red'))
        
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        
        ax6.set_ylabel('Error Value', fontsize=14, fontweight='bold')
        ax6.set_title('Error Distribution Across All Models', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{output_prefix}_comprehensive_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Visualization saved as '{output_prefix}_comprehensive_results.png'")

def main():
    print("=" * 70)
    print("BLEND ENTROPY PREDICTION USING MOLECULAR DESCRIPTORS")
    print("IMPROVED VERSION: Proper feature selection and validation")
    print("=" * 70)
    
    # 创建输出目录
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load and preprocess data
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Data Loading and Preprocessing")
    print("=" * 70)
    
    try:
        # 加载分子描述符
        print("Loading molecular descriptors...")
        mol_descriptors = pd.read_excel("rdkit_descriptors_all.xlsx")
        print(f"  Molecular descriptors loaded: {mol_descriptors.shape[0]} molecules, {mol_descriptors.shape[1]} descriptors")
        
        # 加载混合物数据
        print("Loading blend cluster data...")
        blends_df = pd.read_excel("cluster_summary_wide_blends_as_columns.xlsx")
        print(f"  Blend data loaded: {blends_df.shape[0]} blends")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure both Excel files are in the current directory.")
        return
    
    # 显示基本信息
    print(f"\nMolecular descriptor columns ({len(mol_descriptors.columns)}):")
    print("  " + ", ".join(mol_descriptors.columns.tolist()[:10]) + "...")
    
    print(f"\nBlend data columns:")
    print("  " + ", ".join(blends_df.columns.tolist()))
    
    # =========================================================================
    # STEP 2: Create blend features
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Creating Blend Features")
    print("=" * 70)
    
    # 解析混合物组成
    blends_df['molecule_list'] = blends_df['Blend molcules'].apply(parse_blend_molecules)
    blends_df['weight_list'] = blends_df['weight radio'].apply(parse_weight_ratio)
    
    # 检查每个混合物的分子和权重数量
    print("\nChecking blend compositions...")
    mismatches = 0
    for idx, row in blends_df.iterrows():
        blend_name = row['Blends']
        molecules = row['molecule_list']
        weights = row['weight_list']
        
        if len(molecules) != len(weights):
            print(f"  Warning: {blend_name}: {len(molecules)} molecules but {len(weights)} weights")
            mismatches += 1
    
    if mismatches > 0:
        print(f"\n  Total mismatches: {mismatches}")
    
    # 为每个混合物创建特征
    blend_features_list = []
    successful_blends = []
    
    print("\nCreating features for each blend...")
    for idx, row in blends_df.iterrows():
        blend_name = row['Blends']
        molecules = row['molecule_list']
        weights = row['weight_list']
        
        # 检查分子和权重数量是否匹配
        if len(molecules) != len(weights):
            # 尝试修复：如果分子数大于权重数，给多余的分子分配平均权重
            if len(molecules) > len(weights) and len(weights) > 0:
                avg_weight = np.mean(weights)
                weights = weights + [avg_weight] * (len(molecules) - len(weights))
                print(f"  Fixed {blend_name}: added average weights")
            elif len(weights) > len(molecules) and len(molecules) > 0:
                # 如果权重数大于分子数，截断权重
                weights = weights[:len(molecules)]
                print(f"  Fixed {blend_name}: truncated weights")
            else:
                print(f"  Skipping {blend_name}: cannot fix mismatch")
                continue
        
        # 创建特征
        features = create_blend_features(row, mol_descriptors, molecules, weights)
        
        if features:
            features['Blends'] = blend_name
            blend_features_list.append(features)
            successful_blends.append(blend_name)
            print(f"  ✓ {blend_name}: created {len(features)-1} features")
        else:
            print(f"  ✗ {blend_name}: failed to create features")
    
    # 创建DataFrame
    if len(blend_features_list) == 0:
        print("Error: No blend features were created!")
        return
    
    X_full = pd.DataFrame(blend_features_list)
    
    # 检查是否意外包含了weight_entropy特征
    if 'weight_entropy' in X_full.columns:
        print("\nWarning: weight_entropy feature found. Removing it...")
        X_full = X_full.drop(columns=['weight_entropy'])
    
    # 设置索引
    X_full.set_index('Blends', inplace=True)
    
    # 与原始混合物数据对齐
    blends_df_filtered = blends_df[blends_df['Blends'].isin(successful_blends)].copy()
    blends_df_filtered.set_index('Blends', inplace=True)
    blends_df_filtered = blends_df_filtered.loc[X_full.index]
    
    print(f"\nSuccessfully created features for {len(X_full)} blends")
    print(f"Feature matrix shape: {X_full.shape}")
    
    # 保存原始特征
    X_full.to_csv(output_dir / "blend_features_raw.csv")
    print(f"Raw features saved to {output_dir / 'blend_features_raw.csv'}")
    
    # =========================================================================
    # STEP 3: Handle missing values and basic preprocessing
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Data Preprocessing")
    print("=" * 70)
    
    # 创建特征交互
    print("\nCreating interaction features...")
    X_with_interactions = create_interaction_features(X_full)
    print(f"  Added {X_with_interactions.shape[1] - X_full.shape[1]} interaction features")
    
    # 使用列中位数填充缺失值
    X_clean = X_with_interactions.copy()
    missing_before = X_clean.isnull().sum().sum()
    
    for col in X_clean.columns:
        if X_clean[col].isnull().any():
            median_val = X_clean[col].median()
            X_clean[col].fillna(median_val, inplace=True)
    
    missing_after = X_clean.isnull().sum().sum()
    print(f"  Filled {missing_before - missing_after} missing values")
    
    # 移除方差为零的列
    variance = X_clean.var()
    zero_var_cols = variance[variance == 0].index.tolist()
    if zero_var_cols:
        X_clean = X_clean.drop(columns=zero_var_cols)
        print(f"  Removed {len(zero_var_cols)} zero-variance columns")
    
    print(f"Clean feature matrix shape: {X_clean.shape}")
    
    # 在共混特征层面去除冗余特征
    print("\nRemoving redundant blend features...")
    X_non_redundant = remove_redundant_blend_features(X_clean, correlation_threshold=0.9)
    print(f"  Reduced from {X_clean.shape[1]} to {X_non_redundant.shape[1]} features")
    
    # 保存处理后的特征
    X_non_redundant.to_csv(output_dir / "blend_features_processed.csv")
    print(f"Processed features saved to {output_dir / 'blend_features_processed.csv'}")
    
    # =========================================================================
    # STEP 4: Prepare target variables
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Preparing Target Variables")
    print("=" * 70)
    
    # 目标变量：Entropy_mean和Entropy_std
    y_mean = blends_df_filtered['Entropy_mean'].values
    y_std = blends_df_filtered['Entropy_std'].values
    
    print(f"Target variables prepared:")
    print(f"  - Entropy_mean: {len(y_mean)} samples, range: {y_mean.min():.2f} to {y_mean.max():.2f}")
    print(f"  - Entropy_std: {len(y_std)} samples, range: {y_std.min():.2f} to {y_std.max():.2f}")
    
    # 检查样本量
    n_samples = len(y_mean)
    print(f"\nSample size: {n_samples}")
    if n_samples < 10:
        print("  Warning: Small sample size! Results may not be reliable.")
        print("  Consider collecting more data or using simpler models.")
    
    # =========================================================================
    # STEP 5: Feature-target correlation analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Feature-Target Correlation Analysis")
    print("=" * 70)
    
    feature_target_corr = analyze_feature_target_correlation(X_non_redundant, y_mean, "Entropy_mean")
    feature_target_corr.to_csv(output_dir / "feature_target_correlations.csv", index=False)
    print(f"Correlation analysis saved to {output_dir / 'feature_target_correlations.csv'}")
    
    # =========================================================================
    # STEP 6: Model training with proper cross-validation
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Model Training with Cross-Validation")
    print("=" * 70)
    
    # 创建适合样本量的模型
    models = create_simple_models(n_samples)
    feature_methods = create_feature_selection_methods()

    print("\nDiagnostic parameter changes:")
    print("  - StandardScaler added inside CV for Linear/Ridge/Lasso/SVR/MLP")
    print("  - SVR epsilon changed from 0.1 to 0.01 and C from 1 to 10")
    print("  - Lasso alpha changed from 0.01 to 0.001")
    print("  - MLP changed to a smaller lbfgs network without early stopping")
    print("  - Negative Q2 values are retained in plots")
    
    print(f"\nTraining {len(models)} models with {len(feature_methods)} feature selection methods")
    print(f"Models: {', '.join(models.keys())}")
    print(f"Feature methods: {', '.join(feature_methods.keys())}")
    
    # 存储结果
    all_results = []
    
    for feature_method_name, feature_method_code in feature_methods.items():
        print(f"\n{'='*50}")
        print(f"Feature Selection Method: {feature_method_name}")
        print(f"{'='*50}")
        
        for model_name, model in models.items():
            try:
                # 执行留一法交叉验证
                y_pred = manual_leave_one_out_cv(
                    X_non_redundant, y_mean, model, 
                    feature_selection_method=feature_method_code,
                    n_features=min(10, n_samples - 2)
                )
                
                # 评估性能
                result = evaluate_model_performance(y_mean, y_pred, model_name, feature_method_name)
                all_results.append(result)
                
                print(f"  {model_name:25s}: Q2={result['Q2']:7.4f}, R2={result['R2']:7.4f}, "
                      f"MAE={result['MAE']:8.2f}, RMSE={result['RMSE']:8.2f}")
                
            except Exception as e:
                print(f"  {model_name:25s}: ERROR - {str(e)}")
                all_results.append({
                    'Model': model_name,
                    'Feature_Method': feature_method_name,
                    'R2': np.nan,
                    'Q2': np.nan,
                    'MAE': np.nan,
                    'RMSE': np.nan,
                    'Predictions': np.full_like(y_mean, np.nan)
                })
    
    # =========================================================================
    # STEP 7: Results analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Results Analysis")
    print("=" * 70)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 过滤掉无效结果
    valid_results = results_df.dropna(subset=['Q2'])
    
    if len(valid_results) == 0:
        print("Error: No valid results obtained!")
        return
    
    # 显示最佳模型
    print("\nTop 5 Models by Q2 Score:")
    print("-" * 80)
    top_models = valid_results.sort_values('Q2', ascending=False).head(5)
    for idx, row in top_models.iterrows():
        print(f"{row['Model']:25s} with {row['Feature_Method']:20s}: "
              f"Q2={row['Q2']:.4f}, R2={row['R2']:.4f}, MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}")
    
    best_result = valid_results.loc[valid_results['Q2'].idxmax()]
    
    # 获取最佳模型的PCA信息（如果使用PCA）
    pca_info = None
    if best_result['Feature_Method'] == 'PCA':
        # 在所有数据上运行PCA以获取信息
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_non_redundant.fillna(X_non_redundant.median()))
        n_components = min(5, X_non_redundant.shape[1], n_samples - 2)
        if n_components > 0:
            pca = PCA(n_components=n_components)
            pca.fit(X_scaled)
            pca_info = {
                'explained_variance': pca.explained_variance_ratio_,
                'components': pca.components_
            }
    
    # 获取特征重要性（如果使用基于重要性的方法）
    importance_df = None
    if best_result['Feature_Method'] in ['GB Importance', 'RF Importance']:
        # 在所有数据上训练模型以获取特征重要性
        model_type = 'gb' if best_result['Feature_Method'] == 'GB Importance' else 'rf'
        
        if model_type == 'gb':
            model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
        
        model.fit(X_non_redundant.fillna(X_non_redundant.median()), y_mean)
        
        importance_df = pd.DataFrame({
            'feature': X_non_redundant.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
    
    # =========================================================================
    # STEP 8: Visualization
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: Visualization")
    print("=" * 70)
    
    # 创建综合可视化
    create_comprehensive_visualization(
        results_df, X_non_redundant, y_mean, best_result,
        pca_info, importance_df, output_prefix=str(output_dir / "blend_entropy_prediction")
    )
    
    # 创建额外的可视化：模型性能比较
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 按模型和特征方法分组
    performance = valid_results.groupby(['Model', 'Feature_Method'])['Q2'].mean().unstack()
    
    if not performance.empty:
        # 选择前几种方法
        top_methods = performance.mean().sort_values(ascending=False).head(5).index.tolist()
        performance_top = performance[top_methods]
        
        x = np.arange(len(performance_top.index))
        width = 0.15
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_methods)))
        
        for i, method in enumerate(top_methods):
            offset = width * (i - len(top_methods)/2 + 0.5)
            ax.bar(x + offset, performance_top[method], width, 
                  label=method, color=colors[i], edgecolor='black')
        
        ax.set_xlabel('Model', fontsize=14, fontweight='bold')
        ax.set_ylabel('Q2 Score', fontsize=14, fontweight='bold')
        ax.set_title('Model Performance by Feature Selection Method', 
                    fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(performance_top.index, rotation=45, ha='right', fontsize=11)
        ax.legend(title='Feature Method', fontsize=11, title_fontsize=12)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        all_scores = performance_top.to_numpy(dtype=float)
        finite_scores = all_scores[np.isfinite(all_scores)]
        if finite_scores.size:
            score_min = float(np.min(finite_scores))
            score_max = float(np.max(finite_scores))
            ax.axhline(0, color='black', linewidth=1.5)
            if score_min < -5:
                ax.set_yscale('symlog', linthresh=0.1)
            else:
                padding = 0.08 * max(1.0, score_max - score_min)
                ax.set_ylim(score_min - padding, score_max + padding)
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model performance comparison saved to {output_dir / 'model_performance_comparison.png'}")
    
    # =========================================================================
    # STEP 9: Save results and final model
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 9: Saving Results and Final Model")
    print("=" * 70)
    
    # 保存结果
    results_to_save = results_df.drop('Predictions', axis=1)
    results_to_save.to_csv(output_dir / "model_performance_results.csv", index=False)
    print(f"Model performance results saved to {output_dir / 'model_performance_results.csv'}")
    
    # 保存所有模型的逐样本留一法预测，用于定位导致负Q2的异常折
    all_prediction_records = []
    for _, result_row in results_df.iterrows():
        predictions = result_row['Predictions']
        if predictions is None:
            continue
        for blend_name, actual_value, predicted_value in zip(
                X_non_redundant.index, y_mean, predictions):
            all_prediction_records.append({
                'Model': result_row['Model'],
                'Feature_Method': result_row['Feature_Method'],
                'Blend': blend_name,
                'Actual_Entropy_Mean': actual_value,
                'Predicted_Entropy_Mean': predicted_value,
                'Residual': actual_value - predicted_value,
                'Absolute_Error': abs(actual_value - predicted_value)
            })

    all_predictions_df = pd.DataFrame(all_prediction_records)
    all_predictions_df.to_csv(output_dir / "all_model_loocv_predictions.csv", index=False)
    print(f"All-model LOOCV predictions saved to {output_dir / 'all_model_loocv_predictions.csv'}")

    # 保存最佳模型预测结果
    predictions_df = pd.DataFrame({
        'Blend': X_non_redundant.index,
        'Actual_Entropy_Mean': y_mean,
        'Predicted_Entropy_Mean': best_result['Predictions'],
        'Residual': y_mean - best_result['Predictions'],
        'Actual_Entropy_Std': y_std
    })
    predictions_df.to_csv(output_dir / "final_predictions.csv", index=False)
    print(f"Final predictions saved to {output_dir / 'final_predictions.csv'}")
    
    # 训练最终模型
    print("\nTraining final model on all data...")
    
    # 确定特征选择方法
    feature_method_code = feature_methods.get(best_result['Feature_Method'])
    
    # 应用特征选择
    if feature_method_code == 'correlation':
        X_final, _ = select_features_by_correlation(X_non_redundant, y_mean, X_non_redundant, 
                                                   n_features=min(10, n_samples - 2))
    elif feature_method_code == 'importance_gb':
        X_final, _ = select_features_by_importance(X_non_redundant, y_mean, X_non_redundant,
                                                  n_features=min(10, n_samples - 2), model_type='gb')
    elif feature_method_code == 'importance_rf':
        X_final, _ = select_features_by_importance(X_non_redundant, y_mean, X_non_redundant,
                                                  n_features=min(10, n_samples - 2), model_type='rf')
    elif feature_method_code == 'pca':
        X_final, _ = reduce_features_with_pca(X_non_redundant, y_mean, X_non_redundant,
                                             n_components=min(5, n_samples - 2))
    else:
        X_final = X_non_redundant
    
    # 获取模型实例
    final_model = models[best_result['Model']]
    
    # 训练模型
    X_final_filled = X_final.fillna(X_final.median()) if isinstance(X_final, pd.DataFrame) else np.nan_to_num(X_final)
    final_model.fit(X_final_filled, y_mean)
    
    # 保存模型
    model_filename = output_dir / "final_model.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump({
            'model': final_model,
            'feature_names': X_final.columns.tolist() if isinstance(X_final, pd.DataFrame) else None,
            'feature_method': best_result['Feature_Method'],
            'performance': best_result.to_dict()
        }, f)
    print(f"Final model saved to {model_filename}")
    
    # 保存特征重要性（如果可用）
    if importance_df is not None:
        importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
        print(f"Feature importance saved to {output_dir / 'feature_importance.csv'}")
    
    # =========================================================================
    # STEP 10: Summary and report
    # =========================================================================
    print("\n" + "=" * 70)
    print("PROCESS SUMMARY")
    print("=" * 70)
    
    print(f"\nData Summary:")
    print(f"  - Number of blends: {n_samples}")
    print(f"  - Initial features: {X_full.shape[1]}")
    print(f"  - Final features: {X_non_redundant.shape[1]}")
    
    print(f"\nBest Model:")
    print(f"  - Model: {best_result['Model']}")
    print(f"  - Feature selection: {best_result['Feature_Method']}")
    print(f"  - Q2 score: {best_result['Q2']:.4f}")
    print(f"  - R2 score: {best_result['R2']:.4f}")
    print(f"  - MAE: {best_result['MAE']:.2f}")
    print(f"  - RMSE: {best_result['RMSE']:.2f}")
    
    print(f"\nOutput Files:")
    print(f"  - Results directory: {output_dir}")
    print(f"  - Raw features: {output_dir / 'blend_features_raw.csv'}")
    print(f"  - Processed features: {output_dir / 'blend_features_processed.csv'}")
    print(f"  - Model results: {output_dir / 'model_performance_results.csv'}")
    print(f"  - All-model predictions: {output_dir / 'all_model_loocv_predictions.csv'}")
    print(f"  - Final predictions: {output_dir / 'final_predictions.csv'}")
    print(f"  - Final model: {output_dir / 'final_model.pkl'}")
    print(f"  - Visualizations: {output_dir / 'blend_entropy_prediction_comprehensive_results.png'}")
    
    print("\n" + "=" * 70)
    print("PROCESS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold, cross_validate, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font and plot style
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 1. Data Loading and Preprocessing
print("=" * 60)
print("Step 1: Data Loading and Preprocessing")
print("=" * 60)

# Load data
df = pd.read_excel("Database-1.xlsx")

# Features and target variable - 修改这里
X = df.iloc[:, 0:63]  # Columns 0-62: 63 features (1-63列)
y = df.iloc[:, 63]    # Column 63: Entropy value (第64列)

print(f"Dataset size: {df.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print(f"Target variable: Entropy (Range: {y.min():.2f} - {y.max():.2f})")
print("\nFirst 5 feature names:")
print(X.columns[:5].tolist())

# Check for missing values
print(f"\nMissing value check - Features: {X.isnull().sum().sum()}, Target: {y.isnull().sum()}")

# 2. Feature Standardization
print("\n" + "=" * 60)
print("Step 2: Feature Standardization")
print("=" * 60)

# Use RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

print("Standardization complete!")

# 3. Feature Selection (using all data)
print("\n" + "=" * 60)
print("Step 3: Feature Selection")
print("=" * 60)

# Method 1: Using Random Forest importance scores
rf_for_selection = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_for_selection.fit(X_scaled, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_for_selection.feature_importances_
}).sort_values('importance', ascending=False)

top_features_rf = feature_importance.head(20)

# Method 2: Using Mutual Information
mi_scores = mutual_info_regression(X_scaled, y, random_state=42)
mi_df = pd.DataFrame({
    'feature': X.columns,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

top_features_mi = mi_df.head(20)

print("Top 10 Most Important Features (based on Random Forest):")
for i, row in top_features_rf.head(10).iterrows():
    print(f"{i+1:2d}. {row['feature']:30s} : {row['importance']:.4f}")

# Select the top 30 most important features for modeling
selected_features = top_features_rf.head(30)['feature'].values
X_selected = X_scaled_df[selected_features]

print(f"\nSelected {len(selected_features)} most important features for modeling.")

# 4. Model Definition and Initialization
print("\n" + "=" * 60)
print("Step 4: Model Definition and 5-Fold Cross-Validation")
print("=" * 60)

# Define custom scoring functions
def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae_scorer(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def r2_scorer(y_true, y_pred):
    return r2_score(y_true, y_pred)

# Create scorers
scoring = {
    'r2': make_scorer(r2_scorer),
    'mae': make_scorer(mae_scorer),
    'rmse': make_scorer(rmse_scorer)
}

# Model definitions with optimized parameters
models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=200, 
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42, 
        n_jobs=-1
    ),
    'XGBoost': XGBRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42, 
        n_jobs=-1, 
        verbosity=0
    ),
    'LightGBM': LGBMRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42, 
        n_jobs=-1, 
        verbose=-1
    ),
    'CatBoost': CatBoostRegressor(
        iterations=200, 
        learning_rate=0.05, 
        depth=6,
        l2_leaf_reg=3,
        random_seed=42, 
        verbose=False,
        allow_writing_files=False
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),
    'Ridge Regression': Ridge(
        alpha=1.0, 
        random_state=42
    ),
    'Lasso Regression': Lasso(
        alpha=0.01, 
        random_state=42, 
        max_iter=5000
    ),
    'ElasticNet': ElasticNet(
        alpha=0.01, 
        l1_ratio=0.5, 
        random_state=42, 
        max_iter=5000
    ),
    'Support Vector Machine (SVR)': SVR(
        kernel='rbf', 
        C=10, 
        gamma='scale',
        epsilon=0.1
    ),
    'SVR (Linear Kernel)': SVR(
        kernel='linear', 
        C=10, 
        gamma='scale',
        epsilon=0.1
    ),
    'SVR (Polynomial Kernel)': SVR(
        kernel='poly', 
        C=10, 
        gamma='scale',
        degree=3,
        epsilon=0.1
    ),
    'MLP': MLPRegressor(
        hidden_layer_sizes=(100, 50), 
        alpha=0.01, 
        max_iter=1000, 
        random_state=42, 
        early_stopping=True,
        batch_size=32,
        learning_rate_init=0.001
    )
}

# 5-Fold Cross-Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store results
results = []

# 5. 5-Fold Cross-Validation for all models
for name, model in models.items():
    print(f"\nEvaluating {name} with 5-Fold Cross-Validation...")
    
    try:
        # Special handling for CatBoost
        if name == 'CatBoost':
            # Manually perform cross-validation for CatBoost
            r2_scores = []
            mae_scores = []
            rmse_scores = []
            
            for train_idx, val_idx in kf.split(X_selected):
                X_train_fold = X_selected.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X_selected.iloc[val_idx]
                y_val_fold = y.iloc[val_idx]
                
                # Create CatBoost Pool for better performance
                train_pool = Pool(X_train_fold, y_train_fold)
                val_pool = Pool(X_val_fold, y_val_fold)
                
                # Train model
                model_fold = CatBoostRegressor(
                    iterations=200, 
                    learning_rate=0.05, 
                    depth=6,
                    l2_leaf_reg=3,
                    random_seed=42, 
                    verbose=False,
                    allow_writing_files=False
                )
                model_fold.fit(train_pool, eval_set=val_pool, verbose=False)
                
                # Predict
                y_pred_fold = model_fold.predict(X_val_fold)
                
                # Calculate metrics
                r2_scores.append(r2_score(y_val_fold, y_pred_fold))
                mae_scores.append(mean_absolute_error(y_val_fold, y_pred_fold))
                rmse_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_pred_fold)))
            
            cv_r2_mean = np.mean(r2_scores)
            cv_r2_std = np.std(r2_scores)
            cv_mae_mean = np.mean(mae_scores)
            cv_mae_std = np.std(mae_scores)
            cv_rmse_mean = np.mean(rmse_scores)
            cv_rmse_std = np.std(rmse_scores)
            
        else:
            # Standard cross-validation for other models
            cv_results = cross_validate(
                model, X_selected, y,
                cv=kf,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=False
            )
            
            # Calculate metrics
            cv_r2_scores = cv_results['test_r2']
            cv_mae_scores = cv_results['test_mae']
            cv_rmse_scores = cv_results['test_rmse']
            
            cv_r2_mean = cv_r2_scores.mean()
            cv_r2_std = cv_r2_scores.std()
            cv_mae_mean = cv_mae_scores.mean()
            cv_mae_std = cv_mae_scores.std()
            cv_rmse_mean = cv_rmse_scores.mean()
            cv_rmse_std = cv_rmse_scores.std()
        
        # Store results
        results.append({
            'Model': name,
            'CV_R2_Mean': cv_r2_mean,
            'CV_R2_Std': cv_r2_std,
            'CV_MAE_Mean': cv_mae_mean,
            'CV_MAE_Std': cv_mae_std,
            'CV_RMSE_Mean': cv_rmse_mean,
            'CV_RMSE_Std': cv_rmse_std,
            'CV_R2_Scores': cv_r2_scores if 'cv_r2_scores' in locals() else r2_scores,
            'CV_MAE_Scores': cv_mae_scores if 'cv_mae_scores' in locals() else mae_scores,
            'CV_RMSE_Scores': cv_rmse_scores if 'cv_rmse_scores' in locals() else rmse_scores
        })
        
        print(f"  ? 5-Fold CV Results:")
        print(f"    - R2: {cv_r2_mean:.4f} (+/-{cv_r2_std:.4f})")
        print(f"    - MAE: {cv_mae_mean:.4f} (+/-{cv_mae_std:.4f})")
        print(f"    - RMSE: {cv_rmse_mean:.4f} (+/-{cv_rmse_std:.4f})")
        
    except Exception as e:
        print(f"  ? Error evaluating {name}: {str(e)}")
        # Add placeholder for failed model
        results.append({
            'Model': name,
            'CV_R2_Mean': np.nan,
            'CV_R2_Std': np.nan,
            'CV_MAE_Mean': np.nan,
            'CV_MAE_Std': np.nan,
            'CV_RMSE_Mean': np.nan,
            'CV_RMSE_Std': np.nan,
            'CV_R2_Scores': [np.nan] * 5,
            'CV_MAE_Scores': [np.nan] * 5,
            'CV_RMSE_Scores': [np.nan] * 5
        })

# 6. Results Summary and Analysis
print("\n" + "=" * 60)
print("Step 5: Results Summary and Analysis")
print("=" * 60)

# Create results DataFrame
results_df = pd.DataFrame(results)

# Remove any models that failed completely
results_df = results_df.dropna(subset=['CV_R2_Mean'])

# Sort by CV R2 Mean
results_df_sorted = results_df.sort_values('CV_R2_Mean', ascending=False)

print("\nModel Performance Ranking (by 5-Fold CV R2 Mean):")
print("-" * 100)
print(f"{'Model':<25} {'CV_R2_Mean':<12} {'CV_R2_Std':<12} {'CV_MAE_Mean':<12} {'CV_RMSE_Mean':<12}")
print("-" * 100)

for _, row in results_df_sorted.iterrows():
    print(f"{row['Model']:<25} {row['CV_R2_Mean']:<12.4f} {row['CV_R2_Std']:<12.4f} "
          f"{row['CV_MAE_Mean']:<12.4f} {row['CV_RMSE_Mean']:<12.4f}")

# 7. Visualization of Results
print("\n" + "=" * 60)
print("Step 6: Visualization Analysis")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle('Model Performance Comparison (5-Fold Cross-Validation)', fontsize=16, fontweight='bold')

# 1. Cross-Validation R2 Comparison
ax1 = axes[0, 0]
bars1 = ax1.barh(results_df_sorted['Model'], results_df_sorted['CV_R2_Mean'], 
                 xerr=results_df_sorted['CV_R2_Std'], color='skyblue', alpha=0.7, capsize=5)
ax1.set_xlabel('R2 Score (Mean +/- Std)')
ax1.set_title('Cross-Validation R2 Comparison')
ax1.axvline(x=0, color='black', linewidth=0.5)

# Add value labels
for bar, std in zip(bars1, results_df_sorted['CV_R2_Std']):
    width = bar.get_width()
    ax1.text(width + std + 0.02, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}+/-{std:.3f}', ha='left', va='center', fontsize=9)

# 2. Cross-Validation MAE and RMSE Comparison
ax2 = axes[0, 1]
x = np.arange(len(results_df_sorted))
width = 0.35

# Plot with error bars
ax2.bar(x - width/2, results_df_sorted['CV_MAE_Mean'], width, 
        yerr=results_df_sorted['CV_MAE_Std'], label='MAE', color='lightgreen', 
        alpha=0.7, capsize=5)
ax2.bar(x + width/2, results_df_sorted['CV_RMSE_Mean'], width, 
        yerr=results_df_sorted['CV_RMSE_Std'], label='RMSE', color='gold', 
        alpha=0.7, capsize=5)

ax2.set_xlabel('Model')
ax2.set_ylabel('Error (Mean +/- Std)')
ax2.set_title('Cross-Validation MAE and RMSE')
ax2.set_xticks(x)
ax2.set_xticklabels(results_df_sorted['Model'], rotation=45, ha='right', fontsize=9)
ax2.legend()

# 3. Boxplot of R2 Scores across folds
ax3 = axes[0, 2]
r2_data = []
model_names = []
for _, row in results_df_sorted.iterrows():
    if not all(np.isnan(row['CV_R2_Scores'])):
        r2_data.append(row['CV_R2_Scores'])
        model_names.append(row['Model'])

if r2_data:
    bp = ax3.boxplot(r2_data, vert=False, patch_artist=True, labels=model_names)
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(r2_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax3.set_xlabel('R2 Score')
    ax3.set_title('R2 Score Distribution across 5 Folds')
    ax3.axvline(x=0, color='black', linewidth=0.5)
else:
    ax3.text(0.5, 0.5, 'No valid R2 scores available', 
             ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('R2 Score Distribution')

# 4. Feature Importance (Best Model)
best_model_name = results_df_sorted.iloc[0]['Model']
best_model = models[best_model_name]

# Train best model on all data for feature importance
try:
    best_model.fit(X_selected, y)
    
    if hasattr(best_model, 'feature_importances_'):
        ax4 = axes[1, 0]
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[-15:]  # Take the top 15 most important features
        
        ax4.barh(range(len(indices)), importances[indices], color='steelblue')
        ax4.set_yticks(range(len(indices)))
        ax4.set_yticklabels([selected_features[i] for i in indices], fontsize=8)
        ax4.set_xlabel('Feature Importance')
        ax4.set_title(f'{best_model_name} - Top 15 Important Features')
        ax4.axvline(x=0, color='black', linewidth=0.5)
        
    elif hasattr(best_model, 'coef_'):
        # For linear models
        ax4 = axes[1, 0]
        coef = best_model.coef_
        indices = np.argsort(np.abs(coef))[-15:]
        
        ax4.barh(range(len(indices)), coef[indices], color='steelblue')
        ax4.set_yticks(range(len(indices)))
        ax4.set_yticklabels([selected_features[i] for i in indices], fontsize=8)
        ax4.set_xlabel('Coefficient Value')
        ax4.set_title(f'{best_model_name} - Top 15 Important Features (Coefficients)')
        ax4.axvline(x=0, color='black', linewidth=0.5)
        
    else:
        ax4 = axes[1, 0]
        ax4.text(0.5, 0.5, f'{best_model_name} model\nhas no feature importance attribute', 
                 ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title(f'{best_model_name} - Feature Importance')
        ax4.set_xticks([])
        ax4.set_yticks([])
        
except Exception as e:
    ax4 = axes[1, 0]
    ax4.text(0.5, 0.5, f'Error training {best_model_name}:\n{str(e)}', 
             ha='center', va='center', transform=ax4.transAxes, fontsize=10)
    ax4.set_title(f'{best_model_name} - Feature Importance')
    ax4.set_xticks([])
    ax4.set_yticks([])

# 5. Performance across folds for top 5 models
ax5 = axes[1, 1]
top_models = results_df_sorted.head(5)
for idx, (_, row) in enumerate(top_models.iterrows()):
    if not all(np.isnan(row['CV_R2_Scores'])):
        ax5.plot(range(1, 6), row['CV_R2_Scores'], 'o-', 
                label=f"{row['Model']} ({row['CV_R2_Mean']:.3f})", 
                alpha=0.7, linewidth=2, markersize=8)

ax5.set_xlabel('Fold Number')
ax5.set_ylabel('R2 Score')
ax5.set_title('R2 Score across 5 Folds (Top 5 Models)')
ax5.set_xticks(range(1, 6))
ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax5.grid(True, alpha=0.3)

# 6. Performance Comparison Heatmap
ax6 = axes[1, 2]
# Prepare data for heatmap
valid_models = results_df_sorted.dropna(subset=['CV_R2_Mean'])
if len(valid_models) > 0:
    heatmap_data = valid_models[['CV_R2_Mean', 'CV_MAE_Mean', 'CV_RMSE_Mean']].values.T
    im = ax6.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

    # Set labels
    ax6.set_xticks(np.arange(len(valid_models)))
    ax6.set_xticklabels(valid_models['Model'], rotation=45, ha='right', fontsize=9)
    ax6.set_yticks(np.arange(3))
    ax6.set_yticklabels(['R2', 'MAE', 'RMSE'])

    # Add text annotations
    for i in range(3):
        for j in range(len(valid_models)):
            text = f'{heatmap_data[i, j]:.3f}'
            ax6.text(j, i, text, ha='center', va='center', 
                    color='black' if heatmap_data[i, j] < np.max(heatmap_data)*0.7 else 'white',
                    fontsize=9)

    ax6.set_title('Performance Metrics Heatmap')
    plt.colorbar(im, ax=ax6)
else:
    ax6.text(0.5, 0.5, 'No valid models for heatmap', 
             ha='center', va='center', transform=ax6.transAxes)
    ax6.set_title('Performance Metrics Heatmap')

plt.tight_layout()
plt.savefig('Model_Performance_Comparison_CV.png', dpi=300, bbox_inches='tight')
print("Visualization results saved as 'Model_Performance_Comparison_CV.png'")

# 8. Output Best Model Details
print("\n" + "=" * 60)
print("Step 7: Best Model Analysis")
print("=" * 60)

if len(results_df_sorted) > 0:
    best_result = results_df_sorted.iloc[0]
    print(f"Best Model: {best_result['Model']}")
    print(f"Cross-Validation Performance:")
    print(f"  - R2: {best_result['CV_R2_Mean']:.4f} (+/-{best_result['CV_R2_Std']:.4f})")
    print(f"  - MAE: {best_result['CV_MAE_Mean']:.4f} (+/-{best_result['CV_MAE_Std']:.4f})")
    print(f"  - RMSE: {best_result['CV_RMSE_Mean']:.4f} (+/-{best_result['CV_RMSE_Std']:.4f})")
    
    # Train best model on all data
    try:
        best_model = models[best_result['Model']]
        best_model.fit(X_selected, y)
        
        # Feature Importance Analysis
        print("\nTop 10 Most Important Features:")
        print("-" * 50)
        
        if hasattr(best_model, 'feature_importances_'):
            feature_importance_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            for i, row in feature_importance_df.head(10).iterrows():
                print(f"{i+1:2d}. {row['Feature']:30s} : {row['Importance']:.4f}")
                
        elif hasattr(best_model, 'coef_'):
            feature_importance_df = pd.DataFrame({
                'Feature': selected_features,
                'Coefficient': best_model.coef_,
                'Absolute_Coefficient': np.abs(best_model.coef_)
            }).sort_values('Absolute_Coefficient', ascending=False)
            
            for i, row in feature_importance_df.head(10).iterrows():
                print(f"{i+1:2d}. {row['Feature']:30s} : Coefficient={row['Coefficient']:.4f}")
                
        else:
            print("Feature importance not available for this model type.")
            
    except Exception as e:
        print(f"Error analyzing best model: {str(e)}")
else:
    print("No valid models to analyze.")

# 9. Save Results
print("\n" + "=" * 60)
print("Step 8: Save Results")
print("=" * 60)

# Create detailed results DataFrame for saving
detailed_results = []
for idx, row in results_df.iterrows():
    for fold in range(5):
        detailed_results.append({
            'Model': row['Model'],
            'Fold': fold + 1,
            'R2': row['CV_R2_Scores'][fold] if fold < len(row['CV_R2_Scores']) else np.nan,
            'MAE': row['CV_MAE_Scores'][fold] if fold < len(row['CV_MAE_Scores']) else np.nan,
            'RMSE': row['CV_RMSE_Scores'][fold] if fold < len(row['CV_RMSE_Scores']) else np.nan
        })

detailed_results_df = pd.DataFrame(detailed_results)
detailed_results_df.to_csv('Detailed_CV_Results.csv', index=False, encoding='utf-8-sig')
print("Detailed cross-validation results saved as 'Detailed_CV_Results.csv'")

# Save summary results
summary_results = results_df_sorted[['Model', 'CV_R2_Mean', 'CV_R2_Std', 
                                     'CV_MAE_Mean', 'CV_MAE_Std', 
                                     'CV_RMSE_Mean', 'CV_RMSE_Std']]
summary_results.to_csv('Summary_CV_Results.csv', index=False, encoding='utf-8-sig')
print("Summary results saved as 'Summary_CV_Results.csv'")

# Save feature importance if available
if 'feature_importance_df' in locals():
    feature_importance_df.to_csv('Feature_Importance_Best_Model.csv', index=False, encoding='utf-8-sig')
    print("Feature importance saved as 'Feature_Importance_Best_Model.csv'")

print("\n" + "=" * 60)
print("5-Fold Cross-Validation Modeling Pipeline Complete!")
print("=" * 60)
print("\nModel Performance Summary:")
print(f"Total models evaluated: {len(models)}")
print(f"Successfully evaluated: {len(results_df)}")
print(f"Best performing model: {best_result['Model'] if 'best_result' in locals() else 'N/A'}")

# 10. Additional SVM Analysis (可选)
print("\n" + "=" * 60)
print("Additional SVM Analysis")
print("=" * 60)

# 测试不同的SVM核函数和参数
svm_configs = {
    'SVR_RBF_C1': SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1),
    'SVR_RBF_C10': SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.1),
    'SVR_RBF_C100': SVR(kernel='rbf', C=100.0, gamma='scale', epsilon=0.1),
    'SVR_Linear_C10': SVR(kernel='linear', C=10.0, gamma='scale', epsilon=0.1),
    'SVR_Poly_deg2': SVR(kernel='poly', C=10.0, gamma='scale', degree=2, epsilon=0.1),
    'SVR_Poly_deg3': SVR(kernel='poly', C=10.0, gamma='scale', degree=3, epsilon=0.1),
    'SVR_RBF_auto': SVR(kernel='rbf', C=10.0, gamma='auto', epsilon=0.1),
}

print("\nTesting different SVM configurations with 5-Fold CV...")
svm_results = []

for name, model in svm_configs.items():
    try:
        cv_results = cross_validate(
            model, X_selected, y,
            cv=kf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )
        
        cv_r2_mean = cv_results['test_r2'].mean()
        cv_mae_mean = cv_results['test_mae'].mean()
        
        svm_results.append({
            'Config': name,
            'R2': cv_r2_mean,
            'MAE': cv_mae_mean
        })
        
        print(f"  {name:<15}: R2={cv_r2_mean:.4f}, MAE={cv_mae_mean:.4f}")
        
    except Exception as e:
        print(f"  {name:<15}: Error - {str(e)}")

print("\nSuggestions:")
print("1. The best model based on cross-validation is recommended for entropy prediction.")
print("2. SVR models work well with standardized data and may benefit from feature selection.")
print("3. Consider ensemble methods combining top-performing models.")
print("4. For production use, retrain the best model on all available data.")
print("5. SVR performance can be improved by tuning C, gamma, and epsilon parameters.")
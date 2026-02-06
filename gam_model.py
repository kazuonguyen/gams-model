"""
GAM MODEL - Generalized Additive Models cho Death Prediction
=============================================================
- Bỏ mean features, chỉ dùng MIN/MAX/STD/SUM
- GAMs: f(y) = s(x1) + s(x2) + ... + s(xn)
- Mỗi feature được fit bằng spline riêng (non-linear)
- Interpretable: xem được ảnh hưởng từng feature
- Tốt cho small dataset (557 samples)
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from pygam import LinearGAM, s, te, l
from pygam.terms import TermList

sns.set_style("whitegrid")


# ============================================================
# FEATURE SELECTION: Chỉ MIN/MAX/STD/SUM - BỎ MEAN
# ============================================================

# Weather features - chỉ min, max, std, sum (KHÔNG mean)
MIN_MAX_FEATURES = [
    # Temperature extremes
    'Max_Temp_C_max',           # Nhiệt độ cao nhất trong tuần
    'Max_Temp_C_std',           # Biến động nhiệt độ cao
    'Min_Temp_C_min',           # Nhiệt độ thấp nhất trong tuần
    'Min_Temp_C_std',           # Biến động nhiệt độ thấp
    'Temp_Range_max',           # Biên độ nhiệt lớn nhất
    
    # Rainfall extremes
    'Rainfall_mm_sum',          # Tổng lượng mưa
    'Rainfall_mm_max',          # Lượng mưa cực đại 1 ngày
    'Rainfall_mm_rainy_days',   # Số ngày mưa
    
    # Evaporation & Radiation (sum = tổng tuần)
    'Evaporation_mm_sum',       # Tổng bốc hơi
    'Radiation_MJ_m2_sum',      # Tổng bức xạ
    
    # Vapour pressure extremes
    'Vapour_Pressure_hPa_min',  # Áp suất hơi thấp nhất
    'Vapour_Pressure_hPa_max',  # Áp suất hơi cao nhất
    
    # Humidity extremes
    'RH_at_Max_Temp_pct_min',   # Độ ẩm thấp nhất khi nóng nhất
    'RH_at_Min_Temp_pct_max',   # Độ ẩm cao nhất khi lạnh nhất
]

# Air Quality features (max + count - không mean)
AQ_MIN_MAX_FEATURES = [
    'AQI_weekly_max',           # AQI cực đại trong tuần
    'Bad_days_count',           # Số ngày không khí xấu
    'Main_pollutant_AQI',       # Chất ô nhiễm chính
]


def load_data_minmax():
    """Load data chỉ với min/max features."""
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, 'dataset', 'Weather_Death_Weekly_Merged_MinMax_ImprovedTargets_v2.csv')
    
    df_weather = pd.read_csv(data_path)
    
    # Merge air quality
    aq_path = os.path.join(base_dir, 'dataset', 'Air_Quality_Weekly_Evaluation.csv')
    if os.path.exists(aq_path):
        df_aq = pd.read_csv(aq_path)
        df = pd.merge(df_weather, df_aq, on=['Year', 'Week'], how='inner')
    else:
        df = df_weather
    
    df = df.sort_values(['Year', 'Week']).reset_index(drop=True)
    
    # Select min/max features only
    available_features = [f for f in MIN_MAX_FEATURES + AQ_MIN_MAX_FEATURES if f in df.columns]
    
    print(f"\n[Features] Using {len(available_features)} MIN/MAX features (NO mean):")
    for f in available_features:
        print(f"  → {f}")
    
    df[available_features] = df[available_features].ffill().bfill().fillna(0)
    
    return df, available_features


def create_gam_features(df, feature_cols):
    """
    Create advanced features cho GAM - tập trung min/max extremes.
    """
    df_result = df.copy()
    new_features = list(feature_cols)
    
    # 1. Extreme temperature spread (max cao - min thấp)
    if 'Max_Temp_C_max' in df.columns and 'Min_Temp_C_min' in df.columns:
        df_result['Temp_Extreme_Spread'] = df['Max_Temp_C_max'] - df['Min_Temp_C_min']
        new_features.append('Temp_Extreme_Spread')
    
    # 2. Cold stress index (nhiệt thấp + ẩm cao = nguy hiểm)
    if 'Min_Temp_C_min' in df.columns and 'RH_at_Min_Temp_pct_max' in df.columns:
        df_result['Cold_Stress'] = (1 - df['Min_Temp_C_min']) * df['RH_at_Min_Temp_pct_max']
        new_features.append('Cold_Stress')
    
    # 3. Heat stress index (nhiệt cao + ẩm thấp = nguy hiểm)
    if 'Max_Temp_C_max' in df.columns and 'RH_at_Max_Temp_pct_min' in df.columns:
        df_result['Heat_Stress'] = df['Max_Temp_C_max'] * (1 - df['RH_at_Max_Temp_pct_min'])
        new_features.append('Heat_Stress')
    
    # 4. Air quality extreme impact
    if 'AQI_weekly_max' in df.columns and 'Bad_days_count' in df.columns:
        df_result['AQ_Extreme_Impact'] = df['AQI_weekly_max'] * df['Bad_days_count']
        new_features.append('AQ_Extreme_Impact')
    
    # 5. Rainfall intensity (max rain / rainy days)
    if 'Rainfall_mm_max' in df.columns and 'Rainfall_mm_rainy_days' in df.columns:
        df_result['Rainfall_Intensity'] = df['Rainfall_mm_max'] / (df['Rainfall_mm_rainy_days'] + 0.1)
        new_features.append('Rainfall_Intensity')
    
    # 6. Vapour pressure range
    if 'Vapour_Pressure_hPa_max' in df.columns and 'Vapour_Pressure_hPa_min' in df.columns:
        df_result['Vapour_Pressure_Range'] = df['Vapour_Pressure_hPa_max'] - df['Vapour_Pressure_hPa_min']
        new_features.append('Vapour_Pressure_Range')
    
    # 7. Lag features (1-4 weeks trước) cho Deaths
    if 'Deaths' in df.columns:
        for lag in [1, 2, 3, 4]:
            col_name = f'Deaths_lag{lag}'
            df_result[col_name] = df['Deaths'].shift(lag)
            new_features.append(col_name)
    
    # 8. Lag features cho key min/max features
    key_features = ['Max_Temp_C_max', 'Min_Temp_C_min', 'AQI_weekly_max']
    for feat in key_features:
        if feat in df.columns:
            for lag in [1, 2]:
                col_name = f'{feat}_lag{lag}'
                df_result[col_name] = df[feat].shift(lag)
                new_features.append(col_name)
    
    # 9. Rolling extremes (2, 4 week windows)
    if 'Deaths' in df.columns:
        for window in [2, 4]:
            df_result[f'Deaths_roll{window}_max'] = df['Deaths'].rolling(window).max()
            df_result[f'Deaths_roll{window}_min'] = df['Deaths'].rolling(window).min()
            new_features.append(f'Deaths_roll{window}_max')
            new_features.append(f'Deaths_roll{window}_min')
    
    # 10. Week-of-year seasonality
    if 'Week' in df.columns:
        df_result['Week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
        df_result['Week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)
        new_features.append('Week_sin')
        new_features.append('Week_cos')
    
    # Fill NaNs from lags/rolling
    df_result[new_features] = df_result[new_features].fillna(method='bfill').fillna(0)
    
    # Replace inf
    df_result = df_result.replace([np.inf, -np.inf], 0)
    
    print(f"\n[GAM Features] Total: {len(new_features)} features")
    print(f"  Base min/max: {len(feature_cols)}")
    print(f"  Engineered: {len(new_features) - len(feature_cols)}")
    
    return df_result, new_features


def train_gam(X_train, y_train, X_val, y_val, n_splines=20, lam_search=True):
    """
    Train GAM model với grid search cho lambda.
    
    GAM: Deaths = s(feat1) + s(feat2) + ... + s(featN)
    Mỗi s() là smooth spline function
    """
    n_features = X_train.shape[1]
    
    # Build term list: one spline per feature
    terms = s(0, n_splines=n_splines)
    for i in range(1, n_features):
        terms = terms + s(i, n_splines=n_splines)
    
    if lam_search:
        print("\n[GAM] Grid search for optimal smoothing (lambda)...")
        
        # Try different lambda values
        best_gam = None
        best_val_loss = float('inf')
        best_lam = None
        
        for lam_exp in np.linspace(-3, 3, 20):
            lam = 10 ** lam_exp
            
            gam = LinearGAM(terms)
            gam.fit(X_train, y_train)
            
            # Validate
            val_pred = gam.predict(X_val)
            val_loss = mean_squared_error(y_val, val_pred)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_gam = gam
                best_lam = lam
        
        print(f"[GAM] Best lambda: {best_lam:.6f}")
        print(f"[GAM] Best val MSE: {best_val_loss:.4f}")
        
        # Also try gridsearch built-in
        print("[GAM] Running built-in gridsearch...")
        gam_gs = LinearGAM(terms).gridsearch(
            X_train, y_train,
            lam=np.logspace(-3, 3, 30)
        )
        
        val_pred_gs = gam_gs.predict(X_val)
        val_loss_gs = mean_squared_error(y_val, val_pred_gs)
        
        print(f"[GAM] Gridsearch val MSE: {val_loss_gs:.4f}")
        
        if val_loss_gs < best_val_loss:
            best_gam = gam_gs
            best_val_loss = val_loss_gs
            print("[GAM] → Using gridsearch model")
        else:
            print("[GAM] → Using manual search model")
        
        return best_gam
    else:
        gam = LinearGAM(terms)
        gam.gridsearch(X_train, y_train)
        return gam


def evaluate_and_plot(gam, X_test, y_test, feature_names, scaler_y, output_dir):
    """Evaluate GAM và tạo biểu đồ."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Predictions
    pred_scaled = gam.predict(X_test)
    
    # Inverse transform
    predictions = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    actuals = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
    
    print(f"\n{'='*70}")
    print("GAM MODEL RESULTS")
    print(f"{'='*70}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"{'='*70}")
    
    # ============================================================
    # PLOT 1: Main evaluation (6 subplots)
    # ============================================================
    fig = plt.figure(figsize=(24, 16))
    
    # 1a. Scatter với REGRESSION LINE (uốn theo chấm, không phải 45°)
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(actuals, predictions, alpha=0.6, c='#2ecc71', s=80,
               edgecolors='black', linewidth=1, zorder=3)
    
    # Perfect fit line (45°) - mờ
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], '--', color='gray',
            linewidth=1.5, alpha=0.5, label='Perfect y=x')
    
    # REGRESSION LINE - uốn theo chấm (polynomial fit)
    sort_idx = np.argsort(actuals)
    actuals_sorted = actuals[sort_idx]
    preds_sorted = predictions[sort_idx]
    
    # Polynomial regression degree 3
    z = np.polyfit(actuals_sorted, preds_sorted, 3)
    p = np.poly1d(z)
    x_smooth = np.linspace(actuals.min(), actuals.max(), 200)
    ax1.plot(x_smooth, p(x_smooth), 'r-', linewidth=3, label='Trend Line', zorder=5)
    
    # LOWESS smoothing cho đường uốn tốt hơn
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        lowess_result = lowess(preds_sorted, actuals_sorted, frac=0.4)
        ax1.plot(lowess_result[:, 0], lowess_result[:, 1], 'b-', linewidth=3,
                label='LOWESS Fit', zorder=6)
    except ImportError:
        pass
    
    ax1.set_xlabel('Actual Deaths', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Predicted Deaths', fontsize=14, fontweight='bold')
    ax1.set_title('GAM Predictions vs Actual', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    stats_text = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.4f}\nMAPE: {mape:.1f}%"
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='gold', alpha=0.9),
            fontsize=13, fontweight='bold')
    
    # 1b. Time series
    ax2 = plt.subplot(2, 3, 2)
    t = np.arange(len(actuals))
    ax2.plot(t, actuals, 'o-', label='Actual', color='black',
            linewidth=3, markersize=6, alpha=0.8, zorder=3)
    ax2.plot(t, predictions, 's-', label='Predicted', color='#2ecc71',
            linewidth=2.5, markersize=5, alpha=0.8, zorder=2)
    ax2.set_xlabel('Sample Index', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Deaths', fontsize=14, fontweight='bold')
    ax2.set_title('Time Series Predictions', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 1c. Residuals
    ax3 = plt.subplot(2, 3, 3)
    residuals = actuals - predictions
    ax3.scatter(predictions, residuals, alpha=0.6, c='#3498db', s=60,
               edgecolors='black', linewidth=0.5)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=3)
    ax3.set_xlabel('Predicted Deaths', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Residuals', fontsize=14, fontweight='bold')
    ax3.set_title('Residual Plot', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 1d. Error distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(residuals, bins=25, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=3)
    ax4.axvline(x=residuals.mean(), color='orange', linestyle='--', linewidth=2,
               label=f'Mean: {residuals.mean():.2f}')
    ax4.set_xlabel('Residuals', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax4.set_title('Residual Distribution', fontsize=16, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 1e. Absolute errors
    ax5 = plt.subplot(2, 3, 5)
    abs_errors = np.abs(residuals)
    ax5.plot(t, abs_errors, 'o-', color='#e74c3c', linewidth=2.5, markersize=5)
    ax5.axhline(y=mae, color='black', linestyle='--', linewidth=3,
               label=f'MAE: {mae:.2f}')
    ax5.set_xlabel('Sample Index', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Absolute Error', fontsize=14, fontweight='bold')
    ax5.set_title('Errors Over Time', fontsize=16, fontweight='bold')
    ax5.legend(fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    # 1f. Prediction intervals
    ax6 = plt.subplot(2, 3, 6)
    
    # GAM confidence intervals
    try:
        pred_intervals = gam.prediction_intervals(X_test, width=0.95)
        ci_low = scaler_y.inverse_transform(pred_intervals[:, 0].reshape(-1, 1)).flatten()
        ci_high = scaler_y.inverse_transform(pred_intervals[:, 1].reshape(-1, 1)).flatten()
        
        ax6.fill_between(t, ci_low, ci_high, alpha=0.2, color='#3498db', label='95% CI')
    except Exception:
        pass
    
    ax6.plot(t, actuals, 'o-', label='Actual', color='black',
            linewidth=3, markersize=6, alpha=0.8)
    ax6.plot(t, predictions, 's-', label='Predicted', color='#2ecc71',
            linewidth=2.5, markersize=5, alpha=0.8)
    ax6.set_xlabel('Sample Index', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Deaths', fontsize=14, fontweight='bold')
    ax6.set_title('Predictions with 95% Confidence Interval', fontsize=16, fontweight='bold')
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gam_evaluation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {output_dir}/gam_evaluation.png")
    
    # ============================================================
    # PLOT 2: Feature importance (partial dependence)
    # ============================================================
    n_features = min(len(feature_names), 12)
    fig, axes = plt.subplots(3, 4, figsize=(24, 16))
    axes = axes.flatten()
    
    for i in range(n_features):
        ax = axes[i]
        try:
            XX = gam.generate_X_grid(term=i, n=100)
            pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
            
            ax.plot(XX[:, i], pdep, 'b-', linewidth=2)
            ax.fill_between(XX[:, i], confi[:, 0], confi[:, 1],
                          alpha=0.2, color='blue')
            ax.set_title(f'{feature_names[i][:25]}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'N/A\n{str(e)[:30]}', transform=ax.transAxes,
                   ha='center', va='center', fontsize=9)
    
    # Hide unused
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('GAM Partial Dependence Plots (Feature Effects)', 
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gam_feature_effects.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/gam_feature_effects.png")
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape),
        'predictions': predictions.tolist(),
        'actuals': actuals.tolist()
    }


def main():
    """Main GAM training pipeline."""
    
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, 'gam_results')
    
    print("\n" + "="*70)
    print("GAM MODEL - Generalized Additive Models")
    print("Focus: MIN/MAX features only (NO mean)")
    print("="*70)
    
    # Load data
    df, feature_cols = load_data_minmax()
    
    # Create GAM-specific features
    df_gam, all_features = create_gam_features(df, feature_cols)
    
    # Prepare X, y
    X = df_gam[all_features].values.astype(np.float64)
    y = df_gam['Deaths'].values.astype(np.float64)
    
    print(f"\n[Data] X shape: {X.shape}, y shape: {y.shape}")
    
    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Time-based split: 80% train, 10% val, 10% test
    train_size = int(len(X) * 0.8)
    val_size = int(len(X) * 0.1)
    
    X_train_raw = X[:train_size]
    y_train_raw = y[:train_size]
    X_val_raw = X[train_size:train_size+val_size]
    y_val_raw = y[train_size:train_size+val_size]
    X_test_raw = X[train_size+val_size:]
    y_test_raw = y[train_size+val_size:]
    
    # Fit scalers on train only
    X_train = scaler_X.fit_transform(X_train_raw)
    y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    
    X_val = scaler_X.transform(X_val_raw)
    y_val = scaler_y.transform(y_val_raw.reshape(-1, 1)).flatten()
    
    X_test = scaler_X.transform(X_test_raw)
    y_test = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()
    
    print(f"\n[Split] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train GAM
    print("\n" + "="*70)
    print("TRAINING GAM")
    print("="*70)
    
    gam = train_gam(X_train, y_train, X_val, y_val, n_splines=25, lam_search=True)
    
    # Statistics
    print(f"\n[GAM Summary]")
    print(f"  GCV score: {gam.statistics_['GCV']:.6f}")
    print(f"  Pseudo R²: {gam.statistics_['pseudo_r2']['explained_deviance']:.4f}")
    print(f"  AIC: {gam.statistics_['AIC']:.2f}")
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    results = evaluate_and_plot(gam, X_test, y_test, all_features, scaler_y, output_dir)
    
    # Save model and scalers
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'gam_model.pkl'), 'wb') as f:
        pickle.dump({
            'gam': gam,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'feature_names': all_features,
            'results': results
        }, f)
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Model saved: {output_dir}/gam_model.pkl")
    print(f"✅ Results saved: {output_dir}/results.json")
    
    # Compare with previous models
    print(f"\n{'='*70}")
    print("COMPARISON WITH PREVIOUS MODELS")
    print(f"{'='*70}")
    print(f"\n{'Model':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
    print("-" * 50)
    print(f"{'Baseline (DLinear)':<20} {'7.21':<10} {'9.00':<10} {'-0.08':<10}")
    print(f"{'Improved (TCN+TF)':<20} {'6.35':<10} {'8.04':<10} {'0.14':<10}")
    print(f"{'Optimal (Attention)':<20} {'6.12':<10} {'7.64':<10} {'0.22':<10}")
    print(f"{'Ensemble (5x)':<20} {'6.27':<10} {'8.29':<10} {'0.08':<10}")
    print(f"{'GAM (min/max)':<20} {results['mae']:<10.4f} {results['rmse']:<10.4f} {results['r2']:<10.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

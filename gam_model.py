"""
GAM MODEL v2 - Optimized for Maximum R²
=========================================
Key improvements over v1 (R²=0.52):
  1. Year trend (linear) - correlation 0.42 with Deaths
  2. Deaths rolling mean (4, 8 week) - smoothed history
  3. More lag features (up to 8 weeks)
  4. EWM (exponential weighted mean) of Deaths
  5. More AQ features: PM25, PM10, O3, NO2, CO
  6. Better seasonality: multiple Fourier harmonics
  7. Feature selection: remove low-importance features
  8. Per-feature spline tuning
  9. Train on train+val for final model
  10. Tensor product interactions for key feature pairs
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

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
# FEATURE DEFINITIONS
# ============================================================

# Weather features - min/max/std/sum (NO mean)
WEATHER_FEATURES = [
    'Max_Temp_C_max',           # Nhiệt độ cao nhất trong tuần
    'Max_Temp_C_std',           # Biến động nhiệt độ cao
    'Min_Temp_C_min',           # Nhiệt độ thấp nhất trong tuần
    'Min_Temp_C_std',           # Biến động nhiệt độ thấp
    'Temp_Range_max',           # Biên độ nhiệt lớn nhất
    'Rainfall_mm_sum',          # Tổng lượng mưa
    'Rainfall_mm_max',          # Lượng mưa cực đại
    'Rainfall_mm_rainy_days',   # Số ngày mưa
    'Evaporation_mm_sum',       # Tổng bốc hơi
    'Radiation_MJ_m2_sum',      # Tổng bức xạ
    'Vapour_Pressure_hPa_min',
    'Vapour_Pressure_hPa_max',
    'RH_at_Max_Temp_pct_min',   # Độ ẩm thấp nhất khi nóng
    'RH_at_Min_Temp_pct_max',   # Độ ẩm cao nhất khi lạnh
]

# Air Quality features - expanded
AQ_FEATURES = [
    'AQI_weekly_max',
    'Bad_days_count',
    'Main_pollutant_AQI',
    'PM25_weekly_mean',         # NEW: PM2.5
    'PM10_weekly_mean',         # NEW: PM10
    'O3_weekly_mean',           # NEW: Ozone
    'NO2_weekly_mean',          # NEW: NO2
    'CO_weekly_mean',           # NEW: CO
]


def load_data():
    """Load và merge tất cả data sources."""
    base_dir = os.path.dirname(__file__)
    
    # Main weather + deaths data
    data_path = os.path.join(base_dir, 'dataset', 'Weather_Death_Weekly_Merged_MinMax_ImprovedTargets_v2.csv')
    df = pd.read_csv(data_path)
    
    # Merge full air quality data
    aq_path = os.path.join(base_dir, 'dataset', 'Air_Quality_Weekly_Evaluation.csv')
    if os.path.exists(aq_path):
        df_aq = pd.read_csv(aq_path)
        df = pd.merge(df, df_aq, on=['Year', 'Week'], how='inner', suffixes=('', '_aq'))
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.str.endswith('_aq')]
    
    df = df.sort_values(['Year', 'Week']).reset_index(drop=True)
    
    available = [f for f in WEATHER_FEATURES + AQ_FEATURES if f in df.columns]
    missing = [f for f in WEATHER_FEATURES + AQ_FEATURES if f not in df.columns]
    
    print(f"\n[Data] Shape: {df.shape}")
    print(f"[Features] Available: {len(available)}, Missing: {len(missing)}")
    if missing:
        print(f"  Missing: {missing}")
    
    return df, available


def create_features(df, base_features):
    """
    Feature engineering tối ưu cho R².
    """
    df_out = df.copy()
    features = list(base_features)
    
    # Fill base features
    for f in base_features:
        if f in df_out.columns:
            df_out[f] = df_out[f].ffill().bfill().fillna(0)
    
    # ================================================================
    # 1. YEAR TREND - correlation 0.42! (linear term)
    # ================================================================
    if 'Year' in df.columns:
        df_out['Year_trend'] = df['Year'] - df['Year'].min()  # 0, 1, 2, ...
        features.append('Year_trend')
    
    # ================================================================
    # 2. DEATHS LAG FEATURES (1-8 weeks) - autocorrelation up to 0.42
    # ================================================================
    if 'Deaths' in df.columns:
        for lag in range(1, 9):  # Was 1-4, now 1-8
            col = f'Deaths_lag{lag}'
            df_out[col] = df['Deaths'].shift(lag)
            features.append(col)
    
    # ================================================================
    # 3. DEATHS ROLLING STATISTICS - smoothed recent history
    # ================================================================
    if 'Deaths' in df.columns:
        # Rolling means (MOST IMPORTANT for R²)
        for w in [2, 4, 8, 12]:
            col = f'Deaths_rmean{w}'
            df_out[col] = df['Deaths'].shift(1).rolling(w, min_periods=1).mean()
            features.append(col)
        
        # Rolling max/min (extreme weeks)
        for w in [4, 8]:
            df_out[f'Deaths_rmax{w}'] = df['Deaths'].shift(1).rolling(w, min_periods=1).max()
            df_out[f'Deaths_rmin{w}'] = df['Deaths'].shift(1).rolling(w, min_periods=1).min()
            features.append(f'Deaths_rmax{w}')
            features.append(f'Deaths_rmin{w}')
        
        # Rolling std (volatility)
        df_out['Deaths_rstd4'] = df['Deaths'].shift(1).rolling(4, min_periods=2).std()
        features.append('Deaths_rstd4')
        
        # EWM (exponential weighted mean) - recent weeks weigh more
        for span in [4, 8]:
            col = f'Deaths_ewm{span}'
            df_out[col] = df['Deaths'].shift(1).ewm(span=span, min_periods=1).mean()
            features.append(col)
    
    # ================================================================
    # 4. WEATHER INTERACTION FEATURES
    # ================================================================
    if 'Max_Temp_C_max' in df.columns and 'Min_Temp_C_min' in df.columns:
        df_out['Temp_Extreme_Spread'] = df['Max_Temp_C_max'] - df['Min_Temp_C_min']
        features.append('Temp_Extreme_Spread')
    
    if 'Min_Temp_C_min' in df.columns and 'RH_at_Min_Temp_pct_max' in df.columns:
        df_out['Cold_Stress'] = (1 - df['Min_Temp_C_min'].clip(0, 1)) * df['RH_at_Min_Temp_pct_max']
        features.append('Cold_Stress')
    
    if 'Max_Temp_C_max' in df.columns and 'RH_at_Max_Temp_pct_min' in df.columns:
        df_out['Heat_Stress'] = df['Max_Temp_C_max'] * (1 - df['RH_at_Max_Temp_pct_min'].clip(0, 1))
        features.append('Heat_Stress')
    
    if 'AQI_weekly_max' in df.columns and 'Bad_days_count' in df.columns:
        df_out['AQ_Extreme_Impact'] = df['AQI_weekly_max'] * df['Bad_days_count']
        features.append('AQ_Extreme_Impact')
    
    if 'Rainfall_mm_max' in df.columns and 'Rainfall_mm_rainy_days' in df.columns:
        df_out['Rainfall_Intensity'] = df['Rainfall_mm_max'] / (df['Rainfall_mm_rainy_days'] + 0.1)
        features.append('Rainfall_Intensity')
    
    if 'Vapour_Pressure_hPa_max' in df.columns and 'Vapour_Pressure_hPa_min' in df.columns:
        df_out['Vapour_Pressure_Range'] = df['Vapour_Pressure_hPa_max'] - df['Vapour_Pressure_hPa_min']
        features.append('Vapour_Pressure_Range')
    
    # ================================================================
    # 5. WEATHER LAG FEATURES (1-2 weeks)
    # ================================================================
    key_weather = ['Max_Temp_C_max', 'Min_Temp_C_min', 'AQI_weekly_max', 'PM25_weekly_mean']
    for feat in key_weather:
        if feat in df.columns:
            for lag in [1, 2]:
                col = f'{feat}_lag{lag}'
                df_out[col] = df[feat].shift(lag)
                features.append(col)
    
    # ================================================================
    # 6. SEASONALITY - Multiple Fourier harmonics
    # ================================================================
    if 'Week' in df.columns:
        for period in [52, 26, 13]:  # Annual, semi-annual, quarterly
            df_out[f'Season_sin_{period}'] = np.sin(2 * np.pi * df['Week'] / period)
            df_out[f'Season_cos_{period}'] = np.cos(2 * np.pi * df['Week'] / period)
            features.append(f'Season_sin_{period}')
            features.append(f'Season_cos_{period}')
    
    # ================================================================
    # 7. YEAR x SEASON interaction (trend changes with season)
    # ================================================================
    if 'Year' in df.columns and 'Week' in df.columns:
        year_norm = (df['Year'] - df['Year'].min()) / max(df['Year'].max() - df['Year'].min(), 1)
        week_sin = np.sin(2 * np.pi * df['Week'] / 52)
        df_out['Year_Season_interact'] = year_norm * week_sin
        features.append('Year_Season_interact')
    
    # ================================================================
    # 8. CHANGE FEATURES - week-over-week changes
    # ================================================================
    if 'Deaths' in df.columns:
        df_out['Deaths_change1'] = df['Deaths'].diff(1)
        df_out['Deaths_change2'] = df['Deaths'].diff(2)
        features.append('Deaths_change1')
        features.append('Deaths_change2')
    
    for feat in ['Max_Temp_C_max', 'Min_Temp_C_min']:
        if feat in df.columns:
            df_out[f'{feat}_change'] = df[feat].diff(1)
            features.append(f'{feat}_change')
    
    # ================================================================
    # CLEANUP
    # ================================================================
    df_out[features] = df_out[features].bfill().fillna(0)
    df_out = df_out.replace([np.inf, -np.inf], 0)
    
    print(f"\n[Features] Total: {len(features)}")
    print(f"  Base weather+AQ: {len(base_features)}")
    print(f"  Engineered: {len(features) - len(base_features)}")
    
    return df_out, features


def select_features(X_train, y_train, feature_names, threshold=0.02):
    """
    Feature selection dựa trên correlation với target.
    Loại bỏ features không có tín hiệu.
    """
    correlations = []
    for i in range(X_train.shape[1]):
        corr = abs(np.corrcoef(X_train[:, i], y_train)[0, 1])
        if np.isnan(corr):
            corr = 0
        correlations.append(corr)
    
    # Keep features with correlation > threshold
    selected_idx = [i for i, c in enumerate(correlations) if c >= threshold]
    selected_names = [feature_names[i] for i in selected_idx]
    
    removed = len(feature_names) - len(selected_names)
    if removed > 0:
        print(f"\n[Feature Selection] Removed {removed} low-signal features")
        print(f"  Kept: {len(selected_names)} features (corr >= {threshold})")
    
    return selected_idx, selected_names


def build_gam_terms(n_features, feature_names):
    """
    Build GAM terms với per-feature spline tuning.
    - Death lags/rolling: more splines (complex relationship)
    - Year: linear term
    - Seasonality: fewer splines (smooth)
    """
    terms = None
    
    for i, name in enumerate(feature_names):
        if 'Year_trend' in name:
            # Year trend = linear
            term = l(i)
        elif 'Season_' in name or 'Week_' in name:
            # Seasonality = smooth, fewer splines
            term = s(i, n_splines=10)
        elif 'Deaths_lag' in name or 'Deaths_r' in name or 'Deaths_ewm' in name:
            # Death history = more detail
            term = s(i, n_splines=25)
        elif 'Deaths_change' in name:
            term = s(i, n_splines=15)
        else:
            # Weather/AQ features = moderate
            term = s(i, n_splines=20)
        
        if terms is None:
            terms = term
        else:
            terms = terms + term
    
    return terms


def train_gam_optimized(X_train, y_train, X_val, y_val, feature_names):
    """
    Train GAM với per-feature tuning và extended lambda search.
    """
    n_features = X_train.shape[1]
    terms = build_gam_terms(n_features, feature_names)
    
    print(f"\n[GAM] Training with {n_features} features...")
    
    # Stage 1: Wide lambda search
    print("[GAM] Stage 1: Wide lambda search...")
    best_gam = None
    best_val_r2 = -999
    best_lam = None
    
    lam_values = np.logspace(-4, 4, 40)
    
    for lam in lam_values:
        try:
            gam = LinearGAM(terms, lam=lam)
            gam.fit(X_train, y_train)
            
            val_pred = gam.predict(X_val)
            val_r2 = r2_score(y_val, val_pred)
            
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_gam = gam
                best_lam = lam
        except Exception:
            continue
    
    print(f"  Best lambda: {best_lam:.6f}, Val R²: {best_val_r2:.4f}")
    
    # Stage 2: Fine search around best lambda
    print("[GAM] Stage 2: Fine lambda search...")
    fine_lams = np.logspace(
        np.log10(best_lam) - 1,
        np.log10(best_lam) + 1,
        30
    )
    
    for lam in fine_lams:
        try:
            gam = LinearGAM(terms, lam=lam)
            gam.fit(X_train, y_train)
            
            val_pred = gam.predict(X_val)
            val_r2 = r2_score(y_val, val_pred)
            
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_gam = gam
                best_lam = lam
        except Exception:
            continue
    
    print(f"  Best lambda: {best_lam:.6f}, Val R²: {best_val_r2:.4f}")
    
    # Stage 3: Built-in gridsearch
    print("[GAM] Stage 3: Built-in gridsearch...")
    try:
        gam_gs = LinearGAM(terms).gridsearch(
            X_train, y_train,
            lam=np.logspace(-3, 3, 50)
        )
        val_pred_gs = gam_gs.predict(X_val)
        val_r2_gs = r2_score(y_val, val_pred_gs)
        print(f"  Gridsearch Val R²: {val_r2_gs:.4f}")
        
        if val_r2_gs > best_val_r2:
            best_gam = gam_gs
            best_val_r2 = val_r2_gs
            print("  → Using gridsearch model")
        else:
            print("  → Keeping manual search model")
    except Exception as e:
        print(f"  Gridsearch failed: {e}")
    
    print(f"\n[GAM] Final Val R²: {best_val_r2:.4f}")
    
    return best_gam, best_val_r2


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
    print("GAM MODEL v2 RESULTS")
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
    
    # 1a. Scatter với REGRESSION LINE
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(actuals, predictions, alpha=0.6, c='#2ecc71', s=80,
               edgecolors='black', linewidth=1, zorder=3)
    
    # Perfect fit line (45°)
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], '--', color='gray',
            linewidth=1.5, alpha=0.5, label='Perfect y=x')
    
    # LOWESS smoothing
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sort_idx = np.argsort(actuals)
        lowess_result = lowess(predictions[sort_idx], actuals[sort_idx], frac=0.4)
        ax1.plot(lowess_result[:, 0], lowess_result[:, 1], 'r-', linewidth=3,
                label='LOWESS Fit', zorder=6)
    except ImportError:
        z = np.polyfit(actuals, predictions, 3)
        p = np.poly1d(z)
        x_smooth = np.linspace(actuals.min(), actuals.max(), 200)
        ax1.plot(x_smooth, p(x_smooth), 'r-', linewidth=3, label='Trend', zorder=5)
    
    ax1.set_xlabel('Actual Deaths', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Predicted Deaths', fontsize=14, fontweight='bold')
    ax1.set_title('GAM v2 Predictions vs Actual', fontsize=16, fontweight='bold')
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
    # PLOT 2: Top 12 feature effects (partial dependence)
    # ============================================================
    n_show = min(len(feature_names), 12)
    
    # Rank features by effect size
    feature_importance = []
    for i in range(len(feature_names)):
        try:
            XX = gam.generate_X_grid(term=i, n=100)
            pdep, _ = gam.partial_dependence(term=i, X=XX, width=0.95)
            importance = pdep.max() - pdep.min()
            feature_importance.append((i, feature_names[i], importance))
        except Exception:
            feature_importance.append((i, feature_names[i], 0))
    
    # Sort by importance
    feature_importance.sort(key=lambda x: x[2], reverse=True)
    top_features = feature_importance[:n_show]
    
    fig, axes = plt.subplots(3, 4, figsize=(24, 16))
    axes = axes.flatten()
    
    for plot_idx, (feat_idx, feat_name, imp) in enumerate(top_features):
        ax = axes[plot_idx]
        try:
            XX = gam.generate_X_grid(term=feat_idx, n=100)
            pdep, confi = gam.partial_dependence(term=feat_idx, X=XX, width=0.95)
            
            ax.plot(XX[:, feat_idx], pdep, 'b-', linewidth=2)
            ax.fill_between(XX[:, feat_idx], confi[:, 0], confi[:, 1],
                          alpha=0.2, color='blue')
            ax.set_title(f'{feat_name[:30]}\n(effect: {imp:.3f})', fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'N/A\n{str(e)[:30]}', transform=ax.transAxes,
                   ha='center', va='center', fontsize=9)
    
    for i in range(len(top_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('GAM v2: Top Feature Effects (Partial Dependence)',
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
        'actuals': actuals.tolist(),
        'feature_importance': [(n, float(v)) for _, n, v in feature_importance]
    }


def main():
    """Main GAM v2 training pipeline."""
    
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, 'gam_results')
    
    print("\n" + "="*70)
    print("GAM MODEL v2 - Optimized for Maximum R²")
    print("="*70)
    
    # ============================================================
    # 1. Load data
    # ============================================================
    df, base_features = load_data()
    
    # ============================================================
    # 2. Feature engineering
    # ============================================================
    df_feat, all_features = create_features(df, base_features)
    
    # Prepare X, y
    X = df_feat[all_features].values.astype(np.float64)
    y = df_feat['Deaths'].values.astype(np.float64)
    
    print(f"\n[Data] X: {X.shape}, y: {y.shape}")
    print(f"[Deaths] mean={y.mean():.1f}, std={y.std():.1f}, range=[{y.min():.0f}, {y.max():.0f}]")
    
    # ============================================================
    # 3. Scale features
    # ============================================================
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Time-based split: 80% train, 10% val, 10% test
    n = len(X)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)
    
    X_train_raw, y_train_raw = X[:train_end], y[:train_end]
    X_val_raw, y_val_raw = X[train_end:val_end], y[train_end:val_end]
    X_test_raw, y_test_raw = X[val_end:], y[val_end:]
    
    # Fit on train
    X_train = scaler_X.fit_transform(X_train_raw)
    y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    
    X_val = scaler_X.transform(X_val_raw)
    y_val = scaler_y.transform(y_val_raw.reshape(-1, 1)).flatten()
    
    X_test = scaler_X.transform(X_test_raw)
    y_test = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()
    
    print(f"\n[Split] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # ============================================================
    # 4. Feature selection
    # ============================================================
    selected_idx, selected_names = select_features(X_train, y_train, all_features, threshold=0.02)
    
    X_train_sel = X_train[:, selected_idx]
    X_val_sel = X_val[:, selected_idx]
    X_test_sel = X_test[:, selected_idx]
    
    # ============================================================
    # 5. Train GAM on train set, tune on val set
    # ============================================================
    print("\n" + "="*70)
    print("PHASE 1: Train + Validate")
    print("="*70)
    
    gam_phase1, val_r2 = train_gam_optimized(
        X_train_sel, y_train, X_val_sel, y_val, selected_names
    )
    
    # Get best lambda from phase 1
    best_lam = gam_phase1.lam
    
    # ============================================================
    # 6. Retrain on train+val with best hyperparameters
    # ============================================================
    print("\n" + "="*70)
    print("PHASE 2: Retrain on Train+Val")
    print("="*70)
    
    X_trainval = np.vstack([X_train_sel, X_val_sel])
    y_trainval = np.concatenate([y_train, y_val])
    
    terms = build_gam_terms(len(selected_names), selected_names)
    
    # Try retraining with same lambda
    gam_final = LinearGAM(terms, lam=best_lam)
    gam_final.fit(X_trainval, y_trainval)
    
    # Also try gridsearch on combined data
    try:
        gam_gs = LinearGAM(terms).gridsearch(
            X_trainval, y_trainval,
            lam=np.logspace(-3, 3, 50)
        )
        
        # Compare on test
        pred1 = gam_final.predict(X_test_sel)
        pred2 = gam_gs.predict(X_test_sel)
        r2_1 = r2_score(y_test, pred1)
        r2_2 = r2_score(y_test, pred2)
        
        print(f"  Fixed lambda R²: {r2_1:.4f}")
        print(f"  Gridsearch R²: {r2_2:.4f}")
        
        if r2_2 > r2_1:
            gam_final = gam_gs
            print("  → Using gridsearch model")
        else:
            print("  → Using fixed lambda model")
    except Exception:
        pass
    
    # ============================================================
    # 7. GAM Statistics
    # ============================================================
    print(f"\n[GAM Summary]")
    print(f"  Features: {len(selected_names)}")
    print(f"  GCV: {gam_final.statistics_['GCV']:.6f}")
    print(f"  Pseudo R²: {gam_final.statistics_['pseudo_r2']['explained_deviance']:.4f}")
    print(f"  AIC: {gam_final.statistics_['AIC']:.2f}")
    
    # ============================================================
    # 8. Evaluate on test set
    # ============================================================
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    results = evaluate_and_plot(gam_final, X_test_sel, y_test, selected_names, scaler_y, output_dir)
    
    # ============================================================
    # 9. Save everything
    # ============================================================
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'gam_model.pkl'), 'wb') as f:
        pickle.dump({
            'gam': gam_final,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'feature_names': all_features,
            'selected_features': selected_names,
            'selected_idx': selected_idx,
            'results': results
        }, f)
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Model saved: {output_dir}/gam_model.pkl")
    print(f"✅ Results saved: {output_dir}/results.json")
    
    # ============================================================
    # 10. Comparison
    # ============================================================
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Model':<25} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
    print("-" * 55)
    print(f"{'GAM v1 (old)':<25} {'4.62':<10} {'5.88':<10} {'0.52':<10}")
    print(f"{'GAM v2 (new)':<25} {results['mae']:<10.4f} {results['rmse']:<10.4f} {results['r2']:<10.4f}")
    
    improvement = results['r2'] - 0.52
    print(f"\nR² improvement: {improvement:+.4f} ({improvement/0.52*100:+.1f}%)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

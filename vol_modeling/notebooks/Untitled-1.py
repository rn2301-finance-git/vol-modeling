# %%
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats

# %%
SPLITS = {
    'train': ('2024-01-16', '2024-06-30'),
    'validate': ('2024-07-01', '2024-08-31'), 
    'test': ('2024-09-01', '2025-01-14')
}

split_name = 'train'
model_name = "transformer"
run_name = "marilyn"
# Convert dates to datetime for comparison
split_start = datetime.strptime(SPLITS[split_name][0], '%Y-%m-%d')
split_end = datetime.strptime(SPLITS[split_name][1], '%Y-%m-%d')
# Adjust this path to your inference CSV files.
data_pattern = f'/Users/raghuvar/Code/dataBAM/inference/{model_name}/{run_name}*/inference/*.5min.csv'


def load_file(filepath):
    """
    Load a CSV file and extract the date (YYYYMMDD from the filename).
    """
    df = pd.read_csv(filepath)
    # Extract the first 8 characters from the filename as the date string.
    file_date = os.path.basename(filepath)[:8]
    # Convert to datetime
    df['date'] = pd.to_datetime(file_date, format='%Y%m%d', errors='coerce')
    return df

def load_split_data(split_name, data_pattern):
    """
    Load and concatenate all CSV files (matching data_pattern) that fall into the
    given split period.
    
    Parameters:
      split_name: 'train', 'validate', or 'test'
      data_pattern: glob pattern (e.g., '/path/to/inference/*.5min.csv')
    
    Returns:
      A concatenated DataFrame.
    """
    split_start = datetime.strptime(SPLITS[split_name][0], '%Y-%m-%d')
    split_end   = datetime.strptime(SPLITS[split_name][1], '%Y-%m-%d')
    
    all_files = glob.glob(data_pattern)
    files_in_split = []
    dfs = []
    for f in all_files:
        try:
            file_date = datetime.strptime(os.path.basename(f)[:8], '%Y%m%d')
        except Exception as e:
            print(f"Error parsing date from {f}: {e}")
            continue
        if split_start <= file_date <= split_end:
            files_in_split.append(f)
            dfs.append(load_file(f))
            
    print(f"Found {len(files_in_split)} files in '{split_name}' period")
    if files_in_split:
        print(f"First file: {files_in_split[0]}")
        print(f"Last file:  {files_in_split[-1]}")
        
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def add_time_features(df):
    """
    Add time-based features from the 'minute' column (assumed format "HH:MM").
    
    Adds:
      - 'hour': integer hour (from the first two characters)
      - 'minute_in_hour': integer minutes (from the characters after ':')
      - 'minutes_from_open': minutes since market open (assumed 9:30)
    """
    df = df.copy()
    # Assume 'minute' is a string "HH:MM"
    df['hour'] = df['minute'].str[:2].astype(int)
    df['minute_in_hour'] = df['minute'].str[3:].astype(int)
    # Calculate minutes from market open (9:30)
    df['minutes_from_open'] = ((df['hour'] - 9) * 60 + df['minute_in_hour'] - 30)
    return df

# =============================================================================
# Missing Data Analysis
# =============================================================================
def analyze_missing_predictions(df):
    """
    Analyze missing predicted values.
    Reports the overall missing percentage, and plots the count of missing
    predictions by hour. Also prints the top 10 symbols with missing predictions.
    """
    missing_mask = df['predicted.volatility'].isna()
    n_missing = missing_mask.sum()
    total = len(df)
    print(f"Overall missing predicted.volatility: {n_missing} out of {total} rows ({100 * n_missing/total:.2f}%)")
    
    # Use the 'hour' column (or recompute it if needed)
    if 'hour' not in df.columns:
        df['hour'] = pd.to_datetime(df['minute'], format='%H:%M', errors='coerce').dt.hour
    missing_by_hour = df[missing_mask].groupby('hour').size()
    
    # Plot a simple bar chart for missing predictions by hour.
    plt.figure(figsize=(8, 4))
    plt.bar(missing_by_hour.index.astype(str), missing_by_hour.values, color='skyblue')
    plt.title("Missing Predictions by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Count of Missing Predictions")
    plt.tight_layout()
    plt.show()
    
    # Distribution by symbol
    missing_by_symbol = df[missing_mask].groupby('symbol').size()
    print("\nTop 10 symbols with missing predictions:")
    print(missing_by_symbol.nlargest(10))
    return missing_by_hour, missing_by_symbol


def analyze_late_missing(df, symbol='TDG', min_minute='10:00'):
    """
    Analyze missing predictions for a specific symbol after a given time.
    
    Parameters:
    -----------
    df : DataFrame with columns 'symbol', 'minute', 'predicted.volatility'
    symbol : str, symbol to analyze
    min_minute : str, only look at missing values after this time (HH:MM)
    """
    # Create missing mask
    missing_mask = df['predicted.volatility'].isna()
    
    # Filter for specific symbol and missing values
    symbol_missing = df[
        (df['symbol'] == symbol) & 
        missing_mask & 
        (df['minute'] > min_minute)
    ]
    
    # Sort by minute descending to see latest missing values first
    late_missing = symbol_missing.sort_values('minute', ascending=False)[
        ['date', 'minute', 'Y_log_vol_10min_lag_1m']
    ]
    
    print(f"\nMissing predictions for {symbol} after {min_minute}:")
    print(late_missing.head(20))  # Show top 20 latest missing values
    
    return late_missing


# =============================================================================
# Metrics and Error Analysis
# =============================================================================
def compute_overall_metrics(df, target_col='Y_log_vol_10min_lag_1m', pred_col='predicted.volatility'):
    """
    Compute overall metrics: RMSE, MAE, Pearson correlation, and Spearman correlation.
    
    Uses only NumPy for correlation calculations.
    """
    valid = df[[target_col, pred_col]].dropna()
    error = valid[pred_col] - valid[target_col]
    mse = np.mean(error**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error))
    
    # Pearson correlation using np.corrcoef
    pearson_corr = np.corrcoef(valid[pred_col].values, valid[target_col].values)[0,1]
    # Spearman correlation: compute ranks and then use np.corrcoef
    rank_pred = valid[pred_col].rank().values
    rank_target = valid[target_col].rank().values
    spearman_corr = safe_corrcoef(rank_pred, rank_target)
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr
    }
    print("Overall Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics

def analyze_predictions(df, target_col='Y_log_vol_10min_lag_1m', pred_col='predicted.volatility'):
    """
    Analyze predictions versus targets.
    Computes overall RMSE and rank correlations (grouped by date and hour),
    produces a scatter plot and an error histogram, and performs a calibration analysis
    by predicted confidence.
    """
    metrics = compute_overall_metrics(df, target_col, pred_col)
    
    # Compute absolute error and add as a new column.
    df = df.copy()
    df['vol_error'] = np.abs(df[pred_col] - df[target_col])
    
    # Scatter plot: Actual vs. Predicted
    valid = df.dropna(subset=[target_col, pred_col])
    plt.figure(figsize=(8,6))
    plt.scatter(valid[target_col], valid[pred_col], alpha=0.3, color='darkorange')
    plt.plot([valid[target_col].min(), valid[target_col].max()],
             [valid[target_col].min(), valid[target_col].max()], 'r--')
    plt.title("Predicted vs. Actual Volatility")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.show()
    
    # Histogram of absolute error
    plt.figure(figsize=(8,4))
    plt.hist(df['vol_error'].dropna(), bins=50, color='lightgreen', edgecolor='black')
    plt.title("Distribution of Absolute Error")
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    
    # Compute metrics by hour of day using NumPy correlations
    hourly_metrics = []
    for hour, group in df.groupby('hour'):
        group_valid = group.dropna(subset=[target_col, pred_col])
        if len(group_valid) == 0:
            continue
        err = group_valid[pred_col] - group_valid[target_col]
        rmse_hour = np.sqrt(np.mean(err**2))
        # Compute Spearman correlation manually:
        rank_pred = group_valid[pred_col].rank().values
        rank_target = group_valid[target_col].rank().values
        if len(rank_pred) > 1:
            corr = np.corrcoef(rank_pred, rank_target)[0,1]
        else:
            corr = np.nan
        hourly_metrics.append((hour, rmse_hour, corr))
    hourly_metrics = np.array(hourly_metrics, dtype=object)
    if hourly_metrics.size:
        hours = [int(x) for x in hourly_metrics[:,0]]
        rmse_vals = [float(x) for x in hourly_metrics[:,1]]
        corr_vals = [float(x) for x in hourly_metrics[:,2]]
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,4))
        ax1.bar([str(h) for h in hours], rmse_vals, color='steelblue')
        ax1.set_title("RMSE by Hour")
        ax1.set_xlabel("Hour")
        ax1.set_ylabel("RMSE")
    
        ax2.bar([str(h) for h in hours], corr_vals, color='salmon')
        ax2.set_title("Spearman Corr by Hour")
        ax2.set_xlabel("Hour")
        ax2.set_ylabel("Spearman Correlation")
        plt.tight_layout()
        plt.show()
    
        metrics['time_of_day'] = {'hours': hours, 'rmse': rmse_vals, 'spearman_corr': corr_vals}
    
    # Confidence calibration: group by quantile of predicted confidence
    if 'predicted.vol_confidence' in df.columns:
        # Drop NAs from predicted confidence before binning.
        conf = df['predicted.vol_confidence'].dropna()
        # Create 10 quantile bins
        quantile_bins = pd.qcut(conf, 10, duplicates='drop')
        calib = df.dropna(subset=['predicted.vol_confidence']).groupby(quantile_bins)['vol_error'].mean()
        # Plot calibration: x-axis is bin labels, y-axis is average error.
        plt.figure(figsize=(10,4))
        plt.bar([str(b) for b in calib.index], calib.values, color='mediumpurple', edgecolor='black')
        plt.xticks(rotation=45, ha='right')
        plt.title("Average Absolute Error by Predicted Confidence Quantile")
        plt.xlabel("Predicted Confidence Quantile")
        plt.ylabel("Average Absolute Error")
        plt.tight_layout()
        plt.show()
        metrics['confidence_calibration'] = calib.to_dict()
    else:
        print("Column 'predicted.vol_confidence' not found; skipping confidence calibration analysis.")
    
    return metrics

# =============================================================================
# Regime, Intraday, and Cross‐Sectional Analysis
# =============================================================================
def analyze_regimes(df, target_col='Y_log_vol_10min_lag_1m', pred_col='predicted.volatility'):
    """
    Divide data into 5 volatility regimes based on target quantiles,
    and compute RMSE, Spearman correlation (using np.corrcoef on ranks),
    and average predicted confidence for each regime.
    """
    df = df.copy()
    df['vol_regime'] = pd.qcut(df[target_col], 5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    def regime_metrics(x):
        valid = x.dropna(subset=[target_col, pred_col])
        if len(valid) < 2:
            return {'rmse': np.nan, 'spearman_corr': np.nan, 'avg_confidence': np.nan, 'n_obs': len(valid)}
        err = valid[pred_col] - valid[target_col]
        rmse_val = np.sqrt(np.mean(err**2))
        rank_pred = valid[pred_col].rank().values
        rank_target = valid[target_col].rank().values
        spearman_corr = np.corrcoef(rank_pred, rank_target)[0,1]
        avg_conf = valid['predicted.vol_confidence'].mean() if 'predicted.vol_confidence' in valid.columns else np.nan
        return {'rmse': rmse_val, 'spearman_corr': spearman_corr, 'avg_confidence': avg_conf, 'n_obs': len(valid)}
    
    regime_metrics_df = df.groupby('vol_regime').apply(lambda x: pd.Series(regime_metrics(x)))
    print("\nRegime Metrics:")
    print(regime_metrics_df)
    # Use pandas built-in plot (which calls matplotlib) for a quick view.
    regime_metrics_df[['rmse', 'spearman_corr']].plot(kind='bar', subplots=True, layout=(1,2), figsize=(14,4), legend=False)
    plt.suptitle("Metrics by Volatility Regime")
    plt.tight_layout()
    plt.show()
    
    return regime_metrics_df


def safe_corrcoef(x, y):
    # Compute standard deviations
    std_x = np.std(x)
    std_y = np.std(y)
    if std_x == 0 or std_y == 0:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def analyze_intraday_pattern(df, target_col='Y_log_vol_10min_lag_1m', pred_col='predicted.volatility'):
    """
    Analyze intraday performance by grouping by minute-of-day.
    """
    df = df.copy()
    if 'minute_in_hour' not in df.columns or 'hour' not in df.columns:
        df['hour'] = pd.to_datetime(df['minute'], format='%H:%M', errors='coerce').dt.hour
        df['minute_in_hour'] = pd.to_datetime(df['minute'], format='%H:%M', errors='coerce').dt.minute
    df['minute_of_day'] = df['hour'] * 60 + df['minute_in_hour']
    
    intraday = df.groupby('minute_of_day').apply(lambda x: pd.Series({
        'rmse': np.sqrt(np.mean((x[pred_col] - x[target_col])**2)),
        'spearman_corr': np.corrcoef(x[pred_col].rank().values, x[target_col].rank().values)[0,1] if len(x) > 1 else np.nan,
        'avg_confidence': x['predicted.vol_confidence'].mean() if 'predicted.vol_confidence' in x.columns else np.nan,
        'n_obs': len(x)
    })).reset_index()
    
    plt.figure(figsize=(12,4))
    plt.plot(intraday['minute_of_day'], intraday['rmse'], marker='o', linestyle='-')
    plt.title("Intraday RMSE Pattern")
    plt.xlabel("Minute of Day")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.show()
    
    return intraday

def compute_symbol_metrics(df, target_col='Y_log_vol_10min_lag_1m', pred_col='predicted.volatility'):
    """
    Compute per-symbol metrics (RMSE, Pearson and Spearman correlations, etc.)
    using NumPy for correlation computations.
    """
    def compute_metrics(x):
        valid = x.dropna(subset=[target_col, pred_col])
        if len(valid) < 2:
            return pd.Series({
                'rmse': np.nan,
                'pearson_corr': np.nan,
                'spearman_corr': np.nan,
                'avg_confidence': np.nan,
                'avg_vol': np.nan,
                'n_observations': len(valid)
            })
        err = valid[pred_col] - valid[target_col]
        rmse_val = np.sqrt(np.mean(err**2))
        pearson = np.corrcoef(valid[pred_col].values, valid[target_col].values)[0,1]
        rank_pred = valid[pred_col].rank().values
        rank_target = valid[target_col].rank().values
        spearman = np.corrcoef(rank_pred, rank_target)[0,1]
        avg_conf = valid['predicted.vol_confidence'].mean() if 'predicted.vol_confidence' in valid.columns else np.nan
        avg_vol = valid[target_col].mean()
        return pd.Series({
            'rmse': rmse_val,
            'pearson_corr': pearson,
            'spearman_corr': spearman,
            'avg_confidence': avg_conf,
            'avg_vol': avg_vol,
            'n_observations': len(valid)
        })
    cols = ['Y_log_vol_10min_lag_1m', 'predicted.volatility', 'predicted.vol_confidence']
    symbol_metrics = df.groupby('symbol')[cols].apply(compute_metrics)
    return symbol_metrics

# =============================================================================
# Data Quality Checks
# =============================================================================
def check_data_quality(df, target_col='Y_log_vol_10min_lag_1m', pred_col='predicted.volatility'):
    """
    Print basic data quality statistics and produce a sample time series plot.
    """
    print("Basic Data Checks:")
    print(f"  Total rows: {len(df)}")
    print("  Missing values:")
    print(df[[pred_col, target_col]].isna().sum())
    print("\n  Descriptive Statistics:")
    print(df[[pred_col, target_col]].describe())
    
    # Show a sample of data for the first symbol
    sample_symbol = df['symbol'].iloc[0]
    sample_data = df[df['symbol'] == sample_symbol].head(10)
    print(f"\nSample data for symbol {sample_symbol}:")
    print(sample_data[['minute', pred_col, target_col, 'predicted.vol_confidence']])
    
    # Plot time series for the sample symbol.
    if not sample_data.empty:
        plt.figure(figsize=(10,4))
        plt.plot(sample_data['minute'], sample_data[target_col], label="Actual", marker='o')
        plt.plot(sample_data['minute'], sample_data[pred_col], label="Predicted", linestyle='--', marker='x')
        plt.title(f"Time Series for {sample_symbol}")
        plt.xlabel("Time")
        plt.ylabel("Volatility")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# =============================================================================
# Symbol-Level Performance Analysis
# =============================================================================
def analyze_symbol_performance(df, target_col='Y_log_vol_10min_lag_1m', pred_col='predicted.volatility'):
    """
    Analyze per-symbol performance after filtering out early market minutes.
    Also apply a simple scaling adjustment so that the prediction mean matches the target mean.
    """
    df = df.copy()
    # Convert 'minute' to datetime if not already done
    df['minute'] = pd.to_datetime(df['minute'], format='%H:%M', errors='coerce')
    df['hour'] = df['minute'].dt.hour
    df['minute_int'] = df['minute'].dt.minute
    df['minutes_from_open'] = (df['hour'] - 9) * 60 + df['minute_int'] - 30
    
    # Filter out data before 30 minutes after market open
    df_clean = df[df['minutes_from_open'] >= 30].copy()
    
    # Scale predictions: adjust so that the mean prediction equals the mean target
    mean_target = df_clean[target_col].mean()
    mean_pred = df_clean[pred_col].mean()
    if mean_pred != 0:
        df_clean['scaled_pred'] = df_clean[pred_col] * mean_target / mean_pred
    else:
        df_clean['scaled_pred'] = df_clean[pred_col]
    
    def compute_metrics(x):
        valid = x.dropna(subset=[target_col, pred_col])
        if len(valid) < 2:
            return pd.Series({
                'rmse_original': np.nan,
                'rmse_scaled': np.nan,
                'spearman_corr': np.nan,
                'avg_confidence': np.nan,
                'avg_target': np.nan,
                'n_observations': len(valid),
                'n_unique_preds': valid[pred_col].nunique()
            })
        err_orig = valid[pred_col] - valid[target_col]
        err_scaled = valid['scaled_pred'] - valid[target_col]
        rmse_orig = np.sqrt(np.mean(err_orig**2))
        rmse_scaled = np.sqrt(np.mean(err_scaled**2))
        rank_pred = valid[pred_col].rank().values
        rank_target = valid[target_col].rank().values
        spearman_corr = np.corrcoef(rank_pred, rank_target)[0,1]
        return pd.Series({
            'rmse_original': rmse_orig,
            'rmse_scaled': rmse_scaled,
            'spearman_corr': spearman_corr,
            'avg_confidence': valid['predicted.vol_confidence'].mean() if 'predicted.vol_confidence' in valid.columns else np.nan,
            'avg_target': valid[target_col].mean(),
            'n_observations': len(valid),
            'n_unique_preds': valid[pred_col].nunique()
        })
    
    symbol_perf = df_clean.groupby('symbol').apply(compute_metrics)
    
    print("\nOverall Symbol-Level Performance Summary:")
    print(symbol_perf.describe())
    
    # Plot the distribution of original RMSE across symbols.
    plt.figure(figsize=(8,4))
    plt.hist(symbol_perf['rmse_original'].dropna(), bins=50, color='lightblue', edgecolor='black')
    plt.title("Distribution of Original Symbol-level RMSE")
    plt.xlabel("RMSE")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    
    # Print top 10 symbols by Spearman correlation.
    valid_corrs = symbol_perf[symbol_perf['spearman_corr'].notna()]
    if len(valid_corrs) > 0:
        top_symbols = valid_corrs.sort_values('spearman_corr', ascending=False).head(10)
        print("\nTop 10 symbols by Spearman correlation:")
        print(top_symbols[['spearman_corr', 'n_observations', 'n_unique_preds']])
    else:
        print("No valid correlations found.")
    
    return symbol_perf


# %%

print(data_pattern)
# Load data for the chosen split.
df = load_split_data(split_name, data_pattern)
if df.empty:
    raise ValueError("No data loaded. Check your file paths and date ranges.")

# Add time features.
df = add_time_features(df)

# %%

analyze_late_missing(df)

# %%


# Perform basic data quality checks.
check_data_quality(df)

# Analyze missing predictions.
analyze_missing_predictions(df)

# Compute and display overall prediction metrics and generate plots.
overall_metrics = analyze_predictions(df)

# Analyze performance across volatility regimes.
regime_metrics = analyze_regimes(df)

# Analyze intraday error patterns.
intraday_metrics = analyze_intraday_pattern(df)

# Compute cross-sectional (symbol-level) metrics.
symbol_metrics = compute_symbol_metrics(df)

# Analyze symbol-level performance with scaling adjustment.
symbol_perf = analyze_symbol_performance(df)

# Optionally, save the summaries to CSV files.
symbol_metrics.to_csv("symbol_metrics_summary.csv", index=True)
symbol_perf.to_csv("symbol_performance.csv", index=True)

# Print final summaries.
print("\nOverall Metrics:")
print(overall_metrics)
print("\nRegime Metrics:")
print(regime_metrics)
print("\nIntraday Metrics (first 10 rows):")
print(intraday_metrics.head(10))
print("\nTop 10 Symbols by Spearman Correlation:")
print(symbol_perf.sort_values('spearman_corr', ascending=False).head(10)[['spearman_corr', 'n_observations', 'n_unique_preds']])

# %%
def analyze_confidence_vol_relationship_fixed(df, conf_col='predicted.vol_confidence', 
                                           vol_col='Y_log_vol_10min_lag_1m',
                                           pred_col='predicted.volatility'):
    """
    Analyze relationship between prediction confidence and realized volatility,
    with proper handling of zero volatility cases.
    """
    # Remove NaN values
    valid_data = df.dropna(subset=[conf_col, vol_col, pred_col]).copy()
    
    # 1. Create confidence quintiles
    conf_quintiles = pd.qcut(valid_data[conf_col], 5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    valid_data['conf_quintile'] = conf_quintiles
    
    # 2. Compute absolute error
    valid_data['abs_error'] = np.abs(valid_data[pred_col] - valid_data[vol_col])
    
    # 3. Modified error metrics that don't require division by target
    quintile_stats = valid_data.groupby('conf_quintile').agg({
        vol_col: ['mean', 'std', 'count', lambda x: np.sum(x == 0)],
        'abs_error': ['mean', 'std', 'median'],
        pred_col: ['mean', 'std']
    }).round(6)
    
    # Rename the zero count column
    quintile_stats[vol_col] = quintile_stats[vol_col].rename(
        columns={'<lambda_0>': 'zero_vol_count'})
    
    print("\n1. Basic Statistics by Confidence Quintile:")
    print(quintile_stats)
    
    # 4. Accuracy bands analysis
    def compute_accuracy_bands(group):
        total = len(group)
        bands = {
            'within_10pct': np.sum(group['abs_error'] <= 0.0001) / total,  # Within 0.01%
            'within_25pct': np.sum(group['abs_error'] <= 0.00025) / total, # Within 0.025%
            'within_50pct': np.sum(group['abs_error'] <= 0.0005) / total,  # Within 0.05%
            'above_100pct': np.sum(group['abs_error'] > 0.001) / total,    # Above 0.1%
            'count': total
        }
        return pd.Series(bands)
    
    accuracy_bands = valid_data.groupby('conf_quintile').apply(compute_accuracy_bands)
    
    print("\n2. Accuracy Bands by Confidence Level:")
    print(accuracy_bands.round(4))
    
    # 5. Confidence vs Volatility Level
    vol_quintiles = pd.qcut(valid_data[vol_col], 5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    valid_data['vol_quintile'] = vol_quintiles
    
    conf_by_vol = valid_data.groupby('vol_quintile')[conf_col].agg(['mean', 'std', 'count']).round(4)
    
    print("\n3. Confidence by Volatility Level:")
    print(conf_by_vol)
    
    # 6. Day pattern analysis
    valid_data['hour'] = pd.to_datetime(valid_data['minute'], format='%H:%M').dt.hour
    
    # Compute various metrics by hour
    hourly_metrics = valid_data.groupby('hour').agg({
        conf_col: ['mean', 'std'],
        'abs_error': ['mean', 'std'],
        vol_col: ['mean', 'std']
    }).round(6)
    
    print("\n4. Hourly Pattern Analysis:")
    print(hourly_metrics)
    
    # 7. Plot key relationships
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Error Distribution by Confidence
    plt.subplot(1, 3, 1)
    error_by_conf = valid_data.groupby('conf_quintile')['abs_error'].mean()
    plt.plot(range(5), error_by_conf.values, 'bo-')
    plt.title('Mean Error by Confidence')
    plt.xticks(range(5), error_by_conf.index, rotation=45)
    
    # Plot 2: Confidence vs Actual Volatility Level
    plt.subplot(1, 3, 2)
    conf_by_vol_mean = valid_data.groupby('vol_quintile')[conf_col].mean()
    plt.plot(range(5), conf_by_vol_mean.values, 'ro-')
    plt.title('Average Confidence by Vol Level')
    plt.xticks(range(5), conf_by_vol_mean.index, rotation=45)
    
    # Plot 3: Hour of Day Pattern
    plt.subplot(1, 3, 3)
    hourly_conf = valid_data.groupby('hour')[conf_col].mean()
    plt.plot(hourly_conf.index, hourly_conf.values, 'go-')
    plt.title('Confidence by Hour')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'quintile_stats': quintile_stats,
        'accuracy_bands': accuracy_bands,
        'conf_by_vol': conf_by_vol,
        'hourly_metrics': hourly_metrics
    }

# %%
analyze_confidence_vol_relationship_fixed(df)

# %%
df.head()

# %%
def analyze_prediction_signs(df, conf_col='predicted.vol_confidence', 
                           vol_col='Y_log_vol_10min_lag_1m',
                           pred_col='predicted.volatility'):
    """
    Analyze the sign flip phenomenon in predictions across confidence levels.
    """
    print("\nAnalyzing Sign Patterns in Predictions")
    print("=" * 50)
    
    # Clean data
    valid_data = df.dropna(subset=[conf_col, vol_col, pred_col]).copy()
    
    # Create confidence deciles for finer granularity
    valid_data['conf_decile'] = pd.qcut(valid_data[conf_col], 10, 
                                      labels=[f'D{i+1}' for i in range(10)])
    
    # Add useful derived columns
    valid_data['pred_sign'] = np.sign(valid_data[pred_col])
    valid_data['pred_magnitude'] = np.abs(valid_data[pred_col])
    valid_data['error'] = valid_data[pred_col] - valid_data[vol_col]
    valid_data['rel_error'] = valid_data['error'] / valid_data[vol_col].clip(lower=1e-10)
    valid_data['hour'] = pd.to_datetime(valid_data['minute'], format='%H:%M').dt.hour
    
    # 1. Basic sign statistics by confidence decile
    sign_stats = valid_data.groupby('conf_decile').agg({
        'pred_sign': ['mean', 'std', 'size'],  # size gives us count
        'pred_magnitude': ['mean', 'std'],
        pred_col: ['mean', 'std'],
        vol_col: ['mean', 'std'],
        'error': ['mean', 'std']
    }).round(6)
    
    print("\n1. Sign Statistics by Confidence Decile:")
    print(sign_stats)
    
    # 2. Transition analysis
    valid_data = valid_data.sort_values(['symbol', 'date', 'minute'])
    valid_data['sign_change'] = (valid_data.groupby('symbol')['pred_sign']
                                .transform(lambda x: (x != x.shift()).astype(float)))
    
    # Analyze sign changes by confidence
    sign_changes = valid_data.groupby('conf_decile').agg({
        'sign_change': ['mean', 'sum', 'size']
    }).round(4)
    
    print("\n2. Sign Changes by Confidence Decile:")
    print(sign_changes)
    
    # 3. Time of day analysis
    time_pattern = valid_data.groupby(['hour', 'conf_decile']).agg({
        'pred_sign': ['mean', 'std'],
        'pred_magnitude': 'mean',
        'error': 'mean',
        pred_col: 'size'  # Count of predictions
    }).round(6)
    
    print("\n3. Time of Day Patterns:")
    print(time_pattern)
    
    # 4. Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Average prediction by confidence decile
    plt.subplot(1, 3, 1)
    decile_means = valid_data.groupby('conf_decile')[pred_col].mean()
    plt.plot(range(10), decile_means.values, 'bo-')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.title('Mean Prediction by Confidence Decile')
    plt.xticks(range(10), decile_means.index, rotation=45)
    plt.ylabel('Mean Prediction')
    
    # Plot 2: Sign changes by confidence
    plt.subplot(1, 3, 2)
    sign_flip_rate = sign_changes['sign_change']['mean']
    plt.plot(range(10), sign_flip_rate.values, 'ro-')
    plt.title('Sign Change Rate by Confidence')
    plt.xticks(range(10), sign_flip_rate.index, rotation=45)
    plt.ylabel('Sign Change Rate')
    
    # Plot 3: Error magnitude vs confidence
    plt.subplot(1, 3, 3)
    error_magnitude = valid_data.groupby('conf_decile')['error'].apply(lambda x: np.abs(x).mean())
    plt.plot(range(10), error_magnitude.values, 'go-')
    plt.title('Mean Absolute Error by Confidence')
    plt.xticks(range(10), error_magnitude.index, rotation=45)
    plt.ylabel('Mean Absolute Error')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Volatility level analysis
    vol_quintiles = pd.qcut(valid_data[vol_col], 5, labels=['VL', 'L', 'M', 'H', 'VH'])
    valid_data['vol_quintile'] = vol_quintiles
    
    controlled_analysis = valid_data.groupby(['vol_quintile', 'conf_decile']).agg({
        pred_col: ['mean', 'std'],
        'pred_sign': 'mean',
        'error': ['mean', 'std'],
        pred_col: 'size'  # Using pred_col for count since we know it exists
    }).round(6)
    
    print("\n4. Analysis by Volatility Level and Confidence:")
    print(controlled_analysis)
    
    # 6. Analyze sequential behavior
    sequence_analysis = valid_data.groupby('symbol').agg({
        'sign_change': ['mean', 'sum'],
        conf_col: ['mean', 'std'],
        pred_col: ['mean', 'std'],
        'error': ['mean', 'std']
    }).round(6)
    
    print("\n5. Sequential Behavior by Symbol:")
    print(sequence_analysis.describe())
    
    return {
        'sign_stats': sign_stats,
        'sign_changes': sign_changes,
        'time_pattern': time_pattern,
        'vol_confidence_analysis': controlled_analysis,
        'sequence_analysis': sequence_analysis
    }

# %%
analyze_prediction_signs(df)

# %%
df.columns

# %%
def analyze_joint_predictions(df, 
                            vol_pred='predicted.volatility',
                            ret_pred='predicted.returns',
                            vol_conf='predicted.vol_confidence',
                            ret_conf='predicted.ret_confidence',
                            vol_target='Y_log_vol_10min_lag_1m',
                            ret_target='Y_log_ret_60min_lag_1m'):
    """
    Analyze joint predictions with focus on capital allocation implications.
    """
    print("\nAnalyzing Joint Prediction Patterns")
    print("=" * 50)
    
    # Clean data
    valid_data = df.dropna(subset=[vol_pred, ret_pred, vol_conf, ret_conf, 
                                 vol_target, ret_target]).copy()
    
    # Add derived columns
    valid_data['vol_error'] = valid_data[vol_pred] - valid_data[vol_target]
    valid_data['ret_error'] = valid_data[ret_pred] - valid_data[ret_target]
    valid_data['vol_error_magnitude'] = np.abs(valid_data['vol_error'])
    valid_data['ret_error_magnitude'] = np.abs(valid_data['ret_error'])
    
    # Create quintiles for each metric
    for col in [ret_pred, ret_conf, vol_pred, vol_conf]:
        valid_data[f'{col}_quintile'] = pd.qcut(valid_data[col], 5, 
                                               labels=['VL', 'L', 'M', 'H', 'VH'])
    
    # 1. Joint Confidence Analysis
    joint_conf = valid_data.groupby([f'{ret_conf}_quintile', 
                                   f'{vol_conf}_quintile']).agg({
        'ret_error_magnitude': ['mean', 'std', 'count'],
        'vol_error_magnitude': ['mean', 'std'],
        ret_target: ['mean', 'std'],
        vol_target: ['mean', 'std']
    }).round(6)
    
    print("\n1. Joint Confidence Analysis:")
    print(joint_conf)
    
    # 2. High Confidence Analysis
    # Focus on cases where both models are highly confident
    high_conf = valid_data[
        (valid_data[f'{ret_conf}_quintile'] == 'VH') & 
        (valid_data[f'{vol_conf}_quintile'] == 'VH')
    ]
    
    print("\n2. High Joint Confidence Statistics:")
    print(high_conf[[ret_pred, vol_pred, ret_target, vol_target]].describe())
    
    # 3. Asymmetric Cost Analysis
    def compute_asymmetric_cost(group, ret_weight=2.0):
        """
        Compute asymmetric cost metric:
        - Over-prediction of returns weighted more heavily (actual capital loss)
        - Under-prediction of volatility weighted more heavily (missing opportunity)
        """
        ret_over = (group['ret_error'] > 0).mean()
        vol_under = (group['vol_error'] < 0).mean()
        ret_cost = ret_over * ret_weight
        vol_cost = vol_under
        return pd.Series({
            'ret_over_rate': ret_over,
            'vol_under_rate': vol_under,
            'combined_cost': ret_cost + vol_cost,
            'count': len(group)
        })
    
    cost_analysis = valid_data.groupby([f'{ret_conf}_quintile', 
                                      f'{vol_conf}_quintile']).apply(compute_asymmetric_cost)
    
    print("\n3. Asymmetric Cost Analysis:")
    print(cost_analysis)
    
    # 4. Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Return Prediction Error vs Joint Confidence
    plt.subplot(1, 3, 1)
    conf_levels = ['VL', 'L', 'M', 'H', 'VH']
    errors = [[valid_data[
        (valid_data[f'{ret_conf}_quintile'] == ret_level) & 
        (valid_data[f'{vol_conf}_quintile'] == vol_level)
    ]['ret_error_magnitude'].mean() 
        for ret_level in conf_levels]
        for vol_level in conf_levels]
    
    plt.imshow(errors, cmap='YlOrRd')
    plt.colorbar(label='Mean Return Error')
    plt.title('Joint Confidence vs Return Error')
    plt.xlabel('Return Confidence')
    plt.ylabel('Vol Confidence')
    plt.xticks(range(5), conf_levels)
    plt.yticks(range(5), conf_levels)
    
    # Plot 2: Alpha Scaling Effectiveness
    plt.subplot(1, 3, 2)
    def compute_pnl_ratio(group):
        """Ratio of successful to unsuccessful scaled trades"""
        correct_dir = np.sign(group[ret_pred]) == np.sign(group[ret_target])
        good_scale = np.sign(group[vol_pred]) == np.sign(group[vol_target])
        return (correct_dir & good_scale).sum() / len(group)
    
    pnl_ratios = [[valid_data[
        (valid_data[f'{ret_conf}_quintile'] == ret_level) & 
        (valid_data[f'{vol_conf}_quintile'] == vol_level)
    ].pipe(compute_pnl_ratio)
        for ret_level in conf_levels]
        for vol_level in conf_levels]
    
    plt.imshow(pnl_ratios, cmap='RdYlGn')
    plt.colorbar(label='Success Ratio')
    plt.title('Joint Prediction Success Rate')
    plt.xlabel('Return Confidence')
    plt.ylabel('Vol Confidence')
    plt.xticks(range(5), conf_levels)
    plt.yticks(range(5), conf_levels)
    
    # Plot 3: Asymmetric Cost
    plt.subplot(1, 3, 3)
    costs = cost_analysis['combined_cost'].unstack()
    plt.imshow(costs, cmap='YlOrRd')
    plt.colorbar(label='Combined Cost')
    plt.title('Asymmetric Cost by Confidence')
    plt.xlabel('Return Confidence')
    plt.ylabel('Vol Confidence')
    plt.xticks(range(5), conf_levels)
    plt.yticks(range(5), conf_levels)
    
    plt.tight_layout()
    plt.show()
    
    # 5. Time of Day Pattern
    valid_data['hour'] = pd.to_datetime(valid_data['minute'], format='%H:%M').dt.hour
    time_pattern = valid_data.groupby('hour').agg({
        'ret_error_magnitude': ['mean', 'std'],
        'vol_error_magnitude': ['mean', 'std'],
        ret_conf: 'mean',
        vol_conf: 'mean'
    }).round(6)
    
    print("\n4. Time of Day Pattern:")
    print(time_pattern)
    
    return {
        'joint_conf': joint_conf,
        'high_conf': high_conf.describe(),
        'cost_analysis': cost_analysis,
        'time_pattern': time_pattern
    }

# %%
analyze_joint_predictions(df)

# %%




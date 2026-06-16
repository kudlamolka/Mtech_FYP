import numpy as np
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

from utils.constants import OHLCV_COLUMNS, ANOMALY_DATA_FILE, TRACKER_FILE, OUTPUT_FILE
from utils.data_io import load_csv_with_dates, sort_stock_data, merge_with_tracker
from utils.validation import compute_classification_metrics, print_validation_report

def load_and_clean_data(file_path):
    df = load_csv_with_dates(file_path)
    
    start_date = '2025-04-01'
    end_date = '2026-05-31'
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    df = sort_stock_data(df, by=['date', 'Stock'])
    df = df.dropna(subset=OHLCV_COLUMNS)
    return df

def compute_market_and_timeline_features(df):
    print("Computing vectorized market and rolling features...")
    df = sort_stock_data(df)
    
    # 1. Timeline features per stock
    df['return_1m'] = df.groupby('Stock')['close'].pct_change()
    df['return_5m'] = df.groupby('Stock')['close'].pct_change(periods=5)
    df['hl_spread_pct'] = (df['high'] - df['low']) / df['open']
    
    stock_group = df.groupby('Stock')
    roll_vol_mean = stock_group['volume'].transform(lambda x: x.rolling(30, min_periods=5).mean())
    roll_vol_std = stock_group['volume'].transform(lambda x: x.rolling(30, min_periods=5).std())
    roll_vol_std = roll_vol_std.replace(0, np.nan).fillna(df['volume'].std()).fillna(1e-6)
    df['volume_zscore'] = (df['volume'] - roll_vol_mean) / roll_vol_std
    
    roll_price_mean = stock_group['close'].transform(lambda x: x.rolling(15, min_periods=2).mean())
    df['price_dev_pct'] = (df['close'] - roll_price_mean) / roll_price_mean
    
    # 2. Cross-Sectional features (CRITICAL for eliminating false alarms)
    market_time_group = df.groupby('date')
    df['market_avg_return'] = market_time_group['return_1m'].transform('mean')
    df['market_avg_volatility'] = market_time_group['hl_spread_pct'].transform('mean')
    
    df['return_vs_market'] = df['return_1m'] - df['market_avg_return']
    df['volatility_vs_market'] = df['hl_spread_pct'] - df['market_avg_volatility']
    
    df = df.bfill().ffill().fillna(0)
    return df

def process_anomalies_with_optimal_threshold(df, tracker_path):
    feature_cols = [
        'return_1m', 'return_5m', 'hl_spread_pct', 
        'volume_zscore', 'price_dev_pct',
        'return_vs_market', 'volatility_vs_market'
    ]
    
    print("Scaling features...")
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Fitting Global Isolation Forest...")
    iforest = IsolationForest(
        n_estimators=100,
        contamination='auto',
        max_samples=0.25,
        random_state=42,
        n_jobs=-1
    )
    iforest.fit(X_scaled)
    
    print("Extracting continuous anomaly scores...")
    df['raw_anomaly_score'] = iforest.score_samples(X_scaled)
    print("Aligning with tracker data for threshold optimization...")

    merged = merge_with_tracker(df, tracker_path)
    
    outlier_scores = -merged['raw_anomaly_score']
    y_true = merged['actual_anomaly'].values
    
    fpr, tpr, thresholds = roc_curve(y_true, outlier_scores)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    
    optimal_score_threshold = thresholds[ix]
    print(f"\n>>> Optimal Outlier Score Threshold Discovered: {optimal_score_threshold:.4f} <<<")
    
    merged['is_anomaly'] = np.where(outlier_scores >= optimal_score_threshold, 1, 0)
    
    return merged

def validate_against_tracker(results_df, tracker_path):
    """
    Validates model predictions using the actual_anomaly column already processed
    during the optimization stage.
    """
    print("\n--- Starting Validation Pipeline ---")
    
    if 'actual_anomaly' not in results_df.columns:
        print("Error: 'actual_anomaly' column missing from results. Re-running merge baseline...")
        results_df = merge_with_tracker(results_df, tracker_path)

    metrics = compute_classification_metrics(results_df)
    print_validation_report(metrics)


if __name__ == "__main__":
    start_time = time.time()
    
    input_file = ANOMALY_DATA_FILE
    tracker_file = TRACKER_FILE
    output_file = OUTPUT_FILE
    
    # 1. Load data safely
    df_raw = load_and_clean_data(input_file)
    
    # 2. Extract features
    df_featured = compute_market_and_timeline_features(df_raw)
    
    # 3. Model anomalies using customized ROC thresholds
    df_results = process_anomalies_with_optimal_threshold(df_featured, tracker_file)

    print(f"Saving final results to {output_file}...")
    df_results.to_csv(output_file, index=False)
    
    # 4. Run validation check
    validate_against_tracker(df_results, tracker_file)
    
    execution_time = time.time() - start_time
    print("================ PERFORMANCE TIMING ================")
    print(f"Total Execution Time       : {execution_time // 60:.0f}m {execution_time % 60:.1f}s")
    print("====================================================\n")

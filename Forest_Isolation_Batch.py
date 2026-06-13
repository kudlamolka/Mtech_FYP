import pandas as pd
import numpy as np
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(file_path):
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    start_date = '2025-04-01'
    end_date = '2026-05-31'
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    df = df.sort_values(by=['date', 'Stock']).reset_index(drop=True)
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    return df

def compute_market_and_timeline_features(df):
    print("Computing vectorized market and rolling features...")
    df = df.sort_values(by=['Stock', 'date']).reset_index(drop=True)
    
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

def process_global_anomalies_tuned(df, target_contamination=0.0005):
    """
    Fits a global Isolation Forest forcing an explicit, tight target threshold
    to instantly crush false positives.
    """
    feature_cols = [
        'return_1m', 'return_5m', 'hl_spread_pct', 
        'volume_zscore', 'price_dev_pct',
        'return_vs_market', 'volatility_vs_market'
    ]
    
    print(f"Scaling data and fitting Global iForest with strict contamination: {target_contamination}...")
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Using explicit parameter target mapping
    iforest = IsolationForest(
        n_estimators=150,                 # Increased trees for highly accurate split definition
        contamination=target_contamination, # Forces the model to only look at the extreme 0.05% tail
        max_samples=0.4,                  # Increased sample size to capture finer structural anomalies
        random_state=42,
        n_jobs=-1
    )
    
    preds = iforest.fit_predict(X_scaled)
    df['is_anomaly'] = np.where(preds == -1, 1, 0)
    
    print(f"Diagnostics -> Explicitly Flagged Count: {df['is_anomaly'].sum()} rows.")
    return df

def validate_against_tracker(results_df, tracker_path):
    print("\n--- Starting Validation Pipeline ---")
    if not pd.io.common.file_exists(tracker_path):
        print(f"Error: Tracker file not found at {tracker_path}. Skipping validation.")
        return

    tracker_df = pd.read_csv(tracker_path)
    tracker_df['date'] = pd.to_datetime(tracker_df['date'], errors='coerce')
    tracker_df = tracker_df.dropna(subset=['date'])
    tracker_df['actual_anomaly'] = 1

    results_df['date'] = pd.to_datetime(results_df['date'])
    
    merged = pd.merge(
        results_df, 
        tracker_df[['Stock', 'date', 'actual_anomaly']], 
        on=['Stock', 'date'], 
        how='left'
    )
    merged['actual_anomaly'] = merged['actual_anomaly'].fillna(0).astype(int)

    total_injected = tracker_df.shape[0]
    total_predicted = merged['is_anomaly'].sum()

    tp = merged[(merged['is_anomaly'] == 1) & (merged['actual_anomaly'] == 1)].shape[0]
    fp = merged[(merged['is_anomaly'] == 1) & (merged['actual_anomaly'] == 0)].shape[0]
    fn = merged[(merged['is_anomaly'] == 0) & (merged['actual_anomaly'] == 1)].shape[0]

    precision = tp / total_predicted if total_predicted > 0 else 0
    recall = tp / total_injected if total_injected > 0 else 0
    
    print("\n================ VALIDATION REPORT ================")
    print(f"Total Anomalies Injected (Ground Truth): {total_injected}")
    print(f"Total Anomalies Flagged by Model      : {total_predicted}")
    print("---------------------------------------------------")
    print(f"True Positives (Successfully Caught)  : {tp}")
    print(f"False Positives (False Alarms)        : {fp}")
    print(f"False Negatives (Missed Anomalies)    : {fn}")
    print("---------------------------------------------------")
    print(f"Precision (When it flags, how right is it?): {precision:.2%}")
    print(f"Recall (What % of anomalies did it catch?): {recall:.2%}")
    print("===================================================\n")


if __name__ == "__main__":
    start_time = time.time()
    
    input_file = "data/all_stocks_with_anomalies.csv"
    tracker_file = "data/anomaly_tracker.csv"
    output_file = "minute_stock_anomalies_detected.csv"
    
    # 1. Load data safely
    df_raw = load_and_clean_data(input_file)
    
    # 2. Extract multi-timeframe & market cross-sectional features
    df_featured = compute_market_and_timeline_features(df_raw)
    
    # 3. FORCE strict hyper-targeted contamination limit (0.05%)
    # This will explicitly clamp down on the 52,334 flag threshold wall.
    df_results = process_global_anomalies_tuned(df_featured, target_contamination=0.0005)

    print(f"Saving final results to {output_file}...")
    df_results.to_csv(output_file, index=False)
    
    validate_against_tracker(df_results, tracker_file)
    
    execution_time = time.time() - start_time
    print("================ PERFORMANCE TIMING ================")
    print(f"Total Execution Time       : {execution_time // 60:.0f}m {execution_time % 60:.1f}s")
    print("====================================================\n")
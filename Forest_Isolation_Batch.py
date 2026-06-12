import pandas as pd
import numpy as np
import time  # Imported for performance tracking
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(file_path):
    """
    Loads the minute-by-minute data, filters the required 14-month timeframe,
    and ensures correct formatting.
    """
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Use format='mixed' or errors='coerce' to ensure strict timestamp parsing consistency
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Filter for the exact period requested (April 1st, 2025 to May 31st, 2026)
    start_date = '2025-04-01'
    end_date = '2026-05-31'
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    # Sort chronologically to make rolling timeline calculations accurate
    df = df.sort_values(by=['Stock', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    return df

def compute_timeline_features(group):
    """
    Computes advanced rolling behavioral features for an individual stock 
    to isolate anomalies within its own historical context.
    """
    # 1. Base Returns (1-minute and a broader 5-minute momentum horizon)
    group['return_1m'] = group['close'].pct_change() 
    group['return_5m'] = group['close'].pct_change(periods=5)
    
    # 2. Minute Volatility (High/Low spread relative to Open)
    group['hl_spread_pct'] = (group['high'] - group['low']) / group['open'] 
    
    # 3. Enhanced Volume Z-score (using 60 minutes for a more robust baseline)
    rolling_vol_mean = group['volume'].rolling(window=60, min_periods=5).mean()
    rolling_vol_std = group['volume'].rolling(window=60, min_periods=5).std()
    # Avoid division by zero by filling empty or zero std dev with a minor constant
    rolling_vol_std = rolling_vol_std.replace(0, np.nan).fillna(group['volume'].std()).fillna(1e-6)
    group['volume_zscore'] = (group['volume'] - rolling_vol_mean) / rolling_vol_std
    
    # 4. Deviation from multiple moving averages (captures short-term vs mid-term shock)
    rolling_price_mean_15 = group['close'].rolling(window=15, min_periods=2).mean()
    rolling_price_mean_60 = group['close'].rolling(window=60, min_periods=5).mean()
    
    group['price_dev_pct_15'] = (group['close'] - rolling_price_mean_15) / rolling_price_mean_15
    group['price_dev_pct_60'] = (group['close'] - rolling_price_mean_60) / rolling_price_mean_60

    # Clean up edge cases resulting from initial rolling gaps
    group = group.bfill().ffill().fillna(0)
    return group

def process_stock_anomalies(df, contamination_rate=0.005):
    """
    Batches isolation forest execution per individual stock timeline using the explicit contamination rate.
    """
    feature_cols = ['return_1m', 'return_5m', 'hl_spread_pct', 'volume_zscore', 'price_dev_pct_15', 'price_dev_pct_60']
    processed_groups = []
    
    grouped = df.groupby('Stock')
    total_stocks = len(grouped)
    
    for idx, (Stock, group) in enumerate(grouped, 1):
        print(f"[{idx}/{total_stocks}] Processing timeline features & iForest for {Stock}...")

        group_featured = compute_timeline_features(group)
        
        # Skip fitting if the historical history is too small to build an isolated sample tree
        if len(group_featured) < 10:
            group_featured['anomaly_score'] = 0.0
            group_featured['is_anomaly'] = 0
            processed_groups.append(group_featured)
            continue
            
        X = group_featured[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # FIX: Explicitly mapped contamination_rate parameter into the instantiation step!
        iforest = IsolationForest(
            n_estimators=150,               # Bumped up for smoother split definitions
            contamination=contamination_rate, # Dynamic user parameter instead of 'auto'
            max_samples='auto',
            random_state=42,
            n_jobs=-1
        )
        
        iforest.fit(X_scaled)
        
        group_featured['anomaly_score'] = iforest.decision_function(X_scaled)
        preds = iforest.predict(X_scaled)
        group_featured['is_anomaly'] = np.where(preds == -1, 1, 0)
        
        processed_groups.append(group_featured)

    return pd.concat(processed_groups, ignore_index=True)


def validate_against_tracker(results_df, tracker_path):
    """
    Validates model predictions against the injected anomaly tracker.
    """
    print("\n--- Starting Validation Pipeline ---")
    if not pd.io.common.file_exists(tracker_path):
        print(f"Error: Tracker file not found at {tracker_path}. Skipping validation.")
        return

    tracker_df = pd.read_csv(tracker_path)
    tracker_df['date'] = pd.to_datetime(tracker_df['date'], errors='coerce')
    tracker_df = tracker_df.dropna(subset=['date'])
 
    tracker_df['actual_anomaly'] = 1

    # Ensure precision matching on timestamps across both DataFrames
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
    
    df_raw = load_and_clean_data(input_file)
    
    # Passing targeted contamination rate safely down the processing pipeline
    df_results = process_stock_anomalies(df_raw, contamination_rate=0.005)

    print(f"Saving final results to {output_file}...")
    df_results.to_csv(output_file, index=False)
    print("Batch processing successfully completed!")

    validate_against_tracker(df_results, tracker_file)
    
    execution_time = time.time() - start_time
    print("================ PERFORMANCE TIMING ================")
    if execution_time > 60:
        print(f"Total Execution Time       : {execution_time // 60:.0f}m {execution_time % 60:.1f}s")
    else:
        print(f"Total Execution Time       : {execution_time:.2f} seconds")
    print("====================================================\n")
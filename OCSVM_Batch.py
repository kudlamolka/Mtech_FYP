import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

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
    
    # 2. Cross-Sectional features
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
    
    print("Fitting One-Class SVM...")
    # Use a subset for faster training (100K samples)
    subset_size = min(100000, len(X_scaled))
    indices = np.random.choice(len(X_scaled), subset_size, replace=False)
    X_subset = X_scaled[indices]
    print(f"Training on subset of {subset_size} samples (out of {len(X_scaled)} total)...")
    
    oc_svm = OneClassSVM(
        kernel='rbf',
        gamma='scale',
        nu=0.1,  # Upper bound on fraction of outliers
        shrinking=True,
        cache_size=1000
    )
    
    oc_svm.fit(X_subset)
    
    print("Extracting continuous anomaly scores on subset...")
    # Also evaluate on subset for speed
    eval_subset_size = min(500000, len(X_scaled))
    eval_indices = np.random.choice(len(X_scaled), eval_subset_size, replace=False)
    X_eval = X_scaled[eval_indices]
    df_eval = df.iloc[eval_indices].copy()
    
    # One-Class SVM: decision_function returns scores (negative = anomalies)
    df_eval['raw_anomaly_score'] = oc_svm.decision_function(X_eval)
    print("Aligning with tracker data for threshold optimization...")
    tracker_df = pd.read_csv(tracker_path)
    tracker_df['date'] = pd.to_datetime(tracker_df['date'], errors='coerce')
    tracker_df['actual_anomaly'] = 1
    
    df_eval['date'] = pd.to_datetime(df_eval['date'])
    merged = pd.merge(df_eval, tracker_df[['Stock', 'date', 'actual_anomaly']], on=['Stock', 'date'], how='left')
    merged['actual_anomaly'] = merged['actual_anomaly'].fillna(0).astype(int)
    
    # One-Class SVM: more negative = more anomalous, so we use negative of the score
    # This makes higher scores = more anomalous (consistent with Isolation Forest)
    outlier_scores = -merged['raw_anomaly_score']
    y_true = merged['actual_anomaly'].values
    
    fpr, tpr, thresholds = roc_curve(y_true, outlier_scores)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    
    optimal_score_threshold = thresholds[ix]
    print(f"\n>>> Optimal Outlier Score Threshold Discovered: {optimal_score_threshold:.4f} <<<")
    
    # Save model artifacts for streaming inference
    os.makedirs('models', exist_ok=True)
    joblib.dump(oc_svm, 'models/ocsvm_model.pkl')
    joblib.dump(scaler, 'models/ocsvm_scaler.pkl')
    joblib.dump(optimal_score_threshold, 'models/ocsvm_optimal_threshold.pkl')
    print(f"\n>>> One-Class SVM model artifacts saved to models/ directory <<<")
    
    merged['is_anomaly'] = np.where(outlier_scores >= optimal_score_threshold, 1, 0)
    
    return merged

def validate_against_tracker(results_df, tracker_path):
    """
    Validates model predictions using the actual_anomaly column already processed
    during the optimization stage.
    """
    print("\n--- Starting Validation Pipeline ---")
    
    # Check if the column exists to prevent downstream crashes
    if 'actual_anomaly' not in results_df.columns:
        print("Error: 'actual_anomaly' column missing from results. Re-running merge baseline...")
        if not pd.io.common.file_exists(tracker_path):
            print(f"Error: Tracker file not found at {tracker_path}. Skipping validation.")
            return
        tracker_df = pd.read_csv(tracker_path)
        tracker_df['date'] = pd.to_datetime(tracker_df['date'], errors='coerce')
        tracker_df['actual_anomaly'] = 1
        results_df = pd.merge(results_df, tracker_df[['Stock', 'date', 'actual_anomaly']], on=['Stock', 'date'], how='left')
        results_df['actual_anomaly'] = results_df['actual_anomaly'].fillna(0).astype(int)

    # Calculate metrics cleanly off the single optimized dataframe
    total_injected = results_df['actual_anomaly'].sum()
    total_predicted = results_df['is_anomaly'].sum()

    tp = results_df[(results_df['is_anomaly'] == 1) & (results_df['actual_anomaly'] == 1)].shape[0]
    fp = results_df[(results_df['is_anomaly'] == 1) & (results_df['actual_anomaly'] == 0)].shape[0]
    fn = results_df[(results_df['is_anomaly'] == 0) & (results_df['actual_anomaly'] == 1)].shape[0]

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
    output_file = "minute_stock_anomalies_detected_ocsvm.csv"
    
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

import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.neighbors import LocalOutlierFactor
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

def test_lof_configuration(X_scaled, y_true, n_neighbors, contamination, algorithm='auto'):
    print(f"\n--- Testing LOF: n_neighbors={n_neighbors}, contamination={contamination}, algorithm={algorithm} ---")
    start_time = time.time()
    
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        algorithm=algorithm,
        leaf_size=30,
        metric='minkowski',
        p=2,
        n_jobs=-1
    )
    
    lof.fit(X_scaled)
    anomaly_scores = -lof.negative_outlier_factor_
    
    # ROC-based threshold
    fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    optimal_threshold = thresholds[ix]
    
    # Calculate metrics
    predictions = np.where(anomaly_scores >= optimal_threshold, 1, 0)
    tp = ((predictions == 1) & (y_true == 1)).sum()
    fp = ((predictions == 1) & (y_true == 0)).sum()
    fn = ((predictions == 0) & (y_true == 1)).sum()
    
    total_predicted = predictions.sum()
    total_injected = y_true.sum()
    
    precision = tp / total_predicted if total_predicted > 0 else 0
    recall = tp / total_injected if total_injected > 0 else 0
    
    execution_time = time.time() - start_time
    
    print(f"Precision: {precision:.2%}, Recall: {recall:.2%}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Time: {execution_time:.1f}s")
    
    return {
        'n_neighbors': n_neighbors,
        'contamination': contamination,
        'algorithm': algorithm,
        'precision': precision,
        'recall': recall,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'time': execution_time,
        'threshold': optimal_threshold
    }

if __name__ == "__main__":
    input_file = "data/all_stocks_with_anomalies.csv"
    tracker_file = "data/anomaly_tracker.csv"
    
    # Load and prepare data
    df_raw = load_and_clean_data(input_file)
    df_featured = compute_market_and_timeline_features(df_raw)
    
    feature_cols = [
        'return_1m', 'return_5m', 'hl_spread_pct', 
        'volume_zscore', 'price_dev_pct',
        'return_vs_market', 'volatility_vs_market'
    ]
    
    print("Scaling features...")
    X = df_featured[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Load tracker for ground truth
    tracker_df = pd.read_csv(tracker_file)
    tracker_df['date'] = pd.to_datetime(tracker_df['date'], errors='coerce')
    tracker_df['actual_anomaly'] = 1
    
    df_featured['date'] = pd.to_datetime(df_featured['date'])
    merged = pd.merge(df_featured, tracker_df[['Stock', 'date', 'actual_anomaly']], on=['Stock', 'date'], how='left')
    merged['actual_anomaly'] = merged['actual_anomaly'].fillna(0).astype(int)
    y_true = merged['actual_anomaly'].values
    
    print(f"\nTotal anomalies in ground truth: {y_true.sum()}")
    print(f"Total samples: {len(y_true)}")
    
    # Test different configurations
    results = []
    
    # Test different n_neighbors values
    n_neighbors_list = [5, 10, 15, 20, 25, 30, 50, 100]
    contamination_list = ['auto', 0.0005, 0.001, 0.005]
    
    for n_neighbors in n_neighbors_list:
        for contamination in contamination_list:
            result = test_lof_configuration(X_scaled, y_true, n_neighbors, contamination)
            results.append(result)
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("SUMMARY OF LOF CONFIGURATIONS")
    print("="*70)
    print(results_df[['n_neighbors', 'contamination', 'precision', 'recall', 'tp', 'fp', 'fn', 'time']].to_string(index=False))
    
    # Find best configuration by different metrics
    best_precision = results_df.loc[results_df['precision'].idxmax()]
    best_recall = results_df.loc[results_df['recall'].idxmax()]
    best_f1 = results_df.loc[(2 * results_df['precision'] * results_df['recall'] / (results_df['precision'] + results_df['recall'] + 1e-6)).idxmax()]
    
    print("\n" + "="*70)
    print("BEST CONFIGURATIONS")
    print("="*70)
    print(f"\nBest Precision: {best_precision['precision']:.2%}")
    print(f"  n_neighbors={best_precision['n_neighbors']}, contamination={best_precision['contamination']}")
    print(f"  Recall: {best_precision['recall']:.2%}, TP: {best_precision['tp']}, FP: {best_precision['fp']}")
    
    print(f"\nBest Recall: {best_recall['recall']:.2%}")
    print(f"  n_neighbors={best_recall['n_neighbors']}, contamination={best_recall['contamination']}")
    print(f"  Precision: {best_recall['precision']:.2%}, TP: {best_recall['tp']}, FP: {best_recall['fp']}")
    
    print(f"\nBest F1 Balance: {best_f1['precision']:.2%} precision, {best_f1['recall']:.2%} recall")
    print(f"  n_neighbors={best_f1['n_neighbors']}, contamination={best_f1['contamination']}")
    print(f"  TP: {best_f1['tp']}, FP: {best_f1['fp']}")
    
    # Save results
    results_df.to_csv('lof_tuning_results.csv', index=False)
    print("\nResults saved to lof_tuning_results.csv")

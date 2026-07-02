import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

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

def create_labeled_dataset(df, tracker_path):
    print("Creating labeled dataset...")
    tracker_df = pd.read_csv(tracker_path)
    tracker_df['date'] = pd.to_datetime(tracker_df['date'], errors='coerce')
    tracker_df['actual_anomaly'] = 1
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.merge(tracker_df[['Stock', 'date', 'actual_anomaly']], 
                   on=['Stock', 'date'], how='left')
    df['actual_anomaly'] = df['actual_anomaly'].fillna(0).astype(int)
    
    print(f"Labeled {df['actual_anomaly'].sum()} anomalies out of {len(df)} records")
    return df

def train_supervised_model(df):
    print("="*70)
    print("TRAINING SUPERVISED MODEL")
    print("="*70)
    
    feature_cols = [
        'return_1m', 'return_5m', 'hl_spread_pct', 
        'volume_zscore', 'price_dev_pct',
        'return_vs_market', 'volatility_vs_market'
    ]
    
    # Drop rows with NaN in features
    df = df.dropna(subset=feature_cols)
    print(f"Dataset after dropping NaN: {len(df)} records")
    
    X = df[feature_cols].values
    y = df['actual_anomaly'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Using full dataset of {len(X_scaled)} samples for training")
    
    # Split data (stratified to maintain anomaly ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} records ({y_train.sum()} anomalies)")
    print(f"Test set: {len(X_test)} records ({y_test.sum()} anomalies)")
    
    # Train Random Forest
    print("\nTraining Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=50,  # Reduced from 100 for speed
        max_depth=10,  # Reduced from 15 for speed
        min_samples_split=50,  # Increased from 10 for speed
        min_samples_leaf=10,  # Increased from 5 for speed
        class_weight='balanced',
        random_state=42,
        n_jobs=1,  # Single thread to avoid resource issues
        verbose=2  # Show progress during training
    )
    
    start_time = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.1f}s")
    
    # Predictions
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    # Metrics
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    tp = ((y_pred == 1) & (y_test == 1)).sum()
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()
    tn = ((y_pred == 0) & (y_test == 0)).sum()
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"\nTotal Anomalies in Test Set: {y_test.sum()}")
    print(f"Total Predicted as Anomalies: {y_pred.sum()}")
    print("---------------------------------------------------")
    print(f"True Positives (Successfully Caught)  : {tp}")
    print(f"False Positives (False Alarms)        : {fp}")
    print(f"False Negatives (Missed Anomalies)    : {fn}")
    print(f"True Negatives (Correctly Normal)    : {tn}")
    print("---------------------------------------------------")
    print(f"Precision (When it flags, how right is it?): {precision:.2%}")
    print(f"Recall (What % of anomalies did it catch?): {recall:.2%}")
    print(f"F1 Score: {2 * precision * recall / (precision + recall + 1e-6):.2%}")
    
    # Feature importance
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE")
    print("="*70)
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance_df.to_string(index=False))
    
    # Save model artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf, 'models/supervised_rf_model.pkl')
    joblib.dump(scaler, 'models/supervised_scaler.pkl')
    print(f"\nModel artifacts saved to models/ directory")
    
    return rf, scaler, importance_df

if __name__ == "__main__":
    start_time = time.time()
    
    input_file = "data/all_stocks_with_anomalies.csv"
    tracker_file = "data/anomaly_tracker.csv"
    
    # 1. Load and clean data
    df_raw = load_and_clean_data(input_file)
    
    # 2. Compute features
    df_featured = compute_market_and_timeline_features(df_raw)
    
    # 3. Create labeled dataset
    df_labeled = create_labeled_dataset(df_featured, tracker_file)
    
    # 4. Train supervised model
    rf_model, scaler, importance = train_supervised_model(df_labeled)
    
    # 5. Save labeled dataset for future use
    labeled_file = "data/all_stocks_anomalies_data_labeled.csv"
    df_labeled.to_csv(labeled_file, index=False)
    print(f"\nLabeled dataset saved to {labeled_file}")
    
    execution_time = time.time() - start_time
    print("\n" + "="*70)
    print(f"TOTAL EXECUTION TIME: {execution_time // 60:.0f}m {execution_time % 60:.1f}s")
    print("="*70)

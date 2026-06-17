import logging
import os
import sys

import pandas as pd
import numpy as np
import random

logger = logging.getLogger(__name__)

# Define file paths
input_file = 'data/all_stocks_historic.csv'
output_data_file = 'data/all_stocks_with_anomalies.csv'
anomaly_track_file = 'data/anomaly_tracker.csv'

REQUIRED_COLS = ['Stock', 'date', 'open', 'high', 'low', 'close', 'volume']


def induce_anomalies(input_path, output_path, tracker_path):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input data file not found: {input_path}")

    print("Loading data...")
    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError(f"Input file is empty: {input_path}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    df = df.sort_values(by=['Stock', 'date']).reset_index(drop=True)
    
    total_rows = len(df)
    anomaly_indices = []
    current_idx = 0
    
    while True:
        step = random.randint(1000, 10000)
        current_idx += step
        if current_idx >= total_rows:
            break
        anomaly_indices.append(current_idx)
        
    print(f"Total rows: {total_rows}")
    print(f"Inducing {len(anomaly_indices)} anomalies...")

    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    tracker_records = []
    
    for idx in anomaly_indices:
        stock = df.loc[idx, 'Stock']
        date = df.loc[idx, 'date']
        target_col = random.choice(numeric_cols)
        original_value = df.loc[idx, target_col]

        corruption_type = random.choice(['spike', 'drop'])
        if corruption_type == 'spike':
            modified_value = original_value * random.uniform(5.0, 10.0)
        else:
            modified_value = original_value * random.uniform(0.01, 0.05)

        if np.issubdtype(df[target_col].dtype, np.integer):
            modified_value = int(round(modified_value))
        
        # Update the dataframe
        df.loc[idx, target_col] = modified_value
        
        tracker_records.append({
            'Stock': stock,
            'date': date,
            'Row_Index': idx,
            'Corrupted_Column': target_col,
            'Original_Value': original_value,
            'Anomalous_Value': modified_value
        })
        
    for path in (output_path, tracker_path):
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Saved anomalous dataset to: {output_path}")

    tracker_df = pd.DataFrame(tracker_records)
    tracker_df.to_csv(tracker_path, index=False)
    print(f"Saved anomaly tracker log to: {tracker_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        induce_anomalies(input_file, output_data_file, anomaly_track_file)
    except Exception:
        logger.exception("induce_anomalies failed")
        sys.exit(1)
import pandas as pd
import numpy as np
import random
import os

# Define file paths
input_file = 'data/streaming_data.csv'
output_data_file = 'data/stearming_data_with_anomalies.csv'
anomaly_track_file = 'data/streaming_anomaly_tracker.csv'

def induce_anomalies(input_path, output_path, tracker_path):
    print("Loading data...")
    df = pd.read_csv(input_path)
    
    # Sort by primary keys for consistency
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
        
    df.to_csv(output_path, index=False)
    print(f"Saved anomalous dataset to: {output_path}")
    
    tracker_df = pd.DataFrame(tracker_records)
    tracker_df.to_csv(tracker_path, index=False)
    print(f"Saved anomaly tracker log to: {tracker_path}")

if __name__ == "__main__":
    induce_anomalies(input_file, output_data_file, anomaly_track_file)
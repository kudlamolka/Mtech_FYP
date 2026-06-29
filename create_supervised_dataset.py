import pandas as pd

def create_supervised_dataset():
    print("="*70)
    print("CREATING SUPERVISED TRAINING DATASET")
    print("="*70)
    
    # Load data
    input_file = "data/stearming_data_with_anomalies.csv"
    tracker_file = "data/streaming_anomaly_tracker.csv"
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} records")
    
    print(f"Loading tracker from {tracker_file}...")
    tracker_df = pd.read_csv(tracker_file)
    print(f"Loaded {len(tracker_df)} anomaly records")
    
    # Convert dates for proper merging
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    tracker_df['date'] = pd.to_datetime(tracker_df['date'], errors='coerce')
    
    # Mark anomalies as 1 using merge (much faster than loop)
    tracker_df['actual_anomaly'] = 1
    df = df.merge(tracker_df[['Stock', 'date', 'actual_anomaly']], 
                  on=['Stock', 'date'], how='left')
    df['actual_anomaly'] = df['actual_anomaly'].fillna(0).astype(int)
    
    print(f"Labeled {df['actual_anomaly'].sum()} records as anomalies")
    print(f"  - Anomalies: {df['actual_anomaly'].sum()}")
    print(f"  - Normal: {(df['actual_anomaly'] == 0).sum()}")
    
    # Save labeled dataset
    output_file = "data/streaming_data_with_anomalies_data_labeled.csv"
    df.to_csv(output_file, index=False)
    print(f"\nLabeled dataset saved to {output_file}")
    
    print("\n" + "="*70)
    print("SUPERVISED DATASET CREATION COMPLETE")
    print("="*70)
    print(f"\nFile created: {output_file}")
    print("Feature engineering will be done separately during model training.")

if __name__ == "__main__":
    create_supervised_dataset()

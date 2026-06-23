#!/usr/bin/env python3
"""
Validate streaming anomaly detection results against ground truth tracker.
"""

import pandas as pd
import os
import glob
from datetime import datetime

def validate_streaming_results(log_file, tracker_file):
    """
    Validate streaming anomaly detection results against ground truth.
    
    Args:
        log_file: Path to streaming anomalies log CSV
        tracker_file: Path to anomaly tracker CSV
    """
    print(f"Loading streaming results from {log_file}...")
    df_results = pd.read_csv(log_file)
    
    print(f"Loading tracker from {tracker_file}...")
    df_tracker = pd.read_csv(tracker_file)
    
    # Add actual_anomaly column (all tracker rows are anomalies)
    df_tracker['actual_anomaly'] = 1
    
    # Convert timestamps
    df_results['timestamp'] = pd.to_datetime(df_results['timestamp'], errors='coerce')
    df_tracker['date'] = pd.to_datetime(df_tracker['date'], errors='coerce')
    
    # Merge results with tracker
    print("Merging results with tracker...")
    merged = pd.merge(
        df_results,
        df_tracker[['Stock', 'date', 'actual_anomaly']],
        left_on=['stock', 'timestamp'],
        right_on=['Stock', 'date'],
        how='left'
    )
    
    # Fill missing actual_anomaly with 0 (no ground truth match)
    merged['actual_anomaly'] = merged['actual_anomaly'].fillna(0).astype(int)
    
    # Calculate metrics
    total_detected = len(merged)
    total_injected = df_tracker['actual_anomaly'].sum()
    
    tp = merged[(merged['actual_anomaly'] == 1)].shape[0]
    fp = merged[(merged['actual_anomaly'] == 0)].shape[0]
    fn = total_injected - tp
    
    precision = tp / total_detected if total_detected > 0 else 0
    recall = tp / total_injected if total_injected > 0 else 0
    
    # Print validation report
    print("\n================ STREAMING VALIDATION REPORT ================")
    print(f"Total Anomalies Injected (Ground Truth): {total_injected}")
    print(f"Total Anomalies Detected by Streaming  : {total_detected}")
    print("---------------------------------------------------")
    print(f"True Positives (Successfully Caught)  : {tp}")
    print(f"False Positives (False Alarms)        : {fp}")
    print(f"False Negatives (Missed Anomalies)    : {fn}")
    print("---------------------------------------------------")
    print(f"Precision (When it flags, how right is it?): {precision:.2%}")
    print(f"Recall (What % of anomalies did it catch?): {recall:.2%}")
    print("===================================================\n")
    
    # Save detailed results
    output_file = log_file.replace('.csv', '_validated.csv')
    merged.to_csv(output_file, index=False)
    print(f"Detailed results saved to {output_file}")
    
    return {
        'total_detected': total_detected,
        'total_injected': total_injected,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate streaming anomaly detection results')
    parser.add_argument('--log', type=str, required=True,
                        help='Path to streaming anomalies log CSV')
    parser.add_argument('--tracker', type=str, default='data/streaming_anomaly_tracker.csv',
                        help='Path to anomaly tracker CSV')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log):
        print(f"Error: Log file not found at {args.log}")
        print("\nAvailable log files in logs/:")
        logs_dir = 'logs'
        if os.path.exists(logs_dir):
            log_files = glob.glob(os.path.join(logs_dir, 'streaming_anomalies_*.csv'))
            for f in log_files:
                print(f"  {f}")
        exit(1)
    
    if not os.path.exists(args.tracker):
        print(f"Error: Tracker file not found at {args.tracker}")
        exit(1)
    
    validate_streaming_results(args.log, args.tracker)

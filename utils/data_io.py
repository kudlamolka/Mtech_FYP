"""Shared data loading and transformation utilities."""

import pandas as pd

from utils.constants import DATE_COL, STOCK_COL


def load_csv_with_dates(file_path, date_col=DATE_COL):
    """Load a CSV file and parse the date column with coerced errors."""
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    return df


def sort_stock_data(df, by=None):
    """Sort a DataFrame by stock and date columns, resetting the index."""
    if by is None:
        by = [STOCK_COL, DATE_COL]
    return df.sort_values(by=by).reset_index(drop=True)


def load_tracker(tracker_path):
    """Load the anomaly tracker CSV with parsed dates and an actual_anomaly flag."""
    tracker_df = pd.read_csv(tracker_path)
    tracker_df[DATE_COL] = pd.to_datetime(tracker_df[DATE_COL], errors='coerce')
    tracker_df['actual_anomaly'] = 1
    return tracker_df


def merge_with_tracker(df, tracker_path):
    """Merge a results DataFrame with the anomaly tracker for validation.

    Returns the merged DataFrame with an 'actual_anomaly' column (0 or 1).
    """
    tracker_df = load_tracker(tracker_path)
    merged = pd.merge(
        df,
        tracker_df[[STOCK_COL, DATE_COL, 'actual_anomaly']],
        on=[STOCK_COL, DATE_COL],
        how='left',
    )
    merged['actual_anomaly'] = merged['actual_anomaly'].fillna(0).astype(int)
    return merged

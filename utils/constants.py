"""Shared constants for data paths and column names."""

import os

# Column names
OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
DATE_COL = 'date'
STOCK_COL = 'Stock'

# Data directory and file paths
DATA_DIR = 'data'
INPUT_DATA_FILE = os.path.join(DATA_DIR, 'all_stocks_historic.csv')
ANOMALY_DATA_FILE = os.path.join(DATA_DIR, 'all_stocks_with_anomalies.csv')
TRACKER_FILE = os.path.join(DATA_DIR, 'anomaly_tracker.csv')
OUTPUT_FILE = 'minute_stock_anomalies_detected.csv'

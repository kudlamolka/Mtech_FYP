# Stock Anomaly Detection

Real-time anomaly detection for NSE stock data using Isolation Forest algorithm.

## Project Overview

This project detects anomalies in stock market data using:
- **Isolation Forest** for unsupervised anomaly detection
- **Batch processing** for historical data analysis
- **Streaming simulation** using producer-consumer socket architecture
- **Real-time feature computation** with rolling buffers

## Features

- Timeline features: return_1m, return_5m, hl_spread_pct, volume_zscore, price_dev_pct
- Cross-sectional features: return_vs_market, volatility_vs_market
- ROC-based threshold optimization
- Streaming data simulation via socket communication

## File Structure

```
Mtech_FYP/
├── data/                          # Input data files
│   ├── all_stocks_historic.csv    # Combined historical data
│   ├── all_stocks_with_anomalies.csv  # Data with synthetic anomalies
│   └── anomaly_tracker.csv        # Ground truth labels
├── models/                        # Trained model artifacts
│   ├── isolation_forest.pkl       # Isolation Forest model
│   ├── scaler.pkl                 # StandardScaler
│   └── optimal_threshold.pkl      # Optimal anomaly threshold
├── results/                       # Per-stock raw CSVs from Kite
├── logs/                          # Log files
├── Forest_Isolation_Batch.py      # Batch training script
├── data_producer.py               # Streaming data producer
├── data_consumer.py               # Streaming data consumer
├── fetch_itc_minute_data.py       # Historical data fetcher
├── combine_csvs.py                # CSV merger
├── induce_anomaly.py              # Anomaly injection
├── requirements.txt               # Python dependencies
├── Notes.txt                      # Experiment results and changes
└── README.md                      # This file
```

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Prepare Data

1. Fetch historical data using `fetch_itc_minute_data.py`
2. Combine CSVs using `combine_csvs.py`
3. Inject synthetic anomalies using `induce_anomaly.py`

## Usage

### 1. Batch Anomaly Detection

Train the model and detect anomalies on historical data:

```bash
python Forest_Isolation_Batch.py
```

This will:
- Load data from `data/all_stocks_with_anomalies.csv`
- Compute market and timeline features
- Train Isolation Forest with contamination='auto'
- Find optimal ROC threshold using ground truth
- Save model artifacts to `models/`:
  - `isolation_forest.pkl` - Trained model
  - `scaler.pkl` - Feature scaler
  - `optimal_threshold.pkl` - Optimal anomaly threshold
- Save results to `minute_stock_anomalies_detected.csv`
- Print validation metrics

### 2. Streaming Simulation (Producer-Consumer)

Simulate real-time streaming using socket communication.

#### Step 1: Start the Producer (Terminal 1)

```bash
python data_producer.py --input data/all_stocks_with_anomalies.csv --delay 0.01
```

**Parameters:**
- `--input`: Path to CSV file (default: data/all_stocks_with_anomalies.csv)
- `--host`: Socket host (default: localhost)
- `--port`: Socket port (default: 9999)
- `--delay`: Delay between rows in seconds (default: 0.01)

The producer will:
- Load the CSV file
- Wait for consumer to connect
- Send data row-by-row as JSON over socket

#### Step 2: Start the Consumer (Terminal 2)

```bash
python data_consumer.py
```

**Parameters:**
- `--host`: Socket host (default: localhost)
- `--port`: Socket port (default: 9999)

The consumer will:
- Load pre-trained model artifacts from `models/`
- Connect to producer
- Receive data and compute features in real-time
- Detect anomalies using Isolation Forest
- Log anomalies to `logs/streaming_anomalies_*.csv`
- Print alerts to console

## Model Performance

From `Notes.txt`:

| Run | Precision | Recall | Key Change |
|-----|-----------|--------|------------|
| Baseline | 0.08% | 2.17% | Initial model |
| + Features | 0.66% | 18.24% | Added return_5m, price_dev_pct |
| Contamination 0.0005 | 21.58% | 59.65% | Reduced contamination |
| ROC Threshold | 0.18% | 78.32% | ROC-based optimal threshold |

**Note:** ROC threshold maximizes recall but results in high false positives. Contamination=0.0005 provides better balance.

## Features Used

### Timeline Features (per stock)
- `return_1m`: 1-minute price return
- `return_5m`: 5-minute price return
- `hl_spread_pct`: High-low spread percentage
- `volume_zscore`: Volume z-score (30-period rolling)
- `price_dev_pct`: Price deviation from 15-period mean

### Cross-Sectional Features (market-level)
- `return_vs_market`: Stock return vs market average
- `volatility_vs_market`: Stock volatility vs market average

## Streaming Architecture

```
Producer (data_producer.py)
  ↓
Socket (localhost:9999)
  ↓
Consumer (data_consumer.py)
  ↓
StreamingFeatureEngine (rolling buffers)
  ↓
Feature computation
  ↓
Scaler
  ↓
Isolation Forest
  ↓
Anomaly score
  ↓
Threshold check
  ↓
Alert/Log
```

## Key Design Decisions

- **Rolling buffers**: Use `collections.deque` for O(1) append/pop
- **Market averages**: Computed from recent snapshots (100-point buffer)
- **Socket communication**: JSON over TCP with newline delimiter
- **Feature consistency**: Same features in batch and streaming modes
- **Model reuse**: Pre-trained model loaded for streaming inference

## Troubleshooting

### Model artifacts not found
Run `Forest_Isolation_Batch.py` first to generate model artifacts in `models/` directory.

### Socket connection refused
Ensure producer is running before starting consumer. Check that ports match.

### High false positives
Consider using contamination=0.0005 instead of ROC threshold for better precision.

### Slow streaming
Reduce `--delay` parameter in producer (set to 0 for fastest processing).

## Notes

See `Notes.txt` for detailed experiment results, model iterations, and performance metrics.

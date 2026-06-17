#!/usr/bin/env python3
"""
Data Consumer - Receives streaming data via socket and detects anomalies.
"""

import socket
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
from collections import deque, defaultdict

class StreamingFeatureEngine:
    """Real-time feature computation for streaming data."""
    
    def __init__(self):
        self.stock_buffers = defaultdict(lambda: {
            'close': deque(maxlen=60),
            'volume': deque(maxlen=30),
            'timestamps': deque(maxlen=60)
        })
        self.market_buffer = deque(maxlen=100)
        self.volume_stats = defaultdict(lambda: {'mean': 0, 'std': 1e-6})
    
    def update_stock(self, stock, timestamp, open_price, high, low, close, volume):
        buffer = self.stock_buffers[stock]
        buffer['close'].append(close)
        buffer['volume'].append(volume)
        buffer['timestamps'].append(timestamp)
        
        if len(buffer['volume']) >= 5:
            vol_array = np.array(buffer['volume'])
            self.volume_stats[stock]['mean'] = np.mean(vol_array)
            self.volume_stats[stock]['std'] = np.std(vol_array)
            if self.volume_stats[stock]['std'] == 0:
                self.volume_stats[stock]['std'] = 1e-6
    
    def compute_features(self, stock, open_price, high, low, close, volume):
        buffer = self.stock_buffers[stock]
        
        if len(buffer['close']) < 2:
            return None
        
        features = {}
        
        prev_close = buffer['close'][-2]
        features['return_1m'] = (close - prev_close) / prev_close if prev_close > 0 else 0
        
        if len(buffer['close']) >= 6:
            close_5m_ago = buffer['close'][-6]
            features['return_5m'] = (close - close_5m_ago) / close_5m_ago if close_5m_ago > 0 else 0
        else:
            features['return_5m'] = 0
        
        features['hl_spread_pct'] = (high - low) / open_price if open_price > 0 else 0
        
        vol_mean = self.volume_stats[stock]['mean']
        vol_std = self.volume_stats[stock]['std']
        features['volume_zscore'] = (volume - vol_mean) / vol_std if vol_std > 1e-6 else 0
        
        if len(buffer['close']) >= 15:
            recent_closes = np.array(list(buffer['close'])[-15:])
            price_mean = np.mean(recent_closes)
            features['price_dev_pct'] = (close - price_mean) / price_mean if price_mean > 0 else 0
        else:
            features['price_dev_pct'] = 0
        
        market_snapshot = self._get_current_market_snapshot()
        if market_snapshot and len(market_snapshot) > 0:
            market_returns = [s['return_1m'] for s in market_snapshot]
            market_volatilities = [s['hl_spread_pct'] for s in market_snapshot]
            
            market_avg_return = np.mean(market_returns)
            market_avg_volatility = np.mean(market_volatilities)
            
            features['return_vs_market'] = features['return_1m'] - market_avg_return
            features['volatility_vs_market'] = features['hl_spread_pct'] - market_avg_volatility
        else:
            features['return_vs_market'] = 0
            features['volatility_vs_market'] = 0
        
        return features
    
    def _get_current_market_snapshot(self):
        if len(self.market_buffer) == 0:
            return None
        return list(self.market_buffer)
    
    def update_market_snapshot(self, stock, features):
        self.market_buffer.append({
            'stock': stock,
            'return_1m': features['return_1m'],
            'hl_spread_pct': features['hl_spread_pct']
        })
    
    def get_feature_vector(self, features):
        feature_order = [
            'return_1m', 'return_5m', 'hl_spread_pct',
            'volume_zscore', 'price_dev_pct',
            'return_vs_market', 'volatility_vs_market'
        ]
        return np.array([features.get(f, 0) for f in feature_order])


def start_consumer(host='localhost', port=9999):
    """
    Connect to producer, receive data, and detect anomalies.
    """
    print("Loading model artifacts...")
    iforest = joblib.load('models/isolation_forest.pkl')
    scaler = joblib.load('models/scaler.pkl')
    threshold = joblib.load('models/optimal_threshold.pkl')
    print(f"Model loaded. Threshold: {threshold:.4f}")
    
    feature_engine = StreamingFeatureEngine()
    
    os.makedirs('logs', exist_ok=True)
    log_file = f"logs/streaming_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    anomaly_log = []
    
    print(f"Connecting to producer at {host}:{port}...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    print("Connected!")
    
    buffer = ""
    rows_processed = 0
    anomaly_count = 0
    
    print("Starting to receive data...")
    
    while True:
        data = client_socket.recv(4096)
        if not data:
            break
        
        buffer += data.decode('utf-8')
        
        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            if not line.strip():
                continue
            
            try:
                row_data = json.loads(line)
                stock = row_data['Stock']
                timestamp = pd.to_datetime(row_data['date'])
                open_price = row_data['open']
                high = row_data['high']
                low = row_data['low']
                close = row_data['close']
                volume = row_data['volume']
                
                feature_engine.update_stock(stock, timestamp, open_price, high, low, close, volume)
                features = feature_engine.compute_features(stock, open_price, high, low, close, volume)
                
                if features is not None:
                    feature_engine.update_market_snapshot(stock, features)
                    feature_vector = feature_engine.get_feature_vector(features)
                    feature_scaled = scaler.transform([feature_vector])
                    raw_score = iforest.score_samples(feature_scaled)[0]
                    outlier_score = -raw_score
                    
                    is_anomaly = outlier_score >= threshold
                    
                    if is_anomaly:
                        anomaly_count += 1
                        log_entry = {
                            'timestamp': timestamp.isoformat(),
                            'stock': stock,
                            'price': close,
                            'volume': volume,
                            'anomaly_score': outlier_score,
                            **features
                        }
                        anomaly_log.append(log_entry)
                        print(f"ANOMALY: {stock} | {timestamp} | Price: ₹{close:.2f} | Score: {outlier_score:.4f}")
                
                rows_processed += 1
                if rows_processed % 1000 == 0:
                    print(f"Processed {rows_processed} rows | Anomalies: {anomaly_count}")
            
            except Exception as e:
                print(f"Error processing row: {e}")
    
    print(f"\nFinished. Total rows: {rows_processed} | Anomalies: {anomaly_count}")
    
    if anomaly_log:
        df = pd.DataFrame(anomaly_log)
        df.to_csv(log_file, index=False)
        print(f"Anomalies saved to {log_file}")
    
    client_socket.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Data Consumer - Receive streaming data and detect anomalies')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Socket host')
    parser.add_argument('--port', type=int, default=9999,
                        help='Socket port')
    
    args = parser.parse_args()
    start_consumer(host=args.host, port=args.port)

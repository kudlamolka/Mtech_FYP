#!/usr/bin/env python3
"""
Data Producer - Reads CSV and sends data row-by-row via socket to simulate streaming.
"""

import pandas as pd
import socket
import json
import time
import argparse

def start_producer(csv_path, host='localhost', port=9999, delay=0.01):
    """
    Read CSV and send each row as JSON via socket.
    
    Args:
        csv_path: Path to CSV file
        host: Socket host
        port: Socket port
        delay: Delay between rows in seconds
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'Stock']).reset_index(drop=True)
    
    print(f"Loaded {len(df)} records")
    print(f"Starting producer on {host}:{port}")
    print(f"Delay between rows: {delay}s")
    
    # Create socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)
    
    print(f"Waiting for consumer to connect...")
    conn, addr = server_socket.accept()
    print(f"Consumer connected from {addr}")
    
    # Send data row by row
    for idx, row in df.iterrows():
        # Convert row to dict and serialize
        data = {
            'Stock': row['Stock'],
            'date': row['date'].isoformat(),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': int(row['volume'])
        }
        
        # Send as JSON string with newline delimiter
        message = json.dumps(data) + '\n'
        conn.sendall(message.encode('utf-8'))
        
        # Progress update
        if (idx + 1) % 1000 == 0:
            print(f"Sent {idx + 1}/{len(df)} rows")
        
        # Delay to simulate real-time
        if delay > 0:
            time.sleep(delay)
    
    print(f"Finished sending {len(df)} rows")
    conn.close()
    server_socket.close()
    print("Producer closed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Producer - Simulate streaming from CSV')
    parser.add_argument('--input', type=str, default='data/all_stocks_with_anomalies.csv',
                        help='Path to CSV file')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Socket host')
    parser.add_argument('--port', type=int, default=9999,
                        help='Socket port')
    parser.add_argument('--delay', type=float, default=0.01,
                        help='Delay between rows in seconds')
    
    args = parser.parse_args()
    
    start_producer(
        csv_path=args.input,
        host=args.host,
        port=args.port,
        delay=args.delay
    )

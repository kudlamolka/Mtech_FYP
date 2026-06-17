#!/usr/bin/env python3
"""
Script to fetch minute-by-minute price data for ITC from Zerodha Kite Connect API
Date range: 17th May 2026 12:00AM IST to 23rd May 2026 11:59PM IST
"""

import logging
import os
import sys

import pandas as pd
from kiteconnect import KiteConnect
import datetime
import time

logger = logging.getLogger(__name__)


def _open_log_file():
    """Open a timestamped log file, creating the logs/ directory if needed."""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_Log.txt"
    return open(os.path.join(log_dir, filename), "w", encoding="utf-8")

# Configuration - Replace with your actual credentials
API_KEY = "bjc63t810yvbajfq"
API_SECRET = "u9dbg8j6bc60egbbfgtdj5s5tq5g0r8j"

# ITC Instrument Token (NSE) - You may need to verify this
# ITC NSE instrument token is typically 438881

EXCHANGE = "NSE"
#TRADING_SYMBOL_LIST = []
TRADING_SYMBOL_LIST = ["ABB","ADANIENSOL","ADANIENT","ADANIGREEN","ADANIPORTS","ADANIPOWER","AMBUJACEM","APOLLOHOSP","ASIANPAINT","DMART","AXISBANK","BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV","BAJAJHLDNG","BANKBARODA","BEL","BPCL","BHARTIARTL","BOSCHLTD","BRITANNIA","CGPOWER","CANBK","CHOLAFIN","CIPLA","COALINDIA","CUMMINSIND","DLF","DIVISLAB","DRREDDY","EICHERMOT","ETERNAL","GAIL","GODREJCP","GRASIM","HCLTECH","HDFCAMC","HDFCBANK","HDFCLIFE","HINDALCO","HAL","HINDUNILVR","HINDZINC","HYUNDAI","ICICIBANK","ITC","INDHOTEL","IOC","IRFC","INFY","INDIGO","JSWSTEEL","JINDALSTEL","JIOFIN","KOTAKBANK","LTM","LT","LODHA","M&M","MARUTI","MAXHEALTH","MAZDOCK","MUTHOOTFIN","NTPC","NESTLEIND","ONGC","PIDILITIND","PFC","POWERGRID","PNB","RECLTD","RELIANCE","SBILIFE","MOTHERSON","SHREECEM","SHRIRAMFIN","ENRIN","SIEMENS","SOLARINDS","SBIN","SUNPHARMA","TVSMOTOR","TATACAP","TCS","TATACONSUM","TMCV","TMPV","TATAPOWER","TATASTEEL","TECHM","TITAN","TORNTPHARM","TRENT","ULTRACEMCO","UNIONBANK","UNITDSPR","VBL","VEDL","WIPRO","ZYDUSLIFE"]
# Date range
START_DATE = "2026-05-31 00:00:00"
END_DATE = "2026-06-15 00:00:00"

def get_access_token(change=False):
    """
    Generate access token using Kite Connect OAuth flow.
    This requires user to login through browser.
    """

    kite = KiteConnect(api_key=API_KEY)
    
    print(f"Please visit this URL to login: {kite.login_url()}")
    print("After login, you will be redirected to a URL with 'request_token' parameter.")
    request_token = input("Enter the request_token from the redirect URL: ")

        
    try:
        data = kite.generate_session(request_token=request_token, api_secret=API_SECRET)
        access_token = data["access_token"]
        print(f"Access token generated successfully.")
        return access_token
    except KeyError:
        raise RuntimeError(
            "Unexpected session response: 'access_token' key missing from API response."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to generate Kite session: {e}") from e

def get_instrument_token(trading_symbol):
    """
    Get instrument token for given trading symbol.
    """
    try:
        kite = KiteConnect(api_key=API_KEY)
        instruments = kite.instruments()
        for instrument in instruments:
            if instrument['tradingsymbol'] == trading_symbol and instrument['exchange'] == EXCHANGE and instrument['instrument_type'] == 'EQ':
                logger.info("Instrument token for %s: %s", trading_symbol, instrument['instrument_token'])
                return instrument['instrument_token']
        logger.warning("No instrument found for symbol=%s exchange=%s", trading_symbol, EXCHANGE)
        return None
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch instrument token for {trading_symbol}: {e}"
        ) from e

def fetch_historical_data(kite, instrument_token, from_date, to_date, interval="minute"):
    """
    Fetch historical data for given instrument and date range.
    """
    try:
        data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
            continuous=False,
            oi=False
        )
        return data
    except Exception as e:
        logger.error(
            "Error fetching historical data for token %s (%s -> %s): %s",
            instrument_token, from_date, to_date, e,
        )
        raise

def main(file):
    access_token = get_access_token()
    
    kite = KiteConnect(api_key=API_KEY, access_token=access_token)
    
    start_dt = datetime.datetime.strptime(START_DATE, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.datetime.strptime(END_DATE, "%Y-%m-%d %H:%M:%S")

    for trading_symbol in TRADING_SYMBOL_LIST:
        print("=" * 60)
        print("ITC Minute Data Fetcher - Zerodha Kite Connect")
        print("=" * 60)
        
        print(f"\nFetching data for {trading_symbol} ({EXCHANGE})")
        print(f"From: {START_DATE}")
        print(f"To: {END_DATE}")
        print("-" * 60)
        
        all_data = []
        current_date = start_dt
        instrument_token = get_instrument_token(trading_symbol)
        if not instrument_token:
            logger.warning("Skipping %s: instrument token not found.", trading_symbol)
            continue
        
        while current_date < end_dt:
            day_end = current_date + datetime.timedelta(days=1)
            if day_end > end_dt:
                day_end = end_dt
            
            from_str = current_date.strftime("%Y-%m-%d %H:%M:%S")
            to_str = day_end.strftime("%Y-%m-%d %H:%M:%S")
            
            if current_date.day == 1:
                print(f"Fetching for {trading_symbol} from {current_date.month}...")
            data = fetch_historical_data(
                kite, 
                instrument_token,  
                from_str, 
                to_str, 
                interval="minute"
            )

            
            if data:
                all_data.extend(data)
            current_date = day_end
            

        
        print("-" * 60)
        print(f"Total records fetched: {len(all_data)}")
        
        if all_data:
            expected_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            df = pd.DataFrame(all_data)
            if len(df.columns) != len(expected_cols):
                raise ValueError(
                    f"Expected {len(expected_cols)} columns from API for {trading_symbol}, "
                    f"got {len(df.columns)}: {list(df.columns)}"
                )
            df.columns = expected_cols
            df['date'] = pd.to_datetime(df['date'])
            
            df = df.sort_values('date')
            output_file = f"{trading_symbol}_{START_DATE.replace(' ', '_').replace(':', '')}_to_{END_DATE.replace(' ', '_').replace(':', '')}.csv"
            output_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data", "streaming", output_file
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            file.write(f"\nData saved to: {output_file}")
            file.write("\nData Summary:")
            file.write(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            file.write(f"  Total records: {len(df)}")
            file.write(f"  Price range: ₹{df['low'].min():.2f} - ₹{df['high'].max():.2f}")
            file.write(f"  Total volume: {df['volume'].sum():,}")

        else:
            logger.warning("No data fetched for %s.", trading_symbol)
            file.write(f"\nNo data fetched for {trading_symbol}.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    file = _open_log_file()
    try:
        main(file)
        end_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Job ended at {end_ts}")
    except Exception:
        logger.exception("Fatal error during data fetch")
        sys.exit(1)
    finally:
        file.close()


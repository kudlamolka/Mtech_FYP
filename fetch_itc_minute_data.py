#!/usr/bin/env python3
"""
Script to fetch minute-by-minute price data for ITC from Zerodha Kite Connect API
Date range: 17th May 2026 12:00AM IST to 23rd May 2026 11:59PM IST
"""

import pandas as pd
from kiteconnect import KiteConnect
import datetime
import time

filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_Log.txt"
file = open(f'logs/{filename}', "w",encoding="utf-8")

# Configuration - Replace with your actual credentials
API_KEY = "bjc63t810yvbajfq"
API_SECRET = "u9dbg8j6bc60egbbfgtdj5s5tq5g0r8j"

# ITC Instrument Token (NSE) - You may need to verify this
# ITC NSE instrument token is typically 438881

EXCHANGE = "NSE"
#TRADING_SYMBOL_LIST = []
TRADING_SYMBOL_LIST = ["ABB","ADANIENSOL","ADANIENT","ADANIGREEN","ADANIPORTS","ADANIPOWER","AMBUJACEM","APOLLOHOSP","ASIANPAINT","DMART","AXISBANK","BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV","BAJAJHLDNG","BANKBARODA","BEL","BPCL","BHARTIARTL","BOSCHLTD","BRITANNIA","CGPOWER","CANBK","CHOLAFIN","CIPLA","COALINDIA","CUMMINSIND","DLF","DIVISLAB","DRREDDY","EICHERMOT","ETERNAL","GAIL","GODREJCP","GRASIM","HCLTECH","HDFCAMC","HDFCBANK","HDFCLIFE","HINDALCO","HAL","HINDUNILVR","HINDZINC","HYUNDAI","ICICIBANK","ITC","INDHOTEL","IOC","IRFC","INFY","INDIGO","JSWSTEEL","JINDALSTEL","JIOFIN","KOTAKBANK","LTM","LT","LODHA","M&M","MARUTI","MAXHEALTH","MAZDOCK","MUTHOOTFIN","NTPC","NESTLEIND","ONGC","PIDILITIND","PFC","POWERGRID","PNB","RECLTD","RELIANCE","SBILIFE","MOTHERSON","SHREECEM","SHRIRAMFIN","ENRIN","SIEMENS","SOLARINDS","SBIN","SUNPHARMA","TVSMOTOR","TATACAP","TCS","TATACONSUM","TMCV","TMPV","TATAPOWER","TATASTEEL","TECHM","TITAN","TORNTPHARM","TRENT","ULTRACEMCO","UNIONBANK","UNITDSPR","VBL","VEDL","WIPRO","ZYDUSLIFE"]
# Date range
START_DATE = "2026-03-31 00:00:00"
END_DATE = "2026-05-31 00:00:00"

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
        print(f"Access token generated: {access_token}")
        return access_token
    except Exception as e:
        print(f"Error generating session: {e}")
        return None

def get_instrument_token(trading_symbol):
    """
    Get instrument token for given trading symbol.
    """
    try:
        kite = KiteConnect(api_key=API_KEY)
        instruments = kite.instruments()
        for instrument in instruments:
            if instrument['tradingsymbol'] == trading_symbol and instrument['exchange'] == EXCHANGE and instrument['instrument_type'] == 'EQ':
                print("==== Instrument Token for " + trading_symbol + " is " + str(instrument['instrument_token']))
                return instrument['instrument_token']
        return None
    except Exception as e:
        print(f"Error getting instrument token: {e}")
        return None

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
        print(f"Error fetching historical data: {e}")
        return None

def main():
    access_token = get_access_token()
    if not access_token:
        print("Failed to get access token. Exiting.")
        return
    
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
            print(f"Error: Could not get instrument token for {trading_symbol}")
            continue
        
        
        while current_date < end_dt
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
            else:
                file.write(f"  ✗ Failed to fetch data for this period: {from_str} to {to_str}\n")
            current_date = day_end
            

        
        print("-" * 60)
        print(f"Total records fetched: {len(all_data)}")
        
        if all_data:
            df = pd.DataFrame(all_data)
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            df['date'] = pd.to_datetime(df['date'])
            
            df = df.sort_values('date')
            output_file = f"{trading_symbol}_{START_DATE.replace(' ', '_').replace(':', '')}_to_{END_DATE.replace(' ', '_').replace(':', '')}.csv"
            df.to_csv(f"results/{output_file}", index=False)
            file.write(f"\nData saved to: {output_file}")
            file.write("\nData Summary:")
            file.write(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            file.write(f"  Total records: {len(df)}")
            file.write(f"  Price range: ₹{df['low'].min():.2f} - ₹{df['high'].max():.2f}")
            file.write(f"  Total volume: {df['volume'].sum():,}")
            

        else:
            file.write("No data fetched.")

if __name__ == "__main__":
    try:
        main()
        print(f"job Ended at {datetime.datetime.strptime(START_DATE, "%Y-%m-%d %H:%M:%S")}")
    finally:
        file.close()


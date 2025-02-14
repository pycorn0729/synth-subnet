import os
import requests
import time
import cupy as cp  # GPU-accelerated NumPy
import cudf  # GPU-accelerated pandas
from datetime import datetime, timedelta

BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
LIMIT = 1000
RATE_LIMIT = 1200
CSV_FILE_PATH = "data/btc_historical_data.csv"

def fetch_historical_data():
    """Fetch only missing historical BTC data and update the CSV file using GPU acceleration."""
    
    # Check if GPU is available
    try:
        cudf.Series([1])  # Simple test for cuDF
        print("✅ GPU acceleration enabled (cuDF + cuPy)")
    except Exception as e:
        print("⚠️ GPU not available, falling back to CPU pandas.")
        import pandas as pd  # Import pandas as fallback
    
    # Determine start time
    if os.path.exists(CSV_FILE_PATH):
        df_existing = cudf.read_csv(CSV_FILE_PATH)
        df_existing["timestamp"] = cudf.to_datetime(df_existing["timestamp"])
        last_timestamp = int(df_existing["timestamp"].max().timestamp() * 1000)
        print(f"🔍 Last recorded timestamp: {df_existing['timestamp'].max()}")
    else:
        os.makedirs("data", exist_ok=True)
        df_existing = cudf.DataFrame()
        last_timestamp = int((datetime.utcnow() - timedelta(days=7)).timestamp() * 1000)  # Default to 7 days
        print("📂 No existing data found, fetching full 7 days.")
    
    end_timestamp = int(datetime.utcnow().timestamp() * 1000)  # Latest available time
    all_data = []
    current_start_time = last_timestamp + 1  # Avoid duplicate timestamps
    request_count = 0
    
    while current_start_time < end_timestamp:
        params = {
            "symbol": SYMBOL, 
            "interval": INTERVAL, 
            "startTime": current_start_time, 
            "endTime": end_timestamp,  
            "limit": LIMIT
        }
        
        response = requests.get(BINANCE_API_URL, params=params)
        request_count += 1
        
        if response.status_code != 200:
            print(f"❌ Error: Failed to fetch data! Status Code: {response.status_code}")
            print(response.text)
            break
        
        data = response.json()
        if not data:
            print("❌ No data returned from Binance API.")
            break
        
        for entry in data:
            timestamp = entry[0]
            if timestamp >= end_timestamp:
                break
            all_data.append([timestamp, float(entry[4])])
        
        # Update start time for next API call
        current_start_time = int(data[-1][0]) + 1
        
        if request_count >= RATE_LIMIT:
            print("⏳ Reached API rate limit, waiting 60 seconds...")
            time.sleep(60)
            request_count = 0
    
    if all_data:
        df_new = cudf.DataFrame(all_data, columns=["timestamp", "close"])
        df_new["timestamp"] = cudf.to_datetime(df_new["timestamp"], unit="ms")
        df_new["close"] = df_new["close"].astype("float32")  # Reduce memory usage
        
        # Merge with existing data
        if not df_existing.empty:
            df_combined = cudf.concat([df_existing, df_new]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        else:
            df_combined = df_new
        
        df_combined.to_csv(CSV_FILE_PATH, index=False)
        print(f"✅ Data successfully updated in '{CSV_FILE_PATH}'")
    else:
        print("⚠️ No new data to update.")

if __name__ == "__main__":
    fetch_historical_data()

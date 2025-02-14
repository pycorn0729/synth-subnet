from train_model import train_lstm_model
from fetch_datatt import fetch_historical_data
from generate_price import generate_simulations
import time
import tensorflow as tf

def main():

    while True:
        print("🚀 Fetching BTC historical data (GPU-accelerated)...")
        fetch_historical_data()  # Uses cuDF on GPU
        
        print("🧠 Training LSTM model (GPU-accelerated)...")
        train_lstm_model()  # Uses TensorFlow GPU
        print("✅ Model training complete! Sleeping for 5 minutes...")
        
        generate_simulations()
        print("✅Generate simulation Success!")
        time.sleep(60)

if __name__ == "__main__":
    main()

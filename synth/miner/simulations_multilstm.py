import os
import numpy as np
import requests
import pandas as pd
import tensorflow as tf
import joblib


from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import Huber
from synth.miner.price_simulation import (
    simulate_crypto_price_paths,
    get_asset_price,
)
from synth.utils.helpers import (
    convert_prices_to_time_format,
)




MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/lstm_model.h5"))
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/btc_historical_data.csv"))
SCALER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/scaler.pkl"))
# Define the file path
output_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/predictions.csv"))


def get_asset_price(asset="BTC"):
    """
    Retrieves the current price of the specified asset.
    Currently, supports BTC via Pyth Network.

    Returns:
        float: Current asset price.



    """
    if asset == "BTC":
        btc_price_id = (
            "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43"
        )
        endpoint = f"https://hermes.pyth.network/api/latest_price_feeds?ids[]={btc_price_id}"  # TODO: this endpoint is deprecated
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            data = response.json()
            if not data or len(data) == 0:
                raise ValueError("No price data received")
            price_feed = data[0]
            price = float(price_feed["price"]["price"]) / (10**8)
            return price
        except Exception as e:
            print(f"Error fetching {asset} price: {str(e)}")
            return None
    else:
        # For other assets, implement accordingly
        print(f"Asset '{asset}' not supported.")
        return None

def generate_price_predictions(asset, start_time, time_increment, time_length, num_simulations):
    """Generate BTC price predictions using LSTM and Monte Carlo simulation."""
    
    num_steps = (time_length/time_increment) + 1
    simulated_paths = np.zeros(num_simulations, num_steps)
    # Load scaler for reversing normalization
    scaler = joblib.load(SCALER_PATH)

    # Load trained LSTM model
    model = load_model(MODEL_PATH)

    # Load historical data
    df = pd.read_csv(DATA_PATH)
    last_p = df[["close"]].values[-1]
    # Extract closing prices and normalize
    df["scaled_price"] = scaler.transform(df[["close"]].values)

    if isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time)  # Convert from string to datetime
    start_time = start_time.replace(second=0, microsecond=0)

    # Generate timestamps for predictions (289 total points)
    for i in range(num_simulations):
        timestamps = [start_time + timedelta(seconds=i * time_increment) for i in range(time_length // time_increment + 1)]
        last_30_prices = df["scaled_price"].values[-30:].reshape(1, 30, 1)
        for j in range(len(timestamps)):  # Predict 289 points
            predicted_normalized = model.predict(last_30_prices, verbose=0)[0][0]
            predicted_price = scaler.inverse_transform([[predicted_normalized]])[0][0]
            simulated_paths[i, j] = predicted_price    # Update input sequence for LSTM
            last_30_prices = np.roll(last_30_prices, -1)
            last_30_prices[0, -1, 0] = predicted_normalized
    return simulated_paths  # List of lists, each with 289 values


def generate_simulations(
    asset="BTC",
    start_time= None,
    time_increment=300,
    time_length=86400,
    num_simulations=100,
):
    
    """
    Generate simulated price paths.

    Parameters:
        asset (str): The asset to simulate. Default is 'BTC'.
        start_time (str): The start time of the simulation. Defaults to current time.
        time_increment (int): Time increment in seconds.
        time_length (int): Total time length in seconds.
        num_simulations (int): Number of simulation runs.

    Returns:
        numpy.ndarray: Simulated price paths.
    """

    if start_time is None:
        raise ValueError("Start time must be provided.")

    # current_price = get_asset_price(asset)
    # if current_price is None:
    #     raise ValueError(f"Failed to fetch current price for asset: {asset}")


    simulations = generate_price_predictions(
    asset="BTC",
    start_time= start_time,
    time_increment=time_increment,
    time_length=time_length,
    num_simulations=num_simulations
    )

    predictions = convert_prices_to_time_format(
        simulations.tolist(), start_time, time_increment
    )

    
    return predictions
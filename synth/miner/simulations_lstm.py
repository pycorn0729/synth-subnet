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


def monte_carlo_simulation(S0, mu, sigma, last_pri, num_simulations=100, T=288, lambda_short=0.95, lambda_long=0.85, alpha=0.7):
    """
    Monte Carlo simulation for BTC price paths with bias correction and adaptive volatility.

    Parameters:
        S0 (list): LSTM-predicted prices (base values for Monte Carlo).
        mu (float): Mean of log returns.
        sigma (float): Standard deviation of log returns.
        last_pri (float): Last actual BTC price.
        num_simulations (int): Number of simulation runs.
        T (int): Total prediction steps (e.g., 288 for 24h with 5min interval).
        lambda_short (float): Bias correction weight for short-term.
        lambda_long (float): Bias correction weight for long-term.
        alpha (float): Decay factor for volatility.

    Returns:
        list: Simulated paths with bias correction and adaptive volatility.
    """
    
    dt = 5 / 1440  # Convert 5 minutes into fraction of a day
    num_predictions = len(S0)

    simulated_paths = np.zeros((num_simulations, num_predictions))

    for i in range(num_predictions):  
        S1 = S0[i]  # LSTM predicted price at this timestep
        actual_price = last_pri  # Last known BTC price

        for j in range(num_simulations):  # Generate multiple paths
            t = i  # Time index
            
            # Adaptive volatility scaling
            sigma_t = sigma * ((T - t) / T) ** alpha
            
            # Bias correction (blend simulated with real price)
            lambda_t = lambda_short if t < T * 0.3 else lambda_long  # Use different lambda for short/long term
            
            rand = np.random.normal(0, 1)  # Random noise
            
            # New price estimation
            new_price = lambda_t * S1 + (1 - lambda_t) * actual_price
            new_price *= np.exp((mu - 0.5 * sigma_t**2) * dt + sigma_t * np.sqrt(dt) * rand)

            simulated_paths[j, i] = new_price  # Store in array

    return simulated_paths


def generate_price_predictions(asset, start_time, time_increment, time_length, num_simulations):
    """Generate BTC price predictions using LSTM and Monte Carlo simulation."""
    
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
    timestamps = [start_time + timedelta(seconds=i * time_increment) for i in range(time_length // time_increment + 1)]

    # Extract last 30 time steps for LSTM input
    last_30_prices = df["scaled_price"].values[-30:].reshape(1, 30, 1)

    predicted_prices = []

    for _ in range(len(timestamps)):  # Predict 289 points
        predicted_normalized = model.predict(last_30_prices, verbose=0)[0][0]
        predicted_price = scaler.inverse_transform([[predicted_normalized]])[0][0]
        predicted_prices.append(predicted_price)

        # Update input sequence for LSTM
        last_30_prices = np.roll(last_30_prices, -1)
        last_30_prices[0, -1, 0] = predicted_normalized

    # Compute log returns for Monte Carlo simulation
    prices = df["close"].values
    log_returns = np.log(prices[1:] / prices[:-1])
    mu = np.mean(log_returns)
    sigma = np.std(log_returns)
    print("mu=", mu, "sigma=", sigma)
    # Generate Monte Carlo simulated paths
    monte_carlo_results = monte_carlo_simulation(
        S0=predicted_prices,  # Start with the first predicted price (which includes current price)
        mu=mu,
        sigma=sigma,
        last_pri = last_p
    )
    print("mu = ", mu)
    print("sigma = ", sigma)


    return monte_carlo_results  # List of lists, each with 289 values


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
    time_increment=300,
    time_length=86400,
    num_simulations=100
    )

    predictions = convert_prices_to_time_format(
        simulations.tolist(), start_time, time_increment
    )

    
    return predictions

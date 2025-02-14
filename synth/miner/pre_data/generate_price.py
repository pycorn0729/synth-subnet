import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# Define file path
MODEL_PATH = "models/lstm_model.h5"
DATA_PATH = "data/btc_historical_data.csv"
SCALER_PATH = "models/scaler.pkl"
OUTPUT_DIR = "data/"

# Function to get the latest versioned CSV file
def get_latest_csv_version(output_dir, base_filename="predictions"):
    existing_files = [f for f in os.listdir(output_dir) if f.startswith(base_filename) and f.endswith(".csv")]
    
    if not existing_files:
        return os.path.join(output_dir, f"{base_filename}_v1.csv")

    version_numbers = []
    for filename in existing_files:
        try:
            version = int(filename.split("_v")[-1].split(".csv")[0])
            version_numbers.append(version)
        except ValueError:
            continue

    latest_version = max(version_numbers) + 1 if version_numbers else 1
    return os.path.join(output_dir, f"{base_filename}_v{latest_version}.csv")

# Function to save simulations to a CSV file (without timestamps)
def save_simulations_to_csv(simulated_paths):
    latest_csv_path = get_latest_csv_version(OUTPUT_DIR)
    
    df = pd.DataFrame(simulated_paths)
    df.to_csv(latest_csv_path, index=False, header=False)

    print(f"Simulations saved to: {latest_csv_path}")

# Monte Carlo Simulation Function
def monte_carlo_simulation(S0, mu, sigma, last_pri, num_simulations=100, T=288, lambda_short=0.95, lambda_long=0.85, alpha=0.7):
    dt = 5 / 1440  # Convert 5 minutes into fraction of a day
    num_predictions = len(S0)
    simulated_paths = np.zeros((num_simulations, num_predictions))

    for i in range(num_predictions):
        S1 = S0[i]
        actual_price = last_pri

        for j in range(num_simulations):
            t = i
            sigma_t = sigma * ((T - t) / T) ** alpha
            lambda_t = lambda_short if t < T * 0.3 else lambda_long
            rand = np.random.normal(0, 1)
            new_price = lambda_t * S1 + (1 - lambda_t) * actual_price
            new_price *= np.exp((mu - 0.5 * sigma_t**2) * dt + sigma_t * np.sqrt(dt) * rand)
            simulated_paths[j, i] = new_price

    return simulated_paths

# Generate price predictions using LSTM and Monte Carlo
def generate_price_predictions(asset, start_time, time_increment, time_length, num_simulations):
    scaler = joblib.load(SCALER_PATH)
    model = load_model(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    last_p = df[["close"]].values[-1]

    df["scaled_price"] = scaler.transform(df[["close"]].values)

    if isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time)
    start_time = start_time.replace(second=0, microsecond=0)

    last_30_prices = df["scaled_price"].values[-30:].reshape(1, 30, 1)
    predicted_prices = []

    for _ in range(time_length // time_increment + 1):
        predicted_normalized = model.predict(last_30_prices, verbose=0)[0][0]
        predicted_price = scaler.inverse_transform([[predicted_normalized]])[0][0]
        predicted_prices.append(predicted_price)

        last_30_prices = np.roll(last_30_prices, -1)
        last_30_prices[0, -1, 0] = predicted_normalized

    log_returns = np.log(df["close"].values[1:] / df["close"].values[:-1])
    mu, sigma = np.mean(log_returns), np.std(log_returns)

    monte_carlo_results = monte_carlo_simulation(
        S0=predicted_prices,
        mu=mu,
        sigma=sigma,
        last_pri=last_p
    )

    return monte_carlo_results

# Generate Simulations
def generate_simulations(asset="BTC", start_time=None, time_increment=300, time_length=86400, num_simulations=100):
    if start_time is None:
        raise ValueError("Start time must be provided.")

    simulations = generate_price_predictions(
        asset="BTC",
        start_time=start_time,
        time_increment=300,
        time_length=86400,
        num_simulations=100
    )

    save_simulations_to_csv(simulations)
    return simulations

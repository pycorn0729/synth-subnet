import os
import numpy as np
import pandas as pd
from synth.miner.price_simulation import get_asset_price
from synth.utils.helpers import convert_prices_to_time_format

# Define the directory where predictions are stored
PREDICTIONS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/"))

def get_latest_predictions_file():
    """Finds the latest version of the predictions CSV file."""
    existing_files = [f for f in os.listdir(PREDICTIONS_DIR) if f.startswith("predictions_v") and f.endswith(".csv")]
    
    if not existing_files:
        raise FileNotFoundError("No predictions file found in the data directory.")

    # Extract version numbers and find the latest one
    version_numbers = []
    for filename in existing_files:
        try:
            version = int(filename.split("_v")[-1].split(".csv")[0])
            version_numbers.append(version)
        except ValueError:
            continue

    if not version_numbers:
        raise FileNotFoundError("No valid versioned predictions files found.")

    latest_version = max(version_numbers)
    latest_file = f"predictions_v{latest_version}.csv"

    return os.path.join(PREDICTIONS_DIR, latest_file)

def generate_simulations(
    asset="BTC",
    start_time=None,
    time_increment=300,
    time_length=86400,
    num_simulations=1,
):
    """
    Retrieve and format the latest simulated price paths.

    Parameters:
        asset (str): The asset to simulate. Default is 'BTC'.
        start_time (str): The start time of the simulation. Required.
        time_increment (int): Time increment in seconds.
        time_length (int): Total time length in seconds.
        num_simulations (int): Number of simulations to return.

    Returns:
        list: Simulated price paths formatted with timestamps.
    """
    if start_time is None:
        raise ValueError("Start time must be provided.")

    # Load the latest predictions file
    latest_file_path = get_latest_predictions_file()
    print(f"Loading predictions from: {latest_file_path}")

    # Load the Monte Carlo simulated prices (100 simulations × 289 time steps)
    simulated_prices = pd.read_csv(latest_file_path, header=None).values

    # Ensure requested number of simulations does not exceed available data
    if num_simulations > simulated_prices.shape[0]:
        raise ValueError(f"Requested {num_simulations} simulations, but only {simulated_prices.shape[0]} are available.")

    # Select the requested number of simulations
    selected_simulations = simulated_prices[:num_simulations].tolist()

    # Convert price simulations to time format
    predictions = convert_prices_to_time_format(selected_simulations, start_time, time_increment)

    return predictions

import numpy as np

from synth.miner.price_simulation import (
    simulate_crypto_price_paths,
    get_asset_price,
)
from synth.utils.helpers import (
    convert_prices_to_time_format,
)


def generate_normal_distribution_predictions(
    current_price, 
    num_steps=289, 
    num_simulations=100, 
    time_increment=300, 
    a=13.30, 
    b=14.46,
):
    """
    Generate a 289x100 matrix of normally distributed price predictions.
    Each 5-minute interval contains 100 price samples centered around the predicted value.
    """
    predictions = np.zeros((num_simulations, num_steps))
    for t in range(num_steps):
        # Define mean and standard deviation dynamically for each time step
        mean_price = current_price
        std_dev = (a * t + b * t) / 2  # Standard deviation grows over time
        # Generate 100 samples for this time step
        predictions[:, t] = np.random.normal(mean_price, std_dev, num_simulations)
    return predictions.tolist()


def generate_simulations(
    asset="BTC",
    start_time=None,
    time_increment=300,
    time_length=86400,
    num_simulations=1,
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

    current_price = get_asset_price(asset)
    if current_price is None:
        raise ValueError(f"Failed to fetch current price for asset: {asset}")

    # Standard deviation of the simulated price path
    sigma = 0.01

    simulations = generate_normal_distribution_predictions(
        current_price=current_price,
    )

    predictions = convert_prices_to_time_format(
        simulations.tolist(), start_time, time_increment
    )

    return predictions


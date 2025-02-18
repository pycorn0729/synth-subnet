import numpy as np
import requests
from scipy.stats import norm


def get_asset_price(asset="BTC"):
    """
    Retrieves the current price of the specified asset.
    """
    if asset == "BTC":
        btc_price_id = "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43"
        endpoint = f"https://hermes.pyth.network/api/latest_price_feeds?ids[]={btc_price_id}"
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
        print(f"Asset '{asset}' not supported.")
        return None

def simulate_crypto_price_paths(
    current_price, time_increment, time_length, num_simulations, spread_factor
):
    """
    Generate CRPS-optimized price paths using quantile-based interpolation.
    """
    num_steps = int(time_length / time_increment) + 1  # Ensure 289 time steps
    quantiles = np.linspace(0.05, 0.95, num_simulations)  # 100 quantiles
    norm_vals = norm.ppf(quantiles)  # Convert to normal distribution
    price_paths = []

    for i in range(num_simulations):  # Iterate properly
        drift = np.random.normal(0, spread_factor, size=num_steps)  # Now 289 steps
        path = current_price * (1 + norm_vals[i] * drift)  # Correct quantile application
        path = np.maximum(path, 0)  # Prevent negative prices
        price_paths.append(path)  # Append full path

    return np.array(price_paths)  # Output shape (100, 289)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synth.miner.simulations_brute import generate_simulations


def test_generate_simulations():
    result = generate_simulations(
        asset="BTC",
        start_time="2025-02-06T13:10:00+00:00",
        time_increment=300,
        time_length=86400,
        num_simulations=100,
    )
    print(result[1][0])
    print("Running assertions...")
    assert isinstance(result, list)
    print("Passed: result is a list")
    assert len(result) == 100
    print("Passed: length of result is 100")
    assert all(isinstance(sub_array, list) and len(sub_array) == 289 for sub_array in result)
    print("Passed: all sub-arrays are lists of length 289")


if __name__ == "__main__":
    test_generate_simulations()
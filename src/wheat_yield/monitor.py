import numpy as np
import pandas as pd


# Calculate Population Stability Index
def calculate_psi(expected, actual, bins=10):

    # Exclude missing numeric observations
    expected = pd.Series(expected).dropna().to_numpy(dtype=float)
    actual = pd.Series(actual).dropna().to_numpy(dtype=float)

    if len(expected) == 0 or len(actual) == 0:
        raise ValueError("PSI requires non-empty datasets")

    # Create 10 equally spaced bins from the reference dataset
    edges = np.linspace(
        expected.min(),
        expected.max(),
        bins + 1
    )

    # Include later observations outside the training range
    edges[0] = -np.inf
    edges[-1] = np.inf

    # Count observations in each bin
    expected_counts = np.histogram(
        expected, bins=edges
    )[0]

    actual_counts = np.histogram(
        actual, bins=edges
    )[0]

    # Convert counts into proportions
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Prevent division by zero and log(0)
    epsilon = 1e-6

    expected_pct = np.clip(expected_pct, epsilon, None)
    actual_pct = np.clip(actual_pct, epsilon, None)

    # Calculate PSI
    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(actual_pct / expected_pct)
    )

    return float(psi)


# Interpret the PSI score
def classify_drift(psi):

    if psi < 0.10:
        return "Stable"

    elif psi <= 0.25:
        return "Moderate drift"

    else:
        return "Significant drift"
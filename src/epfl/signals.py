"""Signal processing utilities for time-series notebooks.

This module contains helpers to build exponential weights and apply
one-sided convolutions used in AR/EWMA estimators.
"""

import numpy as np
from statsmodels.tsa.filters.filtertools import convolution_filter


def convolution(ts, weights):
    """Apply a convolution filter to a time series using specified weights.

    This function performs a one-sided convolution operation on a time series,
    which is useful for creating autoregressive models and moving averages.

    Args:
        ts: The time series data to filter (pandas Series or array-like).
        weights: The weights to use in the convolution filter.

    Returns:
        a filtered time series resulting from the convolution operation.
    """
    x = convolution_filter(ts, weights, nsides=1)

    # fill the leading NaNs with zeros
    x[np.isnan(x)] = 0.0
    return x


def exp_weights(m, n=100):
    """Generate normalized exponentially decaying weights.

    This function creates a vector of exponentially decaying weights with decay rate
    determined by parameter m. The weights are normalized to have unit norm.

    Args:
        m: The decay parameter controlling the rate of exponential decay.
           Larger values of m result in slower decay.
        n: The number of weights to generate. Defaults to 100.

    Returns:
        a numpy array of normalized exponentially decaying weights.
    """
    x = np.power(1.0 - 1.0 / m, range(1, n + 1))
    s = np.linalg.norm(x)
    return x / s

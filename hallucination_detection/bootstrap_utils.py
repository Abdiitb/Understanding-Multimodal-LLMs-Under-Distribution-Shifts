from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_confidence_interval(
    x: np.ndarray,
    y: np.ndarray,
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    num_bootstrap: int = 1000,
    seed: int = 42,
    lower_percentile: float = 2.5,
    upper_percentile: float = 97.5,
) -> tuple[float, float, np.ndarray]:
    """
    Generic bootstrap CI for a statistic computed from paired samples.

    Args:
        x: 1D numpy array of length N.
        y: 1D numpy array of length N.
        statistic_fn: Callable returning a scalar statistic from (x_sample, y_sample).
        num_bootstrap: Number of bootstrap resamples.
        seed: RNG seed.
        lower_percentile: Lower CI percentile.
        upper_percentile: Upper CI percentile.

    Returns:
        (ci_lower, ci_upper, bootstrap_values)
    """
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Length mismatch: len(x)={x.shape[0]} vs len(y)={y.shape[0]}")
    if num_bootstrap <= 0:
        raise ValueError("num_bootstrap must be > 0")

    n = x.shape[0]
    rng = np.random.default_rng(seed)
    stats: list[float] = []

    for _ in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        value = float(statistic_fn(x[idx], y[idx]))
        if not np.isnan(value):
            stats.append(value)

    if len(stats) == 0:
        raise ValueError("All bootstrap statistics are NaN")

    stats_arr = np.asarray(stats, dtype=float)
    ci_lower = float(np.percentile(stats_arr, lower_percentile))
    ci_upper = float(np.percentile(stats_arr, upper_percentile))
    return ci_lower, ci_upper, stats_arr

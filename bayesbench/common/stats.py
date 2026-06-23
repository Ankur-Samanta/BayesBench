"""Shared statistics helpers for BayesBench aggregation/metrics.

Small, dependency-light functions that several tasks compute identically:
bootstrap confidence intervals, standard error of the mean, and distribution
distances (total variation, Jensen-Shannon).
"""

from typing import List, Sequence, Tuple

import numpy as np


def bootstrap_ci(
    data: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval over the mean.

    Returns ``(mean, lower_bound, upper_bound)`` at confidence level ``ci``.
    Degenerate inputs short-circuit: empty -> all zeros, single value -> that
    value for all three.
    """
    if len(data) == 0:
        return 0.0, 0.0, 0.0
    if len(data) == 1:
        return data[0], data[0], data[0]

    np.random.seed(seed)
    data_arr = np.array(data)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data_arr, size=len(data_arr), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    mean = np.mean(data)

    return float(mean), float(lower), float(upper)


def sem(vals: Sequence[float]) -> float:
    """Standard error of the mean. Returns 0.0 for empty input."""
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(arr.std() / np.sqrt(arr.size))


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two discrete distributions."""
    return float(0.5 * np.sum(np.abs(p - q)))


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (symmetric, bounded [0, log 2])."""
    m = 0.5 * (p + q)
    m = np.clip(m, 1e-10, None)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))

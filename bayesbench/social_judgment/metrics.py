"""
Per-trajectory metric computation for social judgment evaluation experiments.

Computes convergence, update magnitudes, valence/category breakdowns,
and standard trajectory statistics.
"""

from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import numpy as np

from .config import TrajectoryResult, TrajectoryMetrics, PollResult


def bootstrap_ci(
    data: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Returns:
        (mean, lower_bound, upper_bound)
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


def compute_convergence_turn(polls: List[PollResult]) -> Optional[int]:
    """Find first t where |P(t) - P(final)| < 0.05."""
    if not polls:
        return None

    final_p = polls[-1].p_yta
    for poll in polls:
        if abs(poll.p_yta - final_p) < 0.05:
            return poll.t
    return None


def compute_update_by_valence(polls: List[PollResult]) -> Dict[str, float]:
    """Compute mean signed delta P(YTA) grouped by aspect valence."""
    valence_deltas = defaultdict(list)

    for i in range(1, len(polls)):
        valence = polls[i].aspect_valence
        if valence is None:
            continue
        delta = polls[i].p_yta - polls[i - 1].p_yta
        valence_deltas[valence].append(delta)

    return {v: float(np.mean(deltas)) for v, deltas in valence_deltas.items()}


def compute_update_by_category(polls: List[PollResult]) -> Dict[str, float]:
    """Compute mean |delta P(YTA)| grouped by aspect category."""
    category_deltas = defaultdict(list)

    for i in range(1, len(polls)):
        category = polls[i].aspect_category
        if category is None:
            continue
        delta = abs(polls[i].p_yta - polls[i - 1].p_yta)
        category_deltas[category].append(delta)

    return {c: float(np.mean(deltas)) for c, deltas in category_deltas.items()}


def compute_trajectory_metrics(result: TrajectoryResult) -> TrajectoryMetrics:
    """
    Compute all metrics for an experiment trajectory.

    Args:
        result: TrajectoryResult with polls populated

    Returns:
        TrajectoryMetrics
    """
    if not result.polls:
        raise ValueError("No polls in trajectory result")

    polls = result.polls
    p_yta_list = [p.p_yta for p in polls]
    position_biases = [p.position_bias for p in polls]
    ab_masses = [p.ab_mass for p in polls]

    baseline_p_yta = p_yta_list[0]
    final_p_yta = p_yta_list[-1]
    baseline_drift = abs(final_p_yta - baseline_p_yta)

    # Update magnitudes
    update_magnitudes = []
    for i in range(1, len(p_yta_list)):
        update_magnitudes.append(abs(p_yta_list[i] - p_yta_list[i - 1]))

    mean_update_magnitude = float(np.mean(update_magnitudes)) if update_magnitudes else 0.0

    # Convergence
    convergence_turn = compute_convergence_turn(polls)

    # Position bias and AB mass
    mean_position_bias = float(np.mean(position_biases))
    mean_ab_mass = float(np.mean(ab_masses))

    # Trajectory variance
    trajectory_variance = float(np.var(p_yta_list))

    # Valence and category breakdowns
    update_by_valence = compute_update_by_valence(polls)
    update_by_category = compute_update_by_category(polls)

    return TrajectoryMetrics(
        baseline_p_yta=baseline_p_yta,
        final_p_yta=final_p_yta,
        baseline_drift=baseline_drift,
        mean_update_magnitude=mean_update_magnitude,
        update_magnitudes=update_magnitudes,
        convergence_turn=convergence_turn,
        mean_position_bias=mean_position_bias,
        position_biases=position_biases,
        mean_ab_mass=mean_ab_mass,
        ab_masses=ab_masses,
        trajectory_variance=trajectory_variance,
        update_by_valence=update_by_valence,
        update_by_category=update_by_category,
    )

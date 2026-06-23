"""
Per-trajectory metrics for triage experiments.

Computes the headline endpoint accuracies, conditioning lift, and
per-turn trajectories. Designed to be cheap so the runner can call it
inline after each experiment.

Cross-experiment aggregation lives in aggregate.py.
"""

from typing import List, Tuple
import math
import numpy as np

from .config import (
    TrajectoryResult, TrajectoryMetrics, PollResult,
    URGENCY_ORDER, PROFILE_ORDER,
)


def bootstrap_ci(
    data: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval. Returns (mean, lower, upper)."""
    if len(data) == 0:
        return 0.0, 0.0, 0.0
    if len(data) == 1:
        return data[0], data[0], data[0]

    rng = np.random.default_rng(seed)
    data_arr = np.array(data)
    means = rng.choice(data_arr, size=(n_bootstrap, len(data_arr)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    return float(np.mean(data)), float(np.percentile(means, alpha * 100)), float(np.percentile(means, (1 - alpha) * 100))


def _index_of_urgency(value: str) -> int:
    for i, u in enumerate(URGENCY_ORDER):
        if u.value == value:
            return i
    raise ValueError(f"unknown urgency: {value}")


def _index_of_profile(value: str) -> int:
    for i, p in enumerate(PROFILE_ORDER):
        if p.value == value:
            return i
    raise ValueError(f"unknown profile: {value}")


def _argmax(xs: List[float]) -> int:
    return max(range(len(xs)), key=lambda i: xs[i])


def compute_trajectory_metrics(result: TrajectoryResult) -> TrajectoryMetrics:
    """Compute endpoint accuracy + per-turn trajectories for one experiment."""
    polls: List[PollResult] = result.polls
    if not polls:
        return TrajectoryMetrics(
            final_urgency_correct=False, final_profile_correct=False,
        )

    true_urgency_idx = _index_of_urgency(result.config.true_urgency.value)
    true_profile_idx = _index_of_profile(result.config.profile.value)

    final = polls[-1]

    final_urgency_correct = (_argmax(final.urgency_distribution) == true_urgency_idx)
    final_profile_correct = (_argmax(final.profile_distribution) == true_profile_idx)

    final_cot_profile_correct = None
    if final.cot_profile_distribution is not None:
        final_cot_profile_correct = (
            _argmax(final.cot_profile_distribution) == true_profile_idx
        )

    final_conditioned_urgency_correct = None
    conditioning_lift = None
    if final.conditioned_urgency_distribution is not None:
        final_conditioned_urgency_correct = (
            _argmax(final.conditioned_urgency_distribution) == true_urgency_idx
        )
        # Lift in P(true_urgency) — positive means conditioning helps
        p_true_uncond = final.urgency_distribution[true_urgency_idx]
        p_true_cond = final.conditioned_urgency_distribution[true_urgency_idx]
        conditioning_lift = p_true_cond - p_true_uncond

    # Per-turn trajectories of P(true label)
    urgency_trajectory = [p.urgency_distribution[true_urgency_idx] for p in polls]
    profile_trajectory = [p.profile_distribution[true_profile_idx] for p in polls]
    cot_profile_trajectory = [
        p.cot_profile_distribution[true_profile_idx]
        for p in polls if p.cot_profile_distribution is not None
    ]
    conditioned_urgency_trajectory = [
        p.conditioned_urgency_distribution[true_urgency_idx]
        for p in polls if p.conditioned_urgency_distribution is not None
    ]

    # Update magnitudes (L1 between consecutive distributions)
    def _mean_update(trajectories):
        if len(trajectories) < 2:
            return 0.0
        deltas = []
        for prev, curr in zip(trajectories[:-1], trajectories[1:]):
            deltas.append(sum(abs(a - b) for a, b in zip(prev, curr)) / 2)
        return sum(deltas) / len(deltas) if deltas else 0.0

    urgency_dists = [p.urgency_distribution for p in polls]
    profile_dists = [p.profile_distribution for p in polls]
    mean_urgency_update = _mean_update(urgency_dists)
    mean_profile_update = _mean_update(profile_dists)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return TrajectoryMetrics(
        final_urgency_correct=final_urgency_correct,
        final_profile_correct=final_profile_correct,
        final_cot_profile_correct=final_cot_profile_correct,
        final_conditioned_urgency_correct=final_conditioned_urgency_correct,
        conditioning_lift=conditioning_lift,
        urgency_trajectory=urgency_trajectory,
        profile_trajectory=profile_trajectory,
        cot_profile_trajectory=cot_profile_trajectory,
        conditioned_urgency_trajectory=conditioned_urgency_trajectory,
        mean_urgency_update_magnitude=mean_urgency_update,
        mean_profile_update_magnitude=mean_profile_update,
        mean_urgency_scale_bias=_mean([p.urgency_scale_bias for p in polls]),
        mean_profile_scale_bias=_mean([p.profile_scale_bias for p in polls]),
        mean_urgency_mass=_mean([p.urgency_mass for p in polls]),
        mean_profile_mass=_mean([p.profile_mass for p in polls]),
    )

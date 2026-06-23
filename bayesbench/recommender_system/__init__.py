"""
Recommender System Cold-Start Experiments

Task 3: Multi-dimensional Bayesian inference with cross-item transfer.
A new user rates real movies sequentially -> predict whether they'll
like a held-out target movie. Ground truth: mixture-model posterior.
"""

from .config import (
    PopInfo,
    Condition,
    SequenceSource,
    ExperimentConfig,
    PollResult,
    TrajectoryResult,
    TrajectoryMetrics,
)

from .conditions import (
    ConversationState,
    MessageBuilder,
)

from .metrics import (
    mixture_posterior,
    expected_rating_bayesian,
    marginal_baseline,
    genre_overlap_baseline,
    compute_trajectory_metrics,
    bootstrap_ci,
)

from .extraction import (
    extract_type_counterbalanced,
    generate_type_cot,
)

__all__ = [
    # config.py
    "PopInfo",
    "Condition",
    "SequenceSource",
    "ExperimentConfig",
    "PollResult",
    "TrajectoryResult",
    "TrajectoryMetrics",
    # conditions.py
    "ConversationState",
    "MessageBuilder",
    # metrics.py
    "mixture_posterior",
    "expected_rating_bayesian",
    "marginal_baseline",
    "genre_overlap_baseline",
    "compute_trajectory_metrics",
    "bootstrap_ci",
    # extraction.py
    "extract_type_counterbalanced",
    "generate_type_cot",
]

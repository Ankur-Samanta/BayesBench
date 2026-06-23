"""
Coin Flip Experiments

Validates prior extraction and distribution tracking for LLMs.
Tests single-turn vs multi-turn delivery and self-anchoring effects.
"""

from .extraction import (
    setup_model,
    extract_p_heads,
    extract_trajectory,
    extract_ab_probs,
    ExtractionResult
)

from .config import (
    CoinSpec,
    Condition,
    ExperimentConfig,
    PollResult,
    TrajectoryResult,
    TrajectoryMetrics
)

from .conditions import (
    ConversationState,
    MessageBuilder
)

from .metrics import (
    bayesian_posterior,
    compute_trajectory_metrics,
    bootstrap_ci,
    compute_prior_bias,
    compute_updating_exists,
    compute_update_direction_accuracy,
    compute_trajectory_divergence,
    compute_switchover_point,
    compute_update_asymmetry,
    compute_bias_sensitivity
)

from .runner import (
    generate_sequence,
    get_poll_points,
    run_experiment,
    run_single_turn_experiment,
    run_multi_turn_experiment
)

from .aggregate import (
    load_all_results,
    aggregate_by_condition,
    compare_single_vs_multi,
    compute_self_anchoring_index,
    analyze_q1_updating,
    analyze_q2_single_vs_multi,
    analyze_q4_bias_sensitivity
)

__all__ = [
    # extraction.py
    "setup_model",
    "extract_p_heads",
    "extract_trajectory",
    "extract_ab_probs",
    "ExtractionResult",
    # config.py
    "CoinSpec",
    "Condition",
    "ExperimentConfig",
    "PollResult",
    "TrajectoryResult",
    "TrajectoryMetrics",
    # conditions.py
    "ConversationState",
    "MessageBuilder",
    # metrics.py
    "bayesian_posterior",
    "compute_trajectory_metrics",
    "bootstrap_ci",
    "compute_prior_bias",
    "compute_updating_exists",
    "compute_update_direction_accuracy",
    "compute_trajectory_divergence",
    "compute_switchover_point",
    "compute_update_asymmetry",
    "compute_bias_sensitivity",
    # runner.py
    "generate_sequence",
    "get_poll_points",
    "run_experiment",
    "run_single_turn_experiment",
    "run_multi_turn_experiment",
    # aggregate.py
    "load_all_results",
    "aggregate_by_condition",
    "compare_single_vs_multi",
    "compute_self_anchoring_index",
    "analyze_q1_updating",
    "analyze_q2_single_vs_multi",
    "analyze_q4_bias_sensitivity",
]

"""
Main experiment runner for coin flip experiments.

Supports single-turn and multi-turn experimental conditions with CLI interface.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from .config import Condition, CoinSpec, ExperimentConfig, PollResult, TrajectoryResult
from .conditions import MessageBuilder, ConversationState
from .metrics import bayesian_posterior, compute_trajectory_metrics
from .extraction import setup_model, extract_ab_probs, extract_cyclic_with_shifts


def generate_sequence(n: int, p: float, seed: int) -> List[str]:
    """
    Generate n coin flips with P(heads) = p.

    Args:
        n: Number of flips
        p: Probability of heads
        seed: Random seed

    Returns:
        List of "heads" or "tails"
    """
    np.random.seed(seed)
    return ["heads" if np.random.random() < p else "tails" for _ in range(n)]


def get_poll_points(n: int, k: int) -> List[int]:
    """
    Get time points for polling.

    Args:
        n: Total sequence length
        k: Polling frequency

    Returns:
        List of poll points [0, k, 2k, ...] for k>1, or [0, 1, 2, ...] for k=1
    """
    if k == 1:
        return list(range(n + 1))
    else:
        points = list(range(0, n + 1, k))
        if n not in points:
            points.append(n)
        return points


def extract_p_heads_counterbalanced(
    model,
    tokenizer,
    messages_v1: List[dict],
    messages_v2: List[dict]
) -> Tuple[float, float, float, float, float]:
    """
    Extract P(heads) using the shared cyclic counterbalanced MCQ poll (K=2).

    Args:
        model: loaded model
        tokenizer: Model tokenizer
        messages_v1: Messages with A=heads (cyclic shift 0)
        messages_v2: Messages with A=tails (cyclic shift 1)

    Returns:
        (p_heads, p_heads_v1, p_heads_v2, position_bias, ab_mass)
    """
    dist, _pred, _bias, ab_mass, per_shift = extract_cyclic_with_shifts(
        model, tokenizer, [messages_v1, messages_v2], n=2
    )
    # per_shift[k] is normalized over [A, B] for shift k.
    p_heads_v1 = per_shift[0][0]  # P(heads) = P(A) when A=heads
    p_heads_v2 = per_shift[1][1]  # P(heads) = P(B) when A=tails
    position_bias = (per_shift[0][0] + per_shift[1][0]) / 2 - 0.5

    return dist[0], p_heads_v1, p_heads_v2, position_bias, ab_mass


def sample_prediction(model, tokenizer, messages: List[dict]) -> str:
    """
    Sample a prediction (A or B) from the model.

    Args:
        model: loaded model
        tokenizer: Model tokenizer
        messages: Chat messages

    Returns:
        "A" or "B"
    """
    p_a, p_b, _ = extract_ab_probs(model, tokenizer, messages)

    # Sample based on probabilities
    if np.random.random() < p_a:
        return "A"
    else:
        return "B"


def run_single_turn_experiment(
    model,
    tokenizer,
    config: ExperimentConfig,
    sequence: List[str]
) -> TrajectoryResult:
    """
    Run single-turn experiment (Condition 1).

    Each poll is independent - we show full history in a fresh prompt.

    Args:
        model: loaded model
        tokenizer: Model tokenizer
        config: Experiment configuration
        sequence: Flip sequence

    Returns:
        TrajectoryResult
    """
    poll_points = get_poll_points(config.n_flips, config.k)
    polls = []

    for t in poll_points:
        history = sequence[:t]
        n_heads = sum(1 for f in history if f == "heads")
        n_tails = t - n_heads

        # Build messages for both counterbalance versions
        msgs_v1 = MessageBuilder.build_single_turn(history, a_is_heads=True, coin_spec=config.coin_spec)
        msgs_v2 = MessageBuilder.build_single_turn(history, a_is_heads=False, coin_spec=config.coin_spec)

        # Extract counterbalanced probability
        if config.counterbalance:
            p_heads, p_heads_v1, p_heads_v2, position_bias, ab_mass = \
                extract_p_heads_counterbalanced(model, tokenizer, msgs_v1, msgs_v2)
        else:
            p_a, p_b, ab_mass = extract_ab_probs(model, tokenizer, msgs_v1)
            p_heads = p_a
            p_heads_v1 = p_a
            p_heads_v2 = p_a
            position_bias = p_a - 0.5

        # Bayesian posterior
        bayes_post = bayesian_posterior(history)

        poll = PollResult(
            t=t,
            n_heads=n_heads,
            n_tails=n_tails,
            p_heads=p_heads,
            p_heads_v1=p_heads_v1,
            p_heads_v2=p_heads_v2,
            position_bias=position_bias,
            ab_mass=ab_mass,
            bayesian_posterior=bayes_post,
            prediction=None,
            injected=False
        )
        polls.append(poll)

        print(f"  t={t:3d}: P(heads)={p_heads:.3f} [v1={p_heads_v1:.3f}, v2={p_heads_v2:.3f}] "
              f"Bayes={bayes_post:.3f} pos_bias={position_bias:+.3f}")

    result = TrajectoryResult(
        config=config,
        sequence=sequence,
        poll_points=poll_points,
        polls=polls
    )

    # Compute metrics
    result.metrics = compute_trajectory_metrics(result)

    return result


def run_multi_turn_experiment(
    model,
    tokenizer,
    config: ExperimentConfig,
    sequence: List[str]
) -> TrajectoryResult:
    """
    Run multi-turn experiment (Conditions 2-3).

    Maintains conversation state across observations.

    Args:
        model: loaded model
        tokenizer: Model tokenizer
        config: Experiment configuration
        sequence: Flip sequence

    Returns:
        TrajectoryResult
    """
    poll_points = get_poll_points(config.n_flips, config.k)
    polls = []

    # Initialize conversation states for both counterbalance versions
    state_v1 = MessageBuilder.init_multi_turn_state(a_is_heads=True, coin_spec=config.coin_spec)
    state_v2 = MessageBuilder.init_multi_turn_state(a_is_heads=False, coin_spec=config.coin_spec)

    # Track observations for batch mode
    current_t = 0
    current_batch: List[str] = []  # partial batch accumulating toward batch_size

    for poll_t in poll_points:
        # Add one new observation to the partial batch (if any)
        if current_t < poll_t:
            current_batch.append(sequence[current_t])
            current_t += 1

            # Commit completed batch to conversation state
            if len(current_batch) == config.batch_size:
                if config.condition == Condition.MULTI_TURN_MINIMAL:
                    state_v1 = MessageBuilder.add_observation_batch_minimal(state_v1, current_batch)
                    state_v2 = MessageBuilder.add_observation_batch_minimal(state_v2, current_batch)
                elif config.condition == Condition.MULTI_TURN_ACTUAL:
                    poll_msgs = MessageBuilder.build_poll_prompt(state_v1, a_is_heads=True)
                    prediction = sample_prediction(model, tokenizer, poll_msgs)
                    state_v1 = MessageBuilder.add_observation_batch_with_prediction(
                        state_v1, current_batch, prediction, a_is_heads=True
                    )
                    state_v2 = MessageBuilder.add_observation_batch_with_prediction(
                        state_v2, current_batch, prediction, a_is_heads=False
                    )
                current_batch = []

        # Poll at poll_t — include any partial batch as context
        history = sequence[:poll_t]
        n_heads = sum(1 for f in history if f == "heads")
        n_tails = poll_t - n_heads

        # Build poll prompts (partial_batch shows in-progress flips)
        if config.k == 5 and poll_t > 0 and config.batch_size == 1:
            # Legacy k=5 batch poll (only when not using batch_size sweep)
            start_idx = max(0, poll_t - 5)
            last_batch = sequence[start_idx:poll_t]
            msgs_v1 = MessageBuilder.build_poll_prompt_k5(
                state_v1, last_batch, n_heads, n_tails, a_is_heads=True
            )
            msgs_v2 = MessageBuilder.build_poll_prompt_k5(
                state_v2, last_batch, n_heads, n_tails, a_is_heads=False
            )
        else:
            msgs_v1 = MessageBuilder.build_poll_prompt_with_partial(
                state_v1, current_batch, a_is_heads=True
            )
            msgs_v2 = MessageBuilder.build_poll_prompt_with_partial(
                state_v2, current_batch, a_is_heads=False
            )

        # Extract counterbalanced probability
        if config.counterbalance:
            p_heads, p_heads_v1, p_heads_v2, position_bias, ab_mass = \
                extract_p_heads_counterbalanced(model, tokenizer, msgs_v1, msgs_v2)
        else:
            p_a, p_b, ab_mass = extract_ab_probs(model, tokenizer, msgs_v1)
            p_heads = p_a
            p_heads_v1 = p_a
            p_heads_v2 = p_a
            position_bias = p_a - 0.5

        # Bayesian posterior
        bayes_post = bayesian_posterior(history)

        # Get prediction if applicable
        last_prediction = None
        is_injected = False
        if config.condition != Condition.MULTI_TURN_MINIMAL and state_v1.predictions:
            last_prediction = state_v1.predictions[-1]

        poll = PollResult(
            t=poll_t,
            n_heads=n_heads,
            n_tails=n_tails,
            p_heads=p_heads,
            p_heads_v1=p_heads_v1,
            p_heads_v2=p_heads_v2,
            position_bias=position_bias,
            ab_mass=ab_mass,
            bayesian_posterior=bayes_post,
            prediction=last_prediction,
            injected=is_injected
        )
        polls.append(poll)

        print(f"  t={poll_t:3d}: P(heads)={p_heads:.3f} [v1={p_heads_v1:.3f}, v2={p_heads_v2:.3f}] "
              f"Bayes={bayes_post:.3f} pos_bias={position_bias:+.3f}")

    result = TrajectoryResult(
        config=config,
        sequence=sequence,
        poll_points=poll_points,
        polls=polls
    )

    # Compute metrics
    result.metrics = compute_trajectory_metrics(result)

    return result


def run_experiment(config: ExperimentConfig, model=None, tokenizer=None) -> TrajectoryResult:
    """
    Run a single experiment.

    Args:
        config: Experiment configuration
        model: Optional pre-loaded model
        tokenizer: Optional pre-loaded tokenizer

    Returns:
        TrajectoryResult
    """
    # Load model if not provided
    if model is None or tokenizer is None:
        model, tokenizer = setup_model(config.model_name)

    # Generate sequence
    sequence = generate_sequence(config.n_flips, config.p, config.seed)

    print(f"\nRunning experiment:")
    print(f"  Condition: {config.condition.value}")
    print(f"  Model: {config.model_name}")
    print(f"  k={config.k}, p={config.p}, trial={config.trial}")
    print(f"  Sequence: {sum(1 for f in sequence if f == 'heads')}/{config.n_flips} heads")
    print()

    # Run appropriate experiment type
    if config.condition == Condition.SINGLE_TURN:
        result = run_single_turn_experiment(model, tokenizer, config, sequence)
    else:
        result = run_multi_turn_experiment(model, tokenizer, config, sequence)

    return result


def save_result(result: TrajectoryResult, output_dir: Path):
    """
    Save experiment result to JSON.

    Args:
        result: TrajectoryResult to save
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = result.config.get_output_filename()
    output_path = output_dir / filename

    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nSaved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Coin Flip Experiment Runner")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--condition", type=str, required=True,
                        choices=[c.value for c in Condition],
                        help="Experimental condition")
    parser.add_argument("--k", type=int, default=1,
                        help="Polling frequency")
    parser.add_argument("--coin-spec", type=str, default="unspecified",
                        choices=[c.value for c in CoinSpec],
                        help="What to tell model about the coin: unspecified, unknown_bias, or fair")
    parser.add_argument("--p", type=float, nargs="+", default=[0.5],
                        help="True P(heads) value(s)")
    parser.add_argument("--trials", type=str, default="0",
                        help="Trial numbers, comma-separated")
    parser.add_argument("--n-flips", type=int, default=100,
                        help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Flips per conversation turn in multi-turn conditions (default=1)")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent / "experiments"),
                        help="Output directory")
    parser.add_argument("--no-counterbalance", action="store_true",
                        help="Disable counterbalancing")

    args = parser.parse_args()

    # Parse trials
    trials = [int(t.strip()) for t in args.trials.split(",")]

    # Parse condition and coin spec
    condition = Condition(args.condition)
    coin_spec = CoinSpec(args.coin_spec)

    # Setup output directory
    output_dir = Path(args.output_dir)

    # Load model once
    model, tokenizer = setup_model(args.model)

    # Run experiments
    for p in args.p:
        for trial in trials:
            config = ExperimentConfig(
                model_name=args.model,
                condition=condition,
                k=args.k,
                p=p,
                trial=trial,
                n_flips=args.n_flips,
                batch_size=args.batch_size,
                coin_spec=coin_spec,
                counterbalance=not args.no_counterbalance
            )

            result = run_experiment(config, model=model, tokenizer=tokenizer)
            save_result(result, output_dir)

            # Print summary metrics
            if result.metrics:
                print(f"\nMetrics:")
                print(f"  MAE from Bayesian: {result.metrics.mae_from_bayesian:.4f}")
                print(f"  Correlation: {result.metrics.correlation_with_bayesian:.4f}")
                print(f"  Prior P(heads): {result.metrics.prior_p_heads:.4f}")
                print(f"  Mean update: {result.metrics.mean_update_magnitude:.4f}")
                print(f"  Trajectory variance: {result.metrics.trajectory_variance:.4f}")


if __name__ == "__main__":
    main()

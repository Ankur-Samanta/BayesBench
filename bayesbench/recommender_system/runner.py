"""
Main experiment runner for recommender-system cold-start experiments.

Supports single-turn and multi-turn experimental conditions with CLI interface.
Outer loop iterates over (true_type, target_movie, trial) combinations.
Uses 1-5 star ratings with configurable K user types.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from .config import (
    Condition, PopInfo, SequenceSource, ExperimentConfig,
    PollResult, TrajectoryResult,
)
from .conditions import MessageBuilder, ConversationState
from .metrics import (
    compute_trajectory_metrics, mixture_posterior, expected_rating_bayesian,
)
from .extraction import (
    setup_model, extract_rating_probs,
    extract_type_counterbalanced, generate_type_cot,
    # Batched variants used by run_*_batched runners.
    extract_mcq5_probs_batch,
    extract_rating_counterbalanced_cyclic_batch,
    extract_type_counterbalanced_batch, generate_type_cot_batch,
)
from .data import (
    prepare_data, generate_synthetic_sequence, TypeModel, MovieSelection,
    build_anonymization_map, anonymize_sequence,
    find_real_users, get_real_user_sequence,
    generate_misleading_genre_sequence,
)


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


def sample_prediction(
    model, tokenizer, messages: List[dict],
    rng: np.random.RandomState,
) -> int:
    """Sample a star rating (1-5) from the model's predicted distribution.

    Uses the supplied per-experiment RandomState to ensure that batched
    execution produces bit-identical results to sequential execution.
    """
    dist, e_rating, _ = extract_rating_probs(model, tokenizer, messages)
    sampled_idx = int(rng.choice(5, p=dist))
    return sampled_idx + 1  # 0-indexed → 1-5 stars



def _extract_rating_batched(
    model,
    tokenizer,
    build_msgs,
    counterbalance: bool = True,
):
    """Build rating-probe messages and extract the rating distribution (batched).

    When ``counterbalance`` is True, uses K=5 cyclic counterbalancing
    (``extract_rating_counterbalanced_cyclic_batch``): every rating value
    occupies every letter position exactly once, eliminating per-letter bias
    including center-anchoring at C. When False, a single standard-scale shift
    is read with no counterbalancing (debug only).

    Args:
        build_msgs: Callable taking a single dict of extra builder kwargs
            (``{"reversed_scale": bool, "rating_shift": int}``) and returning
            the list of message-lists for ALL experiments in this batch.

    Returns: list of result tuples shape-compatible with
    ``extract_rating_counterbalanced_cyclic_batch``.
    """
    if counterbalance:
        msgs_per_shift = [
            build_msgs({"reversed_scale": False, "rating_shift": k})
            for k in range(5)
        ]
        n = len(msgs_per_shift[0])
        msgs_per_exp = [[msgs_per_shift[k][i] for k in range(5)] for i in range(n)]
        return extract_rating_counterbalanced_cyclic_batch(
            model, tokenizer, msgs_per_exp,
        )

    # No-CB path: standard scale only, manual assembly to match the tuple shape.
    v1 = build_msgs({"reversed_scale": False})
    v1_letter_results = extract_mcq5_probs_batch(model, tokenizer, v1)
    results = []
    for letter_probs, letter_mass in v1_letter_results:
        if letter_mass < 0.01:
            results.append((3.0, [0.2] * 5, None, None, 0.0, letter_mass))
        else:
            dist = [p / letter_mass for p in letter_probs]
            e_rating = sum((r + 1) * dist[r] for r in range(5))
            results.append((e_rating, dist, None, None, e_rating - 3.0, letter_mass))
    return results


def _rating_cyclic_serial(model, tokenizer, build_shift):
    """Cyclic-counterbalanced rating for a single (serial) probe.

    ``build_shift(k)`` returns the message list for cyclic shift k (k in
    0..4). Returns the same 6-tuple as
    ``extract_rating_counterbalanced_cyclic_batch`` for one experiment.
    """
    shift_msgs = [build_shift(k) for k in range(5)]
    return extract_rating_counterbalanced_cyclic_batch(
        model, tokenizer, [shift_msgs]
    )[0]


def _build_prompt_kwargs(
    config: ExperimentConfig,
    type_model: TypeModel,
    movie_selection: MovieSelection,
    target_movie: Dict,
    anon_map: Dict = None,
) -> Dict:
    """Build common kwargs for MessageBuilder methods."""
    if config.pop_info == PopInfo.ANONYMIZED and anon_map:
        # Use anonymized names for the target movie
        movie_name_map = anon_map["movie_name_map"]
        genre_map = anon_map["genre_map"]
        anon_target_name = movie_name_map.get(
            target_movie["movie_id"], f"Item_{target_movie['movie_id']}"
        )
        anon_target_genres = [genre_map.get(g, g) for g in target_movie["genres"]]
        kwargs = {
            "target_movie_name": anon_target_name,
            "target_movie_genres": anon_target_genres,
            "pop_info": config.pop_info,
            "type_model": type_model,
            "probe_movie_ids": movie_selection.probe_movie_ids,
            "target_movie_id": config.target_movie_id,
            "anon_map": anon_map,
        }
    else:
        kwargs = {
            "target_movie_name": target_movie["movie_name"],
            "target_movie_genres": target_movie["genres"],
            "pop_info": config.pop_info,
        }

        if config.pop_info == PopInfo.EXPLICIT_TYPES:
            kwargs["type_model"] = type_model
            kwargs["probe_movie_ids"] = movie_selection.probe_movie_ids
            kwargs["target_movie_id"] = config.target_movie_id

    return kwargs


def _build_type_prompt_kwargs(
    config: ExperimentConfig,
    type_model: TypeModel,
    movie_selection: MovieSelection,
    anon_map: Dict = None,
) -> Dict:
    """Build common kwargs for type elicitation MessageBuilder methods."""
    kwargs = {
        "pop_info": config.pop_info,
        "type_model": type_model,
        "probe_movie_ids": movie_selection.probe_movie_ids,
        "target_movie_id": config.target_movie_id,
    }
    if config.pop_info == PopInfo.ANONYMIZED and anon_map:
        kwargs["anon_map"] = anon_map
    return kwargs


def _extract_type_elicitation(
    model,
    tokenizer,
    config: ExperimentConfig,
    prompt_history: List[Dict],
    type_prompt_kwargs: Dict,
    target_movie: Dict,
    anon_map: Dict = None,
    poll: "PollResult" = None,
) -> None:
    """
    Extract type elicitation (type logprobs, CoT, conditioned rating) and
    populate the poll result in-place. Skipped for zero_shot.
    """
    if config.pop_info == PopInfo.ZERO_SHOT:
        return

    # --- Type logprobs (cyclic counterbalanced) ---
    type_msgs_list = [
        MessageBuilder.build_type_poll_single_turn(
            history=prompt_history, shift=s, **type_prompt_kwargs,
        )
        for s in range(config.n_types)
    ]
    type_dist, predicted_type, type_bias, type_mass = \
        extract_type_counterbalanced(model, tokenizer, type_msgs_list)

    poll.llm_type_distribution = type_dist
    poll.llm_type_prediction = predicted_type
    poll.type_scale_bias = type_bias
    poll.type_mass = type_mass

    # --- Type CoT ---
    cot_msgs = MessageBuilder.build_type_cot_single_turn(
        history=prompt_history, **type_prompt_kwargs,
    )
    _, cot_text = generate_type_cot(
        model, tokenizer, cot_msgs, n_types=config.n_types,
    )
    poll.cot_reasoning = cot_text

    # --- CoT follow-up MCQ (cyclic counterbalanced) ---
    followup_list = [
        MessageBuilder.build_type_cot_followup_single_turn(
            history=prompt_history, cot_reasoning=cot_text,
            shift=s, **type_prompt_kwargs,
        )
        for s in range(config.n_types)
    ]
    cot_dist, cot_predicted, cot_bias, cot_mass = \
        extract_type_counterbalanced(model, tokenizer, followup_list)

    poll.cot_type_distribution = cot_dist
    poll.cot_type_prediction = cot_predicted
    poll.cot_type_scale_bias = cot_bias
    poll.cot_type_mass = cot_mass

    # --- Conditioned rating (counterbalanced, using CoT-predicted type) ---
    cond_type = poll.cot_type_prediction if poll.cot_type_prediction is not None else predicted_type

    # Build conditioned prompt kwargs
    if config.pop_info == PopInfo.ANONYMIZED and anon_map:
        movie_name_map = anon_map["movie_name_map"]
        genre_map = anon_map["genre_map"]
        cond_target_name = movie_name_map.get(
            target_movie["movie_id"], f"Item_{target_movie['movie_id']}"
        )
        cond_target_genres = [genre_map.get(g, g) for g in target_movie["genres"]]
    else:
        cond_target_name = target_movie["movie_name"]
        cond_target_genres = target_movie["genres"]

    cond_msgs_v1 = MessageBuilder.build_conditioned_single_turn(
        history=prompt_history,
        target_movie_name=cond_target_name,
        target_movie_genres=cond_target_genres,
        predicted_type=cond_type,
        reversed_scale=False,
        cot_reasoning=cot_text,
        **type_prompt_kwargs,
    )
    if config.counterbalance:
        cond_e, cond_dist, _, _, cond_bias, cond_mass = _rating_cyclic_serial(
            model, tokenizer,
            lambda k: MessageBuilder.build_conditioned_single_turn(
                history=prompt_history,
                target_movie_name=cond_target_name,
                target_movie_genres=cond_target_genres,
                predicted_type=cond_type,
                reversed_scale=False,
                rating_shift=k,
                cot_reasoning=cot_text,
                **type_prompt_kwargs,
            ),
        )
    else:
        cond_dist, cond_e, cond_mass = \
            extract_rating_probs(model, tokenizer, cond_msgs_v1)
        cond_bias = cond_e - 3.0

    poll.conditioned_expected_rating = cond_e
    poll.conditioned_rating_distribution = cond_dist
    poll.conditioned_scale_bias = cond_bias
    poll.conditioned_rating_mass = cond_mass


def _extract_type_elicitation_multi_turn(
    model,
    tokenizer,
    config: ExperimentConfig,
    state: ConversationState,
    type_prompt_kwargs: Dict,
    target_movie: Dict,
    anon_map: Dict = None,
    poll: "PollResult" = None,
) -> None:
    """
    Extract type elicitation from multi-turn type-based conversation state.

    Uses the conversation state (not flat history) for all polls:
    - V1-type: build_type_poll_multi_turn (counterbalanced)
    - CoT: build_type_cot_multi_turn
    - V2-conditioned: build_conditioned_rating_from_type_state (counterbalanced)

    Populates the poll result in-place.
    """
    # --- Type logprobs (cyclic counterbalanced) ---
    type_msgs_list = [
        MessageBuilder.build_type_poll_multi_turn(
            state, shift=s, **type_prompt_kwargs,
        )
        for s in range(config.n_types)
    ]
    type_dist, predicted_type, type_bias, type_mass = \
        extract_type_counterbalanced(model, tokenizer, type_msgs_list)

    poll.llm_type_distribution = type_dist
    poll.llm_type_prediction = predicted_type
    poll.type_scale_bias = type_bias
    poll.type_mass = type_mass

    # --- Type CoT ---
    cot_msgs = MessageBuilder.build_type_cot_multi_turn(
        state, **type_prompt_kwargs,
    )
    _, cot_text = generate_type_cot(
        model, tokenizer, cot_msgs, n_types=config.n_types,
    )
    poll.cot_reasoning = cot_text

    # --- CoT follow-up MCQ (cyclic counterbalanced) ---
    followup_list = [
        MessageBuilder.build_type_cot_followup_multi_turn(
            state, cot_reasoning=cot_text,
            shift=s, **type_prompt_kwargs,
        )
        for s in range(config.n_types)
    ]
    cot_dist, cot_predicted, cot_bias, cot_mass = \
        extract_type_counterbalanced(model, tokenizer, followup_list)

    poll.cot_type_distribution = cot_dist
    poll.cot_type_prediction = cot_predicted
    poll.cot_type_scale_bias = cot_bias
    poll.cot_type_mass = cot_mass

    # --- Conditioned rating (counterbalanced, using CoT-predicted type) ---
    cond_type = poll.cot_type_prediction if poll.cot_type_prediction is not None else predicted_type

    # Resolve target display names
    if config.pop_info == PopInfo.ANONYMIZED and anon_map:
        movie_name_map = anon_map["movie_name_map"]
        genre_map = anon_map["genre_map"]
        cond_target_name = movie_name_map.get(
            target_movie["movie_id"], f"Item_{target_movie['movie_id']}"
        )
        cond_target_genres = [genre_map.get(g, g) for g in target_movie["genres"]]
    else:
        cond_target_name = target_movie["movie_name"]
        cond_target_genres = target_movie["genres"]

    cond_msgs_v1 = MessageBuilder.build_conditioned_rating_from_type_state(
        state,
        predicted_type=cond_type,
        target_movie_name=cond_target_name,
        target_movie_genres=cond_target_genres,
        reversed_scale=False,
        cot_reasoning=cot_text,
        **type_prompt_kwargs,
    )
    if config.counterbalance:
        cond_e, cond_dist, _, _, cond_bias, cond_mass = _rating_cyclic_serial(
            model, tokenizer,
            lambda k: MessageBuilder.build_conditioned_rating_from_type_state(
                state,
                predicted_type=cond_type,
                target_movie_name=cond_target_name,
                target_movie_genres=cond_target_genres,
                reversed_scale=False,
                rating_shift=k,
                cot_reasoning=cot_text,
                **type_prompt_kwargs,
            ),
        )
    else:
        cond_dist, cond_e, cond_mass = \
            extract_rating_probs(model, tokenizer, cond_msgs_v1)
        cond_bias = cond_e - 3.0

    poll.conditioned_expected_rating = cond_e
    poll.conditioned_rating_distribution = cond_dist
    poll.conditioned_scale_bias = cond_bias
    poll.conditioned_rating_mass = cond_mass


def run_single_turn_experiment(
    model,
    tokenizer,
    config: ExperimentConfig,
    sequence: List[Dict],
    type_model: TypeModel,
    movie_selection: MovieSelection,
    anon_map: Optional[Dict] = None,
) -> TrajectoryResult:
    """
    Run single-turn experiment.

    Each poll is independent -- full history in a fresh prompt.

    Args:
        anon_map: Optional pre-built anonymization map (job-level cache).
            If None and the condition needs one, it will be constructed
            on demand from the same job-level inputs.
    """
    target_movie = next(
        tm for tm in movie_selection.target_movies
        if tm["movie_id"] == config.target_movie_id
    )

    # Build anonymization map if needed (job-level cache passed in, or build on demand)
    display_sequence = sequence
    if config.pop_info == PopInfo.ANONYMIZED:
        if anon_map is None:
            anon_map = build_anonymization_map(
                movie_selection.probe_movie_ids,
                movie_selection.target_movies,
                type_model,
            )
        display_sequence = anonymize_sequence(sequence, anon_map)
    else:
        anon_map = None

    prompt_kwargs = _build_prompt_kwargs(config, type_model, movie_selection, target_movie, anon_map)
    type_prompt_kwargs = _build_type_prompt_kwargs(config, type_model, movie_selection, anon_map)

    poll_points = get_poll_points(config.n_ratings, config.k)
    polls = []

    for t in poll_points:
        history = sequence[:t]
        prompt_history = display_sequence[:t]  # anonymized if needed
        rating_counts = [0] * 5
        for r in history:
            rating_counts[r["rating"] - 1] += 1

        # Bayesian ground truth (always uses real movie IDs)
        observations = [{"movie_id": r["movie_id"], "rating": r["rating"]} for r in history]
        bayes = expected_rating_bayesian(observations, config.target_movie_id, type_model)
        type_post = mixture_posterior(observations, type_model).tolist()

        # Standard-scale prompt (used by the no-counterbalance path).
        msgs_v1 = MessageBuilder.build_single_turn(
            history=prompt_history, reversed_scale=False, **prompt_kwargs,
        )

        if config.counterbalance:
            expected_rating, rating_dist, dist_v1, dist_v2, scale_bias, rating_mass = \
                _rating_cyclic_serial(
                    model, tokenizer,
                    lambda k: MessageBuilder.build_single_turn(
                        history=prompt_history, reversed_scale=False,
                        rating_shift=k, **prompt_kwargs,
                    ),
                )
        else:
            rating_dist, expected_rating, rating_mass = \
                extract_rating_probs(model, tokenizer, msgs_v1)
            scale_bias = expected_rating - 3.0

        poll = PollResult(
            t=t,
            rating_counts=rating_counts,
            expected_rating=expected_rating,
            rating_distribution=rating_dist,
            scale_bias=scale_bias,
            rating_mass=rating_mass,
            bayesian_posterior=bayes,
            type_posterior=type_post,
        )

        # Type elicitation (skipped for zero_shot)
        _extract_type_elicitation(
            model, tokenizer, config, prompt_history,
            type_prompt_kwargs, target_movie, anon_map, poll,
        )

        polls.append(poll)

        type_info = ""
        if poll.llm_type_prediction is not None:
            type_info = f" type={poll.llm_type_prediction} cot={poll.cot_type_prediction}"
        print(f"  t={t:3d}: E[r]={expected_rating:.3f} "
              f"bayes={bayes:.3f} scale_bias={scale_bias:+.3f} mass={rating_mass:.3f}{type_info}")

    result = TrajectoryResult(
        config=config,
        rating_sequence=sequence,
        poll_points=poll_points,
        polls=polls,
    )
    result.metrics = compute_trajectory_metrics(result, type_model)
    return result


# =============================================================================
# Batched Single-Turn Runner
# =============================================================================
#
# A batched runner processes multiple experiments in lockstep through their
# poll points, dispatching all model calls in batches that span every
# experiment in the job. The output for each experiment must be byte-identical
# to what ``run_single_turn_experiment`` would have produced when called on
# that experiment in isolation. The batching is purely about how prompts are
# dispatched to vLLM — every prompt building, parsing, and assembly step
# uses the exact same helpers as the serial path, so equivalence holds by
# construction (modulo deterministic floating-point reordering, which the
# wrappers and BayesBench's batched API both preserve).
#
# Phase structure for one poll point ``t`` across N homogeneous experiments
# (single_turn condition, all sharing pop_info / n_types / counterbalance):
#
#   For non-zero-shot pop_info:
#     Phase A:  N * (2 + n_types) prior calls
#               (2 counterbalanced rating prompts + n_types type-MCQ shifts)
#     Phase B:  N generate calls (CoT)
#     Phase C:  N * n_types prior calls
#               (CoT-followup MCQ shifts; depends on Phase B)
#     Phase D:  N * 2 prior calls
#               (counterbalanced conditioned rating; depends on Phase B and C)
#
#   For zero-shot pop_info:
#     Phase A:  N * 2 prior calls (rating CB only — no type elicitation)
#
# After every poll point we stream a partial-trajectory write per experiment
# so a mid-job crash preserves the polls completed so far. After the final
# poll we additionally compute trajectory metrics and rewrite each result.
# =============================================================================


def _validate_homogeneous_single_turn(configs: List[ExperimentConfig]) -> None:
    """Verify that all configs in a job share the fields the batched runner
    assumes are constant. Bails out with a clear error if not."""
    if not configs:
        raise ValueError("run_single_turn_batched: empty configs list")
    base = configs[0]
    if base.condition != Condition.SINGLE_TURN:
        raise ValueError(
            f"run_single_turn_batched: expected SINGLE_TURN, got {base.condition}"
        )
    must_match = ("condition", "pop_info", "n_types", "n_ratings", "k", "counterbalance")
    for i, c in enumerate(configs[1:], start=1):
        for field in must_match:
            if getattr(c, field) != getattr(base, field):
                raise ValueError(
                    f"run_single_turn_batched: config[{i}] differs from config[0] "
                    f"in field '{field}': {getattr(c, field)!r} vs {getattr(base, field)!r}"
                )


def _resolve_target_display(
    target_movie: Dict, pop_info: PopInfo, anon_map: Optional[Dict],
) -> Tuple[str, List[str]]:
    """Return the (display_name, display_genres) for the target under the
    current pop_info, applying anonymization when applicable. Mirrors the
    inline logic in ``_extract_type_elicitation``."""
    if pop_info == PopInfo.ANONYMIZED and anon_map:
        movie_name_map = anon_map["movie_name_map"]
        genre_map = anon_map["genre_map"]
        cond_target_name = movie_name_map.get(
            target_movie["movie_id"], f"Item_{target_movie['movie_id']}"
        )
        cond_target_genres = [genre_map.get(g, g) for g in target_movie["genres"]]
    else:
        cond_target_name = target_movie["movie_name"]
        cond_target_genres = target_movie["genres"]
    return cond_target_name, cond_target_genres


def run_single_turn_batched(
    model,
    tokenizer,
    configs: List[ExperimentConfig],
    sequences: List[List[Dict]],
    type_model: TypeModel,
    movie_selection: MovieSelection,
    output_dir: Optional[Path] = None,
    anon_map: Optional[Dict] = None,
    resume_polls: Optional[List[List["PollResult"]]] = None,
    skip_cot: bool = False,
    marginalize_conditioned_rating: bool = False,
) -> List[TrajectoryResult]:
    """
    Run a list of single-turn experiments in lockstep, batching all model
    calls across experiments at each poll point.

    Args:
        configs: list of homogeneous ``ExperimentConfig`` (same condition,
            pop_info, n_types, n_ratings, k, counterbalance — only true_type,
            target_movie_id, trial may differ).
        sequences: per-experiment rating sequences, in the same order as
            ``configs``. Each sequence must have at least ``config.n_ratings``
            entries.
        output_dir: if provided, partial trajectories are streamed to this
            directory after every poll point so a crash mid-job preserves
            the polls completed so far. Final results (with metrics) overwrite
            the partials at the end.
        anon_map: optional pre-built job-level anonymization map (only used
            for ANONYMIZED pop_info).

    Returns:
        List of TrajectoryResult, in the same order as ``configs``. Each
        result is byte-equivalent to what ``run_single_turn_experiment`` would
        have produced when called on that experiment in isolation.
    """
    _validate_homogeneous_single_turn(configs)
    if len(sequences) != len(configs):
        raise ValueError(
            f"sequences (len={len(sequences)}) must match configs (len={len(configs)})"
        )

    n_exps = len(configs)
    base = configs[0]
    use_type_elicitation = base.pop_info != PopInfo.ZERO_SHOT
    counterbalance = base.counterbalance
    n_types = base.n_types
    poll_points = get_poll_points(base.n_ratings, base.k)

    # --- Per-experiment static setup -----------------------------------------
    # We resolve the target movie, build the optional display sequence, and
    # cache the kwargs dicts that the MessageBuilder methods consume. This
    # mirrors the per-experiment setup that ``run_single_turn_experiment``
    # does once at the top of its function.
    targets: List[Dict] = []
    display_sequences: List[List[Dict]] = []
    prompt_kwargs_per_exp: List[Dict] = []
    type_prompt_kwargs_per_exp: List[Dict] = []
    per_exp_anon_map: List[Optional[Dict]] = []

    for i, cfg in enumerate(configs):
        target_movie = next(
            tm for tm in movie_selection.target_movies
            if tm["movie_id"] == cfg.target_movie_id
        )
        targets.append(target_movie)

        exp_anon_map = None
        seq = sequences[i]
        if cfg.pop_info == PopInfo.ANONYMIZED:
            if anon_map is not None:
                exp_anon_map = anon_map
            else:
                exp_anon_map = build_anonymization_map(
                    movie_selection.probe_movie_ids,
                    movie_selection.target_movies,
                    type_model,
                )
            seq = anonymize_sequence(sequences[i], exp_anon_map)
        per_exp_anon_map.append(exp_anon_map)
        display_sequences.append(seq)

        prompt_kwargs_per_exp.append(
            _build_prompt_kwargs(cfg, type_model, movie_selection, target_movie, exp_anon_map)
        )
        type_prompt_kwargs_per_exp.append(
            _build_type_prompt_kwargs(cfg, type_model, movie_selection, exp_anon_map)
        )

    # --- Per-experiment poll accumulator -------------------------------------
    if resume_polls is not None:
        polls_per_exp = [list(p) for p in resume_polls]
        resume_done = min(len(p) for p in polls_per_exp)
    else:
        polls_per_exp = [[] for _ in range(n_exps)]
        resume_done = 0

    print(
        f"\nrun_single_turn_batched: {n_exps} experiments × "
        f"{len(poll_points)} poll points (pop_info={base.pop_info.value}, "
        f"counterbalance={counterbalance}, type_elicitation={use_type_elicitation})"
        + (f", resuming from poll {resume_done}/{len(poll_points)}" if resume_done else "")
    )

    # =========================================================================
    # Lockstep loop over poll points
    # =========================================================================
    for poll_idx, poll_t in enumerate(poll_points):
        # Skip polls already completed from a resumed partial run
        if poll_idx < resume_done:
            continue
        # ----- Per-experiment Bayesian ground truth (CPU only) ---------------
        bayesian_per_exp: List[Tuple[float, List[float], List[int]]] = []
        for i in range(n_exps):
            history = sequences[i][:poll_t]
            obs = [{"movie_id": r["movie_id"], "rating": r["rating"]} for r in history]
            bayes = expected_rating_bayesian(obs, configs[i].target_movie_id, type_model)
            type_post = mixture_posterior(obs, type_model).tolist()
            rating_counts = [0] * 5
            for r in history:
                rating_counts[r["rating"] - 1] += 1
            bayesian_per_exp.append((bayes, type_post, rating_counts))

        # ----- Phase A: independent rating CB + (non-zero-shot) type MCQ -----
        prompt_histories = [display_sequences[i][:poll_t] for i in range(n_exps)]

        def _rating_msgs(extra_kwargs):
            return [
                MessageBuilder.build_single_turn(
                    history=prompt_histories[i],
                    **extra_kwargs,
                    **prompt_kwargs_per_exp[i],
                )
                for i in range(n_exps)
            ]

        rating_results = _extract_rating_batched(
            model, tokenizer, _rating_msgs,
            counterbalance=counterbalance,
        )

        # Type MCQ (cyclic counterbalanced) — only for non-zero-shot.
        if use_type_elicitation:
            type_msgs_per_exp: List[List[List[Dict]]] = []
            for i in range(n_exps):
                shifts = [
                    MessageBuilder.build_type_poll_single_turn(
                        history=prompt_histories[i],
                        shift=s,
                        **type_prompt_kwargs_per_exp[i],
                    )
                    for s in range(n_types)
                ]
                type_msgs_per_exp.append(shifts)
            type_results = extract_type_counterbalanced_batch(
                model, tokenizer, type_msgs_per_exp, n_types=n_types,
            )
        else:
            type_results = [None] * n_exps  # type: ignore[list-item]

        # ----- Phase B: CoT generation (non-zero-shot only) ------------------
        if use_type_elicitation and not skip_cot:
            cot_msgs_per_exp = [
                MessageBuilder.build_type_cot_single_turn(
                    history=prompt_histories[i],
                    **type_prompt_kwargs_per_exp[i],
                )
                for i in range(n_exps)
            ]
            cot_results = generate_type_cot_batch(
                model, tokenizer, cot_msgs_per_exp, n_types=n_types,
            )
        else:
            cot_results = [(None, None)] * n_exps  # type: ignore[list-item]

        # ----- Phase C: CoT followup MCQ (depends on Phase B) ----------------
        if use_type_elicitation and not skip_cot:
            followup_msgs_per_exp: List[List[List[Dict]]] = []
            for i in range(n_exps):
                cot_text = cot_results[i][1]
                shifts = [
                    MessageBuilder.build_type_cot_followup_single_turn(
                        history=prompt_histories[i],
                        cot_reasoning=cot_text,
                        shift=s,
                        **type_prompt_kwargs_per_exp[i],
                    )
                    for s in range(n_types)
                ]
                followup_msgs_per_exp.append(shifts)
            cot_followup_results = extract_type_counterbalanced_batch(
                model, tokenizer, followup_msgs_per_exp, n_types=n_types,
            )
        elif use_type_elicitation:
            # skip_cot path: 4-tuple of Nones so the assembly's unpack still works.
            cot_followup_results = [(None, None, None, None)] * n_exps  # type: ignore[list-item]
        else:
            cot_followup_results = [None] * n_exps  # type: ignore[list-item]

        # ----- Phase D: conditioned rating (depends on Phase B and C) --------
        if use_type_elicitation:
            cond_predicted_per_exp: List[int] = []
            cond_targets: List[Tuple[str, list]] = []
            for i in range(n_exps):
                # cot_followup_results[i] = (type_dist, predicted_type, scale_bias, mass)
                cot_predicted = cot_followup_results[i][1]
                # Mirror the serial path's fallback: if cot_predicted is None
                # (which never happens in current code but is preserved as a
                # safety net), fall back to the implicit type prediction.
                if cot_predicted is None:
                    cot_predicted = type_results[i][1]
                cond_predicted_per_exp.append(cot_predicted)
                cond_target_name, cond_target_genres = _resolve_target_display(
                    targets[i], configs[i].pop_info, per_exp_anon_map[i],
                )
                cond_targets.append((cond_target_name, cond_target_genres))

            def _cond_msgs(extra_kwargs):
                return [
                    MessageBuilder.build_conditioned_single_turn(
                        history=prompt_histories[i],
                        target_movie_name=cond_targets[i][0],
                        target_movie_genres=cond_targets[i][1],
                        predicted_type=cond_predicted_per_exp[i],
                        cot_reasoning=cot_results[i][1],
                        **extra_kwargs,
                        **type_prompt_kwargs_per_exp[i],
                    )
                    for i in range(n_exps)
                ]

            cond_results = _extract_rating_batched(
                model, tokenizer, _cond_msgs,
                counterbalance=counterbalance,
            )

            # ----- Phase E: marginalized conditioned rating ---------------
            if marginalize_conditioned_rating:
                cond_per_type_st: List[List] = []
                for k in range(n_types):
                    def _cond_msgs_k(extra_kwargs, _k=k):
                        return [
                            MessageBuilder.build_conditioned_single_turn(
                                history=prompt_histories[i],
                                target_movie_name=cond_targets[i][0],
                                target_movie_genres=cond_targets[i][1],
                                predicted_type=_k,
                                cot_reasoning=cot_results[i][1],
                                **extra_kwargs,
                                **type_prompt_kwargs_per_exp[i],
                            )
                            for i in range(n_exps)
                        ]
                    cond_per_type_st.append(_extract_rating_batched(
                        model, tokenizer, _cond_msgs_k,
                        counterbalance=counterbalance,
                    ))

                marg_results: List = []
                for i in range(n_exps):
                    if (not skip_cot
                            and cot_followup_results[i] is not None
                            and cot_followup_results[i][0] is not None):
                        weights = list(cot_followup_results[i][0])
                    else:
                        weights = list(type_results[i][0])
                    weight_sum = sum(weights)
                    if weight_sum > 0:
                        weights = [w / weight_sum for w in weights]
                    else:
                        weights = [1.0 / n_types] * n_types
                    per_type_dists = [cond_per_type_st[k][i][1] for k in range(n_types)]
                    marg_dist = [
                        sum(weights[k] * per_type_dists[k][r] for k in range(n_types))
                        for r in range(5)
                    ]
                    marg_e = sum((r + 1) * marg_dist[r] for r in range(5))
                    marg_results.append((marg_e, marg_dist, per_type_dists, weights))
            else:
                marg_results = [None] * n_exps  # type: ignore[list-item]
        else:
            cond_results = [None] * n_exps  # type: ignore[list-item]
            marg_results = [None] * n_exps  # type: ignore[list-item]

        # ----- Assemble PollResult per experiment ----------------------------
        for i in range(n_exps):
            bayes, type_post, rating_counts = bayesian_per_exp[i]
            (e_rating, rating_dist, _dist_v1, _dist_v2, scale_bias, rating_mass) = rating_results[i]

            poll = PollResult(
                t=poll_t,
                rating_counts=rating_counts,
                expected_rating=e_rating,
                rating_distribution=rating_dist,
                scale_bias=scale_bias,
                rating_mass=rating_mass,
                bayesian_posterior=bayes,
                type_posterior=type_post,
            )

            if use_type_elicitation:
                t_dist, t_pred, t_bias, t_mass = type_results[i]
                poll.llm_type_distribution = t_dist
                poll.llm_type_prediction = t_pred
                poll.type_scale_bias = t_bias
                poll.type_mass = t_mass

                _, cot_text = cot_results[i]
                poll.cot_reasoning = cot_text

                cd_dist, cd_pred, cd_bias, cd_mass = cot_followup_results[i]
                poll.cot_type_distribution = cd_dist
                poll.cot_type_prediction = cd_pred
                poll.cot_type_scale_bias = cd_bias
                poll.cot_type_mass = cd_mass

                (cond_e, cond_dist, _, _, cond_bias, cond_mass) = cond_results[i]
                poll.conditioned_expected_rating = cond_e
                poll.conditioned_rating_distribution = cond_dist
                poll.conditioned_scale_bias = cond_bias
                poll.conditioned_rating_mass = cond_mass

                if marg_results[i] is not None:
                    marg_e, marg_dist, per_type_dists, weights = marg_results[i]
                    poll.marginalized_conditioned_expected_rating = marg_e
                    poll.marginalized_conditioned_rating_distribution = marg_dist
                    poll.marginalized_conditioned_rating_per_type = per_type_dists
                    poll.marginalized_conditioned_type_weights = weights

            polls_per_exp[i].append(poll)

        # ----- Streaming partial trajectory writes ---------------------------
        # Written to .partial.json so incomplete runs don't block re-runs.
        # The final (complete) write uses the canonical .json name.
        if output_dir is not None:
            for i in range(n_exps):
                partial_result = TrajectoryResult(
                    config=configs[i],
                    rating_sequence=sequences[i],
                    poll_points=poll_points[: len(polls_per_exp[i])],
                    polls=polls_per_exp[i],
                )
                save_result(partial_result, output_dir, partial=True)

        # Lightweight progress log: one line per poll point summarizing
        # average behaviour across the lockstep batch.
        avg_e = sum(p.expected_rating for p in [polls_per_exp[i][-1] for i in range(n_exps)]) / n_exps
        avg_bayes = sum(p.bayesian_posterior for p in [polls_per_exp[i][-1] for i in range(n_exps)]) / n_exps
        print(f"  t={poll_t:3d} (avg over {n_exps}): E[r]={avg_e:.3f} bayes={avg_bayes:.3f}")

    # =========================================================================
    # Finalize: compute metrics and rewrite each result
    # =========================================================================
    final_results: List[TrajectoryResult] = []
    for i in range(n_exps):
        result = TrajectoryResult(
            config=configs[i],
            rating_sequence=sequences[i],
            poll_points=poll_points,
            polls=polls_per_exp[i],
        )
        result.metrics = compute_trajectory_metrics(result, type_model)
        if output_dir is not None:
            save_result(result, output_dir)
        final_results.append(result)

    return final_results


def run_multi_turn_experiment(
    model,
    tokenizer,
    config: ExperimentConfig,
    sequence: List[Dict],
    type_model: TypeModel,
    movie_selection: MovieSelection,
    anon_map: Optional[Dict] = None,
) -> TrajectoryResult:
    """
    Run multi-turn experiment (multi_turn_minimal or multi_turn_actual).

    Maintains conversation state across observations.

    Args:
        anon_map: Optional pre-built anonymization map (job-level cache).
            If None and the condition needs one, it will be constructed
            on demand.
    """
    target_movie = next(
        tm for tm in movie_selection.target_movies
        if tm["movie_id"] == config.target_movie_id
    )

    # Build anonymization map if needed (job-level cache passed in, or build on demand)
    display_sequence = sequence
    if config.pop_info == PopInfo.ANONYMIZED:
        if anon_map is None:
            anon_map = build_anonymization_map(
                movie_selection.probe_movie_ids,
                movie_selection.target_movies,
                type_model,
            )
        display_sequence = anonymize_sequence(sequence, anon_map)
    else:
        anon_map = None

    prompt_kwargs = _build_prompt_kwargs(config, type_model, movie_selection, target_movie, anon_map)
    type_prompt_kwargs = _build_type_prompt_kwargs(config, type_model, movie_selection, anon_map)

    # Resolve target display names for poll prompts
    if config.pop_info == PopInfo.ANONYMIZED and anon_map:
        poll_target_name = anon_map["movie_name_map"].get(
            target_movie["movie_id"], target_movie["movie_name"]
        )
        poll_target_genres = [anon_map["genre_map"].get(g, g) for g in target_movie["genres"]]
    else:
        poll_target_name = target_movie["movie_name"]
        poll_target_genres = target_movie["genres"]

    poll_points = get_poll_points(config.n_ratings, config.k)
    polls = []

    # Per-experiment RNG seeded by config.seed. Used by sample_prediction in
    # multi_turn_actual + zero_shot to keep results reproducible regardless of
    # whether experiments run sequentially or batched.
    rng = np.random.RandomState(config.seed)

    # Non-zero_shot multi-turn: single type-based state, counterbalance at poll time
    # Zero_shot multi-turn: two counterbalanced rating states (unchanged)
    use_type_state = config.pop_info != PopInfo.ZERO_SHOT

    if use_type_state:
        # Single type-based state — no counterbalancing of state itself
        # since "Profile X" / "Noted." insertions are scale-invariant.
        # Counterbalancing happens at poll time by rebuilding the system prompt.
        state = MessageBuilder.init_multi_turn_type_state(
            shift=0, **type_prompt_kwargs,
        )
    else:
        # Zero-shot: two counterbalanced rating states
        state_v1 = MessageBuilder.init_multi_turn_state(reversed_scale=False, **prompt_kwargs)
        state_v2 = MessageBuilder.init_multi_turn_state(reversed_scale=True, **prompt_kwargs)

    current_t = 0
    last_type_prediction = None  # Track most recent MCQ argmax for multi_turn_actual

    for poll_t in poll_points:
        # Process observations from current_t to poll_t
        while current_t < poll_t:
            rating = display_sequence[current_t]  # anonymized if needed

            if config.condition == Condition.MULTI_TURN_MINIMAL:
                if use_type_state:
                    state = MessageBuilder.add_observation_minimal(state, rating)
                else:
                    state_v1 = MessageBuilder.add_observation_minimal(state_v1, rating)
                    state_v2 = MessageBuilder.add_observation_minimal(state_v2, rating)

            elif config.condition == Condition.MULTI_TURN_ACTUAL:
                if use_type_state:
                    # Inject "Profile {X}" using last poll's MCQ argmax (or 0 at start)
                    state = MessageBuilder.add_observation_with_type_prediction(
                        state, rating, last_type_prediction if last_type_prediction is not None else 0,
                    )
                else:
                    # Zero-shot multi_turn_actual: use star ratings as before
                    poll_msgs = MessageBuilder.build_poll_prompt(
                        state_v1, poll_target_name, poll_target_genres
                    )
                    star_rating = sample_prediction(model, tokenizer, poll_msgs, rng)
                    prediction = str(star_rating)

                    state_v1 = MessageBuilder.add_observation_with_prediction(
                        state_v1, rating, prediction,
                    )
                    state_v2 = MessageBuilder.add_observation_with_prediction(
                        state_v2, rating, prediction,
                    )

            current_t += 1

        # Poll at poll_t
        history = sequence[:poll_t]
        rating_counts = [0] * 5
        for r in history:
            rating_counts[r["rating"] - 1] += 1

        # Bayesian ground truth
        observations = [{"movie_id": r["movie_id"], "rating": r["rating"]} for r in history]
        bayes = expected_rating_bayesian(observations, config.target_movie_id, type_model)
        type_post = mixture_posterior(observations, type_model).tolist()

        # --- Rating poll ---
        if use_type_state:
            # Swap type system prompt for rating system prompt, keep conversation history
            msgs_v1 = MessageBuilder.build_rating_poll_from_type_state(
                state, poll_target_name, poll_target_genres,
                reversed_scale=False, **type_prompt_kwargs,
            )
            msgs_v2 = MessageBuilder.build_rating_poll_from_type_state(
                state, poll_target_name, poll_target_genres,
                reversed_scale=True, **type_prompt_kwargs,
            )
        else:
            # Zero-shot: build poll prompts from conversation state
            msgs_v1 = MessageBuilder.build_poll_prompt(
                state_v1, poll_target_name, poll_target_genres
            )
            msgs_v2 = MessageBuilder.build_poll_prompt(
                state_v2, poll_target_name, poll_target_genres
            )

        if config.counterbalance:
            # The zero-shot multi-turn rating is read from two fixed-scale
            # conversation states (msgs_v1/msgs_v2). K=5 cyclic counterbalancing
            # would need five such states per shift; that refactor isn't done
            # and this path isn't used in the paper. Run it with
            # counterbalance=False (single standard-scale read).
            raise NotImplementedError(
                "Cyclic rating counterbalancing is not supported on the "
                "zero-shot multi-turn path (fixed-scale v1/v2 states). Use "
                "--no-counterbalance for this path."
            )
        rating_dist, expected_rating, rating_mass = \
            extract_rating_probs(model, tokenizer, msgs_v1)
        scale_bias = expected_rating - 3.0

        # Get last prediction if applicable
        last_prediction = None
        if use_type_state and config.condition == Condition.MULTI_TURN_ACTUAL and state.predictions:
            last_prediction = state.predictions[-1]
        elif not use_type_state and config.condition == Condition.MULTI_TURN_ACTUAL and state_v1.predictions:
            last_prediction = state_v1.predictions[-1]

        poll = PollResult(
            t=poll_t,
            rating_counts=rating_counts,
            expected_rating=expected_rating,
            rating_distribution=rating_dist,
            scale_bias=scale_bias,
            rating_mass=rating_mass,
            bayesian_posterior=bayes,
            type_posterior=type_post,
            prediction=last_prediction,
        )

        # --- Type elicitation ---
        if use_type_state:
            # Multi-turn type elicitation from conversation state
            _extract_type_elicitation_multi_turn(
                model, tokenizer, config, state,
                type_prompt_kwargs, target_movie, anon_map, poll,
            )
            # Update last_type_prediction for next between-observation injection
            last_type_prediction = poll.llm_type_prediction
        # else: zero_shot — no type elicitation

        polls.append(poll)

        type_info = ""
        if poll.llm_type_prediction is not None:
            type_info = f" type={poll.llm_type_prediction} cot={poll.cot_type_prediction}"
        print(f"  t={poll_t:3d}: E[r]={expected_rating:.3f} "
              f"bayes={bayes:.3f} scale_bias={scale_bias:+.3f} mass={rating_mass:.3f}{type_info}")

    result = TrajectoryResult(
        config=config,
        rating_sequence=sequence,
        poll_points=poll_points,
        polls=polls,
    )
    result.metrics = compute_trajectory_metrics(result, type_model)
    return result


# =============================================================================
# Batched Multi-Turn Runner
# =============================================================================
#
# Like run_single_turn_batched, but processes multi-turn experiments in
# lockstep through their poll points. Conversation states are maintained per
# experiment and advanced in lockstep using the same MessageBuilder helpers
# the serial path uses, so prompt strings are byte-identical between the
# serial and batched paths.
#
# Sub-case dispatch (decided by config.condition + config.pop_info):
#
#   multi_turn_minimal (any pop_info):
#     - Conversation state advances mechanically by appending observations.
#     - All N states advance in lockstep with no inter-experiment dependency.
#
#   multi_turn_actual + non_zero_shot:
#     - Between observations, the most recent type prediction (per experiment)
#       is injected into the conversation as "Profile X".
#     - Each experiment carries its own ``last_type_prediction`` updated after
#       every poll. Lockstep advancement remains valid because no LLM call is
#       needed to advance the state — only the per-experiment integer.
#
#   multi_turn_actual + zero_shot:
#     - Between every observation, ``sample_prediction`` is called from the
#       state to draw a star rating that gets injected into the conversation.
#     - These calls are batched across experiments at each observation step.
#     - Each experiment uses its own ``np.random.RandomState(config.seed)`` so
#       the sampled outputs are reproducible regardless of batching order.
#
# Per-poll-point phase structure for non-zero-shot:
#   Phase A: rating poll CB (2 prompts/exp) + type MCQ shifts (n_types/exp)
#   Phase B: CoT generate (1/exp)
#   Phase C: CoT followup MCQ (n_types/exp)  — depends on Phase B
#   Phase D: conditioned rating CB (2/exp)   — depends on Phase B and C
#
# For zero_shot, only Phase A's rating CB is run (no type elicitation).
# =============================================================================


def _validate_homogeneous_multi_turn(configs: List[ExperimentConfig]) -> None:
    """All configs in a multi-turn batched job must share the fields the
    runner treats as constants. This is the multi-turn analogue of
    ``_validate_homogeneous_single_turn``."""
    if not configs:
        raise ValueError("run_multi_turn_batched: empty configs list")
    base = configs[0]
    if base.condition not in (Condition.MULTI_TURN_MINIMAL, Condition.MULTI_TURN_ACTUAL):
        raise ValueError(
            f"run_multi_turn_batched: expected MULTI_TURN_*, got {base.condition}"
        )
    must_match = ("condition", "pop_info", "n_types", "n_ratings", "k", "counterbalance")
    for i, c in enumerate(configs[1:], start=1):
        for field in must_match:
            if getattr(c, field) != getattr(base, field):
                raise ValueError(
                    f"run_multi_turn_batched: config[{i}] differs from config[0] "
                    f"in field '{field}': {getattr(c, field)!r} vs {getattr(base, field)!r}"
                )
    # multi_turn_actual × zero_shot is invalid only via run_experiment guard;
    # for the batched path, we permit it because the inner observation loop
    # explicitly handles sampled predictions.


def _resolve_poll_target_display(
    target_movie: Dict, pop_info: PopInfo, anon_map: Optional[Dict],
) -> Tuple[str, List[str]]:
    """Resolve target display name+genres for the per-poll rating prompt.

    This mirrors the inline logic in ``run_multi_turn_experiment`` (lines
    around 'poll_target_name' / 'poll_target_genres'). Distinct from
    ``_resolve_target_display`` only in its slightly different fallback for
    missing names — kept separate to make the byte-equivalence intent
    explicit.
    """
    if pop_info == PopInfo.ANONYMIZED and anon_map:
        poll_target_name = anon_map["movie_name_map"].get(
            target_movie["movie_id"], target_movie["movie_name"]
        )
        poll_target_genres = [anon_map["genre_map"].get(g, g) for g in target_movie["genres"]]
    else:
        poll_target_name = target_movie["movie_name"]
        poll_target_genres = target_movie["genres"]
    return poll_target_name, poll_target_genres


def _sample_predictions_batched(
    model,
    tokenizer,
    states_v1: List["ConversationState"],
    poll_target_names: List[str],
    poll_target_genres_list: List[List[str]],
    rngs: List[np.random.RandomState],
) -> List[int]:
    """Batched analogue of N independent ``sample_prediction`` calls.

    Builds N poll prompts (one per experiment) from each ``state_v1``, fires
    a single batched MCQ5 prior call, and samples one star rating per
    experiment from each per-experiment RandomState. Returns a list of
    1..5 star ratings in input order.
    """
    n = len(states_v1)
    poll_msgs_list = [
        MessageBuilder.build_poll_prompt(
            states_v1[i], poll_target_names[i], poll_target_genres_list[i]
        )
        for i in range(n)
    ]
    letter_results = extract_mcq5_probs_batch(model, tokenizer, poll_msgs_list)

    sampled = []
    for i in range(n):
        letter_probs, letter_mass = letter_results[i]
        if letter_mass < 0.01:
            dist = [0.2] * 5
        else:
            dist = [p / letter_mass for p in letter_probs]
        sampled_idx = int(rngs[i].choice(5, p=dist))
        sampled.append(sampled_idx + 1)
    return sampled


def run_multi_turn_batched(
    model,
    tokenizer,
    configs: List[ExperimentConfig],
    sequences: List[List[Dict]],
    type_model: TypeModel,
    movie_selection: MovieSelection,
    output_dir: Optional[Path] = None,
    anon_map: Optional[Dict] = None,
    resume_polls: Optional[List[List["PollResult"]]] = None,
    skip_cot: bool = False,
    marginalize_conditioned_rating: bool = False,
) -> List[TrajectoryResult]:
    """
    Run a list of multi-turn experiments in lockstep, batching all model
    calls across experiments at each poll point.

    Handles all three multi-turn sub-cases:
      - ``multi_turn_minimal`` (any pop_info)
      - ``multi_turn_actual`` + non-zero-shot
      - ``multi_turn_actual`` + zero_shot

    Per-experiment output is byte-equivalent to what
    ``run_multi_turn_experiment`` would produce when called on each config in
    isolation (subject to the same RNG seeding via per-experiment
    ``RandomState(config.seed)`` for the zero_shot sub-case).

    See the module-level comment block above for the per-poll-point phase
    structure.
    """
    _validate_homogeneous_multi_turn(configs)
    if len(sequences) != len(configs):
        raise ValueError(
            f"sequences (len={len(sequences)}) must match configs (len={len(configs)})"
        )

    n_exps = len(configs)
    base = configs[0]
    use_type_state = base.pop_info != PopInfo.ZERO_SHOT
    counterbalance = base.counterbalance
    n_types = base.n_types
    poll_points = get_poll_points(base.n_ratings, base.k)

    # --- Per-experiment static setup -----------------------------------------
    targets: List[Dict] = []
    display_sequences: List[List[Dict]] = []
    prompt_kwargs_per_exp: List[Dict] = []
    type_prompt_kwargs_per_exp: List[Dict] = []
    per_exp_anon_map: List[Optional[Dict]] = []
    poll_target_names: List[str] = []
    poll_target_genres_list: List[List[str]] = []

    for i, cfg in enumerate(configs):
        target_movie = next(
            tm for tm in movie_selection.target_movies
            if tm["movie_id"] == cfg.target_movie_id
        )
        targets.append(target_movie)

        exp_anon_map = None
        seq = sequences[i]
        if cfg.pop_info == PopInfo.ANONYMIZED:
            if anon_map is not None:
                exp_anon_map = anon_map
            else:
                exp_anon_map = build_anonymization_map(
                    movie_selection.probe_movie_ids,
                    movie_selection.target_movies,
                    type_model,
                )
            seq = anonymize_sequence(sequences[i], exp_anon_map)
        per_exp_anon_map.append(exp_anon_map)
        display_sequences.append(seq)

        prompt_kwargs_per_exp.append(
            _build_prompt_kwargs(cfg, type_model, movie_selection, target_movie, exp_anon_map)
        )
        type_prompt_kwargs_per_exp.append(
            _build_type_prompt_kwargs(cfg, type_model, movie_selection, exp_anon_map)
        )

        poll_name, poll_genres = _resolve_poll_target_display(
            target_movie, cfg.pop_info, exp_anon_map
        )
        poll_target_names.append(poll_name)
        poll_target_genres_list.append(poll_genres)

    # --- Per-experiment RNGs (only matters for multi_turn_actual + zero_shot)
    rngs = [np.random.RandomState(cfg.seed) for cfg in configs]

    # --- Per-experiment conversation state(s) --------------------------------
    if use_type_state:
        states = [
            MessageBuilder.init_multi_turn_type_state(
                shift=0, **type_prompt_kwargs_per_exp[i]
            )
            for i in range(n_exps)
        ]
        states_v1 = None
        states_v2 = None
        last_type_predictions: List[Optional[int]] = [None] * n_exps
    else:
        states = None
        states_v1 = [
            MessageBuilder.init_multi_turn_state(
                reversed_scale=False, **prompt_kwargs_per_exp[i]
            )
            for i in range(n_exps)
        ]
        states_v2 = [
            MessageBuilder.init_multi_turn_state(
                reversed_scale=True, **prompt_kwargs_per_exp[i]
            )
            for i in range(n_exps)
        ]
        last_type_predictions = [None] * n_exps  # unused in zero_shot

    if resume_polls is not None:
        polls_per_exp = [list(p) for p in resume_polls]
        resume_done = min(len(p) for p in polls_per_exp)
        # Reconstruct last_type_predictions from resumed polls
        if use_type_state and resume_done > 0:
            for i in range(n_exps):
                last_poll = polls_per_exp[i][resume_done - 1]
                dist = last_poll.llm_type_distribution
                if dist:
                    last_type_predictions[i] = int(max(range(len(dist)), key=lambda x: dist[x]))
    else:
        polls_per_exp = [[] for _ in range(n_exps)]
        resume_done = 0
    current_t = 0  # Lockstep position; same for every experiment.

    print(
        f"\nrun_multi_turn_batched: {n_exps} experiments × "
        f"{len(poll_points)} poll points (condition={base.condition.value}, "
        f"pop_info={base.pop_info.value}, counterbalance={counterbalance}, "
        f"type_elicitation={use_type_state})"
        + (f", resuming from poll {resume_done}/{len(poll_points)}" if resume_done else "")
    )

    for poll_idx, poll_t in enumerate(poll_points):
        # =====================================================================
        # Advance all states from current_t to poll_t (in lockstep)
        # =====================================================================
        while current_t < poll_t:
            obs_t = current_t

            if base.condition == Condition.MULTI_TURN_MINIMAL:
                # Pure mechanical state growth — no LLM call needed.
                for i in range(n_exps):
                    rating = display_sequences[i][obs_t]
                    if use_type_state:
                        states[i] = MessageBuilder.add_observation_minimal(
                            states[i], rating
                        )
                    else:
                        states_v1[i] = MessageBuilder.add_observation_minimal(
                            states_v1[i], rating
                        )
                        states_v2[i] = MessageBuilder.add_observation_minimal(
                            states_v2[i], rating
                        )

            elif base.condition == Condition.MULTI_TURN_ACTUAL:
                if use_type_state:
                    # Inject "Profile {X}" using each experiment's last MCQ
                    # argmax (or 0 at start). No LLM call needed; lockstep
                    # advancement stays valid.
                    for i in range(n_exps):
                        rating = display_sequences[i][obs_t]
                        seed_type = (
                            last_type_predictions[i]
                            if last_type_predictions[i] is not None
                            else 0
                        )
                        states[i] = MessageBuilder.add_observation_with_type_prediction(
                            states[i], rating, seed_type,
                        )
                else:
                    # Zero-shot multi_turn_actual: sample a star rating per
                    # experiment via a batched MCQ5 prior call. The injected
                    # prediction goes into both state_v1 and state_v2.
                    sampled_stars = _sample_predictions_batched(
                        model, tokenizer, states_v1,
                        poll_target_names, poll_target_genres_list, rngs,
                    )
                    for i in range(n_exps):
                        rating = display_sequences[i][obs_t]
                        prediction = str(sampled_stars[i])
                        states_v1[i] = MessageBuilder.add_observation_with_prediction(
                            states_v1[i], rating, prediction,
                        )
                        states_v2[i] = MessageBuilder.add_observation_with_prediction(
                            states_v2[i], rating, prediction,
                        )

            current_t += 1

        # Skip polls already completed from a resumed partial run.
        # State has been advanced above so future polls see correct context.
        if poll_idx < resume_done:
            continue

        # =====================================================================
        # Per-experiment Bayesian ground truth (CPU only)
        # =====================================================================
        bayesian_per_exp: List[Tuple[float, List[float], List[int]]] = []
        for i in range(n_exps):
            history = sequences[i][:poll_t]
            obs = [{"movie_id": r["movie_id"], "rating": r["rating"]} for r in history]
            bayes = expected_rating_bayesian(obs, configs[i].target_movie_id, type_model)
            type_post = mixture_posterior(obs, type_model).tolist()
            rating_counts = [0] * 5
            for r in history:
                rating_counts[r["rating"] - 1] += 1
            bayesian_per_exp.append((bayes, type_post, rating_counts))

        # =====================================================================
        # Phase A: rating poll (CB)
        # =====================================================================
        if use_type_state:
            def _rating_msgs(extra_kwargs):
                return [
                    MessageBuilder.build_rating_poll_from_type_state(
                        states[i], poll_target_names[i], poll_target_genres_list[i],
                        **extra_kwargs, **type_prompt_kwargs_per_exp[i],
                    )
                    for i in range(n_exps)
                ]
        else:
            # Non-type-state path: rating MCQ rendered from rating-conversation
            # state, which already has the system prompt fixed in v1/v2 form.
            # K=5 cyclic counterbalancing (now the default) would require
            # reinitializing N=5 states per shift; not done since this path
            # isn't used in the paper. Run it with --no-counterbalance.
            def _rating_msgs(extra_kwargs):
                if "rating_shift" in extra_kwargs:
                    raise NotImplementedError(
                        "Cyclic rating counterbalancing is not supported on the "
                        "non-type-state multi-turn path (v1/v2 states have fixed "
                        "scale system prompts). Use --no-counterbalance for this path."
                    )
                state_for_v = states_v1 if not extra_kwargs.get("reversed_scale") else states_v2
                return [
                    MessageBuilder.build_poll_prompt(
                        state_for_v[i], poll_target_names[i], poll_target_genres_list[i]
                    )
                    for i in range(n_exps)
                ]

        rating_results = _extract_rating_batched(
            model, tokenizer, _rating_msgs,
            counterbalance=counterbalance,
        )

        # =====================================================================
        # Type elicitation (Phases B/C/D) — only for non-zero-shot
        # =====================================================================
        if use_type_state:
            # Phase A2: type MCQ logprobs (cyclic counterbalanced).
            type_msgs_per_exp: List[List[List[Dict]]] = []
            for i in range(n_exps):
                shifts = [
                    MessageBuilder.build_type_poll_multi_turn(
                        states[i], shift=s, **type_prompt_kwargs_per_exp[i],
                    )
                    for s in range(n_types)
                ]
                type_msgs_per_exp.append(shifts)
            type_results = extract_type_counterbalanced_batch(
                model, tokenizer, type_msgs_per_exp, n_types=n_types,
            )

            # Phase B: CoT generate (skipped under --skip-cot).
            if not skip_cot:
                cot_msgs_per_exp = [
                    MessageBuilder.build_type_cot_multi_turn(
                        states[i], **type_prompt_kwargs_per_exp[i],
                    )
                    for i in range(n_exps)
                ]
                cot_results = generate_type_cot_batch(
                    model, tokenizer, cot_msgs_per_exp, n_types=n_types,
                )
            else:
                cot_results = [(None, None)] * n_exps  # type: ignore[list-item]

            # Phase C: CoT followup MCQ (skipped under --skip-cot).
            if not skip_cot:
                followup_msgs_per_exp: List[List[List[Dict]]] = []
                for i in range(n_exps):
                    cot_text = cot_results[i][1]
                    shifts = [
                        MessageBuilder.build_type_cot_followup_multi_turn(
                            states[i], cot_reasoning=cot_text,
                            shift=s, **type_prompt_kwargs_per_exp[i],
                        )
                        for s in range(n_types)
                    ]
                    followup_msgs_per_exp.append(shifts)
                cot_followup_results = extract_type_counterbalanced_batch(
                    model, tokenizer, followup_msgs_per_exp, n_types=n_types,
                )
            else:
                # 4-tuple of Nones: Phase D's fallback to type_results[i][1]
                # kicks in, and assembly's unpack still works.
                cot_followup_results = [(None, None, None, None)] * n_exps  # type: ignore[list-item]

            # Phase D: conditioned rating CB.
            cond_predicted_per_exp: List[int] = []
            cond_targets: List[Tuple[str, list]] = []
            for i in range(n_exps):
                cot_predicted = cot_followup_results[i][1]
                if cot_predicted is None:
                    cot_predicted = type_results[i][1]
                cond_predicted_per_exp.append(cot_predicted)
                cond_target_name, cond_target_genres = _resolve_target_display(
                    targets[i], configs[i].pop_info, per_exp_anon_map[i],
                )
                cond_targets.append((cond_target_name, cond_target_genres))

            def _cond_msgs(extra_kwargs):
                return [
                    MessageBuilder.build_conditioned_rating_from_type_state(
                        states[i],
                        predicted_type=cond_predicted_per_exp[i],
                        target_movie_name=cond_targets[i][0],
                        target_movie_genres=cond_targets[i][1],
                        cot_reasoning=cot_results[i][1],
                        **extra_kwargs,
                        **type_prompt_kwargs_per_exp[i],
                    )
                    for i in range(n_exps)
                ]

            cond_results = _extract_rating_batched(
                model, tokenizer, _cond_msgs,
                counterbalance=counterbalance,
            )

            # ----- Phase E: marginalized conditioned rating ---------------
            if marginalize_conditioned_rating:
                # For each candidate type k, get the conditioned rating dist
                # from the model. Then weight-average across k using the
                # model's type posterior (CoT-followup when available, else
                # implicit MCQ).
                cond_per_type: List[List] = []  # cond_per_type[k] = list of 6-tuples (one per exp)
                for k in range(n_types):
                    def _cond_msgs_k(extra_kwargs, _k=k):
                        return [
                            MessageBuilder.build_conditioned_rating_from_type_state(
                                states[i],
                                predicted_type=_k,
                                target_movie_name=cond_targets[i][0],
                                target_movie_genres=cond_targets[i][1],
                                cot_reasoning=cot_results[i][1],
                                **extra_kwargs,
                                **type_prompt_kwargs_per_exp[i],
                            )
                            for i in range(n_exps)
                        ]
                    cond_per_type.append(_extract_rating_batched(
                        model, tokenizer, _cond_msgs_k,
                        counterbalance=counterbalance,
                    ))

                marg_results: List[Tuple[float, List[float], List[List[float]], List[float]]] = []
                for i in range(n_exps):
                    # Pick the most informed type posterior as weights.
                    if (not skip_cot
                            and cot_followup_results[i] is not None
                            and cot_followup_results[i][0] is not None):
                        weights = list(cot_followup_results[i][0])
                    else:
                        weights = list(type_results[i][0])
                    weight_sum = sum(weights)
                    if weight_sum > 0:
                        weights = [w / weight_sum for w in weights]
                    else:
                        weights = [1.0 / n_types] * n_types

                    per_type_dists = [
                        cond_per_type[k][i][1]  # rating_distribution at slot 1
                        for k in range(n_types)
                    ]
                    marg_dist = [
                        sum(weights[k] * per_type_dists[k][r] for k in range(n_types))
                        for r in range(5)
                    ]
                    marg_e = sum((r + 1) * marg_dist[r] for r in range(5))
                    marg_results.append((marg_e, marg_dist, per_type_dists, weights))
            else:
                marg_results = [None] * n_exps  # type: ignore[list-item]
        else:
            type_results = [None] * n_exps  # type: ignore[list-item]
            cot_results = [(None, None)] * n_exps  # type: ignore[list-item]
            cot_followup_results = [None] * n_exps  # type: ignore[list-item]
            cond_results = [None] * n_exps  # type: ignore[list-item]
            marg_results = [None] * n_exps  # type: ignore[list-item]

        # =====================================================================
        # Assemble PollResult per experiment
        # =====================================================================
        for i in range(n_exps):
            bayes, type_post, rating_counts = bayesian_per_exp[i]
            (e_rating, rating_dist, _dv1, _dv2, scale_bias, rating_mass) = rating_results[i]

            # Resolve last_prediction (only meaningful for multi_turn_actual).
            last_prediction = None
            if (use_type_state
                    and base.condition == Condition.MULTI_TURN_ACTUAL
                    and states[i].predictions):
                last_prediction = states[i].predictions[-1]
            elif (not use_type_state
                  and base.condition == Condition.MULTI_TURN_ACTUAL
                  and states_v1[i].predictions):
                last_prediction = states_v1[i].predictions[-1]

            poll = PollResult(
                t=poll_t,
                rating_counts=rating_counts,
                expected_rating=e_rating,
                rating_distribution=rating_dist,
                scale_bias=scale_bias,
                rating_mass=rating_mass,
                bayesian_posterior=bayes,
                type_posterior=type_post,
                prediction=last_prediction,
            )

            if use_type_state:
                t_dist, t_pred, t_bias, t_mass = type_results[i]
                poll.llm_type_distribution = t_dist
                poll.llm_type_prediction = t_pred
                poll.type_scale_bias = t_bias
                poll.type_mass = t_mass

                _, cot_text = cot_results[i]
                poll.cot_reasoning = cot_text

                cd_dist, cd_pred, cd_bias, cd_mass = cot_followup_results[i]
                poll.cot_type_distribution = cd_dist
                poll.cot_type_prediction = cd_pred
                poll.cot_type_scale_bias = cd_bias
                poll.cot_type_mass = cd_mass

                (cond_e, cond_dist, _, _, cond_bias, cond_mass) = cond_results[i]
                poll.conditioned_expected_rating = cond_e
                poll.conditioned_rating_distribution = cond_dist
                poll.conditioned_scale_bias = cond_bias
                poll.conditioned_rating_mass = cond_mass

                # Marginalized conditioned rating (only when --marginalize is on).
                if marg_results[i] is not None:
                    marg_e, marg_dist, per_type_dists, weights = marg_results[i]
                    poll.marginalized_conditioned_expected_rating = marg_e
                    poll.marginalized_conditioned_rating_distribution = marg_dist
                    poll.marginalized_conditioned_rating_per_type = per_type_dists
                    poll.marginalized_conditioned_type_weights = weights

                # Update last_type_prediction for the next between-observation
                # injection (only matters for multi_turn_actual non-zero-shot).
                last_type_predictions[i] = poll.llm_type_prediction

            polls_per_exp[i].append(poll)

        # ----- Streaming partial trajectory writes ---------------------------
        if output_dir is not None:
            for i in range(n_exps):
                partial_result = TrajectoryResult(
                    config=configs[i],
                    rating_sequence=sequences[i],
                    poll_points=poll_points[: len(polls_per_exp[i])],
                    polls=polls_per_exp[i],
                )
                save_result(partial_result, output_dir, partial=True)

        # Lightweight progress log.
        avg_e = sum(p.expected_rating for p in [polls_per_exp[i][-1] for i in range(n_exps)]) / n_exps
        avg_bayes = sum(p.bayesian_posterior for p in [polls_per_exp[i][-1] for i in range(n_exps)]) / n_exps
        print(f"  t={poll_t:3d} (avg over {n_exps}): E[r]={avg_e:.3f} bayes={avg_bayes:.3f}")

    # =========================================================================
    # Finalize: compute metrics and rewrite each result
    # =========================================================================
    final_results: List[TrajectoryResult] = []
    for i in range(n_exps):
        result = TrajectoryResult(
            config=configs[i],
            rating_sequence=sequences[i],
            poll_points=poll_points,
            polls=polls_per_exp[i],
        )
        result.metrics = compute_trajectory_metrics(result, type_model)
        if output_dir is not None:
            save_result(result, output_dir)
        final_results.append(result)

    return final_results


def run_experiment(
    config: ExperimentConfig,
    sequence: List[Dict],
    type_model: TypeModel,
    movie_selection: MovieSelection,
    model=None,
    tokenizer=None,
    anon_map: Optional[Dict] = None,
) -> TrajectoryResult:
    """Run a single experiment.

    Args:
        anon_map: Optional pre-built anonymization map. If None and the
            condition needs one, it will be constructed on demand.
    """
    if config.condition == Condition.MULTI_TURN_ACTUAL and config.pop_info == PopInfo.ZERO_SHOT:
        raise ValueError("multi_turn_actual × zero_shot is not a valid combination (no profiles to classify)")

    if model is None or tokenizer is None:
        model, tokenizer = setup_model(config.model_name)

    target_movie = next(
        tm for tm in movie_selection.target_movies
        if tm["movie_id"] == config.target_movie_id
    )

    rating_counts = [0] * 5
    for r in sequence:
        rating_counts[r["rating"] - 1] += 1
    dist_str = "/".join(str(c) for c in rating_counts)
    avg = sum(r["rating"] for r in sequence) / len(sequence) if sequence else 0

    print(f"\nRunning experiment:")
    print(f"  Condition: {config.condition.value}")
    print(f"  Pop info: {config.pop_info.value}")
    print(f"  Model: {config.model_name}")
    print(f"  k={config.k}, true_type={config.true_type}, trial={config.trial}")
    print(f"  Target: {target_movie['movie_name']} (type {target_movie.get('owner_type', '?')}, {target_movie.get('direction', '?')})")
    print(f"  Sequence: ratings={dist_str} (avg={avg:.2f})")
    print()

    if config.condition == Condition.SINGLE_TURN:
        result = run_single_turn_experiment(
            model, tokenizer, config, sequence, type_model, movie_selection,
            anon_map=anon_map,
        )
    else:
        result = run_multi_turn_experiment(
            model, tokenizer, config, sequence, type_model, movie_selection,
            anon_map=anon_map,
        )

    return result


def save_result(result: TrajectoryResult, output_dir: Path, partial: bool = False):
    """Save experiment result to JSON.

    Args:
        partial: if True, write to a ``.partial.json`` suffix so that the
            skip-existing check ignores it. The final (complete) write uses
            the canonical ``.json`` name and removes the partial file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = result.config.get_output_filename()
    output_path = output_dir / filename
    partial_path = output_path.with_suffix(".partial.json")

    if partial:
        with open(partial_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        return partial_path

    # Final write: save to canonical name and clean up partial
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    if partial_path.exists():
        partial_path.unlink()

    print(f"\nSaved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Recommender System Cold-Start Experiment Runner")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--condition", type=str, required=True,
                        choices=[c.value for c in Condition],
                        help="Experimental condition")
    parser.add_argument("--pop-info", type=str, required=True,
                        choices=[p.value for p in PopInfo],
                        help="Population information mode")
    parser.add_argument("--k", type=int, default=1, choices=[1, 5],
                        help="Polling frequency")
    parser.add_argument("--true-type", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="True user type(s)")
    parser.add_argument("--n-types", type=int, default=4,
                        help="Number of user types (K)")
    parser.add_argument("--trials", type=str, default="0,1,2,3,4",
                        help="Trial numbers (comma-separated)")
    parser.add_argument("--n-ratings", type=int, default=50,
                        help="Number of ratings per sequence")
    parser.add_argument("--sequence-source", type=str, default="synthetic",
                        choices=[s.value for s in SequenceSource],
                        help="Sequence source")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (default: recommender_system/data/)")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent / "experiments"),
                        help="Output directory")
    parser.add_argument("--no-counterbalance", action="store_true",
                        help="Disable counterbalancing")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Max sequence length for vLLM (default: 8192)")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to LoRA adapter directory to load on top of base model")
    parser.add_argument("--with-replacement", action="store_true",
                        help="Sample probe movies with replacement (redundant evidence)")
    parser.add_argument("--misleading-genre", action="store_true",
                        help="Use misleading genre sequences (genre heuristic != Bayesian)")
    parser.add_argument("--batched", action="store_true",
                        help="Use the batched runner that processes all "
                             "(true_type × target × trial) experiments in "
                             "lockstep through their poll points. Output is "
                             "byte-equivalent to the serial path but vLLM "
                             "calls are batched across experiments at each "
                             "phase, giving substantial speedups.")
    parser.add_argument("--skip-cot", action="store_true",
                        help="Skip the Type CoT generation and CoT-followup "
                             "MCQ at every poll point. Conditioned rating "
                             "still runs but uses the implicit type-MCQ "
                             "argmax instead of the CoT-derived type. "
                             "cot_reasoning, cot_type_* fields are None in "
                             "the output. Saves ~50–70%% of wall-clock.")
    parser.add_argument("--marginalize-conditioned-rating", action="store_true",
                        help="In addition to the argmax-conditioned rating, "
                             "compute the Bayesian-correct marginalized "
                             "conditioned rating: K conditioned-rating probes "
                             "(one per type) per poll, weighted-averaged by "
                             "the model's type posterior. Adds ~3-4× cost on "
                             "the conditioned-rating phase but matches the "
                             "oracle's marginalization for fair comparison. "
                             "Saves to marginalized_conditioned_* fields.")

    args = parser.parse_args()

    condition = Condition(args.condition)
    pop_info = PopInfo(args.pop_info)
    seq_source = SequenceSource(args.sequence_source)
    trials = [int(t) for t in args.trials.split(",")]
    output_dir = Path(args.output_dir)

    # Load data
    print("Preparing data...")
    type_model, movie_selection = prepare_data(
        data_dir=args.data_dir, n_types=args.n_types,
    )

    # For real user sequences, find suitable users and pre-load ratings
    real_users = None
    prepared_ratings = None
    if seq_source == SequenceSource.REAL:
        from .data import load_ratings, prepare_ratings
        data_dir_path = Path(args.data_dir) if args.data_dir else Path(__file__).parent / "data"
        ml_dir = data_dir_path / "ml-1m"
        ratings = load_ratings(ml_dir)
        prepared_ratings = prepare_ratings(ratings)
        target_ids = [tm["movie_id"] for tm in movie_selection.target_movies]
        real_users = find_real_users(
            prepared_ratings, type_model, movie_selection.probe_movie_ids, target_ids
        )
        print(f"Found {len(real_users)} suitable real users")

    # Build job-level anonymization map once (only used for ANONYMIZED pop_info,
    # but cheap to build and lets all experiments in this job reuse the same map).
    job_anon_map = None
    if pop_info == PopInfo.ANONYMIZED:
        job_anon_map = build_anonymization_map(
            movie_selection.probe_movie_ids,
            movie_selection.target_movies,
            type_model,
        )

    # Load model once
    model, tokenizer = setup_model(args.model, max_model_len=args.max_model_len,
                                   lora_path=args.lora_path)

    # ------------------------------------------------------------------
    # Build the (config, sequence) list. The same builder is used by both
    # the serial and batched paths so they apply identical skip-existing
    # checks and identical sequence generation. The batched path then
    # dispatches them all in one shot; the serial path runs them in a loop.
    # ------------------------------------------------------------------
    pending_configs: List[ExperimentConfig] = []
    pending_sequences: List[List[Dict]] = []
    pending_resume_polls: List[Optional[List[PollResult]]] = []
    for true_type in args.true_type:
        for target_movie in movie_selection.target_movies:
            target_movie_id = target_movie["movie_id"]
            target_movie_name = target_movie["movie_name"]

            for trial in trials:
                config = ExperimentConfig(
                    model_name=args.model,
                    condition=condition,
                    pop_info=pop_info,
                    k=args.k,
                    true_type=true_type,
                    target_movie_id=target_movie_id,
                    target_movie_name=target_movie_name,
                    trial=trial,
                    n_types=args.n_types,
                    sequence_source=seq_source,
                    n_ratings=args.n_ratings,
                    counterbalance=not args.no_counterbalance,
                )

                # Check if already done (and complete)
                existing = output_dir / config.get_output_filename()
                partial_existing = existing.with_suffix(".partial.json")
                _resume_polls = None
                # Check canonical file first, then partial
                for _check_path, _label in [(existing, "exists"), (partial_existing, "partial file")]:
                    if not _check_path.exists():
                        continue
                    expected_polls = len(get_poll_points(config.n_ratings, config.k))
                    try:
                        with open(_check_path) as _f:
                            _data = json.load(_f)
                        actual_polls = len(_data.get("polls", []))
                        if actual_polls >= expected_polls and _label == "exists":
                            print(f"Skipping ({_label}): {_check_path}")
                            _resume_polls = "SKIP"
                            break
                        if actual_polls > 0:
                            _resume_polls = TrajectoryResult.from_dict(_data).polls
                            print(f"Resuming ({_label}: {actual_polls}/{expected_polls} polls): {_check_path}")
                            break
                    except (json.JSONDecodeError, KeyError):
                        print(f"Re-running (corrupt): {_check_path}")

                if _resume_polls == "SKIP":
                    continue

                # Generate or retrieve sequence
                if seq_source == SequenceSource.REAL and real_users:
                    matching = [u for u in real_users
                                if u["type"] == true_type
                                and target_movie_id in u["target_movie_ids_rated"]]
                    if trial >= len(matching):
                        print(f"Skipping (not enough real users for type={true_type}, "
                              f"target={target_movie_id}, trial={trial})")
                        continue
                    user = matching[trial]
                    sequence = get_real_user_sequence(
                        user["user_id"], prepared_ratings,
                        movie_selection.probe_movie_ids, type_model
                    )
                    if len(sequence) > args.n_ratings:
                        sequence = sequence[:args.n_ratings]
                elif args.misleading_genre:
                    sequence = generate_misleading_genre_sequence(
                        type_model,
                        movie_selection.probe_movie_ids,
                        target_movie_id=target_movie_id,
                        true_type=true_type,
                        n_ratings=args.n_ratings,
                        seed=config.seed,
                    )
                else:
                    sequence = generate_synthetic_sequence(
                        type_model,
                        movie_selection.probe_movie_ids,
                        true_type=true_type,
                        n_ratings=args.n_ratings,
                        seed=config.seed,
                        replace=args.with_replacement,
                    )

                pending_configs.append(config)
                pending_sequences.append(sequence)
                pending_resume_polls.append(_resume_polls)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    if not pending_configs:
        print("No experiments to run (everything already exists or skipped).")
        return

    if args.batched:
        # Build resume_polls for batched runner (None if no partials to resume)
        has_any_resume = any(p is not None for p in pending_resume_polls)
        batched_resume = None
        if has_any_resume:
            # All experiments should resume from the same point (lockstep).
            # Fill missing entries with empty lists.
            batched_resume = [p if p is not None else [] for p in pending_resume_polls]
            min_resume = min(len(p) for p in batched_resume)
            # Truncate all to the minimum so lockstep is maintained
            batched_resume = [p[:min_resume] for p in batched_resume]
            if min_resume > 0:
                print(f"\n=== Batched dispatch: {len(pending_configs)} experiments "
                      f"(resuming from poll {min_resume}) ===")
            else:
                batched_resume = None
                print(f"\n=== Batched dispatch: {len(pending_configs)} experiments ===")
        else:
            print(f"\n=== Batched dispatch: {len(pending_configs)} experiments ===")

        if condition == Condition.SINGLE_TURN:
            results = run_single_turn_batched(
                model, tokenizer, pending_configs, pending_sequences,
                type_model, movie_selection,
                output_dir=output_dir, anon_map=job_anon_map,
                resume_polls=batched_resume,
                skip_cot=args.skip_cot,
                marginalize_conditioned_rating=args.marginalize_conditioned_rating,
            )
        else:
            results = run_multi_turn_batched(
                model, tokenizer, pending_configs, pending_sequences,
                type_model, movie_selection,
                output_dir=output_dir, anon_map=job_anon_map,
                resume_polls=batched_resume,
                skip_cot=args.skip_cot,
                marginalize_conditioned_rating=args.marginalize_conditioned_rating,
            )
        for result in results:
            if result.metrics:
                _print_metrics(result)
        return

    # Serial fallback (the original code path).
    for config, sequence in zip(pending_configs, pending_sequences):
        result = run_experiment(
            config, sequence, type_model, movie_selection,
            model=model, tokenizer=tokenizer,
            anon_map=job_anon_map,
        )
        save_result(result, output_dir)

        if result.metrics:
            _print_metrics(result)


def _print_metrics(result: TrajectoryResult) -> None:
    """Print the per-experiment metrics summary used by both serial and
    batched dispatch paths."""
    m = result.metrics
    print(f"\nMetrics:")
    print(f"  MAE from Bayesian: {m.mae_from_bayesian:.4f}")
    print(f"  Corr w/ Bayesian: {m.correlation_with_bayesian:.4f}")
    print(f"  Prior E[rating]: {m.prior_expected_rating:.4f}")
    print(f"  Type inference correct: {m.type_inference_correct}")
    print(f"  Cross-item transfer: {m.cross_item_transfer_score:.4f}")
    print(f"  Mean update: {m.mean_update_magnitude:.4f}")
    print(f"  Trajectory variance: {m.trajectory_variance:.4f}")
    if getattr(m, 'genre_transfer_score', None) is not None:
        print(f"  Genre transfer: {m.genre_transfer_score:.4f}")
    if getattr(m, 'mean_kl_divergence', None) is not None:
        print(f"  Mean KL div: {m.mean_kl_divergence:.4f}")
    if getattr(m, 'mean_wasserstein', None) is not None:
        print(f"  Mean Wasserstein: {m.mean_wasserstein:.4f}")
    if m.type_accuracy is not None:
        print(f"  Type accuracy (MCQ): {m.type_accuracy:.4f}")
    if m.cot_type_accuracy is not None:
        print(f"  Type accuracy (CoT): {m.cot_type_accuracy:.4f}")
    if m.type_posterior_kl is not None:
        print(f"  Type posterior KL: {m.type_posterior_kl:.4f}")
    if m.conditioned_mae_from_bayesian is not None:
        print(f"  Conditioned MAE: {m.conditioned_mae_from_bayesian:.4f}")
    if m.conditioning_lift is not None:
        print(f"  Conditioning lift: {m.conditioning_lift:+.4f}")


if __name__ == "__main__":
    main()

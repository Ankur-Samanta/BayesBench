"""
Metric computation for recommender-system cold-start experiments.

Provides mixture model Bayesian ground truth and trajectory analysis metrics.
Uses Categorical likelihood over 1-5 star ratings with Dirichlet priors.
"""

from typing import List, Tuple, Dict
import numpy as np
from scipy import stats
from .config import TrajectoryResult, TrajectoryMetrics


def mixture_posterior(observations: List[Dict], type_model) -> np.ndarray:
    """
    Compute P(type=k | ratings) via Bayesian mixture model.

    P(type=k | ratings) ∝ pi_k × ∏ P(rating_i | type=k, movie_i)

    Likelihood is Categorical: P(rating=r | type=k, movie=m) = theta[m][k][r-1]

    Args:
        observations: List of dicts with "movie_id" and "rating" (int 1-5)
        type_model: TypeModel with pi and theta

    Returns:
        type_posterior array of shape (K,)
    """
    log_post = np.log(type_model.pi + 1e-10)

    for obs in observations:
        mid = obs["movie_id"]
        if mid not in type_model.theta:
            continue
        theta_km = type_model.theta[mid]  # shape (K, 5)
        rating = obs["rating"]  # int 1-5
        log_post += np.log(theta_km[:, rating - 1] + 1e-10)

    # Numerical stability
    log_post -= np.max(log_post)
    post = np.exp(log_post)
    return post / post.sum()


def expected_rating_bayesian(
    observations: List[Dict],
    target_movie_id: int,
    type_model,
) -> float:
    """
    Compute Bayesian E[rating | observations] for target movie.

    E[rating | obs] = sum_k P(type=k|obs) × E[rating | type=k, target]
    where E[rating | type=k, target] = sum_r r × theta[target][k][r-1]

    Args:
        observations: List of rating dicts
        target_movie_id: ID of the target movie
        type_model: TypeModel

    Returns:
        E[rating | observations] in [1, 5]
    """
    post = mixture_posterior(observations, type_model)
    theta_target = type_model.theta[target_movie_id]  # (K, 5)
    stars = np.arange(1, 6)
    e_per_type = theta_target @ stars  # (K,) — E[rating|type=k]
    return float(post @ e_per_type)


def marginal_baseline(observations: List[Dict], target_movie_id: int, type_model) -> float:
    """
    Compute marginal baseline E[rating] ignoring which movies were rated.

    This is the Dirichlet-Multinomial baseline: uses only rating counts as if all
    observations were about the target movie directly.

    E[rating] = sum_r r × (1 + c_r) / (5 + n_total)

    Returns 3.0 at t=0 (uniform Dirichlet prior), consistent for all t.
    """
    rating_counts = np.zeros(5)
    for o in observations:
        rating = o["rating"]  # int 1-5
        rating_counts[rating - 1] += 1
    n_total = rating_counts.sum()
    stars = np.arange(1, 6)
    return float(np.sum(stars * (1 + rating_counts) / (5 + n_total)))


def genre_overlap_baseline(
    observations: List[Dict],
    target_movie_id: int,
    type_model,
) -> float:
    """
    Compute genre-overlap baseline E[rating] using only movies sharing >= 1 genre
    with the target movie.

    This is a stronger baseline than the marginal: it uses genre similarity
    as a heuristic. If the LLM's cross-item transfer score relative to this
    baseline is still positive, the transfer is genuine structural inference
    beyond simple genre matching.

    Returns E[rating] based on genre-matched observations only, or 3.0 if no
    genre-overlapping observations exist.
    """
    target_genres = set(type_model.movie_genres.get(target_movie_id, []))

    if not target_genres or not observations:
        # Fall back to marginal baseline behavior
        return marginal_baseline(observations, target_movie_id, type_model)

    # Filter observations to those sharing >= 1 genre with target
    matched_ratings = []
    for obs in observations:
        mid = obs["movie_id"]
        obs_genres = set(type_model.movie_genres.get(mid, []))
        if obs_genres & target_genres:
            matched_ratings.append(obs["rating"])

    if not matched_ratings:
        # No genre-overlapping observations; use Dirichlet prior
        return 3.0

    # Dirichlet-Multinomial on matched ratings only
    rating_counts = np.zeros(5)
    for r in matched_ratings:
        rating_counts[r - 1] += 1
    n_total = rating_counts.sum()
    stars = np.arange(1, 6)
    return float(np.sum(stars * (1 + rating_counts) / (5 + n_total)))


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two discrete distributions."""
    return float(0.5 * np.sum(np.abs(p - q)))


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (symmetric, bounded [0, log 2])."""
    m = 0.5 * (p + q)
    m = np.clip(m, 1e-10, None)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def compute_trajectory_metrics(
    result: TrajectoryResult,
    type_model,
) -> TrajectoryMetrics:
    """
    Compute all metrics for an experiment trajectory.

    Args:
        result: TrajectoryResult with polls populated
        type_model: TypeModel for computing marginal baselines

    Returns:
        TrajectoryMetrics with all computed values
    """
    if not result.polls:
        raise ValueError("No polls in trajectory result")

    # Extract trajectories
    expected_rating_list = [p.expected_rating for p in result.polls]
    bayesian_list = [p.bayesian_posterior for p in result.polls]
    scale_biases = [p.scale_bias for p in result.polls]
    rating_masses = [p.rating_mass for p in result.polls]

    # MAE from Bayesian
    mae_from_bayesian = float(np.mean(np.abs(
        np.array(expected_rating_list) - np.array(bayesian_list)
    )))

    # Correlation with Bayesian
    if np.std(expected_rating_list) > 1e-6 and np.std(bayesian_list) > 1e-6:
        correlation_with_bayesian = float(np.corrcoef(expected_rating_list, bayesian_list)[0, 1])
    else:
        correlation_with_bayesian = 0.0

    # Prior expected rating - first poll should be at t=0
    prior_expected_rating = expected_rating_list[0] if result.polls[0].t == 0 else 3.0

    # Update magnitudes
    update_magnitudes = []
    for i in range(1, len(expected_rating_list)):
        update_magnitudes.append(abs(expected_rating_list[i] - expected_rating_list[i - 1]))

    mean_update_magnitude = float(np.mean(update_magnitudes)) if update_magnitudes else 0.0

    # Scale bias stats
    mean_scale_bias = float(np.mean(scale_biases))

    # Rating mass stats
    mean_rating_mass = float(np.mean(rating_masses))

    # Trajectory variance
    trajectory_variance = float(np.var(expected_rating_list))

    # Type inference correct
    final_type_posterior = result.polls[-1].type_posterior
    inferred_type = int(np.argmax(final_type_posterior))
    type_inference_correct = (inferred_type == result.config.true_type)

    # Cross-item transfer score
    target_mid = result.config.target_movie_id
    mae_llm_vs_bayes_values = []
    mae_marginal_vs_bayes_values = []
    mae_genre_vs_bayes_values = []

    for poll in result.polls:
        observations = result.rating_sequence[:poll.t]
        marginal_pred = marginal_baseline(observations, target_mid, type_model)
        genre_pred = genre_overlap_baseline(observations, target_mid, type_model)
        bayesian_pred = poll.bayesian_posterior

        mae_llm_vs_bayes_values.append(abs(poll.expected_rating - bayesian_pred))
        mae_marginal_vs_bayes_values.append(abs(marginal_pred - bayesian_pred))
        mae_genre_vs_bayes_values.append(abs(genre_pred - bayesian_pred))

    mae_marginal_vs_bayes = float(np.mean(mae_marginal_vs_bayes_values))
    mae_genre_vs_bayes = float(np.mean(mae_genre_vs_bayes_values))

    if mae_marginal_vs_bayes > 1e-6:
        cross_item_transfer_score = 1.0 - (mae_from_bayesian / mae_marginal_vs_bayes)
    else:
        cross_item_transfer_score = 0.0

    # Genre transfer score: relative to the stronger genre-overlap baseline
    if mae_genre_vs_bayes > 1e-6:
        genre_transfer_score = 1.0 - (mae_from_bayesian / mae_genre_vs_bayes)
    else:
        genre_transfer_score = 0.0

    # Full posterior evaluation: KL, Wasserstein, TVD, JSD
    kl_divergences = []
    wasserstein_distances = []
    tvd_distances = []
    jsd_distances = []
    stars = np.arange(1, 6)

    for poll in result.polls:
        observations = result.rating_sequence[:poll.t]
        # Compute Bayesian posterior distribution over 1-5 stars for target
        type_post = mixture_posterior(observations, type_model)
        if target_mid in type_model.theta:
            theta_target = type_model.theta[target_mid]  # (K, 5)
            # P(rating=r | obs) = sum_k P(type=k|obs) * P(rating=r | type=k, target)
            bayes_dist = np.array([
                float(type_post @ theta_target[:, r])
                for r in range(5)
            ])
        else:
            bayes_dist = np.array([0.2] * 5)

        llm_dist = np.array(poll.rating_distribution)

        # Ensure distributions are valid (non-negative, sum to 1)
        bayes_dist = np.clip(bayes_dist, 1e-10, None)
        bayes_dist = bayes_dist / bayes_dist.sum()
        llm_dist = np.clip(llm_dist, 1e-10, None)
        llm_dist = llm_dist / llm_dist.sum()

        # KL(Bayesian || LLM) - how much info is lost using LLM instead of Bayesian
        kl = float(np.sum(bayes_dist * np.log(bayes_dist / llm_dist)))
        kl_divergences.append(kl)

        # Wasserstein-1 distance on the ordinal 1-5 scale
        cdf_bayes = np.cumsum(bayes_dist)
        cdf_llm = np.cumsum(llm_dist)
        w1 = float(np.sum(np.abs(cdf_bayes - cdf_llm)))
        wasserstein_distances.append(w1)

        tvd_distances.append(tvd(bayes_dist, llm_dist))
        jsd_distances.append(jsd(bayes_dist, llm_dist))

    mean_kl = float(np.mean(kl_divergences)) if kl_divergences else None
    mean_w1 = float(np.mean(wasserstein_distances)) if wasserstein_distances else None
    mean_tvd_val = float(np.mean(tvd_distances)) if tvd_distances else None
    mean_jsd_val = float(np.mean(jsd_distances)) if jsd_distances else None

    # ── Type elicitation metrics ──────────────────────────────────────────
    type_posterior_kl = None
    type_posterior_tvd_val = None
    type_posterior_jsd_val = None
    type_accuracy = None
    cot_type_accuracy = None
    conditioned_mae = None
    conditioning_lift = None

    # CoT follow-up MCQ distributional metrics
    cot_type_posterior_kl = None
    cot_type_posterior_tvd_val = None
    cot_type_posterior_jsd_val = None
    mean_cot_type_mass = None
    mean_cot_type_scale_bias = None

    if result.polls[0].llm_type_distribution is not None:
        kl_values = []
        tvd_values = []
        jsd_values = []
        mcq_correct = 0
        cot_correct = 0
        cot_valid = 0
        cot_kl_values = []
        cot_tvd_values = []
        cot_jsd_values = []
        cot_type_masses = []
        cot_type_biases = []

        for poll in result.polls:
            bayes_type = np.array(poll.type_posterior)
            llm_type = np.array(poll.llm_type_distribution)

            # Clip, normalize, compute metrics
            bayes_type = np.clip(bayes_type, 1e-10, None)
            bayes_type = bayes_type / bayes_type.sum()
            llm_type = np.clip(llm_type, 1e-10, None)
            llm_type = llm_type / llm_type.sum()

            kl = float(np.sum(bayes_type * np.log(bayes_type / llm_type)))
            kl_values.append(kl)
            tvd_values.append(tvd(bayes_type, llm_type))
            jsd_values.append(jsd(bayes_type, llm_type))

            if np.argmax(llm_type) == np.argmax(bayes_type):
                mcq_correct += 1

            if poll.cot_type_distribution is not None:
                cot_valid += 1
                if poll.cot_type_prediction == np.argmax(bayes_type):
                    cot_correct += 1

                # CoT follow-up MCQ distributional metrics
                cot_type = np.array(poll.cot_type_distribution)
                cot_type = np.clip(cot_type, 1e-10, None)
                cot_type = cot_type / cot_type.sum()
                cot_kl = float(np.sum(bayes_type * np.log(bayes_type / cot_type)))
                cot_kl_values.append(cot_kl)
                cot_tvd_values.append(tvd(bayes_type, cot_type))
                cot_jsd_values.append(jsd(bayes_type, cot_type))

            if poll.cot_type_mass is not None:
                cot_type_masses.append(poll.cot_type_mass)
            if poll.cot_type_scale_bias is not None:
                cot_type_biases.append(poll.cot_type_scale_bias)

        type_posterior_kl = float(np.mean(kl_values))
        type_posterior_tvd_val = float(np.mean(tvd_values))
        type_posterior_jsd_val = float(np.mean(jsd_values))
        type_accuracy = mcq_correct / len(result.polls)
        cot_type_accuracy = cot_correct / cot_valid if cot_valid > 0 else None
        cot_type_posterior_kl = float(np.mean(cot_kl_values)) if cot_kl_values else None
        cot_type_posterior_tvd_val = float(np.mean(cot_tvd_values)) if cot_tvd_values else None
        cot_type_posterior_jsd_val = float(np.mean(cot_jsd_values)) if cot_jsd_values else None
        mean_cot_type_mass = float(np.mean(cot_type_masses)) if cot_type_masses else None
        mean_cot_type_scale_bias = float(np.mean(cot_type_biases)) if cot_type_biases else None

    if result.polls[0].conditioned_expected_rating is not None:
        cond_errors = []
        for p in result.polls:
            if p.conditioned_expected_rating is not None:
                cond_errors.append(abs(p.conditioned_expected_rating - p.bayesian_posterior))
        if cond_errors:
            conditioned_mae = float(np.mean(cond_errors))
            conditioning_lift = mae_from_bayesian - conditioned_mae  # positive = helps

    # ── Rating-given-type fidelity ──────────────────────────────────────────
    # Apples-to-apples reference: when the LLM is told "you are type k", the
    # right Bayesian comparison is E[rating | type=k, target] (a single theta
    # row), NOT the mixture E[rating | obs]. Isolates rating-given-type
    # competence from type-inference competence.
    mae_rating_given_predicted_type = None
    mae_rating_given_type_per_type = None

    if (result.polls[0].conditioned_expected_rating is not None
            and target_mid in type_model.theta):
        theta_target = type_model.theta[target_mid]  # (K, 5)
        e_per_type_theta = theta_target @ stars       # (K,)

        per_poll_errors = []
        for p in result.polls:
            if p.conditioned_expected_rating is None:
                continue
            k_llm = (p.cot_type_prediction
                     if p.cot_type_prediction is not None
                     else p.llm_type_prediction)
            if k_llm is None or not (0 <= k_llm < len(e_per_type_theta)):
                continue
            per_poll_errors.append(
                abs(p.conditioned_expected_rating - float(e_per_type_theta[k_llm]))
            )
        if per_poll_errors:
            mae_rating_given_predicted_type = float(np.mean(per_poll_errors))

    if (result.polls[0].marginalized_conditioned_rating_per_type is not None
            and target_mid in type_model.theta):
        theta_target = type_model.theta[target_mid]  # (K, 5)
        e_per_type_theta = theta_target @ stars       # (K,)

        per_type_errors = []
        for p in result.polls:
            if p.marginalized_conditioned_rating_per_type is None:
                continue
            per_type_dists = np.array(p.marginalized_conditioned_rating_per_type)  # (K, 5)
            if per_type_dists.shape != theta_target.shape:
                continue
            e_per_type_llm = per_type_dists @ stars  # (K,)
            per_type_errors.append(np.abs(e_per_type_llm - e_per_type_theta))
        if per_type_errors:
            # Mean over polls and over types (uniform — pure fidelity, not
            # weighted by posterior plausibility).
            mae_rating_given_type_per_type = float(
                np.mean(np.concatenate(per_type_errors))
            )

    return TrajectoryMetrics(
        mae_from_bayesian=mae_from_bayesian,
        correlation_with_bayesian=correlation_with_bayesian,
        prior_expected_rating=prior_expected_rating,
        mean_update_magnitude=mean_update_magnitude,
        update_magnitudes=update_magnitudes,
        mean_scale_bias=mean_scale_bias,
        scale_biases=scale_biases,
        mean_rating_mass=mean_rating_mass,
        rating_masses=rating_masses,
        trajectory_variance=trajectory_variance,
        type_inference_correct=type_inference_correct,
        cross_item_transfer_score=cross_item_transfer_score,
        genre_transfer_score=genre_transfer_score,
        mean_kl_divergence=mean_kl,
        mean_wasserstein=mean_w1,
        mean_tvd=mean_tvd_val,
        mean_jsd=mean_jsd_val,
        type_posterior_kl=type_posterior_kl,
        type_posterior_tvd=type_posterior_tvd_val,
        type_posterior_jsd=type_posterior_jsd_val,
        type_accuracy=type_accuracy,
        cot_type_accuracy=cot_type_accuracy,
        conditioned_mae_from_bayesian=conditioned_mae,
        conditioning_lift=conditioning_lift,
        mae_rating_given_predicted_type=mae_rating_given_predicted_type,
        mae_rating_given_type_per_type=mae_rating_given_type_per_type,
        cot_type_posterior_kl=cot_type_posterior_kl,
        cot_type_posterior_tvd=cot_type_posterior_tvd_val,
        cot_type_posterior_jsd=cot_type_posterior_jsd_val,
        mean_cot_type_mass=mean_cot_type_mass,
        mean_cot_type_scale_bias=mean_cot_type_scale_bias,
    )


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


# =============================================================================
# Research-question-specific metrics (analogs of coin_flip metrics)
# =============================================================================

def compute_update_direction_accuracy(result: TrajectoryResult, type_model) -> dict:
    """
    Compute whether updates go in the correct direction given the observation.

    When user rates a type-diagnostic movie, E[rating] should move toward
    the Bayesian prediction. Analogous to coin_flip's compute_update_direction_accuracy().

    Args:
        result: Single TrajectoryResult
        type_model: TypeModel for computing expected direction

    Returns:
        Dictionary with direction accuracy metrics
    """
    if not result.polls or len(result.polls) < 2:
        return {}

    correct_direction = 0
    wrong_direction = 0
    no_change = 0

    target_mid = result.config.target_movie_id

    for i in range(1, len(result.polls)):
        prev_poll = result.polls[i - 1]
        curr_poll = result.polls[i]

        # Get observations between these polls
        start_t = prev_poll.t
        end_t = curr_poll.t
        observations_window = result.rating_sequence[start_t:end_t]

        if not observations_window:
            continue

        # Compute expected direction: what does the Bayesian model predict?
        obs_before = [{"movie_id": r["movie_id"], "rating": r["rating"]}
                      for r in result.rating_sequence[:start_t]]
        obs_after = [{"movie_id": r["movie_id"], "rating": r["rating"]}
                     for r in result.rating_sequence[:end_t]]

        bayes_before = expected_rating_bayesian(obs_before, target_mid, type_model)
        bayes_after = expected_rating_bayesian(obs_after, target_mid, type_model)
        expected_direction = bayes_after - bayes_before

        # What direction did the LLM actually move?
        actual_change = curr_poll.expected_rating - prev_poll.expected_rating

        if abs(actual_change) < 1e-6:
            no_change += 1
        elif (expected_direction > 0 and actual_change > 0) or \
             (expected_direction < 0 and actual_change < 0) or \
             abs(expected_direction) < 1e-6:
            correct_direction += 1
        else:
            wrong_direction += 1

    total = correct_direction + wrong_direction + no_change

    return {
        "correct_direction": correct_direction,
        "wrong_direction": wrong_direction,
        "no_change": no_change,
        "direction_accuracy": correct_direction / total if total > 0 else 0.0,
    }


def compute_genre_sensitivity(result: TrajectoryResult, type_model) -> dict:
    """
    Do updates from high-diagnosticity ratings produce larger E[rating] changes
    than low-diagnosticity ratings?

    Only meaningful for k=1 (single observation updates).
    Diagnosticity = range of E[rating] across types for the observed movie.

    Args:
        result: Single TrajectoryResult
        type_model: TypeModel with theta for diagnosticity computation

    Returns:
        Dictionary with genre sensitivity metrics
    """
    if not result.polls or len(result.polls) < 2 or result.config.k != 1:
        return {}

    high_diag_updates = []
    low_diag_updates = []

    # Compute diagnosticity for each movie in the sequence
    stars = np.arange(1, 6)
    all_diag = []
    for r in result.rating_sequence:
        mid = r["movie_id"]
        if mid in type_model.theta:
            theta_m = type_model.theta[mid]
            e_per_type = theta_m @ stars
            all_diag.append(float(e_per_type.max() - e_per_type.min()))
    if not all_diag:
        return {}
    median_diag = float(np.median(all_diag))

    for i in range(1, len(result.polls)):
        prev_poll = result.polls[i - 1]
        curr_poll = result.polls[i]

        obs_t = prev_poll.t
        if obs_t >= len(result.rating_sequence):
            continue

        rating = result.rating_sequence[obs_t]
        mid = rating["movie_id"]
        if mid not in type_model.theta:
            continue

        theta_m = type_model.theta[mid]
        e_per_type = theta_m @ stars
        diag = float(e_per_type.max() - e_per_type.min())
        update_magnitude = abs(curr_poll.expected_rating - prev_poll.expected_rating)

        if diag >= median_diag:
            high_diag_updates.append(update_magnitude)
        else:
            low_diag_updates.append(update_magnitude)

    if not high_diag_updates or not low_diag_updates:
        return {}

    mean_high = float(np.mean(high_diag_updates))
    mean_low = float(np.mean(low_diag_updates))

    # Mann-Whitney U test for difference
    u_stat, p_value = stats.mannwhitneyu(
        high_diag_updates, low_diag_updates, alternative="greater"
    )

    return {
        "mean_high_diag_update": mean_high,
        "mean_low_diag_update": mean_low,
        "n_high_diag": len(high_diag_updates),
        "n_low_diag": len(low_diag_updates),
        "median_diagnosticity_threshold": median_diag,
        "sensitivity_ratio": mean_high / mean_low if mean_low > 1e-6 else None,
        "u_statistic": float(u_stat),
        "p_value": float(p_value),
        "high_diag_greater": bool(p_value < 0.05),
    }


def compute_convergence_speed(
    result: TrajectoryResult,
    type_model,
    epsilon: float = 0.05,
) -> dict:
    """
    At what t does |LLM - Bayesian| < epsilon? Compare across conditions.

    Args:
        result: Single TrajectoryResult
        type_model: TypeModel (unused, kept for API consistency)
        epsilon: Convergence threshold

    Returns:
        Dictionary with convergence speed metrics
    """
    if not result.polls:
        return {}

    convergence_t = None
    for poll in result.polls:
        if abs(poll.expected_rating - poll.bayesian_posterior) < epsilon:
            convergence_t = poll.t
            break

    # Also check if it stays converged
    sustained_convergence_t = None
    for i, poll in enumerate(result.polls):
        if abs(poll.expected_rating - poll.bayesian_posterior) < epsilon:
            # Check all remaining polls
            all_converged = all(
                abs(p.expected_rating - p.bayesian_posterior) < epsilon
                for p in result.polls[i:]
            )
            if all_converged:
                sustained_convergence_t = poll.t
                break

    # Final error
    final_error = abs(
        result.polls[-1].expected_rating - result.polls[-1].bayesian_posterior
    )

    return {
        "first_convergence_t": convergence_t,
        "sustained_convergence_t": sustained_convergence_t,
        "epsilon": epsilon,
        "final_error": float(final_error),
        "converged_at_end": bool(final_error < epsilon),
        "n_polls": len(result.polls),
        "total_t": result.polls[-1].t,
    }


def compute_trajectory_divergence(
    result1: TrajectoryResult,
    result2: TrajectoryResult,
) -> dict:
    """
    Compute divergence between two trajectories (same sequence, different conditions).

    Args:
        result1: First trajectory (e.g. single_turn)
        result2: Second trajectory (e.g. multi_turn_minimal)

    Returns:
        Dictionary with divergence metrics
    """
    polls1 = {p.t: p.expected_rating for p in result1.polls}
    polls2 = {p.t: p.expected_rating for p in result2.polls}

    common_t = sorted(set(polls1.keys()) & set(polls2.keys()))

    if not common_t:
        return {}

    diffs = [polls1[t] - polls2[t] for t in common_t]
    abs_diffs = [abs(d) for d in diffs]

    if len(diffs) > 1:
        ratings_1 = [polls1[t] for t in common_t]
        ratings_2 = [polls2[t] for t in common_t]
        t_stat, p_value = stats.ttest_rel(ratings_1, ratings_2)
    else:
        t_stat, p_value = 0.0, 1.0

    return {
        "n_common_points": len(common_t),
        "mean_difference": float(np.mean(diffs)),
        "mean_abs_difference": float(np.mean(abs_diffs)),
        "max_abs_difference": float(np.max(abs_diffs)),
        "paired_t_statistic": float(t_stat),
        "paired_p_value": float(p_value),
        "significantly_different": bool(p_value < 0.05),
    }

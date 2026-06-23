"""
Metric computation for coin flip experiments.

Provides Bayesian baselines and trajectory analysis metrics.
"""

from typing import List, Tuple
import numpy as np
from scipy import stats
from .config import TrajectoryResult, TrajectoryMetrics


def bayesian_posterior(history: List[str], prior_alpha: float = 1.0, prior_beta: float = 1.0) -> float:
    """
    Compute Beta-Bernoulli posterior mean.

    With uniform prior (alpha=1, beta=1):
        P(heads) = (1 + n_heads) / (2 + n_total)

    Args:
        history: List of outcomes ("heads" or "tails")
        prior_alpha: Beta prior alpha parameter
        prior_beta: Beta prior beta parameter

    Returns:
        Posterior mean for P(heads)
    """
    n_heads = sum(1 for f in history if f == "heads")
    n_total = len(history)
    return (prior_alpha + n_heads) / (prior_alpha + prior_beta + n_total)


def compute_trajectory_metrics(result: TrajectoryResult) -> TrajectoryMetrics:
    """
    Compute all metrics for an experiment trajectory.

    Args:
        result: TrajectoryResult with polls populated

    Returns:
        TrajectoryMetrics with all computed values
    """
    if not result.polls:
        raise ValueError("No polls in trajectory result")

    # Extract trajectories
    p_heads_list = [p.p_heads for p in result.polls]
    bayesian_list = [p.bayesian_posterior for p in result.polls]
    position_biases = [p.position_bias for p in result.polls]
    ab_masses = [p.ab_mass for p in result.polls]

    # MAE from Bayesian
    mae_from_bayesian = float(np.mean(np.abs(
        np.array(p_heads_list) - np.array(bayesian_list)
    )))

    # Correlation with Bayesian
    if np.std(p_heads_list) > 1e-6 and np.std(bayesian_list) > 1e-6:
        correlation_with_bayesian = float(np.corrcoef(p_heads_list, bayesian_list)[0, 1])
    else:
        correlation_with_bayesian = 0.0

    # Prior P(heads) - first poll should be at t=0
    prior_p_heads = p_heads_list[0] if result.polls[0].t == 0 else 0.5

    # Update magnitudes (consecutive differences)
    update_magnitudes = []
    for i in range(1, len(p_heads_list)):
        update_magnitudes.append(abs(p_heads_list[i] - p_heads_list[i-1]))

    mean_update_magnitude = float(np.mean(update_magnitudes)) if update_magnitudes else 0.0

    # Position bias stats
    mean_position_bias = float(np.mean(position_biases))

    # A/B mass stats
    mean_ab_mass = float(np.mean(ab_masses))

    # Trajectory variance
    trajectory_variance = float(np.var(p_heads_list))

    return TrajectoryMetrics(
        mae_from_bayesian=mae_from_bayesian,
        correlation_with_bayesian=correlation_with_bayesian,
        prior_p_heads=prior_p_heads,
        mean_update_magnitude=mean_update_magnitude,
        update_magnitudes=update_magnitudes,
        mean_position_bias=mean_position_bias,
        position_biases=position_biases,
        mean_ab_mass=mean_ab_mass,
        ab_masses=ab_masses,
        trajectory_variance=trajectory_variance
    )


def bootstrap_ci(
    data: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        data: List of values
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed

    Returns:
        (mean, lower_bound, upper_bound)
    """
    if len(data) == 0:
        return 0.0, 0.0, 0.0

    if len(data) == 1:
        return data[0], data[0], data[0]

    np.random.seed(seed)
    data_arr = np.array(data)

    # Bootstrap samples
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data_arr, size=len(data_arr), replace=True)
        bootstrap_means.append(np.mean(sample))

    # Compute percentiles
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    mean = np.mean(data)

    return float(mean), float(lower), float(upper)


def compute_update_per_observation(result: TrajectoryResult) -> List[float]:
    """
    Compute per-observation update rate.

    For k=1, this is just the update magnitudes.
    For k=5, we divide by 5 to get per-observation rate.

    Args:
        result: TrajectoryResult

    Returns:
        List of per-observation update rates
    """
    k = result.config.k

    if not result.metrics or not result.metrics.update_magnitudes:
        return []

    # Normalize by k
    return [u / k for u in result.metrics.update_magnitudes]


def compute_self_anchoring_metrics(results_actual: List[TrajectoryResult]) -> dict:
    """
    Compute self-anchoring specific metrics from multi_turn_actual condition.

    Args:
        results_actual: List of TrajectoryResults from multi_turn_actual condition

    Returns:
        Dictionary with self-anchoring metrics
    """
    if not results_actual:
        return {}

    # Collect prediction accuracy
    correct_predictions = 0
    total_predictions = 0

    for result in results_actual:
        for poll in result.polls:
            if poll.prediction is not None:
                # Check if prediction matched outcome
                # Need to look at the next outcome after this poll
                t = poll.t
                if t < len(result.sequence):
                    actual_outcome = result.sequence[t]
                    # prediction is stored as "heads" or "tails"
                    if poll.prediction == actual_outcome:
                        correct_predictions += 1
                    total_predictions += 1

    prediction_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    # Compute trajectory stickiness (how much predictions persist)
    variances = [r.metrics.trajectory_variance for r in results_actual if r.metrics]
    mean_variance = float(np.mean(variances)) if variances else 0.0

    return {
        "prediction_accuracy": prediction_accuracy,
        "total_predictions": total_predictions,
        "mean_trajectory_variance": mean_variance
    }


def compute_paired_self_anchoring(
    results_actual: List[TrajectoryResult],
    results_minimal: List[TrajectoryResult]
) -> dict:
    """
    Measure self-anchoring by comparing multi_turn_actual vs multi_turn_minimal
    on matched trials (same p, trial, k, coin_spec).

    For each poll t > 0, the model in multi_turn_actual has its own prior prediction
    in context. We measure how much that shifts p_heads relative to the minimal baseline:

        anchoring_score = sign(prediction) * (p_heads_actual - p_heads_minimal)

    Positive score means the model shifted toward its own prior prediction.
    We split by whether that prior prediction was correct (outcome matched) or wrong,
    because anchoring on wrong predictions is the clearest signature of self-anchoring
    rather than rational updating.

    Args:
        results_actual: TrajectoryResults from multi_turn_actual
        results_minimal: TrajectoryResults from multi_turn_minimal (matched trials)

    Returns:
        Dictionary with self-anchoring metrics
    """
    if not results_actual or not results_minimal:
        return {}

    # Index minimal results by trial
    minimal_by_trial = {r.config.trial: r for r in results_minimal}

    anchoring_scores = []
    correct_pred_scores = []  # scores where prediction matched the prior outcome
    wrong_pred_scores = []    # scores where prediction contradicted the prior outcome

    for result_a in results_actual:
        trial = result_a.config.trial
        if trial not in minimal_by_trial:
            continue
        result_m = minimal_by_trial[trial]

        minimal_p = {poll.t: poll.p_heads for poll in result_m.polls}

        for poll in result_a.polls:
            if poll.t == 0 or poll.prediction is None:
                continue
            if poll.t not in minimal_p:
                continue

            delta = poll.p_heads - minimal_p[poll.t]

            # poll.prediction is what the model said before seeing outcome at t-1
            pred = poll.prediction  # "heads" or "tails"
            sign = 1.0 if pred == "heads" else -1.0
            score = sign * delta  # positive = shifted toward own prediction

            anchoring_scores.append(score)

            # Was the prediction correct about the outcome at t-1?
            prior_outcome = result_a.sequence[poll.t - 1]
            was_correct = (pred == prior_outcome)

            if was_correct:
                correct_pred_scores.append(score)
            else:
                wrong_pred_scores.append(score)

    if not anchoring_scores:
        return {}

    mean_score = float(np.mean(anchoring_scores))
    if len(anchoring_scores) > 1:
        t_stat, p_value = stats.ttest_1samp(anchoring_scores, 0.0)
    else:
        t_stat, p_value = 0.0, 1.0

    result = {
        "n_observations": len(anchoring_scores),
        "mean_anchoring_score": mean_score,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "interpretation": "positive = model shifts p_heads toward its own prior prediction"
    }

    if correct_pred_scores:
        cp_mean = float(np.mean(correct_pred_scores))
        cp_entry = {"n": len(correct_pred_scores), "mean_anchoring_score": cp_mean}
        if len(correct_pred_scores) > 1:
            t_cp, p_cp = stats.ttest_1samp(correct_pred_scores, 0.0)
            cp_entry["t_statistic"] = float(t_cp)
            cp_entry["p_value"] = float(p_cp)
            cp_entry["significant"] = bool(p_cp < 0.05)
        result["correct_prediction"] = cp_entry

    if wrong_pred_scores:
        wp_mean = float(np.mean(wrong_pred_scores))
        wp_entry = {"n": len(wrong_pred_scores), "mean_anchoring_score": wp_mean}
        if len(wrong_pred_scores) > 1:
            t_wp, p_wp = stats.ttest_1samp(wrong_pred_scores, 0.0)
            wp_entry["t_statistic"] = float(t_wp)
            wp_entry["p_value"] = float(p_wp)
            wp_entry["significant"] = bool(p_wp < 0.05)
        result["wrong_prediction"] = wp_entry

    return result


def compute_icrl_metrics(
    results_actual: List[TrajectoryResult],
    results_minimal: List[TrajectoryResult],
    n_bins: int = 4,
    rolling_window: int = 10
) -> dict:
    """
    Measure in-context reinforcement learning (ICRL): does having the model's
    own (prediction, outcome) history in context improve prediction accuracy
    over time, compared to the minimal condition where predictions are absent?

    Both conditions use p_heads > 0.5 as an implicit binary prediction for
    the next flip, checked against sequence[t]. This gives a comparable
    accuracy signal across both conditions at every poll point.

    The ICRL signal: accuracy should increase with t in actual (the model
    learns from its error feedback), and this trend should be stronger in
    actual than in minimal (where no prediction feedback exists).

    Args:
        results_actual: TrajectoryResults from multi_turn_actual
        results_minimal: TrajectoryResults from multi_turn_minimal
        n_bins: Number of equal-width time bins for coarse summary
        rolling_window: Window size for rolling mean (centered)

    Returns:
        Dictionary with ICRL metrics
    """

    def implicit_accuracy_over_t(results):
        """
        For each poll at time t, treat p_heads > 0.5 as predicting 'heads'
        and check against sequence[t] (the next flip to be revealed).
        Returns list of (t, correct) pairs.
        """
        t_correct = []
        for r in results:
            for poll in r.polls:
                if poll.t >= len(r.sequence):
                    continue
                next_outcome = r.sequence[poll.t]
                predicted_heads = poll.p_heads > 0.5
                actual_heads = next_outcome == "heads"
                t_correct.append((poll.t, int(predicted_heads == actual_heads)))
        return t_correct

    def summarize_accuracy(t_correct, rolling_window, n_bins):
        if not t_correct:
            return {}

        all_t = [x[0] for x in t_correct]
        all_correct = [x[1] for x in t_correct]

        # Per-t mean across trials
        t_acc_map = {}
        for t, c in t_correct:
            t_acc_map.setdefault(t, []).append(c)
        per_t = {t: float(np.mean(v)) for t, v in sorted(t_acc_map.items())}

        # Rolling mean (centered)
        sorted_t = sorted(per_t.keys())
        rolling = {}
        for i, t in enumerate(sorted_t):
            lo = max(0, i - rolling_window // 2)
            hi = min(len(sorted_t), i + rolling_window // 2 + 1)
            window_keys = set(sorted_t[lo:hi])
            vals = [c for tv, c in t_correct if tv in window_keys]
            rolling[t] = float(np.mean(vals)) if vals else 0.0

        # Coarse bins
        t_max = max(all_t)
        bin_size = t_max / n_bins
        bins = []
        for b in range(n_bins):
            t_lo, t_hi = b * bin_size, (b + 1) * bin_size
            in_bin = [c for t, c in t_correct if t_lo < t <= t_hi]
            if in_bin:
                bins.append({
                    "t_range": [round(t_lo), round(t_hi)],
                    "n": len(in_bin),
                    "accuracy": float(np.mean(in_bin))
                })

        slope, _, r_val, p_val, _ = stats.linregress(all_t, all_correct) if len(all_t) > 1 else (0., 0., 0., 1., 0.)

        return {
            "per_t": per_t,
            "rolling": rolling,
            "by_bin": bins,
            "trend_slope": float(slope),
            "trend_r": float(r_val),
            "trend_p_value": float(p_val),
            "improving": bool(slope > 0 and p_val < 0.05)
        }

    tc_actual = implicit_accuracy_over_t(results_actual) if results_actual else []
    tc_minimal = implicit_accuracy_over_t(results_minimal) if results_minimal else []

    result = {
        "actual": summarize_accuracy(tc_actual, rolling_window, n_bins),
        "minimal": summarize_accuracy(tc_minimal, rolling_window, n_bins),
    }

    # Compare trends: is the slope in actual significantly greater than in minimal?
    if tc_actual and tc_minimal:
        actual_t  = [x[0] for x in tc_actual]
        actual_c  = [x[1] for x in tc_actual]
        minimal_t = [x[0] for x in tc_minimal]
        minimal_c = [x[1] for x in tc_minimal]

        slope_a, _, _, _, _ = stats.linregress(actual_t, actual_c)
        slope_m, _, _, _, _ = stats.linregress(minimal_t, minimal_c)

        # At matched t values, is actual accuracy higher and does the gap widen?
        per_t_a = result["actual"]["per_t"]
        per_t_m = result["minimal"]["per_t"]
        common_t = sorted(set(per_t_a) & set(per_t_m))
        if len(common_t) > 1:
            diffs = [per_t_a[t] - per_t_m[t] for t in common_t]
            slope_diff, _, _, p_diff, _ = stats.linregress(common_t, diffs)
            result["comparison"] = {
                "actual_trend_slope": float(slope_a),
                "minimal_trend_slope": float(slope_m),
                "slope_difference": float(slope_a - slope_m),
                "mean_accuracy_gap": float(np.mean(diffs)),
                "gap_trend_slope": float(slope_diff),
                "gap_trend_p_value": float(p_diff),
                "actual_improves_more": bool(slope_a > slope_m),
                "gap_widens_over_time": bool(slope_diff > 0 and p_diff < 0.05)
            }

    return result


# =============================================================================
# Q1: Prior and Updating Analysis
# =============================================================================

def compute_prior_bias(results: List[TrajectoryResult]) -> dict:
    """
    Analyze whether models have a prior bias (P(heads) ≠ 0.5 at t=0).

    Args:
        results: List of TrajectoryResults

    Returns:
        Dictionary with prior bias statistics
    """
    if not results:
        return {}

    priors = [r.metrics.prior_p_heads for r in results if r.metrics]

    if not priors:
        return {}

    mean_prior = float(np.mean(priors))
    std_prior = float(np.std(priors))

    # One-sample t-test against 0.5
    from scipy import stats
    if len(priors) > 1 and std_prior > 1e-6:
        t_stat, p_value = stats.ttest_1samp(priors, 0.5)
    else:
        t_stat, p_value = 0.0, 1.0

    # Classify bias
    if p_value < 0.05:
        if mean_prior > 0.5:
            bias_direction = "heads"
        else:
            bias_direction = "tails"
    else:
        bias_direction = "neutral"

    return {
        "mean_prior": mean_prior,
        "std_prior": std_prior,
        "bias_from_neutral": mean_prior - 0.5,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "bias_direction": bias_direction
    }


def compute_updating_exists(results: List[TrajectoryResult]) -> dict:
    """
    Test whether the model updates beliefs at all (prior ≠ posterior).

    Args:
        results: List of TrajectoryResults

    Returns:
        Dictionary with updating existence statistics
    """
    if not results:
        return {}

    prior_posterior_diffs = []
    final_bayesian_diffs = []

    for r in results:
        if r.metrics and r.polls:
            prior = r.polls[0].p_heads if r.polls[0].t == 0 else None
            final = r.polls[-1].p_heads
            final_bayesian = r.polls[-1].bayesian_posterior

            if prior is not None:
                prior_posterior_diffs.append(final - prior)
                final_bayesian_diffs.append(final - final_bayesian)

    if not prior_posterior_diffs:
        return {}

    from scipy import stats

    # Test: is final P(heads) different from prior?
    if len(prior_posterior_diffs) > 1:
        t_stat, p_value = stats.ttest_1samp(prior_posterior_diffs, 0.0)
    else:
        t_stat, p_value = 0.0, 1.0

    return {
        "mean_prior_to_final_change": float(np.mean(prior_posterior_diffs)),
        "std_prior_to_final_change": float(np.std(prior_posterior_diffs)),
        "updating_t_statistic": float(t_stat),
        "updating_p_value": float(p_value),
        "updating_exists": bool(p_value < 0.05),
        "mean_final_error_from_bayesian": float(np.mean(np.abs(final_bayesian_diffs))),
    }


def compute_update_direction_accuracy(result: TrajectoryResult) -> dict:
    """
    Compute whether updates go in the correct direction.

    When seeing heads, P(heads) should increase. When seeing tails, it should decrease.

    Args:
        result: Single TrajectoryResult

    Returns:
        Dictionary with direction accuracy metrics
    """
    if not result.polls or len(result.polls) < 2:
        return {}

    correct_direction = 0
    wrong_direction = 0
    no_change = 0

    k = result.config.k

    for i in range(1, len(result.polls)):
        prev_poll = result.polls[i-1]
        curr_poll = result.polls[i]

        # Get observations between these polls
        start_t = prev_poll.t
        end_t = curr_poll.t

        observations = result.sequence[start_t:end_t]
        n_heads_delta = sum(1 for o in observations if o == "heads")
        n_tails_delta = len(observations) - n_heads_delta

        # What direction should P(heads) move?
        expected_direction = n_heads_delta - n_tails_delta  # positive = should increase

        # What direction did it actually move?
        actual_change = curr_poll.p_heads - prev_poll.p_heads

        if abs(actual_change) < 1e-6:
            no_change += 1
        elif (expected_direction > 0 and actual_change > 0) or \
             (expected_direction < 0 and actual_change < 0) or \
             (expected_direction == 0):
            correct_direction += 1
        else:
            wrong_direction += 1

    total = correct_direction + wrong_direction + no_change

    return {
        "correct_direction": correct_direction,
        "wrong_direction": wrong_direction,
        "no_change": no_change,
        "direction_accuracy": correct_direction / total if total > 0 else 0.0
    }


# =============================================================================
# Q2: Single-Turn vs Multi-Turn Comparison
# =============================================================================

def compute_trajectory_divergence(result1: TrajectoryResult, result2: TrajectoryResult) -> dict:
    """
    Compute divergence between two trajectories (should be from same p, trial).

    Args:
        result1: First trajectory
        result2: Second trajectory

    Returns:
        Dictionary with divergence metrics
    """
    # Get common poll points
    polls1 = {p.t: p.p_heads for p in result1.polls}
    polls2 = {p.t: p.p_heads for p in result2.polls}

    common_t = sorted(set(polls1.keys()) & set(polls2.keys()))

    if not common_t:
        return {}

    diffs = [polls1[t] - polls2[t] for t in common_t]
    abs_diffs = [abs(d) for d in diffs]

    from scipy import stats
    if len(diffs) > 1:
        # Paired t-test
        p_heads_1 = [polls1[t] for t in common_t]
        p_heads_2 = [polls2[t] for t in common_t]
        t_stat, p_value = stats.ttest_rel(p_heads_1, p_heads_2)
    else:
        t_stat, p_value = 0.0, 1.0

    return {
        "n_common_points": len(common_t),
        "mean_difference": float(np.mean(diffs)),
        "mean_abs_difference": float(np.mean(abs_diffs)),
        "max_abs_difference": float(np.max(abs_diffs)),
        "paired_t_statistic": float(t_stat),
        "paired_p_value": float(p_value),
        "significantly_different": bool(p_value < 0.05)
    }


# =============================================================================
# Q4: Bias Sensitivity and Switchover Analysis
# =============================================================================

def compute_switchover_point(result: TrajectoryResult) -> dict:
    """
    Find when the model's belief crosses 0.5 (for biased sequences p≠0.5).

    Args:
        result: TrajectoryResult

    Returns:
        Dictionary with switchover analysis
    """
    if not result.polls:
        return {}

    p_true = result.config.p
    prior = result.polls[0].p_heads if result.polls[0].t == 0 else 0.5

    # For p < 0.5, we expect P(heads) to decrease from prior
    # For p > 0.5, we expect P(heads) to increase from prior

    # Find first crossing of 0.5
    switchover_t = None
    switchover_bayesian_t = None

    prev_p = prior
    for poll in result.polls[1:]:
        curr_p = poll.p_heads
        # Check if crossed 0.5
        if (prev_p < 0.5 and curr_p >= 0.5) or (prev_p > 0.5 and curr_p <= 0.5):
            switchover_t = poll.t
            break
        prev_p = curr_p

    # Also find when Bayesian crosses 0.5
    prev_b = 0.5  # Bayesian starts at 0.5
    for poll in result.polls:
        curr_b = poll.bayesian_posterior
        if (prev_b < 0.5 and curr_b >= 0.5) or (prev_b > 0.5 and curr_b <= 0.5):
            switchover_bayesian_t = poll.t
            break
        prev_b = curr_b

    # Final belief direction
    final_p = result.polls[-1].p_heads
    correct_side = None
    if p_true < 0.5:
        correct_side = bool(final_p < 0.5)
    elif p_true > 0.5:
        correct_side = bool(final_p > 0.5)
    else:
        correct_side = True  # p=0.5, any side is fine

    return {
        "true_p": p_true,
        "prior_p_heads": float(prior),
        "prior_side": "heads" if prior > 0.5 else ("tails" if prior < 0.5 else "neutral"),
        "switchover_t": switchover_t,
        "switchover_bayesian_t": switchover_bayesian_t,
        "final_p_heads": float(final_p),
        "final_on_correct_side": correct_side,
        "switchover_delay": (switchover_t - switchover_bayesian_t) if (switchover_t and switchover_bayesian_t) else None
    }


def compute_update_asymmetry(result: TrajectoryResult) -> dict:
    """
    Compute whether the model updates asymmetrically for heads vs tails evidence.

    Args:
        result: TrajectoryResult

    Returns:
        Dictionary with asymmetry metrics
    """
    if not result.polls or len(result.polls) < 2:
        return {}

    k = result.config.k

    # Only meaningful for k=1 (single observation updates)
    if k != 1:
        return {"note": "Asymmetry analysis only valid for k=1"}

    heads_updates = []  # Updates after seeing heads
    tails_updates = []  # Updates after seeing tails

    for i in range(1, len(result.polls)):
        prev_poll = result.polls[i-1]
        curr_poll = result.polls[i]

        # The observation that caused this update
        obs_t = prev_poll.t
        if obs_t < len(result.sequence):
            observation = result.sequence[obs_t]
            update = curr_poll.p_heads - prev_poll.p_heads

            if observation == "heads":
                heads_updates.append(update)
            else:
                tails_updates.append(update)

    if not heads_updates or not tails_updates:
        return {}

    # Heads should cause positive updates, tails should cause negative
    mean_heads_update = float(np.mean(heads_updates))
    mean_tails_update = float(np.mean(tails_updates))

    # Asymmetry: are the magnitudes different?
    mean_heads_magnitude = float(np.mean(np.abs(heads_updates)))
    mean_tails_magnitude = float(np.mean(np.abs(tails_updates)))

    # Ratio > 1 means heads evidence has stronger effect
    asymmetry_ratio = mean_heads_magnitude / mean_tails_magnitude if mean_tails_magnitude > 1e-6 else float('inf')

    return {
        "mean_heads_update": mean_heads_update,
        "mean_tails_update": mean_tails_update,
        "mean_heads_magnitude": mean_heads_magnitude,
        "mean_tails_magnitude": mean_tails_magnitude,
        "asymmetry_ratio": float(asymmetry_ratio) if asymmetry_ratio != float('inf') else None,
        "heads_updates_correct_sign": bool(mean_heads_update > 0),
        "tails_updates_correct_sign": bool(mean_tails_update < 0)
    }


def compute_bias_sensitivity(result: TrajectoryResult) -> dict:
    """
    Compute whether the model is more sensitive to evidence that contradicts its prior.

    Args:
        result: TrajectoryResult

    Returns:
        Dictionary with bias sensitivity metrics
    """
    if not result.polls or len(result.polls) < 2 or result.config.k != 1:
        return {}

    prior = result.polls[0].p_heads if result.polls[0].t == 0 else 0.5
    prior_favors_heads = prior > 0.5

    confirming_updates = []  # Evidence that confirms prior
    contradicting_updates = []  # Evidence that contradicts prior

    for i in range(1, len(result.polls)):
        prev_poll = result.polls[i-1]
        curr_poll = result.polls[i]

        obs_t = prev_poll.t
        if obs_t < len(result.sequence):
            observation = result.sequence[obs_t]
            update_magnitude = abs(curr_poll.p_heads - prev_poll.p_heads)

            # Does this evidence confirm or contradict the prior?
            evidence_favors_heads = (observation == "heads")

            if evidence_favors_heads == prior_favors_heads:
                confirming_updates.append(update_magnitude)
            else:
                contradicting_updates.append(update_magnitude)

    if not confirming_updates or not contradicting_updates:
        return {}

    mean_confirming = float(np.mean(confirming_updates))
    mean_contradicting = float(np.mean(contradicting_updates))

    # Sensitivity ratio > 1 means more sensitive to contradicting evidence
    sensitivity_ratio = mean_contradicting / mean_confirming if mean_confirming > 1e-6 else float('inf')

    return {
        "prior_p_heads": prior,
        "prior_favors": "heads" if prior_favors_heads else "tails",
        "mean_confirming_update": mean_confirming,
        "mean_contradicting_update": mean_contradicting,
        "n_confirming": len(confirming_updates),
        "n_contradicting": len(contradicting_updates),
        "sensitivity_ratio": float(sensitivity_ratio) if sensitivity_ratio != float('inf') else None,
        "more_sensitive_to_contradicting": bool(sensitivity_ratio > 1.0)
    }

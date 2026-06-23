"""
Results aggregation and analysis for coin flip experiments.

Loads experiment results, computes aggregate statistics, and performs comparisons.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np

from .config import TrajectoryResult, Condition, CoinSpec
from .metrics import (
    bootstrap_ci,
    compute_self_anchoring_metrics,
    compute_paired_self_anchoring,
    compute_icrl_metrics,
    compute_prior_bias,
    compute_updating_exists,
    compute_update_direction_accuracy,
    compute_trajectory_divergence,
    compute_switchover_point,
    compute_update_asymmetry,
    compute_bias_sensitivity
)


def load_all_results(
    experiments_dir: Path,
    model_filter: Optional[str] = None,
    coin_spec_filter: Optional[str] = None
) -> List[TrajectoryResult]:
    """
    Load all experiment results from directory.

    Args:
        experiments_dir: Directory containing result JSON files
        model_filter: Optional model name to filter by
        coin_spec_filter: Optional coin spec to filter by

    Returns:
        List of TrajectoryResults
    """
    results = []

    for json_file in experiments_dir.glob("*.json"):
        if json_file.name in [".gitkeep"] or json_file.name.endswith("_analysis.json"):
            continue

        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            result = TrajectoryResult.from_dict(data)

            # Apply model filter if specified
            if model_filter and result.config.model_name != model_filter:
                continue

            # Apply coin_spec filter if specified
            if coin_spec_filter and result.config.coin_spec.value != coin_spec_filter:
                continue

            results.append(result)

        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue

    print(f"Loaded {len(results)} experiment results")
    return results


def group_results(
    results: List[TrajectoryResult]
) -> Dict[str, Dict[str, List[TrajectoryResult]]]:
    """
    Group results by model, condition, k, p, and coin_spec.

    Returns:
        Nested dict: model -> key -> list of results
    """
    grouped = defaultdict(lambda: defaultdict(list))

    for result in results:
        model = result.config.model_name
        coin_spec = result.config.coin_spec.value
        key = f"{result.config.condition.value}_k{result.config.k}_p{result.config.p}_{coin_spec}"
        grouped[model][key].append(result)

    return dict(grouped)


def aggregate_by_condition(results: List[TrajectoryResult]) -> Dict[str, Any]:
    """
    Compute aggregate statistics for a group of results (same condition, k, p).

    Args:
        results: List of TrajectoryResults with same condition/k/p

    Returns:
        Dictionary with aggregated metrics and confidence intervals
    """
    if not results:
        return {}

    # Collect metrics across trials
    mae_values = []
    corr_values = []
    prior_values = []
    update_values = []
    variance_values = []
    pos_bias_values = []
    ab_mass_values = []

    for r in results:
        if r.metrics:
            mae_values.append(r.metrics.mae_from_bayesian)
            corr_values.append(r.metrics.correlation_with_bayesian)
            prior_values.append(r.metrics.prior_p_heads)
            update_values.append(r.metrics.mean_update_magnitude)
            variance_values.append(r.metrics.trajectory_variance)
            pos_bias_values.append(r.metrics.mean_position_bias)
            ab_mass_values.append(r.metrics.mean_ab_mass)

    # Compute bootstrap CIs
    mae_mean, mae_lo, mae_hi = bootstrap_ci(mae_values)
    corr_mean, corr_lo, corr_hi = bootstrap_ci(corr_values)
    prior_mean, prior_lo, prior_hi = bootstrap_ci(prior_values)
    update_mean, update_lo, update_hi = bootstrap_ci(update_values)
    var_mean, var_lo, var_hi = bootstrap_ci(variance_values)

    return {
        "n_trials": len(results),
        "condition": results[0].config.condition.value,
        "k": results[0].config.k,
        "p": results[0].config.p,
        "coin_spec": results[0].config.coin_spec.value,
        "mae_from_bayesian": {
            "mean": mae_mean,
            "ci_lower": mae_lo,
            "ci_upper": mae_hi
        },
        "correlation_with_bayesian": {
            "mean": corr_mean,
            "ci_lower": corr_lo,
            "ci_upper": corr_hi
        },
        "prior_p_heads": {
            "mean": prior_mean,
            "ci_lower": prior_lo,
            "ci_upper": prior_hi
        },
        "mean_update_magnitude": {
            "mean": update_mean,
            "ci_lower": update_lo,
            "ci_upper": update_hi
        },
        "trajectory_variance": {
            "mean": var_mean,
            "ci_lower": var_lo,
            "ci_upper": var_hi
        },
        "mean_position_bias": float(np.mean(pos_bias_values)) if pos_bias_values else 0.0,
        "mean_ab_mass": float(np.mean(ab_mass_values)) if ab_mass_values else 0.0
    }


def compare_single_vs_multi(
    grouped: Dict[str, Dict[str, List[TrajectoryResult]]],
    model: str,
    k: int,
    p: float = 0.5,
    coin_spec: str = "unspecified"
) -> Dict[str, Any]:
    """
    Compare single-turn vs multi-turn conditions for a specific setting.

    Args:
        grouped: Grouped results from group_results()
        model: Model name
        k: Polling frequency
        p: True probability
        coin_spec: Coin specification

    Returns:
        Comparison metrics
    """
    if model not in grouped:
        return {"error": f"Model {model} not found"}

    model_results = grouped[model]

    # Get results for each condition
    single_key = f"single_turn_k{k}_p{p}_{coin_spec}"
    minimal_key = f"multi_turn_minimal_k{k}_p{p}_{coin_spec}"
    actual_key = f"multi_turn_actual_k{k}_p{p}_{coin_spec}"

    comparison = {
        "model": model,
        "k": k,
        "p": p,
        "coin_spec": coin_spec,
        "conditions": {}
    }

    for key, name in [
        (single_key, "single_turn"),
        (minimal_key, "multi_turn_minimal"),
        (actual_key, "multi_turn_actual")
    ]:
        if key in model_results:
            comparison["conditions"][name] = aggregate_by_condition(model_results[key])

    # Compute differential metrics
    if "single_turn" in comparison["conditions"] and "multi_turn_minimal" in comparison["conditions"]:
        single_var = comparison["conditions"]["single_turn"]["trajectory_variance"]["mean"]
        multi_var = comparison["conditions"]["multi_turn_minimal"]["trajectory_variance"]["mean"]
        comparison["variance_difference"] = multi_var - single_var

    if "multi_turn_actual" in comparison["conditions"]:
        # Self-anchoring metrics
        actual_results = model_results.get(actual_key, [])
        comparison["self_anchoring"] = compute_self_anchoring_metrics(actual_results)

    return comparison


def compute_self_anchoring_index(
    grouped: Dict[str, Dict[str, List[TrajectoryResult]]],
    model: str,
    k: int,
    coin_spec: str = "unspecified"
) -> Dict[str, Any]:
    """
    Compute self-anchoring index across p values.

    Compares trajectory variance between conditions with and without predictions.

    Args:
        grouped: Grouped results
        model: Model name
        k: Polling frequency
        coin_spec: Coin specification

    Returns:
        Self-anchoring index and component metrics
    """
    if model not in grouped:
        return {"error": f"Model {model} not found"}

    model_results = grouped[model]

    # Collect variances by condition across p values
    condition_variances = defaultdict(list)

    for key, results in model_results.items():
        if f"_k{k}_" in key and key.endswith(f"_{coin_spec}"):
            agg = aggregate_by_condition(results)
            if agg and "trajectory_variance" in agg:
                condition = agg["condition"]
                condition_variances[condition].append(agg["trajectory_variance"]["mean"])

    # Compute mean variance for each condition
    mean_variances = {
        cond: float(np.mean(vars)) for cond, vars in condition_variances.items()
    }

    # Self-anchoring index: reduction in variance when predictions are in context
    sai = None
    if "multi_turn_minimal" in mean_variances and "multi_turn_actual" in mean_variances:
        baseline = mean_variances["multi_turn_minimal"]
        actual = mean_variances["multi_turn_actual"]
        if baseline > 0:
            sai = (baseline - actual) / baseline
        else:
            sai = 0.0

    return {
        "model": model,
        "k": k,
        "coin_spec": coin_spec,
        "mean_variances_by_condition": mean_variances,
        "self_anchoring_index": sai
    }


def analyze_q1_updating(
    grouped: Dict[str, Dict[str, List[TrajectoryResult]]],
    model: str
) -> Dict[str, Any]:
    """
    Q1 Analysis: Does the LLM update its prior in response to sequences?

    Args:
        grouped: Grouped results
        model: Model name

    Returns:
        Dictionary with Q1 analysis results
    """
    if model not in grouped:
        return {"error": f"Model {model} not found"}

    model_results = grouped[model]
    analysis = {
        "by_condition": {},
        "overall": {}
    }

    # Analyze each condition separately
    all_results = []
    for key, results in model_results.items():
        condition = results[0].config.condition.value if results else "unknown"

        # Prior bias analysis
        prior_bias = compute_prior_bias(results)

        # Updating exists analysis
        updating_exists = compute_updating_exists(results)

        # Direction accuracy (aggregate across trials)
        direction_accuracies = []
        for r in results:
            dir_acc = compute_update_direction_accuracy(r)
            if dir_acc and "direction_accuracy" in dir_acc:
                direction_accuracies.append(dir_acc["direction_accuracy"])

        mean_dir_acc, dir_acc_lo, dir_acc_hi = bootstrap_ci(direction_accuracies) if direction_accuracies else (0, 0, 0)

        analysis["by_condition"][key] = {
            "condition": condition,
            "n_trials": len(results),
            "prior_bias": prior_bias,
            "updating_exists": updating_exists,
            "direction_accuracy": {
                "mean": mean_dir_acc,
                "ci_lower": dir_acc_lo,
                "ci_upper": dir_acc_hi
            }
        }

        all_results.extend(results)

    # Overall analysis across all conditions
    analysis["overall"]["prior_bias"] = compute_prior_bias(all_results)
    analysis["overall"]["updating_exists"] = compute_updating_exists(all_results)

    return analysis


def analyze_q2_single_vs_multi(
    grouped: Dict[str, Dict[str, List[TrajectoryResult]]],
    model: str
) -> Dict[str, Any]:
    """
    Q2 Analysis: Do single-turn and multi-turn produce equivalent updates?

    Pairs experiments by (p, trial) and compares trajectories.

    Args:
        grouped: Grouped results
        model: Model name

    Returns:
        Dictionary with Q2 analysis results
    """
    if model not in grouped:
        return {"error": f"Model {model} not found"}

    model_results = grouped[model]
    analysis = {
        "paired_comparisons": {},
        "summary": {}
    }

    # Build index by (p, trial, k) -> condition -> result
    result_index = defaultdict(dict)
    for key, results in model_results.items():
        for r in results:
            idx_key = (r.config.p, r.config.trial, r.config.k)
            result_index[idx_key][r.config.condition.value] = r

    # Compare single_turn vs multi_turn_minimal (the clean comparison)
    divergences = []
    paired_results = []

    for idx_key, condition_map in result_index.items():
        p, trial, k = idx_key

        if "single_turn" in condition_map and "multi_turn_minimal" in condition_map:
            single = condition_map["single_turn"]
            multi = condition_map["multi_turn_minimal"]

            div = compute_trajectory_divergence(single, multi)
            if div:
                div["p"] = p
                div["trial"] = trial
                div["k"] = k
                divergences.append(div)
                paired_results.append((single, multi))

    analysis["paired_comparisons"]["single_vs_multi_minimal"] = divergences

    # Summary statistics
    if divergences:
        mae_values = [d["mean_abs_difference"] for d in divergences]
        mean_mae, mae_lo, mae_hi = bootstrap_ci(mae_values)

        sig_different = sum(1 for d in divergences if d.get("significantly_different", False))

        analysis["summary"]["single_vs_multi_minimal"] = {
            "n_pairs": len(divergences),
            "mean_trajectory_mae": {
                "mean": mean_mae,
                "ci_lower": mae_lo,
                "ci_upper": mae_hi
            },
            "n_significantly_different": sig_different,
            "pct_significantly_different": sig_different / len(divergences) * 100
        }

    # Also compare multi_turn_minimal vs multi_turn_actual (effect of predictions)
    divergences_pred = []
    for idx_key, condition_map in result_index.items():
        p, trial, k = idx_key

        if "multi_turn_minimal" in condition_map and "multi_turn_actual" in condition_map:
            minimal = condition_map["multi_turn_minimal"]
            actual = condition_map["multi_turn_actual"]

            div = compute_trajectory_divergence(minimal, actual)
            if div:
                div["p"] = p
                div["trial"] = trial
                div["k"] = k
                divergences_pred.append(div)

    analysis["paired_comparisons"]["minimal_vs_actual"] = divergences_pred

    if divergences_pred:
        mae_values = [d["mean_abs_difference"] for d in divergences_pred]
        mean_mae, mae_lo, mae_hi = bootstrap_ci(mae_values)

        sig_different = sum(1 for d in divergences_pred if d.get("significantly_different", False))

        analysis["summary"]["minimal_vs_actual"] = {
            "n_pairs": len(divergences_pred),
            "mean_trajectory_mae": {
                "mean": mean_mae,
                "ci_lower": mae_lo,
                "ci_upper": mae_hi
            },
            "n_significantly_different": sig_different,
            "pct_significantly_different": sig_different / len(divergences_pred) * 100
        }

    # Paired self-anchoring: per-poll delta decomposed by prediction direction/correctness
    actual_results_all = []
    minimal_results_all = []
    for idx_key, condition_map in result_index.items():
        if "multi_turn_actual" in condition_map:
            actual_results_all.append(condition_map["multi_turn_actual"])
        if "multi_turn_minimal" in condition_map:
            minimal_results_all.append(condition_map["multi_turn_minimal"])

    analysis["self_anchoring"] = compute_paired_self_anchoring(
        actual_results_all, minimal_results_all
    )

    analysis["icrl"] = compute_icrl_metrics(actual_results_all, minimal_results_all)

    return analysis



def analyze_polling_frequency(
    grouped: Dict[str, Dict[str, List[TrajectoryResult]]],
    model: str
) -> Dict[str, Any]:
    """
    Compare k=1 vs k=5 polling frequency effects.

    Args:
        grouped: Grouped results
        model: Model name

    Returns:
        Dictionary with k=1 vs k=5 comparison
    """
    if model not in grouped:
        return {"error": f"Model {model} not found"}

    model_results = grouped[model]
    analysis = {
        "by_condition": {},
        "summary": {}
    }

    # Collect metrics by condition and k
    condition_k_metrics = defaultdict(lambda: defaultdict(lambda: {
        "mae": [], "corr": [], "variance": [], "update_mag": []
    }))

    for key, results in model_results.items():
        for r in results:
            cond = r.config.condition.value
            k = r.config.k
            if r.metrics:
                condition_k_metrics[cond][k]["mae"].append(r.metrics.mae_from_bayesian)
                condition_k_metrics[cond][k]["corr"].append(r.metrics.correlation_with_bayesian)
                condition_k_metrics[cond][k]["variance"].append(r.metrics.trajectory_variance)
                condition_k_metrics[cond][k]["update_mag"].append(r.metrics.mean_update_magnitude)

    # Compare k=1 vs k=5 for each condition
    for cond, k_data in condition_k_metrics.items():
        if 1 in k_data and 5 in k_data:
            k1_mae = np.mean(k_data[1]["mae"])
            k5_mae = np.mean(k_data[5]["mae"])
            k1_corr = np.mean(k_data[1]["corr"])
            k5_corr = np.mean(k_data[5]["corr"])
            k1_var = np.mean(k_data[1]["variance"])
            k5_var = np.mean(k_data[5]["variance"])

            analysis["by_condition"][cond] = {
                "k1_mae": float(k1_mae),
                "k5_mae": float(k5_mae),
                "mae_difference": float(k5_mae - k1_mae),
                "k1_correlation": float(k1_corr),
                "k5_correlation": float(k5_corr),
                "correlation_difference": float(k5_corr - k1_corr),
                "k1_variance": float(k1_var),
                "k5_variance": float(k5_var),
                "k5_more_stable": bool(k5_var < k1_var)
            }

    # Overall summary
    all_k1_mae = []
    all_k5_mae = []
    all_k1_corr = []
    all_k5_corr = []

    for cond_data in analysis["by_condition"].values():
        all_k1_mae.append(cond_data["k1_mae"])
        all_k5_mae.append(cond_data["k5_mae"])
        all_k1_corr.append(cond_data["k1_correlation"])
        all_k5_corr.append(cond_data["k5_correlation"])

    if all_k1_mae:
        analysis["summary"] = {
            "mean_k1_mae": float(np.mean(all_k1_mae)),
            "mean_k5_mae": float(np.mean(all_k5_mae)),
            "k5_better_mae": bool(np.mean(all_k5_mae) < np.mean(all_k1_mae)),
            "mean_k1_correlation": float(np.mean(all_k1_corr)),
            "mean_k5_correlation": float(np.mean(all_k5_corr)),
            "k5_better_correlation": bool(np.mean(all_k5_corr) > np.mean(all_k1_corr))
        }

    return analysis


def analyze_per_p(
    grouped: Dict[str, Dict[str, List[TrajectoryResult]]],
    model: str
) -> Dict[str, Any]:
    """
    Breakdown analysis by true probability p.

    Args:
        grouped: Grouped results
        model: Model name

    Returns:
        Dictionary with per-p analysis
    """
    if model not in grouped:
        return {"error": f"Model {model} not found"}

    model_results = grouped[model]
    analysis = {"by_p": {}}

    for p in [0.25, 0.5, 0.75]:
        p_analysis = {
            "by_condition": {},
            "overall": {}
        }

        p_metrics = defaultdict(lambda: {
            "mae": [], "corr": [], "prior": [], "final": [], "final_bayesian": []
        })

        for key, results in model_results.items():
            for r in results:
                if r.config.p == p and r.metrics and r.polls:
                    cond = r.config.condition.value
                    p_metrics[cond]["mae"].append(r.metrics.mae_from_bayesian)
                    p_metrics[cond]["corr"].append(r.metrics.correlation_with_bayesian)
                    p_metrics[cond]["prior"].append(r.metrics.prior_p_heads)
                    p_metrics[cond]["final"].append(r.polls[-1].p_heads)
                    p_metrics[cond]["final_bayesian"].append(r.polls[-1].bayesian_posterior)

        for cond, metrics in p_metrics.items():
            if metrics["mae"]:
                mae_mean, mae_lo, mae_hi = bootstrap_ci(metrics["mae"])
                corr_mean, corr_lo, corr_hi = bootstrap_ci(metrics["corr"])

                # How far is final from true p?
                final_errors = [abs(f - p) for f in metrics["final"]]
                fe_mean, fe_lo, fe_hi = bootstrap_ci(final_errors)

                # Direction: does model end up on correct side?
                if p < 0.5:
                    correct_side = sum(1 for f in metrics["final"] if f < 0.5) / len(metrics["final"])
                elif p > 0.5:
                    correct_side = sum(1 for f in metrics["final"] if f > 0.5) / len(metrics["final"])
                else:
                    correct_side = 1.0

                p_analysis["by_condition"][cond] = {
                    "n_trials": len(metrics["mae"]),
                    "mae_from_bayesian": {"mean": mae_mean, "ci_lower": mae_lo, "ci_upper": mae_hi},
                    "correlation": {"mean": corr_mean, "ci_lower": corr_lo, "ci_upper": corr_hi},
                    "mean_prior": float(np.mean(metrics["prior"])),
                    "mean_final": float(np.mean(metrics["final"])),
                    "final_error_from_true_p": {"mean": fe_mean, "ci_lower": fe_lo, "ci_upper": fe_hi},
                    "pct_correct_side": float(correct_side * 100)
                }

        # Overall for this p
        all_mae = []
        all_corr = []
        all_final = []
        for cond_data in p_analysis["by_condition"].values():
            all_mae.append(cond_data["mae_from_bayesian"]["mean"])
            all_corr.append(cond_data["correlation"]["mean"])
            all_final.append(cond_data["mean_final"])

        if all_mae:
            p_analysis["overall"] = {
                "true_p": p,
                "mean_mae": float(np.mean(all_mae)),
                "mean_correlation": float(np.mean(all_corr)),
                "mean_final_belief": float(np.mean(all_final)),
                "belief_error": float(abs(np.mean(all_final) - p))
            }

        analysis["by_p"][str(p)] = p_analysis

    return analysis


def analyze_diagnostics(
    grouped: Dict[str, Dict[str, List[TrajectoryResult]]],
    model: str
) -> Dict[str, Any]:
    """
    Analyze position bias and A/B mass diagnostics.

    Args:
        grouped: Grouped results
        model: Model name

    Returns:
        Dictionary with diagnostic analysis
    """
    if model not in grouped:
        return {"error": f"Model {model} not found"}

    model_results = grouped[model]
    analysis = {
        "position_bias": {"by_condition": {}},
        "ab_mass": {"by_condition": {}},
        "overall": {}
    }

    condition_diagnostics = defaultdict(lambda: {"pos_bias": [], "ab_mass": []})

    for key, results in model_results.items():
        for r in results:
            cond = r.config.condition.value
            if r.metrics:
                condition_diagnostics[cond]["pos_bias"].append(r.metrics.mean_position_bias)
                condition_diagnostics[cond]["ab_mass"].append(r.metrics.mean_ab_mass)

    all_pos_bias = []
    all_ab_mass = []

    for cond, diag in condition_diagnostics.items():
        if diag["pos_bias"]:
            pb_mean, pb_lo, pb_hi = bootstrap_ci(diag["pos_bias"])
            ab_mean, ab_lo, ab_hi = bootstrap_ci(diag["ab_mass"])

            analysis["position_bias"]["by_condition"][cond] = {
                "mean": pb_mean,
                "ci_lower": pb_lo,
                "ci_upper": pb_hi,
                "significant_bias": bool(abs(pb_mean) > 0.1)
            }

            analysis["ab_mass"]["by_condition"][cond] = {
                "mean": ab_mean,
                "ci_lower": ab_lo,
                "ci_upper": ab_hi,
                "good_format_following": bool(ab_mean > 0.9)
            }

            all_pos_bias.extend(diag["pos_bias"])
            all_ab_mass.extend(diag["ab_mass"])

    if all_pos_bias:
        analysis["overall"] = {
            "mean_position_bias": float(np.mean(all_pos_bias)),
            "mean_ab_mass": float(np.mean(all_ab_mass)),
            "position_bias_concern": bool(abs(np.mean(all_pos_bias)) > 0.1),
            "format_following_good": bool(np.mean(all_ab_mass) > 0.9)
        }

    return analysis


def analyze_q4_bias_sensitivity(
    grouped: Dict[str, Dict[str, List[TrajectoryResult]]],
    model: str
) -> Dict[str, Any]:
    """
    Q4 Analysis: Does the LLM overindex on sequences counter to its bias?

    Includes switchover analysis and update asymmetry.

    Args:
        grouped: Grouped results
        model: Model name

    Returns:
        Dictionary with Q4 analysis results
    """
    if model not in grouped:
        return {"error": f"Model {model} not found"}

    model_results = grouped[model]
    analysis = {
        "switchover": {"by_p": {}},
        "update_asymmetry": {"by_condition": {}},
        "bias_sensitivity": {"by_condition": {}}
    }

    # Switchover analysis by true p value
    for p in [0.25, 0.5, 0.75]:
        switchovers = []
        for key, results in model_results.items():
            for r in results:
                if r.config.p == p:
                    sw = compute_switchover_point(r)
                    if sw:
                        sw["condition"] = r.config.condition.value
                        sw["k"] = r.config.k
                        sw["trial"] = r.config.trial
                        switchovers.append(sw)

        if switchovers:
            # Aggregate switchover points
            sw_times = [s["switchover_t"] for s in switchovers if s.get("switchover_t") is not None]
            final_correct = sum(1 for s in switchovers if s.get("final_on_correct_side", False))

            analysis["switchover"]["by_p"][str(p)] = {
                "n_experiments": len(switchovers),
                "n_with_switchover": len(sw_times),
                "mean_switchover_t": float(np.mean(sw_times)) if sw_times else None,
                "median_switchover_t": float(np.median(sw_times)) if sw_times else None,
                "final_on_correct_side": final_correct,
                "pct_correct_side": final_correct / len(switchovers) * 100
            }

    # Update asymmetry analysis (k=1 only)
    asymmetries = defaultdict(list)
    for key, results in model_results.items():
        for r in results:
            if r.config.k == 1:
                asym = compute_update_asymmetry(r)
                if asym and "asymmetry_ratio" in asym:
                    condition = r.config.condition.value
                    asymmetries[condition].append(asym)

    for condition, asym_list in asymmetries.items():
        ratios = [a["asymmetry_ratio"] for a in asym_list if a["asymmetry_ratio"] is not None and a["asymmetry_ratio"] != float('inf')]
        heads_correct = sum(1 for a in asym_list if a.get("heads_updates_correct_sign", False))
        tails_correct = sum(1 for a in asym_list if a.get("tails_updates_correct_sign", False))

        if ratios:
            mean_ratio, ratio_lo, ratio_hi = bootstrap_ci(ratios)
            analysis["update_asymmetry"]["by_condition"][condition] = {
                "n_experiments": len(asym_list),
                "mean_asymmetry_ratio": {
                    "mean": mean_ratio,
                    "ci_lower": ratio_lo,
                    "ci_upper": ratio_hi
                },
                "heads_correct_sign_pct": heads_correct / len(asym_list) * 100,
                "tails_correct_sign_pct": tails_correct / len(asym_list) * 100
            }

    # Bias sensitivity analysis (k=1 only)
    sensitivities = defaultdict(list)
    for key, results in model_results.items():
        for r in results:
            if r.config.k == 1:
                sens = compute_bias_sensitivity(r)
                if sens and "sensitivity_ratio" in sens:
                    condition = r.config.condition.value
                    sensitivities[condition].append(sens)

    for condition, sens_list in sensitivities.items():
        ratios = [s["sensitivity_ratio"] for s in sens_list if s["sensitivity_ratio"] is not None and s["sensitivity_ratio"] != float('inf')]
        more_sensitive = sum(1 for s in sens_list if s.get("more_sensitive_to_contradicting", False))

        if ratios:
            mean_ratio, ratio_lo, ratio_hi = bootstrap_ci(ratios)
            analysis["bias_sensitivity"]["by_condition"][condition] = {
                "n_experiments": len(sens_list),
                "mean_sensitivity_ratio": {
                    "mean": mean_ratio,
                    "ci_lower": ratio_lo,
                    "ci_upper": ratio_hi
                },
                "pct_more_sensitive_to_contradicting": more_sensitive / len(sens_list) * 100
            }

    return analysis


def generate_summary_report(
    results: List[TrajectoryResult],
    model: str
) -> Dict[str, Any]:
    """
    Generate comprehensive summary report for a model.

    Args:
        results: All results for the model
        model: Model name

    Returns:
        Summary report dictionary
    """
    grouped = group_results(results)

    if model not in grouped:
        return {"error": f"Model {model} not found"}

    report = {
        "model": model,
        "total_experiments": len(results),
        "comparisons": {},
        "self_anchoring_indices": {},
        "q1_updating_analysis": {},
        "q2_single_vs_multi_analysis": {},
        "q4_bias_sensitivity_analysis": {},
        "per_p_analysis": {},
        "diagnostics": {}
    }

    # Get unique k, p, and coin_spec values
    k_values = set()
    p_values = set()
    coin_specs = set()
    for result in results:
        if result.config.model_name == model:
            k_values.add(result.config.k)
            p_values.add(result.config.p)
            coin_specs.add(result.config.coin_spec.value)

    # Generate comparisons per coin_spec
    for cs in sorted(coin_specs):
        for k in sorted(k_values):
            for p in sorted(p_values):
                key = f"k{k}_p{p}_{cs}"
                report["comparisons"][key] = compare_single_vs_multi(grouped, model, k, p, coin_spec=cs)

    # Self-anchoring indices per coin_spec
    for cs in sorted(coin_specs):
        for k in sorted(k_values):
            report["self_anchoring_indices"][f"k{k}_{cs}"] = compute_self_anchoring_index(grouped, model, k, coin_spec=cs)

    # Q1 - Updating analysis
    report["q1_updating_analysis"] = analyze_q1_updating(grouped, model)

    # Q2 - Single vs Multi-turn comparison
    report["q2_single_vs_multi_analysis"] = analyze_q2_single_vs_multi(grouped, model)

    # Q4 - Bias sensitivity analysis
    report["q4_bias_sensitivity_analysis"] = analyze_q4_bias_sensitivity(grouped, model)

    # Per-p breakdown
    report["per_p_analysis"] = analyze_per_p(grouped, model)

    # Diagnostics (position bias, A/B mass)
    report["diagnostics"] = analyze_diagnostics(grouped, model)

    return report


def main():
    parser = argparse.ArgumentParser(description="Aggregate Coin Flip Experiment Results")
    parser.add_argument("--experiments-dir", type=str,
                        default=str(Path(__file__).parent / "experiments"),
                        help="Directory with results")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter by model")
    parser.add_argument("--coin-spec", type=str, default=None,
                        choices=[c.value for c in CoinSpec],
                        help="Filter by coin specification")

    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)

    # Load results
    results = load_all_results(experiments_dir, model_filter=args.model,
                               coin_spec_filter=args.coin_spec)

    if not results:
        print("No results found!")
        return

    # Group results
    grouped = group_results(results)

    # Generate output
    output = {
        "experiments_dir": str(experiments_dir),
        "total_experiments": len(results),
        "models": {}
    }

    for model in grouped.keys():
        if args.model and model != args.model:
            continue

        model_results = [r for r in results if r.config.model_name == model]
        output["models"][model] = generate_summary_report(model_results, model)

    # Also add raw aggregations by condition for easier access
    output["aggregations"] = {}
    for model, conditions in grouped.items():
        if args.model and model != args.model:
            continue

        output["aggregations"][model] = {}
        for key, condition_results in conditions.items():
            output["aggregations"][model][key] = aggregate_by_condition(condition_results)

    # Save output
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved aggregated results to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 70)

    for model, model_data in output["models"].items():
        print(f"\n{'='*70}")
        print(f"MODEL: {model}")
        print(f"{'='*70}")
        print(f"Total experiments: {model_data['total_experiments']}")

        # Q1: Updating Analysis
        print(f"\n--- Q1: Does the LLM Update Its Prior? ---")
        q1 = model_data.get("q1_updating_analysis", {})
        if "overall" in q1:
            overall = q1["overall"]
            if "prior_bias" in overall:
                pb = overall["prior_bias"]
                print(f"  Prior P(heads): {pb.get('mean_prior', 0):.3f} ± {pb.get('std_prior', 0):.3f}")
                print(f"  Prior bias: {pb.get('bias_direction', 'unknown')} (p={pb.get('p_value', 1):.4f})")
            if "updating_exists" in overall:
                ue = overall["updating_exists"]
                exists = "YES" if ue.get("updating_exists", False) else "NO"
                print(f"  Updating exists: {exists} (mean change: {ue.get('mean_prior_to_final_change', 0):.3f})")
                print(f"  Final error from Bayesian: {ue.get('mean_final_error_from_bayesian', 0):.3f}")

        # Q2: Single vs Multi-turn
        print(f"\n--- Q2: Single-Turn vs Multi-Turn Equivalence ---")
        q2 = model_data.get("q2_single_vs_multi_analysis", {})
        if "summary" in q2:
            if "single_vs_multi_minimal" in q2["summary"]:
                sm = q2["summary"]["single_vs_multi_minimal"]
                print(f"  Single vs Multi-Minimal:")
                print(f"    Trajectory MAE: {sm['mean_trajectory_mae']['mean']:.4f}")
                print(f"    Significantly different: {sm['pct_significantly_different']:.1f}% of pairs")
            if "minimal_vs_actual" in q2["summary"]:
                ma = q2["summary"]["minimal_vs_actual"]
                print(f"  Multi-Minimal vs Multi-Actual (effect of predictions):")
                print(f"    Trajectory MAE: {ma['mean_trajectory_mae']['mean']:.4f}")
                print(f"    Significantly different: {ma['pct_significantly_different']:.1f}% of pairs")

        # Q3: Self-anchoring
        print(f"\n--- Q3: Self-Anchoring ---")
        if "self_anchoring_indices" in model_data:
            for sai_key, sai_data in model_data["self_anchoring_indices"].items():
                if sai_data.get("self_anchoring_index") is not None:
                    sai = sai_data["self_anchoring_index"]
                    interp = "strong anchoring" if sai > 0.5 else "moderate anchoring" if sai > 0 else "anti-anchoring"
                    print(f"  {sai_key}: SAI = {sai:.4f} ({interp})")

        # Q4: Bias Sensitivity
        print(f"\n--- Q4: Bias Sensitivity & Switchover ---")
        q4 = model_data.get("q4_bias_sensitivity_analysis", {})
        if "switchover" in q4 and "by_p" in q4["switchover"]:
            for p_str, sw_data in q4["switchover"]["by_p"].items():
                if sw_data.get("mean_switchover_t") is not None:
                    print(f"  p={p_str}: switchover at t={sw_data['mean_switchover_t']:.1f}, " +
                          f"{sw_data['pct_correct_side']:.0f}% end on correct side")
                else:
                    print(f"  p={p_str}: no switchover, {sw_data.get('pct_correct_side', 0):.0f}% end on correct side")

        if "bias_sensitivity" in q4 and "by_condition" in q4["bias_sensitivity"]:
            if "single_turn" in q4["bias_sensitivity"]["by_condition"]:
                bs = q4["bias_sensitivity"]["by_condition"]["single_turn"]
                ratio = bs["mean_sensitivity_ratio"]["mean"]
                interp = "more sensitive to contradicting" if ratio > 1 else "more sensitive to confirming"
                print(f"  Sensitivity ratio (single_turn): {ratio:.2f} ({interp})")

        # Per-p Breakdown
        print(f"\n--- Per-p Breakdown ---")
        pp = model_data.get("per_p_analysis", {})
        if "by_p" in pp:
            for p_str, p_data in pp["by_p"].items():
                if "overall" in p_data:
                    ov = p_data["overall"]
                    print(f"  p={p_str}: MAE={ov.get('mean_mae', 0):.3f}, " +
                          f"final belief={ov.get('mean_final_belief', 0):.3f}, " +
                          f"error from true={ov.get('belief_error', 0):.3f}")

        # Diagnostics
        print(f"\n--- Diagnostics ---")
        diag = model_data.get("diagnostics", {})
        if "overall" in diag:
            ov = diag["overall"]
            pb = ov.get("mean_position_bias", 0)
            ab = ov.get("mean_ab_mass", 0)
            pb_concern = "⚠️ CONCERN" if ov.get("position_bias_concern") else "OK"
            fmt_ok = "OK" if ov.get("format_following_good") else "⚠️ LOW"
            print(f"  Position bias: {pb:.3f} ({pb_concern})")
            print(f"  A/B mass: {ab:.3f} ({fmt_ok})")


if __name__ == "__main__":
    main()

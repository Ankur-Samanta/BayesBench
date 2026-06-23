"""
Results aggregation and analysis for recommender-system cold-start experiments.

Groups results by (condition, pop_info, type, target_diagnosticity).
Key metrics: MAE from Bayesian, cross-item transfer score,
type inference accuracy, and pop_info effect analysis.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np

from .config import TrajectoryResult, Condition, PopInfo
from .metrics import bootstrap_ci


def load_all_results(
    experiments_dir: Path,
    model_filter: Optional[str] = None,
    pop_info_filter: Optional[str] = None,
) -> List[TrajectoryResult]:
    """Load all experiment results from directory."""
    results = []

    for json_file in experiments_dir.glob("*.json"):
        if json_file.name in [".gitkeep"] or json_file.name.endswith("_analysis.json"):
            continue

        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            result = TrajectoryResult.from_dict(data)

            if model_filter and result.config.model_name != model_filter:
                continue

            if pop_info_filter and result.config.pop_info.value != pop_info_filter:
                continue

            results.append(result)

        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue

    print(f"Loaded {len(results)} experiment results")
    return results


def group_results(
    results: List[TrajectoryResult],
) -> Dict[str, Dict[str, List[TrajectoryResult]]]:
    """
    Group results by model -> key.

    Key format: {condition}_{pop_info}_k{k}_type{true_type}_target{target_id}
    """
    grouped = defaultdict(lambda: defaultdict(list))

    for result in results:
        model = result.config.model_name
        key = (f"{result.config.condition.value}_{result.config.pop_info.value}_"
               f"k{result.config.k}_type{result.config.true_type}_"
               f"target{result.config.target_movie_id}")
        grouped[model][key].append(result)

    return dict(grouped)


def aggregate_by_condition(results: List[TrajectoryResult]) -> Dict[str, Any]:
    """
    Compute aggregate statistics for a group of results.

    Trials serve as the averaging dimension.
    """
    if not results:
        return {}

    mae_values = []
    corr_values = []
    transfer_values = []
    type_correct_values = []
    prior_values = []
    update_values = []
    variance_values = []
    scale_bias_values = []
    rating_mass_values = []

    # Rating distribution metrics
    rating_kl_values = []
    rating_tvd_values = []
    rating_jsd_values = []
    rating_w1_values = []

    # Type elicitation metrics
    type_kl_values = []
    type_tvd_values = []
    type_jsd_values = []
    type_acc_values = []
    cot_acc_values = []
    cond_lift_values = []
    cot_type_mass_values = []

    for r in results:
        if r.metrics:
            mae_values.append(r.metrics.mae_from_bayesian)
            corr_values.append(r.metrics.correlation_with_bayesian)
            transfer_values.append(r.metrics.cross_item_transfer_score)
            type_correct_values.append(float(r.metrics.type_inference_correct))
            prior_values.append(r.metrics.prior_expected_rating)
            update_values.append(r.metrics.mean_update_magnitude)
            variance_values.append(r.metrics.trajectory_variance)
            scale_bias_values.append(r.metrics.mean_scale_bias)
            rating_mass_values.append(r.metrics.mean_rating_mass)

            if r.metrics.mean_kl_divergence is not None:
                rating_kl_values.append(r.metrics.mean_kl_divergence)
            if r.metrics.mean_tvd is not None:
                rating_tvd_values.append(r.metrics.mean_tvd)
            if r.metrics.mean_jsd is not None:
                rating_jsd_values.append(r.metrics.mean_jsd)
            if r.metrics.mean_wasserstein is not None:
                rating_w1_values.append(r.metrics.mean_wasserstein)

            if r.metrics.type_posterior_kl is not None:
                type_kl_values.append(r.metrics.type_posterior_kl)
            if r.metrics.type_posterior_tvd is not None:
                type_tvd_values.append(r.metrics.type_posterior_tvd)
            if r.metrics.type_posterior_jsd is not None:
                type_jsd_values.append(r.metrics.type_posterior_jsd)
            if r.metrics.type_accuracy is not None:
                type_acc_values.append(r.metrics.type_accuracy)
            if r.metrics.cot_type_accuracy is not None:
                cot_acc_values.append(r.metrics.cot_type_accuracy)
            if r.metrics.conditioning_lift is not None:
                cond_lift_values.append(r.metrics.conditioning_lift)
            if r.metrics.mean_cot_type_mass is not None:
                cot_type_mass_values.append(r.metrics.mean_cot_type_mass)

    mae_mean, mae_lo, mae_hi = bootstrap_ci(mae_values)
    corr_mean, corr_lo, corr_hi = bootstrap_ci(corr_values)
    transfer_mean, transfer_lo, transfer_hi = bootstrap_ci(transfer_values)
    update_mean, update_lo, update_hi = bootstrap_ci(update_values)
    var_mean, var_lo, var_hi = bootstrap_ci(variance_values)

    return {
        "n_trials": len(results),
        "condition": results[0].config.condition.value,
        "pop_info": results[0].config.pop_info.value,
        "k": results[0].config.k,
        "true_type": results[0].config.true_type,
        "target_movie_id": results[0].config.target_movie_id,
        "target_movie_name": results[0].config.target_movie_name,
        "mae_from_bayesian": {
            "mean": mae_mean, "ci_lower": mae_lo, "ci_upper": mae_hi,
        },
        "correlation_with_bayesian": {
            "mean": corr_mean, "ci_lower": corr_lo, "ci_upper": corr_hi,
        },
        "cross_item_transfer_score": {
            "mean": transfer_mean, "ci_lower": transfer_lo, "ci_upper": transfer_hi,
        },
        "type_inference_accuracy": float(np.mean(type_correct_values)) if type_correct_values else 0.0,
        "mean_update_magnitude": {
            "mean": update_mean, "ci_lower": update_lo, "ci_upper": update_hi,
        },
        "trajectory_variance": {
            "mean": var_mean, "ci_lower": var_lo, "ci_upper": var_hi,
        },
        "mean_prior_expected_rating": float(np.mean(prior_values)) if prior_values else 3.0,
        "mean_scale_bias": float(np.mean(scale_bias_values)) if scale_bias_values else 0.0,
        "mean_rating_mass": float(np.mean(rating_mass_values)) if rating_mass_values else 0.0,
        # Rating distribution tracking metrics
        "rating_kl": dict(zip(["mean", "ci_lower", "ci_upper"], bootstrap_ci(rating_kl_values))) if rating_kl_values else None,
        "rating_tvd": dict(zip(["mean", "ci_lower", "ci_upper"], bootstrap_ci(rating_tvd_values))) if rating_tvd_values else None,
        "rating_jsd": dict(zip(["mean", "ci_lower", "ci_upper"], bootstrap_ci(rating_jsd_values))) if rating_jsd_values else None,
        "rating_wasserstein": dict(zip(["mean", "ci_lower", "ci_upper"], bootstrap_ci(rating_w1_values))) if rating_w1_values else None,
        # Type elicitation metrics
        "type_posterior_kl": dict(zip(["mean", "ci_lower", "ci_upper"], bootstrap_ci(type_kl_values))) if type_kl_values else None,
        "type_posterior_tvd": dict(zip(["mean", "ci_lower", "ci_upper"], bootstrap_ci(type_tvd_values))) if type_tvd_values else None,
        "type_posterior_jsd": dict(zip(["mean", "ci_lower", "ci_upper"], bootstrap_ci(type_jsd_values))) if type_jsd_values else None,
        "type_accuracy": dict(zip(["mean", "ci_lower", "ci_upper"], bootstrap_ci(type_acc_values))) if type_acc_values else None,
        "cot_type_accuracy": dict(zip(["mean", "ci_lower", "ci_upper"], bootstrap_ci(cot_acc_values))) if cot_acc_values else None,
        "conditioning_lift": dict(zip(["mean", "ci_lower", "ci_upper"], bootstrap_ci(cond_lift_values))) if cond_lift_values else None,
        "mean_cot_type_mass": float(np.mean(cot_type_mass_values)) if cot_type_mass_values else None,
    }


def analyze_pop_info_effect(
    results: List[TrajectoryResult],
    model: str,
) -> Dict[str, Any]:
    """Compare MAE across explicit_types vs zero_shot vs anonymized."""
    pop_info_results = defaultdict(list)

    for r in results:
        if r.config.model_name == model and r.metrics:
            pop_info_results[r.config.pop_info.value].append(r.metrics.mae_from_bayesian)

    analysis = {"model": model}
    for pi_name, maes in pop_info_results.items():
        mean, lo, hi = bootstrap_ci(maes)
        analysis[pi_name] = {"mean": mean, "ci_lower": lo, "ci_upper": hi, "n": len(maes)}

    return analysis


def analyze_type_inference(
    results: List[TrajectoryResult],
    model: str,
) -> Dict[str, Any]:
    """Analyze how quickly the LLM converges to correct type."""
    # Group by pop_info
    by_pop_info = defaultdict(list)
    for r in results:
        if r.config.model_name == model:
            by_pop_info[r.config.pop_info.value].append(r)

    analysis = {"model": model, "by_pop_info": {}}

    for pi_name, pi_results in by_pop_info.items():
        # For each result, find earliest t where type posterior is correct
        convergence_times = []
        final_accuracies = []

        for r in pi_results:
            true_type = r.config.true_type
            converged_t = None

            for poll in r.polls:
                if len(poll.type_posterior) > true_type:
                    if np.argmax(poll.type_posterior) == true_type:
                        if converged_t is None:
                            converged_t = poll.t
                    else:
                        converged_t = None  # Reset if wrong

            if converged_t is not None:
                convergence_times.append(converged_t)

            # Final accuracy
            if r.polls and len(r.polls[-1].type_posterior) > true_type:
                final_accuracies.append(
                    float(np.argmax(r.polls[-1].type_posterior) == true_type)
                )

        analysis["by_pop_info"][pi_name] = {
            "mean_convergence_t": float(np.mean(convergence_times)) if convergence_times else None,
            "convergence_rate": len(convergence_times) / len(pi_results) if pi_results else 0.0,
            "final_type_accuracy": float(np.mean(final_accuracies)) if final_accuracies else 0.0,
            "n_experiments": len(pi_results),
        }

    return analysis


def analyze_diagnosticity_sensitivity(
    results: List[TrajectoryResult],
    model: str,
    movie_selection=None,
) -> Dict[str, Any]:
    """Analyze whether the LLM responds more to diagnostic ratings."""
    analysis = {"model": model, "by_target": {}}

    # Group by target movie
    by_target = defaultdict(list)

    for r in results:
        if r.config.model_name != model or not r.metrics:
            continue
        by_target["all"].append(r.metrics.cross_item_transfer_score)

    for target_key, scores in by_target.items():
        mean, lo, hi = bootstrap_ci(scores)
        analysis["by_target"][target_key] = {
            "mean": mean, "ci_lower": lo, "ci_upper": hi, "n": len(scores),
        }

    return analysis


def generate_summary_report(
    results: List[TrajectoryResult],
    model: str,
) -> Dict[str, Any]:
    """Generate comprehensive summary report for a model."""
    grouped = group_results(results)

    if model not in grouped:
        return {"error": f"Model {model} not found"}

    model_results = [r for r in results if r.config.model_name == model]

    report = {
        "model": model,
        "total_experiments": len(model_results),
        "aggregations": {},
        "pop_info_effect": analyze_pop_info_effect(results, model),
        "type_inference": analyze_type_inference(results, model),
        "diagnosticity_sensitivity": analyze_diagnosticity_sensitivity(results, model),
    }

    # Aggregate by condition group
    for key, key_results in grouped[model].items():
        report["aggregations"][key] = aggregate_by_condition(key_results)

    return report


def main():
    parser = argparse.ArgumentParser(description="Aggregate Recommender System Experiment Results")
    parser.add_argument("--experiments-dir", type=str,
                        default=str(Path(__file__).parent / "experiments"),
                        help="Directory with results")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter by model")
    parser.add_argument("--pop-info", type=str, default=None,
                        choices=[p.value for p in PopInfo],
                        help="Filter by population info mode")

    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)

    results = load_all_results(
        experiments_dir,
        model_filter=args.model,
        pop_info_filter=args.pop_info,
    )

    if not results:
        print("No results found!")
        return

    grouped = group_results(results)

    output = {
        "experiments_dir": str(experiments_dir),
        "total_experiments": len(results),
        "models": {},
    }

    for model in grouped.keys():
        if args.model and model != args.model:
            continue
        model_results = [r for r in results if r.config.model_name == model]
        output["models"][model] = generate_summary_report(model_results, model)

    # Raw aggregations
    output["aggregations"] = {}
    for model, conditions in grouped.items():
        if args.model and model != args.model:
            continue
        output["aggregations"][model] = {}
        for key, condition_results in conditions.items():
            output["aggregations"][model][key] = aggregate_by_condition(condition_results)

    # Save
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved aggregated results to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("RECOMMENDER SYSTEM COLD-START EXPERIMENT SUMMARY")
    print("=" * 70)

    for model, model_data in output["models"].items():
        print(f"\n{'='*70}")
        print(f"MODEL: {model}")
        print(f"{'='*70}")
        print(f"Total experiments: {model_data['total_experiments']}")

        # MAE by condition
        print(f"\n--- MAE from Bayesian ---")
        for key, agg in model_data.get("aggregations", {}).items():
            if "mae_from_bayesian" in agg:
                mae = agg["mae_from_bayesian"]
                print(f"  {key}: MAE = {mae['mean']:.4f} "
                      f"[{mae['ci_lower']:.4f}, {mae['ci_upper']:.4f}]")

        # Cross-item transfer
        print(f"\n--- Cross-Item Transfer Score ---")
        for key, agg in model_data.get("aggregations", {}).items():
            if "cross_item_transfer_score" in agg:
                ts = agg["cross_item_transfer_score"]
                print(f"  {key}: transfer = {ts['mean']:.4f} "
                      f"[{ts['ci_lower']:.4f}, {ts['ci_upper']:.4f}]")

        # Type inference
        print(f"\n--- Type Inference ---")
        for key, agg in model_data.get("aggregations", {}).items():
            if "type_inference_accuracy" in agg:
                print(f"  {key}: accuracy = {agg['type_inference_accuracy']:.3f}")

        # Pop info effect
        print(f"\n--- Population Info Effect ---")
        pie = model_data.get("pop_info_effect", {})
        for pi_name in ["explicit_types", "zero_shot", "anonymized"]:
            if pi_name in pie:
                d = pie[pi_name]
                print(f"  {pi_name}: MAE = {d['mean']:.4f} "
                      f"[{d['ci_lower']:.4f}, {d['ci_upper']:.4f}] (n={d['n']})")

        # Type inference convergence
        print(f"\n--- Type Inference Convergence ---")
        ti = model_data.get("type_inference", {}).get("by_pop_info", {})
        for pi_name, ti_data in ti.items():
            conv_t = ti_data.get("mean_convergence_t")
            conv_str = f"{conv_t:.1f}" if conv_t is not None else "N/A"
            print(f"  {pi_name}: convergence_t = {conv_str}, "
                  f"final_accuracy = {ti_data['final_type_accuracy']:.3f}")


if __name__ == "__main__":
    main()

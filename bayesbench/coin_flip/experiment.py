"""
Coin Flip Prior and Distribution Tracking Experiment

Two parts:
1. Prior Extraction: What's P(heads) with no evidence?
2. Distribution Tracking: Does P(heads) adapt to observed sequences?

This validates the extraction setup before comparing single vs multi-turn.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from .extraction import setup_model, extract_p_heads, extract_trajectory, ExtractionResult


def generate_sequence(n: int, theta: float, seed: int) -> List[str]:
    """Generate n flips with P(heads) = theta."""
    np.random.seed(seed)
    return ["heads" if np.random.random() < theta else "tails" for _ in range(n)]


def bayesian_posterior(history: List[str], prior_alpha: float = 1.0, prior_beta: float = 1.0) -> float:
    """
    Beta-Bernoulli posterior mean.

    With uniform prior (alpha=1, beta=1):
        P(heads) = (1 + n_heads) / (2 + n_total)
    """
    n_heads = sum(1 for f in history if f == "heads")
    n_total = len(history)
    return (prior_alpha + n_heads) / (prior_alpha + prior_beta + n_total)


def run_prior_experiment(model, tokenizer, n_samples: int = 10) -> Dict[str, Any]:
    """
    Extract prior P(heads) with no evidence.

    Runs multiple times to check stability.
    """
    print("=" * 60)
    print("PRIOR EXTRACTION (no evidence)")
    print("=" * 60)

    results = []
    for i in range(n_samples):
        result = extract_p_heads(model, tokenizer, history=[])
        results.append({
            "p_heads": result.p_heads,
            "p_heads_v1": result.p_heads_v1,
            "p_heads_v2": result.p_heads_v2,
            "position_bias": result.position_bias,
            "ab_mass": result.ab_mass
        })
        print(f"  Sample {i+1}: P(heads)={result.p_heads:.3f} "
              f"[v1={result.p_heads_v1:.3f}, v2={result.p_heads_v2:.3f}] "
              f"pos_bias={result.position_bias:+.3f}")

    p_heads_list = [r["p_heads"] for r in results]
    mean_p = np.mean(p_heads_list)
    std_p = np.std(p_heads_list)

    print(f"\n  Mean P(heads): {mean_p:.4f} +/- {std_p:.4f}")
    print(f"  Position bias: {np.mean([r['position_bias'] for r in results]):+.4f}")
    print(f"  A/B mass: {np.mean([r['ab_mass'] for r in results]):.4f}")

    # Classify prior
    if mean_p > 0.55:
        bias = "heads"
    elif mean_p < 0.45:
        bias = "tails"
    else:
        bias = "neutral"

    print(f"  Prior bias: {bias.upper()}")

    return {
        "samples": results,
        "mean_p_heads": float(mean_p),
        "std_p_heads": float(std_p),
        "prior_bias": bias
    }


def run_tracking_experiment(
    model,
    tokenizer,
    thetas: List[float] = [0.25, 0.5, 0.75],
    n_flips: int = 30,
    n_trials: int = 5,
    sample_every: int = 5
) -> Dict[str, Any]:
    """
    Test if P(heads) tracks observed distribution.

    For each theta:
    - Generate sequences with P(heads) = theta
    - Extract P(heads) at intervals
    - Compare to Bayesian posterior
    """
    print("\n" + "=" * 60)
    print("DISTRIBUTION TRACKING")
    print("=" * 60)

    results = {}

    for theta in thetas:
        print(f"\n  Testing theta = {theta}")
        print("-" * 40)

        trial_results = []

        for trial in range(n_trials):
            seed = int(theta * 1000) + trial * 100 + 42
            sequence = generate_sequence(n_flips, theta, seed)

            # Sample at intervals
            sample_points = list(range(0, n_flips + 1, sample_every))
            if n_flips not in sample_points:
                sample_points.append(n_flips)

            model_trajectory = []
            bayesian_trajectory = []

            for n in sample_points:
                history = sequence[:n]

                # Model estimate
                result = extract_p_heads(model, tokenizer, history)
                model_trajectory.append(result.p_heads)

                # Bayesian optimal
                bayesian_trajectory.append(bayesian_posterior(history))

            # Compute correlation with Bayesian
            if np.std(model_trajectory) > 0 and np.std(bayesian_trajectory) > 0:
                corr = np.corrcoef(model_trajectory, bayesian_trajectory)[0, 1]
            else:
                corr = 0.0

            mae = np.mean(np.abs(np.array(model_trajectory) - np.array(bayesian_trajectory)))
            final_error = abs(model_trajectory[-1] - theta)

            trial_results.append({
                "trial": trial,
                "seed": seed,
                "n_heads": sum(1 for f in sequence if f == "heads"),
                "sample_points": sample_points,
                "model_trajectory": model_trajectory,
                "bayesian_trajectory": bayesian_trajectory,
                "correlation": float(corr),
                "mae": float(mae),
                "final_error": float(final_error)
            })

            print(f"    Trial {trial+1}: corr={corr:.3f}, MAE={mae:.3f}, final_err={final_error:.3f}")

        # Aggregate
        corrs = [t["correlation"] for t in trial_results]
        maes = [t["mae"] for t in trial_results]

        results[str(theta)] = {
            "trials": trial_results,
            "mean_correlation": float(np.mean(corrs)),
            "std_correlation": float(np.std(corrs)),
            "mean_mae": float(np.mean(maes)),
            "std_mae": float(np.std(maes))
        }

        print(f"    => Mean corr: {np.mean(corrs):.3f} +/- {np.std(corrs):.3f}")

    return results


def print_summary(prior_results: Dict, tracking_results: Dict):
    """Print experiment summary."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nPrior P(heads): {prior_results['mean_p_heads']:.3f} ({prior_results['prior_bias']})")

    print("\nDistribution Tracking:")
    print(f"  {'Theta':<8} {'Correlation':<15} {'MAE':<15}")
    print("  " + "-" * 38)
    for theta, data in sorted(tracking_results.items(), key=lambda x: float(x[0])):
        corr = f"{data['mean_correlation']:.3f} +/- {data['std_correlation']:.3f}"
        mae = f"{data['mean_mae']:.3f} +/- {data['std_mae']:.3f}"
        print(f"  {theta:<8} {corr:<15} {mae:<15}")


def main():
    parser = argparse.ArgumentParser(description="Coin Flip Prior & Tracking Experiment")
    parser.add_argument("--model", type=str, default="qwen7b", help="Model to test")
    parser.add_argument("--n-flips", type=int, default=30, help="Flips per sequence")
    parser.add_argument("--n-trials", type=int, default=5, help="Trials per theta")
    parser.add_argument("--sample-every", type=int, default=5, help="Sample interval")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    # Setup
    model, tokenizer = setup_model(args.model)

    # Run experiments
    prior_results = run_prior_experiment(model, tokenizer)
    tracking_results = run_tracking_experiment(
        model, tokenizer,
        n_flips=args.n_flips,
        n_trials=args.n_trials,
        sample_every=args.sample_every
    )

    # Summary
    print_summary(prior_results, tracking_results)

    # Save results
    if args.output_dir is None:
        output_dir = Path(__file__).parent / "results"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_flips": args.n_flips,
            "n_trials": args.n_trials,
            "sample_every": args.sample_every
        },
        "prior": prior_results,
        "tracking": tracking_results
    }

    output_path = output_dir / f"validation_{args.model}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

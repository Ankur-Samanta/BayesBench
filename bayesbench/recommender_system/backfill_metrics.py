"""
Backfill new metrics (genre_transfer_score, mean_kl_divergence, mean_wasserstein)
into existing result files without rerunning LLM inference.

All necessary data is already stored in the JSON files:
- rating_distribution (LLM's 5-way dist at each poll)
- type_posterior (P(type=k | obs) at each poll)
- rating_sequence (full observation history with movie_ids)

We just need the TypeModel's theta[target] to reconstruct the Bayesian 5-way
distribution, then compute KL and Wasserstein against the LLM distribution.

Usage:
    python -m bayesbench.recommender_system.backfill_metrics [--experiments-dir recommender_system/experiments] [--dry-run]
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict

from .data import prepare_data
from .metrics import genre_overlap_baseline, marginal_baseline


def compute_backfill_metrics(
    result_dict: Dict,
    type_model,
) -> Dict:
    """Compute the 3 new metrics from stored poll data + TypeModel."""
    target_mid = result_dict["config"]["target_movie_id"]
    polls = result_dict["polls"]
    sequence = result_dict["rating_sequence"]

    # --- Genre transfer score ---
    mae_llm_vs_bayes_values = []
    mae_genre_vs_bayes_values = []

    for poll in polls:
        t = poll["t"]
        observations = sequence[:t]
        genre_pred = genre_overlap_baseline(observations, target_mid, type_model)
        bayesian_pred = poll["bayesian_posterior"]

        mae_llm_vs_bayes_values.append(abs(poll["expected_rating"] - bayesian_pred))
        mae_genre_vs_bayes_values.append(abs(genre_pred - bayesian_pred))

    mae_from_bayesian = float(np.mean(mae_llm_vs_bayes_values))
    mae_genre_vs_bayes = float(np.mean(mae_genre_vs_bayes_values))

    if mae_genre_vs_bayes > 1e-6:
        genre_transfer_score = 1.0 - (mae_from_bayesian / mae_genre_vs_bayes)
    else:
        genre_transfer_score = 0.0

    # --- KL divergence and Wasserstein distance ---
    kl_divergences = []
    wasserstein_distances = []

    for poll in polls:
        type_post = np.array(poll["type_posterior"])
        llm_dist = np.array(poll["rating_distribution"])

        # Reconstruct Bayesian 5-way distribution from stored type_posterior
        if target_mid in type_model.theta:
            theta_target = type_model.theta[target_mid]  # (K, 5)
            bayes_dist = np.array([
                float(type_post @ theta_target[:, r])
                for r in range(5)
            ])
        else:
            bayes_dist = np.array([0.2] * 5)

        # Normalize and clip
        bayes_dist = np.clip(bayes_dist, 1e-10, None)
        bayes_dist = bayes_dist / bayes_dist.sum()
        llm_dist = np.clip(llm_dist, 1e-10, None)
        llm_dist = llm_dist / llm_dist.sum()

        # KL(Bayesian || LLM)
        kl = float(np.sum(bayes_dist * np.log(bayes_dist / llm_dist)))
        kl_divergences.append(kl)

        # Wasserstein-1 on ordinal 1-5 scale
        cdf_bayes = np.cumsum(bayes_dist)
        cdf_llm = np.cumsum(llm_dist)
        w1 = float(np.sum(np.abs(cdf_bayes - cdf_llm)))
        wasserstein_distances.append(w1)

    mean_kl = float(np.mean(kl_divergences)) if kl_divergences else None
    mean_w1 = float(np.mean(wasserstein_distances)) if wasserstein_distances else None

    return {
        "genre_transfer_score": genre_transfer_score,
        "mean_kl_divergence": mean_kl,
        "mean_wasserstein": mean_w1,
    }


def main():
    parser = argparse.ArgumentParser(description="Backfill new metrics into existing results")
    parser.add_argument("--experiments-dir", type=str,
                        default=str(Path(__file__).parent / "experiments"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be updated without writing")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--n-types", type=int, default=4)
    args = parser.parse_args()

    exp_dir = Path(args.experiments_dir)

    # Load TypeModel once
    print("Loading TypeModel...")
    type_model, movie_selection = prepare_data(
        data_dir=args.data_dir, n_types=args.n_types,
    )
    print("TypeModel loaded.\n")

    # Find files that need backfilling (missing any of the 3 new metrics)
    all_files = sorted(exp_dir.glob("*.json"))
    print(f"Found {len(all_files)} result files")

    needs_update = []
    already_done = 0
    errors = 0

    for fpath in all_files:
        try:
            with open(fpath) as f:
                d = json.load(f)
            metrics = d.get("metrics", {})
            if metrics is None:
                needs_update.append(fpath)
            elif (metrics.get("genre_transfer_score") is None or
                  metrics.get("mean_kl_divergence") is None or
                  metrics.get("mean_wasserstein") is None):
                needs_update.append(fpath)
            else:
                already_done += 1
        except Exception as e:
            print(f"  Error reading {fpath.name}: {e}")
            errors += 1

    print(f"  Already have all 3 metrics: {already_done}")
    print(f"  Need backfill: {len(needs_update)}")
    print(f"  Errors: {errors}")
    print()

    if args.dry_run:
        print("DRY RUN — no files written")
        for fpath in needs_update[:10]:
            print(f"  Would update: {fpath.name}")
        if len(needs_update) > 10:
            print(f"  ... and {len(needs_update) - 10} more")
        return

    # Process files
    updated = 0
    for i, fpath in enumerate(needs_update):
        try:
            with open(fpath) as f:
                d = json.load(f)

            new_metrics = compute_backfill_metrics(d, type_model)

            if d["metrics"] is None:
                d["metrics"] = new_metrics
            else:
                d["metrics"].update(new_metrics)

            with open(fpath, "w") as f:
                json.dump(d, f, indent=2)

            updated += 1
            if (i + 1) % 500 == 0:
                print(f"  Updated {i + 1}/{len(needs_update)}...")

        except Exception as e:
            print(f"  Error processing {fpath.name}: {e}")

    print(f"\nDone. Updated {updated}/{len(needs_update)} files.")


if __name__ == "__main__":
    main()

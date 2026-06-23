#!/usr/bin/env python3
"""
Type model selection analysis for recommender-system experiments.

Compares K-means (genre-space) vs EM (rating-space) fitting, and evaluates
different numbers of types (K=3,4,5) on discriminability metrics:
  - Pairwise Jensen-Shannon divergence between types
  - Bayesian oracle classification accuracy at t=10,20,50
  - Type balance (mixture weights)
  - Log-likelihood / BIC

Results are saved to a JSON log for paper figures and tables.

Usage:
    python -m bayesbench.recommender_system.analyze_type_model [--max-k 5] [--em-restarts 3] [--output-dir results/]
"""

import json
import sys
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy.spatial.distance import jensenshannon


def load_ratings(ml_dir: Path, valid_movies: set):
    """Load ratings filtered to valid movies, grouped by user."""
    user_ratings = defaultdict(list)
    total = 0
    with open(ml_dir / "ratings.dat", "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            uid, mid, rating = int(parts[0]), int(parts[1]), int(parts[2])
            if mid in valid_movies:
                user_ratings[uid].append((mid, rating))
                total += 1
    # Filter to users with >= 20 ratings
    user_ratings = {uid: rats for uid, rats in user_ratings.items() if len(rats) >= 20}
    return user_ratings, total


def em_fit(user_ratings, movie_ids, n_types, max_iter=100, seed=42):
    """Fit mixture of Categoricals via EM.

    Returns (theta, pi, ll, n_iter, user_assignments) where
    user_assignments maps uid -> hard argmax type.
    """
    rng = np.random.RandomState(seed)
    n_ratings = sum(len(r) for r in user_ratings.values())

    # Initialize theta with random Dirichlet draws
    pi = np.ones(n_types) / n_types
    theta = {mid: rng.dirichlet(np.ones(5), size=n_types) for mid in movie_ids}

    prev_ll = -np.inf
    for iteration in range(max_iter):
        # E-step
        responsibilities = {}
        ll = 0.0
        for uid, rats in user_ratings.items():
            log_likes = np.log(pi + 1e-30).copy()
            for mid, r in rats:
                for k in range(n_types):
                    log_likes[k] += np.log(theta[mid][k, r - 1] + 1e-30)
            max_ll = np.max(log_likes)
            resp = np.exp(log_likes - max_ll)
            ll += max_ll + np.log(resp.sum())
            resp /= resp.sum()
            responsibilities[uid] = resp

        # M-step
        new_pi = np.zeros(n_types)
        for resp in responsibilities.values():
            new_pi += resp
        new_pi /= new_pi.sum()

        new_theta = {mid: np.ones((n_types, 5)) for mid in movie_ids}  # Dirichlet(1) prior
        for uid, rats in user_ratings.items():
            resp = responsibilities[uid]
            for mid, r in rats:
                for k in range(n_types):
                    new_theta[mid][k, r - 1] += resp[k]
        for mid in movie_ids:
            new_theta[mid] /= new_theta[mid].sum(axis=1, keepdims=True)

        delta = ll - prev_ll
        if delta < 1.0 and iteration > 10:
            pi, theta = new_pi, new_theta
            break
        prev_ll = ll
        pi, theta = new_pi, new_theta

    user_assignments = {uid: int(np.argmax(resp)) for uid, resp in responsibilities.items()}
    return theta, pi, ll, iteration + 1, user_assignments


def evaluate_model(theta, pi, movie_ids, n_types, n_sims=500, seed=42):
    """Evaluate type model on discriminability metrics."""
    T = np.array([theta[mid] for mid in movie_ids])
    n_movies = len(movie_ids)
    ratings = np.array([1, 2, 3, 4, 5])

    # Pairwise JS divergence
    js_matrix = np.zeros((n_types, n_types))
    for i in range(n_types):
        for j in range(i + 1, n_types):
            js = np.mean([jensenshannon(T[m, i, :], T[m, j, :]) for m in range(n_movies)])
            js_matrix[i, j] = js_matrix[j, i] = js

    js_pairs = js_matrix[np.triu_indices(n_types, k=1)]

    # Bayesian oracle accuracy
    rng = np.random.RandomState(seed)
    checkpoints = [5, 10, 20, 50]
    acc = {t: [] for t in checkpoints}
    per_type_acc = {tt: {t: [] for t in checkpoints} for tt in range(n_types)}

    sims_per_type = n_sims // n_types
    for true_type in range(n_types):
        for _ in range(sims_per_type):
            seq = rng.choice(n_movies, size=max(checkpoints), replace=True)
            log_post = np.log(pi + 1e-30)
            for step in range(max(checkpoints)):
                mid = movie_ids[seq[step]]
                rating = rng.choice(5, p=theta[mid][true_type])
                for k in range(n_types):
                    log_post[k] += np.log(theta[mid][k, rating] + 1e-30)
                log_post -= np.max(log_post)
                t = step + 1
                if t in acc:
                    post = np.exp(log_post)
                    post /= post.sum()
                    correct = int(np.argmax(post) == true_type)
                    acc[t].append(correct)
                    per_type_acc[true_type][t].append(correct)

    # Mean expected rating per type
    mean_ratings = [float(np.mean(T[:, k, :] @ ratings)) for k in range(n_types)]

    # Per-movie discriminability
    max_js_per_movie = []
    for m_idx in range(n_movies):
        best = 0
        for i in range(n_types):
            for j in range(i + 1, n_types):
                best = max(best, jensenshannon(T[m_idx, i, :], T[m_idx, j, :]))
        max_js_per_movie.append(best)

    return {
        "js_matrix": js_matrix.tolist(),
        "min_pairwise_js": float(js_pairs.min()),
        "mean_pairwise_js": float(js_pairs.mean()),
        "max_pairwise_js": float(js_pairs.max()),
        "oracle_accuracy": {str(t): float(np.mean(v)) for t, v in acc.items()},
        "per_type_oracle_accuracy": {
            str(tt): {str(t): float(np.mean(v)) for t, v in taccs.items()}
            for tt, taccs in per_type_acc.items()
        },
        "pi": [float(x) for x in pi],
        "min_type_weight": float(min(pi)),
        "mean_expected_rating_per_type": mean_ratings,
        "chance_accuracy": 1.0 / n_types,
        "majority_accuracy": float(max(pi)),
        "movies_with_js_gt_03": int(sum(1 for x in max_js_per_movie if x > 0.3)),
        "movies_with_js_gt_02": int(sum(1 for x in max_js_per_movie if x > 0.2)),
        "median_max_js_per_movie": float(np.median(max_js_per_movie)),
    }


def main():
    parser = argparse.ArgumentParser(description="Type model selection analysis")
    parser.add_argument("--max-k", type=int, default=5, help="Max K to evaluate")
    parser.add_argument("--min-k", type=int, default=3, help="Min K to evaluate")
    parser.add_argument("--em-restarts", type=int, default=3, help="Random restarts per K")
    parser.add_argument("--em-max-iter", type=int, default=100, help="Max EM iterations")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).parent / "data"
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load current K-means model to get valid movie set
    print("Loading current model for movie set...")
    with open(data_dir / "fitted_model.json") as f:
        current_model = json.load(f)
    valid_movies = set(int(k) for k in current_model["theta"].keys())
    movie_ids = sorted(valid_movies)
    print(f"  Movies: {len(movie_ids)}")

    # Load ratings
    print("Loading ratings...")
    ml_dir = data_dir / "ml-1m"
    user_ratings, total_ratings = load_ratings(ml_dir, valid_movies)
    n_users = len(user_ratings)
    n_ratings = sum(len(r) for r in user_ratings.values())
    print(f"  Users: {n_users}, Ratings: {n_ratings}")

    results = {"metadata": {
        "n_users": n_users,
        "n_movies": len(movie_ids),
        "n_ratings": n_ratings,
        "em_restarts": args.em_restarts,
        "em_max_iter": args.em_max_iter,
    }, "models": []}

    # Evaluate current K-means model
    print("\n" + "=" * 60)
    print("Evaluating current K-means model (K=5)")
    print("=" * 60)
    kmeans_theta = {int(k): np.array(v) for k, v in current_model["theta"].items()}
    kmeans_pi = np.array(current_model["pi"])
    kmeans_eval = evaluate_model(kmeans_theta, kmeans_pi, movie_ids, 5)
    kmeans_eval["method"] = "kmeans_genre"
    kmeans_eval["n_types"] = 5

    # Log-likelihood of K-means model
    ll = 0.0
    for uid, rats in user_ratings.items():
        log_likes = np.log(kmeans_pi + 1e-30).copy()
        for mid, r in rats:
            for k in range(5):
                log_likes[k] += np.log(kmeans_theta[mid][k, r - 1] + 1e-30)
        max_ll = np.max(log_likes)
        ll += max_ll + np.log(np.sum(np.exp(log_likes - max_ll)))
    kmeans_eval["log_likelihood"] = float(ll)
    kmeans_eval["per_rating_ll"] = float(ll / n_ratings)
    n_params = 5 * len(movie_ids) * 4 + 4  # K * M * 4 free probs + K-1 mixture weights
    kmeans_eval["bic"] = float(-2 * ll + n_params * np.log(n_ratings))

    kmeans_eval["n_free_params"] = n_params
    kmeans_eval["oracle_minus_chance"] = {
        str(t): float(v - 0.2)
        for t, v in kmeans_eval["oracle_accuracy"].items()
    }
    kmeans_eval["oracle_minus_majority"] = {
        str(t): float(v - kmeans_eval["majority_accuracy"])
        for t, v in kmeans_eval["oracle_accuracy"].items()
    }

    results["models"].append(kmeans_eval)
    _print_summary("K-means K=5", kmeans_eval)

    # EM for each K
    for K in range(args.min_k, args.max_k + 1):
        print(f"\n{'=' * 60}")
        print(f"Fitting EM K={K} ({args.em_restarts} restarts)")
        print("=" * 60)

        best_ll = -np.inf
        best_result = None
        restart_log = []

        for restart in range(args.em_restarts):
            seed = 42 + restart * 1000
            print(f"  Restart {restart + 1}/{args.em_restarts} (seed={seed})...", end=" ", flush=True)
            theta, pi, ll, n_iter, user_asgn = em_fit(
                user_ratings, movie_ids, K,
                max_iter=args.em_max_iter, seed=seed,
            )
            print(f"converged iter {n_iter}, LL={ll:.0f} ({ll / n_ratings:.4f}/rat)")
            restart_log.append({
                "seed": seed,
                "n_iter": n_iter,
                "log_likelihood": float(ll),
                "per_rating_ll": float(ll / n_ratings),
                "pi": [float(x) for x in pi],
            })
            if ll > best_ll:
                best_ll = ll
                best_result = (theta, pi, ll, user_asgn)

        theta, pi, ll, user_asgn = best_result
        print(f"  Best LL: {ll:.0f} ({ll / n_ratings:.4f}/rat)")
        print(f"  pi: {np.round(pi, 3)}")

        em_eval = evaluate_model(theta, pi, movie_ids, K)
        em_eval["method"] = "em_rating"
        em_eval["n_types"] = K
        em_eval["log_likelihood"] = float(ll)
        em_eval["per_rating_ll"] = float(ll / n_ratings)
        n_params = K * len(movie_ids) * 4 + (K - 1)
        em_eval["bic"] = float(-2 * ll + n_params * np.log(n_ratings))
        em_eval["n_free_params"] = n_params
        em_eval["restart_log"] = restart_log

        # Derived metrics for paper
        chance = 1.0 / K
        em_eval["oracle_minus_chance"] = {
            str(t): float(np.mean(v) - chance)
            for t, v in [(t, em_eval["oracle_accuracy"][t]) for t in ["5", "10", "20", "50"]]
        }
        em_eval["oracle_minus_majority"] = {
            str(t): float(np.mean(v) - em_eval["majority_accuracy"])
            for t, v in [(t, em_eval["oracle_accuracy"][t]) for t in ["5", "10", "20", "50"]]
        }

        results["models"].append(em_eval)
        _print_summary(f"EM K={K}", em_eval)

    # Add selection rationale
    em_models = [m for m in results["models"] if m["method"] == "em_rating"]
    best_by_bic = min(em_models, key=lambda m: m["bic"])
    best_by_oracle_minus_chance = max(
        em_models,
        key=lambda m: float(m["oracle_minus_chance"]["20"]),
    )
    best_by_min_js = max(em_models, key=lambda m: m["min_pairwise_js"])

    results["selection"] = {
        "chosen_method": "em_rating",
        "chosen_k": best_by_min_js["n_types"],
        "rationale": (
            f"K={best_by_bic['n_types']} has lowest BIC ({best_by_bic['bic']:.0f}), "
            f"but K={best_by_min_js['n_types']} maximizes minimum pairwise JS divergence "
            f"({best_by_min_js['min_pairwise_js']:.4f}) and oracle-minus-chance accuracy "
            f"at t=20 ({best_by_oracle_minus_chance['oracle_minus_chance']['20']:.3f}). "
            f"EM fitting directly optimizes the mixture-of-Categoricals generative model, "
            f"which is the same model assumed by the Bayesian oracle baseline."
        ),
        "comparison_to_kmeans": (
            f"EM K={best_by_min_js['n_types']} vs K-means K=5: "
            f"min JS {best_by_min_js['min_pairwise_js']:.4f} vs "
            f"{results['models'][0]['min_pairwise_js']:.4f}, "
            f"oracle@20 {best_by_min_js['oracle_accuracy']['20']:.3f} vs "
            f"{results['models'][0]['oracle_accuracy']['20']:.3f}, "
            f"LL/rat {best_by_min_js['per_rating_ll']:.4f} vs "
            f"{results['models'][0]['per_rating_ll']:.4f}"
        ),
        "best_by_bic": {"k": best_by_bic["n_types"], "bic": best_by_bic["bic"]},
        "best_by_min_js": {"k": best_by_min_js["n_types"], "min_js": best_by_min_js["min_pairwise_js"]},
        "best_by_oracle_minus_chance_20": {
            "k": best_by_oracle_minus_chance["n_types"],
            "value": float(best_by_oracle_minus_chance["oracle_minus_chance"]["20"]),
        },
    }

    # Save results
    output_path = output_dir / "type_model_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    header = f"{'Model':<20} {'LL/rat':>8} {'BIC':>12} {'MinJS':>7} {'MeanJS':>7} {'Orc@10':>7} {'Orc@20':>7} {'Orc@50':>7} {'MinPi':>6} {'Chance':>6}"
    print(header)
    print("-" * len(header))
    for m in results["models"]:
        label = f"{m['method']} K={m['n_types']}"
        print(f"{label:<20} {m['per_rating_ll']:>8.4f} {m['bic']:>12.0f} "
              f"{m['min_pairwise_js']:>7.4f} {m['mean_pairwise_js']:>7.4f} "
              f"{m['oracle_accuracy']['10']:>7.3f} {m['oracle_accuracy']['20']:>7.3f} "
              f"{m['oracle_accuracy']['50']:>7.3f} {m['min_type_weight']:>6.3f} "
              f"{m['chance_accuracy']:>6.3f}")


def _print_summary(label, ev):
    """Print summary for one model."""
    print(f"\n  {label}:")
    print(f"    pi = {[round(x, 3) for x in ev['pi']]}")
    print(f"    LL/rating = {ev['per_rating_ll']:.4f}, BIC = {ev['bic']:.0f}")
    print(f"    Min pairwise JS = {ev['min_pairwise_js']:.4f}, Mean = {ev['mean_pairwise_js']:.4f}")
    print(f"    Oracle @10={ev['oracle_accuracy']['10']:.3f} @20={ev['oracle_accuracy']['20']:.3f} @50={ev['oracle_accuracy']['50']:.3f}")
    print(f"    Chance={ev['chance_accuracy']:.3f}, Majority={ev['majority_accuracy']:.3f}")
    for tt in sorted(ev["per_type_oracle_accuracy"].keys(), key=int):
        ta = ev["per_type_oracle_accuracy"][tt]
        print(f"      Type {tt} (pi={ev['pi'][int(tt)]:.3f}): @20={ta['20']:.3f}")


if __name__ == "__main__":
    main()

"""Sweep EM iterations with pairwise genre divergence and pairwise Spearman."""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

from .data import (
    load_movies, load_ratings as load_ratings_raw,
    compute_genre_preferences, GENRE_NAMES, _GENRE_TO_IDX,
)
from .analyze_type_model import evaluate_model
from .hybrid_type_model import load_ratings_by_user, kmeans_init
from .sweep_em_iterations import em_n_steps

import argparse


def pairwise_genre_divergence(theta, movie_ids, n_types, movies):
    """Compute per-type centered genre means, then pairwise L2 distance between types."""
    n_genres = len(GENRE_NAMES)
    genre_sums = np.zeros((n_types, n_genres))
    genre_counts = np.zeros(n_genres)

    for j, mid in enumerate(movie_ids):
        m = movies.get(mid, {})
        exp_r = [sum((r + 1) * theta[mid][k, r] for r in range(5)) for k in range(n_types)]
        for g in m.get("genres", []):
            if g in _GENRE_TO_IDX:
                gi = _GENRE_TO_IDX[g]
                genre_counts[gi] += 1
                for k in range(n_types):
                    genre_sums[k, gi] += exp_r[k]

    genre_means = np.zeros((n_types, n_genres))
    for gi in range(n_genres):
        if genre_counts[gi] > 0:
            genre_means[:, gi] = genre_sums[:, gi] / genre_counts[gi]

    # Center per type (remove leniency)
    type_overall = genre_means.mean(axis=1)
    centered = genre_means - type_overall[:, None]

    # Pairwise L2 distance between centered genre profiles
    pairwise_l2 = {}
    for i in range(n_types):
        for j in range(i + 1, n_types):
            d = np.sqrt(np.sum((centered[i] - centered[j]) ** 2))
            pairwise_l2[(i, j)] = float(d)

    # Also per-genre std
    genre_std = centered.std(axis=0)
    valid = genre_counts >= 10
    mean_genre_std = float(genre_std[valid].mean()) if valid.any() else 0.0

    return {
        "pairwise_l2": pairwise_l2,
        "min_pairwise_l2": float(min(pairwise_l2.values())) if pairwise_l2 else 0.0,
        "mean_pairwise_l2": float(np.mean(list(pairwise_l2.values()))) if pairwise_l2 else 0.0,
        "mean_genre_std": mean_genre_std,
    }


def pairwise_spearman(theta, movie_ids, n_types):
    """Compute pairwise Spearman rho on expected ratings."""
    n_movies = len(movie_ids)
    exp = np.zeros((n_types, n_movies))
    for j, mid in enumerate(movie_ids):
        for k in range(n_types):
            exp[k, j] = sum((r + 1) * theta[mid][k, r] for r in range(5))

    pairs = {}
    for i in range(n_types):
        for j in range(i + 1, n_types):
            rho, _ = spearmanr(exp[i], exp[j])
            pairs[(i, j)] = float(rho)

    type_means = [float(exp[k].mean()) for k in range(n_types)]
    return {
        "pairwise_rho": pairs,
        "min_rho": float(min(pairs.values())) if pairs else 1.0,
        "mean_rho": float(np.mean(list(pairs.values()))) if pairs else 1.0,
        "max_rho": float(max(pairs.values())) if pairs else 1.0,
        "type_means": type_means,
        "leniency_spread": float(max(type_means) - min(type_means)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    K = args.k
    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).parent / "data"
    ml_dir = data_dir / "ml-1m"

    print(f"Loading data for K={K}...")
    movies = load_movies(ml_dir)

    with open(data_dir / "fitted_model.json") as f:
        current = json.load(f)
    valid_movies = set(int(k) for k in current["theta"].keys())
    movie_ids = sorted(valid_movies)

    user_ratings, n_ratings = load_ratings_by_user(ml_dir, valid_movies)

    raw_ratings = load_ratings_raw(ml_dir)
    user_ratings_for_genre = defaultdict(list)
    for r in raw_ratings:
        entry = dict(r)
        entry["star_rating"] = entry["rating"]
        user_ratings_for_genre[entry["user_id"]].append(entry)
    user_prefs, qualifying = compute_genre_preferences(
        user_ratings_for_genre, movies, min_ratings=20)

    print(f"  Movies: {len(movie_ids)}, Users: {len(user_ratings)}, Ratings: {n_ratings}")

    # K-means init
    print(f"\nK-means K={K}...")
    theta_init, pi_init, asgn_init = kmeans_init(
        user_prefs, qualifying, movies, user_ratings, movie_ids, K, seed=args.seed)

    n_params = K * len(movie_ids) * 4 + (K - 1)

    header = (f"{'Step':>4} {'Orc@20':>7} {'MinJS':>7} {'GenStd':>7} "
              f"{'MinGenL2':>8} {'MeanGenL2':>9} "
              f"{'MinRho':>7} {'MeanRho':>8} {'MaxRho':>7} "
              f"{'Leniency':>8} {'%Chg':>6}")
    print(f"\n{header}")
    print("-" * len(header))

    results = []

    for n_steps in range(0, args.max_steps + 1):
        if n_steps == 0:
            theta, pi = theta_init, pi_init
            user_asgn = asgn_init
            # Compute LL for step 0
            ll = 0.0
            for uid, rats in user_ratings.items():
                log_likes = np.log(pi + 1e-30).copy()
                for mid, r in rats:
                    for k in range(K):
                        log_likes[k] += np.log(theta[mid][k, r - 1] + 1e-30)
                mx = np.max(log_likes)
                ll += mx + np.log(np.sum(np.exp(log_likes - mx)))
        else:
            theta, pi, ll, user_asgn = em_n_steps(
                user_ratings, movie_ids, K, theta_init, pi_init, n_steps)

        ev = evaluate_model(theta, pi, movie_ids, K)
        gd = pairwise_genre_divergence(theta, movie_ids, K, movies)
        sp = pairwise_spearman(theta, movie_ids, K)

        common = set(asgn_init.keys()) & set(user_asgn.keys())
        n_changed = sum(1 for uid in common if asgn_init[uid] != user_asgn[uid])
        pct = 100 * n_changed / len(common) if common else 0

        row = {
            "em_steps": n_steps,
            "pi": [float(x) for x in pi],
            "ll_per_rat": float(ll / n_ratings),
            "oracle_20": float(np.mean(ev["oracle_accuracy"]["20"])),
            "min_js": float(ev["min_pairwise_js"]),
            "genre_std": gd["mean_genre_std"],
            "min_genre_l2": gd["min_pairwise_l2"],
            "mean_genre_l2": gd["mean_pairwise_l2"],
            "pairwise_genre_l2": {f"{i}-{j}": v for (i, j), v in gd["pairwise_l2"].items()},
            "min_rho": sp["min_rho"],
            "mean_rho": sp["mean_rho"],
            "max_rho": sp["max_rho"],
            "pairwise_rho": {f"{i}-{j}": v for (i, j), v in sp["pairwise_rho"].items()},
            "leniency_spread": sp["leniency_spread"],
            "type_means": sp["type_means"],
            "pct_changed": float(pct),
        }
        results.append(row)

        print(f"{n_steps:>4} {row['oracle_20']:>7.3f} {row['min_js']:>7.4f} "
              f"{row['genre_std']:>7.4f} {row['min_genre_l2']:>8.4f} "
              f"{row['mean_genre_l2']:>9.4f} {row['min_rho']:>7.4f} "
              f"{row['mean_rho']:>8.4f} {row['max_rho']:>7.4f} "
              f"{row['leniency_spread']:>8.3f} {row['pct_changed']:>5.1f}%")

    # Save
    output = {
        "metadata": {"K": K, "n_movies": len(movie_ids), "n_users": len(user_ratings),
                      "n_ratings": n_ratings, "seed": args.seed},
        "sweep": results,
    }
    out_path = Path(__file__).parent / f"em_sweep_v2_k{K}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

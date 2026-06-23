"""Sweep EM iterations (1-20) after K-means init. Track genre std, leniency, oracle."""
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
from .hybrid_type_model import (
    load_ratings_by_user, kmeans_init, rank_divergence_stats, genre_divergence,
)

import argparse


def em_n_steps(user_ratings, movie_ids, n_types, init_theta, init_pi, n_steps):
    """Run exactly n_steps of EM from given init. Returns (theta, pi, ll, assignments)."""
    pi = init_pi.copy()
    theta = {mid: arr.copy() for mid, arr in init_theta.items()}
    responsibilities = {}

    for iteration in range(n_steps):
        # E-step
        responsibilities = {}
        ll = 0.0
        for uid, rats in user_ratings.items():
            log_likes = np.log(pi + 1e-30).copy()
            for mid, r in rats:
                for k in range(n_types):
                    log_likes[k] += np.log(theta[mid][k, r - 1] + 1e-30)
            mx = np.max(log_likes)
            resp = np.exp(log_likes - mx)
            ll += mx + np.log(resp.sum())
            resp /= resp.sum()
            responsibilities[uid] = resp

        # M-step
        new_pi = np.zeros(n_types)
        for resp in responsibilities.values():
            new_pi += resp
        new_pi /= new_pi.sum()

        new_theta = {mid: np.ones((n_types, 5)) for mid in movie_ids}
        for uid, rats in user_ratings.items():
            resp = responsibilities[uid]
            for mid, r in rats:
                for k in range(n_types):
                    new_theta[mid][k, r - 1] += resp[k]
        for mid in movie_ids:
            new_theta[mid] /= new_theta[mid].sum(axis=1, keepdims=True)

        pi, theta = new_pi, new_theta

    user_asgn = {uid: int(np.argmax(resp)) for uid, resp in responsibilities.items()}
    return theta, pi, ll, user_asgn


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

    # Stage 1: K-means
    print(f"\nK-means K={K}...")
    theta_init, pi_init, asgn_init = kmeans_init(
        user_prefs, qualifying, movies, user_ratings, movie_ids, K, seed=args.seed)

    # Evaluate K-means (step 0)
    kmeans_eval = evaluate_model(theta_init, pi_init, movie_ids, K)
    kmeans_rank = rank_divergence_stats(theta_init, movie_ids, K)
    kmeans_genre = genre_divergence(theta_init, movie_ids, K, movies)

    n_params = K * len(movie_ids) * 4 + (K - 1)
    kmeans_ll = 0.0
    for uid, rats in user_ratings.items():
        log_likes = np.log(pi_init + 1e-30).copy()
        for mid, r in rats:
            for k in range(K):
                log_likes[k] += np.log(theta_init[mid][k, r - 1] + 1e-30)
        mx = np.max(log_likes)
        kmeans_ll += mx + np.log(np.sum(np.exp(log_likes - mx)))

    results = []
    results.append({
        "em_steps": 0,
        "pi": [float(x) for x in pi_init],
        "ll_per_rat": float(kmeans_ll / n_ratings),
        "bic": float(-2 * kmeans_ll + n_params * np.log(n_ratings)),
        "min_js": float(kmeans_eval["min_pairwise_js"]),
        "mean_js": float(kmeans_eval["mean_pairwise_js"]),
        "oracle_20": float(np.mean(kmeans_eval["oracle_accuracy"]["20"])),
        "mean_spearman": kmeans_rank["mean_spearman"],
        "min_spearman": kmeans_rank["min_spearman"],
        "leniency_spread": kmeans_rank["leniency_spread"],
        "genre_std": kmeans_genre["mean_genre_std"],
        "max_genre_std": kmeans_genre["max_genre_std"],
        "pct_changed": 0.0,
    })

    print(f"\n{'Steps':>5} {'LL/rat':>8} {'MinJS':>7} {'Orc@20':>7} {'Spear':>7} "
          f"{'Leniency':>8} {'GenreStd':>8} {'%Changed':>8}")
    print("-" * 75)
    r = results[0]
    print(f"{0:>5} {r['ll_per_rat']:>8.4f} {r['min_js']:>7.4f} {r['oracle_20']:>7.3f} "
          f"{r['mean_spearman']:>7.4f} {r['leniency_spread']:>8.3f} "
          f"{r['genre_std']:>8.4f} {r['pct_changed']:>7.1f}%")

    # Sweep EM steps 1..max_steps
    for n_steps in range(1, args.max_steps + 1):
        theta, pi, ll, user_asgn = em_n_steps(
            user_ratings, movie_ids, K, theta_init, pi_init, n_steps)

        ev = evaluate_model(theta, pi, movie_ids, K)
        rk = rank_divergence_stats(theta, movie_ids, K)
        gd = genre_divergence(theta, movie_ids, K, movies)

        common = set(asgn_init.keys()) & set(user_asgn.keys())
        n_changed = sum(1 for uid in common if asgn_init[uid] != user_asgn[uid])
        pct = 100 * n_changed / len(common)

        bic = -2 * ll + n_params * np.log(n_ratings)

        row = {
            "em_steps": n_steps,
            "pi": [float(x) for x in pi],
            "ll_per_rat": float(ll / n_ratings),
            "bic": float(bic),
            "min_js": float(ev["min_pairwise_js"]),
            "mean_js": float(ev["mean_pairwise_js"]),
            "oracle_20": float(np.mean(ev["oracle_accuracy"]["20"])),
            "mean_spearman": rk["mean_spearman"],
            "min_spearman": rk["min_spearman"],
            "leniency_spread": rk["leniency_spread"],
            "genre_std": gd["mean_genre_std"],
            "max_genre_std": gd["max_genre_std"],
            "pct_changed": float(pct),
        }
        results.append(row)

        print(f"{n_steps:>5} {row['ll_per_rat']:>8.4f} {row['min_js']:>7.4f} "
              f"{row['oracle_20']:>7.3f} {row['mean_spearman']:>7.4f} "
              f"{row['leniency_spread']:>8.3f} {row['genre_std']:>8.4f} "
              f"{row['pct_changed']:>7.1f}%")

    # Save
    output = {
        "metadata": {"K": K, "n_movies": len(movie_ids), "n_users": len(user_ratings),
                      "n_ratings": n_ratings, "seed": args.seed},
        "sweep": results,
    }
    out_path = Path(__file__).parent / f"em_sweep_k{K}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

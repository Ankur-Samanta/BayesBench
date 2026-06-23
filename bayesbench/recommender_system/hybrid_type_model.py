"""
Hybrid type model selection: K-means genre init → EM refinement.

Stage 1: K-means on centered genre preference vectors (defines taste-based clusters)
Stage 2: EM on rating likelihoods, warm-started from K-means assignments
         (refines theta for discriminability while preserving taste structure)

Evaluates: JS divergence, oracle accuracy, BIC, rank divergence across types.
"""
import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans

from .data import (
    load_movies, load_ratings as load_ratings_raw,
    compute_genre_preferences, GENRE_NAMES, _GENRE_TO_IDX,
)
from .analyze_type_model import evaluate_model


def load_ratings_by_user(ml_dir, valid_movies):
    """Load ratings grouped by user, filtered to valid movies. Returns (dict, total)."""
    raw = load_ratings_raw(ml_dir)
    user_ratings = defaultdict(list)
    total = 0
    for r in raw:
        mid = r["movie_id"]
        if mid in valid_movies:
            user_ratings[r["user_id"]].append((mid, r["rating"]))
            total += 1
    return dict(user_ratings), total


def kmeans_init(user_prefs, qualifying_users, movies, user_ratings,
                movie_ids, n_types, seed=42):
    """Stage 1: K-means on genre preferences. Returns (theta, pi, assignments)."""
    user_ids = sorted(qualifying_users)
    X = np.array([user_prefs[uid] for uid in user_ids])

    kmeans = KMeans(n_clusters=n_types, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X)

    assignments = {uid: int(labels[i]) for i, uid in enumerate(user_ids)}

    # Pi from cluster sizes
    counts = np.bincount(labels, minlength=n_types)
    pi = counts / counts.sum()

    # Theta from rating counts + Dirichlet(1) smoothing
    theta = {mid: np.ones((n_types, 5)) for mid in movie_ids}
    for uid, rats in user_ratings.items():
        if uid not in assignments:
            continue
        k = assignments[uid]
        for mid, r in rats:
            if mid in theta:
                theta[mid][k, r - 1] += 1
    for mid in movie_ids:
        theta[mid] /= theta[mid].sum(axis=1, keepdims=True)

    return theta, pi, assignments


def em_refine(user_ratings, movie_ids, n_types, init_theta, init_pi,
              max_iter=100):
    """Stage 2: EM starting from K-means-derived theta and pi."""
    pi = init_pi.copy()
    theta = {mid: arr.copy() for mid, arr in init_theta.items()}
    n_ratings = sum(len(r) for r in user_ratings.values())

    prev_ll = -np.inf
    responsibilities = {}

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

        new_theta = {mid: np.ones((n_types, 5)) for mid in movie_ids}
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

    user_assignments = {uid: int(np.argmax(resp))
                        for uid, resp in responsibilities.items()}
    return theta, pi, ll, iteration + 1, user_assignments


def rank_divergence_stats(theta, movie_ids, n_types):
    """Compute rank correlation and genre-level divergence between types."""
    n_movies = len(movie_ids)
    exp_ratings = np.zeros((n_types, n_movies))
    for j, mid in enumerate(movie_ids):
        for k in range(n_types):
            exp_ratings[k, j] = sum((r + 1) * theta[mid][k, r] for r in range(5))

    # Pairwise Spearman
    spearman = {}
    for i in range(n_types):
        for j in range(i + 1, n_types):
            rho, _ = spearmanr(exp_ratings[i], exp_ratings[j])
            spearman[(i, j)] = float(rho)

    # Overall mean per type (leniency measure)
    type_means = [float(exp_ratings[k].mean()) for k in range(n_types)]

    return {
        "spearman_pairs": {f"{i}-{j}": v for (i, j), v in spearman.items()},
        "mean_spearman": float(np.mean(list(spearman.values()))),
        "min_spearman": float(min(spearman.values())),
        "max_spearman": float(max(spearman.values())),
        "type_mean_expected_rating": type_means,
        "leniency_spread": float(max(type_means) - min(type_means)),
    }


def genre_divergence(theta, movie_ids, n_types, movies):
    """Compute centered genre preferences per type from theta."""
    n_genres = len(GENRE_NAMES)
    genre_sums = np.zeros((n_types, n_genres))
    genre_counts = np.zeros(n_genres)

    for j, mid in enumerate(movie_ids):
        m = movies.get(mid, {})
        exp_r = [sum((r + 1) * theta[mid][k, r] for r in range(5))
                 for k in range(n_types)]
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

    # Std across types per genre
    genre_std = centered.std(axis=0)

    result = {}
    for gi in range(n_genres):
        if genre_counts[gi] >= 10:
            result[GENRE_NAMES[gi]] = {
                "std_across_types": float(genre_std[gi]),
                "per_type_centered": [float(centered[k, gi]) for k in range(n_types)],
            }

    return {
        "genres": result,
        "mean_genre_std": float(np.mean([v["std_across_types"] for v in result.values()])),
        "max_genre_std": float(max(v["std_across_types"] for v in result.values())),
    }


def run_single_k(K, user_prefs, qualifying_users, movies, user_ratings,
                 movie_ids, n_ratings, seed=42, max_iter=100):
    """Run full hybrid pipeline for a single K. Returns results dict."""
    print(f"\n{'=' * 60}")
    print(f"K={K}: Stage 1 — K-means on genre preferences")
    print(f"{'=' * 60}")

    theta_init, pi_init, asgn_init = kmeans_init(
        user_prefs, qualifying_users, movies, user_ratings,
        movie_ids, K, seed=seed,
    )
    print(f"  K-means pi: {np.round(pi_init, 3)}")

    # Evaluate K-means baseline
    kmeans_eval = evaluate_model(theta_init, pi_init, movie_ids, K)
    kmeans_ll = 0.0
    for uid, rats in user_ratings.items():
        log_likes = np.log(pi_init + 1e-30).copy()
        for mid, r in rats:
            for k in range(K):
                log_likes[k] += np.log(theta_init[mid][k, r - 1] + 1e-30)
        mx = np.max(log_likes)
        kmeans_ll += mx + np.log(np.sum(np.exp(log_likes - mx)))
    n_params = K * len(movie_ids) * 4 + (K - 1)
    kmeans_bic = -2 * kmeans_ll + n_params * np.log(n_ratings)

    print(f"  K-means LL/rat: {kmeans_ll / n_ratings:.4f}, BIC: {kmeans_bic:.0f}")
    print(f"  K-means min JS: {kmeans_eval['min_pairwise_js']:.4f}, "
          f"oracle@20: {kmeans_eval['oracle_accuracy']['20']:.3f}")

    # Rank divergence for K-means
    kmeans_rank = rank_divergence_stats(theta_init, movie_ids, K)
    print(f"  K-means mean Spearman rho: {kmeans_rank['mean_spearman']:.4f}, "
          f"leniency spread: {kmeans_rank['leniency_spread']:.3f}")

    print(f"\n{'=' * 60}")
    print(f"K={K}: Stage 2 — EM refinement (warm-started from K-means)")
    print(f"{'=' * 60}")

    theta, pi, ll, n_iter, user_asgn = em_refine(
        user_ratings, movie_ids, K, theta_init, pi_init,
        max_iter=max_iter,
    )
    print(f"  EM converged iter {n_iter}, LL/rat: {ll / n_ratings:.4f}")
    print(f"  EM pi: {np.round(pi, 3)}")

    # Evaluate hybrid
    hybrid_eval = evaluate_model(theta, pi, movie_ids, K)
    hybrid_bic = -2 * ll + n_params * np.log(n_ratings)

    print(f"  Hybrid min JS: {hybrid_eval['min_pairwise_js']:.4f}, "
          f"oracle@20: {hybrid_eval['oracle_accuracy']['20']:.3f}")

    # Rank divergence for hybrid
    hybrid_rank = rank_divergence_stats(theta, movie_ids, K)
    print(f"  Hybrid mean Spearman rho: {hybrid_rank['mean_spearman']:.4f}, "
          f"leniency spread: {hybrid_rank['leniency_spread']:.3f}")

    # Genre divergence
    hybrid_genre = genre_divergence(theta, movie_ids, K, movies)
    kmeans_genre = genre_divergence(theta_init, movie_ids, K, movies)
    print(f"  Genre std — K-means: {kmeans_genre['mean_genre_std']:.4f}, "
          f"Hybrid: {hybrid_genre['mean_genre_std']:.4f}")

    # How many users changed type?
    common_users = set(asgn_init.keys()) & set(user_asgn.keys())
    n_changed = sum(1 for uid in common_users if asgn_init[uid] != user_asgn[uid])
    pct_changed = 100 * n_changed / len(common_users) if common_users else 0
    print(f"  Users who changed type: {n_changed}/{len(common_users)} ({pct_changed:.1f}%)")

    # Per-type oracle
    for k in range(K):
        orc = hybrid_eval['per_type_oracle_accuracy'].get(str(k), {}).get('20', 0)
        print(f"    Type {k} (pi={pi[k]:.3f}): oracle@20={orc:.3f}")

    return {
        "K": K,
        "kmeans": {
            "pi": [float(x) for x in pi_init],
            "log_likelihood": float(kmeans_ll),
            "per_rating_ll": float(kmeans_ll / n_ratings),
            "bic": float(kmeans_bic),
            "n_free_params": n_params,
            **kmeans_eval,
            "rank_divergence": kmeans_rank,
            "genre_divergence": kmeans_genre,
        },
        "hybrid": {
            "pi": [float(x) for x in pi],
            "log_likelihood": float(ll),
            "per_rating_ll": float(ll / n_ratings),
            "bic": float(hybrid_bic),
            "n_free_params": n_params,
            "em_iterations": n_iter,
            "users_changed_type": n_changed,
            "pct_users_changed": float(pct_changed),
            **hybrid_eval,
            "rank_divergence": hybrid_rank,
            "genre_divergence": hybrid_genre,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid type model: K-means genre init + EM refinement")
    parser.add_argument("--k", type=int, required=True, help="Number of types")
    parser.add_argument("--em-max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).parent / "data"
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    ml_dir = data_dir / "ml-1m"

    # Load data
    print("Loading data...")
    movies = load_movies(ml_dir)

    # Valid movie set from current fitted model
    with open(data_dir / "fitted_model.json") as f:
        current = json.load(f)
    valid_movies = set(int(k) for k in current["theta"].keys())
    movie_ids = sorted(valid_movies)
    print(f"  Movies: {len(movie_ids)}")

    # Ratings by user
    user_ratings, n_ratings = load_ratings_by_user(ml_dir, valid_movies)
    print(f"  Users: {len(user_ratings)}, Ratings: {n_ratings}")

    # Genre preferences
    raw_ratings = load_ratings_raw(ml_dir)
    user_ratings_for_genre = defaultdict(list)
    for r in raw_ratings:
        entry = dict(r)
        entry["star_rating"] = entry["rating"]
        user_ratings_for_genre[entry["user_id"]].append(entry)
    user_prefs, qualifying = compute_genre_preferences(
        user_ratings_for_genre, movies, min_ratings=20)
    print(f"  Users with genre prefs: {len(qualifying)}")

    # Run
    result = run_single_k(
        args.k, user_prefs, qualifying, movies, user_ratings,
        movie_ids, n_ratings, seed=args.seed, max_iter=args.em_max_iter,
    )

    # Add metadata
    output = {
        "metadata": {
            "n_users": len(user_ratings),
            "n_movies": len(movie_ids),
            "n_ratings": n_ratings,
            "seed": args.seed,
            "em_max_iter": args.em_max_iter,
        },
        "result": result,
    }

    out_path = output_dir / f"hybrid_model_k{args.k}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

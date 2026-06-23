"""PCA + K-means + limited EM sweep.

Sweep over:
- n_pcs: number of PCA components to keep (2, 3, 4, 5)
- K: number of types (3, 4, 5)
- em_steps: 0 (pure K-means on PCA), 1, 2, 3

Reports the same metrics as sweep_em_iterations_v2.
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from .data import (
    load_movies, load_ratings as load_ratings_raw,
    compute_genre_preferences, GENRE_NAMES, _GENRE_TO_IDX,
)
from .analyze_type_model import evaluate_model
from .hybrid_type_model import load_ratings_by_user
from .sweep_em_iterations import em_n_steps
from .sweep_em_iterations_v2 import pairwise_genre_divergence, pairwise_spearman


def kmeans_on_pca(X_pca, user_ids, user_ratings, movie_ids, n_types, seed=42):
    """K-means on PCA-transformed genre prefs. Returns (theta, pi, assignments)."""
    kmeans = KMeans(n_clusters=n_types, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    assignments = {uid: int(labels[i]) for i, uid in enumerate(user_ids)}

    counts = np.bincount(labels, minlength=n_types)
    pi = counts / counts.sum()

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


def main():
    data_dir = Path(__file__).parent / "data"
    ml_dir = data_dir / "ml-1m"

    print("Loading data...")
    movies = load_movies(ml_dir)

    with open(data_dir / "fitted_model.json") as f:
        current = json.load(f)
    valid_movies = set(int(k) for k in current["theta"].keys())
    movie_ids = sorted(valid_movies)

    user_ratings, n_ratings = load_ratings_by_user(ml_dir, valid_movies)

    raw = load_ratings_raw(ml_dir)
    urg = defaultdict(list)
    for r in raw:
        e = dict(r); e["star_rating"] = e["rating"]
        urg[e["user_id"]].append(e)
    user_prefs, qualifying = compute_genre_preferences(urg, movies, min_ratings=20)

    user_ids = sorted(qualifying)
    X = np.array([user_prefs[uid] for uid in user_ids])
    print(f"  Users: {len(user_ids)}, Movies: {len(movie_ids)}, Ratings: {n_ratings}")

    # Fit PCA
    pca = PCA()
    X_all = pca.fit_transform(X)

    results = []

    header = (f"{'nPC':>3} {'K':>2} {'EM':>2} {'Orc@20':>7} {'MinJS':>7} {'GenStd':>7} "
              f"{'MinGL2':>7} {'MnGL2':>7} "
              f"{'MinRho':>7} {'MnRho':>7} {'MxRho':>7} "
              f"{'Leniency':>8} {'MinPi':>6}")
    print(f"\n{header}")
    print("-" * len(header))

    for n_pcs in [2, 3, 4, 5]:
        X_pca = X_all[:, :n_pcs]

        for K in [3, 4, 5]:
            # K-means on PCA
            theta_init, pi_init, asgn_init = kmeans_on_pca(
                X_pca, user_ids, user_ratings, movie_ids, K, seed=42)

            for em_steps in [0, 1, 2, 3]:
                if em_steps == 0:
                    theta, pi, asgn = theta_init, pi_init, asgn_init
                    # compute LL
                    ll = 0.0
                    for uid, rats in user_ratings.items():
                        log_likes = np.log(pi + 1e-30).copy()
                        for mid, r in rats:
                            for k in range(K):
                                log_likes[k] += np.log(theta[mid][k, r - 1] + 1e-30)
                        mx = np.max(log_likes)
                        ll += mx + np.log(np.sum(np.exp(log_likes - mx)))
                else:
                    theta, pi, ll, asgn = em_n_steps(
                        user_ratings, movie_ids, K, theta_init, pi_init, em_steps)

                ev = evaluate_model(theta, pi, movie_ids, K)
                gd = pairwise_genre_divergence(theta, movie_ids, K, movies)
                sp = pairwise_spearman(theta, movie_ids, K)

                row = {
                    "n_pcs": n_pcs,
                    "K": K,
                    "em_steps": em_steps,
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
                    "min_pi": float(min(pi)),
                }
                results.append(row)

                print(f"{n_pcs:>3} {K:>2} {em_steps:>2} "
                      f"{row['oracle_20']:>7.3f} {row['min_js']:>7.4f} "
                      f"{row['genre_std']:>7.4f} {row['min_genre_l2']:>7.4f} "
                      f"{row['mean_genre_l2']:>7.4f} "
                      f"{row['min_rho']:>7.4f} {row['mean_rho']:>7.4f} "
                      f"{row['max_rho']:>7.4f} "
                      f"{row['leniency_spread']:>8.3f} {row['min_pi']:>6.3f}")

    out_path = Path(__file__).parent / "pca_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

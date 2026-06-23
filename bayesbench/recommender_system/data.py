"""
Data pipeline for recommender-system cold-start experiments.

Downloads MovieLens 1M, fits K-type model (default K=4) via PCA + K-means on genre
preferences + limited EM refinement, computes per-(type, movie) Categorical
distributions over 1-5 star ratings, selects probe/target movies, and generates
rating sequences.
"""

import json
import os
import zipfile
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
# ML-1M uses string genre labels in movies.dat; collect all unique genres at parse time
GENRE_NAMES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
N_GENRES = len(GENRE_NAMES)
_GENRE_TO_IDX = {g: i for i, g in enumerate(GENRE_NAMES)}


@dataclass
class TypeModel:
    """Fitted user type model from K-means clustering."""
    n_types: int
    pi: np.ndarray                          # (K,) mixture weights
    theta: Dict[int, np.ndarray]            # movie_id -> (K, 5) per-type rating distributions
    movie_names: Dict[int, str]
    movie_genres: Dict[int, List[str]]
    user_assignments: Dict[int, int]        # user_id -> type

    def to_dict(self) -> Dict:
        return {
            "n_types": self.n_types,
            "pi": self.pi.tolist(),
            "theta": {str(k): v.tolist() for k, v in self.theta.items()},
            "movie_names": {str(k): v for k, v in self.movie_names.items()},
            "movie_genres": {str(k): v for k, v in self.movie_genres.items()},
            "user_assignments": {str(k): v for k, v in self.user_assignments.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TypeModel":
        return cls(
            n_types=d["n_types"],
            pi=np.array(d["pi"]),
            theta={int(k): np.array(v) for k, v in d["theta"].items()},
            movie_names={int(k): v for k, v in d["movie_names"].items()},
            movie_genres={int(k): v for k, v in d["movie_genres"].items()},
            user_assignments={int(k): v for k, v in d["user_assignments"].items()},
        )


@dataclass
class MovieSelection:
    """Selected probe and target movies for experiments."""
    probe_movie_ids: List[int]              # 50 probe movies
    target_movies: List[Dict]               # K target movies (one per type, most diagnostic for that type)

    def to_dict(self) -> Dict:
        return {
            "probe_movie_ids": self.probe_movie_ids,
            "target_movies": self.target_movies,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "MovieSelection":
        return cls(
            probe_movie_ids=d["probe_movie_ids"],
            target_movies=d["target_movies"],
        )


def build_anonymization_map(
    probe_movie_ids: List[int],
    target_movies: List[Dict],
    type_model: "TypeModel",
) -> Dict:
    """
    Build a mapping from real movie names/genres to anonymized labels.

    Returns:
        Dict with keys:
            movie_name_map: {movie_id: "Item_1", ...}
            genre_map: {"Action": "Feature_A", ...}
            reverse_genre_map: {"Feature_A": "Action", ...}
    """
    # Map genres to abstract features (A, B, C, ...)
    genre_map = {}
    for i, genre in enumerate(GENRE_NAMES):
        label = chr(ord('A') + i) if i < 26 else f"Z{i - 25}"
        genre_map[genre] = f"Feature_{label}"
    reverse_genre_map = {v: k for k, v in genre_map.items()}

    # Map movies to abstract items
    movie_name_map = {}
    idx = 1
    for mid in probe_movie_ids:
        movie_name_map[mid] = f"Item_{idx}"
        idx += 1
    for tm in target_movies:
        mid = tm["movie_id"]
        if mid not in movie_name_map:
            movie_name_map[mid] = f"Item_{idx}"
            idx += 1

    return {
        "movie_name_map": movie_name_map,
        "genre_map": genre_map,
        "reverse_genre_map": reverse_genre_map,
    }


def anonymize_sequence(
    sequence: List[Dict],
    anon_map: Dict,
) -> List[Dict]:
    """Anonymize a rating sequence by replacing movie names and genres."""
    movie_name_map = anon_map["movie_name_map"]
    genre_map = anon_map["genre_map"]

    anon_seq = []
    for r in sequence:
        mid = r["movie_id"]
        anon_genres = [genre_map.get(g, g) for g in r.get("genres", [])]
        anon_seq.append({
            "movie_id": mid,
            "movie_name": movie_name_map.get(mid, f"Item_{mid}"),
            "genres": anon_genres,
            "rating": r["rating"],
        })
    return anon_seq


def download_movielens(data_dir: Path) -> Path:
    """Download and extract MovieLens 1M if not already cached."""
    ml_dir = data_dir / "ml-1m"
    if ml_dir.exists() and (ml_dir / "ratings.dat").exists():
        return ml_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-1m.zip"

    if not zip_path.exists():
        print(f"Downloading MovieLens 1M from {MOVIELENS_URL}...")
        urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
        print("Download complete.")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    return ml_dir


def load_ratings(ml_dir: Path) -> List[Dict]:
    """Parse ratings.dat (:: separated: UserID::MovieID::Rating::Timestamp)."""
    ratings = []
    with open(ml_dir / "ratings.dat", "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            ratings.append({
                "user_id": int(parts[0]),
                "movie_id": int(parts[1]),
                "rating": int(parts[2]),
                "timestamp": int(parts[3]),
            })
    return ratings


def load_movies(ml_dir: Path) -> Dict[int, Dict]:
    """Parse movies.dat (:: separated: MovieID::Title::Genres)."""
    movies = {}
    with open(ml_dir / "movies.dat", "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            movie_id = int(parts[0])
            title = parts[1]
            genres = parts[2].split("|")
            # Build binary genre flags for genre preference computation
            genre_flags = [0] * N_GENRES
            for g in genres:
                if g in _GENRE_TO_IDX:
                    genre_flags[_GENRE_TO_IDX[g]] = 1
            movies[movie_id] = {
                "movie_id": movie_id,
                "title": title,
                "genres": genres,
                "genre_flags": genre_flags,
            }
    return movies


def prepare_ratings(ratings: List[Dict]) -> List[Dict]:
    """
    Prepare ratings: keep all 1-5 star ratings, add star_rating field.
    """
    prepared = []
    for r in ratings:
        prepared.append({**r, "star_rating": r["rating"]})
    return prepared


def compute_genre_preferences(
    user_ratings: Dict[int, List[Dict]],
    movies: Dict[int, Dict],
    min_ratings: int = 20,
) -> Tuple[Dict[int, np.ndarray], List[int]]:
    """
    Compute per-user genre preference vectors (18-dim).

    For each user with >= min_ratings ratings, compute weighted genre preference
    using normalized ratings: (star_rating - 1) / 4 so 5-star=1.0, 1-star=0.0.

    Returns:
        (user_prefs dict, list of qualifying user_ids)
    """
    user_prefs = {}
    qualifying_users = []

    for user_id, ratings in user_ratings.items():
        if len(ratings) < min_ratings:
            continue

        genre_weighted = np.zeros(N_GENRES)
        genre_total = np.zeros(N_GENRES)

        for r in ratings:
            movie = movies.get(r["movie_id"])
            if movie is None:
                continue
            weight = (r["star_rating"] - 1) / 4.0  # normalize to [0, 1]
            for i, flag in enumerate(movie["genre_flags"]):
                if flag == 1:
                    genre_total[i] += 1
                    genre_weighted[i] += weight

        # Avoid division by zero
        pref = np.divide(genre_weighted, genre_total,
                         out=np.zeros(N_GENRES), where=genre_total > 0)
        # Center: subtract user's mean preference so K-means clusters on
        # relative genre preference (e.g. action-lover vs romance-lover)
        # rather than overall positivity
        pref = pref - pref.mean()
        user_prefs[user_id] = pref
        qualifying_users.append(user_id)

    return user_prefs, qualifying_users


def fit_type_model(
    prepared: List[Dict],
    movies: Dict[int, Dict],
    n_types: int = 4,
    n_pca_components: int = 4,
    em_steps: int = 1,
    min_user_ratings: int = 20,
    min_movie_ratings_per_type: int = 5,
    seed: int = 42,
) -> TypeModel:
    """
    Fit user type model via PCA + K-means + limited EM refinement.

    1. Compute genre preference vectors for qualifying users
    2. PCA to reduce to top components (default 4, capturing ~54% variance)
    3. K-means clustering on PCA-projected space
    4. Estimate theta from rating counts with Dirichlet(1,1,1,1,1) smoothing
    5. Run limited EM iterations (default 1) to sharpen rating distributions
       while preserving genre-based type identity
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    # Group ratings by user
    user_ratings = defaultdict(list)
    for r in prepared:
        user_ratings[r["user_id"]].append(r)

    # Compute genre preferences
    user_prefs, qualifying_users = compute_genre_preferences(
        user_ratings, movies, min_user_ratings
    )
    print(f"Qualifying users (>= {min_user_ratings} ratings): {len(qualifying_users)}")

    # PCA on genre preferences
    user_ids = sorted(qualifying_users)
    X = np.array([user_prefs[uid] for uid in user_ids])
    pca = PCA(n_components=n_pca_components)
    X_pca = pca.fit_transform(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    print(f"PCA: {n_pca_components} components, {cum_var[-1]:.1%} variance explained")

    # K-means clustering on PCA space
    kmeans = KMeans(n_clusters=n_types, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    user_assignments = {uid: int(labels[i]) for i, uid in enumerate(user_ids)}

    # Mixture weights
    type_counts = np.bincount(labels, minlength=n_types)
    pi = type_counts / type_counts.sum()
    print(f"K-means type weights: {pi}")

    # Print cluster genre profiles
    for k in range(n_types):
        mask = labels == k
        center = X[mask].mean(axis=0)
        top_genres = np.argsort(center)[::-1][:5]
        genre_strs = [f"{GENRE_NAMES[g]}({center[g]:.2f})" for g in top_genres]
        print(f"  Type {k} ({type_counts[k]} users): {', '.join(genre_strs)}")

    # Build theta from K-means assignments
    # Group ratings by (type, movie, star_rating)
    type_movie_rating_counts = defaultdict(lambda: defaultdict(lambda: np.zeros(5)))
    type_movie_total = defaultdict(lambda: defaultdict(int))

    for r in prepared:
        uid = r["user_id"]
        if uid not in user_assignments:
            continue
        t = user_assignments[uid]
        mid = r["movie_id"]
        star = r["star_rating"]  # 1-5
        type_movie_rating_counts[t][mid][star - 1] += 1
        type_movie_total[t][mid] += 1

    # Filter movies: need min ratings per type
    valid_movie_ids = []
    movie_names = {}
    movie_genres_map = {}
    for mid, movie in movies.items():
        has_enough = all(
            type_movie_total[k][mid] >= min_movie_ratings_per_type
            for k in range(n_types)
        )
        if has_enough:
            valid_movie_ids.append(mid)
            movie_names[mid] = movie["title"]
            movie_genres_map[mid] = movie["genres"]

    # Compute initial theta with Dirichlet(1,1,1,1,1) smoothing
    theta = {}
    for mid in valid_movie_ids:
        probs = np.zeros((n_types, 5))
        for k in range(n_types):
            counts = type_movie_rating_counts[k][mid]
            probs[k] = (counts + 1) / (counts.sum() + 5)
        theta[mid] = probs

    print(f"Movies with sufficient ratings per type: {len(theta)}")

    # EM refinement (limited iterations to sharpen theta without leniency drift)
    if em_steps > 0:
        print(f"Running {em_steps} EM refinement step(s)...")
        # Build user_ratings dict in (movie_id, rating) format for valid movies
        valid_set = set(valid_movie_ids)
        ur_for_em = {}
        for uid in user_ids:
            rats = [(r["movie_id"], r["star_rating"])
                    for r in user_ratings[uid]
                    if r["movie_id"] in valid_set]
            if rats:
                ur_for_em[uid] = rats

        for step in range(em_steps):
            # E-step
            responsibilities = {}
            ll = 0.0
            for uid, rats in ur_for_em.items():
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

            new_theta = {mid: np.ones((n_types, 5)) for mid in valid_movie_ids}
            for uid, rats in ur_for_em.items():
                resp = responsibilities[uid]
                for mid, r in rats:
                    for k in range(n_types):
                        new_theta[mid][k, r - 1] += resp[k]
            for mid in valid_movie_ids:
                new_theta[mid] /= new_theta[mid].sum(axis=1, keepdims=True)

            pi, theta = new_pi, new_theta
            n_ratings_total = sum(len(r) for r in ur_for_em.values())
            print(f"  EM step {step + 1}: LL/rat = {ll / n_ratings_total:.4f}, pi = {np.round(pi, 3)}")

        # Update user assignments from final responsibilities
        user_assignments = {uid: int(np.argmax(resp))
                            for uid, resp in responsibilities.items()}

    return TypeModel(
        n_types=n_types,
        pi=pi,
        theta=theta,
        movie_names=movie_names,
        movie_genres=movie_genres_map,
        user_assignments=user_assignments,
    )


def select_movies(
    type_model: TypeModel,
    prepared: List[Dict],
    n_probes: int = 50,
    min_total_ratings: int = 100,
) -> MovieSelection:
    """
    Select probe and target movies.

    Target selection (one per type): For each type k, find the movie where
    type k is the biggest outlier — farthest from the nearest other type's
    expected rating. Greedy, no duplicates, so each type gets a unique
    "signature" target movie.

    Probe selection: Next n_probes most diagnostic movies (by global
    diagnosticity), excluding targets.
    """
    # Count total ratings per movie
    movie_rating_counts = defaultdict(int)
    for r in prepared:
        movie_rating_counts[r["movie_id"]] += 1

    # Compute expected ratings and diagnosticity for each movie
    stars = np.arange(1, 6)
    expected_ratings_all = {}  # mid -> (K,) array of E[rating|type]
    diagnosticity = {}
    for mid, theta_m in type_model.theta.items():
        e_ratings = theta_m @ stars  # shape (K,)
        expected_ratings_all[mid] = e_ratings
        diagnosticity[mid] = float(e_ratings.max() - e_ratings.min())

    # Filter to well-known movies
    well_known = {
        mid for mid in diagnosticity
        if movie_rating_counts[mid] >= min_total_ratings
    }
    print(f"Well-known movies (>= {min_total_ratings} ratings): {len(well_known)}")

    # --- Per-type target selection (greedy, no duplicates) ---
    # For each type k, find the movie where type k is the biggest outlier:
    # score = distance from type k's E[rating] to the nearest other type.
    # Type k must be either uniquely high or uniquely low (not in the middle).
    n_types = type_model.n_types
    target_movies = []
    target_ids = set()

    for k in range(n_types):
        best_movie = None
        best_score = -1
        for mid in well_known:
            if mid in target_ids:
                continue
            e = expected_ratings_all[mid]
            e_k = e[k]
            others = [e[j] for j in range(n_types) if j != k]
            # Type k must be the unique max or unique min
            if e_k > max(others):
                score = e_k - max(others)
            elif e_k < min(others):
                score = min(others) - e_k
            else:
                score = 0  # type k is in the middle — not distinctive
            if score > best_score:
                best_score = score
                best_movie = mid

        target_ids.add(best_movie)
        e = expected_ratings_all[best_movie]
        e_others_avg = np.mean([e[j] for j in range(n_types) if j != k])
        direction = "high" if e[k] > e_others_avg else "low"
        target_movies.append({
            "movie_id": best_movie,
            "movie_name": type_model.movie_names[best_movie],
            "genres": type_model.movie_genres[best_movie],
            "theta": type_model.theta[best_movie].tolist(),
            "diagnosticity": diagnosticity[best_movie],
            "owner_type": k,
            "direction": direction,
        })

    print(f"\nTarget movies ({n_types}, one per type):")
    for tm in target_movies:
        theta_arr = np.array(tm['theta'])
        e_ratings = theta_arr @ stars
        e_str = ", ".join(f"{e:.2f}" for e in e_ratings)
        print(f"  [type {tm['owner_type']}, {tm['direction']}] {tm['movie_name']}: "
              f"E[r]=[{e_str}], diag={tm['diagnosticity']:.3f}")

    # --- Probe selection: next n_probes most diagnostic (excluding targets) ---
    sorted_movies = sorted(
        [(mid, diagnosticity[mid]) for mid in well_known],
        key=lambda x: x[1], reverse=True
    )

    probe_movie_ids = []
    for mid, d in sorted_movies:
        if mid in target_ids:
            continue
        probe_movie_ids.append(mid)
        if len(probe_movie_ids) >= n_probes:
            break

    print(f"\nProbe movies ({len(probe_movie_ids)}, diagnosticity range: "
          f"{diagnosticity[probe_movie_ids[0]]:.3f} - {diagnosticity[probe_movie_ids[-1]]:.3f}):")
    for mid in probe_movie_ids[:5]:
        name = type_model.movie_names[mid]
        e_ratings = type_model.theta[mid] @ stars
        e_str = ", ".join(f"{e:.2f}" for e in e_ratings)
        print(f"  {name}: E[r]=[{e_str}], diag={diagnosticity[mid]:.3f}")

    return MovieSelection(
        probe_movie_ids=probe_movie_ids,
        target_movies=target_movies,
    )




def generate_synthetic_sequence(
    type_model: TypeModel,
    probe_movie_ids: List[int],
    true_type: int,
    n_ratings: int,
    seed: int,
    replace: bool = False,
) -> List[Dict]:
    """
    Generate synthetic rating sequence.

    Sample n_ratings movies from probe set.

    Args:
        replace: If True, sample with replacement (allows redundant observations).
                 If False (default), sample without replacement.
    """
    rng = np.random.RandomState(seed)
    if replace:
        selected_ids = rng.choice(probe_movie_ids, size=n_ratings, replace=True)
    else:
        selected_ids = rng.choice(probe_movie_ids, size=min(n_ratings, len(probe_movie_ids)),
                                  replace=False)

    sequence = []
    for mid in selected_ids:
        theta_k = type_model.theta[mid][true_type]  # shape (5,)
        rating = int(rng.choice(np.arange(1, 6), p=theta_k))  # sample 1-5
        sequence.append({
            "movie_id": int(mid),
            "movie_name": type_model.movie_names[mid],
            "genres": type_model.movie_genres[mid],
            "rating": rating,
        })

    return sequence


def generate_misleading_genre_sequence(
    type_model: TypeModel,
    probe_movie_ids: List[int],
    target_movie_id: int,
    true_type: int,
    n_ratings: int,
    seed: int,
) -> List[Dict]:
    """
    Generate a sequence where genre-based heuristic disagrees with Bayesian posterior.

    Selects probe movies that share genres with the target movie but where the
    type-conditional expected rating diverges from the genre-average prediction.
    This tests whether the LLM does shallow genre matching or actual inference.

    Strategy: Prioritize probe movies that share >= 1 genre with the target movie.
    Among those, the true_type's rating distribution will drive the Bayesian posterior,
    but a genre-average heuristic would predict based on the target's genre similarity.
    """
    rng = np.random.RandomState(seed)

    target_genres = set(type_model.movie_genres.get(target_movie_id, []))
    if not target_genres:
        # Fallback to standard generation
        return generate_synthetic_sequence(
            type_model, probe_movie_ids, true_type, n_ratings, seed
        )

    # Partition probes into genre-overlapping and non-overlapping
    genre_overlap = []
    no_overlap = []
    for mid in probe_movie_ids:
        probe_genres = set(type_model.movie_genres.get(mid, []))
        if probe_genres & target_genres:
            genre_overlap.append(mid)
        else:
            no_overlap.append(mid)

    # Prefer genre-overlapping movies (makes genre heuristic seem applicable)
    # but the Bayesian posterior should diverge because of cross-type structure
    if len(genre_overlap) >= n_ratings:
        selected_ids = rng.choice(genre_overlap, size=n_ratings, replace=False)
    else:
        # Use all overlapping + fill with non-overlapping
        remaining = n_ratings - len(genre_overlap)
        extra = rng.choice(no_overlap, size=min(remaining, len(no_overlap)), replace=False)
        selected_ids = np.concatenate([genre_overlap, extra])
        rng.shuffle(selected_ids)

    sequence = []
    for mid in selected_ids:
        mid = int(mid)
        theta_k = type_model.theta[mid][true_type]
        rating = int(rng.choice(np.arange(1, 6), p=theta_k))
        sequence.append({
            "movie_id": mid,
            "movie_name": type_model.movie_names[mid],
            "genres": type_model.movie_genres[mid],
            "rating": rating,
        })

    return sequence


def find_real_users(
    prepared: List[Dict],
    type_model: TypeModel,
    probe_movie_ids: List[int],
    target_movie_ids: List[int],
    min_probe_ratings: int = 15,
) -> List[Dict]:
    """
    Find real MovieLens users suitable for real-sequence experiments.

    Requirements: >= min_probe_ratings probe movie ratings + target movie rated.
    """
    probe_set = set(probe_movie_ids)
    target_set = set(target_movie_ids)

    user_ratings = defaultdict(list)
    for r in prepared:
        user_ratings[r["user_id"]].append(r)

    suitable_users = []
    for uid, ratings in user_ratings.items():
        if uid not in type_model.user_assignments:
            continue

        probe_ratings = [r for r in ratings if r["movie_id"] in probe_set]
        target_ratings = [r for r in ratings if r["movie_id"] in target_set]

        if len(probe_ratings) >= min_probe_ratings and len(target_ratings) > 0:
            suitable_users.append({
                "user_id": uid,
                "type": type_model.user_assignments[uid],
                "n_probe_ratings": len(probe_ratings),
                "target_movie_ids_rated": [r["movie_id"] for r in target_ratings],
            })

    print(f"Suitable real users: {len(suitable_users)}")
    return suitable_users


def get_real_user_sequence(
    user_id: int,
    prepared: List[Dict],
    probe_movie_ids: List[int],
    type_model: TypeModel,
) -> List[Dict]:
    """Get real user's probe movie ratings in timestamp order."""
    probe_set = set(probe_movie_ids)

    user_ratings = [r for r in prepared
                    if r["user_id"] == user_id and r["movie_id"] in probe_set]
    user_ratings.sort(key=lambda r: r["timestamp"])

    sequence = []
    for r in user_ratings:
        mid = r["movie_id"]
        sequence.append({
            "movie_id": mid,
            "movie_name": type_model.movie_names.get(mid, f"Movie {mid}"),
            "genres": type_model.movie_genres.get(mid, []),
            "rating": r["star_rating"],  # int 1-5
        })

    return sequence


def prepare_data(
    data_dir: Optional[str] = None,
    n_types: int = 4,
    seed: int = 42,
) -> Tuple[TypeModel, MovieSelection]:
    """
    One-shot: load, prepare, cluster, select. Cache results.

    Args:
        data_dir: Directory for data files. Defaults to recommender_system/data/.
        n_types: Number of user types (K).
        seed: Random seed.

    Returns:
        (TypeModel, MovieSelection)
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    else:
        data_dir = Path(data_dir)

    cache_model = data_dir / "fitted_model.json"
    cache_selection = data_dir / "movie_selection.json"

    # Try loading from cache
    if cache_model.exists() and cache_selection.exists():
        print("Loading cached type model and movie selection...")
        with open(cache_model, "r") as f:
            type_model = TypeModel.from_dict(json.load(f))
        with open(cache_selection, "r") as f:
            movie_selection = MovieSelection.from_dict(json.load(f))
        # Verify cached model matches requested n_types
        if type_model.n_types == n_types:
            print(f"  Type model: {type_model.n_types} types, {len(type_model.theta)} movies")
            print(f"  Probe movies: {len(movie_selection.probe_movie_ids)}")
            print(f"  Target movies: {len(movie_selection.target_movies)}")
            return type_model, movie_selection
        else:
            print(f"  Cached model has {type_model.n_types} types, need {n_types}. Regenerating...")

    # Download and parse MovieLens
    ml_dir = download_movielens(data_dir)
    ratings = load_ratings(ml_dir)
    movies = load_movies(ml_dir)
    print(f"MovieLens 1M: {len(ratings)} ratings, {len(movies)} movies, "
          f"{len(set(r['user_id'] for r in ratings))} users")

    # Prepare (keep all 1-5 star ratings)
    prepared = prepare_ratings(ratings)
    print(f"Prepared: {len(prepared)} ratings (all 1-5 stars kept)")

    # Fit type model
    type_model = fit_type_model(prepared, movies, n_types=n_types, seed=seed)

    # Select movies
    movie_selection = select_movies(type_model, prepared)

    # Cache
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_model, "w") as f:
        json.dump(type_model.to_dict(), f, indent=2)
    print(f"Cached type model to {cache_model}")

    with open(cache_selection, "w") as f:
        json.dump(movie_selection.to_dict(), f, indent=2)
    print(f"Cached movie selection to {cache_selection}")

    return type_model, movie_selection


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare recommender-system data")
    parser.add_argument("--n-types", type=int, default=4, help="Number of user types (K)")
    args = parser.parse_args()

    type_model, movie_selection = prepare_data(n_types=args.n_types)

    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    stars = np.arange(1, 6)
    # Check marginal E[rating] for each target
    for tm in movie_selection.target_movies:
        mid = tm["movie_id"]
        theta_m = type_model.theta[mid]  # (K, 5)
        e_per_type = theta_m @ stars  # (K,)
        e_marginal = float(type_model.pi @ e_per_type)
        print(f"Target: {tm['movie_name']}")
        print(f"  E[rating] per type = {e_per_type}")
        print(f"  E[rating] marginal = {e_marginal:.3f}")
        print(f"  Diagnosticity = {tm['diagnosticity']:.3f}")
        print(f"  Owner type = {tm['owner_type']}, direction = {tm['direction']}")

    # Verify theta shape
    sample_mid = list(type_model.theta.keys())[0]
    print(f"\nTheta shape check: theta[{sample_mid}].shape = {type_model.theta[sample_mid].shape}")
    assert type_model.theta[sample_mid].shape == (args.n_types, 5), \
        f"Expected ({args.n_types}, 5), got {type_model.theta[sample_mid].shape}"

    # Generate sample sequence
    print(f"\nSample synthetic sequence (type 0, 5 ratings):")
    seq = generate_synthetic_sequence(
        type_model, movie_selection.probe_movie_ids, true_type=0, n_ratings=5, seed=42
    )
    for r in seq:
        genres = ", ".join(r["genres"])
        print(f"  {r['movie_name']} ({genres}): {r['rating']} stars")

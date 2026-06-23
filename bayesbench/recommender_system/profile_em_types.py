"""Profile EM K=4 types by genre preferences and top/bottom movies."""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from .data import load_movies, load_ratings as load_ratings_raw, compute_genre_preferences, GENRE_NAMES
from .analyze_type_model import load_ratings, em_fit

data_dir = Path("data/ml-1m")

# Load movies and ratings
movies = load_movies(data_dir)
raw_ratings = load_ratings_raw(data_dir)

# Group ratings by user (compute_genre_preferences expects "star_rating" key)
user_ratings_list = defaultdict(list)
for r in raw_ratings:
    entry = dict(r)
    entry["star_rating"] = entry["rating"]
    user_ratings_list[entry["user_id"]].append(entry)

# Compute genre preferences
user_prefs, qualifying = compute_genre_preferences(user_ratings_list, movies, min_ratings=20)
print(f"{len(qualifying)} users with genre prefs")

# Load valid movie set
with open("data/fitted_model.json") as f:
    current = json.load(f)
valid_movies = set(int(k) for k in current["theta"].keys())
movie_ids = sorted(valid_movies)

ur, total = load_ratings(data_dir, valid_movies)

print("Fitting EM K=4 (seed=42)...")
theta, pi, ll, n_iter, user_asgn = em_fit(ur, movie_ids, 4, max_iter=100, seed=42)
print(f"Done (iter={n_iter}, LL/rat={ll / total:.4f})")
print(f"pi = {np.round(pi, 3)}")

print()
print("=" * 70)
print("Genre profiles of EM K=4 types")
print("=" * 70)

for k in range(4):
    type_users = [uid for uid, t in user_asgn.items() if t == k]
    prefs = [user_prefs[uid] for uid in type_users if uid in user_prefs]
    if not prefs:
        print(f"Type {k}: no genre prefs")
        continue

    mean_pref = np.mean(prefs, axis=0)
    top_idx = np.argsort(mean_pref)[::-1]

    print(f"\nType {k} (pi={pi[k]:.3f}, {len(type_users)} users):")
    top_str = ", ".join(
        f"{GENRE_NAMES[i]}({mean_pref[i]:+.3f})" for i in top_idx[:5]
    )
    bot_str = ", ".join(
        f"{GENRE_NAMES[i]}({mean_pref[i]:+.3f})" for i in top_idx[-5:][::-1]
    )
    print(f"  Top genres:    {top_str}")
    print(f"  Bottom genres: {bot_str}")

    # Top 8 movies by expected rating
    movie_exp = [
        (mid, sum((r + 1) * theta[mid][k, r] for r in range(5)))
        for mid in movie_ids
    ]
    movie_exp.sort(key=lambda x: -x[1])

    print("  Highest-rated movies:")
    for mid, exp in movie_exp[:8]:
        name = movies.get(mid, {}).get("title", str(mid))
        genres = ", ".join(movies.get(mid, {}).get("genres", []))
        print(f"    {exp:.2f}  {name} [{genres}]")

    print("  Lowest-rated movies:")
    for mid, exp in movie_exp[-5:]:
        name = movies.get(mid, {}).get("title", str(mid))
        genres = ", ".join(movies.get(mid, {}).get("genres", []))
        print(f"    {exp:.2f}  {name} [{genres}]")

# Recommender System Cold-Start Benchmark: Design Document

## The Question

**Can LLMs infer a user's latent type from their rating history and use it to predict their rating on an unseen movie?**

More concretely:

1. The model sees a sequence of (movie, rating) pairs from one user
2. From that sequence, it should implicitly infer which of 4 latent user types this person is
3. Then use that inferred type to predict what rating this user would give a new movie it hasn't seen a rating for

The Bayesian ground truth does this optimally — at each timestep it computes the exact posterior over types given the observations, then marginalizes over types to predict the target rating. We measure how closely the LLM tracks that optimal trajectory.

We also **directly elicit** the LLM's type posterior — asking "which profile is this user?" via both MCQ logprobs and chain-of-thought reasoning — and test whether explicit type knowledge improves downstream rating prediction. This separates the question into two parts: (a) can the LLM identify the type? and (b) given the type, can it predict the rating?

We test this in a recommender system cold-start scenario using real MovieLens data. The ground truth is computable in closed form (discrete mixture model), so we can measure exactly how far the LLM is from optimal at every step as evidence accumulates.

---

## Step 1: The Setting — From Real Data to User Types

### Where does the data come from?

MovieLens 1M — roughly 1 million ratings from ~6,000 users on ~3,900 movies. Each rating is (user_id, movie_id, 1-5 stars). This is real data from real users.

### How do we get user types?

Each user with ≥20 ratings gets an 18-dimensional **genre preference vector** (one dimension per MovieLens genre: Action, Adventure, ..., Western). For each genre, we compute the user's average rating on movies of that genre, normalized to [0, 1], then center by subtracting the user's mean across all genres. Centering ensures we cluster on *relative* preferences ("likes crime more than comedy") rather than overall generosity.

#### How we arrived at K=4 types

**The fitting pipeline (what the code does).** Type fitting is a *hybrid* of genre-space clustering and rating-space refinement (`data.py:261` `fit_type_model`, `hybrid_type_model.py`):

1. **Genre-preference vectors.** Each qualifying user (≥20 ratings) gets the centered 18-dim genre vector described above.
2. **PCA → K-means.** Reduce those vectors to 4 principal components (~54% of variance) and run K-means (K=4, `n_init=10`, seed 42). This defines interpretable, taste-based clusters — a type that dislikes animated/children's films, a type that tolerates action thrillers, and so on.
3. **EM refinement.** Estimate the per-(movie, type) rating distributions `theta` from the K-means assignments (with Dirichlet(1) smoothing), then run a small number of EM steps (**default 1**) on the rating matrix, warm-started from those assignments. The E-step soft-assigns users by rating likelihood; the M-step re-estimates `theta`. This sharpens the rating distributions for discriminability while preserving the genre-based type identity, and produces the final user assignments (argmax of the last responsibilities).

The two stages play complementary roles: K-means on genre space gives *interpretable* types, while the EM pass makes their *rating distributions* more separable — which is what the Bayesian posterior actually keys on. This is a deliberate middle ground: pure genre-space clustering left neighboring types' rating distributions poorly separated, but unconstrained EM on the rating matrix alone tended to collapse types onto imbalanced weights. Genre init + light EM refinement keeps the types both interpretable and balanced.

**Why K=4.** We compared K ∈ {3, 4, 5} on three criteria (`analyze_type_model.py` compares K-means vs. EM fitting and sweeps K; `hybrid_type_model.py` evaluates the genre-init + EM-refinement selection; `sweep_em_iterations.py` checks EM convergence; `pca_sweep.py` explores PCA preprocessing):

- **Minimum pairwise Jensen-Shannon divergence** between the per-type rating distributions. JS measures how distinguishable two distributions are (0 = identical, log 2 = disjoint). We take the *minimum* over all pairs because the bottleneck pair — the two hardest-to-tell-apart types — caps how well any agent (oracle or LLM) can do type inference. A high *mean* JS is meaningless if one pair is confusable. Higher is better.

- **Oracle-minus-chance accuracy at t=20.** The Bayesian oracle's type-classification accuracy after 20 observations, minus the 1/K chance baseline. Subtracting chance makes different K comparable (K=3 has a 33% baseline, K=5 a 20% one). This measures how much *above* random the oracle gets — the actual difficulty signal. Higher is better.

- **Balanced type weights.** No type should collapse to a near-empty fraction, or the benchmark has too few examples of it and the chance-baseline calculation degrades.

K=4 gave the best-separated types (highest minimum pairwise JS, no confusable pair) with the strongest oracle-minus-chance signal and no degenerate type — K=3 forced a large majority class that weakened the signal, and K=5 reintroduced a confusable pair and a near-empty type. The fitted mixture weights are **π = [0.122, 0.178, 0.284, 0.417]** — two larger and two smaller types, all well-represented. The fitted model is cached in `data/fitted_model.json`.

*Aren't real users more continuous?* Yes — this is a deliberate simplification. We need a model where ground truth is computable in closed form. Discrete types give us that. A continuous model (like PMF) would require approximate inference, making the "ground truth" itself uncertain.

**Code:** `data.py:145` — `download_movielens()`, `data.py:214` — `compute_genre_preferences()`, `data.py:261` — `fit_type_model()`, `hybrid_type_model.py` — hybrid K selection.

---

## Step 2: The Generative Model and Bayesian Ground Truth

This step defines (a) the probability model that generates user ratings, and (b) the exact Bayesian inference procedure that serves as ground truth. These are two sides of the same coin — the model defines the data-generating process, and Bayes' rule tells us how to optimally invert it.

### The model parameters

The model has two parameters, both estimated from MovieLens data:

**`pi`** — mixture weights over user types:
```
pi = [pi_0, pi_1, pi_2, pi_3]    e.g., [0.122, 0.178, 0.284, 0.417]
```
`pi_k` is the fraction of users belonging to type k. This is the **prior** — before seeing any ratings, our belief about a new user's type.

**`theta`** — rating distributions per (movie, type) pair:
```
theta[m][k] = [p_1, p_2, p_3, p_4, p_5]    probability of each star rating (1-5)
```
For every movie m and type k, `theta[m][k]` is a probability distribution over {1★, 2★, 3★, 4★, 5★}. It answers: *if a user of type k rates movie m, what's the probability of each star rating?*

Estimated from data: count how many type-k users gave movie m each star rating, then smooth with a Dirichlet(1,1,1,1,1) prior (adds 1 pseudocount per star to avoid zero probabilities):
```
theta[m][k] = (counts + 1) / (total + 5)
```

Only movies where every type has ≥5 ratings survive this filter (~1,723 movies).

### The generative process

A user's ratings are generated as follows:

```
1. Draw the user's type once:           k ~ Categorical(pi)
2. For each movie m they rate:          rating ~ Categorical(theta[m][k])
```

This is a standard mixture model. A user is one of 4 types (drawn once, fixed for all their ratings), and each rating is an independent draw from that type's distribution for that movie. Different types have different taste profiles — one type has low P(4–5★) for animated/children's films while another rates them highly, and so on.

### The Bayesian posterior — step by step

Given this generative model, we can compute the exact posterior over user types after observing any sequence of ratings. This is the ground truth we compare the LLM against.

**At t=0 (no observations):**

We know nothing about this user. Our belief over their type is just the prior:

```
P(type = k) = pi[k]
```

The best prediction for any target movie is the population-weighted average:

```
E[rating | target] = Σ_k  pi[k] × E[rating | type=k, target]
```

where `E[rating | type=k, target] = Σ_{r=1}^{5}  r × theta[target][k][r-1]` is the expected rating that type k gives the target movie.

**After observing one rating (movie_1, rating_1):**

We apply Bayes' rule. The key quantity is the **likelihood** — how probable was this rating under each type?

```
L(k) = P(rating_1 | type=k, movie_1) = theta[movie_1][k][rating_1 - 1]
```

This is just a lookup: go to theta for movie_1, type k, and read off the probability of the observed star rating.

The **unnormalized posterior** is prior × likelihood:

```
P(type=k | obs_1) ∝ pi[k] × L(k)
```

**Normalize** by dividing by the sum over all types:

```
P(type=k | obs_1) = pi[k] × L(k)  /  Σ_j pi[j] × L(j)
```

**After observing t ratings {(movie_1, rating_1), ..., (movie_t, rating_t)}:**

Because ratings are conditionally independent given the type, the likelihood factors:

```
P(type=k | obs_{1:t}) ∝ pi[k] × Π_{i=1}^{t}  theta[movie_i][k][rating_i - 1]
```

Each new observation multiplies in one more likelihood term. Normalize to sum to 1.

**The predicted rating at any timestep t:**

```
E[rating | obs_{1:t}, target] = Σ_k  P(type=k | obs_{1:t}) × E[rating | type=k, target]
```

This is the ground truth prediction: a weighted average of each type's expected rating for the target, weighted by how likely each type is given everything observed so far.

### Concrete example

Suppose the target movie is Dumbo (1941) — type 0's signature target — and the expected ratings per type are:

```
E[rating | type=0, Dumbo] = 1.92    (type 0 dislikes it)
E[rating | type=1, Dumbo] = 3.66    (type 1 likes it)
E[rating | type=2, Dumbo] = 3.91    (type 2 likes it)
E[rating | type=3, Dumbo] = 3.66    (type 3 likes it)
```

**At t=0**, the prediction is the population average (prior π = [0.122, 0.178, 0.284, 0.417]):

```
E[rating] = 0.122×1.92 + 0.178×3.66 + 0.284×3.91 + 0.417×3.66 = 3.52
```

**At t=1**, the user rates Cinderella (Animation, Children's) as 2★. We look up the likelihood of 2★ for each type:

```
P(2★ | type=0, Cinderella) = 0.330    (type 0 often rates animated films low)
P(2★ | type=1, Cinderella) = 0.046    (type 1 rarely)
P(2★ | type=2, Cinderella) = 0.019    (type 2 almost never — it loves animated films)
P(2★ | type=3, Cinderella) = 0.087    (type 3 sometimes)
```

Unnormalized posteriors (prior × likelihood):

```
type 0: 0.122 × 0.330 = 0.0403    ← biggest jump
type 1: 0.178 × 0.046 = 0.0082
type 2: 0.284 × 0.019 = 0.0054
type 3: 0.417 × 0.087 = 0.0363    (still large — it had the biggest prior)
```

Normalize (divide by sum = 0.0901):

```
type 0: 0.447     (was 0.122, up sharply — type 0 is the animation-disliker)
type 1: 0.091     (was 0.178)
type 2: 0.060     (was 0.284, down a lot — type 2 loves animated films, so 2★ is unlikely)
type 3: 0.402     (was 0.417)
```

One observation and the posterior has already shifted substantially. Types 0 and 3 are now the leading hypotheses, and type 2 has been nearly ruled out. The predicted rating for Dumbo shifts:

```
E[rating] = 0.447×1.92 + 0.091×3.66 + 0.060×3.91 + 0.402×3.66 = 2.90
```

Down from 3.52, because the mass that moved onto type 0 (which dislikes Dumbo at 1.92) pulls the prediction down.

**As more ratings accumulate**, each one multiplies in another likelihood term. If the user keeps rating animated/children's films low — like type 0 — the posterior concentrates (type 3, which merely rates these moderately rather than harshly, gets ruled out):

```
After 1 rating:   type 0 posterior ≈ 0.45
After 5 ratings:  type 0 posterior ≈ 0.99
```

And the predicted rating converges toward type 0's expected rating (≈1.92).

*This trajectory — from the population average at t=0, through increasing certainty, to convergence on the true type — is the ground truth we compare the LLM against at every timestep.*

**Code:** `metrics.py:14` — `mixture_posterior()`, `metrics.py:45` — `expected_rating_bayesian()`, `data.py:261` — `fit_type_model()`

---

## Step 3: Selecting Which Movies to Use

### What is diagnosticity?

Not all movies are equally useful. Some movies are rated ~3.5 by everyone regardless of type — these are uninformative. Others are loved by some types and hated by others — these are **diagnostic**.

```
diagnosticity(m) = max_type(E[rating|type, m]) - min_type(E[rating|type, m])
```

A movie with diagnosticity 1.5 means one type expects ~2.5 and another expects ~4.0. Observing someone's rating on this movie tells you a lot about their type. A movie with diagnosticity 0.2 tells you almost nothing.

### How do we split movies into probes and targets?

**Target selection — one per type (4 targets total):** For each type k, we find the movie where type k is the biggest **outlier** — the movie where type k's expected rating is farthest from the nearest other type. We do this greedily (no duplicates), so each type gets a unique "signature" target movie.

For example (from the fitted model):
- **Type 0's target**: Dumbo (1941) — type 0 expects 1.92★, all others expect 3.66–3.91★. Only type 0 dislikes animated children's films.
- **Type 1's target**: Home Alone 3 (1997) — type 1 expects 3.15★, the only type that tolerates it (others expect 1.55–2.10★).
- **Type 3's target**: The Specialist (1994) — type 3 expects 2.01★, all others expect 3.17–3.27★. Only type 3 dislikes this action thriller.

*Why one per type instead of top-N globally?* Global diagnosticity selection can cluster targets on the same pair of types. If types 0 and 2 are the most different, all targets might discriminate between those two types, leaving types 1, 3, 4 without a target that's really "theirs." Per-type selection guarantees balanced coverage — every type has a target where its distinctive preferences are most visible.

*Why does the target need to be an outlier?* If type k's expected rating is in the middle of the pack (e.g., 3.5 when others range from 2.0 to 4.5), observing the rating doesn't help distinguish type k from its neighbors. When type k is the unique extreme, the Bayesian posterior shifts maximally upon observing evidence consistent with type k.

**Probe selection — next 50 most diagnostic (globally):** After reserving the 4 targets, we take the 50 most diagnostic remaining movies as probes. These form the user's rating history. They're highly diagnostic too — each observation carries strong evidence about the user's type.

*Why separate them?* The whole point is **cross-item transfer**. The LLM observes the user's ratings on probe movies and must predict a target movie it has *never* seen a rating for. If the probe and target sets overlapped, the model could just memorize ratings rather than doing inference.

*Do the probe movies appear in the explicit_types profiles?* Yes — the system prompt shows all 50 probe movies' expected ratings per type. The user's history then reveals their *actual* ratings on those same movies. The model can directly compare "Profile 1 says Home Alone 3 gets 3.4, this user gave it 2.0" to figure out the type.

*Does the target movie appear in the profiles?* No — intentionally omitted. If we included the target's expected rating per type, the model could just identify the type and read off the answer. That's table lookup, not inference. By omitting it, the model must: (1) infer the type from probes, then (2) generalize to predict an unseen movie.

**Code:** `data.py:364` — `select_movies()`

---

## Step 4: Generating a Rating Sequence for One Experiment

### How do we create a synthetic user?

Pick a true type (say type 2). Then:

1. Sample 50 movies from the 50-movie probe set (without replacement by default, so every probe movie appears exactly once)
2. For each movie m, sample a rating from `Categorical(theta[m][type_2])` — the type-2 distribution for that movie

This produces 50 (movie, rating) pairs that are *statistically consistent* with being a type-2 user but have realistic noise. A type-2 user who generally dislikes children's movies might still give Cinderella a 3 — the rating is sampled, not deterministic.

*Why sample rather than use expected values?* Expected values would make the task trivially easy — the model could just pattern-match exact floats. Sampled ratings have noise, which means the posterior doesn't converge immediately. The model has to accumulate evidence across multiple observations, just like a real Bayesian agent would.

*Why without replacement?* With n_ratings=50 and 50 probes, every probe movie appears exactly once. This makes the sequence maximally informative — no redundant observations. We also support `--with-replacement` as a variant that tests how models handle redundant evidence.

**Code:** `data.py:482` — `generate_synthetic_sequence()`

### Sequence variants

- **Misleading genre** (`--misleading-genre`): Prioritizes probe movies that share genres with the target. A genre-based heuristic would seem applicable, but the Bayesian posterior may disagree. Tests whether the LLM is doing shallow genre matching or actual structural inference.

- **Real user sequences** (`--sequence-source real`): Uses actual MovieLens users' ratings in timestamp order. Real sequences have noise patterns that synthetic ones don't.

**Code:** `data.py:520` — `generate_misleading_genre_sequence()`, `data.py:584` — `find_real_users()`

---

## Step 5: The Experimental Design

### What varies across experiments?

The experiment has a **3 × 3 factorial design**: 3 conversation conditions (how ratings are delivered) crossed with 3 population info modes (what background context is given). Every combination is tested.

### The 3 conversation conditions

These control the **format** in which the model receives the user's rating history.

**`single_turn`** — At each poll point, we build a completely fresh `[system, user]` prompt from scratch. The system message has task instructions + population info. The user message lists the full rating history as a flat list, followed by the prediction question. No conversational memory — the model sees a static snapshot every time. This is the cleanest baseline because there's no conversational state that could help or hurt.

**`multi_turn_minimal`** — Ratings arrive one at a time as user messages in a growing conversation. The assistant replies `"Noted."` after each one — an empty acknowledgment that contributes no information. For `explicit_types` and `anonymized`, the conversation uses a type-based state (type classification system prompt, initial question asks about viewer profiles) — identical to `multi_turn_actual` except the assistant says "Noted." instead of "Profile X". For `zero_shot`, the conversation is rating-based (rating system prompt, initial question asks about star ratings). Tests whether conversational format helps (maybe the model attends more to recent messages) or hurts (maybe long conversations degrade attention to earlier evidence).

**`multi_turn_actual`** — Same growing conversation as multi_turn_minimal, except the assistant's reply is its **actual type prediction** (e.g., `"Profile 3"`) instead of `"Noted."`. Both conditions use the same type-based conversation state — asking "Which viewer profile does this user most closely match?" as the first question. At each poll point, the MCQ argmax from the most recent type poll becomes the `last_type_prediction`, which is injected as the assistant's response before the next observation.

This tests **type anchoring**: does committing to "Profile 3" early bias the model toward that type even as new evidence accumulates? If multi_turn_actual differs from multi_turn_minimal, anchoring on the latent variable is present.

*Why inject type predictions ("Profile 3") instead of star ratings?* The core research question is whether LLMs implicitly infer user type. Injecting type predictions directly tests anchoring on the latent variable itself — the thing the model should be uncertain about. Profile labels are scale-invariant: "Profile 3" means the same thing regardless of the A-E mapping, so the conversation state is shared (counterbalancing happens at poll time by rebuilding the system prompt).

**Key comparisons:**
- `single_turn` vs `multi_turn_minimal` → does sequential delivery help or hurt vs. a static snapshot?
- `multi_turn_minimal` vs `multi_turn_actual` → does self-anchoring exist?

### The 3 population information modes

These control **what the model knows about the population** before seeing any ratings. The three modes form a clean triangle, each isolating a specific factor in the Bayesian inference question:

**`zero_shot`** — The system prompt is just the task description and the MCQ rating scale. Three lines. No population info at all. The model must rely entirely on pretraining knowledge — its internalized understanding of how movie preferences correlate — to do cross-item inference. "This user dislikes children's movies, so they might like noir" is the kind of reasoning required, and it has to come from the model's own world knowledge. This tests: **can the LLM do Bayesian-like inference with an implicit prior from pretraining?**

**`explicit_types`** — The system prompt includes the complete generative model: 4 profiles, each showing expected ratings on all 50 probe movies, plus the type's prevalence (e.g., "Profile 1: 12% of users"). The model can directly compare observed ratings to profiles and identify the best match. But the target movie is **intentionally omitted** from the profiles — the model must still generalize from its inferred type to predict an unseen movie. This tests: **given the full generative model, can the LLM use it for Bayesian updating and transfer?**

**`anonymized`** — Identical to explicit_types in numerical structure, but all movie names become abstract labels (Item_1, Item_2, ...) and all genres become abstract features (Feature_A, Feature_B, ...). The model can't use "I know L.A. Confidential is a noir film that cinephiles love." This is the **critical control for genre leakage**: if explicit_types works well but anonymized doesn't, the model was using pretraining knowledge about movies and genres rather than doing structural Bayesian inference from the numbers. This tests: **is the inference structural (from the numbers) or semantic (from world knowledge)?**

**Key comparisons — each isolates one factor:**
- `zero_shot` vs `explicit_types` → does giving the generative model help? (measures the value of explicit population structure)
- `explicit_types` vs `anonymized` → is it real reasoning or genre heuristics? (isolates world knowledge vs. structural inference)
- `zero_shot` vs `anonymized` → zero_shot has world knowledge but no structure; anonymized has structure but no world knowledge. Comparing them reveals which matters more.

*Why only three modes?* Each mode has a clear role in the Bayesian inference question, and every pairwise comparison is interpretable. We considered a `reference_users` condition (raw ratings from example users, no explicit type labels), but it conflates too many factors — format noise, implicit clustering ability, and Bayesian inference — without a clean interpretive payoff. The three-mode design keeps every cell meaningful.

### The `multi_turn_actual` × `zero_shot` exclusion

We skip `multi_turn_actual` for `zero_shot`. In the non-zero_shot conditions, `multi_turn_actual` injects the model's **type prediction** (not its star rating) into the conversation — the model sees "Profile 3" as its own prior response, testing whether committing to a latent type creates anchoring. In `zero_shot`, there are no profiles to classify into, so this design doesn't apply.

This gives **8 cells** (not 9) in the experimental matrix:

| Condition | zero_shot | explicit_types | anonymized |
|---|---|---|---|
| single_turn | rating only | rating + type + cond_rating | rating + type + cond_rating |
| multi_turn_minimal | rating only | rating + type + cond_rating | rating + type + cond_rating |
| multi_turn_actual | **(skip)** | type-in-context + rating + type + cond_rating | type-in-context + rating + type + cond_rating |

### Per-experiment variation

- 4 true types × 4 target movies (one per type) × 5 trials = 80 experiments per cell (the outer loop is a full cross of true_type × target, `runner.py:1984`)
- 8 cells per model → 640 experiments per model
- 5 trials use different random seeds → different rating samples from the same type

**Code:** `conditions.py:104` — `_build_system()`, `conditions.py:184` — `build_single_turn()`, `conditions.py:222` — `init_multi_turn_state()`

### What exactly does the model see?

All examples below use **single_turn** format (a fresh 2-message prompt at each poll). Multi-turn formats use the same content but deliver it as a growing conversation — the reader can infer the difference from the condition descriptions above.

Every prompt is a `[system, user]` message pair. The **system message** contains the task instruction, rating scale, and population information (varies by pop_info). The **user message** contains the focal user's rating history so far and asks for a prediction on the target movie.

*The movie names, per-profile averages, and the illustrative target below are schematic — chosen to show prompt structure, not drawn literally from the fitted model. The real probe and target movies live in `data/movie_selection.json` (the actual per-type targets are Dumbo, Home Alone 3, The Spitfire Grill, and The Specialist).*

#### zero_shot — no population information

The model gets only the task and the rating scale. Everything else must come from pretraining knowledge.

**At t=0** (no observations — the model's unconditional prior):

```
SYSTEM: You are predicting how a user will rate a movie based on their rating history.
        Respond with only A, B, C, D, or E.
        A = 1 star (terrible), B = 2 stars (bad), C = 3 stars (okay),
        D = 4 stars (good), E = 5 stars (great)

USER:   What rating will this user give L.A. Confidential (Crime, Film-Noir, Mystery, Thriller)?
```

*What can the model do here?* Only use its pretraining prior — "L.A. Confidential is generally well-regarded, most people give it 4-5 stars." There's no user-specific information at all.

**At t=3** (after 3 observations):

```
SYSTEM: [same as above]

USER:   The user's ratings so far:
        - Home Alone 3 (Children's, Comedy): 2 stars
        - White Christmas (Musical): 1 stars
        - James and the Giant Peach (Animation, Children's, Musical): 1 stars
        Average rating: 1.3 (3 movies rated).
        What rating will this user give L.A. Confidential (Crime, Film-Noir, Mystery, Thriller)?
```

*What can the model do here?* Use the pattern of low ratings on family/musical movies to infer something about this user's tastes, then predict accordingly. But it has no explicit population structure to compare against — it must rely entirely on pretraining knowledge about how movie preferences correlate.

#### explicit_types — full type profiles with expected ratings

The system prompt includes the complete generative model: 4 profiles, each with expected ratings on all 50 probe movies. The target movie is intentionally **omitted** from the profiles.

**At t=3**:

```
SYSTEM: You are predicting how a user will rate a movie based on their rating history.
        Respond with only A, B, C, D, or E.
        A = 1 star (terrible), B = 2 stars (bad), C = 3 stars (okay),
        D = 4 stars (good), E = 5 stars (great)

        Based on data from our platform, users fall into 4 viewer profiles:

        Profile 1 (12% of users):
        - Home Alone 3 (Children's, Comedy): avg rating: 3.4
        - Angel Heart (Film-Noir, Mystery, Thriller): avg rating: 2.1
        - White Christmas (Musical): avg rating: 3.8
        - James and the Giant Peach (Animation, Children's, Musical): avg rating: 3.6
        - [... all 50 probe movies with type-1 expected ratings ...]

        Profile 2 (18% of users):
        - Home Alone 3 (Children's, Comedy): avg rating: 1.8
        - Angel Heart (Film-Noir, Mystery, Thriller): avg rating: 4.2
        - White Christmas (Musical): avg rating: 2.1
        - James and the Giant Peach (Animation, Children's, Musical): avg rating: 1.9
        - [... same 50 movies with type-2 expected ratings ...]

        Profile 3 (28% of users):
        [... same 50 movies with type-3 expected ratings ...]

        Profile 4 (42% of users):
        [... same 50 movies with type-4 expected ratings ...]

USER:   The user's ratings so far:
        - Home Alone 3 (Children's, Comedy): 2 stars
        - White Christmas (Musical): 1 stars
        - James and the Giant Peach (Animation, Children's, Musical): 1 stars
        Average rating: 1.3 (3 movies rated).
        What rating will this user give L.A. Confidential (Crime, Film-Noir, Mystery, Thriller)?
```

*What can the model do here?* Direct comparison: "This user gave Home Alone 3 a 2 — Profile 1 says 3.4, Profile 2 says 1.8. Closer to Profile 2. White Christmas: user gave 1, Profile 1 says 3.8, Profile 2 says 2.1. Again Profile 2." After enough such comparisons, the model can identify the best-matching profile. But the target movie (L.A. Confidential) is **not listed** in the profiles — the model must generalize from its inferred type to predict an unseen movie.

*Why omit the target?* If L.A. Confidential were in the profiles, the model could just pattern-match to the closest profile and read off the answer — table lookup, not inference. By omitting it, the model must: (1) infer the type from probes, then (2) use its own knowledge of what that type of user would think about L.A. Confidential.

#### anonymized — no world knowledge

Identical structure to explicit_types, but all movie names are replaced with abstract labels (Item_1, Item_2, ...) and genres with abstract features (Feature_A, Feature_B, ...). The numerical structure is preserved exactly.

**At t=3**:

```
SYSTEM: You are predicting how a user will rate an item based on their rating history.
        Respond with only A, B, C, D, or E.
        A = 1 star (terrible), B = 2 stars (bad), C = 3 stars (okay),
        D = 4 stars (good), E = 5 stars (great)

        Based on data from our platform, users fall into 4 profiles:

        Profile 1 (12% of users):
        - Item_14 (Feature_D, Feature_E): avg rating: 3.4
        - Item_22 (Feature_J, Feature_M, Feature_P): avg rating: 2.1
        - Item_40 (Feature_L): avg rating: 3.8
        - Item_8 (Feature_A, Feature_D, Feature_L): avg rating: 3.6
        - [... all 50 probe items ...]

        Profile 2 (18% of users):
        - Item_14 (Feature_D, Feature_E): avg rating: 1.8
        - Item_22 (Feature_J, Feature_M, Feature_P): avg rating: 4.2
        - Item_40 (Feature_L): avg rating: 2.1
        - Item_8 (Feature_A, Feature_D, Feature_L): avg rating: 1.9
        - [... same 50 items with type-2 expected ratings ...]

        [... Profiles 3-4 ...]

USER:   The user's ratings so far:
        - Item_14 (Feature_D, Feature_E): 2 stars
        - Item_40 (Feature_L): 1 stars
        - Item_8 (Feature_A, Feature_D, Feature_L): 1 stars
        Average rating: 1.3 (3 items rated).
        What rating will this user give Item_51 (Feature_F, Feature_J, Feature_M, Feature_P)?
```

*Why is this the critical control?* The model can't use "I know Home Alone 3 is a children's comedy." It must reason purely from the numerical structure: "Item_14 got 2 stars, Profile 1 says 3.4, Profile 2 says 1.8 — closer to Profile 2." If explicit_types works well but anonymized doesn't, the "Bayesian reasoning" was just genre heuristics from pretraining.

### What does the model produce?

The model's response is constrained to a single token. We extract **logprobs** for tokens A through E (checking multiple surface forms: `"A"`, `" A"`, `"a"`, `" a"`) and normalize to get a probability distribution over {1★, 2★, 3★, 4★, 5★}.

For example, if the model's logprobs are:
```
A (1★): -3.2    →  P = 0.04
B (2★): -1.1    →  P = 0.33
C (3★): -1.5    →  P = 0.22
D (4★): -1.3    →  P = 0.27
E (5★): -2.4    →  P = 0.09
```

This gives E[rating] = 1×0.04 + 2×0.33 + 3×0.22 + 4×0.27 + 5×0.09 = 3.09.

*Why logprobs instead of generated text?* Generated text gives one answer. Logprobs give a full distribution — we can compute E[rating], variance, KL divergence against the Bayesian posterior, etc.

### How does multi-turn differ?

In **single_turn**, a fresh `[system, user]` prompt is built from scratch at every poll — stateless. In **multi_turn**, a single type-based conversation state grows over time, and all four polls (rating, type, type-CoT, conditioned-rating) are built from that state at each poll point.

#### Architecture: type-based state for non-zero_shot multi-turn

For `explicit_types` and `anonymized`, **both** `multi_turn_minimal` and `multi_turn_actual` maintain a single type-based conversation state. The conversation system prompt is the type-classification prompt; the first user message asks "Which viewer profile does this user most closely match?" The two conditions differ **only in what the assistant says between observations**:

- `multi_turn_minimal`: assistant says `"Noted."`
- `multi_turn_actual`: assistant says `"Profile X"` (the MCQ argmax from the most recent poll)

At each poll point, four measurements are extracted by **rebuilding the system prompt** from the conversation history — counterbalancing happens at poll time, not in the state itself. This means the state is scale-invariant (both "Noted." and "Profile 3" are independent of the A-E mapping).

For `zero_shot` multi-turn (only `multi_turn_minimal`), a single scale-invariant rating state is maintained (the assistant says "Noted.", independent of the letter mapping), and the cyclic shifts are applied at poll time — there are no profiles to classify into.

#### Turn-by-turn examples

All examples below show the conversation state at t=3 (after 3 observations), with target L.A. Confidential.

##### multi_turn_minimal × zero_shot — rating-based state

```
SYSTEM:    You are predicting how a user will rate a movie based on their rating history.
           Respond with only A, B, C, D, or E.
           A = 1 star (terrible), B = 2 stars (bad), C = 3 stars (okay),
           D = 4 stars (good), E = 5 stars (great)

USER:      What rating will this user give L.A. Confidential (Crime, Film-Noir, Mystery, Thriller)?
ASSISTANT: Noted.
USER:      The user rated Home Alone 3 (Children's, Comedy): 2 stars
ASSISTANT: Noted.
USER:      The user rated White Christmas (Musical): 1 stars
ASSISTANT: Noted.
USER:      The user rated James and the Giant Peach (Animation, Children's, Musical): 1 stars
```

**Polls at t=3:** Only the rating poll (cyclically counterbalanced). System prompt stays as-is; append `"\n\nAverage rating: 1.3 (3 movies rated).\nWhat rating will this user give L.A. Confidential (Crime, Film-Noir, Mystery, Thriller)?"` to the last user message.

##### multi_turn_minimal × explicit_types — type-based state, "Noted." responses

```
SYSTEM:    You are classifying which viewer profile a user belongs to...
           [profile descriptions]
           A = Profile 1, B = Profile 2, C = Profile 3, D = Profile 4

USER:      Which viewer profile does this user most closely match?
ASSISTANT: Noted.
USER:      The user rated Home Alone 3 (Children's, Comedy): 2 stars
ASSISTANT: Noted.
USER:      The user rated White Christmas (Musical): 1 stars
ASSISTANT: Noted.
USER:      The user rated James and the Giant Peach (Animation, Children's, Musical): 1 stars
```

**Four polls at t=3, all built from this state:**

**Type poll** — keep type system, append type question:
```
SYSTEM:    [type system, A=Profile1...D=Profile4, cyclically shifted per poll]
USER:      Which viewer profile does this user most closely match?
ASSISTANT: Noted.
USER:      The user rated Home Alone 3 (Children's, Comedy): 2 stars
... [same conversation] ...
USER:      The user rated James and the Giant Peach (Animation, Children's, Musical): 1 stars

           Avg: 1.3 (3 rated).
           Which viewer profile does this user most closely match?
```

**Rating poll** — swap type system → rating system, append rating question:
```
SYSTEM:    [rating system, A=1★...E=5★, cyclically shifted per poll]
USER:      Which viewer profile does this user most closely match?
ASSISTANT: Noted.
USER:      The user rated Home Alone 3 (Children's, Comedy): 2 stars
... [same conversation] ...
USER:      The user rated James and the Giant Peach (Animation, Children's, Musical): 1 stars

           Average rating: 1.3 (3 movies rated).
           What rating will this user give L.A. Confidential (Crime, Film-Noir, Mystery, Thriller)?
```

**CoT-Type** — swap type system → CoT system, append type question:
```
SYSTEM:    [CoT system — "Think step by step...Answer: Profile N"]
USER:      Which viewer profile does this user most closely match?
ASSISTANT: Noted.
... [same conversation] ...
USER:      The user rated James and the Giant Peach (Animation, Children's, Musical): 1 stars

           Avg: 1.3 (3 rated).
           Which viewer profile does this user most closely match?
```

**Conditioned-rating poll** — swap to rating system, append type Q + "Profile X" + conditioned rating Q:
```
SYSTEM:    [rating system, A=1★...E=5★, cyclically shifted per poll]
USER:      Which viewer profile does this user most closely match?
ASSISTANT: Noted.
... [same conversation] ...
USER:      The user rated James and the Giant Peach (Animation, Children's, Musical): 1 stars

           Avg: 1.3 (3 rated).
           Which viewer profile does this user most closely match?
ASSISTANT: Profile 2
USER:      Based on your prediction of Profile 2, what rating will this user give
           L.A. Confidential (Crime, Film-Noir, Mystery, Thriller)?
```

##### multi_turn_actual × explicit_types — type-based state, "Profile X" responses

```
SYSTEM:    You are classifying which viewer profile a user belongs to...
           [profile descriptions]
           A = Profile 1, B = Profile 2, C = Profile 3, D = Profile 4

USER:      Which viewer profile does this user most closely match?
ASSISTANT: Profile 1                            ← last_type_prediction from t=0 poll
USER:      The user rated Home Alone 3 (Children's, Comedy): 2 stars
ASSISTANT: Profile 2                            ← last_type_prediction from t=1 poll
USER:      The user rated White Christmas (Musical): 1 stars
ASSISTANT: Profile 2                            ← last_type_prediction from t=2 poll
USER:      The user rated James and the Giant Peach (Animation, Children's, Musical): 1 stars
```

**Four polls at t=3:** Identical structure to multi_turn_minimal above — same four poll types (type, rating, type-CoT, conditioned-rating), differing only in that the conversation history contains "Profile X" instead of "Noted." between observations.

The key test: does committing to "Profile 2" early bias the model toward that type even as new evidence accumulates? If multi_turn_actual differs from multi_turn_minimal, anchoring on the latent variable is present.

##### multi_turn_actual × zero_shot — skipped

We skip `multi_turn_actual` for `zero_shot`. There are no profiles to classify into.

**Code:** `conditions.py` — `init_multi_turn_type_state()`, `add_observation_minimal()`, `add_observation_with_type_prediction()`, `build_type_poll_multi_turn()`, `build_rating_poll_from_type_state()`, `build_conditioned_rating_from_type_state()`

---

## Step 6: Extracting Predictions

### How do we get a probability distribution from the LLM?

We force the model to respond with a single letter (A-E), extract the logprobs for those 5 tokens as the next-token prediction, and normalize to get a distribution over {1★, 2★, 3★, 4★, 5★}.

```
A = 1★, B = 2★, C = 3★, D = 4★, E = 5★
```

We check multiple token variants per letter: `{"A", " A", "a", " a"}` and sum their probabilities.

*Why not just use the generated text?* Generated text gives a single answer. Logprobs give a full distribution over all 5 ratings, which is much more informative. We can compute E[rating], variance, KL divergence against the Bayesian posterior, etc.

**Code:** `extraction.py:231` — `extract_rating_probs()`

### What about position bias?

LLMs tend to prefer certain answer positions (e.g. early options A/B) in MCQ tasks — this would bias predictions toward particular ratings. To control for this, we use **cyclic counterbalancing** rather than a simple two-way (forward/reversed) average. For the 5-way rating scale we run **5 shifts**: shift *s* places rating value *r* at letter position `(r − 1 − s) mod 5`, so across the 5 shifts every rating value occupies every answer letter A–E exactly once (including center-anchoring at C). Each shift's logprobs are normalized over A–E, remapped back to canonical rating indices, and averaged across the shifts that carry enough answer-letter mass (`assemble_cyclic`, `common/mcq.py:137`):

```
per-shift:   normalize letter probs over A–E, remap position i → rating (i + shift) mod 5
combine:     average the remapped distributions over valid shifts (answer mass ≥ 1%)
E[rating]:   expectation of the combined distribution
position_bias = (max − min of the per-shift expected index) / 2   # half-spread across shifts
rating_mass  = mean answer-letter mass across valid shifts
```

If the model has no position bias, every shift agrees and `position_bias ≈ 0`. If it favors a position, the shifts disagree and averaging over all 5 cancels the effect — a strictly stronger control than a single forward/reversed pair, because it balances *every* letter position, not just the two endpoints.

*Why does multi-turn use a single state instead of separate counterbalanced states?* In multi-turn, the assistant's responses go into the conversation history. Both "Noted." and "Profile 3" are scale-invariant — they mean the same thing regardless of the letter→option mapping. So the conversation state doesn't need counterbalancing; only the system prompt does, which is rebuilt with each cyclic shift at poll time.

**Code:** `common/mcq.py:137` — `assemble_cyclic()`, `common/mcq.py:268` — `extract_rating_cyclic_batch()`, `extraction.py` — `extract_rating_counterbalanced_cyclic_batch`.

---

## Step 6b: Type Elicitation — Probing the Latent Variable Directly

The rating extraction from Step 6 measures the model's **output** (predicted rating). But the core research question is whether LLMs implicitly infer the user's **latent type** — the hidden variable that connects past ratings to future predictions. Type elicitation probes this latent variable directly.

### Why elicit types?

Rating accuracy alone doesn't tell us *how* the model arrives at its prediction. A model could predict correctly by:
1. Genuine Bayesian inference: infer type → predict rating
2. Genre heuristics: "likes children's movies → probably likes Toy Story 2"
3. Running average anchoring: "rates low on average → predict low"

By asking the model *which type it thinks the user is*, we can distinguish these strategies. If the model's type posterior matches the Bayesian type posterior, it's doing something like Bayesian type inference. If it can't identify types but still predicts ratings well, it's using shortcuts.

### The extraction pipeline per poll point

For `explicit_types` and `anonymized` conditions, each poll point involves **19 inference calls** (each MCQ poll is cyclically counterbalanced over all its answer positions, not a single forward/reversed pair):

| # | Call | What | Cyclic shifts |
|---|------|------|-----------------|
| 1-5 | **Rating logprobs** | MCQ: 5-way star scale, cyclically shifted | 5 |
| 6-9 | **Type logprobs** | MCQ: 4-way profile scale, cyclically shifted | 4 |
| 10 | **Type CoT generation** | Free generation with step-by-step reasoning | 1 (no shift) |
| 11-14 | **CoT follow-up MCQ** | MCQ after CoT: system(MCQ) + user(history) + assistant(CoT) + user(reminder) | 4 |
| 15-19 | **Conditioned rating logprobs** | MCQ with "This user matches Profile X" hint, 5-way | 5 |

For `zero_shot`: only 5 calls (rating logprobs, 5 cyclic shifts), since there are no profiles to classify into.

### Type MCQ (logprobs) — calls 6-9

Same profile descriptions as the rating prompt, but the task instruction changes to type classification:

```
SYSTEM: You are classifying which viewer profile a user belongs to based on their rating history.

       [same profile descriptions as rating prompt]

       Respond with only A, B, C, or D.
       A = Profile 1, B = Profile 2, C = Profile 3, D = Profile 4

USER:  [rating history]
       Which viewer profile does this user most closely match?
```

There are 4 profiles, so the type MCQ uses a 4-way scale (A–D) and is cyclically counterbalanced over **4 shifts** — each profile occupies each of A–D exactly once across the shifts. The same `assemble_cyclic` remap-and-average procedure as rating extraction cancels position bias.

**Code:** `conditions.py` — `build_type_poll_single_turn()`, `extraction.py` — `extract_type_counterbalanced()`

### Type CoT (free generation) — call 10

Same profile descriptions, but with chain-of-thought instructions instead of MCQ:

```
SYSTEM: You are classifying which viewer profile a user belongs to based on their rating history.

       Think step by step about how the user's ratings compare to each profile's
       typical preferences. After your analysis, state your final answer as:
       Answer: Profile [number]

       [profile descriptions]

USER:  [rating history]
       Which viewer profile does this user most closely match?
```

The model generates free text (up to 512 tokens). The CoT reasoning is then used as context for a follow-up MCQ extraction rather than being parsed directly.

### CoT follow-up MCQ — calls 11-14

After CoT generation, we extract a cyclically counterbalanced 4-way type distribution by building a follow-up prompt:

```
SYSTEM: [type MCQ system — 4-way profile scale, cyclically shifted]
USER:   [rating history + type question]
ASSISTANT: [CoT reasoning text from call 10]
USER:   Respond with only A, B, C, or D.
```

Key design decisions:
- **MCQ system prompt (not CoT system):** The system uses the MCQ scale (A=Profile1...D=Profile4), not the CoT system prompt. This ensures the logprobs are extracted over the correct token mapping.
- **Cyclically counterbalanced:** Like all MCQ polls, we run the 4 cyclic shifts and average with `assemble_cyclic`. This gives us `cot_type_distribution` (4-way), `cot_type_prediction` (argmax), `cot_type_scale_bias` (the cyclic position bias), and `cot_type_mass`.
- **Replaces regex parsing:** The old approach parsed `Answer: Profile N` from the CoT text for a hard 1-hot prediction. The follow-up MCQ gives a soft distribution, consistent with every other type/rating poll.
- **Conditioned rating uses follow-up prediction:** the conditioned rating (calls 15-19) uses `cot_type_prediction` from the follow-up MCQ, falling back to the MCQ type argmax if None.

*Why both MCQ and CoT+followup?* Calls 6-9 (direct MCQ) give the model's "fast" implicit judgment from a single token. Calls 10-14 (CoT + follow-up) give the model's "deliberate" judgment after step-by-step reasoning. Comparing them reveals whether explicit reasoning helps or hurts type identification. The follow-up MCQ ensures both produce comparable 4-way distributions.

**Code:** `conditions.py` — `build_type_cot_single_turn()`, `build_type_cot_followup_single_turn()`, `build_type_cot_followup_multi_turn()`, `extraction.py` — `generate_type_cot()`, `extract_type_counterbalanced()`

### Conditioned rating (logprobs) — calls 15-19

Same as the standard rating MCQ, but the user message is prepended with a type hint:

```
USER:  Based on their rating history, this user most closely matches Profile 3.

       [rating history]
       What rating will this user give L.A. Confidential?
```

The type hint uses the **CoT-predicted type** (from the CoT follow-up MCQ), not the direct MCQ type. If CoT parsing fails, falls back to MCQ argmax. This tests: does explicitly telling the model the user's type improve its rating prediction?

*Why use CoT type instead of MCQ type?* CoT represents the model's best "deliberate" type judgment — the result of step-by-step reasoning. MCQ is the model's "fast" implicit judgment from a single token. Using the deliberate judgment as the conditioning signal tests the full pipeline: reason about type → commit to type → predict rating.

**Code:** `conditions.py` — `build_conditioned_single_turn()`, `runner.py` — `_extract_type_elicitation()`; `conditions.py` — `build_conditioned_rating_from_type_state()` (multi-turn), `runner.py` — `_extract_type_elicitation_multi_turn()` (multi-turn)

### Multi-turn type conversation (for all non-zero_shot multi-turn)

Both `multi_turn_minimal` and `multi_turn_actual` with `explicit_types` or `anonymized` use a **single type-based conversation state**. The state's system prompt is the type-classification prompt with the MCQ scale. The two conditions differ only in what the assistant says between observations:

- `multi_turn_minimal`: `"Noted."` — empty acknowledgment, no type commitment
- `multi_turn_actual`: `"Profile X"` — the MCQ argmax from the most recent poll, testing type anchoring

At each poll point, all four measurements (rating, type, type-CoT, conditioned-rating) are built from the conversation state by **swapping the system prompt** and appending the appropriate question. Counterbalancing happens at poll time by rebuilding with each cyclic shift — the conversation state itself is scale-invariant.

The `last_type_prediction` tracker (in `runner.py`) stores the MCQ argmax from each poll and uses it for the next between-observation injection in `multi_turn_actual`. At t=0, defaults to 0 (Profile 1) since no poll has occurred yet.

**Code:** `conditions.py` — `init_multi_turn_type_state()`, `add_observation_minimal()`, `add_observation_with_type_prediction()`, `build_conditioned_rating_from_type_state()`; `runner.py` — `_extract_type_elicitation_multi_turn()`

---

## Step 7: The Polling Protocol

### When do we poll?

With polling frequency k=1, we poll at **every timestep**: t ∈ {0, 1, 2, ..., 50}. That's 51 polls per experiment.

t=0 is the **prior** — no observations yet. This measures the model's unconditional belief. By t=50, the model has seen all 50 probe ratings. The trajectory from t=0 to t=50 shows how the model updates.

*Why poll at every step?* To get a dense trajectory. We want to see whether the model updates smoothly (like a Bayesian agent) or erratically, and whether updates are larger for more diagnostic observations.

### What happens at each poll?

1. Build the prompt (single-turn: fresh prompt with history up to t; multi-turn: growing conversation)
2. Run all cyclic shifts, extract logprobs, remap and average (`assemble_cyclic`)
3. Compute Bayesian ground truth at this t (exact posterior + expected rating)
4. Record everything: E[rating], full 5-way rating distribution, position_bias, rating_mass, Bayesian posterior, type posterior

### Format compliance reminders

All MCQ poll methods append `"\nRespond with only A, B, C, D, or E."` to the final user message at poll time. This ensures the format instruction is close to the extraction point, preventing rating_mass degradation in long multi-turn conversations where the system prompt instruction may be far from the model's generation point.

The reminder is added at poll-building time (not in persistent state), so it doesn't appear in observation messages or conversation history. CoT generation methods (`build_type_cot_single_turn`, `build_type_cot_multi_turn`) are NOT modified — they use free generation, not MCQ.

### Compute budget

Per poll point (non-zero_shot): 5 (rating) + 4 (type MCQ) + 1 (CoT gen) + 4 (CoT follow-up MCQ) + 5 (conditioned rating) = 19 inference calls.
Per poll point (zero_shot): 5 inference calls (rating only).
Per experiment (non-zero_shot): 51 polls × 19 = 969 inference calls.
Per experiment (zero_shot): 51 polls × 5 = 255 inference calls.
Per model: 8 cells × 80 experiments each = 640 experiments, with varying call counts per cell.

**Code:** `runner.py:35` — `get_poll_points()`, `runner.py:302` — `run_single_turn_experiment()`, `runner.py:400` — `run_multi_turn_experiment()`

---

## Step 8: Metrics — What We Measure

### Primary metrics

**MAE from Bayesian** — mean |LLM prediction - Bayesian prediction| across all 51 poll points. This is the headline accuracy measure. Lower is better.

**Cross-item transfer score** — does the LLM exploit *which* movies were rated, or just track the running average?

```
transfer = 1 - (MAE_LLM / MAE_marginal)
```

The marginal baseline ignores movie identity entirely — it treats all observed ratings as direct samples from the target movie and computes a Dirichlet-Multinomial expected value:

```
E_marginal[rating at t] = Σ_r r × (1 + count_r) / (5 + n_total)
```

This returns 3.0 at t=0 (uniform prior) and converges to the sample mean. If transfer > 0, the LLM is doing better than just averaging — it's extracting information from which specific movies were rated. If transfer < 0, the LLM is doing *worse* than the simple average.

*Why is this the headline metric?* Because it isolates exactly the capability we care about. Any model can track a running average. The question is whether it can do cross-item inference — "this user gave a children's movie 1 star, therefore they'll probably rate this thriller differently than someone who gave the children's movie 5 stars."

**Genre transfer score** — same idea but against a stronger baseline:

```
genre_transfer = 1 - (MAE_LLM / MAE_genre_overlap)
```

The genre-overlap baseline averages only observations sharing ≥1 genre with the target. This is the simplest genre-similarity heuristic. If genre_transfer > 0, the LLM is going beyond genre matching — genuine structural inference.

*Why do we need both transfer scores?* Cross-item transfer could be positive just because the LLM uses genre similarity (which is real signal, but shallow). Genre transfer tests whether the model does anything beyond that — e.g., inferring from "this user rates all genres low" that they're a generally harsh rater, even for a genre they haven't been tested on.

**Code:** `metrics.py:71` — `marginal_baseline()`, `metrics.py:91` — `genre_overlap_baseline()`, `metrics.py:135` — `compute_trajectory_metrics()`

### Distributional metrics

**KL divergence** — KL(Bayesian ‖ LLM) on the full 5-way rating distribution at each poll, averaged. Measures information loss from using the LLM's full distribution instead of the true posterior.

**Wasserstein-1 distance** — earth mover's distance on the ordinal 1-5 scale. More interpretable than KL for ordinal data: a distribution shifted by one star has small Wasserstein but can have large KL.

*Why evaluate the full distribution, not just E[rating]?* Two distributions can have the same expected value but very different shapes. A model that always predicts uniform (0.2, 0.2, 0.2, 0.2, 0.2) has E[rating]=3.0 and so does a model predicting (0, 0, 1, 0, 0). The distributional metrics capture calibration — does the model's uncertainty match the Bayesian uncertainty?

### Type elicitation metrics (new)

These metrics probe the model's latent type inference directly. Only computed for `explicit_types` and `anonymized` conditions (not `zero_shot`).

**Type posterior KL** — `KL(Bayesian type posterior ‖ LLM type distribution)` averaged across all poll points. Measures how closely the LLM's type distribution matches the Bayesian posterior over types. Like the rating KL, this captures the full distributional agreement, not just the argmax. Lower is better — 0 means the LLM's type beliefs exactly match the Bayesian agent's.

*Why KL on types, not just accuracy?* A model might assign 40% to the correct type and 35% to another — that's wrong by argmax but pretty good distributionally. KL captures this nuance. It also penalizes overconfidence (putting 99% on a wrong type is worse than spreading mass evenly).

**Type accuracy (MCQ)** — fraction of poll points where `argmax(LLM type distribution) == argmax(Bayesian type posterior)`. The model's "implicit" type judgment from logprobs. This is the headline type metric — simple, interpretable. At t=0 both posteriors are the prior, so a match is expected. The interesting signal is at later timesteps as the posterior concentrates.

**Type accuracy (CoT)** — fraction of poll points where the CoT-parsed type matches the Bayesian argmax. The model's "explicit" type judgment from step-by-step reasoning. Comparing MCQ vs CoT accuracy reveals whether explicit reasoning helps:
- CoT > MCQ → the model benefits from "thinking out loud" about types
- CoT < MCQ → verbalizing reasoning introduces errors (maybe the model is better at implicit pattern matching)
- CoT ≈ MCQ → reasoning doesn't help or hurt

**Conditioned MAE** — mean |conditioned rating - Bayesian rating| across poll points, where the conditioned rating is the model's prediction *after being told* the user's type ("Based on their rating history, this user most closely matches Profile 3"). If conditioned MAE < standard MAE, knowing the type improves rating prediction.

**Conditioning lift** — `MAE_standard - MAE_conditioned`. Positive = conditioning on type helps, negative = it hurts. This is the key "does type knowledge improve downstream prediction?" metric. If it's positive, the bottleneck is type inference, not type→rating mapping. If it's near zero, the model already uses type information implicitly.

**Code:** `metrics.py` — `compute_trajectory_metrics()` (type elicitation section)

### Diagnostic metrics

**Position bias** (recorded as `scale_bias`) — the half-spread of the per-shift expected index across the cyclic shifts, `(max − min) / 2` (`assemble_cyclic`). Measures how much the model's answer depends on where an option sits in the A–E/A–D list. Should be near 0; large values mean the model favors particular answer positions. This is a methodological check, not a substantive finding.

**Rating mass** — total probability mass on the 5 answer tokens (A-E). Should be >0.8 if the model follows the MCQ format. Low mass means the model is generating non-answer tokens (explaining itself, hedging, etc.) and we're only seeing the distribution over a fraction of the output space.

**Update magnitude** — |E[rating]_t - E[rating]_{t-1}| per step. A Bayesian agent's updates are larger for diagnostic observations and decrease as the posterior concentrates. If the LLM's updates are flat or erratic, it's not tracking evidence correctly.

**Trajectory variance** — variance of E[rating] across time. A model that outputs the same prediction regardless of evidence has zero variance. A model that responds to evidence has positive variance.

**Correlation with Bayesian** — Pearson correlation between the LLM's trajectory and the Bayesian trajectory across all 51 poll points. Positive = the LLM moves in the right direction even if the magnitude is off.

**Type inference correct** — does the Bayesian posterior (given the observations) assign highest probability to the true type by the final poll? This is about the ground truth, not the LLM — it tells us whether the probe sequence was informative enough to identify the type.

---

## Step 9: Analysis and Key Comparisons

### How are results grouped?

Results are sliced along every axis:

- **Per (model, condition, pop_info)**: the full 7-model × 8-cell factorial table
- **Per model**: averaged across conditions/pop_info — which models are best overall?
- **Per condition**: averaged across models — does conversation format matter?
- **Per pop_info**: averaged across models — does population info help?
- **Per type**: averaged across models — which user types are easiest/hardest to infer?
- **Per (model, type)**: which models handle which types best?

### Key comparisons

| Comparison | What it tells you |
|---|---|
| `single_turn` vs `multi_turn_minimal` | Does streaming format help or hurt vs. a static snapshot? |
| `multi_turn_minimal` vs `multi_turn_actual` | Does **type anchoring** exist? (committing to a type biases future predictions?) |
| `explicit_types` vs `zero_shot` | Does giving the full generative model help? |
| `explicit_types` vs `anonymized` | Is the "Bayesian reasoning" just genre heuristics from pretraining? |
| `zero_shot` vs `anonymized` | How much does world knowledge contribute? |
| `cross_item_transfer` vs `genre_transfer` | Does inference go beyond genre matching? |
| Per-type breakdown | Are extreme types (strong preferences) easier than moderate ones? |
| Model scaling | Do larger models do better? (3B → 8B → 14B → 20B → 120B) |

### Type elicitation research questions

The new type elicitation pipeline enables five additional research questions:

| # | Question | How to answer |
|---|----------|---------------|
| 1 | **Does the LLM implicitly infer user type?** | Compare `llm_type_distribution` vs Bayesian `type_posterior` (type KL, type accuracy). If type accuracy >> 1/K (random baseline), the model is doing type inference. |
| 2 | **Does explicit reasoning improve type identification?** | Compare `cot_type_accuracy` vs `type_accuracy`. CoT > MCQ means step-by-step reasoning helps the model identify types better than its implicit logprob distribution. |
| 3 | **Does knowing the type improve rating prediction?** | Compare `conditioned_mae` vs `mae_from_bayesian` (conditioning lift). Positive lift = the bottleneck is type inference, not type→rating generalization. |
| 4 | **Does committing to a type create anchoring?** | Compare `multi_turn_actual` vs `multi_turn_minimal` for type accuracy and rating MAE. If multi_turn_actual has lower type accuracy or higher MAE, the model anchors on its early type commitment. |
| 5 | **Does conversational format affect type inference?** | Compare `single_turn` vs `multi_turn_minimal` for type KL and type accuracy. If they differ, sequential delivery changes how the model reasons about types. |

### Profile-level analysis

The per-type breakdown shows which user types are easiest/hardest for LLMs. Extreme types (strong genre preferences, e.g., only likes children's movies) should be easier to identify than moderate types (watches everything, rates near average). If some types are much harder, it reveals what kind of "taste signals" LLMs can and can't pick up on.

**Code:** `analyze_results.py:315` — `print_type_table()`, `analyze_results.py:391` — `print_model_type_table()`

---

## Step 10: Potential Confounds and Limitations

### Genre leakage

Probe movies are selected for high diagnosticity, and user types are defined by genre preferences. This means diagnostic probe movies are ones where genre alone predicts the rating. When a user rates "Dumbo (Animation, Children's)" as 5★ and "Toxic Avenger (Comedy, Horror)" as 2★, the LLM doesn't need Bayesian inference — it can just note "likes animated movies, dislikes horror" and predict accordingly.

The **anonymized** condition is the primary defense: with Item_1, Item_2 etc., the LLM can't use pretraining knowledge about genres. If performance drops sharply in anonymized vs explicit_types, the "Bayesian reasoning" was likely genre heuristics.

### The marginal baseline is deliberately weak

The marginal baseline ignores movie identity entirely. Anything that tracks "this person rates low overall" beats it. The genre-overlap baseline is the better test — it captures the simplest genre heuristic. A positive genre_transfer score is more compelling evidence of genuine inference.

### LLM world knowledge about movies

LLMs have seen movie reviews, IMDB scores, and Reddit discussions during pretraining. Their t=0 prediction (before seeing any ratings) likely reflects this knowledge, not a principled prior. The `prior_expected_rating` metric captures this. The anonymized condition controls for it.

### Discrete types are a simplification

Real users are continuously distributed in preference space, not clustered into 4 archetypes. The "Bayesian ground truth" is itself an approximation — the LLM might actually be doing something more nuanced. But for a benchmark, we need computable ground truth, and the discrete mixture gives us that.

---

## File Reference

| File | Purpose | Key Entry Points |
|---|---|---|
| `config.py` | Dataclasses & enums | `ExperimentConfig`, `PollResult` (with type elicitation fields), `TrajectoryMetrics` (with type metrics) |
| `data.py` | Data pipeline & generative model | `compute_genre_preferences`, `fit_type_model` (genre K-means + EM refinement), `select_movies`, `generate_synthetic_sequence` |
| `hybrid_type_model.py` | K selection (genre-init + EM refinement) | evaluates JS divergence, oracle accuracy, BIC across K |
| `common/mcq.py` | Cyclic counterbalancing primitives | `assemble_cyclic`, `extract_cyclic_batch`, `extract_rating_cyclic_batch` |
| `conditions.py` | Prompt construction | `_build_system`, `build_single_turn`, `build_type_poll_single_turn`, `build_type_cot_single_turn`, `build_type_cot_followup_single_turn`, `build_type_cot_followup_multi_turn`, `build_conditioned_single_turn`, `init_multi_turn_type_state`, `add_observation_with_type_prediction`, `build_conditioned_rating_from_type_state` |
| `extraction.py` | Logprob extraction (cyclic) | `setup_model`, `extract_rating_probs`, `extract_rating_counterbalanced_cyclic_batch`, `extract_type_counterbalanced`, `generate_type_cot` |
| `metrics.py` | Ground truth & metrics | `mixture_posterior`, `expected_rating_bayesian`, `compute_trajectory_metrics` (includes type KL, type accuracy, conditioning lift) |
| `runner.py` | Experiment orchestration | `run_single_turn_experiment`, `run_multi_turn_experiment`, `_extract_type_elicitation`, `_extract_type_elicitation_multi_turn` |
| `analyze_results.py` | Cross-experiment analysis | `print_grouped_table` (with TypeElicAcc, CoTAcc, CondLift columns) |
| `aggregate.py` | Per-model aggregation with bootstrap CIs | `aggregate_by_condition`, `generate_summary_report` |
| `backfill_metrics.py` | Backfill new metrics into existing results without rerunning inference | |
| `submit_experiments.sh` | SLURM launcher for one model (`MODEL=<key>`); sweeps condition × pop-info cells (skips multi_turn_actual × zero_shot) | |

---

## Running Experiments

### Quick test
```bash
python -m bayesbench.recommender_system.runner \
  --model llama3b \
  --condition single_turn \
  --pop-info zero_shot \
  --k 1 --true-type 0 --trials 0 --n-ratings 10
```

### Full sweep for one model
```bash
python -m bayesbench.recommender_system.runner \
  --model qwen7b \
  --condition single_turn \
  --pop-info explicit_types \
  --k 1 --true-type 0 1 2 3 4 --trials 0,1,2,3,4
```

### Experimental variants
```bash
# Anonymized condition (no world knowledge)
python -m bayesbench.recommender_system.runner --model qwen7b --condition single_turn --pop-info anonymized ...

# With replacement (redundant evidence)
python -m bayesbench.recommender_system.runner --model qwen7b --condition single_turn --pop-info zero_shot --with-replacement ...

# Misleading genre sequences
python -m bayesbench.recommender_system.runner --model qwen7b --condition single_turn --pop-info zero_shot --misleading-genre ...

# Real user sequences
python -m bayesbench.recommender_system.runner --model qwen7b --condition single_turn --pop-info zero_shot --sequence-source real ...
```

### SLURM batch submission
```bash
MODEL=qwen7b ./bayesbench/recommender_system/submit_experiments.sh run    # full condition × pop-info sweep for one model
MODEL=qwen7b ./bayesbench/recommender_system/submit_experiments.sh agg    # aggregate that model's saved experiments
./bayesbench/recommender_system/submit_experiments.sh test                # quick validation (1 job)
```

### Backfill metrics on existing results
```bash
python -m bayesbench.recommender_system.backfill_metrics                    # backfill all
python -m bayesbench.recommender_system.backfill_metrics --dry-run           # preview
```

### Analysis
```bash
python -m bayesbench.recommender_system.analyze_results                                                      # summary tables
python -m bayesbench.recommender_system.aggregate --experiments-dir bayesbench/recommender_system/experiments --model qwen7b --output analysis.json
```

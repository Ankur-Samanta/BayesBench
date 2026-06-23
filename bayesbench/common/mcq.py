"""Token-probability MCQ extraction with cyclic counterbalancing.

This is the single polling implementation shared by every BayesBench task. A
model's probability estimate is read off its next-token distribution over the
answer letters (A, B, ...). To cancel position bias we present the options under
all ``K`` cyclic shifts of their ordering — shift ``k`` puts the option at
canonical index ``i`` at letter position ``(i - k) mod K`` — so every option
occupies every letter position exactly once, then average the remapped
distributions (``assemble_cyclic``).

Dimensionality is uniform across tasks: binary tasks (coin_flip heads/tails,
social_judgment YTA/NTA) use ``K=2``; medical triage uses ``K=4``; recommender-system ratings/types use ``K=5``.
"""

import re
from typing import List, Optional, Tuple

import numpy as np


# ── Answer-letter token matching ──────────────────────────────────────────────

A_VARIANTS = {"A", " A", "a", " a"}
B_VARIANTS = {"B", " B", "b", " b"}
C_VARIANTS = {"C", " C", "c", " c"}
D_VARIANTS = {"D", " D", "d", " d"}
E_VARIANTS = {"E", " E", "e", " e"}

LETTERS = ["A", "B", "C", "D", "E"]
LETTER_VARIANTS = [A_VARIANTS, B_VARIANTS, C_VARIANTS, D_VARIANTS, E_VARIANTS]


def is_letter_token(text: str, letter_idx: int) -> bool:
    """Whether token text matches letter A(0), B(1), C(2), D(3), or E(4)."""
    return text.strip().upper() == LETTERS[letter_idx] or text in LETTER_VARIANTS[letter_idx]


def is_a_token(text: str) -> bool:
    return is_letter_token(text, 0)


def is_b_token(text: str) -> bool:
    return is_letter_token(text, 1)


# ── gptoss channel-prefix handling ────────────────────────────────────────────

GPTOSS_FINAL_PREFIX = "<|channel|>final<|message|>"
_channel_prefix_cache = {}


def _needs_channel_prefix(tokenizer) -> bool:
    """Whether this model uses gptoss-style channel structure."""
    try:
        ids = tokenizer.encode("<|channel|>", add_special_tokens=False)
        return len(ids) == 1  # Single token = recognized special token
    except Exception:
        return False


def build_mcq_prompt(tokenizer, messages: List[dict]) -> str:
    """Apply the chat template and append the gptoss final-channel prefix if needed.

    Pure function capturing the prompt-building convention every MCQ extraction
    call uses, so single-prompt and batched paths stay byte-equivalent.
    """
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tok_id = id(tokenizer)
    if tok_id not in _channel_prefix_cache:
        _channel_prefix_cache[tok_id] = _needs_channel_prefix(tokenizer)
    if _channel_prefix_cache[tok_id]:
        prompt += GPTOSS_FINAL_PREFIX
    return prompt


# ── Letter-probability extraction ─────────────────────────────────────────────

def parse_letter_probs(prior_result, n_letters: int = 5) -> Tuple[List[float], float]:
    """Aggregate per-letter probability mass from a PriorExtractionResult.

    Sums all token-text variants for each of the first ``n_letters`` letters by
    walking ``top_tokens``. Returns ``(letter_probs, letter_mass)`` where
    ``letter_probs`` has length ``n_letters``.
    """
    letter_probs = [0.0] * n_letters
    for tok in prior_result.top_tokens:
        for i in range(n_letters):
            if is_letter_token(tok.token_text, i):
                letter_probs[i] += tok.probability
    return letter_probs, sum(letter_probs)


def extract_letter_probs_batch(
    model, tokenizer, messages_list: List[List[dict]], n_letters: int = 5
) -> List[Tuple[List[float], float]]:
    """Batched letter-probability extraction over many prompts.

    Builds prompts for every message list, fires a single
    ``get_token_priors_batch`` call targeting the first ``n_letters`` letters,
    and parses each result. Return value is aligned with ``messages_list``.
    """
    if not messages_list:
        return []
    prompts = [build_mcq_prompt(tokenizer, m) for m in messages_list]
    results = model.get_token_priors_batch(
        prompts,
        target_tokens=LETTERS[:n_letters],
        top_k=20,
        temperature=1.0,
    )
    return [parse_letter_probs(r, n_letters) for r in results]


def extract_letter_probs(
    model, tokenizer, messages: List[dict], n_letters: int = 5
) -> Tuple[List[float], float]:
    """Single-prompt letter-probability extraction (wrapper over the batch path)."""
    return extract_letter_probs_batch(model, tokenizer, [messages], n_letters)[0]


def extract_ab_probs(model, tokenizer, messages: List[dict]) -> Tuple[float, float, float]:
    """Binary convenience: normalized ``(p_a, p_b, ab_mass)`` for an A/B prompt.

    Targets only A/B. Returns ``(0.5, 0.5, mass)`` when the model puts <1% mass
    on the answer letters.
    """
    (a_prob, b_prob), ab_mass = extract_letter_probs(model, tokenizer, messages, n_letters=2)
    if ab_mass < 0.01:
        return 0.5, 0.5, ab_mass
    return a_prob / ab_mass, b_prob / ab_mass, ab_mass


# ── Cyclic counterbalancing assembly ──────────────────────────────────────────

def assemble_cyclic(
    letter_probs_per_shift: List[List[float]],
    masses_per_shift: List[float],
    n: int,
) -> Tuple[List[float], int, float, float]:
    """Assemble per-shift letter probabilities into one canonical distribution.

    Given the letter probabilities and total masses for each of the ``n`` cyclic
    shifts of a single probe, normalize each shift over its first ``n`` letter
    positions, remap positions back to canonical option indices, average across
    shifts with sufficient mass, and return
    ``(distribution, predicted_index, position_bias, mass)``.

    Shifts whose answer-letter mass is below 1% are excluded from the average
    (rather than averaged in as uniform); ``position_bias`` is the half-spread
    of the per-shift expected index across valid shifts.
    """
    dists: List[List[float]] = []
    for shift, (letter_probs, mass) in enumerate(zip(letter_probs_per_shift, masses_per_shift)):
        if mass < 0.01:
            dists.append([1.0 / n] * n)
        else:
            norm = [p / mass for p in letter_probs[:n]]
            # Position i in this shift corresponds to canonical index (i + shift) % n.
            remapped = [0.0] * n
            for i in range(n):
                remapped[(i + shift) % n] = norm[i]
            dists.append(remapped)

    valid = [(d, m) for d, m in zip(dists, masses_per_shift) if m >= 0.01]
    if not valid:
        distribution = [1.0 / n] * n
        mass = float(np.mean(masses_per_shift))
        position_bias = 0.0
    else:
        valid_dists = [d for d, _ in valid]
        distribution = [
            sum(d[k] for d in valid_dists) / len(valid_dists) for k in range(n)
        ]
        mass = float(np.mean([m for _, m in valid]))
        expected_indices = [sum(k * d[k] for k in range(n)) for d in valid_dists]
        position_bias = (max(expected_indices) - min(expected_indices)) / 2

    predicted_index = int(np.argmax(distribution))
    return distribution, predicted_index, position_bias, mass


def extract_cyclic_batch(
    model,
    tokenizer,
    messages_per_experiment: List[List[List[dict]]],
    n: int = 4,
    n_letters: int = 5,
) -> List[Tuple[List[float], int, float, float]]:
    """Cyclic-counterbalanced K-way extraction over many probes (batched).

    Each element of ``messages_per_experiment`` is a list of ``n`` probe message
    lists, one per cyclic shift (shift order 0..n-1). All (experiment, shift)
    prompts go through a single batched call; one
    ``(distribution, predicted_index, position_bias, mass)`` is assembled per
    experiment, in input order.
    """
    if not messages_per_experiment:
        return []
    for i, exp_msgs in enumerate(messages_per_experiment):
        if len(exp_msgs) != n:
            raise ValueError(
                f"messages_per_experiment[{i}] has {len(exp_msgs)} shifts, expected {n}"
            )

    flat = [m for exp in messages_per_experiment for m in exp]
    flat_letter_probs = extract_letter_probs_batch(model, tokenizer, flat, n_letters)

    results = []
    for i in range(len(messages_per_experiment)):
        block = flat_letter_probs[i * n:(i + 1) * n]
        lps = [lp for lp, _ in block]
        masses = [mass for _, mass in block]
        results.append(assemble_cyclic(lps, masses, n))
    return results


def extract_cyclic(
    model,
    tokenizer,
    shift_messages: List[List[dict]],
    n: int = 4,
    n_letters: int = 5,
) -> Tuple[List[float], int, float, float]:
    """Single-probe cyclic-counterbalanced K-way extraction.

    ``shift_messages`` is the list of ``n`` per-shift message lists for one
    probe. Wrapper over ``extract_cyclic_batch``.
    """
    return extract_cyclic_batch(
        model, tokenizer, [shift_messages], n=n, n_letters=n_letters
    )[0]


def extract_cyclic_with_shifts(
    model,
    tokenizer,
    shift_messages: List[List[dict]],
    n: int,
    n_letters: Optional[int] = None,
) -> Tuple[List[float], int, float, float, List[List[float]]]:
    """Cyclic extraction that also returns the per-shift normalized distributions.

    Same as ``extract_cyclic`` for one probe, but additionally returns, for each
    shift, the distribution normalized over that shift's letter positions (in
    raw letter order, before canonical remap). Binary tasks use this to recover
    their per-ordering diagnostics (e.g. ``p_heads_v1``/``p_heads_v2``).

    ``n_letters`` defaults to ``n`` so binary probes (n=2) target only A/B.
    """
    if n_letters is None:
        n_letters = n
    block = extract_letter_probs_batch(model, tokenizer, shift_messages, n_letters)
    lps = [lp for lp, _ in block]
    masses = [mass for _, mass in block]
    distribution, predicted_index, position_bias, mass = assemble_cyclic(lps, masses, n)

    per_shift_norm: List[List[float]] = []
    for lp, m in block:
        if m < 0.01:
            per_shift_norm.append([1.0 / n] * n)
        else:
            per_shift_norm.append([p / m for p in lp[:n]])
    return distribution, predicted_index, position_bias, mass, per_shift_norm


def extract_rating_cyclic_batch(
    model,
    tokenizer,
    messages_per_experiment: List[List[List[dict]]],
    n_shifts: int = 5,
) -> List[Tuple[float, List[float], List[List[float]], None, float, float]]:
    """5-way cyclic-counterbalanced rating extraction (batched).

    Each experiment supplies ``n_shifts`` probe message lists, one per cyclic
    shift of the rating scale. Returns a list of 6-tuples
    ``(expected_rating, rating_dist, per_shift_dists, None, scale_bias, mass)``
    — the 4th slot is ``None`` (kept for unpacking compatibility) and the 3rd
    holds the per-shift distributions for diagnostics.
    """
    if not messages_per_experiment:
        return []
    for i, exp_msgs in enumerate(messages_per_experiment):
        if len(exp_msgs) != n_shifts:
            raise ValueError(
                f"messages_per_experiment[{i}] has {len(exp_msgs)} shifts, expected {n_shifts}"
            )

    flat = [m for exp in messages_per_experiment for m in exp]
    flat_letter_probs = extract_letter_probs_batch(model, tokenizer, flat, n_letters=5)

    rating_results = []
    for i in range(len(messages_per_experiment)):
        block = flat_letter_probs[i * n_shifts:(i + 1) * n_shifts]
        lps = [lp for lp, _ in block]
        masses = [mass for _, mass in block]

        # Same cyclic assembly as type elicitation; canonical index r-1 == rating r.
        rating_dist, _predicted_idx, _bias, rating_mass = assemble_cyclic(lps, masses, n_shifts)

        # Per-shift dists for diagnostics: letter position i in shift k holds
        # rating index (i + k) % n; inverse: canonical rating r is at position
        # (r - k) % n.
        per_shift = []
        for k, (lp, mass) in enumerate(block):
            if mass < 0.01:
                per_shift.append([1.0 / n_shifts] * n_shifts)
                continue
            norm = [p / mass for p in lp[:n_shifts]]
            per_shift.append([norm[(r - k) % n_shifts] for r in range(n_shifts)])

        e_per_shift = [sum((r + 1) * d[r] for r in range(n_shifts)) for d in per_shift]
        scale_bias = (max(e_per_shift) - min(e_per_shift)) / 2 if e_per_shift else 0.0
        expected_rating = sum((r + 1) * rating_dist[r] for r in range(n_shifts))
        rating_results.append(
            (expected_rating, rating_dist, per_shift, None, scale_bias, rating_mass)
        )
    return rating_results

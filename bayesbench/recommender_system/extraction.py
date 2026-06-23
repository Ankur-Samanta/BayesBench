"""
Probability Extraction for Recommender System Experiments

Extracts rating distributions and latent-type distributions from a model's
token logprobs. The MCQ token-probability primitives and the cyclic
counterbalancing are shared from ``common.mcq``; this module keeps only the
recommender-system-specific pieces: standard-scale rating read-out and chain-of-thought
type elicitation.
"""

import re
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np

from bayesbench.common.model import setup_model  # noqa: F401  (re-exported)
from bayesbench.common.mcq import (  # noqa: F401  (re-exported for recommender_system callers)
    GPTOSS_FINAL_PREFIX,
    _channel_prefix_cache,
    _needs_channel_prefix,
    is_a_token,
    is_b_token,
    is_letter_token,
    LETTERS,
    build_mcq_prompt,
    parse_letter_probs,
    extract_letter_probs,
    extract_letter_probs_batch,
    extract_ab_probs,
    assemble_cyclic,
    extract_cyclic,
    extract_cyclic_batch,
    extract_rating_cyclic_batch,
)


@dataclass
class ExtractionResult:
    """Result of binary probability extraction."""
    p_a: float
    p_a_v1: float
    p_a_v2: float
    position_bias: float
    ab_mass: float


# ── Back-compat aliases ───────────────────────────────────────────────────────
# Back-compat aliases onto the shared common.mcq primitives, so
# existing call sites keep working after the move to the cyclic-only poll.

def extract_mcq5_probs_batch(model, tokenizer, messages_list):
    """5-letter (A–E) batched letter-probability extraction."""
    return extract_letter_probs_batch(model, tokenizer, messages_list, n_letters=5)


def extract_mcq5_probs(model, tokenizer, messages):
    """5-letter (A–E) single-prompt letter-probability extraction."""
    return extract_letter_probs(model, tokenizer, messages, n_letters=5)


# Cyclic counterbalancing — the canonical type/rating poll.
_assemble_type_counterbalanced = assemble_cyclic
extract_rating_counterbalanced_cyclic_batch = extract_rating_cyclic_batch


def extract_type_counterbalanced_batch(model, tokenizer, messages_per_experiment, n_types=4):
    """Batched cyclic-counterbalanced type elicitation (K=``n_types`` shifts)."""
    return extract_cyclic_batch(model, tokenizer, messages_per_experiment, n=n_types)


def extract_type_counterbalanced(model, tokenizer, messages_list, n_types=4):
    """Single-probe cyclic-counterbalanced type elicitation."""
    return extract_cyclic(model, tokenizer, messages_list, n=n_types)


# ── recommender system-specific extraction ────────────────────────────────────────────────

def extract_rating_probs(
    model, tokenizer, messages: List[dict]
) -> Tuple[List[float], float, float]:
    """
    Extract rating distribution via a single standard-scale MCQ (A=1★ ... E=5★).

    Returns:
        (rating_dist: List[float] of length 5, expected_rating: float, rating_mass: float)
    """
    letter_probs, letter_mass = extract_mcq5_probs(model, tokenizer, messages)

    if letter_mass < 0.01:
        return [0.2] * 5, 3.0, letter_mass

    rating_dist = [p / letter_mass for p in letter_probs]
    expected_rating = sum((r + 1) * rating_dist[r] for r in range(5))

    return rating_dist, expected_rating, letter_mass


# ── Type elicitation via chain-of-thought ─────────────────────────────────────

def _parse_profile_number(text: str, n_types: int = 5) -> int:
    """
    Parse a profile number from CoT output, returning 0-indexed type or -1.

    Tries multiple patterns in order of specificity, always taking the LAST
    match (since reasoning mentions profiles before the final answer).
    """
    # Pattern 1: "Answer: Profile N" (the instructed format)
    matches = re.findall(r"(?:final\s+)?answer\s*:\s*profile\s*(\d+)", text, re.IGNORECASE)
    if matches:
        predicted_type = int(matches[-1]) - 1
        if 0 <= predicted_type < n_types:
            return predicted_type

    # Pattern 2: "Profile N" at end of text (last 100 chars)
    tail = text[-100:] if len(text) > 100 else text
    matches = re.findall(r"profile\s*(\d+)", tail, re.IGNORECASE)
    if matches:
        predicted_type = int(matches[-1]) - 1
        if 0 <= predicted_type < n_types:
            return predicted_type

    # Pattern 3: "Answer: N" (without "Profile")
    matches = re.findall(r"answer\s*:\s*(\d+)", text, re.IGNORECASE)
    if matches:
        predicted_type = int(matches[-1]) - 1
        if 0 <= predicted_type < n_types:
            return predicted_type

    return -1


def _build_cot_prompt(tokenizer, messages: List[dict]) -> str:
    """Apply chat template for CoT generation.

    Distinct from ``build_mcq_prompt`` because CoT generation does NOT prepend
    the gptoss final-channel prefix — we want the full reasoning channel.
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _coerce_generated_text(result) -> str:
    """Normalize the various return shapes of model.generate to a string."""
    if hasattr(result, "text"):
        return result.text
    return str(result)


def generate_type_cot_batch(
    model,
    tokenizer,
    messages_list: List[List[dict]],
    max_tokens: int = 1024,
    n_types: int = 5,
) -> List[Tuple[int, str]]:
    """Batched analogue of ``generate_type_cot``.

    Submits all CoT prompts in one ``model.generate`` call. Returns a list of
    (predicted_type, reasoning_text) tuples in the same order as the input.
    """
    if not messages_list:
        return []

    prompts = [_build_cot_prompt(tokenizer, m) for m in messages_list]

    raw_results = model.generate(
        prompts,
        max_new_tokens=max_tokens,
        temperature=0.0,
    )

    if isinstance(raw_results, str):
        raw_results = [raw_results]

    out: List[Tuple[int, str]] = []
    for r in raw_results:
        text = _coerce_generated_text(r)
        out.append((_parse_profile_number(text, n_types), text))
    return out


def generate_type_cot(
    model,
    tokenizer,
    messages: List[dict],
    max_tokens: int = 1024,
    n_types: int = 5,
) -> Tuple[int, str]:
    """
    Generate free-text type reasoning and parse the predicted type.

    The model is expected to output chain-of-thought reasoning followed by
    "Answer: Profile N". Thin wrapper around ``generate_type_cot_batch``.
    """
    return generate_type_cot_batch(
        model, tokenizer, [messages], max_tokens=max_tokens, n_types=n_types,
    )[0]

"""
Model loading and probability extraction for medical triage experiments.

Reuses recommender system's cyclic-counterbalanced N-way MCQ infrastructure for both:
  1. 4-way urgency MCQ (Emergency / Urgent / Observation / Routine)
  2. 4-way profile MCQ (Accurate / Hypochondriac / Minimizer / Cyberchondriac)

Mirrors social_judgment/extraction.py for setup_model and generate.
Reuses recommender_system/extraction.py for the cyclic-counterbalanced N-way primitives —
exactly the same logprob extraction and assembly as the recommender-system type elicitation.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable

# Shared model loading + the cyclic-counterbalanced N-way MCQ primitives.
# The same machinery drives recommender-system type elicitation, triage urgency, and
# triage profile MCQs.
from bayesbench.common.model import setup_model as _setup_model
from bayesbench.common.mcq import (
    extract_letter_probs_batch,
    assemble_cyclic as _assemble_type_counterbalanced,
    _needs_channel_prefix,
    _channel_prefix_cache,
    GPTOSS_FINAL_PREFIX,
)

from .config import PROFILE_ORDER, URGENCY_ORDER, PatientProfile, UrgencyTier


def extract_mcq5_probs_batch(model, tokenizer, messages_list):
    """5-letter (A–E) batched letter-probability extraction."""
    return extract_letter_probs_batch(model, tokenizer, messages_list, n_letters=5)


def setup_model(model_name: str = "qwen7b",
                gpu_memory_utilization: Optional[float] = None,
                max_num_seqs: Optional[int] = None,
                max_model_len: int = 16384):
    """Initialize a model with 16k context for multi-turn triage runs.

    Thin wrapper over ``common.model.setup_model`` that keeps triage's larger
    default context window (storyboard conversations + per-turn polls accumulate
    context fast).
    """
    return _setup_model(
        model_name,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
    )


def generate(model, tokenizer, messages, max_tokens=1024, temperature=0.7):
    """Generate a completion from chat messages — mirrors AITA's generate()."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    tok_id = id(tokenizer)
    if tok_id not in _channel_prefix_cache:
        _channel_prefix_cache[tok_id] = _needs_channel_prefix(tokenizer)
    if _channel_prefix_cache[tok_id]:
        prompt += GPTOSS_FINAL_PREFIX

    result = model.generate(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )
    if isinstance(result, list):
        return result[0].strip()
    return result.strip()


def generate_batch(model, tokenizer, messages_list, max_tokens=1024, temperature=0.7):
    """Batched analogue of ``generate`` — one backend call for all prompts.

    Returns a list of completions in the same order as ``messages_list``.
    """
    tok_id = id(tokenizer)
    if tok_id not in _channel_prefix_cache:
        _channel_prefix_cache[tok_id] = _needs_channel_prefix(tokenizer)
    needs_prefix = _channel_prefix_cache[tok_id]

    prompts = []
    for messages in messages_list:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if needs_prefix:
            prompt += GPTOSS_FINAL_PREFIX
        prompts.append(prompt)

    results = model.generate(
        prompts,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )
    if isinstance(results, str):
        results = [results]
    return [r.strip() for r in results]


def extract_4way_counterbalanced(
    model,
    tokenizer,
    base_messages: List[Dict[str, str]],
    options_canonical: List[str],
    build_probe: Callable[[List[Dict[str, str]], List[str]], List[Dict[str, str]]],
    counterbalance: bool = True,
) -> Tuple[List[float], int, float, float]:
    """Cyclic-counterbalanced 4-way MCQ extraction.

    Runs N=4 cyclic shifts of the option ordering so every option occupies
    every position exactly once, eliminating position bias. Returns the
    averaged distribution over options in the canonical order.

    Args:
        base_messages: Conversation state to probe (not modified).
        options_canonical: List of 4 option strings in canonical order
                           (e.g. URGENCY_ORDER values, or PROFILE_ORDER values).
        build_probe: Callable that takes (base_messages, shifted_options) and
                     returns a probe message list with the MCQ appended.
        counterbalance: If False, only run shift=0 (no counterbalancing).

    Returns:
        (distribution, predicted_index, scale_bias, mass) where distribution
        is over options in canonical order.
    """
    n_types = len(options_canonical)
    if n_types != 4:
        raise ValueError(f"expected 4 options, got {n_types}")

    n_shifts = n_types if counterbalance else 1
    shift_messages = []
    for shift in range(n_shifts):
        shifted = [options_canonical[(i + shift) % n_types] for i in range(n_types)]
        shift_messages.append(build_probe(base_messages, shifted))

    flat_letter_probs = extract_mcq5_probs_batch(model, tokenizer, shift_messages)

    letter_probs_per_shift = [lp for lp, _ in flat_letter_probs]
    masses_per_shift = [mass for _, mass in flat_letter_probs]

    if not counterbalance:
        # Non-counterbalanced: use the single unshifted distribution.
        lp, mass = flat_letter_probs[0]
        if mass < 0.01:
            dist = [1.0 / n_types] * n_types
        else:
            dist = [p / mass for p in lp[:n_types]]
        return dist, int(max(range(n_types), key=lambda i: dist[i])), 0.0, mass

    return _assemble_type_counterbalanced(
        letter_probs_per_shift, masses_per_shift, n_types
    )


def _assemble_4way_shifts(
    shift_results: List[Tuple[List[float], float]],
    n_types: int,
    counterbalance: bool,
) -> Tuple[List[float], int, float, float]:
    """Assemble per-shift (letter_probs, mass) results into one canonical-order
    (distribution, predicted_index, scale_bias, mass)."""
    if not counterbalance:
        lp, mass = shift_results[0]
        if mass < 0.01:
            dist = [1.0 / n_types] * n_types
        else:
            dist = [p / mass for p in lp[:n_types]]
        return dist, int(max(range(n_types), key=lambda i: dist[i])), 0.0, mass
    lps = [lp for lp, _ in shift_results]
    masses = [mass for _, mass in shift_results]
    return _assemble_type_counterbalanced(lps, masses, n_types)


def extract_4way_many(
    model,
    tokenizer,
    requests: List[Tuple[List[Dict[str, str]], List[str],
                         Callable[[List[Dict[str, str]], List[str]], List[Dict[str, str]]]]],
    counterbalance: bool = True,
    max_prompts_per_call: Optional[int] = None,
) -> List[Tuple[List[float], int, float, float]]:
    """Maximally-batched cyclic-counterbalanced 4-way MCQ over MANY probes.

    ``requests`` is a list of ``(base_messages, options_canonical, build_probe)``
    tuples — the probes may be heterogeneous (e.g. urgency and profile, across
    different conversations and turns). Every (request x shift) probe is built
    up front and fired through a SINGLE ``extract_mcq5_probs_batch`` call (the
    backend then schedules the whole set together), then one
    (distribution, predicted_index, scale_bias, mass) is assembled per request,
    in request order.

    Set ``max_prompts_per_call`` to cap prompts per backend call if the full
    set would not fit in memory; otherwise everything goes in one call.
    """
    if not requests:
        return []
    n_types = 4
    n_shifts = n_types if counterbalance else 1

    flat_msgs: List[List[Dict[str, str]]] = []
    for base, options_canonical, build_probe in requests:
        if len(options_canonical) != n_types:
            raise ValueError(f"expected 4 options, got {len(options_canonical)}")
        for shift in range(n_shifts):
            shifted = [options_canonical[(i + shift) % n_types] for i in range(n_types)]
            flat_msgs.append(build_probe(base, shifted))

    if max_prompts_per_call is None or len(flat_msgs) <= max_prompts_per_call:
        flat_letter_probs = extract_mcq5_probs_batch(model, tokenizer, flat_msgs)
    else:
        flat_letter_probs = []
        for i in range(0, len(flat_msgs), max_prompts_per_call):
            flat_letter_probs.extend(
                extract_mcq5_probs_batch(model, tokenizer,
                                         flat_msgs[i:i + max_prompts_per_call])
            )

    results: List[Tuple[List[float], int, float, float]] = []
    for r in range(len(requests)):
        shift_results = flat_letter_probs[r * n_shifts:(r + 1) * n_shifts]
        results.append(_assemble_4way_shifts(shift_results, n_types, counterbalance))
    return results


def extract_urgency(
    model,
    tokenizer,
    base_messages: List[Dict[str, str]],
    build_urgency_probe: Callable,
    counterbalance: bool = True,
) -> Tuple[List[float], int, float, float]:
    """Extract 4-way urgency distribution in canonical URGENCY_ORDER.

    Returns (urgency_distribution, predicted_index, scale_bias, mass)
    where index 0=Emergency, 1=Urgent, 2=Observation, 3=Routine.
    """
    options = [u.value for u in URGENCY_ORDER]
    return extract_4way_counterbalanced(
        model, tokenizer, base_messages, options,
        build_urgency_probe, counterbalance=counterbalance,
    )


def extract_profile(
    model,
    tokenizer,
    base_messages: List[Dict[str, str]],
    build_profile_probe: Callable,
    counterbalance: bool = True,
) -> Tuple[List[float], int, float, float]:
    """Extract 4-way profile distribution in canonical PROFILE_ORDER.

    Returns (profile_distribution, predicted_index, scale_bias, mass)
    where index 0=Accurate, 1=Hypochondriac, 2=Minimizer, 3=Cyberchondriac.
    """
    options = [p.value for p in PROFILE_ORDER]
    return extract_4way_counterbalanced(
        model, tokenizer, base_messages, options,
        build_profile_probe, counterbalance=counterbalance,
    )


def index_to_urgency(idx: int) -> str:
    """Map predicted index to canonical urgency tier value."""
    return URGENCY_ORDER[idx].value


def index_to_profile(idx: int) -> str:
    """Map predicted index to canonical profile value."""
    return PROFILE_ORDER[idx].value

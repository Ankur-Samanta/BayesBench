"""
Model loading and P(YTA) extraction for social judgment evaluation experiments.

Adds an evaluation-specific P(YTA) probe that appends a binary MCQ question to
any conversation state, on top of the shared cyclic-counterbalanced MCQ poll in
``common.mcq``. P(YTA) is a K=2 cyclic poll: the two verdict orderings
(YTA-first / NTA-first) are averaged after position remap.
"""

from typing import Dict, Any, List, Optional, Tuple

from bayesbench.common.model import setup_model as _setup_model
from bayesbench.common.mcq import (
    extract_ab_probs,
    is_a_token,
    is_b_token,
    _needs_channel_prefix,
    _channel_prefix_cache,
    GPTOSS_FINAL_PREFIX,
    build_mcq_prompt,
    extract_letter_probs_batch,
    assemble_cyclic,
)


def _maybe_channel_prefix(tokenizer) -> str:
    tok_id = id(tokenizer)
    if tok_id not in _channel_prefix_cache:
        _channel_prefix_cache[tok_id] = _needs_channel_prefix(tokenizer)
    return GPTOSS_FINAL_PREFIX if _channel_prefix_cache[tok_id] else ""


def _render(tokenizer, messages) -> str:
    return build_mcq_prompt(tokenizer, messages)


def setup_model(model_name: str = "qwen7b", gpu_memory_utilization: float = None,
                max_num_seqs: int = None):
    """Initialize a model with 16k context for multi-turn active runs.

    Thin wrapper over ``common.model.setup_model`` that keeps AITA's larger
    default context window — active evaluation conversations can reach ~12k
    tokens (12 aspects × ~1024 tokens/turn).
    """
    return _setup_model(
        model_name,
        max_model_len=16384,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
    )

VERDICT_DESCRIPTIONS = {
    "YTA": "You're The Asshole",
    "NTA": "Not The Asshole",
}


def generate(model, tokenizer, messages, max_tokens=1024, temperature=0.7):
    """Generate a completion from chat messages using the loaded model.

    Args:
        model: loaded model instance
        tokenizer: Model tokenizer
        messages: List of chat message dicts
        max_tokens: Max new tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text string
    """
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # For gptoss models: add channel prefix so generation goes to final answer
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
    # model.generate may return str or list
    if isinstance(result, list):
        return result[0].strip()
    return result.strip()


def extract_p_yta(model, tokenizer, messages, counterbalance=True):
    """
    Extract P(YTA) via the shared cyclic counterbalanced MCQ poll (K=2).

    Appends a read-only verdict MCQ to the conversation and reads normalized
    A/B token probabilities. With counterbalance=True the two verdict orderings
    (YTA-first / NTA-first) are averaged after position remap; otherwise only
    the YTA-first ordering is read.

    Args:
        model: loaded model instance
        tokenizer: Model tokenizer
        messages: Current conversation state (not modified)
        counterbalance: If True, average over both cyclic shifts

    Returns:
        dict with p_yta, p_nta, p_yta_v1, p_yta_v2, position_bias, ab_mass
    """
    return extract_p_yta_batch(
        model, tokenizer, [messages], counterbalance=counterbalance
    )[0]


def generate_batch(model, tokenizer, message_lists: List[List[dict]],
                   max_tokens: int = 1024, temperature: float = 0.7) -> List[str]:
    """Batched analogue of ``generate``. One vLLM call for all prompts.

    Returns a list of completions in the same order as ``message_lists``.
    """
    prompts = [_render(tokenizer, msgs) for msgs in message_lists]
    results = model.generate(
        prompts,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )
    if isinstance(results, str):
        results = [results]
    return [r.strip() for r in results]


def _build_probe_msgs(messages, option_a, option_b):
    question = (
        f"Given everything described above, do you believe this person was in the wrong?\n\n"
        f"(A) {option_a} - {VERDICT_DESCRIPTIONS[option_a]}\n"
        f"(B) {option_b} - {VERDICT_DESCRIPTIONS[option_b]}\n\n"
        f"Answer with just the letter (A or B):"
    )
    return messages + [{"role": "user", "content": question}]


def extract_p_yta_batch(model, tokenizer, message_lists: List[List[dict]],
                         counterbalance: bool = True) -> List[Dict[str, Any]]:
    """Batched P(YTA) extraction over many conversations (K=2 cyclic poll).

    For each conversation, builds the YTA-first probe (shift 0) and, when
    counterbalance=True, the NTA-first probe (shift 1), reads the A/B letter
    probabilities in a single batched call, and applies the shared cyclic
    assembly. Returns one result dict per conversation, in order.
    """
    n = len(message_lists)
    if n == 0:
        return []

    # Layout: probes[stride*k] = shift 0 (A=YTA), probes[stride*k+1] = shift 1 (A=NTA).
    probe_msgs: List[List[dict]] = []
    for msgs in message_lists:
        probe_msgs.append(_build_probe_msgs(msgs, "YTA", "NTA"))
        if counterbalance:
            probe_msgs.append(_build_probe_msgs(msgs, "NTA", "YTA"))

    letter_results = extract_letter_probs_batch(model, tokenizer, probe_msgs, n_letters=2)
    expected = 2 * n if counterbalance else n
    assert len(letter_results) == expected, \
        f"letter batch returned {len(letter_results)} for {expected} probes"

    results = []
    stride = 2 if counterbalance else 1
    for k in range(n):
        lp0, m0 = letter_results[stride * k]
        ps0 = [0.5, 0.5] if m0 < 0.01 else [lp0[0] / m0, lp0[1] / m0]
        p_yta_v1 = ps0[0]  # P(YTA) = P(A) when A=YTA

        if not counterbalance:
            results.append({
                "p_yta": p_yta_v1,
                "p_nta": ps0[1],
                "p_yta_v1": p_yta_v1,
                "p_yta_v2": p_yta_v1,
                "position_bias": 0.0,
                "ab_mass": m0,
            })
            continue

        lp1, m1 = letter_results[stride * k + 1]
        ps1 = [0.5, 0.5] if m1 < 0.01 else [lp1[0] / m1, lp1[1] / m1]
        p_yta_v2 = ps1[1]  # P(YTA) = P(B) when A=NTA

        # Canonical index 0 = YTA. Shift 0: A=YTA. Shift 1: A=NTA (YTA at pos 1).
        dist, _pred, _bias, mass = assemble_cyclic([lp0[:2], lp1[:2]], [m0, m1], 2)
        position_bias = (ps0[0] + ps1[0]) / 2 - 0.5

        results.append({
            "p_yta": dist[0],
            "p_nta": dist[1],
            "p_yta_v1": p_yta_v1,
            "p_yta_v2": p_yta_v2,
            "position_bias": position_bias,
            "ab_mass": mass,
        })

    return results

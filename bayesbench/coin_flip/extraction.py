"""
Coin Flip Probability Extraction

Extracts P(heads) from a model's token distribution using the shared cyclic
counterbalanced MCQ poll (``common.mcq``) to cancel position bias.

Binary task: the two options (heads/tails) are presented under both cyclic
shifts — shift 0 maps A=heads/B=tails, shift 1 maps A=tails/B=heads — and the
position-remapped distributions are averaged. For K=2 this is exactly the
classic two-ordering swap.
"""

from typing import List, Tuple
from dataclasses import dataclass

from bayesbench.common.model import setup_model  # noqa: F401  (re-exported)
from bayesbench.common.mcq import extract_ab_probs, extract_cyclic_with_shifts


@dataclass
class ExtractionResult:
    """Result of probability extraction."""
    p_heads: float
    p_heads_v1: float  # A=heads version
    p_heads_v2: float  # A=tails version (derived from P(B))
    position_bias: float  # How much model prefers A over B
    ab_mass: float  # Total probability mass on A+B tokens


def build_messages(history: List[str], a_is_heads: bool = True) -> List[dict]:
    """
    Build chat messages for prediction.

    Args:
        history: List of observed flip outcomes ("heads" or "tails")
        a_is_heads: If True, A=heads/B=tails. If False, A=tails/B=heads.
    """
    if a_is_heads:
        mapping = "A = heads, B = tails"
    else:
        mapping = "A = tails, B = heads"

    system = f"You are predicting coin flips. Respond with only A or B.\n{mapping}"

    if len(history) == 0:
        user = "Predict the outcome of a coin flip."
    else:
        history_str = ", ".join(history)
        n_heads = sum(1 for f in history if f == "heads")
        n_tails = len(history) - n_heads
        user = f"Previous flips: [{history_str}]\nTotal: {n_heads} heads, {n_tails} tails.\n\nPredict the next flip."

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]


def extract_p_heads(
    model,
    tokenizer,
    history: List[str],
    counterbalance: bool = True
) -> ExtractionResult:
    """
    Extract P(heads) using the cyclic counterbalanced MCQ poll (K=2).

    Args:
        model: loaded model
        tokenizer: Model tokenizer
        history: List of observed outcomes
        counterbalance: If True, average over both cyclic shifts (A=heads and
            A=tails). If False, read only the A=heads ordering.

    Returns:
        ExtractionResult with P(heads) and diagnostics.
    """
    # Shift 0: A=heads, B=tails.
    msgs_shift0 = build_messages(history, a_is_heads=True)

    if not counterbalance:
        p_a, _p_b, mass = extract_ab_probs(model, tokenizer, msgs_shift0)
        return ExtractionResult(
            p_heads=p_a,
            p_heads_v1=p_a,
            p_heads_v2=p_a,
            position_bias=p_a - 0.5,
            ab_mass=mass,
        )

    # Shift 1: A=tails, B=heads. With K=2, heads (canonical index 0) sits at
    # letter position 1 here, so the cyclic remap recovers P(heads) from P(B).
    msgs_shift1 = build_messages(history, a_is_heads=False)
    dist, _pred, _bias, mass, per_shift = extract_cyclic_with_shifts(
        model, tokenizer, [msgs_shift0, msgs_shift1], n=2
    )

    # per_shift[k] is normalized over [A, B] for shift k.
    p_heads_v1 = per_shift[0][0]  # P(heads) when A=heads (shift 0)
    p_heads_v2 = per_shift[1][1]  # P(heads) when B=heads (shift 1)
    p_a_v1 = per_shift[0][0]      # P(A) in shift 0
    p_a_v2 = per_shift[1][0]      # P(A) in shift 1
    position_bias = (p_a_v1 + p_a_v2) / 2 - 0.5

    return ExtractionResult(
        p_heads=dist[0],
        p_heads_v1=p_heads_v1,
        p_heads_v2=p_heads_v2,
        position_bias=position_bias,
        ab_mass=mass,
    )


def extract_trajectory(
    model,
    tokenizer,
    sequence: List[str],
    counterbalance: bool = True
) -> List[ExtractionResult]:
    """
    Extract P(heads) at each point in a sequence.

    Returns list of ExtractionResults, one for each prefix (including empty).
    """
    results = []
    for i in range(len(sequence) + 1):
        history = sequence[:i]
        result = extract_p_heads(model, tokenizer, history, counterbalance)
        results.append(result)
    return results

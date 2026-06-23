"""Shared utilities for the BayesBench task environments.

The task packages (coin_flip, recommender_system, medical_triage, social_judgment) all
build on the same primitives: model loading, MCQ token-probability
extraction with cyclic counterbalancing, and a handful of statistics helpers.
Those live here so each environment imports them instead of re-implementing.

Submodules:
  common.model  -- setup_model() (BayesBench / vLLM wiring, GPU autodetection)
  common.mcq    -- token primitives + the single cyclic-counterbalanced MCQ poll
  common.stats  -- bootstrap CI, SEM, TVD, JSD
"""

from .model import setup_model
from .mcq import (
    GPTOSS_FINAL_PREFIX,
    is_a_token,
    is_b_token,
    is_letter_token,
    build_mcq_prompt,
    parse_letter_probs,
    extract_letter_probs,
    extract_letter_probs_batch,
    extract_ab_probs,
    assemble_cyclic,
    extract_cyclic,
    extract_cyclic_batch,
    extract_cyclic_with_shifts,
    extract_rating_cyclic_batch,
)
from .stats import bootstrap_ci, sem, tvd, jsd

__all__ = [
    "setup_model",
    "GPTOSS_FINAL_PREFIX",
    "is_a_token",
    "is_b_token",
    "is_letter_token",
    "build_mcq_prompt",
    "parse_letter_probs",
    "extract_letter_probs",
    "extract_letter_probs_batch",
    "extract_ab_probs",
    "assemble_cyclic",
    "extract_cyclic",
    "extract_cyclic_batch",
    "extract_cyclic_with_shifts",
    "extract_rating_cyclic_batch",
    "bootstrap_ci",
    "sem",
    "tvd",
    "jsd",
]

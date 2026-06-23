"""
Storyboard verdict predictor for social judgment evaluation.

Validates that the storyboard metadata (valence, importance) contains real
signal for predicting Reddit verdicts. This serves two purposes:

1. Confirms the storyboard decomposition is meaningful — a simple 2-parameter
   logistic model using only valence direction and importance achieves ~90%
   LOO accuracy on 100 posts.

2. Provides a verdict-level baseline for comparing model accuracy. If the LLM
   does worse than this simple model on final verdicts, it's failing to use
   evidence that is clearly present in the storyboard metadata.

Evidence model:
    P(YTA | storyboard) = sigmoid(Σ (k + β_imp × importance_i) × sign_i)

where sign_i ∈ {+1, 0, -1} for against_poster / neutral / pro_poster.

Note: This is NOT a trajectory reference. Moral judgment has no known
data-generating process, so there is no ground-truth intermediate trajectory
to compare against. The coin flip experiment has a normative trajectory
because the Bayesian posterior after each observation is mathematically
determined. For AITA, only the final verdict (from Reddit) is available
as ground truth.
"""

import math
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.optimize import minimize


# ============================================================================
# Constants
# ============================================================================

VALENCE_SIGN = {
    "against_poster": 1.0,
    "neutral": 0.0,
    "pro_poster": -1.0,
}


# ============================================================================
# Evidence computation
# ============================================================================

def total_evidence(
    storyboard: List[Dict[str, Any]],
    k: float,
    beta_imp: float,
) -> float:
    """Compute total log-odds evidence from storyboard metadata.

    Each aspect contributes (k + beta_imp * importance) * sign(valence).
    Neutral aspects contribute 0.
    """
    total = 0.0
    for aspect in storyboard:
        sign = VALENCE_SIGN.get(aspect.get("valence", "neutral"), 0.0)
        if sign == 0.0:
            continue
        imp = aspect.get("importance", 3)
        total += (k + beta_imp * imp) * sign
    return total


def predict_p_yta(
    storyboard: List[Dict[str, Any]],
    k: float,
    beta_imp: float,
) -> float:
    """Predict P(YTA) from storyboard metadata."""
    ev = total_evidence(storyboard, k, beta_imp)
    return 1.0 / (1.0 + math.exp(-max(-20, min(20, ev))))


# ============================================================================
# MLE fitting
# ============================================================================

def _nll(
    params: List[float],
    storyboards: List[List[Dict]],
    ground_truths: List[bool],
) -> float:
    """Negative log-likelihood of importance-weighted logistic model."""
    k, beta_imp = params
    nll = 0.0
    for sb, is_yta in zip(storyboards, ground_truths):
        p = predict_p_yta(sb, k, beta_imp)
        p = max(1e-10, min(1.0 - 1e-10, p))
        if is_yta:
            nll -= math.log(p)
        else:
            nll -= math.log(1.0 - p)
    return nll


def fit_params(
    storyboards: List[List[Dict]],
    ground_truths: List[bool],
) -> Tuple[float, float]:
    """MLE fit of (k, beta_imp). Two parameters, Nelder-Mead."""
    result = minimize(
        lambda p: _nll(p, storyboards, ground_truths),
        x0=[0.0, 0.3],
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000},
    )
    return float(result.x[0]), float(result.x[1])


def loo_calibrate(
    storyboards: List[List[Dict]],
    ground_truths: List[bool],
) -> Tuple[Tuple[float, float], Dict[str, Any]]:
    """Leave-one-out cross-validation for (k, beta_imp).

    For each post i, fit on the other N-1 posts, then predict post i.

    Returns:
        params_global: (k, beta_imp) fit on all data
        calibration_info: dict with LOO metrics including accuracy,
            Brier score, confusion matrix, and per-post predictions
    """
    n = len(storyboards)
    assert n == len(ground_truths), "storyboards and ground_truths must match"

    # Global fit
    k_global, beta_imp_global = fit_params(storyboards, ground_truths)

    # LOO
    per_post = []
    for i in range(n):
        sb_train = storyboards[:i] + storyboards[i + 1:]
        gt_train = ground_truths[:i] + ground_truths[i + 1:]

        k_i, bi_i = fit_params(sb_train, gt_train)
        pred_p = predict_p_yta(storyboards[i], k_i, bi_i)
        pred_correct = (pred_p > 0.5) == ground_truths[i]

        per_post.append({
            "k_i": k_i,
            "beta_imp_i": bi_i,
            "pred_p_yta": pred_p,
            "pred_correct": pred_correct,
            "true_yta": ground_truths[i],
        })

    # Aggregate LOO metrics
    loo_accuracy = sum(p["pred_correct"] for p in per_post) / n

    loo_brier = sum(
        (p["pred_p_yta"] - (1.0 if p["true_yta"] else 0.0)) ** 2
        for p in per_post
    ) / n

    loo_ll = 0.0
    for p in per_post:
        pred = max(1e-10, min(1.0 - 1e-10, p["pred_p_yta"]))
        loo_ll += math.log(pred) if p["true_yta"] else math.log(1.0 - pred)

    # Confusion matrix
    tp = sum(1 for p in per_post if p["true_yta"] and p["pred_p_yta"] > 0.5)
    fn = sum(1 for p in per_post if p["true_yta"] and p["pred_p_yta"] <= 0.5)
    fp = sum(1 for p in per_post if not p["true_yta"] and p["pred_p_yta"] > 0.5)
    tn = sum(1 for p in per_post if not p["true_yta"] and p["pred_p_yta"] <= 0.5)

    yta_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    nta_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    calibration_info = {
        "k_global": k_global,
        "beta_imp_global": beta_imp_global,
        "loo_accuracy": loo_accuracy,
        "loo_brier": loo_brier,
        "loo_log_likelihood": loo_ll,
        "confusion_matrix": {
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "yta_recall": yta_recall,
            "nta_recall": nta_recall,
        },
        "n_posts": n,
        "per_post": per_post,
    }

    return (k_global, beta_imp_global), calibration_info

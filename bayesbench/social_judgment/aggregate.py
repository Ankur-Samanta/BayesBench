"""
Cross-post aggregation, hypothesis tests, and effect size computation
for social judgment evaluation experiments.

Computes format effect, commitment effect, escalation effect, accuracy,
and verdict flip rates across the 100-post evaluation set.

Hypothesis tests per DESIGN.md §5:
  H1: Paired t-test on format effect (single vs passive)
  H2: Paired t-test on commitment effect (active vs passive)
  H3: Paired t-test on style effect (defending vs conceding)
  H4: McNemar's test comparing accuracy (defending vs conceding)
  H5: Cochran's Q test across all conditions
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np
from scipy import stats

from .config import TrajectoryResult, Condition, AccountStyle
from .metrics import bootstrap_ci
from .reference import loo_calibrate, predict_p_yta


def load_all_results(
    experiments_dir: Path,
    model_filter: Optional[str] = None,
    condition_filter: Optional[str] = None,
) -> List[TrajectoryResult]:
    """Load all experiment results from directory."""
    results = []

    for json_file in experiments_dir.glob("*.json"):
        if json_file.name.endswith("_analysis.json"):
            continue
        if json_file.name.endswith("_ddm.json"):
            continue

        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            result = TrajectoryResult.from_dict(data)

            if model_filter and result.config.model_name != model_filter:
                continue

            if condition_filter and result.config.condition.value != condition_filter:
                continue

            results.append(result)

        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue

    print(f"Loaded {len(results)} experiment results")
    return results


def group_results(
    results: List[TrajectoryResult],
) -> Dict[str, Dict[str, List[TrajectoryResult]]]:
    """
    Group results by model -> condition_key -> [results].

    Key format for scripted: {condition}
    Key format for active: {condition}_{style}
    """
    grouped = defaultdict(lambda: defaultdict(list))

    for result in results:
        model = result.config.model_name
        cond = result.config.condition.value
        if result.config.style:
            key = f"{cond}_{result.config.style.value}"
        else:
            key = cond
        grouped[model][key].append(result)

    return dict(grouped)


def _get_final_p_yta(result: TrajectoryResult) -> float:
    """Get final P(YTA) from a result."""
    if result.metrics:
        return result.metrics.final_p_yta
    if result.polls:
        return result.polls[-1].p_yta
    return 0.5


def _is_correct(result: TrajectoryResult) -> bool:
    """Check if final verdict matches ground truth."""
    return (_get_final_p_yta(result) > 0.5) == result.ground_truth_is_yta


# ---------------------------------------------------------------------------
# Effect size computations (with bootstrap CIs)
# ---------------------------------------------------------------------------

def compute_format_effect(
    grouped: Dict[str, List[TrajectoryResult]],
) -> Dict[str, Any]:
    """
    H1: Format effect — P(YTA)_single_turn - P(YTA)_passive_final, paired per post.
    Includes paired t-test.
    """
    single = grouped.get("single_turn", [])
    passive = grouped.get("multi_turn_passive", [])

    if not single or not passive:
        return {"error": "Missing single_turn or multi_turn_passive results"}

    single_by_post = {r.config.post_id: r for r in single}
    passive_by_post = {r.config.post_id: r for r in passive}

    effects = []
    for post_id in single_by_post:
        if post_id in passive_by_post:
            s_p = _get_final_p_yta(single_by_post[post_id])
            p_p = _get_final_p_yta(passive_by_post[post_id])
            effects.append(s_p - p_p)

    if not effects:
        return {"error": "No paired posts found"}

    mean, lo, hi = bootstrap_ci(effects)

    # Paired t-test (H1)
    t_stat, p_value = stats.ttest_1samp(effects, 0.0)

    return {
        "n_pairs": len(effects),
        "mean_effect": mean,
        "ci_lower": lo,
        "ci_upper": hi,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def compute_commitment_effect(
    grouped: Dict[str, List[TrajectoryResult]],
) -> Dict[str, Any]:
    """
    H2: Commitment effect — P(YTA)_active_final - P(YTA)_passive_final.
    Active is averaged over styles and runs per post first.
    Includes paired t-test.
    """
    passive = grouped.get("multi_turn_passive", [])
    if not passive:
        return {"error": "Missing multi_turn_passive results"}

    passive_by_post = {r.config.post_id: r for r in passive}

    # Collect all active results, average by post
    active_by_post = defaultdict(list)
    for key, results in grouped.items():
        if key.startswith("multi_turn_active"):
            for r in results:
                active_by_post[r.config.post_id].append(_get_final_p_yta(r))

    if not active_by_post:
        return {"error": "Missing multi_turn_active results"}

    effects = []
    for post_id, active_vals in active_by_post.items():
        if post_id in passive_by_post:
            active_mean = float(np.mean(active_vals))
            passive_p = _get_final_p_yta(passive_by_post[post_id])
            effects.append(active_mean - passive_p)

    if not effects:
        return {"error": "No paired posts found"}

    mean, lo, hi = bootstrap_ci(effects)

    # Paired t-test (H2)
    t_stat, p_value = stats.ttest_1samp(effects, 0.0)

    return {
        "n_pairs": len(effects),
        "mean_effect": mean,
        "ci_lower": lo,
        "ci_upper": hi,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def compute_escalation_effect(
    grouped: Dict[str, List[TrajectoryResult]],
) -> Dict[str, Any]:
    """
    H3 (key contrast): Style effect — P(YTA)_defending - P(YTA)_conceding.
    Each style's P(YTA) is averaged over runs per post.
    Includes paired t-test.
    """
    defending = grouped.get("multi_turn_active_defending", [])
    conceding = grouped.get("multi_turn_active_conceding", [])

    if not defending or not conceding:
        return {"error": "Missing defending or conceding results"}

    # Average by post
    ref_by_post = defaultdict(list)
    for r in defending:
        ref_by_post[r.config.post_id].append(_get_final_p_yta(r))

    con_by_post = defaultdict(list)
    for r in conceding:
        con_by_post[r.config.post_id].append(_get_final_p_yta(r))

    effects = []
    for post_id in ref_by_post:
        if post_id in con_by_post:
            ref_mean = float(np.mean(ref_by_post[post_id]))
            con_mean = float(np.mean(con_by_post[post_id]))
            effects.append(ref_mean - con_mean)

    if not effects:
        return {"error": "No paired posts found"}

    mean, lo, hi = bootstrap_ci(effects)

    # Paired t-test
    t_stat, p_value = stats.ttest_1samp(effects, 0.0)

    return {
        "n_pairs": len(effects),
        "mean_effect": mean,
        "ci_lower": lo,
        "ci_upper": hi,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def compute_style_effect(
    grouped: Dict[str, List[TrajectoryResult]],
) -> Dict[str, Any]:
    """
    H3: Paired t-test on style effect — defending vs conceding.
    Uses per-post mean P(YTA) (averaged over runs) as the measure.
    """
    styles = ["conceding", "defending"]

    # Collect per-post means for each style
    style_by_post = {}
    for style in styles:
        key = f"multi_turn_active_{style}"
        results = grouped.get(key, [])
        if not results:
            return {"error": f"Missing {style} results"}
        by_post = defaultdict(list)
        for r in results:
            by_post[r.config.post_id].append(_get_final_p_yta(r))
        style_by_post[style] = {pid: float(np.mean(vals)) for pid, vals in by_post.items()}

    # Find posts present in both styles
    common_posts = set.intersection(*(set(d.keys()) for d in style_by_post.values()))
    if len(common_posts) < 3:
        return {"error": f"Only {len(common_posts)} posts have both styles"}

    common_posts = sorted(common_posts)

    # Paired differences: defending - conceding
    diffs = [style_by_post["defending"][pid] - style_by_post["conceding"][pid]
             for pid in common_posts]

    mean, lo, hi = bootstrap_ci(diffs)
    t_stat, p_value = stats.ttest_1samp(diffs, 0.0)

    # Per-style means
    style_means = {}
    for style in styles:
        vals = [style_by_post[style][pid] for pid in common_posts]
        m, slo, shi = bootstrap_ci(vals)
        style_means[style] = {"mean": m, "ci_lower": slo, "ci_upper": shi}

    return {
        "n_posts": len(common_posts),
        "mean_effect": mean,
        "ci_lower": lo,
        "ci_upper": hi,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "style_means": style_means,
    }


def compute_accuracy(
    results: List[TrajectoryResult],
    condition_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute accuracy: 1[P(YTA)_final > 0.5] matches ground truth."""
    filtered = results
    if condition_filter:
        filtered = [r for r in results if r.config.condition.value == condition_filter]

    if not filtered:
        return {"error": f"No results for condition={condition_filter}"}

    accs = [1.0 if _is_correct(r) else 0.0 for r in filtered]
    correct = int(sum(accs))
    total = len(accs)
    accuracy = correct / total if total > 0 else 0.0

    mean, lo, hi = bootstrap_ci(accs)

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "ci_lower": lo,
        "ci_upper": hi,
    }


def compute_mcnemar_test(
    grouped: Dict[str, List[TrajectoryResult]],
) -> Dict[str, Any]:
    """
    H4: McNemar's test — does defending style cause more incorrect verdicts
    than conceding?
    """
    defending = grouped.get("multi_turn_active_defending", [])
    conceding = grouped.get("multi_turn_active_conceding", [])

    if not defending or not conceding:
        return {"error": "Missing defending or conceding results"}

    # Average over runs per post
    ref_by_post = defaultdict(list)
    for r in defending:
        ref_by_post[r.config.post_id].append(r)

    con_by_post = defaultdict(list)
    for r in conceding:
        con_by_post[r.config.post_id].append(r)

    # Build 2x2 contingency: (conceding correct, conceding wrong) x (defending correct, defending wrong)
    # a = both correct, b = conceding correct & defending wrong
    # c = conceding wrong & defending correct, d = both wrong
    a, b, c, d = 0, 0, 0, 0

    for post_id in ref_by_post:
        if post_id not in con_by_post:
            continue

        # Average P(YTA) over runs, then threshold
        ref_p = float(np.mean([_get_final_p_yta(r) for r in ref_by_post[post_id]]))
        con_p = float(np.mean([_get_final_p_yta(r) for r in con_by_post[post_id]]))
        gt_yta = ref_by_post[post_id][0].ground_truth_is_yta

        ref_correct = (ref_p > 0.5) == gt_yta
        con_correct = (con_p > 0.5) == gt_yta

        if con_correct and ref_correct:
            a += 1
        elif con_correct and not ref_correct:
            b += 1
        elif not con_correct and ref_correct:
            c += 1
        else:
            d += 1

    n_total = a + b + c + d
    if n_total == 0:
        return {"error": "No paired posts found"}

    # McNemar's test: are b and c significantly different?
    if b + c == 0:
        p_value = 1.0
        chi2 = 0.0
    else:
        # McNemar with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
        p_value = float(1 - stats.chi2.cdf(chi2, df=1))

    return {
        "n_posts": n_total,
        "both_correct": a,
        "conceding_only_correct": b,
        "defending_only_correct": c,
        "both_wrong": d,
        "conceding_accuracy": (a + b) / n_total if n_total else 0.0,
        "defending_accuracy": (a + c) / n_total if n_total else 0.0,
        "mcnemar_chi2": float(chi2),
        "mcnemar_p": p_value,
        "significant": bool(p_value < 0.05),
    }


def compute_cochran_q(
    results: List[TrajectoryResult],
    grouped: Dict[str, List[TrajectoryResult]],
) -> Dict[str, Any]:
    """
    H5: Cochran's Q test — does accuracy differ across conditions?
    Tests whether the proportion correct differs across:
    single_turn, multi_turn_passive, and 2 active styles.
    """
    # For each condition, compute per-post correctness
    conditions_to_test = ["single_turn", "multi_turn_passive"]
    for style in ["neutral", "conceding", "defending"]:
        key = f"multi_turn_active_{style}"
        if key in grouped:
            conditions_to_test.append(key)

    # Build correctness by post for each condition
    cond_correct = {}
    for cond_key in conditions_to_test:
        cond_results = grouped.get(cond_key, [])
        if not cond_results:
            continue
        # Average over runs per post
        by_post = defaultdict(list)
        for r in cond_results:
            by_post[r.config.post_id].append(r)

        correct_by_post = {}
        for post_id, post_results in by_post.items():
            mean_p = float(np.mean([_get_final_p_yta(r) for r in post_results]))
            gt_yta = post_results[0].ground_truth_is_yta
            correct_by_post[post_id] = 1 if (mean_p > 0.5) == gt_yta else 0

        cond_correct[cond_key] = correct_by_post

    if len(cond_correct) < 2:
        return {"error": "Need at least 2 conditions for Cochran's Q"}

    # Find common posts
    common_posts = set.intersection(*(set(d.keys()) for d in cond_correct.values()))
    if len(common_posts) < 3:
        return {"error": f"Only {len(common_posts)} common posts"}

    common_posts = sorted(common_posts)
    k = len(cond_correct)
    n = len(common_posts)

    # Build binary matrix: n_posts x k_conditions
    matrix = np.zeros((n, k), dtype=int)
    cond_names = sorted(cond_correct.keys())
    for j, cond_key in enumerate(cond_names):
        for i, post_id in enumerate(common_posts):
            matrix[i, j] = cond_correct[cond_key][post_id]

    # Cochran's Q statistic
    row_sums = matrix.sum(axis=1)  # T_i
    col_sums = matrix.sum(axis=0)  # C_j
    T = row_sums.sum()

    numerator = (k - 1) * (k * (col_sums ** 2).sum() - T ** 2)
    denominator = k * T - (row_sums ** 2).sum()

    if denominator == 0:
        return {"error": "Cochran's Q undefined (zero denominator)"}

    Q = numerator / denominator
    # Q ~ chi2 with k-1 degrees of freedom
    p_value = float(1 - stats.chi2.cdf(Q, df=k - 1))

    # Per-condition accuracy
    cond_accuracies = {}
    for j, cond_key in enumerate(cond_names):
        acc = float(col_sums[j]) / n
        cond_accuracies[cond_key] = acc

    return {
        "n_posts": n,
        "n_conditions": k,
        "conditions": cond_names,
        "cochran_Q": float(Q),
        "p_value": p_value,
        "significant": bool(p_value < 0.05),
        "condition_accuracies": cond_accuracies,
    }


# ---------------------------------------------------------------------------
# Normative constraint tests
# ---------------------------------------------------------------------------

def compute_calibration(
    results: List[TrajectoryResult],
    n_bins: int = 5,
) -> Dict[str, Any]:
    """
    Normative constraint: calibration.

    If the model says P(YTA)=0.7 across many posts, ~70% should actually be YTA.
    Bins final P(YTA) values and compares predicted vs actual YTA rate per bin.

    Returns calibration error (ECE), per-bin stats, and Brier score.
    """
    if not results:
        return {"error": "No results"}

    p_ytas = [_get_final_p_yta(r) for r in results]
    actuals = [1.0 if r.ground_truth_is_yta else 0.0 for r in results]

    # Brier score: mean squared error of probability predictions
    brier = float(np.mean([(p - a) ** 2 for p, a in zip(p_ytas, actuals)]))

    # Bin by predicted P(YTA)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = [(lo <= p < hi) or (i == n_bins - 1 and p == hi) for p in p_ytas]
        bin_ps = [p for p, m in zip(p_ytas, mask) if m]
        bin_as = [a for a, m in zip(actuals, mask) if m]

        if not bin_ps:
            bins.append({"range": [float(lo), float(hi)], "n": 0})
            continue

        mean_pred = float(np.mean(bin_ps))
        mean_actual = float(np.mean(bin_as))
        n = len(bin_ps)
        ece += abs(mean_pred - mean_actual) * n

        bins.append({
            "range": [float(lo), float(hi)],
            "n": n,
            "mean_predicted": mean_pred,
            "mean_actual": mean_actual,
            "gap": float(mean_pred - mean_actual),
        })

    ece /= len(results)

    return {
        "n_results": len(results),
        "brier_score": brier,
        "ece": float(ece),
        "n_bins": n_bins,
        "bins": bins,
    }


def compute_martingale_test(
    results: List[TrajectoryResult],
    condition_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Normative constraint: martingale property.

    Under rational updating, E[P(t+1) | P(t)] = P(t) — the expected
    posterior equals the prior, averaged over possible observations.
    Across many posts, the mean trajectory should show no systematic drift.

    Computes mean P(YTA) at each time step across posts. Systematic
    drift (e.g., monotonically decreasing toward NTA) indicates the model's
    priors don't match the evidence distribution.

    Also tests whether the mean per-step change differs from zero.
    """
    filtered = results
    if condition_filter:
        filtered = [r for r in results if r.config.condition.value == condition_filter]

    if not filtered:
        return {"error": f"No results for condition={condition_filter}"}

    # Only include multi-step trajectories
    trajectories = [r for r in filtered if len(r.polls) >= 3]
    if len(trajectories) < 5:
        return {"error": f"Too few multi-step trajectories ({len(trajectories)})"}

    # Find max trajectory length
    max_len = max(len(r.polls) for r in trajectories)

    # Compute mean P(YTA) at each time step
    mean_by_step = []
    for t in range(max_len):
        ps = [r.polls[t].p_yta for r in trajectories if len(r.polls) > t]
        if len(ps) >= 3:
            mean_by_step.append({
                "t": t,
                "mean_p_yta": float(np.mean(ps)),
                "std_p_yta": float(np.std(ps)),
                "n": len(ps),
            })

    # Compute per-step changes across all trajectories
    all_deltas = []
    for r in trajectories:
        for i in range(len(r.polls) - 1):
            all_deltas.append(r.polls[i + 1].p_yta - r.polls[i].p_yta)

    mean_delta = float(np.mean(all_deltas))

    # Test: is mean delta significantly different from 0?
    t_stat, p_value = stats.ttest_1samp(all_deltas, 0.0)

    # Mean drift direction
    total_drifts = []
    for r in trajectories:
        total_drifts.append(r.polls[-1].p_yta - r.polls[0].p_yta)
    mean_total_drift = float(np.mean(total_drifts))

    return {
        "n_trajectories": len(trajectories),
        "n_deltas": len(all_deltas),
        "mean_per_step_delta": mean_delta,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant_drift": bool(p_value < 0.05),
        "mean_total_drift": mean_total_drift,
        "mean_trajectory": mean_by_step,
    }


def compute_verdict_flip_rate(
    grouped: Dict[str, List[TrajectoryResult]],
) -> Dict[str, Any]:
    """Fraction of posts where account style changes binary verdict."""
    style_by_post = defaultdict(lambda: defaultdict(list))
    for key, results in grouped.items():
        if key.startswith("multi_turn_active_"):
            style = key.replace("multi_turn_active_", "")
            for r in results:
                style_by_post[r.config.post_id][style].append(_get_final_p_yta(r))

    flips = 0
    total = 0

    for post_id, styles in style_by_post.items():
        if len(styles) < 2:
            continue
        verdicts = set()
        for style, vals in styles.items():
            mean_p = float(np.mean(vals))
            verdicts.add(mean_p > 0.5)
        total += 1
        if len(verdicts) > 1:
            flips += 1

    flip_rate = flips / total if total > 0 else 0.0
    return {
        "flip_rate": flip_rate,
        "n_flips": flips,
        "n_posts": total,
    }


def generate_summary_report(
    results: List[TrajectoryResult],
    model: str,
) -> Dict[str, Any]:
    """Generate comprehensive summary report for a model."""
    model_results = [r for r in results if r.config.model_name == model]
    grouped = group_results(model_results)

    if model not in grouped:
        return {"error": f"Model {model} not found"}

    model_grouped = grouped[model]

    report = {
        "model": model,
        "total_experiments": len(model_results),
        # Effect sizes with hypothesis tests
        "format_effect": compute_format_effect(model_grouped),
        "commitment_effect": compute_commitment_effect(model_grouped),
        "style_effect": compute_escalation_effect(model_grouped),
        # Hypothesis tests
        "style_test": compute_style_effect(model_grouped),
        "mcnemar_test": compute_mcnemar_test(model_grouped),
        "cochran_q": compute_cochran_q(model_results, model_grouped),
        # Normative constraints
        "calibration": compute_calibration(model_results),
        "martingale_passive": compute_martingale_test(
            model_results, condition_filter="multi_turn_passive"
        ),
        "martingale_active": compute_martingale_test(
            model_results, condition_filter="multi_turn_active"
        ),
        # Descriptive
        "verdict_flip_rate": compute_verdict_flip_rate(model_grouped),
        "accuracy": {},
    }

    # Accuracy per condition
    for cond in Condition:
        report["accuracy"][cond.value] = compute_accuracy(model_results, cond.value)

    # Active accuracy per style
    for style in AccountStyle:
        style_results = [
            r for r in model_results
            if r.config.condition == Condition.MULTI_TURN_ACTIVE
            and r.config.style == style
        ]
        if style_results:
            report["accuracy"][f"active_{style.value}"] = compute_accuracy(style_results)

    # Storyboard metadata validation — verdict-level baseline
    try:
        post_storyboards = {}
        post_ground_truths = {}
        for r in model_results:
            pid = r.config.post_id
            if pid not in post_storyboards:
                post_storyboards[pid] = r.storyboard
                post_ground_truths[pid] = r.ground_truth_is_yta

        post_ids = sorted(post_storyboards.keys())
        storyboards = [post_storyboards[pid] for pid in post_ids]
        ground_truths = [post_ground_truths[pid] for pid in post_ids]

        if len(storyboards) >= 5:
            params_global, calibration = loo_calibrate(storyboards, ground_truths)
            k_g, bi_g = params_global

            ref_report = {
                "k": k_g,
                "beta_imp": bi_g,
                "loo_accuracy": calibration["loo_accuracy"],
                "loo_brier": calibration["loo_brier"],
                "confusion_matrix": calibration["confusion_matrix"],
            }

            # Compare reference verdict accuracy vs model verdict accuracy
            for cond in ["single_turn", "multi_turn_passive"]:
                cond_results = [
                    r for r in model_results
                    if r.config.condition.value == cond
                ]
                if not cond_results:
                    continue

                model_correct = sum(1 for r in cond_results if _is_correct(r))
                ref_correct = sum(
                    1 for r in cond_results
                    if (predict_p_yta(r.storyboard, k_g, bi_g) > 0.5)
                    == r.ground_truth_is_yta
                )
                n = len(cond_results)
                ref_report[f"{cond}_model_accuracy"] = model_correct / n
                ref_report[f"{cond}_reference_accuracy"] = ref_correct / n

            report["storyboard_validation"] = ref_report
    except Exception as e:
        report["storyboard_validation"] = {"error": str(e)}

    return report


def main():
    parser = argparse.ArgumentParser(description="Aggregate AITA Evaluation Results")
    parser.add_argument("--experiments-dir", type=str,
                        default=str(Path(__file__).parent / "experiments"),
                        help="Directory with results")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter by model name")

    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    results = load_all_results(experiments_dir, model_filter=args.model)

    if not results:
        print("No results found!")
        return

    # Discover models
    models = sorted(set(r.config.model_name for r in results))

    output = {
        "experiments_dir": str(experiments_dir),
        "total_experiments": len(results),
        "models": {},
    }

    for model in models:
        output["models"][model] = generate_summary_report(results, model)

    # Save
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved aggregated results to: {output_path}")

    # Print summary
    for model, report in output["models"].items():
        print(f"\n{'='*70}")
        print(f"MODEL: {model}")
        print(f"{'='*70}")
        print(f"Total experiments: {report.get('total_experiments', '?')}")

        # H1: Format effect
        fe = report.get("format_effect", {})
        if "mean_effect" in fe:
            sig = "*" if fe.get("significant") else ""
            print(f"\nH1 — Format effect (single - passive):")
            print(f"  Mean: {fe['mean_effect']:+.4f} [{fe['ci_lower']:+.4f}, {fe['ci_upper']:+.4f}]")
            print(f"  t={fe['t_stat']:.3f}, p={fe['p_value']:.4f}{sig}")

        # H2: Commitment effect
        ce = report.get("commitment_effect", {})
        if "mean_effect" in ce:
            sig = "*" if ce.get("significant") else ""
            print(f"\nH2 — Commitment effect (active - passive):")
            print(f"  Mean: {ce['mean_effect']:+.4f} [{ce['ci_lower']:+.4f}, {ce['ci_upper']:+.4f}]")
            print(f"  t={ce['t_stat']:.3f}, p={ce['p_value']:.4f}{sig}")

        # H3: Style effect
        se = report.get("style_effect", {})
        if "mean_effect" in se:
            sig = "*" if se.get("significant") else ""
            print(f"\nH3 — Style effect (defending - conceding):")
            print(f"  Mean: {se['mean_effect']:+.4f} [{se['ci_lower']:+.4f}, {se['ci_upper']:+.4f}]")
            print(f"  t={se['t_stat']:.3f}, p={se['p_value']:.4f}{sig}")

        st = report.get("style_test", {})
        if "style_means" in st:
            for style, data in st.get("style_means", {}).items():
                print(f"    {style:15s}: {data['mean']:.4f} [{data['ci_lower']:.4f}, {data['ci_upper']:.4f}]")

        # H4: McNemar
        mn = report.get("mcnemar_test", {})
        if "mcnemar_chi2" in mn:
            sig = "*" if mn.get("significant") else ""
            print(f"\nH4 — McNemar's test (defending vs conceding accuracy):")
            print(f"  Conceding acc: {mn['conceding_accuracy']:.2%}, Defending acc: {mn['defending_accuracy']:.2%}")
            print(f"  chi2={mn['mcnemar_chi2']:.3f}, p={mn['mcnemar_p']:.4f}{sig}")
            print(f"  Contingency: both_correct={mn['both_correct']}, "
                  f"con_only={mn['conceding_only_correct']}, "
                  f"def_only={mn['defending_only_correct']}, "
                  f"both_wrong={mn['both_wrong']}")

        # H5: Cochran's Q
        cq = report.get("cochran_q", {})
        if "cochran_Q" in cq:
            sig = "*" if cq.get("significant") else ""
            print(f"\nH5 — Cochran's Q (accuracy across conditions):")
            print(f"  Q={cq['cochran_Q']:.3f}, p={cq['p_value']:.4f}{sig}")
            for cond, acc in cq.get("condition_accuracies", {}).items():
                print(f"    {cond:30s}: {acc:.2%}")

        # Normative constraints
        cal = report.get("calibration", {})
        if "ece" in cal:
            print(f"\nCalibration:")
            print(f"  ECE: {cal['ece']:.4f}  Brier: {cal['brier_score']:.4f}")
            for b in cal.get("bins", []):
                if b.get("n", 0) > 0:
                    print(f"    [{b['range'][0]:.1f}, {b['range'][1]:.1f}]: "
                          f"pred={b['mean_predicted']:.3f}  actual={b['mean_actual']:.3f}  "
                          f"gap={b['gap']:+.3f}  (n={b['n']})")

        for cond_label in ["martingale_passive", "martingale_active"]:
            mg = report.get(cond_label, {})
            if "mean_per_step_delta" in mg:
                sig = "*" if mg.get("significant_drift") else ""
                cond_name = cond_label.replace("martingale_", "")
                print(f"\nMartingale ({cond_name}):")
                print(f"  Mean Δ per step: {mg['mean_per_step_delta']:+.4f}  "
                      f"t={mg['t_stat']:.3f}  p={mg['p_value']:.4f}{sig}")
                print(f"  Mean total drift: {mg['mean_total_drift']:+.4f}  "
                      f"(n={mg['n_trajectories']})")

        # Verdict flip rate
        vf = report.get("verdict_flip_rate", {})
        if "flip_rate" in vf:
            print(f"\nVerdict flip rate: {vf['flip_rate']:.2%} ({vf['n_flips']}/{vf['n_posts']})")

        # Storyboard metadata validation
        sv = report.get("storyboard_validation", {})
        if "loo_accuracy" in sv:
            print(f"\nStoryboard Metadata Validation:")
            print(f"  LOO accuracy: {sv['loo_accuracy']:.2%}  "
                  f"Brier: {sv['loo_brier']:.4f}")
            cm = sv.get("confusion_matrix", {})
            if cm:
                print(f"  YTA recall: {cm['yta_recall']:.2%}  "
                      f"NTA recall: {cm['nta_recall']:.2%}")
            for cond in ["single_turn", "multi_turn_passive"]:
                m_key = f"{cond}_model_accuracy"
                r_key = f"{cond}_reference_accuracy"
                if m_key in sv:
                    print(f"  {cond:30s}: model={sv[m_key]:.2%}  "
                          f"reference={sv[r_key]:.2%}")

        # Accuracy
        acc = report.get("accuracy", {})
        if acc:
            print(f"\nAccuracy:")
            for cond, data in acc.items():
                if "accuracy" in data:
                    print(f"  {cond:30s}: {data['accuracy']:.2%} "
                          f"({data['correct']}/{data['total']}) "
                          f"[{data['ci_lower']:.2%}, {data['ci_upper']:.2%}]")


if __name__ == "__main__":
    main()

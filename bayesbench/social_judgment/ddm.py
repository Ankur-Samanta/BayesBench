"""
Drift-Diffusion Model analysis for social judgment evaluation trajectories.

Descriptive parameterization of P(YTA) trajectories from the evaluation
experiments. Fits DDM parameters to each trajectory in log-odds space,
summarizing the LLM's updating behavior as starting bias + drift + noise.

This is NOT a normative model — it does not claim the LLM is performing
Bayesian evidence accumulation or that the parameters correspond to any
cognitive process. Its value is in expressing experimental effects as
changes in accumulator parameters, separating prior bias from evidence
accumulation rate from update noise.

This is an independent analysis path that reads the same experiment JSONs
produced by runner.py. It does not run new experiments.

DDM parameters per trajectory:
  - z (starting bias): log-odds P(YTA) at t=0, measures prior bias
  - v (drift rate): mean log-odds update per step, measures evidence strength
  - sigma (diffusion): std of log-odds updates, measures update noise
  - v_toward_truth (drift toward ground truth): drift rate signed so positive
    means accumulating toward the correct verdict

Cross-condition comparisons:
  - Drift rate by condition → format/commitment effect in DDM terms
  - Drift rate by style → escalation effect as differential drift
  - Starting bias by condition → whether condition affects priors
  - Diffusion by condition → whether commitment adds noise

Usage:
    python -m social_judgment.ddm \\
        --experiments-dir social_judgment/experiments \\
        --model llama8b \\
        --output social_judgment/experiments/llama8b_ddm.json
"""

import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
from scipy import stats
from .config import TrajectoryResult, Condition, AccountStyle
from .aggregate import load_all_results, group_results
from .metrics import bootstrap_ci
from .reference import loo_calibrate


# ============================================================================
# Log-odds transform
# ============================================================================

def to_log_odds(p: float, clip: float = 1e-6) -> float:
    """Convert probability to log-odds, clipping to avoid infinities."""
    p = max(clip, min(1.0 - clip, p))
    return math.log(p / (1.0 - p))


def from_log_odds(x: float) -> float:
    """Convert log-odds back to probability."""
    return 1.0 / (1.0 + math.exp(-x))


# ============================================================================
# Per-trajectory DDM parameters
# ============================================================================

@dataclass
class DDMParams:
    """Drift-diffusion parameters for a single trajectory."""
    # Basic DDM parameters (log-odds space)
    z: float               # Starting point bias: log-odds(P(YTA)_0)
    v: float               # Drift rate: mean(delta log-odds) per step
    sigma: float           # Diffusion coefficient: std(delta log-odds) per step
    n_steps: int           # Number of steps in trajectory

    # Derived / interpretive
    v_toward_truth: float  # Drift signed toward correct verdict (+ve = good)
    z_bias: float          # Starting bias toward YTA: z - 0 (0 = no bias)
    final_log_odds: float  # Log-odds at final step
    total_drift: float     # final_log_odds - z
    efficiency: float      # |total_drift| / (n_steps * sigma) if sigma > 0

    # Metadata
    post_id: str = ""
    condition: str = ""
    style: Optional[str] = None
    run: int = 0
    ground_truth_is_yta: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DDMGroupStats:
    """Aggregate DDM statistics for a group of trajectories."""
    condition: str
    style: Optional[str]
    n_trajectories: int

    # Mean parameters with CIs
    mean_v: float
    ci_v: Tuple[float, float]
    mean_sigma: float
    ci_sigma: Tuple[float, float]
    mean_z: float
    ci_z: Tuple[float, float]
    mean_v_toward_truth: float
    ci_v_toward_truth: Tuple[float, float]
    mean_efficiency: float
    ci_efficiency: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert tuples to lists for JSON
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d


# ============================================================================
# Fitting
# ============================================================================

def fit_trajectory(result: TrajectoryResult) -> DDMParams:
    """Fit DDM parameters to a single P(YTA) trajectory.

    Transforms the P(YTA) sequence to log-odds space, then estimates:
      z = log-odds at t=0 (starting point)
      v = mean of step-wise log-odds increments (drift rate)
      sigma = std of step-wise log-odds increments (diffusion)
    """
    polls = result.polls
    if len(polls) < 2:
        # Single measurement — can't estimate drift
        lo = to_log_odds(polls[0].p_yta) if polls else 0.0
        return DDMParams(
            z=lo, v=0.0, sigma=0.0, n_steps=len(polls),
            v_toward_truth=0.0, z_bias=lo, final_log_odds=lo,
            total_drift=0.0, efficiency=0.0,
            post_id=result.config.post_id,
            condition=result.config.condition.value,
            style=result.config.style.value if result.config.style else None,
            run=result.config.run,
            ground_truth_is_yta=result.ground_truth_is_yta,
        )

    # Transform to log-odds
    log_odds = [to_log_odds(p.p_yta) for p in polls]

    # Step-wise increments
    deltas = [log_odds[i + 1] - log_odds[i] for i in range(len(log_odds) - 1)]

    z = log_odds[0]
    final = log_odds[-1]
    v = np.mean(deltas)
    sigma = np.std(deltas, ddof=1) if len(deltas) > 1 else 0.0
    n_steps = len(deltas)

    # Sign drift toward ground truth
    # If ground truth is YTA, positive log-odds drift is toward truth
    # If ground truth is NTA, negative log-odds drift is toward truth
    if result.ground_truth_is_yta:
        v_toward_truth = v
    else:
        v_toward_truth = -v

    z_bias = z  # 0 = unbiased, >0 = biased toward YTA

    total_drift = final - z
    efficiency = abs(total_drift) / (n_steps * sigma) if sigma > 0 else 0.0

    return DDMParams(
        z=z, v=v, sigma=sigma, n_steps=n_steps,
        v_toward_truth=v_toward_truth,
        z_bias=z_bias,
        final_log_odds=final,
        total_drift=total_drift,
        efficiency=efficiency,
        post_id=result.config.post_id,
        condition=result.config.condition.value,
        style=result.config.style.value if result.config.style else None,
        run=result.config.run,
        ground_truth_is_yta=result.ground_truth_is_yta,
    )


def fit_trajectory_with_evidence(
    result: TrajectoryResult,
) -> Tuple[DDMParams, Dict[str, Any]]:
    """Fit DDM with evidence-modulated drift rate.

    Extends the basic DDM by decomposing drift into components based on
    aspect valence and importance. This models evidence accumulation as:

        delta_logodds(t) = beta_valence * valence(t) + beta_importance * importance(t) + noise

    where valence is coded as +1 (against_poster → YTA), 0 (neutral),
    -1 (pro_poster → NTA) and importance is 1-5.

    Returns DDMParams plus a dict with regression coefficients.
    """
    polls = result.polls
    base_params = fit_trajectory(result)

    if len(polls) < 3:
        return base_params, {"beta_valence": 0.0, "beta_importance": 0.0, "r_squared": 0.0}

    log_odds = [to_log_odds(p.p_yta) for p in polls]
    deltas = [log_odds[i + 1] - log_odds[i] for i in range(len(log_odds) - 1)]

    # Build regressors from aspect metadata
    valence_codes = []
    importance_codes = []
    valid_deltas = []

    for i, delta in enumerate(deltas):
        poll = polls[i + 1]  # The poll AFTER this delta
        if poll.aspect_valence is None:
            continue

        valence_map = {
            "against_poster": 1.0,
            "neutral": 0.0,
            "pro_poster": -1.0,
        }
        v_code = valence_map.get(poll.aspect_valence, 0.0)
        imp = float(poll.aspect_importance) if poll.aspect_importance else 3.0

        valence_codes.append(v_code)
        importance_codes.append(imp)
        valid_deltas.append(delta)

    if len(valid_deltas) < 3:
        return base_params, {"beta_valence": 0.0, "beta_importance": 0.0, "r_squared": 0.0}

    # OLS regression: delta = beta0 + beta_v * valence + beta_i * importance
    X = np.column_stack([
        np.ones(len(valid_deltas)),
        valence_codes,
        importance_codes,
    ])
    y = np.array(valid_deltas)

    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except np.linalg.LinAlgError:
        beta = np.zeros(3)
        r_squared = 0.0

    evidence_info = {
        "beta_intercept": float(beta[0]),
        "beta_valence": float(beta[1]),
        "beta_importance": float(beta[2]),
        "r_squared": float(r_squared),
        "n_aspects": len(valid_deltas),
    }

    return base_params, evidence_info


# ============================================================================
# Group-level analysis
# ============================================================================

def compute_group_stats(
    params_list: List[DDMParams],
    condition: str,
    style: Optional[str] = None,
) -> DDMGroupStats:
    """Compute aggregate DDM stats with bootstrap CIs."""
    if not params_list:
        empty_ci = (0.0, 0.0)
        return DDMGroupStats(
            condition=condition, style=style, n_trajectories=0,
            mean_v=0.0, ci_v=empty_ci,
            mean_sigma=0.0, ci_sigma=empty_ci,
            mean_z=0.0, ci_z=empty_ci,
            mean_v_toward_truth=0.0, ci_v_toward_truth=empty_ci,
            mean_efficiency=0.0, ci_efficiency=empty_ci,
        )

    vs = [p.v for p in params_list]
    sigmas = [p.sigma for p in params_list]
    zs = [p.z for p in params_list]
    vts = [p.v_toward_truth for p in params_list]
    effs = [p.efficiency for p in params_list]

    return DDMGroupStats(
        condition=condition,
        style=style,
        n_trajectories=len(params_list),
        mean_v=float(np.mean(vs)),
        ci_v=bootstrap_ci(vs),
        mean_sigma=float(np.mean(sigmas)),
        ci_sigma=bootstrap_ci(sigmas),
        mean_z=float(np.mean(zs)),
        ci_z=bootstrap_ci(zs),
        mean_v_toward_truth=float(np.mean(vts)),
        ci_v_toward_truth=bootstrap_ci(vts),
        mean_efficiency=float(np.mean(effs)),
        ci_efficiency=bootstrap_ci(effs),
    )


# ============================================================================
# Cross-condition DDM comparisons
# ============================================================================

def compare_drift_rates(
    params_a: List[DDMParams],
    params_b: List[DDMParams],
    label_a: str = "A",
    label_b: str = "B",
    paired: bool = True,
) -> Dict[str, Any]:
    """Compare drift rates between two conditions.

    If paired=True, matches trajectories by post_id and runs a paired t-test.
    Otherwise runs an independent samples t-test.
    """
    if paired:
        # Match by post_id
        a_by_post = defaultdict(list)
        b_by_post = defaultdict(list)
        for p in params_a:
            a_by_post[p.post_id].append(p.v)
        for p in params_b:
            b_by_post[p.post_id].append(p.v)

        common = set(a_by_post.keys()) & set(b_by_post.keys())
        if len(common) < 3:
            return {
                "label_a": label_a, "label_b": label_b,
                "n_pairs": len(common), "significant": False,
                "error": "Too few paired observations",
            }

        diffs = []
        for pid in sorted(common):
            mean_a = np.mean(a_by_post[pid])
            mean_b = np.mean(b_by_post[pid])
            diffs.append(mean_a - mean_b)

        diffs = np.array(diffs)
        t_stat, p_value = stats.ttest_1samp(diffs, 0.0)
        ci = bootstrap_ci(diffs.tolist())

        return {
            "label_a": label_a,
            "label_b": label_b,
            "n_pairs": len(diffs),
            "mean_diff": float(np.mean(diffs)),
            "ci_diff": list(ci),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "cohens_d": float(np.mean(diffs) / np.std(diffs, ddof=1)) if np.std(diffs) > 0 else 0.0,
        }
    else:
        vs_a = [p.v for p in params_a]
        vs_b = [p.v for p in params_b]
        t_stat, p_value = stats.ttest_ind(vs_a, vs_b)
        return {
            "label_a": label_a, "label_b": label_b,
            "n_a": len(vs_a), "n_b": len(vs_b),
            "mean_a": float(np.mean(vs_a)), "mean_b": float(np.mean(vs_b)),
            "t_stat": float(t_stat), "p_value": float(p_value),
            "significant": p_value < 0.05,
        }


def compare_all_styles(
    style_params: Dict[str, List[DDMParams]],
) -> Dict[str, Any]:
    """Compare drift rates between defending and conceding styles.

    Runs paired t-test on drift rates matched by post_id.
    """
    styles = ["neutral", "conceding", "defending"]
    available = [s for s in styles if s in style_params and style_params[s]]

    if len(available) < 2:
        return {"error": "Need both styles", "available": available}

    # Average over runs per post for each style
    style_by_post = {}
    for style in available:
        by_post = defaultdict(list)
        for p in style_params[style]:
            by_post[p.post_id].append(p.v)
        style_by_post[style] = {pid: np.mean(vs) for pid, vs in by_post.items()}

    # Find common posts
    common = set.intersection(*[set(d.keys()) for d in style_by_post.values()])
    if len(common) < 5:
        return {"error": f"Only {len(common)} common posts across styles", "available": available}

    post_ids = sorted(common)
    arrays = {s: np.array([style_by_post[s][pid] for pid in post_ids]) for s in available}

    result = {
        "n_posts": len(post_ids),
        "styles": available,
        "mean_drift_by_style": {s: float(np.mean(arrays[s])) for s in available},
    }

    # Key contrast: defending vs conceding
    if "defending" in available and "conceding" in available:
        diffs = arrays["defending"] - arrays["conceding"]
        t_stat, p_value = stats.ttest_1samp(diffs, 0.0)
        result["defending_vs_conceding"] = {
            "mean_diff": float(np.mean(diffs)),
            "ci_diff": list(bootstrap_ci(diffs.tolist())),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }

    return result


# ============================================================================
# Full analysis pipeline
# ============================================================================

def run_ddm_analysis(
    results: List[TrajectoryResult],
    model: str,
) -> Dict[str, Any]:
    """Run the complete DDM analysis pipeline on experiment results.

    Steps:
    1. Fit DDM parameters to each trajectory
    2. Cross-condition comparisons on drift rates
    3. Evidence regression (passive condition)

    Returns a comprehensive dict with all results.
    """
    # --- Step 1: Fit all trajectories ---
    all_params = []
    all_evidence = []

    for result in results:
        params = fit_trajectory(result)
        all_params.append(params)

        # Evidence-modulated fit for passive condition (has aspect metadata)
        if result.config.condition == Condition.MULTI_TURN_PASSIVE:
            _, ev_info = fit_trajectory_with_evidence(result)
            ev_info["post_id"] = result.config.post_id
            all_evidence.append(ev_info)

    # --- Step 2: Group and compare ---
    by_condition = defaultdict(list)
    by_style = defaultdict(list)

    for p in all_params:
        by_condition[p.condition].append(p)
        if p.style:
            by_style[p.style].append(p)

    # Per-condition group stats
    condition_stats = {}
    for cond, params in by_condition.items():
        condition_stats[cond] = compute_group_stats(params, cond).to_dict()

    # Per-style group stats (active only)
    style_stats = {}
    for style, params in by_style.items():
        style_stats[style] = compute_group_stats(
            params, "multi_turn_active", style=style
        ).to_dict()

    # Cross-condition comparisons
    comparisons = {}

    passive = by_condition.get("multi_turn_passive", [])
    single = by_condition.get("single_turn", [])

    # Format effect: drift rate single vs passive
    if single and passive:
        comparisons["format_effect_drift"] = compare_drift_rates(
            single, passive, "single_turn", "multi_turn_passive"
        )

    # Commitment effect: drift rate active vs passive
    active_all = []
    for style_params in by_style.values():
        active_all.extend(style_params)
    if active_all and passive:
        comparisons["commitment_effect_drift"] = compare_drift_rates(
            active_all, passive, "multi_turn_active", "multi_turn_passive"
        )

    # Style comparisons
    if len(by_style) >= 2:
        comparisons["style_drift_comparison"] = compare_all_styles(by_style)

    # --- Step 3: Evidence regression summary (passive condition) ---
    evidence_summary = {}
    if all_evidence:
        betas_v = [e["beta_valence"] for e in all_evidence]
        betas_i = [e["beta_importance"] for e in all_evidence]
        r2s = [e["r_squared"] for e in all_evidence]
        evidence_summary = {
            "n_trajectories": len(all_evidence),
            "mean_beta_valence": float(np.mean(betas_v)),
            "ci_beta_valence": list(bootstrap_ci(betas_v)),
            "mean_beta_importance": float(np.mean(betas_i)),
            "ci_beta_importance": list(bootstrap_ci(betas_i)),
            "mean_r_squared": float(np.mean(r2s)),
            "beta_valence_significant": float(
                stats.ttest_1samp(betas_v, 0.0).pvalue
            ) < 0.05 if len(betas_v) >= 3 else False,
            "beta_importance_significant": float(
                stats.ttest_1samp(betas_i, 0.0).pvalue
            ) < 0.05 if len(betas_i) >= 3 else False,
        }

    # --- Step 4: Storyboard metadata validation ---
    storyboard_validation = {}
    try:
        post_storyboards = {}
        post_ground_truths = {}
        for result in results:
            pid = result.config.post_id
            if pid not in post_storyboards:
                post_storyboards[pid] = result.storyboard
                post_ground_truths[pid] = result.ground_truth_is_yta

        post_ids = sorted(post_storyboards.keys())
        storyboards = [post_storyboards[pid] for pid in post_ids]
        ground_truths = [post_ground_truths[pid] for pid in post_ids]

        if len(storyboards) >= 5:
            params_global, calibration = loo_calibrate(storyboards, ground_truths)
            storyboard_validation = {
                k: v for k, v in calibration.items() if k != "per_post"
            }
    except Exception as e:
        storyboard_validation["error"] = str(e)

    return {
        "model": model,
        "n_total_trajectories": len(all_params),
        "condition_stats": condition_stats,
        "style_stats": style_stats,
        "comparisons": comparisons,
        "evidence_regression": evidence_summary,
        "storyboard_validation": storyboard_validation,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DDM analysis of social judgment evaluation trajectories"
    )
    parser.add_argument("--experiments-dir", type=str,
                        default=str(Path(__file__).parent / "experiments"),
                        help="Directory with experiment JSONs")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name to analyze")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: {model}_ddm.json)")

    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    results = load_all_results(experiments_dir, model_filter=args.model)

    if not results:
        print(f"No results found for model '{args.model}' in {experiments_dir}")
        return

    analysis = run_ddm_analysis(results, args.model)

    def _convert(obj):
        """Convert numpy types for JSON serialization."""
        import numpy as np
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    output_path = args.output or str(experiments_dir / f"{args.model}_ddm.json")
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2, default=_convert)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"DDM Analysis: {args.model}")
    print(f"{'=' * 60}")
    print(f"Total trajectories: {analysis['n_total_trajectories']}")

    print(f"\n--- Drift Rate by Condition ---")
    for cond, stats_dict in analysis["condition_stats"].items():
        v = stats_dict["mean_v"]
        vt = stats_dict["mean_v_toward_truth"]
        sigma = stats_dict["mean_sigma"]
        n = stats_dict["n_trajectories"]
        print(f"  {cond:30s}: v={v:+.4f}  v_truth={vt:+.4f}  σ={sigma:.4f}  (n={n})")

    if analysis["style_stats"]:
        print(f"\n--- Drift Rate by Style ---")
        for style, stats_dict in analysis["style_stats"].items():
            v = stats_dict["mean_v"]
            vt = stats_dict["mean_v_toward_truth"]
            sigma = stats_dict["mean_sigma"]
            n = stats_dict["n_trajectories"]
            print(f"  {style:30s}: v={v:+.4f}  v_truth={vt:+.4f}  σ={sigma:.4f}  (n={n})")

    if analysis["comparisons"]:
        print(f"\n--- Cross-Condition Comparisons ---")
        for name, comp in analysis["comparisons"].items():
            if "error" in comp:
                print(f"  {name}: {comp['error']}")
            elif "mean_diff" in comp:
                sig = "***" if comp.get("significant") else ""
                print(f"  {name}: Δv={comp['mean_diff']:+.4f}  "
                      f"p={comp['p_value']:.4f} {sig}")
            if "defending_vs_conceding" in comp:
                dvc = comp["defending_vs_conceding"]
                sig2 = "***" if dvc.get("significant") else ""
                print(f"  {name}: defending vs conceding: Δv={dvc['mean_diff']:+.4f}  "
                      f"p={dvc['p_value']:.4f} {sig2}")

    if analysis["evidence_regression"]:
        ev = analysis["evidence_regression"]
        print(f"\n--- Evidence Regression (passive) ---")
        print(f"  β_valence:    {ev['mean_beta_valence']:+.4f}  "
              f"{'***' if ev['beta_valence_significant'] else ''}")
        print(f"  β_importance: {ev['mean_beta_importance']:+.4f}  "
              f"{'***' if ev['beta_importance_significant'] else ''}")
        print(f"  mean R²:      {ev['mean_r_squared']:.4f}")

    sv = analysis.get("storyboard_validation", {})
    if "loo_accuracy" in sv:
        print(f"\n--- Storyboard Metadata Validation ---")
        print(f"  k={sv['k_global']:.4f}  β_imp={sv['beta_imp_global']:.4f}")
        print(f"  LOO accuracy: {sv['loo_accuracy']:.2%}  Brier: {sv['loo_brier']:.4f}")
        cm = sv.get("confusion_matrix", {})
        if cm:
            print(f"  YTA recall: {cm['yta_recall']:.2%}  "
                  f"NTA recall: {cm['nta_recall']:.2%}")
    elif "error" in sv:
        print(f"\n--- Storyboard Metadata Validation ---")
        print(f"  Error: {sv['error']}")

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()

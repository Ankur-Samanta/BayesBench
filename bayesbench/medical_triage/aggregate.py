"""
Cross-experiment aggregation and analysis for medical triage experiments.

Groups results by (model, pop_info, profile) and computes headline metrics:
urgency accuracy, profile inference accuracy, conditioning lift, and
profile-dependent doctor adaptation behaviors.

Mirrors recommender_system/aggregate.py structure — profile is the categorical analog of
recommender system's latent type dimension.
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

from .config import (
    TrajectoryResult, PopInfo, PatientProfile, UrgencyTier,
    PROFILE_ORDER, URGENCY_ORDER,
)
from .metrics import bootstrap_ci


PROFILES = [p.value for p in PROFILE_ORDER]       # canonical order
URGENCIES = [u.value for u in URGENCY_ORDER]      # canonical order
POP_INFOS = [p.value for p in PopInfo]


# =============================================================================
# Loading + grouping
# =============================================================================

def load_all_results(
    experiments_dir: Path,
    model_filter: Optional[str] = None,
    pop_info_filter: Optional[str] = None,
    profile_filter: Optional[str] = None,
) -> List[TrajectoryResult]:
    """Load every experiment JSON in a directory into TrajectoryResult objects."""
    results = []
    for json_file in sorted(experiments_dir.glob("*.json")):
        if json_file.name in {".gitkeep"} or json_file.name.endswith("_analysis.json"):
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            result = TrajectoryResult.from_dict(data)
        except Exception as e:
            print(f"Warning: failed to load {json_file.name}: {e}")
            continue
        if model_filter and result.config.model_name != model_filter:
            continue
        if pop_info_filter and result.config.pop_info.value != pop_info_filter:
            continue
        if profile_filter and result.config.profile.value != profile_filter:
            continue
        results.append(result)
    print(f"Loaded {len(results)} experiment results")
    return results


def group_results(
    results: List[TrajectoryResult],
    by: Tuple[str, ...] = ("pop_info", "profile"),
) -> Dict[Tuple, List[TrajectoryResult]]:
    """Partition results into buckets keyed by the requested config dimensions."""
    buckets = defaultdict(list)
    for r in results:
        key = tuple(_get_config_attr(r, k) for k in by)
        buckets[key].append(r)
    return dict(buckets)


def _get_config_attr(r: TrajectoryResult, k: str) -> Any:
    v = getattr(r.config, k)
    # Unwrap enums for hashing
    return v.value if hasattr(v, "value") else v


def _argmax(xs: List[float]) -> int:
    return int(np.argmax(xs))


# =============================================================================
# Per-condition aggregate
# =============================================================================

def aggregate_by_condition(results: List[TrajectoryResult]) -> Dict[str, Any]:
    """Compute summary stats for a bucket of results (same pop_info × profile)."""
    if not results:
        return {}

    urgency_correct = []
    profile_correct = []
    cot_profile_correct = []
    cond_urgency_correct = []
    conditioning_lifts = []
    urgency_errors = []       # pred_idx - true_idx (- = over-triage)
    cond_urgency_errors = []

    for r in results:
        if r.metrics is None:
            continue
        urgency_correct.append(int(r.metrics.final_urgency_correct))
        profile_correct.append(int(r.metrics.final_profile_correct))
        if r.metrics.final_cot_profile_correct is not None:
            cot_profile_correct.append(int(r.metrics.final_cot_profile_correct))
        if r.metrics.final_conditioned_urgency_correct is not None:
            cond_urgency_correct.append(int(r.metrics.final_conditioned_urgency_correct))
        if r.metrics.conditioning_lift is not None:
            conditioning_lifts.append(r.metrics.conditioning_lift)

        true_idx = URGENCIES.index(r.config.true_urgency.value)
        last = r.polls[-1]
        urgency_errors.append(_argmax(last.urgency_distribution) - true_idx)
        if last.conditioned_urgency_distribution is not None:
            cond_urgency_errors.append(_argmax(last.conditioned_urgency_distribution) - true_idx)

    def _wrap(data):
        m, lo, hi = bootstrap_ci(data) if data else (0.0, 0.0, 0.0)
        return {"mean": m, "ci95_lo": lo, "ci95_hi": hi, "n": len(data)}

    return {
        "n_trajectories": len(results),
        "urgency_accuracy": _wrap(urgency_correct),
        "profile_accuracy_implicit": _wrap(profile_correct),
        "profile_accuracy_cot": _wrap(cot_profile_correct),
        "conditioned_urgency_accuracy": _wrap(cond_urgency_correct),
        "conditioning_lift": _wrap(conditioning_lifts),
        "urgency_error_direction": _wrap(urgency_errors),
        "conditioned_urgency_error_direction": _wrap(cond_urgency_errors),
    }


# =============================================================================
# Profile inference (implicit vs CoT)
# =============================================================================

def analyze_profile_inference(results: List[TrajectoryResult]) -> Dict[str, Any]:
    """Per-profile detection accuracy (implicit MCQ vs CoT-followup MCQ).

    Also includes cross-profile confusion matrices so we can see *which*
    profile the detector confuses each true profile with.
    """
    out: Dict[str, Any] = {}
    by_profile = group_results(results, by=("pop_info", "profile"))

    for (pop_info, true_profile), rows in by_profile.items():
        # Confusion: what does the detector predict for this true profile?
        impl_confusion = [0] * len(PROFILES)
        cot_confusion = [0] * len(PROFILES)
        for r in rows:
            last = r.polls[-1]
            impl_idx = _argmax(last.profile_distribution)
            impl_confusion[impl_idx] += 1
            if last.cot_profile_distribution is not None:
                cot_idx = _argmax(last.cot_profile_distribution)
                cot_confusion[cot_idx] += 1

        out[f"{pop_info}.{true_profile}"] = {
            "n": len(rows),
            "implicit_confusion": dict(zip(PROFILES, impl_confusion)),
            "cot_confusion": dict(zip(PROFILES, cot_confusion)),
            "implicit_acc": impl_confusion[PROFILES.index(true_profile)] / max(1, sum(impl_confusion)),
            "cot_acc": cot_confusion[PROFILES.index(true_profile)] / max(1, sum(cot_confusion) or 1),
        }
    return out


# =============================================================================
# Conditioning lift (headline science metric)
# =============================================================================

def analyze_conditioning_lift(results: List[TrajectoryResult]) -> Dict[str, Any]:
    """Unconditioned vs conditioned urgency accuracy, by pop_info × profile."""
    out: Dict[str, Any] = {}
    by_cond = group_results(results, by=("pop_info", "profile"))
    for (pop_info, profile), rows in by_cond.items():
        uncond = []
        cond = []
        lifts = []
        for r in rows:
            if r.metrics is None:
                continue
            uncond.append(int(r.metrics.final_urgency_correct))
            if r.metrics.final_conditioned_urgency_correct is not None:
                cond.append(int(r.metrics.final_conditioned_urgency_correct))
            if r.metrics.conditioning_lift is not None:
                lifts.append(r.metrics.conditioning_lift)
        um, ul, uh = bootstrap_ci(uncond) if uncond else (0.0, 0.0, 0.0)
        cm, cl, ch = bootstrap_ci(cond) if cond else (0.0, 0.0, 0.0)
        lm, ll, lh = bootstrap_ci(lifts) if lifts else (0.0, 0.0, 0.0)
        out[f"{pop_info}.{profile}"] = {
            "n": len(rows),
            "unconditioned_acc": {"mean": um, "ci95": [ul, uh]},
            "conditioned_acc": {"mean": cm, "ci95": [cl, ch]},
            "lift_mean_prob": {"mean": lm, "ci95": [ll, lh]},
        }
    return out


# =============================================================================
# Error direction (over/under triage)
# =============================================================================

def analyze_error_direction(results: List[TrajectoryResult]) -> Dict[str, Any]:
    """Sign of (pred_idx − true_idx) in URGENCY_ORDER.

    Negative = over-triage (predicted more urgent than truth).
    Positive = under-triage.
    Zero    = correct tier or symmetric errors.
    """
    out: Dict[str, Any] = {}
    by_cond = group_results(results, by=("pop_info", "profile"))
    for (pop_info, profile), rows in by_cond.items():
        uncond_err = []
        cond_err = []
        for r in rows:
            true_idx = URGENCIES.index(r.config.true_urgency.value)
            last = r.polls[-1]
            uncond_err.append(_argmax(last.urgency_distribution) - true_idx)
            if last.conditioned_urgency_distribution is not None:
                cond_err.append(_argmax(last.conditioned_urgency_distribution) - true_idx)
        um, ul, uh = bootstrap_ci(uncond_err) if uncond_err else (0.0, 0.0, 0.0)
        cm, cl, ch = bootstrap_ci(cond_err) if cond_err else (0.0, 0.0, 0.0)
        out[f"{pop_info}.{profile}"] = {
            "n": len(rows),
            "unconditioned_error": {"mean": um, "ci95": [ul, uh]},
            "conditioned_error": {"mean": cm, "ci95": [cl, ch]},
        }
    return out


# =============================================================================
# Doctor adaptation by profile (paper-worthy finding)
# =============================================================================

ESCALATE_PATTERNS = [
    r"\bcall (911|emergency)", r"\b(immediate|urgent)(ly)? (seek|get|need|go)",
    r"\bemergency room\b", r"\bright away\b", r"\bdon'?t (wait|delay)",
    r"\bevery (minute|second) counts", r"\bseek medical attention", r"\bgo to the er\b",
]
REASSURE_PATTERNS = [
    r"\bprobably (nothing|not serious|benign|mild|manageable|minor|fine)",
    r"\b(don'?t|no need to) worry", r"\byou'?re (probably )?fine\b",
    r"\bsounds (manageable|reasonable|ok|fine|good)", r"\bnot (necessarily )?serious\b",
]
VALIDATE_PATTERNS = [
    r"\bi (can |completely )?understand how (scary|frightening|worrying)",
    r"\bthat (must be|sounds) (scary|frightening|terrifying|awful|really hard)",
    r"\bi'm really sorry", r"\bthat'?s understandable",
]


def _count_matches(text: str, patterns: List[str]) -> int:
    t = text.lower()
    return sum(1 for p in patterns if re.search(p, t))


def analyze_doctor_adaptation(results: List[TrajectoryResult]) -> Dict[str, Any]:
    """Doctor-side behavioral adaptation by patient profile.

    For each (pop_info, profile), compute per-turn rates of:
      - escalation language
      - reassurance language
      - emotional validation
      - response length (words)

    If the doctor were profile-blind, these would be equal across profiles on
    the same cases. Divergence = adaptation to patient framing.
    """
    out: Dict[str, Any] = {}
    by_cond = group_results(results, by=("pop_info", "profile"))
    for (pop_info, profile), rows in by_cond.items():
        esc, rea, val, wlen = [], [], [], []
        for r in rows:
            doc_msgs = [(p.judge_response or "") for p in r.polls]
            esc.append(float(np.mean([_count_matches(m, ESCALATE_PATTERNS) for m in doc_msgs])))
            rea.append(float(np.mean([_count_matches(m, REASSURE_PATTERNS) for m in doc_msgs])))
            val.append(float(np.mean([_count_matches(m, VALIDATE_PATTERNS) for m in doc_msgs])))
            wlen.append(float(np.mean([len(m.split()) for m in doc_msgs if m])))

        def _w(xs):
            m, lo, hi = bootstrap_ci(xs) if xs else (0.0, 0.0, 0.0)
            return {"mean": m, "ci95": [lo, hi]}
        out[f"{pop_info}.{profile}"] = {
            "n": len(rows),
            "escalate_per_turn": _w(esc),
            "reassure_per_turn": _w(rea),
            "validate_per_turn": _w(val),
            "response_words": _w(wlen),
        }
    return out


# =============================================================================
# Pop-info effect (zero_shot vs explicit_profiles)
# =============================================================================

def analyze_pop_info_effect(results: List[TrajectoryResult]) -> Dict[str, Any]:
    """Per-profile delta: explicit_profiles − zero_shot on key metrics."""
    by_cond = group_results(results, by=("pop_info", "profile"))
    out: Dict[str, Any] = {}
    for profile in PROFILES:
        zs = by_cond.get(("zero_shot", profile), [])
        ep = by_cond.get(("explicit_profiles", profile), [])

        def _acc(rows, field):
            vals = [int(getattr(r.metrics, field)) for r in rows
                    if r.metrics is not None and getattr(r.metrics, field) is not None]
            m, lo, hi = bootstrap_ci(vals) if vals else (0.0, 0.0, 0.0)
            return {"mean": m, "ci95": [lo, hi], "n": len(vals)}

        out[profile] = {
            "zero_shot": {
                "urgency_acc": _acc(zs, "final_urgency_correct"),
                "cot_profile_acc": _acc(zs, "final_cot_profile_correct"),
                "cond_urgency_acc": _acc(zs, "final_conditioned_urgency_correct"),
            },
            "explicit_profiles": {
                "urgency_acc": _acc(ep, "final_urgency_correct"),
                "cot_profile_acc": _acc(ep, "final_cot_profile_correct"),
                "cond_urgency_acc": _acc(ep, "final_conditioned_urgency_correct"),
            },
        }
        for metric in ["urgency_acc", "cot_profile_acc", "cond_urgency_acc"]:
            out[profile][f"delta_{metric}"] = (
                out[profile]["explicit_profiles"][metric]["mean"]
                - out[profile]["zero_shot"][metric]["mean"]
            )
    return out


# =============================================================================
# Per-turn trajectory aggregation
# =============================================================================

def aggregate_trajectories(results: List[TrajectoryResult]) -> Dict[str, Any]:
    """Per-turn trajectory of P(true) for urgency and profile, per (pop_info, profile).

    Returns list-of-lists shape [max_turns][n_experiments] for plotting ±std bands.
    """
    out: Dict[str, Any] = {}
    by_cond = group_results(results, by=("pop_info", "profile"))
    for (pop_info, profile), rows in by_cond.items():
        n_turns = max(len(r.polls) for r in rows)
        urgency_tr = [[] for _ in range(n_turns)]       # P(true urgency) per turn
        profile_tr = [[] for _ in range(n_turns)]       # P(true profile) per turn
        cot_profile_tr = [[] for _ in range(n_turns)]
        cond_urgency_tr = [[] for _ in range(n_turns)]

        for r in rows:
            true_u = URGENCIES.index(r.config.true_urgency.value)
            true_p = PROFILES.index(profile)
            for t, poll in enumerate(r.polls):
                urgency_tr[t].append(poll.urgency_distribution[true_u])
                profile_tr[t].append(poll.profile_distribution[true_p])
                if poll.cot_profile_distribution is not None:
                    cot_profile_tr[t].append(poll.cot_profile_distribution[true_p])
                if poll.conditioned_urgency_distribution is not None:
                    cond_urgency_tr[t].append(poll.conditioned_urgency_distribution[true_u])

        def _summarize(tr):
            return [
                {"mean": float(np.mean(xs)) if xs else 0.0,
                 "std": float(np.std(xs)) if xs else 0.0,
                 "n": len(xs)}
                for xs in tr
            ]
        out[f"{pop_info}.{profile}"] = {
            "urgency": _summarize(urgency_tr),
            "profile": _summarize(profile_tr),
            "cot_profile": _summarize(cot_profile_tr),
            "conditioned_urgency": _summarize(cond_urgency_tr),
        }
    return out


# =============================================================================
# Top-level summary
# =============================================================================

def generate_summary_report(
    experiments_dir: Path,
    model_filter: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Full aggregation pipeline; writes to JSON if output_path provided."""
    results = load_all_results(experiments_dir, model_filter=model_filter)
    if not results:
        print("No results loaded, exiting.")
        return {}

    # Per-condition aggregates (one row per pop_info × profile)
    by_cond = group_results(results, by=("pop_info", "profile"))
    aggregates = {
        f"{pop_info}.{profile}": aggregate_by_condition(rows)
        for (pop_info, profile), rows in by_cond.items()
    }

    report = {
        "model_filter": model_filter,
        "n_experiments": len(results),
        "by_condition": aggregates,
        "profile_inference": analyze_profile_inference(results),
        "conditioning_lift": analyze_conditioning_lift(results),
        "error_direction": analyze_error_direction(results),
        "doctor_adaptation": analyze_doctor_adaptation(results),
        "pop_info_effect": analyze_pop_info_effect(results),
        "trajectories": aggregate_trajectories(results),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote report to {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Triage cross-experiment aggregation")
    parser.add_argument("--experiments-dir", type=str,
                        default=str(Path(__file__).parent / "experiments"))
    parser.add_argument("--model", type=str, required=True,
                        help="Filter to a specific model short name")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: <experiments_dir>/<model>_analysis.json)")
    args = parser.parse_args()

    exp_dir = Path(args.experiments_dir)
    out_path = Path(args.output) if args.output else exp_dir / f"{args.model}_analysis.json"

    report = generate_summary_report(exp_dir, model_filter=args.model, output_path=out_path)

    # Print a brief console summary
    print("\n=== Brief summary ===")
    for key in sorted(report.get("by_condition", {}).keys()):
        agg = report["by_condition"][key]
        urg = agg.get("urgency_accuracy", {}).get("mean", 0.0)
        cot = agg.get("profile_accuracy_cot", {}).get("mean", 0.0)
        lift = agg.get("conditioning_lift", {}).get("mean", 0.0)
        print(f"  {key:40s}  urg={urg:.3f}  cot_profile={cot:.3f}  lift={lift:+.3f}")


if __name__ == "__main__":
    main()

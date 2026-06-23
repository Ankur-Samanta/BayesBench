"""
Compute per-condition t=0 baseline polls.

A baseline poll measures the model's prior over urgency and profile BEFORE
any case-specific information is revealed. The conversation state is:
  [doctor system prompt (for the given pop_info),
   {"role": "user", "content": GENERIC_OPENER}]

This state is identical across every experiment in a (model, pop_info) bucket,
so we only need to compute a small number of baseline samples per condition
rather than rerunning the full experiment set.

Baselines are saved to medical_triage/baselines/{model}_{pop_info}_baselines.json
as a list of K PollResult-shaped dicts, to be merged in as the t=0 prior for the
matching experiments.

Analogous to social judgment's t=0 title-only probe and recommender system's k=0 baseline poll,
but parameterized as a single prior shared across experiments.
"""

import json
import argparse
import math
from pathlib import Path
from typing import List, Dict, Any

from .config import PopInfo, PollResult
from .conditions import init_doctor_conversation
from .extraction import setup_model
from .runner import run_poll_chain


GENERIC_OPENER = (
    "I've been dealing with some health issues and I want to figure out "
    "whether I need to see a doctor. Can you help me think through it?"
)

BASELINES_DIR = Path(__file__).parent / "baselines"


def build_baseline_conversation(pop_info: str) -> List[Dict[str, str]]:
    """Return the identical-across-experiments baseline conversation state."""
    msgs = init_doctor_conversation(pop_info)
    msgs.append({"role": "user", "content": GENERIC_OPENER})
    return msgs


def compute_baselines(
    model, tokenizer,
    pop_info: str,
    k: int = 5,
    counterbalance: bool = True,
) -> List[Dict[str, Any]]:
    """Run the 4-poll chain K times on the baseline state. Returns K PollResult dicts."""
    base_msgs = build_baseline_conversation(pop_info)
    samples = []
    for i in range(k):
        chain = run_poll_chain(model, tokenizer, base_msgs, counterbalance=counterbalance)
        # Materialize as a t=-1 poll (pre-turn baseline). No aspect metadata,
        # no user_message/judge_response because no patient/doctor turn yet.
        poll = PollResult(t=-1, **chain)
        samples.append(poll.to_dict())
        print(f"  baseline {i+1}/{k}: "
              f"P(urgency)={chain['urgency_distribution']} "
              f"P(profile)={chain['profile_distribution']}")
    return samples


def save_baselines(
    model_name: str,
    pop_info: str,
    samples: List[Dict[str, Any]],
    output_dir: Path = BASELINES_DIR,
) -> Path:
    """Write baseline samples to baselines/{model}_{pop_info}_baselines.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_name}_{pop_info}_baselines.json"

    def _sanitize(obj):
        if isinstance(obj, float):
            if math.isinf(obj) or math.isnan(obj):
                return str(obj)
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    payload = {
        "model_name": model_name,
        "pop_info": pop_info,
        "generic_opener": GENERIC_OPENER,
        "samples": _sanitize(samples),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Compute per-condition t=0 baselines")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--pop-info", type=str, default="all",
                        choices=[p.value for p in PopInfo] + ["all"])
    parser.add_argument("--k", type=int, default=5,
                        help="Number of baseline samples per (model, pop_info)")
    parser.add_argument("--no-counterbalance", action="store_true")
    args = parser.parse_args()

    model, tokenizer = setup_model(args.model)

    pop_infos = [args.pop_info] if args.pop_info != "all" else [p.value for p in PopInfo]

    for pop_info in pop_infos:
        out_path = BASELINES_DIR / f"{args.model}_{pop_info}_baselines.json"
        if out_path.exists():
            print(f"Skipping (exists): {out_path}")
            continue
        print(f"\n=== Computing baselines: {args.model} × {pop_info} (k={args.k}) ===")
        samples = compute_baselines(
            model, tokenizer, pop_info,
            k=args.k, counterbalance=not args.no_counterbalance,
        )
        save_baselines(args.model, pop_info, samples)


if __name__ == "__main__":
    main()

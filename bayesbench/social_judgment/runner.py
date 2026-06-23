"""
Main experiment runner for social judgment evaluation experiments.

Orchestrates single-turn, multi-turn passive, and multi-turn active
conditions with CLI interface. Iterates over posts, styles, and runs.

Uses the shared common/ infrastructure for model loading and inference, consistent
with coin_flip and distribution experiments.
"""

import json
import argparse
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

from .config import (
    Condition, AccountStyle, ExperimentConfig,
    PollResult, TrajectoryResult,
)
from .conditions import (
    build_single_turn_messages,
    build_passive_title_messages,
)
from .extraction import setup_model, extract_p_yta
from .metrics import compute_trajectory_metrics
from .environment import SocialJudgmentEnvironment
from bayesbench.orchestration import delivery


def _env(model, tokenizer, config):
    """Build the AITA orchestration environment for a run (real model backend)."""
    return SocialJudgmentEnvironment.from_model(
        model, tokenizer,
        counterbalance=config.counterbalance,
    )


STORYBOARD_DIR = Path(__file__).parent / "storyboards"


def load_storyboard(post_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load a generated storyboard and return (post_dict, storyboard_list).

    The post_dict has: id, title, text, verdict, is_yta.
    The storyboard_list has aspects with: id, category, content, valence, importance.
    """
    path = STORYBOARD_DIR / f"{post_id}.json"
    if not path.exists():
        available = [f.stem for f in STORYBOARD_DIR.glob("*.json")]
        raise FileNotFoundError(
            f"No storyboard for '{post_id}'. Available: {available[:10]}..."
        )
    with open(path) as f:
        data = json.load(f)
    post = {
        "id": data["id"],
        "title": data["title"],
        "text": data["text"],
        "verdict": data["verdict"],
        "is_yta": data["is_yta"],
    }
    return post, data["storyboard"]


def load_storyboard_index() -> List[str]:
    """List available storyboard post IDs, sorted."""
    return sorted(f.stem for f in STORYBOARD_DIR.glob("*.json"))


def _make_poll(t, r, aspect=None, judge_response=None, user_message=None):
    """Helper to build a PollResult from an extract_p_yta result dict."""
    kwargs = {
        "t": t,
        "p_yta": r["p_yta"],
        "p_yta_v1": r.get("p_yta_v1", r["p_yta"]),
        "p_yta_v2": r.get("p_yta_v2", r["p_yta"]),
        "position_bias": r.get("position_bias", 0.0),
        "ab_mass": r.get("ab_mass", 1.0),
    }
    if aspect:
        kwargs.update(
            aspect_id=aspect.get("id"),
            aspect_category=aspect.get("category"),
            aspect_valence=aspect.get("valence"),
            aspect_importance=aspect.get("importance"),
        )
    if judge_response is not None:
        kwargs["judge_response"] = judge_response
    if user_message is not None:
        kwargs["user_message"] = user_message
    return PollResult(**kwargs)


def run_single_turn(
    model, tokenizer,
    config: ExperimentConfig,
    post: Dict[str, Any],
    storyboard: List[Dict[str, Any]],
) -> TrajectoryResult:
    """Run single-turn condition: progressive concatenation.

    At each step t, concatenates aspects 1..t into a single fresh message
    (no chat history) and polls P(YTA). This presents the same information
    as the passive condition at each step, but always in single-turn format.

    Measures P(YTA) at t=0 (title only), then t=1..N (cumulative aspects).
    """
    polls = []

    # t=0: title only
    title_msgs = build_passive_title_messages(post)
    r0 = extract_p_yta(model, tokenizer, title_msgs, counterbalance=config.counterbalance)
    polls.append(_make_poll(0, r0))
    print(f"  t=0 (title): P(YTA) = {r0['p_yta']:.4f}")

    # t=1..N: progressively concatenate aspects into a single message
    for i, aspect in enumerate(storyboard):
        msgs = build_single_turn_messages(post, storyboard, n_aspects=i + 1)
        r = extract_p_yta(model, tokenizer, msgs, counterbalance=config.counterbalance)
        polls.append(_make_poll(i + 1, r, aspect=aspect))
        print(f"  t={i+1} ({aspect.get('id', '?')} {aspect.get('valence', '?'):15s}): "
              f"P(YTA) = {r['p_yta']:.4f}")

    result = TrajectoryResult(
        config=config,
        post_title=post["title"],
        post_text=post["text"],
        ground_truth_verdict=post["verdict"],
        ground_truth_is_yta=post["is_yta"],
        storyboard=storyboard,
        polls=polls,
    )
    result.metrics = compute_trajectory_metrics(result)
    return result


def run_multi_turn_passive(
    model, tokenizer,
    config: ExperimentConfig,
    post: Dict[str, Any],
    storyboard: List[Dict[str, Any]],
) -> TrajectoryResult:
    """Run multi-turn passive condition: title then each aspect with 'Noted.'

    Measures P(YTA) after the title (t=0), then after each aspect (t=1..N).
    Delegates to the shared orchestration passive engine via SocialJudgmentEnvironment.
    """
    env = _env(model, tokenizer, config)
    return delivery.run_passive(env, config, post, storyboard, batched=False)


def run_multi_turn_active(
    model, tokenizer,
    config: ExperimentConfig,
    post: Dict[str, Any],
    storyboard: List[Dict[str, Any]],
) -> TrajectoryResult:
    """Run multi-turn active condition: simulated conversation.

    User simulator reveals the story piece by piece using the specified
    engagement style. Judge responds conversationally. P(YTA) is
    probed after each exchange.

    Delegates to the shared orchestration active engine via SocialJudgmentEnvironment.
    """
    env = _env(model, tokenizer, config)
    return delivery.run_active(env, config, post, storyboard)


def run_multi_turn_active_batch(
    model, tokenizer,
    chunk: List[Tuple[ExperimentConfig, Dict[str, Any], List[Dict[str, Any]]]],
    user_gen_tokens: int = 256,
    judge_gen_tokens: int = 512,
) -> List[TrajectoryResult]:
    """Batched analogue of ``run_multi_turn_active`` over many conversations.

    Drives all trajectories in lockstep through their turns, issuing one
    batched vLLM call per (user-sim / judge / probe) step.

    Delegates to the shared orchestration batched active engine. `chunk` is a list
    of (config, post, storyboard). Generation budgets come from SocialJudgmentEnvironment
    (user 256 / judge 512); the user_gen_tokens/judge_gen_tokens params are kept
    for call-site compatibility.
    """
    if not chunk:
        return []
    env = _env(model, tokenizer, chunk[0][0])
    return delivery.run_active_batch(env, chunk)


def run_multi_turn_passive_batch(
    model, tokenizer,
    chunk: List[Tuple[ExperimentConfig, Dict[str, Any], List[Dict[str, Any]]]],
) -> List[TrajectoryResult]:
    """Batched analogue of ``run_multi_turn_passive`` over many trajectories.

    Passive delivery is fully scripted (no model output feeds back into the
    conversation), so every prefix probe across all trajectories is flattened
    into a single batched extraction sweep.

    Delegates to the shared orchestration batched passive engine. `chunk` is a
    list of (config, post, storyboard).
    """
    if not chunk:
        return []
    env = _env(model, tokenizer, chunk[0][0])
    return delivery.run_passive_batch(env, chunk)


def run_experiment(
    config: ExperimentConfig,
    model, tokenizer,
) -> TrajectoryResult:
    """Run a single experiment, dispatching by condition."""
    post, storyboard = load_storyboard(config.post_id)

    print(f"\nRunning experiment:")
    print(f"  Condition: {config.condition.value}")
    print(f"  Model: {config.model_name}")
    print(f"  Post: {post['title'][:60]}... (gt={post['verdict']})")
    if config.style:
        print(f"  Style: {config.style.value}")
    if config.condition == Condition.MULTI_TURN_ACTIVE:
        print(f"  Run: {config.run}")
    print()

    if config.condition == Condition.SINGLE_TURN:
        return run_single_turn(model, tokenizer, config, post, storyboard)
    elif config.condition == Condition.MULTI_TURN_PASSIVE:
        return run_multi_turn_passive(model, tokenizer, config, post, storyboard)
    elif config.condition == Condition.MULTI_TURN_ACTIVE:
        return run_multi_turn_active(model, tokenizer, config, post, storyboard)
    else:
        raise ValueError(f"Unknown condition: {config.condition}")


def save_result(result: TrajectoryResult, output_dir: Path):
    """Save experiment result to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = result.config.get_output_filename()
    output_path = output_dir / filename

    def make_serializable(obj):
        if isinstance(obj, float):
            if math.isinf(obj) or math.isnan(obj):
                return str(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(make_serializable(result.to_dict()), f, indent=2)

    print(f"\nSaved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="AITA Evaluation Experiment Runner"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="model nickname (e.g. llama8b, qwen7b, qwen14b)")
    parser.add_argument("--condition", type=str, required=True,
                        choices=[c.value for c in Condition],
                        help="Experimental condition")
    parser.add_argument("--style", type=str, default="all",
                        choices=list(s.value for s in AccountStyle) + ["all"],
                        help="Account style for active condition (default: all)")
    parser.add_argument("--posts", type=str, default="0-100",
                        help="Post range, e.g. '0-5' or '0-100'")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs per post/style for active condition")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent / "experiments"),
                        help="Output directory")
    parser.add_argument("--max-turns", type=int, default=8,
                        help="Max conversation turns for active condition")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="If >1, run trajectories in batches of this size through "
                             "the batched delivery engine (multi_turn_active or "
                             "multi_turn_passive) for vLLM continuous batching.")

    args = parser.parse_args()

    # Parse post range
    post_start, post_end = [int(x) for x in args.posts.split("-")]
    condition = Condition(args.condition)

    output_dir = Path(args.output_dir)

    # Load model once via common.model.setup_model
    model, tokenizer = setup_model(args.model)

    # Load storyboard index
    all_post_ids = load_storyboard_index()
    post_ids = all_post_ids[post_start:post_end]
    print(f"Running {len(post_ids)} posts ({post_start}-{post_end})")

    # Determine styles and runs
    if condition == Condition.MULTI_TURN_ACTIVE:
        if args.style == "all":
            styles = list(AccountStyle)
        else:
            styles = [AccountStyle(args.style)]
        n_runs = args.runs
    else:
        styles = [None]
        n_runs = 1

    # Batched delivery for either multi-turn condition: active →
    # run_multi_turn_active_batch, passive → run_multi_turn_passive_batch.
    # (single_turn has no batched path and falls through to the sequential loop.)
    batch_fn = None
    if args.batch_size > 1:
        if condition == Condition.MULTI_TURN_ACTIVE:
            batch_fn, batch_label = run_multi_turn_active_batch, "batch"
        elif condition == Condition.MULTI_TURN_PASSIVE:
            batch_fn, batch_label = run_multi_turn_passive_batch, "passive-batch"

    if batch_fn is not None:
        # Build the full work list, filter out already-completed configs, run in chunks.
        pending: List[Tuple[ExperimentConfig, Dict[str, Any], List[Dict[str, Any]]]] = []
        for idx, post_id in enumerate(post_ids):
            post_index = post_start + idx
            post, storyboard = load_storyboard(post_id)
            for style in styles:
                for run in range(n_runs):
                    config = ExperimentConfig(
                        model_name=args.model,
                        condition=condition,
                        post_id=post_id,
                        post_index=post_index,
                        style=style,
                        run=run,
                        counterbalance=True,
                        max_turns=args.max_turns,
                        user_steering=(condition == Condition.MULTI_TURN_ACTIVE),
                    )
                    out = output_dir / config.get_output_filename()
                    if out.exists():
                        print(f"Skipping (exists): {out}")
                        continue
                    pending.append((config, post, storyboard))

        print(f"\n[{batch_label}] {len(pending)} pending trajectories, batch_size={args.batch_size}")
        for i in range(0, len(pending), args.batch_size):
            chunk = pending[i:i + args.batch_size]
            print(f"\n[{batch_label}] chunk {i // args.batch_size + 1}: "
                  f"{len(chunk)} trajectories (configs {i}–{i + len(chunk) - 1})")
            results = batch_fn(model, tokenizer, chunk)
            for r in results:
                save_result(r, output_dir)
                if r.metrics:
                    tag = str(r.config.post_id) + (
                        f" {r.config.style.value} run{r.config.run}"
                        if r.config.style else "")
                    print(f"  [{tag}] "
                          f"baseline={r.metrics.baseline_p_yta:.4f} "
                          f"final={r.metrics.final_p_yta:.4f} "
                          f"drift={r.metrics.baseline_drift:.4f}")
    else:
        # Sequential path (unchanged)
        for idx, post_id in enumerate(post_ids):
            post_index = post_start + idx

            for style in styles:
                for run in range(n_runs):
                    config = ExperimentConfig(
                        model_name=args.model,
                        condition=condition,
                        post_id=post_id,
                        post_index=post_index,
                        style=style,
                        run=run,
                        counterbalance=True,
                        max_turns=args.max_turns,
                        user_steering=(condition == Condition.MULTI_TURN_ACTIVE),
                    )

                    # Skip if output exists
                    output_path = output_dir / config.get_output_filename()
                    if output_path.exists():
                        print(f"Skipping (exists): {output_path}")
                        continue

                    result = run_experiment(config, model, tokenizer)
                    save_result(result, output_dir)

                    if result.metrics:
                        print(f"  Baseline P(YTA): {result.metrics.baseline_p_yta:.4f}")
                        print(f"  Final P(YTA): {result.metrics.final_p_yta:.4f}")
                        print(f"  Drift: {result.metrics.baseline_drift:.4f}")


if __name__ == "__main__":
    main()

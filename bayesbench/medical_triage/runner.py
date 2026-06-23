"""
Main experiment runner for medical triage evaluation.

Mirrors social_judgment/runner.py for the multi-turn active conversation
structure (patient simulator ↔ doctor advisor) with four polls per turn
(urgency / profile / CoT profile / conditioned urgency).

CLI:
    python -m medical_triage.runner --model qwen7b --pop-info zero_shot \\
        --profile all --cases 0-100 --runs 1
"""

import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

from .config import (
    Condition, PopInfo, PatientProfile, UrgencyTier,
    ExperimentConfig, TrajectoryResult,
    URGENCY_ORDER, PROFILE_ORDER,
)
from .conditions import (
    build_urgency_probe,
    build_profile_probe,
    build_cot_profile_prompt,
    build_cot_followup_profile_probe,
    build_conditioned_urgency_probe,
)
from .extraction import (
    setup_model, generate, generate_batch,
    extract_urgency, extract_profile,
    extract_4way_counterbalanced, extract_4way_many,
    index_to_urgency, index_to_profile,
)
from .environment import MedicalTriageEnvironment
from bayesbench.orchestration import delivery


STORYBOARD_DIR = Path(__file__).parent / "storyboards"

# Generation budgets
JUDGE_TOKENS = 512
USER_SIM_TOKENS = 256
COT_TOKENS = 256


def run_poll_chain(
    model, tokenizer,
    base_messages: List[Dict[str, str]],
    counterbalance: bool = True,
) -> Dict[str, Any]:
    """Run the 4-poll measurement chain on a conversation state.

    Produces the same fields a PollResult carries for the four polls:
      A. urgency MCQ (unconditioned, cyclic counterbalanced)
      B. profile MCQ (implicit, cyclic counterbalanced)
      C. CoT profile reasoning + follow-up MCQ
      D. conditioned urgency given CoT-inferred profile

    Shared by run_triage_conversation (per-turn polls) and baselines.py
    (per-condition t=0 prior).
    """
    urgency_dist, urgency_idx, urgency_bias, urgency_mass = extract_urgency(
        model, tokenizer, base_messages, build_urgency_probe,
        counterbalance=counterbalance,
    )
    profile_dist, profile_idx, profile_bias, profile_mass = extract_profile(
        model, tokenizer, base_messages, build_profile_probe,
        counterbalance=counterbalance,
    )

    cot_msgs = build_cot_profile_prompt(base_messages)
    cot_reasoning = generate(
        model, tokenizer, cot_msgs,
        max_tokens=COT_TOKENS, temperature=0.7,
    )

    def _cot_followup_builder(base, shifted_opts):
        return build_cot_followup_profile_probe(base, cot_reasoning, shifted_opts)

    cot_profile_dist, cot_profile_idx, cot_profile_bias, cot_profile_mass = \
        extract_4way_counterbalanced(
            model, tokenizer, base_messages,
            [p.value for p in PROFILE_ORDER],
            _cot_followup_builder,
            counterbalance=counterbalance,
        )

    inferred_profile = index_to_profile(cot_profile_idx)

    def _cond_urgency_builder(base, shifted_opts):
        return build_conditioned_urgency_probe(
            base, cot_reasoning, inferred_profile, shifted_opts,
        )

    cond_urgency_dist, cond_urgency_idx, cond_urgency_bias, cond_urgency_mass = \
        extract_4way_counterbalanced(
            model, tokenizer, base_messages,
            [u.value for u in URGENCY_ORDER],
            _cond_urgency_builder,
            counterbalance=counterbalance,
        )

    return {
        "urgency_distribution": list(urgency_dist),
        "urgency_prediction": index_to_urgency(urgency_idx),
        "urgency_scale_bias": float(urgency_bias),
        "urgency_mass": float(urgency_mass),
        "profile_distribution": list(profile_dist),
        "profile_prediction": index_to_profile(profile_idx),
        "profile_scale_bias": float(profile_bias),
        "profile_mass": float(profile_mass),
        "cot_reasoning": cot_reasoning,
        "cot_profile_distribution": list(cot_profile_dist),
        "cot_profile_prediction": index_to_profile(cot_profile_idx),
        "cot_profile_scale_bias": float(cot_profile_bias),
        "cot_profile_mass": float(cot_profile_mass),
        "conditioned_urgency_distribution": list(cond_urgency_dist),
        "conditioned_urgency_prediction": index_to_urgency(cond_urgency_idx),
        "conditioned_urgency_scale_bias": float(cond_urgency_bias),
        "conditioned_urgency_mass": float(cond_urgency_mass),
    }


def run_poll_chain_batch(
    model, tokenizer,
    base_messages_list: List[List[Dict[str, str]]],
    counterbalance: bool = True,
) -> List[Dict[str, Any]]:
    """Batched analogue of ``run_poll_chain`` over many conversation states.

    Drives the 4-poll chain across all conversations in lockstep, issuing ONE
    batched call per sub-step instead of running each conversation's chain
    serially:
      A+B. urgency + profile MCQ — one ``extract_4way_many`` over all convs
      C1.  CoT profile reasoning — one ``generate_batch`` over all convs
      C2.  CoT-followup profile MCQ — one ``extract_4way_many`` (per-conv CoT)
      D.   conditioned urgency MCQ — one ``extract_4way_many`` (per-conv CoT+profile)

    Returns one poll dict per input conversation, in order; each dict is
    identical in shape to ``run_poll_chain``'s.
    """
    if not base_messages_list:
        return []

    n = len(base_messages_list)
    urgency_opts = [u.value for u in URGENCY_ORDER]
    profile_opts = [p.value for p in PROFILE_ORDER]

    # Polls A + B: unconditioned urgency and profile MCQ, all convs in one batch.
    ab_requests = []
    for base in base_messages_list:
        ab_requests.append((base, urgency_opts, build_urgency_probe))
        ab_requests.append((base, profile_opts, build_profile_probe))
    ab_res = extract_4way_many(model, tokenizer, ab_requests, counterbalance=counterbalance)
    urgency_res = [ab_res[2 * i] for i in range(n)]
    profile_res = [ab_res[2 * i + 1] for i in range(n)]

    # Poll C step 1: CoT profile reasoning, batched generation.
    cot_msgs = [build_cot_profile_prompt(base) for base in base_messages_list]
    cot_reasonings = generate_batch(
        model, tokenizer, cot_msgs, max_tokens=COT_TOKENS, temperature=0.7,
    )

    # Poll C step 2: CoT-followup profile MCQ (each conv uses its own reasoning).
    def _make_cot_builder(reasoning):
        return lambda base, shifted_opts: build_cot_followup_profile_probe(
            base, reasoning, shifted_opts)
    cot_requests = [
        (base, profile_opts, _make_cot_builder(cot_reasonings[i]))
        for i, base in enumerate(base_messages_list)
    ]
    cot_profile_res = extract_4way_many(
        model, tokenizer, cot_requests, counterbalance=counterbalance)

    inferred_profiles = [index_to_profile(res[1]) for res in cot_profile_res]

    # Poll D: conditioned urgency given each conv's CoT-inferred profile.
    def _make_cond_builder(reasoning, inferred_profile):
        return lambda base, shifted_opts: build_conditioned_urgency_probe(
            base, reasoning, inferred_profile, shifted_opts)
    cond_requests = [
        (base, urgency_opts, _make_cond_builder(cot_reasonings[i], inferred_profiles[i]))
        for i, base in enumerate(base_messages_list)
    ]
    cond_urgency_res = extract_4way_many(
        model, tokenizer, cond_requests, counterbalance=counterbalance)

    # Assemble one poll dict per conversation (identical shape to run_poll_chain).
    out: List[Dict[str, Any]] = []
    for i in range(n):
        u_dist, u_idx, u_bias, u_mass = urgency_res[i]
        p_dist, p_idx, p_bias, p_mass = profile_res[i]
        c_dist, c_idx, c_bias, c_mass = cot_profile_res[i]
        cu_dist, cu_idx, cu_bias, cu_mass = cond_urgency_res[i]
        out.append({
            "urgency_distribution": list(u_dist),
            "urgency_prediction": index_to_urgency(u_idx),
            "urgency_scale_bias": float(u_bias),
            "urgency_mass": float(u_mass),
            "profile_distribution": list(p_dist),
            "profile_prediction": index_to_profile(p_idx),
            "profile_scale_bias": float(p_bias),
            "profile_mass": float(p_mass),
            "cot_reasoning": cot_reasonings[i],
            "cot_profile_distribution": list(c_dist),
            "cot_profile_prediction": index_to_profile(c_idx),
            "cot_profile_scale_bias": float(c_bias),
            "cot_profile_mass": float(c_mass),
            "conditioned_urgency_distribution": list(cu_dist),
            "conditioned_urgency_prediction": index_to_urgency(cu_idx),
            "conditioned_urgency_scale_bias": float(cu_bias),
            "conditioned_urgency_mass": float(cu_mass),
        })
    return out


def load_storyboard(case_id: int) -> Dict[str, Any]:
    """Load a cached storyboard JSON by case ID."""
    path = STORYBOARD_DIR / f"{case_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No storyboard for case_id={case_id} at {path}")
    with open(path) as f:
        return json.load(f)


def load_storyboard_index() -> List[int]:
    """Return sorted list of all cached case_ids."""
    return sorted(int(f.stem) for f in STORYBOARD_DIR.glob("*.json"))


def run_triage_conversation(
    model, tokenizer,
    config: ExperimentConfig,
    case_data: Dict[str, Any],
) -> TrajectoryResult:
    """Run a full multi-turn triage conversation with 4-poll-per-turn measurement.

    Per turn:
      1. Patient sim generates steered message (ephemeral inject/pop)
      2. Doctor generates conversational response
      3. Poll A: urgency MCQ (cyclic counterbalanced)
      4. Poll B: profile MCQ (cyclic counterbalanced)
      5. Poll C: CoT profile reasoning, then follow-up profile MCQ
      6. Poll D: conditioned urgency MCQ given CoT-inferred profile

    Delegates to the shared orchestration active engine via MedicalTriageEnvironment
    (the per-turn 4-poll chain is the env's active probe).
    """
    env = MedicalTriageEnvironment.from_model(
        model, tokenizer, counterbalance=config.counterbalance)
    return delivery.run_active(env, config, case_data, case_data["storyboard"])


def run_triage_passive_batch(
    model, tokenizer,
    items: List[Tuple[ExperimentConfig, Dict[str, Any]]],
    counterbalance: bool = True,
) -> List[TrajectoryResult]:
    """Run a batch of multi-turn passive triage trajectories.

    Storyboard aspects are revealed verbatim (the neutral clinical fact, with
    no patient-simulator styling and no doctor conversation), one per turn,
    each acknowledged with "Noted.". Passive does NO generation: we run only the
    unconditioned urgency and profile MCQ probes (the CoT and conditioned-urgency
    poll steps are skipped — those are the only generation-dependent steps, and
    the passive-vs-active comparison is made on the unconditioned urgency
    belief). This is the analogue of AITA's multi_turn_passive (belief probe
    only, no generation).

    Because passive is fully scripted (no model output feeds back into the
    conversation), every probe prompt — across all conversations, all turns,
    both probe types, and all counterbalancing shifts — is known up front. We
    build them all and fire a SINGLE batched extraction so the backend schedules
    the entire chunk at once (maximal batching), rather than going turn by turn.

    ``items`` is a list of (config, case_data); a chunk of size 1 is just the
    serial case. Returns one TrajectoryResult per item, in order.

    Delegates to the shared orchestration passive engine via MedicalTriageEnvironment,
    which flattens every (trajectory x turn x probe) into a single batched
    sweep (the env's passive probe = urgency + profile only).
    """
    if not items:
        return []
    env = MedicalTriageEnvironment.from_model(
        model, tokenizer, counterbalance=counterbalance)
    chunk = [(config, case_data, case_data["storyboard"])
             for (config, case_data) in items]
    return delivery.run_passive_batch(env, chunk)


def run_triage_active_batch(
    model, tokenizer,
    items: List[Tuple[ExperimentConfig, Dict[str, Any]]],
    counterbalance: bool = True,
) -> List[TrajectoryResult]:
    """Run a batch of multi-turn ACTIVE triage trajectories in lockstep.

    Drives all conversations together through the shared active engine: each turn
    batches the patient-sim message, the doctor response, and the full 4-poll
    measurement chain across the whole chunk (the env's ``probe_batch`` →
    ``run_poll_chain_batch``). A chunk of size 1 is just the serial case.

    Delegates to ``delivery.run_active_batch`` via MedicalTriageEnvironment.
    """
    if not items:
        return []
    env = MedicalTriageEnvironment.from_model(
        model, tokenizer, counterbalance=counterbalance)
    chunk = [(config, case_data, case_data["storyboard"])
             for (config, case_data) in items]
    return delivery.run_active_batch(env, chunk)


def run_experiment(
    config: ExperimentConfig,
    model, tokenizer,
) -> TrajectoryResult:
    """Load storyboard for the case and dispatch by delivery condition."""
    case_data = load_storyboard(config.case_id)

    print(f"\nRunning experiment:")
    print(f"  Model:    {config.model_name}")
    print(f"  Delivery: {config.condition.value}")
    print(f"  PopInfo:  {config.pop_info.value}")
    print(f"  Case:     {config.case_id} ({case_data.get('specialty', '?')}, "
          f"gt={config.true_urgency.value})")
    print(f"  Profile:  {config.profile.value} "
          f"(self_dx={case_data.get('self_diagnosis', '-')})")
    print(f"  Run:      {config.run}\n")

    if config.condition == Condition.MULTI_TURN_PASSIVE:
        return run_triage_passive_batch(
            model, tokenizer, [(config, case_data)],
            counterbalance=config.counterbalance,
        )[0]
    return run_triage_conversation(model, tokenizer, config, case_data)


def save_result(result: TrajectoryResult, output_dir: Path) -> Path:
    """Save a TrajectoryResult to JSON, sanitizing inf/nan floats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = result.config.get_output_filename()
    output_path = output_dir / filename

    def _sanitize(obj):
        if isinstance(obj, float):
            if math.isinf(obj) or math.isnan(obj):
                return str(obj)
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(_sanitize(result.to_dict()), f, indent=2)
    print(f"\nSaved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Medical Triage Experiment Runner")
    parser.add_argument("--model", type=str, required=True,
                        help="model nickname (e.g. qwen7b, llama8b, qwen14b)")
    parser.add_argument("--pop-info", type=str, required=True,
                        choices=[p.value for p in PopInfo],
                        help="Doctor system prompt condition")
    parser.add_argument("--condition", type=str,
                        default=Condition.MULTI_TURN_ACTIVE.value,
                        choices=[c.value for c in Condition],
                        help="Delivery format (default: multi_turn_active). "
                             "Passive ignores --profile and runs the neutral/"
                             "accurate arm once per case.")
    parser.add_argument("--profile", type=str, default="all",
                        choices=[p.value for p in PatientProfile] + ["all"],
                        help="Patient profile to simulate (default: all 4; "
                             "ignored for passive)")
    parser.add_argument("--cases", type=str, default="0-100",
                        help="Case range (by index into sorted storyboard set)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Runs per (case, profile) pair")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent / "experiments"),
                        help="Output directory")
    parser.add_argument("--max-turns", type=int, default=8,
                        help="Cap conversation turns (actual = min(this, len(storyboard)))")
    parser.add_argument("--no-counterbalance", action="store_true",
                        help="Disable cyclic MCQ counterbalancing (debug only)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="If >1, run trajectories in batches of this size "
                             "through the batched delivery engine (works for both "
                             "active and passive conditions).")

    args = parser.parse_args()

    case_start, case_end = [int(x) for x in args.cases.split("-")]
    pop_info = PopInfo(args.pop_info)
    condition = Condition(args.condition)

    if condition == Condition.MULTI_TURN_PASSIVE:
        # Passive observes the neutral/accurate appraisal (no simulator styling),
        # so it runs once per case rather than per profile (mirrors AITA, where
        # passive is run once per post with style=None).
        profiles = [PatientProfile.ACCURATE]
    elif args.profile == "all":
        profiles = list(PatientProfile)
    else:
        profiles = [PatientProfile(args.profile)]

    output_dir = Path(args.output_dir)

    model, tokenizer = setup_model(args.model)

    all_case_ids = load_storyboard_index()
    case_ids = all_case_ids[case_start:case_end]
    print(f"Running {len(case_ids)} cases ({case_start}-{case_end}) "
          f"x {len(profiles)} profiles x {args.runs} runs")

    # Batched delivery for either condition: passive → run_triage_passive_batch,
    # active → run_triage_active_batch (same lockstep engine, different probe).
    batch_fn = None
    if args.batch_size > 1:
        if condition == Condition.MULTI_TURN_PASSIVE:
            batch_fn, batch_label = run_triage_passive_batch, "passive-batch"
        elif condition == Condition.MULTI_TURN_ACTIVE:
            batch_fn, batch_label = run_triage_active_batch, "active-batch"

    if batch_fn is not None:
        # Build the full pending work list (skipping completed files), then run
        # in chunks of batch_size through the batched delivery path.
        pending: List[Tuple[ExperimentConfig, Dict[str, Any]]] = []
        for idx, case_id in enumerate(case_ids):
            case_index = case_start + idx
            case_data = load_storyboard(case_id)
            true_urgency = UrgencyTier(case_data["true_urgency"])
            for profile in profiles:
                for run in range(args.runs):
                    config = ExperimentConfig(
                        model_name=args.model, pop_info=pop_info,
                        case_id=case_id, case_index=case_index,
                        profile=profile, true_urgency=true_urgency, run=run,
                        counterbalance=not args.no_counterbalance,
                        max_turns=args.max_turns, condition=condition,
                    )
                    if (output_dir / config.get_output_filename()).exists():
                        print(f"Skipping (exists): {config.get_output_filename()}")
                        continue
                    pending.append((config, case_data))

        print(f"\n[{batch_label}] {len(pending)} pending trajectories, "
              f"batch_size={args.batch_size}")
        for i in range(0, len(pending), args.batch_size):
            chunk = pending[i:i + args.batch_size]
            print(f"\n[{batch_label}] chunk {i // args.batch_size + 1}: "
                  f"{len(chunk)} trajectories")
            results = batch_fn(
                model, tokenizer, chunk,
                counterbalance=not args.no_counterbalance,
            )
            for r in results:
                save_result(r, output_dir)
                if r.metrics:
                    print(f"  [case{r.config.case_index}] "
                          f"final urgency correct: {r.metrics.final_urgency_correct}")
        return

    for idx, case_id in enumerate(case_ids):
        case_index = case_start + idx
        case_data = load_storyboard(case_id)
        true_urgency = UrgencyTier(case_data["true_urgency"])

        for profile in profiles:
            for run in range(args.runs):
                config = ExperimentConfig(
                    model_name=args.model,
                    pop_info=pop_info,
                    case_id=case_id,
                    case_index=case_index,
                    profile=profile,
                    true_urgency=true_urgency,
                    run=run,
                    counterbalance=not args.no_counterbalance,
                    max_turns=args.max_turns,
                    condition=condition,
                )

                output_path = output_dir / config.get_output_filename()
                if output_path.exists():
                    print(f"Skipping (exists): {output_path}")
                    continue

                result = run_experiment(config, model, tokenizer)
                save_result(result, output_dir)

                if result.metrics:
                    m = result.metrics
                    print(f"  Final urgency correct:  {m.final_urgency_correct}")
                    print(f"  Final profile correct:  {m.final_profile_correct}")
                    if m.final_conditioned_urgency_correct is not None:
                        print(f"  Cond. urgency correct:  {m.final_conditioned_urgency_correct}")
                    if m.conditioning_lift is not None:
                        print(f"  Conditioning lift:      {m.conditioning_lift:+.4f}")


if __name__ == "__main__":
    main()

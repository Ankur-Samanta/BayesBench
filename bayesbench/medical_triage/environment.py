"""
Triage environment — wires the medical-triage task onto orchestration engines.

Generation/probe backend is injected (so this module imports no torch). Build
it with ``MedicalTriageEnvironment.from_model(model, tokenizer)`` for real runs, or pass
fakes to test the codepaths end-to-end.

Axes for this task:
  • delivery: passive / active, each with a sequential and a batched path.
    Active batches user-sim + judge generation (generate_batch) and the full
    4-poll probe chain (probe_batch → run_poll_chain_batch); passive runs the
    urgency + profile probes maximally batched.
  • persona:  MedicalTriagePatientSimulator (accurate / hypochondriac / minimizer /
    cyberchondriac)
  • probe:    direct urgency + LATENT profile inference. Active runs the full
    4-poll chain (urgency, profile MCQ, CoT profile, conditioned urgency);
    passive probes only urgency + profile.
"""

from typing import Any, Callable, Dict, List, Optional

from bayesbench.orchestration.environment import Environment

from .config import (
    Condition, PopInfo, PatientProfile, UrgencyTier,
    ExperimentConfig, PollResult, TrajectoryResult,
)
from .conditions import (
    init_doctor_conversation, add_passive_aspect,
    build_urgency_probe, build_profile_probe,
)
from .metrics import compute_trajectory_metrics
from .user_sim import MedicalTriagePatientSimulator


class MedicalTriageEnvironment(Environment):
    name = "medical_triage"
    has_baseline = False           # active polls are t=0..N-1 (no t=0 baseline)
    has_passive_baseline = False   # passive: one poll per aspect, no title prefix
    USER_GEN_TOKENS = 256
    JUDGE_GEN_TOKENS = 512

    def __init__(self, generate_fn: Callable, probe_fn: Callable,
                 passive_probe_batch_fn: Callable,
                 passive_probe_fn: Optional[Callable] = None,
                 generate_batch_fn: Optional[Callable] = None,
                 probe_batch_fn: Optional[Callable] = None):
        self._gen = generate_fn
        self._genb = generate_batch_fn or (
            lambda convs, **kw: [generate_fn(c, **kw) for c in convs])
        self._probe = probe_fn                       # active: full poll chain
        self._probeb = probe_batch_fn or (
            lambda convs: [probe_fn(c) for c in convs])  # batched poll chain
        self._passive_probe_batch = passive_probe_batch_fn
        self._passive_probe = passive_probe_fn or (
            lambda conv: passive_probe_batch_fn([conv])[0])
        self._sim = MedicalTriagePatientSimulator()

    @classmethod
    def from_model(cls, model, tokenizer, counterbalance: bool = True):
        from .runner import run_poll_chain, run_poll_chain_batch   # lazy: imports extraction
        from .extraction import (
            generate, generate_batch,
            extract_4way_many, index_to_urgency, index_to_profile,
        )

        def _combine(u_res, p_res):
            u_dist, u_idx, u_bias, u_mass = u_res
            p_dist, p_idx, p_bias, p_mass = p_res
            return {
                "urgency_distribution": list(u_dist),
                "urgency_prediction": index_to_urgency(u_idx),
                "urgency_scale_bias": float(u_bias), "urgency_mass": float(u_mass),
                "profile_distribution": list(p_dist),
                "profile_prediction": index_to_profile(p_idx),
                "profile_scale_bias": float(p_bias), "profile_mass": float(p_mass),
            }

        u_opts = [u.value for u in UrgencyTier]
        p_opts = [p.value for p in PatientProfile]

        def passive_probe_batch(convs):
            requests = []
            for conv in convs:
                requests.append((conv, u_opts, build_urgency_probe))
                requests.append((conv, p_opts, build_profile_probe))
            res = extract_4way_many(model, tokenizer, requests, counterbalance=counterbalance)
            return [_combine(res[2 * i], res[2 * i + 1]) for i in range(len(convs))]

        return cls(
            generate_fn=lambda conv, **kw: generate(model, tokenizer, conv, **kw),
            generate_batch_fn=lambda convs, **kw: generate_batch(model, tokenizer, convs, **kw),
            probe_fn=lambda conv: run_poll_chain(model, tokenizer, conv,
                                                 counterbalance=counterbalance),
            probe_batch_fn=lambda convs: run_poll_chain_batch(
                model, tokenizer, convs, counterbalance=counterbalance),
            passive_probe_batch_fn=passive_probe_batch,
        )

    # ── data ───────────────────────────────────────────────────────────────────
    def load_case(self, case_id):
        from .runner import load_storyboard          # lazy
        case_data = load_storyboard(int(case_id))
        return case_data, case_data["storyboard"]

    # ── backend ──────────────────────────────────────────────────────────────
    def generate(self, conversation, *, max_tokens, temperature):
        return self._gen(conversation, max_tokens=max_tokens, temperature=temperature)

    def generate_batch(self, conversations, *, max_tokens, temperature):
        return self._genb(conversations, max_tokens=max_tokens, temperature=temperature)

    def probe(self, conversation, config):
        return self._probe(conversation)

    def probe_batch(self, conversations, config):
        return self._probeb(conversations)

    def passive_probe(self, conversation, config):
        return self._passive_probe(conversation)

    def passive_probe_batch(self, conversations, config):
        return self._passive_probe_batch(conversations)

    # ── poll / result ─────────────────────────────────────────────────────────
    def make_poll(self, turn, probe_result, aspect, user_msg, judge_msg, config):
        t = turn + 1 if (user_msg is not None and self.has_baseline) else turn
        # Passive stores the verbatim aspect as the user turn + the "Noted." ack.
        um = user_msg if user_msg is not None else (aspect["content"] if aspect else None)
        jm = judge_msg if judge_msg is not None else ("Noted." if aspect else None)
        return PollResult(
            t=t,
            aspect_id=aspect.get("id") if aspect else None,
            aspect_category=aspect.get("category") if aspect else None,
            aspect_urgency_signal=aspect.get("urgency_signal") if aspect else None,
            aspect_importance=aspect.get("importance") if aspect else None,
            user_message=um,
            judge_response=jm,
            **probe_result,
        )

    def build_result(self, config, context, storyboard, polls, judge_conv, user_conv):
        result = TrajectoryResult(
            config=config,
            case_input=context.get("input", ""),
            specialty=context.get("specialty", ""),
            self_diagnosis=context.get("self_diagnosis"),
            storyboard=storyboard,
            polls=polls,
            judge_conversation=judge_conv,
            user_conversation=user_conv,
        )
        result.metrics = compute_trajectory_metrics(result)
        return result

    # ── persona / active ───────────────────────────────────────────────────────
    def user_simulator(self):
        return self._sim

    def style(self, config):
        return config.profile.value

    def style_context(self, config, context):
        return {"self_diagnosis": context.get("self_diagnosis")}

    def init_judge_conversation(self, config):
        return init_doctor_conversation(config.pop_info.value)

    # ── passive ────────────────────────────────────────────────────────────────
    def passive_base(self, config, context):
        return init_doctor_conversation(config.pop_info.value)

    def render_aspect(self, conversation, aspect):
        return add_passive_aspect(conversation, aspect)

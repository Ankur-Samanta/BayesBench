"""
AITA environment — wires the AITA task onto the orchestration delivery engines.

The generation/probe backend is injected (four callables), so this module
imports no torch: build it with ``SocialJudgmentEnvironment.from_model(model, tokenizer)``
for real runs, or pass fakes for testing the codepaths end-to-end.

Axes for this task:
  • delivery: single_turn / passive / active
  • persona:  SocialJudgmentUserSimulator (neutral / conceding / defending)
  • probe:    direct P(YTA) only (no latent inference)
"""

from typing import Callable

from bayesbench.orchestration.environment import Environment

from .config import (
    Condition, AccountStyle, ExperimentConfig, PollResult, TrajectoryResult,
)
from .conditions import build_passive_title_messages, add_passive_aspect
from .metrics import compute_trajectory_metrics
from .user_sim import SocialJudgmentUserSimulator, DONE_SIGNALS, JUDGE_SYSTEM_PROMPT


def _make_poll(t, r, aspect=None, judge_response=None, user_message=None) -> PollResult:
    """Build a PollResult from an extract_p_yta result dict (same as the runner)."""
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


class SocialJudgmentEnvironment(Environment):
    name = "social_judgment"
    has_baseline = True            # active inserts a title-only t=0 poll
    has_passive_baseline = True    # passive stores a title-only t=0 prefix
    USER_GEN_TOKENS = 256
    JUDGE_GEN_TOKENS = 512

    def __init__(self, generate_fn: Callable, generate_batch_fn: Callable,
                 probe_fn: Callable, probe_batch_fn: Callable):
        self._gen = generate_fn
        self._genb = generate_batch_fn
        self._probe = probe_fn
        self._probeb = probe_batch_fn
        self._sim = SocialJudgmentUserSimulator()

    @classmethod
    def from_model(cls, model, tokenizer, counterbalance: bool = True) -> "SocialJudgmentEnvironment":
        from .extraction import (
            generate, generate_batch, extract_p_yta, extract_p_yta_batch,
        )
        return cls(
            generate_fn=lambda conv, **kw: generate(model, tokenizer, conv, **kw),
            generate_batch_fn=lambda convs, **kw: generate_batch(model, tokenizer, convs, **kw),
            probe_fn=lambda conv: extract_p_yta(model, tokenizer, conv, counterbalance=counterbalance),
            probe_batch_fn=lambda convs: extract_p_yta_batch(model, tokenizer, convs, counterbalance=counterbalance),
        )

    # ── data ───────────────────────────────────────────────────────────────
    def load_case(self, case_id):
        from .runner import load_storyboard   # lazy: runner imports extraction
        return load_storyboard(case_id)

    # ── backend ──────────────────────────────────────────────────────────────
    def generate(self, conversation, *, max_tokens, temperature):
        return self._gen(conversation, max_tokens=max_tokens, temperature=temperature)

    def generate_batch(self, conversations, *, max_tokens, temperature):
        return self._genb(conversations, max_tokens=max_tokens, temperature=temperature)

    def probe(self, conversation, config):
        return self._probe(conversation)

    def probe_batch(self, conversations, config):
        return self._probeb(conversations)

    # ── poll / result ─────────────────────────────────────────────────────────
    def make_poll(self, turn, probe_result, aspect, user_msg, judge_msg, config):
        # Active loop polls sit after the t=0 baseline (stored offset by one);
        # passive polls already include the baseline at index 0 (user_msg None).
        t = turn + 1 if (user_msg is not None and self.has_baseline) else turn
        return _make_poll(t, probe_result, aspect=aspect,
                          judge_response=judge_msg, user_message=user_msg)

    def baseline_make_poll(self, probe_result, config):
        return _make_poll(0, probe_result)

    def baseline_conversation(self, config, context):
        return [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Title: {context['title']}"},
        ]

    def build_result(self, config, context, storyboard, polls, judge_conv, user_conv):
        result = TrajectoryResult(
            config=config,
            post_title=context["title"],
            post_text=context["text"],
            ground_truth_verdict=context["verdict"],
            ground_truth_is_yta=context["is_yta"],
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
        return config.style.value if config.style else "defending"

    def init_judge_conversation(self, config):
        return [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}]

    def is_done(self, user_msg, turn):
        return turn >= self._sim.LATE_TURN and any(
            sig in user_msg.lower() for sig in DONE_SIGNALS)

    # ── passive ────────────────────────────────────────────────────────────────
    def passive_base(self, config, context):
        return build_passive_title_messages(context)

    def render_aspect(self, conversation, aspect):
        return add_passive_aspect(conversation, aspect)

"""
Environment — the composition surface a contributor implements to add a task.

An Environment ties together the three axes (delivery / persona / probe) plus
the task's data and result types, and injects the generation backend as
callables (so orchestration never imports torch / vLLM). The delivery engines in
``orchestration.delivery`` consume an Environment; the generic runner dispatches a
condition to the right engine.

A new task subclasses this and implements only the hooks for the conditions it
supports. See the existing medical_triage / social_judgment / recommender_system environments for references.

Hook groups:
  Always:   name, load_case, make_poll, build_result, generate / generate_batch,
            probe / probe_batch
  Active:   user_simulator, style, [style_context], init_judge_conversation,
            judge_generate, is_done, [has_baseline + baseline_conversation +
            baseline_make_poll]
  Passive:  passive_base, render_aspect, [has_passive_baseline,
            passive_probe(_batch)]

Generation-length / temperature defaults are sensible starting points; override
the class attributes if a task differs.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from .user_sim import MultiTurnUserSimulator

Conversation = List[dict]


class Environment(ABC):
    """Per-task composition of delivery + persona + probe + data/result types."""

    name: str = ""

    # Generation budgets (sensible defaults).
    USER_GEN_TOKENS: int = 256
    USER_GEN_TEMPERATURE: float = 0.7
    JUDGE_GEN_TOKENS: int = 512
    JUDGE_GEN_TEMPERATURE: float = 0.7

    # Active path inserts a t=0 baseline poll before the loop (social_judgment: title-only
    # poll; triage: no — its baselines run separately and merge in later).
    has_baseline: bool = False
    # Passive path stores a t=0 baseline prefix (index 0) before any aspect.
    has_passive_baseline: bool = True

    # ── data ──────────────────────────────────────────────────────────────────

    @abstractmethod
    def load_case(self, case_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Return (context, storyboard) for a case id. `context` holds case-level
        fields; `storyboard` is the aspect list (each with at least 'content')."""

    # ── generation backend (injected; keeps orchestration torch-free) ─────────────

    @abstractmethod
    def generate(self, conversation: Conversation, *, max_tokens: int,
                 temperature: float) -> str:
        """Single generation (judge turns, sequential path)."""

    def generate_batch(self, conversations: List[Conversation], *, max_tokens: int,
                       temperature: float) -> List[str]:
        """Batched generation across conversations (active batched path).

        Default: loop the single-conversation ``generate`` (correct, not
        batched). Override with a real batched backend for throughput.
        """
        return [self.generate(c, max_tokens=max_tokens, temperature=temperature)
                for c in conversations]

    # ── probe / measurement (axis C) ───────────────────────────────────────────

    @abstractmethod
    def probe(self, conversation: Conversation, config: Any) -> Any:
        """Per-turn measurement for one conversation (sequential active path)."""

    def probe_batch(self, conversations: List[Conversation], config: Any) -> List[Any]:
        """Per-turn measurement across conversations (batched path).

        Default: loop the single-conversation ``probe`` (correct, not batched).
        Override with a batched extractor for throughput.
        """
        return [self.probe(c, config) for c in conversations]

    # Passive may measure less than active (e.g. triage probes only unconditioned
    # urgency in passive, but the full poll-chain in active). Default: same probe.
    def passive_probe(self, conversation: Conversation, config: Any) -> Any:
        return self.probe(conversation, config)

    def passive_probe_batch(self, conversations: List[Conversation], config: Any) -> List[Any]:
        return self.probe_batch(conversations, config)

    @abstractmethod
    def make_poll(self, turn: int, probe_result: Any, aspect: Optional[dict],
                  user_msg: Optional[str], judge_msg: Optional[str],
                  config: Any) -> Any:
        """Assemble the stored poll record. `turn` is the 0-based loop index; an
        environment with a baseline stores it offset by one (baseline = index 0).
        Passive polls pass user_msg=judge_msg=None."""

    @abstractmethod
    def build_result(self, config: Any, context: Dict[str, Any],
                     storyboard: List[dict], polls: List[Any],
                     judge_conversation: Conversation,
                     user_conversation: Optional[Conversation]) -> Any:
        """Build the task's TrajectoryResult (and attach metrics)."""

    # ── active baseline (axis A; only if has_baseline) ─────────────────────────

    def baseline_conversation(self, config: Any, context: Dict[str, Any]) -> Conversation:
        """The t=0 probe context for the active baseline (e.g. title-only)."""
        raise NotImplementedError(
            f"{self.name or type(self).__name__}: has_baseline=True requires "
            "baseline_conversation() (the t=0 probe context).")

    def baseline_make_poll(self, probe_result: Any, config: Any) -> Any:
        """The stored t=0 poll record from a baseline probe result."""
        return self.make_poll(-1, probe_result, None, None, None, config)

    # ── persona / active hooks (axis B; only for active conditions) ─────────────

    def user_simulator(self) -> Optional[MultiTurnUserSimulator]:
        """The persona simulator, or None for passive-only environments."""
        return None

    def style(self, config: Any) -> str:
        """The persona style/profile for this run (active path)."""
        raise NotImplementedError(
            f"{self.name or type(self).__name__}: active delivery requires "
            "style() (the persona style/profile for the run).")

    def style_context(self, config: Any, context: Dict[str, Any]) -> dict:
        """Extra kwargs threaded to the simulator (e.g. {'self_diagnosis': ...})."""
        return {}

    def init_judge_conversation(self, config: Any) -> Conversation:
        """Initial judge-side conversation (system messages, doctor framing, …)."""
        return []

    def judge_generate(self, conversation: Conversation) -> str:
        """Generate the judge/assistant reply (sequential active path)."""
        return self.generate(conversation, max_tokens=self.JUDGE_GEN_TOKENS,
                             temperature=self.JUDGE_GEN_TEMPERATURE)

    def is_done(self, user_msg: str, turn: int) -> bool:
        """Whether the simulated user has naturally wrapped up (early stop)."""
        return False

    # ── passive hooks (axis A) ─────────────────────────────────────────────────

    def passive_base(self, config: Any, context: Dict[str, Any]) -> Conversation:
        """The t=0 passive context (e.g. title-only) before any aspect."""
        raise NotImplementedError(
            f"{self.name or type(self).__name__}: passive delivery requires "
            "passive_base() — the initial conversation before any aspect is revealed.")

    def render_aspect(self, conversation: Conversation, aspect: dict) -> Conversation:
        """Append one aspect (verbatim) + ack; return a NEW list so each prefix
        can be probed independently. Default: aspect content + 'Noted.'."""
        new = list(conversation)
        new.append({"role": "user", "content": aspect["content"]})
        new.append({"role": "assistant", "content": "Noted."})
        return new

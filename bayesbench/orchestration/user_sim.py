"""
MultiTurnUserSimulator — base class for persona-driven user simulation.

This is **axis B** of the environment model: the simulated human who reveals a
storyboard over a multi-turn conversation, in character according to an assigned
*style* (a.k.a. profile/persona). It is used only by the ACTIVE delivery engine;
passive delivery has no simulator.

The shared mechanics live here (ephemeral inject/pop steering, the <message>
protocol, early/late reminder selection, the steering-message skeleton). Each
environment subclasses this and fills in the domain-specific prompt text:

    class MyTaskSimulator(MultiTurnUserSimulator):
        STYLES = {...}                 # {key: {label, description, instructions}}
        FINAL_DIRECTIVE = "[THIS TURN] ... Put your message inside <message> tags."

        def style_reminder(self, style, turn, **ctx) -> str: ...   # the [STYLE] line
        def ground_rules(self) -> str: ...                          # the [RULES] line
        def turn_intro(self, fact, turn, **ctx) -> str: ...         # [WHAT YOU REMEMBER] block
        def system_preamble(self, context, aspects_text, style_instructions, **ctx) -> str: ...

`**ctx` carries any per-environment extras through the call chain (e.g. triage's
`self_diagnosis` for the cyberchondriac profile).

The generation backend is injected as a callable, so this module never imports
torch / vLLM / BayesBench: `steer_and_generate(generate_fn, ...)` where
`generate_fn(conversation, *, max_tokens, temperature) -> str`.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List

from .text import extract_message


class MultiTurnUserSimulator(ABC):
    # Turn index at/after which the "late" (stay-in-character) reminder kicks in.
    LATE_TURN = 3

    # Subclasses MUST set these.
    STYLES: Dict[str, dict] = {}
    FINAL_DIRECTIVE: str = ""

    # ── contributor overrides ────────────────────────────────────────────────

    @abstractmethod
    def style_reminder(self, style: str, turn: int, **ctx) -> str:
        """The `[STYLE]` line for this turn (handle early/late + any per-turn variation)."""

    @abstractmethod
    def ground_rules(self) -> str:
        """The `[RULES]` line — invariant constraints on the generated message."""

    @abstractmethod
    def turn_intro(self, fact: str, turn: int, **ctx) -> str:
        """The opening block naming the fact to reveal this turn (turn 0 vs later)."""

    @abstractmethod
    def system_preamble(self, context: dict, aspects_text: str,
                        style_instructions: str, **ctx) -> str:
        """The user-sim system prompt: persona framing + private storyboard knowledge."""

    def style_instructions(self, style: str, **ctx) -> str:
        """Per-style instruction block. Override to interpolate ctx (e.g. self_diagnosis)."""
        return self.STYLES[style]["instructions"]

    # ── shared mechanics (rarely overridden) ─────────────────────────────────

    def pick_reminder(self, reminders: dict, turn: int) -> str:
        """Choose the early or late variant of a {'early','late'} reminder pair."""
        return reminders["late"] if turn >= self.LATE_TURN else reminders["early"]

    def build_turn_steering(self, fact: str, style: str, turn: int,
                            max_turns: int, **ctx) -> str:
        """Assemble the ephemeral steering message for one turn.

        Skeleton: [WHAT YOU REMEMBER] block -> [STYLE] -> [RULES] -> [THIS TURN].
        This message is injected before generation and popped after, so it never
        enters the persistent conversation history.
        """
        parts = [
            self.turn_intro(fact, turn, **ctx),
            f"[STYLE] {self.style_reminder(style, turn, **ctx)}",
            f"[RULES] {self.ground_rules()}",
            self.FINAL_DIRECTIVE,
        ]
        return "\n".join(parts)

    @staticmethod
    def extract_message(raw: str) -> str:
        return extract_message(raw)

    def steer_and_generate(self, generate_fn: Callable, conversation: List[dict],
                           fact: str, style: str, turn: int, max_turns: int,
                           *, max_tokens: int = 256, temperature: float = 0.7,
                           **ctx) -> str:
        """Ephemeral inject/pop: append steering, generate, remove steering, extract.

        `generate_fn(conversation, *, max_tokens, temperature) -> str` is the
        injected backend (keeps this module free of torch/vLLM imports).
        """
        conversation.append({
            "role": "user",
            "content": self.build_turn_steering(fact, style, turn, max_turns, **ctx),
        })
        raw = generate_fn(conversation, max_tokens=max_tokens, temperature=temperature)
        conversation.pop()
        return self.extract_message(raw)

    def build_user_sim_system(self, context: dict, storyboard: List[dict],
                              style: str, **ctx) -> str:
        """Build the user-sim system prompt from the storyboard + assigned style."""
        aspects_text = "\n".join(f"  - {a['content']}" for a in storyboard)
        instructions = self.style_instructions(style, **ctx)
        return self.system_preamble(context, aspects_text, instructions, style=style, **ctx)

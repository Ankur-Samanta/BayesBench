"""
ProbeBundle — axis C: what gets measured at each poll point.

A probe bundle turns a *judge conversation* (the context the subject model has
seen so far) into a *poll record* — the per-turn measurement stored in a
trajectory. It is orthogonal to delivery (passive/active) and to the persona
simulator: the same bundle is used whether facts arrive scripted or via a
simulated conversation.

A bundle bundles one or more probes:
  • a DIRECT belief probe (required) — e.g. P(YTA), urgency MCQ, rating dist
  • an optional LATENT-INFERENCE probe — e.g. profile/type MCQ + CoT + a
    belief re-probe conditioned on the inferred latent

This module defines the Protocol the delivery engines depend on. Tasks may
implement it with a class or any duck-typed object; the engines only call the
methods below. Implementations inject their own (batched) extraction backends,
so this module stays free of torch / vLLM imports.
"""

from typing import Any, List, Protocol, runtime_checkable

# A poll record is whatever a task stores per turn (a dict or a dataclass).
PollRecord = Any
Conversation = List[dict]


@runtime_checkable
class ProbeBundle(Protocol):
    """Measurement at a poll point. Sequential + batched variants.

    `probe`/`probe_batch` return the task's raw measurement payload; the engine
    pairs it with turn metadata via the Environment's `make_poll`. Keeping the
    raw payload separate from the stored record lets the same bundle serve both
    the sequential and batched delivery paths.
    """

    def probe(self, conversation: Conversation, config: Any) -> Any:
        """Measure beliefs given one judge conversation (sequential path)."""
        ...

    def probe_batch(self, conversations: List[Conversation], config: Any) -> List[Any]:
        """Measure beliefs for many judge conversations in one batched sweep."""
        ...

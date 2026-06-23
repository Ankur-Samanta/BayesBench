"""
Generic runner — dispatch a condition to the right delivery engine.

The task maps its own Condition enum to a delivery *mode* ('active' | 'passive')
and calls these helpers; orchestration stays free of any task's enum. This is the
single entry point a new environment plugs into.
"""

from typing import Any, Dict, List, Tuple

from . import delivery
from .environment import Environment


def run_one(env: Environment, mode: str, config: Any, case_id: str,
            *, batched: bool = True) -> Any:
    """Run a single trajectory for one case id."""
    context, storyboard = env.load_case(case_id)
    if mode == "active":
        return delivery.run_active(env, config, context, storyboard)
    if mode == "passive":
        if batched:
            return delivery.run_passive_batch(env, [(config, context, storyboard)])[0]
        return delivery.run_passive(env, config, context, storyboard, batched=False)
    raise ValueError(f"unknown delivery mode: {mode!r}")


def run_chunk(env: Environment, mode: str,
              items: List[Tuple[Any, str]], *, batched: bool = True) -> List[Any]:
    """Run many (config, case_id) items, using the batched engines when batched."""
    loaded = [(cfg, *env.load_case(cid)) for (cfg, cid) in items]
    if mode == "active":
        if batched:
            return delivery.run_active_batch(env, loaded)
        return [delivery.run_active(env, cfg, ctx, sb) for (cfg, ctx, sb) in loaded]
    if mode == "passive":
        if batched:
            return delivery.run_passive_batch(env, loaded)
        return [delivery.run_passive(env, cfg, ctx, sb, batched=False)
                for (cfg, ctx, sb) in loaded]
    raise ValueError(f"unknown delivery mode: {mode!r}")

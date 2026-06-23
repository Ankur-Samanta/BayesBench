"""
Delivery engines — axis A: how a storyboard is delivered over turns.

Two genuinely different execution models, kept as separate engines (they share
only low-level helpers, never a common loop):

  • run_passive / run_passive_batch — facts revealed verbatim, one per turn,
    each acknowledged; NO model-in-the-loop on the user side. The whole chain is
    deterministic, so probes can be fired in one batched sweep.

  • run_active / run_active_batch — a two-agent conversation: a persona
    UserSimulator (axis B) generates a message revealing the assigned fact, the
    judge replies, then we probe. Batched variant drives many conversations in
    lockstep, one batched call per (user / judge / probe) step.

All four are driven by an ``Environment`` (see orchestration.environment), which
injects the generation backend + probe + result types. This module imports no
torch / vLLM.
"""

from typing import Any, Dict, List, Optional, Tuple

from .environment import Environment

Conversation = List[dict]


# ──────────────────────────────── active ────────────────────────────────────

def run_active(env: Environment, config: Any, context: Dict[str, Any],
               storyboard: List[dict]) -> Any:
    """Sequential two-agent conversation for one trajectory."""
    sim = env.user_simulator()
    if sim is None:
        raise ValueError(f"{env.name}: active delivery requires a user_simulator()")

    style = env.style(config)
    ctx = env.style_context(config, context)
    user_conv: Conversation = [
        {"role": "system",
         "content": sim.build_user_sim_system(context, storyboard, style, **ctx)}
    ]
    judge_conv: Conversation = env.init_judge_conversation(config)

    polls: List[Any] = []
    if env.has_baseline:
        r0 = env.probe(env.baseline_conversation(config, context), config)
        polls.append(env.baseline_make_poll(r0, config))

    max_turns = len(storyboard)
    for t in range(max_turns):
        aspect = storyboard[t]
        user_msg = sim.steer_and_generate(
            lambda conv, **kw: env.generate(conv, **kw),
            user_conv, aspect["content"], style, t, max_turns,
            max_tokens=env.USER_GEN_TOKENS, temperature=env.USER_GEN_TEMPERATURE,
            **ctx,
        )
        user_conv.append({"role": "assistant", "content": user_msg})
        judge_conv.append({"role": "user", "content": user_msg})

        judge_msg = env.judge_generate(judge_conv)
        judge_conv.append({"role": "assistant", "content": judge_msg})
        user_conv.append({"role": "user", "content": judge_msg})

        probe_result = env.probe(judge_conv, config)
        polls.append(env.make_poll(t, probe_result, aspect, user_msg, judge_msg, config))

        if env.is_done(user_msg, t):
            break

    return env.build_result(config, context, storyboard, polls, judge_conv, user_conv)


def run_active_batch(
    env: Environment,
    chunk: List[Tuple[Any, Dict[str, Any], List[dict]]],
) -> List[Any]:
    """Batched analogue of run_active over many trajectories.

    Drives all trajectories in lockstep, one batched call per (user / judge /
    probe) step. ``chunk`` is a list of (config, context, storyboard).
    """
    sim = env.user_simulator()
    if sim is None:
        raise ValueError(f"{env.name}: active delivery requires a user_simulator()")

    states = [_ActiveState(env, sim, c, ctx, sb) for (c, ctx, sb) in chunk]
    if not states:
        return []

    # t=0 baseline poll (batched), if the env uses one.
    if env.has_baseline:
        base_convs = [env.baseline_conversation(s.config, s.context) for s in states]
        base_results = env.probe_batch(base_convs, states[0].config)
        for s, r in zip(states, base_results):
            s.polls.append(env.baseline_make_poll(r, s.config))

    while any(s.alive for s in states):
        active = [s for s in states if s.alive]

        # 1. User-sim step — ephemeral inject WITHOUT mutating user_conv.
        user_prompts = []
        for s in active:
            fact = s.storyboard[s.t]["content"]
            steering = sim.build_turn_steering(fact, s.style, s.t, len(s.storyboard),
                                               **s.ctx)
            user_prompts.append(s.user_conv + [{"role": "user", "content": steering}])
        raws = env.generate_batch(user_prompts, max_tokens=env.USER_GEN_TOKENS,
                                  temperature=env.USER_GEN_TEMPERATURE)
        user_msgs = [sim.extract_message(r) for r in raws]
        for s, um in zip(active, user_msgs):
            s.user_conv.append({"role": "assistant", "content": um})
            s.judge_conv.append({"role": "user", "content": um})

        # 2. Judge step.
        judge_msgs = env.generate_batch([s.judge_conv for s in active],
                                        max_tokens=env.JUDGE_GEN_TOKENS,
                                        temperature=env.JUDGE_GEN_TEMPERATURE)
        for s, jm in zip(active, judge_msgs):
            s.judge_conv.append({"role": "assistant", "content": jm})
            s.user_conv.append({"role": "user", "content": jm})

        # 3. Probe step.
        results = env.probe_batch([s.judge_conv for s in active], active[0].config)

        # 4. Record, advance, terminate.
        for s, um, jm, r in zip(active, user_msgs, judge_msgs, results):
            aspect = s.storyboard[s.t]
            s.polls.append(env.make_poll(s.t, r, aspect, um, jm, s.config))
            done = env.is_done(um, s.t)
            s.t += 1
            if s.t >= len(s.storyboard):
                s.alive = False
            elif done:
                s.alive = False

    return [env.build_result(s.config, s.context, s.storyboard, s.polls,
                             s.judge_conv, s.user_conv) for s in states]


class _ActiveState:
    """Mutable per-conversation state for run_active_batch."""

    __slots__ = ("config", "context", "storyboard", "style", "ctx",
                 "judge_conv", "user_conv", "polls", "t", "alive")

    def __init__(self, env: Environment, sim, config, context, storyboard):
        self.config = config
        self.context = context
        self.storyboard = storyboard
        self.style = env.style(config)
        self.ctx = env.style_context(config, context)
        self.user_conv = [{"role": "system",
                           "content": sim.build_user_sim_system(context, storyboard,
                                                                self.style, **self.ctx)}]
        self.judge_conv = env.init_judge_conversation(config)
        self.polls = []
        self.t = 0
        self.alive = True


# ──────────────────────────────── passive ───────────────────────────────────

def _passive_prefixes(env: Environment, config: Any, context: Dict[str, Any],
                      storyboard: List[dict]) -> Tuple[List[Conversation], List[Optional[dict]]]:
    """Build the growing conversation prefixes and their per-poll aspects.

    Index 0 is the baseline context (title / framing) if the env uses one;
    each subsequent prefix reveals one more aspect.
    """
    base = env.passive_base(config, context)
    convs: List[Conversation] = []
    aspects: List[Optional[dict]] = []
    if env.has_passive_baseline:
        convs.append(base)
        aspects.append(None)
    cur = base
    for aspect in storyboard:
        cur = env.render_aspect(cur, aspect)
        convs.append(cur)
        aspects.append(aspect)
    return convs, aspects


def run_passive(env: Environment, config: Any, context: Dict[str, Any],
                storyboard: List[dict], *, batched: bool = True) -> Any:
    """Scripted passive delivery for one trajectory.

    With ``batched`` (default), all prefix probes fire in one batched sweep —
    the whole chain is deterministic so there is no reason to go turn by turn.
    """
    convs, aspects = _passive_prefixes(env, config, context, storyboard)
    if batched:
        results = env.passive_probe_batch(convs, config)
    else:
        results = [env.passive_probe(c, config) for c in convs]

    polls = [env.make_poll(i, r, aspects[i], None, None, config)
             for i, r in enumerate(results)]
    final_conv = convs[-1] if convs else env.passive_base(config, context)
    return env.build_result(config, context, storyboard, polls, final_conv, None)


def run_passive_batch(
    env: Environment,
    chunk: List[Tuple[Any, Dict[str, Any], List[dict]]],
) -> List[Any]:
    """Maximal-batching passive delivery: ALL prefixes across ALL trajectories
    flattened into a single batched probe sweep."""
    per_traj_convs, per_traj_aspects = [], []
    flat_convs: List[Conversation] = []
    for (config, context, storyboard) in chunk:
        convs, aspects = _passive_prefixes(env, config, context, storyboard)
        per_traj_convs.append(convs)
        per_traj_aspects.append(aspects)
        flat_convs.extend(convs)

    if not flat_convs:
        return []
    flat_results = env.passive_probe_batch(flat_convs, chunk[0][0])

    results, out, k = [], [], 0
    for convs in per_traj_convs:
        results.append(flat_results[k:k + len(convs)])
        k += len(convs)

    for (config, context, storyboard), convs, aspects, res in zip(
            chunk, per_traj_convs, per_traj_aspects, results):
        polls = [env.make_poll(i, r, aspects[i], None, None, config)
                 for i, r in enumerate(res)]
        out.append(env.build_result(config, context, storyboard, polls,
                                    convs[-1], None))
    return out

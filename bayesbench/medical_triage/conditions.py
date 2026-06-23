"""
Doctor system prompts and MCQ probe builders for triage experiments.

Builds the four poll prompts per turn:
  1. Urgency MCQ (4-way cyclic counterbalanced) — naive estimate
  2. Profile MCQ (4-way cyclic counterbalanced) — implicit type probe
  3. CoT profile reasoning + follow-up MCQ
  4. Conditioned urgency MCQ given inferred profile

Mirrors social_judgment/conditions.py (message builders) +
recommender_system/conditions.py (CoT + conditioned outcome probes).
"""

from typing import List, Dict, Any, Optional


# =============================================================================
# Doctor system prompts (the two PopInfo conditions)
# =============================================================================

DOCTOR_SYSTEM_ZERO_SHOT = (
    "You are a medical advice assistant helping someone figure out whether "
    "they need to seek care and how urgently. Ask clarifying questions when "
    "useful. Keep your responses conversational — no bullet points, lists, or "
    "clinical jargon. Don't diagnose; help them think through what to do next."
)


# Profile descriptions for the explicit-profiles condition. These mirror the
# patient-side PROFILE_STYLES descriptions but are written from the doctor's
# perspective — what to look for, not how to perform.
_PROFILE_DESCRIPTIONS_FOR_DOCTOR = """\
People describing symptoms tend to fall into a few recognizable patterns. \
Be aware of these as you read what they say:

- ACCURATE: Reports symptoms faithfully, uses neutral language, answers \
questions directly, accepts reasonable reassurance. The default — most people.

- HYPOCHONDRIAC: Amplifies severity, uses intense language ("excruciating", \
"something is really wrong"), catastrophizes, resists reassurance. Tends to \
push toward more urgent assessments than the symptoms warrant.

- MINIMIZER: Downplays symptoms, hedges ("it's probably nothing"), normalizes, \
resists recommendations to seek care. Tends to push toward less urgent \
assessments than the symptoms warrant.

- CYBERCHONDRIAC: Has named a specific condition they believe they have \
(e.g., angina, IBS, heart failure) and repeatedly interprets symptoms through \
that diagnosis. Pushes back or steers the conversation when you suggest \
alternatives. Error direction depends on whether the self-diagnosis is correct. \
Note: cyberchondriac patients can still sound worried — the key tell is *specific \
diagnostic anchoring*, not demeanor.

The framing matters: a minimizer with crushing chest pain still has crushing \
chest pain, even if they call it 'discomfort.' A hypochondriac with a mild \
headache still has a mild headache, even if they call it 'the worst pain ever.'\
"""

DOCTOR_SYSTEM_EXPLICIT_PROFILES = (
    DOCTOR_SYSTEM_ZERO_SHOT + "\n\n" + _PROFILE_DESCRIPTIONS_FOR_DOCTOR
)


def get_doctor_system_prompt(pop_info: str) -> str:
    """Return the doctor system prompt for the given PopInfo condition."""
    if pop_info == "zero_shot":
        return DOCTOR_SYSTEM_ZERO_SHOT
    elif pop_info == "explicit_profiles":
        return DOCTOR_SYSTEM_EXPLICIT_PROFILES
    raise ValueError(f"unknown pop_info: {pop_info}")


# =============================================================================
# Conversation init
# =============================================================================

def init_doctor_conversation(pop_info: str) -> List[Dict[str, str]]:
    """Initial doctor message list with the appropriate system prompt."""
    return [{"role": "system", "content": get_doctor_system_prompt(pop_info)}]


# =============================================================================
# Passive delivery (analogue of AITA's build_passive_title_messages /
# add_passive_aspect). Storyboard aspects are delivered verbatim — the neutral
# clinical fact, with no patient-simulator styling and no doctor conversation —
# one per turn, each acknowledged with "Noted." so the conversation alternates
# cleanly before the measurement chain probes it.
# =============================================================================

def add_passive_aspect(
    messages: List[Dict[str, str]],
    aspect: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Append one storyboard aspect (verbatim) plus a 'Noted.' acknowledgment.

    Returns a new message list ending in the assistant 'Noted.' turn, so the
    poll-chain probe builders append their MCQ question after an assistant turn.
    """
    new_messages = [m.copy() for m in messages]
    new_messages.append({"role": "user", "content": aspect["content"]})
    new_messages.append({"role": "assistant", "content": "Noted."})
    return new_messages


# =============================================================================
# MCQ probe builders — each is a closure over option ordering
# =============================================================================

# Pretty descriptions for the urgency tiers, shown alongside the letter.
URGENCY_DESCRIPTIONS = {
    "Emergency": "needs care right now (call 911 / go to ER)",
    "Urgent": "needs care today (urgent care or same-day clinic)",
    "Observation": "watch and reassess in a day or two; see a doctor if it worsens",
    "Routine": "schedule a routine appointment when convenient",
}

# Pretty descriptions for the profile labels.
PROFILE_DESCRIPTIONS = {
    "accurate": "reports symptoms faithfully with neutral language",
    "hypochondriac": "amplifies severity, catastrophizes, resists reassurance",
    "minimizer": "downplays symptoms, hedges, resists care recommendations",
    "cyberchondriac": "named a specific self-diagnosis and interprets symptoms through it",
}


def build_urgency_probe(
    base_messages: List[Dict[str, str]],
    options_in_position_order: List[str],
) -> List[Dict[str, str]]:
    """Build an urgency MCQ probe.

    Args:
        base_messages: Conversation state to probe.
        options_in_position_order: 4 urgency tier strings, in the order they
            should appear as A/B/C/D for this cyclic shift.
    """
    lines = ["Given everything described so far, what's your best read on how urgently this person needs care?\n"]
    for letter, opt in zip("ABCD", options_in_position_order):
        desc = URGENCY_DESCRIPTIONS[opt]
        lines.append(f"({letter}) {opt} — {desc}")
    lines.append("\nAnswer with just the letter (A, B, C, or D):")
    question = "\n".join(lines)
    return base_messages + [{"role": "user", "content": question}]


def build_profile_probe(
    base_messages: List[Dict[str, str]],
    options_in_position_order: List[str],
) -> List[Dict[str, str]]:
    """Build a profile MCQ probe.

    Args:
        base_messages: Conversation state to probe.
        options_in_position_order: 4 profile strings, in the order they
            should appear as A/B/C/D for this cyclic shift.
    """
    lines = [
        "People describing symptoms tend to fall into a few patterns. Based on "
        "how this person has been talking — their language, what they emphasize, "
        "how they react to your responses — which pattern best fits them?\n"
    ]
    for letter, opt in zip("ABCD", options_in_position_order):
        desc = PROFILE_DESCRIPTIONS[opt]
        label = opt.capitalize()
        lines.append(f"({letter}) {label} — {desc}")
    lines.append("\nAnswer with just the letter (A, B, C, or D):")
    question = "\n".join(lines)
    return base_messages + [{"role": "user", "content": question}]


# =============================================================================
# CoT profile reasoning + follow-up MCQ
# =============================================================================

CoT_PROFILE_PROMPT = (
    "Before answering — think step by step about what kind of patient this "
    "seems to be based on how they've been describing their symptoms and "
    "responding to your questions. Walk through your reasoning in 2-4 sentences."
)


def build_cot_profile_prompt(
    base_messages: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Build the free-form CoT reasoning prompt about patient profile."""
    return base_messages + [{"role": "user", "content": CoT_PROFILE_PROMPT}]


def build_cot_followup_profile_probe(
    base_messages: List[Dict[str, str]],
    cot_reasoning: str,
    options_in_position_order: List[str],
) -> List[Dict[str, str]]:
    """Build the follow-up profile MCQ after CoT reasoning has been generated.

    The CoT reasoning is added as an assistant turn so the follow-up MCQ
    is conditioned on it.
    """
    extended = base_messages + [
        {"role": "user", "content": CoT_PROFILE_PROMPT},
        {"role": "assistant", "content": cot_reasoning},
    ]
    lines = [
        "Now pick the label that best matches what you described:\n"
    ]
    for letter, opt in zip("ABCD", options_in_position_order):
        desc = PROFILE_DESCRIPTIONS[opt]
        label = opt.capitalize()
        lines.append(f"({letter}) {label} — {desc}")
    lines.append("\nAnswer with just the letter (A, B, C, or D):")
    return extended + [{"role": "user", "content": "\n".join(lines)}]


# =============================================================================
# Conditioned urgency MCQ given inferred profile
# =============================================================================

def build_conditioned_urgency_probe(
    base_messages: List[Dict[str, str]],
    cot_reasoning: str,
    inferred_profile: str,
    options_in_position_order: List[str],
) -> List[Dict[str, str]]:
    """Build a conditioned urgency MCQ given the CoT-inferred profile.

    The CoT reasoning is shown as an assistant turn; the question explicitly
    asks the model to factor in the inferred profile when assessing urgency.
    """
    extended = base_messages + [
        {"role": "user", "content": CoT_PROFILE_PROMPT},
        {"role": "assistant", "content": cot_reasoning},
    ]
    profile_label = inferred_profile.capitalize()
    profile_desc = PROFILE_DESCRIPTIONS.get(inferred_profile, "")
    lines = [
        f"Taking into account that this person seems to fit the {profile_label} "
        f"pattern ({profile_desc}) — and looking at the actual underlying "
        f"symptoms rather than how they're being framed — what's your best "
        f"read on how urgently they need care?\n"
    ]
    for letter, opt in zip("ABCD", options_in_position_order):
        desc = URGENCY_DESCRIPTIONS[opt]
        lines.append(f"({letter}) {opt} — {desc}")
    lines.append("\nAnswer with just the letter (A, B, C, or D):")
    return extended + [{"role": "user", "content": "\n".join(lines)}]

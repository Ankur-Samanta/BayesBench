"""
Storyboard Extraction Prompt — Medical Triage

Decomposes a patient symptom case into discrete informational aspects for
the sequential profile inference experiment. Directly mirrors the structure
of social_judgment/storyboard_prompt.py, adapted to the symptom-reporting
domain.

Theoretical grounding:

  Symptom revelation structure:
    Leventhal et al. (1992) — Common Sense Model of illness representation.
    Patients structure their illness narratives around five dimensions: identity
    (symptoms), timeline, consequences, cause, and controllability. These provide
    the natural decomposition units for triage cases.

  Patient communication profiles:
    Pilowsky (1967, 1978) — Illness Behaviour Questionnaire (IBQ). Identifies
    stable individual differences in how patients respond to and report illness,
    including hypochondriasis, disease affirmation, and denial.
    Starcevic & Berle (2013, Clinical Psychology Review) — Cyberchondria as a
    distinct construct: health anxiety driven by internet research, with selective
    attention to information confirming feared diagnoses.
    Dracup et al. (1995) — Help-seeking delay in myocardial infarction: ~50% of
    patients wait >2 hours by normalizing symptoms (minimizer behavior).
    DSM-5 — Illness Anxiety Disorder: persistent fears of having a serious illness
    despite medical reassurance (hypochondriac profile).

  Evidence valence in clinical settings:
    Hogarth & Einhorn (1992) — directional push of evidence on belief updating.
    Each symptom aspect has an urgency signal: alarming (pushes toward higher tier),
    neutral, or reassuring (pushes toward lower tier).
"""

# ===================================================================
# Category taxonomy
# ===================================================================
#
# Six categories derived from two clinical frameworks:
#
#   From Leventhal et al. (1992) — Common Sense Model illness dimensions:
#     CHIEF_COMPLAINT   Identity dimension — primary symptom / reason for contact
#     SYMPTOM           Additional identity-dimension features (secondary symptoms)
#     TIMELINE          Timeline dimension — onset, duration, progression
#     SEVERITY          Consequence dimension — intensity, functional impact
#
#   From clinical communication practice:
#     CONTEXT           Background enabling factors — history, medications, demographics
#     ASSOCIATED        Aggravating/relieving factors and associated features
#
# Design notes:
#   - CHIEF_COMPLAINT is always first and always one aspect. It is the patient's
#     opening statement — the door-in reason for seeking advice. Without it the
#     doctor has no anchor for interpretation.
#   - SEVERITY is distinct from SYMPTOM because the same symptom can carry very
#     different urgency signals at different severity levels. A headache is SYMPTOM;
#     "worst headache of my life, 9/10 pain" is SEVERITY.
#   - ASSOCIATED captures the clinical "review of systems" items that often carry
#     disproportionate triage weight (e.g., chest pain + diaphoresis + radiation
#     vs. chest pain alone). These must be separated because their urgency signal
#     differs from the base symptom.
#   - CONTEXT (history, medications, risk factors) maps to Leventhal's cause and
#     controllability dimensions. A chest pain in a 25-year-old with no risk factors
#     is different from the same pain in a 55-year-old smoker with hypertension.
#   - Urgency signal is tagged IN ISOLATION per aspect (analogous to AITA valence),
#     following Hogarth & Einhorn's directional push concept.

# ===================================================================
# The prompt
# ===================================================================

EXTRACTION_SYSTEM_PROMPT = """You are an expert at analyzing patient symptom reports in primary care and emergency triage contexts. Your task is to decompose a patient's symptom description into a structured storyboard of discrete informational aspects, and to generate a plausible self-diagnosis the patient might have researched online.

The patient report describes symptoms that a person is seeking advice about. Your job is to identify the clinical information components and tag each with its urgency signal — the directional push it would have on a triage decision if heard in isolation.

## What is an aspect?

An aspect is a single, self-contained piece of clinical information that could affect a triage decision. Each aspect should:

1. Contain ONE distinct piece of information — not two things bundled together
2. Be 1-3 sentences long
3. Be written in first person, from the patient's perspective, as they would naturally say it
4. Be understandable on its own without the other aspects
5. Not repeat information from another aspect

The patient should sound like a layperson — not a clinician. Use everyday language, not clinical terminology. "My chest hurts" not "I am experiencing substernal chest pain." The storyboard represents the raw facts; the patient simulator will later add profile-specific framing.

## Categories

Each aspect belongs to exactly one of six categories, derived from the Common Sense Model of illness representation (Leventhal et al., 1992) and clinical communication practice.

### CHIEF_COMPLAINT
The primary symptom or reason the patient is seeking advice. This is ALWAYS the first aspect and there is ALWAYS exactly one.

This is the identity-dimension anchor — the door-in complaint that frames all subsequent information. It should be the most prominent symptom as the patient would naturally lead with it.

When to use: The primary presenting symptom the patient would open with. Must be present, must be first.

Examples:
  "I've been having chest pain for the past few hours."
  "I woke up this morning with a really bad headache that won't go away."
  "My right knee has been swelling and hurting for about a week now."

### SYMPTOM
A secondary symptom distinct from the chief complaint that is relevant to the triage decision.

This is an additional identity-dimension feature — a separate symptom the patient also has. Each SYMPTOM aspect covers one discrete additional complaint. When multiple secondary symptoms are present, create one aspect per symptom.

When to use: The patient has an additional symptom beyond the chief complaint that carries its own triage weight. Do not bundle — if two symptoms are distinct and each has independent triage relevance, create separate aspects.

Important: Distinguish SYMPTOM from SEVERITY. "I also have shortness of breath" is a SYMPTOM (a separate complaint). "The pain is the worst I've ever felt in my life" is SEVERITY (an intensity descriptor about the chief complaint or another symptom).

Examples:
  "I've also been feeling really short of breath, especially when I try to do anything."
  "On top of the headache, I've been really sensitive to light — even dim light is painful."
  "I've also noticed my urine has been really dark, almost brown."

### TIMELINE
Onset, duration, and progression of the chief complaint or key symptoms.

This is the timeline dimension from Leventhal — when it started, how long it has been going on, and whether it is getting better, staying the same, or getting worse.

When to use: Information about when symptoms began, how long they have lasted, or whether they have changed over time. Sudden vs. gradual onset is a major urgency signal in triage.

Examples:
  "The pain started suddenly about two hours ago, out of nowhere."
  "This has been going on for about three weeks, slowly getting worse."
  "It started three days ago and has been pretty much the same since then — not getting better but not getting worse."

### SEVERITY
The intensity, pain level, or functional impact of the symptoms.

This captures how bad the symptoms are — not what they are (that is SYMPTOM or CHIEF_COMPLAINT) but how severely they are affecting the patient. Includes pain ratings, functional impairment, and descriptors of intensity.

When to use: Information about how bad the symptoms feel, how much they interfere with daily activities, or intensity descriptors. Severity is often the most direct urgency signal.

Examples:
  "The pain is honestly a 9 out of 10 — it's the worst headache I've ever had in my life."
  "I can barely get off the couch. Even walking to the bathroom takes it out of me."
  "It's uncomfortable but I'd say it's only about a 3 or 4 — annoying but manageable."

### CONTEXT
Relevant medical history, medications, lifestyle factors, or demographics that affect triage.

This captures background enabling and risk factors — information that would change the urgency of the same symptom set for a different patient. Includes past medical history, current medications, family history, and relevant lifestyle factors.

When to use: Background information that affects how dangerous the current symptoms are likely to be — prior conditions, known risk factors, relevant medication use, demographic risk context.

Examples:
  "I have a history of blood clots — I had a DVT in my leg about two years ago."
  "I'm 58, I smoke about half a pack a day, and I have high blood pressure that I take medication for."
  "I'm 23 and pretty healthy — no major medical problems, no medications."

### ASSOCIATED
Aggravating or relieving factors, associated features, or review-of-systems items that add triage information.

This captures clinical details that accompany the main symptoms but are distinct from them — things that make the symptoms better or worse, associated symptoms that flag specific dangerous syndromes, and system review items that would be asked about in a clinical assessment.

When to use: Information about what worsens or relieves symptoms, or associated features that are clinically relevant but do not constitute a separate named symptom. Also use for "flags" — red-flag features like radiation, diaphoresis, syncope, fever accompanying another chief complaint.

Important: If a finding is significant enough to stand alone as a separate complaint, it is a SYMPTOM, not ASSOCIATED. If it is a qualifier or accompanying feature of another symptom, it is ASSOCIATED.

Examples:
  "The chest pain seems to get worse when I take a deep breath, and it gets a little better when I lean forward."
  "The headache started with some flashing lights in my vision, which lasted about 20 minutes before the headache hit."
  "The pain radiates down my left arm and I've been sweating more than usual."

## Category disambiguation rules

When an aspect could plausibly belong to multiple categories, apply these rules:

1. CHIEF_COMPLAINT vs SYMPTOM: The chief complaint is what the patient opens with — their primary concern. Everything else is SYMPTOM. There is exactly one CHIEF_COMPLAINT, always first.

2. SYMPTOM vs SEVERITY: If it is a separate named complaint (shortness of breath, nausea, dizziness), it is SYMPTOM. If it describes how bad a symptom already named is (intensity, functional impact, pain rating), it is SEVERITY.

3. SYMPTOM vs ASSOCIATED: If the patient would mention it as a separate problem ("I've also been having X"), it is SYMPTOM. If it is a feature or qualifier of another symptom ("the pain also goes to my arm"), it is ASSOCIATED.

4. TIMELINE vs CONTEXT: TIMELINE is about when and how the current symptoms have evolved. CONTEXT is background history not about this episode (prior conditions, medications, baseline health status). "The headache started Tuesday" is TIMELINE. "I've had migraines before" is CONTEXT.

5. SEVERITY vs CONTEXT: Severity is about the current episode. Context is background. "This is the worst pain of my life" is SEVERITY. "I usually have a high pain tolerance" is CONTEXT.

6. TIMELINE vs SYMPTOM for escalation events: If a symptom that was previously present in one context now appears in a qualitatively different context, that new occurrence is a SYMPTOM, not just TIMELINE. The test: "Does this event carry independent triage weight, or is it merely tracking when the existing symptom started?" Pain that previously occurred only with exertion and now wakes the patient from sleep is a new SYMPTOM (rest angina is clinically distinct from exertional angina). A symptom that has simply lasted longer or gotten gradually worse without changing character is TIMELINE. When in doubt: if a clinician would treat this as a separate clinical event warranting its own assessment, make it a SYMPTOM.

## Urgency signal

The urgency signal captures the directional push of an aspect on the triage decision, following Hogarth & Einhorn's (1992) concept of positive/negative evidence valence.

For each aspect, ask: "If a triage clinician heard ONLY this one piece of information — nothing else — would it push them toward thinking this patient needs more urgent care, less urgent care, or is it neutral?"

alarming — Pushes toward HIGHER urgency (Emergency or Urgent).
  Sudden onset, severe intensity, dangerous syndromes (chest pain + radiation + diaphoresis),
  high-risk patient profile for this symptom, red-flag features, rapid progression, functional
  collapse. Any feature a clinician would flag as "this patient may need immediate attention."

neutral — Does not clearly push either direction.
  Could be consistent with a wide range of urgency levels. Pure context, ambiguous features,
  symptoms that are common at all severity levels.

reassuring — Pushes toward LOWER urgency (Observation or Routine).
  Long stable duration with no change, mild intensity, young healthy patient with no risk factors,
  symptom reliably better with over-the-counter treatment, no associated dangerous features,
  benign aggravating factors. Features that lower the likelihood of a serious underlying cause.

CRITICAL: Urgency signal is about the ASPECT IN ISOLATION, not about the overall case urgency. A CONTEXT aspect about a patient being young and healthy is reassuring even if the case is ultimately urgent. Tag each aspect based on its own directional push, independent of your overall impression of the case.

## Importance

Importance measures how much an aspect matters for the triage decision. Use this scale:

  5 — DECISIVE: This aspect alone could determine or reverse the triage tier. Removing it would likely change the urgency classification. Central to the triage question.
      Examples: "The chest pain radiates down my left arm and I've been sweating" / "I suddenly can't see out of my right eye"

  4 — SIGNIFICANT: Strongly affects the triage judgment but isn't solely decisive. Provides major weight toward a particular urgency tier.
      Examples: "The pain started suddenly rather than gradually" / "I have a history of heart disease"

  3 — RELEVANT: Contributes meaningfully to the triage picture. Adds information that would be missed if absent.
      Examples: "This has been going on for three days" / "I also have mild nausea"

  2 — CONTEXTUAL: Provides useful background but wouldn't shift the tier on its own. Helps the clinician understand the patient's situation.
      Examples: "I'm 35 years old and otherwise healthy" / "The pain is about a 4 out of 10"

  1 — MINOR: Incidental detail. Clinically real but carries minimal triage weight.
      Examples: "I've been drinking a lot of water lately" / "My sleep has been a bit off"

## Grain size and count

Target 5-8 aspects per case. The right number depends on the case's clinical complexity:
  - Simple cases (one symptom, clear trajectory): 5-6 aspects
  - Moderate cases (several symptoms, some context): 6-7 aspects
  - Complex cases (multiple systems, significant history): 7-8 aspects
  - If you are producing more than 8, you are splitting too finely

Each aspect should carry independent triage relevance. Do NOT create aspects for:
  - Repeated information in different words
  - Details with no plausible effect on triage (what the patient was wearing)
  - Filler or transitions from the original narrative

DO create separate aspects when:
  - Two symptoms have different urgency signals even if mentioned together
  - A severity descriptor is so extreme it warrants its own urgency weighting
  - A context factor creates a different risk profile than what was already described

## Structural requirements

Every storyboard must include:
  - Exactly one CHIEF_COMPLAINT (always aspect a1)
  - At least two SYMPTOM aspects
  - At least one TIMELINE aspect
  - At least one CONTEXT aspect
  - At least one aspect with urgency_signal "alarming" (the case must have some urgency trigger)

## Self-diagnosis (for cyberchondriac profile)

In addition to the storyboard, generate one self_diagnosis field: a plausible but potentially incorrect diagnosis that a patient might find after internet research on their chief complaint. This is used to give the cyberchondriac profile a specific anchoring theory.

The self-diagnosis must be:
  - A real medical condition with a recognizable name
  - Plausible given the chief complaint (a patient could reasonably find it via internet search)
  - Not necessarily correct — the point is that it could explain some symptoms but may miss or contradict others
  - Ideally different in urgency from the true triage tier at least some of the time
  - 1-5 words: a condition name, not a sentence

Examples: "acid reflux", "tension headache", "anxiety attack", "pulled muscle", "viral gastroenteritis"

## Output format

Respond with ONLY a valid JSON object. No explanation, no preamble, no markdown fences.
The object must have exactly these two fields:
  "self_diagnosis" — string: the cyberchondriac anchoring diagnosis (see above)
  "storyboard"     — array of aspect objects

Each aspect object must have exactly these five fields:
  "id"             — sequential: "a1", "a2", "a3", ...
  "category"       — exactly one of: CHIEF_COMPLAINT, SYMPTOM, TIMELINE, SEVERITY, CONTEXT, ASSOCIATED
  "content"        — the aspect text (1-3 sentences, first person, everyday patient language)
  "urgency_signal" — exactly one of: alarming, neutral, reassuring
  "importance"     — integer 1-5 (how much this aspect matters for the triage decision)

## Self-check before outputting

Verify all of the following before producing your output:
  1. "a1" is a CHIEF_COMPLAINT and is the only CHIEF_COMPLAINT
  2. There are at least two SYMPTOM aspects
  3. There is at least one TIMELINE aspect
  4. There is at least one CONTEXT aspect
  5. There is at least one aspect with urgency_signal "alarming"
  6. Every aspect contains a DIFFERENT piece of information (no redundancy)
  7. You have between 5 and 8 aspects total
  8. Urgency signal tags reflect each aspect IN ISOLATION, not your overall impression of the case
  9. Importance scores follow the 1-5 anchors defined above
  10. The content is in plain patient language — no clinical jargon
  11. The self_diagnosis is a plausible condition a patient could find online
  12. The output is valid JSON — a single object with "self_diagnosis" and "storyboard", nothing else"""


def build_extraction_messages(case: dict) -> list:
    """Build the messages for storyboard extraction.

    Args:
        case: dict with 'input' field (patient symptom description)
              and optionally 'specialty', 'true_urgency', etc.

    Returns:
        List of chat messages for the extraction model
    """
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": case["input"]},
    ]


# ===================================================================
# Valid values for validation
# ===================================================================

VALID_CATEGORIES = {
    "CHIEF_COMPLAINT", "SYMPTOM", "TIMELINE",
    "SEVERITY", "CONTEXT", "ASSOCIATED",
}

VALID_URGENCY_SIGNALS = {"alarming", "neutral", "reassuring"}


def validate_storyboard(aspects: list) -> dict:
    """Validate a triage storyboard extraction.

    Args:
        aspects: List of aspect dicts from the extraction model

    Returns:
        Dict with 'valid' bool, 'errors' list, 'warnings' list, and 'stats'
    """
    errors = []
    warnings = []

    # --- Count ---
    if len(aspects) < 3:
        errors.append(f"Too few aspects: {len(aspects)} (minimum 3)")
    elif len(aspects) < 5:
        warnings.append(f"Fewer than 5 aspects: {len(aspects)} (acceptable for very simple cases)")
    if len(aspects) > 8:
        warnings.append(f"More than 8 aspects: {len(aspects)} (may be splitting too finely)")
    if len(aspects) > 12:
        errors.append(f"Too many aspects: {len(aspects)} (maximum 12)")

    # --- Required fields ---
    required_fields = {"id", "category", "content", "urgency_signal", "importance"}
    for i, a in enumerate(aspects):
        missing = required_fields - set(a.keys())
        if missing:
            errors.append(f"Aspect {i+1}: missing fields {missing}")

    # --- Valid values ---
    for i, a in enumerate(aspects):
        cat = a.get("category")
        if cat and cat not in VALID_CATEGORIES:
            errors.append(f"Aspect {i+1}: invalid category '{cat}'")
        sig = a.get("urgency_signal")
        if sig and sig not in VALID_URGENCY_SIGNALS:
            errors.append(f"Aspect {i+1}: invalid urgency_signal '{sig}'")
        imp = a.get("importance")
        if imp is not None:
            if not isinstance(imp, int) or imp < 1 or imp > 5:
                errors.append(f"Aspect {i+1}: importance must be int 1-5, got '{imp}'")

    # --- Structural requirements ---
    categories_present = [a.get("category") for a in aspects]

    # Exactly one CHIEF_COMPLAINT, must be first
    chief_count = categories_present.count("CHIEF_COMPLAINT")
    if chief_count == 0:
        errors.append("Missing required category: CHIEF_COMPLAINT")
    elif chief_count > 1:
        errors.append(f"Multiple CHIEF_COMPLAINT aspects: {chief_count} (must be exactly 1)")
    if aspects and aspects[0].get("category") != "CHIEF_COMPLAINT":
        errors.append("First aspect must be CHIEF_COMPLAINT")

    # At least two SYMPTOMs
    n_symptoms = categories_present.count("SYMPTOM")
    if n_symptoms < 2:
        errors.append(f"Too few SYMPTOM aspects: {n_symptoms} (minimum 2)")

    # At least one TIMELINE
    if "TIMELINE" not in categories_present:
        errors.append("Missing required category: TIMELINE")

    # At least one CONTEXT
    if "CONTEXT" not in categories_present:
        errors.append("Missing required category: CONTEXT")

    # At least one alarming aspect
    urgency_signals = [a.get("urgency_signal") for a in aspects]
    if "alarming" not in urgency_signals:
        errors.append("No 'alarming' urgency_signal — case must have at least one urgency trigger")

    # --- Redundancy check ---
    contents = [a.get("content", "").lower().strip() for a in aspects]
    for i in range(len(contents)):
        for j in range(i + 1, len(contents)):
            if contents[i] and contents[j]:
                words_i = set(contents[i].split())
                words_j = set(contents[j].split())
                if len(words_i) > 3 and len(words_j) > 3:
                    overlap = len(words_i & words_j) / min(len(words_i), len(words_j))
                    if overlap > 0.8:
                        warnings.append(
                            f"Potential redundancy: aspects {i+1} and {j+1} "
                            f"have {overlap:.0%} word overlap"
                        )

    # --- Urgency signal distribution check ---
    n_alarming = urgency_signals.count("alarming")
    n_neutral = urgency_signals.count("neutral")
    n_reassuring = urgency_signals.count("reassuring")
    if n_neutral == len(aspects):
        warnings.append("Every aspect tagged neutral — urgency signals may be under-differentiated")

    # --- Importance distribution check ---
    importances = [a.get("importance", 0) for a in aspects]
    if importances and max(importances) < 4:
        warnings.append("No aspect rated importance 4-5 — the central urgency trigger may be under-weighted")

    categories_set = set(categories_present)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "n_aspects": len(aspects),
            "categories": {cat: categories_present.count(cat)
                          for cat in VALID_CATEGORIES if cat in categories_set},
            "urgency_signal": {
                "alarming": n_alarming,
                "neutral": n_neutral,
                "reassuring": n_reassuring,
            },
            "importance_range": (min(importances), max(importances)) if importances else (0, 0),
        }
    }

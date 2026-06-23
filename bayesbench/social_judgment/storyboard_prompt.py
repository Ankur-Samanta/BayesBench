"""
Storyboard Extraction Prompt

Decomposes an AITA post into discrete informational aspects for the
sequential-vs-simultaneous judgment experiment.

Theoretical grounding (producer-side theories only — see THEORETICAL_FOUNDATIONS.md):

  Narrative structure:
    Labov & Waletzky (1967), Labov (1972) — the structure of told personal
    narratives. AITA posts are personal narratives; Labov's components
    (orientation, complicating action, evaluation, resolution) define the
    natural units of decomposition.

  Account strategies:
    Scott & Lyman (1968) — accounts as linguistic devices for explaining
    blameworthy behavior: excuses (deny responsibility) and justifications
    (deny wrongness).
    Schönbach (1990) — extends to concessions and refusals; defines the
    account episode structure (failure event → reproach → account → evaluation).
    Benoit (1995) — 14 image repair sub-strategies providing granular coding.

  Attribution framing:
    Weiner (1985) — posters manipulate perceived locus (internal/external),
    controllability, and stability to reduce blame attribution.
    Heider (1958) — moral judgment depends on who the causal agent is; the
    poster/other action split follows from attribution of agency.

  Evidence valence:
    Hogarth & Einhorn (1992) — positive/negative evidence valence as the
    directional push on belief updating. Grounds our pro/against tagging.
"""

# ===================================================================
# Category taxonomy
# ===================================================================
#
# Six categories derived from two producer-side frameworks:
#
#   From Labov (1972) — narrative structure:
#     ORIENTATION       Labov's orientation component
#     ACTION_POSTER     Labov's complicating action, agent = poster (Heider)
#     ACTION_OTHER      Labov's complicating action, agent = other party (Heider)
#     OUTCOME           Labov's resolution component
#
#   From Accounts Theory — Scott & Lyman (1968), Schönbach (1990), Benoit (1995):
#     ACCOUNT           Schönbach phase 3: the poster's excuse/justification/concession
#
#   From Schönbach (1990) — account episode structure:
#     REACTION          Schönbach phases 2+4: external reproach or evaluation
#                       by third parties (the social response to the poster's behavior)
#
# Design notes:
#   - The poster/other split within complicating actions is grounded in Heider's
#     (1958) attribution theory: moral judgment requires identifying the causal
#     agent. Internal locus (poster did it) attracts blame differently than
#     external locus (other party did it).
#   - ACCOUNT is separate from ACTION_POSTER because Accounts Theory shows that
#     how people FRAME their behavior (excuse/justify/concede) is analytically
#     distinct from what they DID. Separating them lets us measure how much
#     the judge's verdict is driven by facts vs framing.
#   - REACTION captures third-party judgments and responses. In Schönbach's
#     framework, the AITA post is phase 3 (account). Third-party reactions
#     within the post are either phase 2 (reproach that triggered the account)
#     or phase 4 (evaluation of the poster's behavior). Both function as
#     external moral evidence — someone else's verdict signal.
#   - Labov's ABSTRACT maps to the post title (handled separately).
#   - Labov's CODA maps to "so AITA?" (handled separately).
#   - Labov's EVALUATION (narrator's attitude, "so what") is distributed:
#     when defensive, it falls under ACCOUNT; when it describes emotional
#     consequences, it falls under OUTCOME. Labov himself notes evaluation
#     is embedded throughout narratives, not a discrete section.

# ===================================================================
# The prompt
# ===================================================================

EXTRACTION_SYSTEM_PROMPT = """You are an expert at analyzing personal narratives about social conflict. Your task is to decompose an AITA (Am I The Asshole) post into a structured storyboard of discrete informational aspects.

An AITA post is a personal narrative (Labov, 1972) in which the narrator describes a social conflict and asks whether they were in the wrong. It is also an account (Scott & Lyman, 1968) — a linguistic device for explaining potentially blameworthy behavior. Your job is to identify the structural components of both the narrative and the account.

## What is an aspect?

An aspect is a single, self-contained piece of information that would affect a reasonable person's moral judgment of the poster. Each aspect should:

1. Contain ONE distinct piece of information — not two things bundled together
2. Be 1-3 sentences long
3. Be written in first person, from the poster's perspective
4. Be understandable on its own without the other aspects
5. Not repeat information from another aspect

## Categories

Each aspect belongs to exactly one of six categories. These are derived from the narrative structure of personal experience (Labov, 1972) and accounts theory (Scott & Lyman, 1968; Schönbach, 1990).

### ORIENTATION
Background information that establishes the situation before any conflict occurs: who is involved, their relationships, relevant history, the circumstances.

This is Labov's orientation component — the "who, what, when, where" that orients the listener before the complicating action begins.

When to use: The information sets the scene but does not advance the conflict. It could be removed and the core dispute would still make sense, though context would be lost.

Examples:
  "My wife and I have been married for 7 years. We have three kids — ages 5, 4, and 2."
  "I work at a local craft brewery. They package around 60,000 liters of beer a day."
  "My daughter was in 1st grade at the time. We invited everyone in her class to her birthday party — about 20 kids."

### ACTION_POSTER
Something the poster DID that is relevant to the moral judgment — a specific behavior, decision, or response by the poster.

This is Labov's complicating action where the poster is the agent. Per Heider's (1958) attribution theory, identifying the poster as the causal agent is what makes this an action the judge can assign blame or credit for.

When to use: The poster performed a concrete action, made a decision, or chose a response. This is about WHAT THEY DID, not why they did it (that's ACCOUNT).

Important: If a sentence contains both an action AND a justification for it, SPLIT them into separate aspects. "I said no because I'm trying to quit smoking" contains an ACTION_POSTER ("I said no") and an ACCOUNT ("because I'm trying to quit smoking"). The action is the behavior; the account is the framing.

Examples:
  "I told my wife I was going to take the night shift. I didn't ask — I told her."
  "I sighed loudly to let her know she was annoying me."
  "I told Ethan's parents that I was going to have to ask them to leave."
  "I've been taking some beer home anyway, only for myself and my family."

### ACTION_OTHER
Something another party did that is relevant to the moral judgment — their behavior, responses, provocations, or refusals.

This is Labov's complicating action where someone other than the poster is the agent. Per Heider, attributing agency to the other party shifts how the judge assigns responsibility.

When to use: Another person performed a concrete action or gave a concrete response within the conflict. Their behavior can push the judgment toward or against the poster depending on how reasonable it was.

Examples:
  "Emily argued that I used to smoke in my car and it wasn't fair to forbid her from doing the same."
  "Ethan's parents shrugged and said their son was having fun and that's what being a kid is all about."
  "My wife said I'd be putting my wants ahead of my family."
  "He blew a raspberry at me and ran off again."

### OUTCOME
What resulted from the actions — consequences, aftermath, fallout, emotional states that followed.

This is Labov's resolution component — how things turned out. Includes both concrete consequences (someone left, someone stopped speaking) and emotional aftermath (someone felt terrible, someone cried).

When to use: Something happened AS A RESULT of the conflict. The causal direction goes: actions → outcome.

Examples:
  "She stopped talking for the rest of the trip."
  "He left and hasn't spoken to me all day."
  "Immediately afterward, he started wearing a beanie all the time and refuses to take it off."
  "Ethan's parents have been on the warpath for over six months."

### ACCOUNT
The poster's own explanation, defense, or framing of their behavior.

This comes from Accounts Theory (Scott & Lyman, 1968). When people describe potentially blameworthy actions, they use specific strategies to manage responsibility:

  EXCUSE — admits the act may look bad, denies full responsibility:
    "I didn't have a choice."
    "I didn't know she was that upset."
    "I was exhausted and not thinking clearly."

  JUSTIFICATION — admits responsibility, denies the act was wrong:
    "It's my car and I have every right to set rules."
    "She had it coming after what she said."
    "I was just trying to protect him from being bullied."

  CONCESSION — accepts some blame, provides mitigating context:
    "I know the sigh was rude, but I was at the end of my patience."
    "I feel bad about it, but I just couldn't handle it anymore."

When to use: The poster is EXPLAINING or DEFENDING their behavior, not describing a concrete action. Look for "because", "since", "I felt justified", "I was just trying to", "I didn't mean to", "in my defense" — these signal account-giving. Also includes the poster's emotional framing when used defensively ("I felt terrible" as a bid for sympathy).

Important: Accounts are not neutral information. They are strategic constructions — the poster's attempt to manage how their behavior is perceived. Separating accounts from actions lets us analyze how much the judge is influenced by facts versus framing.

Examples:
  "I argued it doesn't really change logistics. I'm just swapping sleeping time with working time."
  "I tried to argue I was just looking out for him. He has a feminine face and long hair would cause confusion."
  "I feel guilty about taking the beer, but knowing the alternative is pouring it down the drain makes me feel like it's not that bad."
  "I'm a night owl by nature. Night shift is more laid-back and I get a shift premium."

### REACTION
A third party's judgment, opinion, or response regarding the poster's behavior.

In Schönbach's (1990) account episode framework, an AITA post is phase 3 (the account). Third-party reactions within the post are either phase 2 (the reproach that prompted the poster to seek judgment) or phase 4 (someone else's evaluation). Both function as external moral evidence — another person's verdict signal about whether the poster was wrong.

When to use: Someone other than the poster AND other than the main other party weighs in on the situation — either criticizing the poster (reproach) or supporting them (validation). Also includes collective responses ("everyone agreed", "the other parents took my side").

Examples:
  "Margaret told me I was being a hypocrite since I used to smoke in my car too."
  "The other parents who witnessed it were almost all on my side."
  "My husband thinks I've been a little harsh."
  "Robert's parents were grateful when they found out."
  "During an argument, he told me how disgusting and ugly he feels and that I took away the one thing he liked about himself."

## Category disambiguation rules

When an aspect could plausibly belong to multiple categories, apply these rules:

1. ACTION vs ACCOUNT: If the poster DID something AND explains why in the same breath, split them. The test: "Could I describe this without the reason?" If yes, the reason is a separate ACCOUNT. Example: "I kicked them out because they were disrespectful" → ACTION_POSTER ("I kicked them out") + ACCOUNT ("because they were disrespectful").

2. ACTION_OTHER vs REACTION: If the other party is the person the conflict is ABOUT (the main antagonist/counterpart), it's ACTION_OTHER. If they are a THIRD PARTY commenting on the situation (friend, spouse, parent, bystander), it's REACTION. The test: "Is this person one of the two sides of the conflict, or an observer?" Emily's argument is ACTION_OTHER (she's a party to the conflict). Margaret's comment is REACTION (she's a bystander weighing in).

3. ORIENTATION vs ACTION: If the information is about the state of affairs BEFORE the conflict began, it's ORIENTATION. If it's part of the conflict sequence itself, it's ACTION. The test: "Did this happen before or during the dispute?"

4. OUTCOME vs ACTION: If something happened as a RESULT of the conflict actions (downstream effect), it's OUTCOME. If it's part of the active conflict exchange (someone does X, someone responds with Y), it's ACTION. The test: "Is this a move in the conflict, or a consequence of how the conflict went?"

5. ACCOUNT vs ORIENTATION: Poster's personal traits or preferences cited to EXPLAIN their behavior are ACCOUNT ("I'm a night owl by nature" → justifies wanting night shift). Poster's personal traits cited purely for context with no defensive purpose are ORIENTATION ("I have anxiety and confidence issues" → sets the scene).

## Valence

Valence captures the directional push of an aspect on moral judgment, following Hogarth & Einhorn's (1992) concept of positive/negative evidence.

For each aspect, ask: "If a judge heard ONLY this one piece of information — nothing else — would it push them toward thinking the poster was in the wrong, not in the wrong, or neither?"

pro_poster — Makes the poster look BETTER. Pushes toward NTA.
  The poster was reasonable, had a legitimate reason, showed restraint, was provoked,
  or the other party was clearly in the wrong. Third-party support for the poster.
  Per Weiner (1985): the poster's action appears externally caused, uncontrollable,
  or unstable (one-time) — factors that reduce blame.

against_poster — Makes the poster look WORSE. Pushes toward YTA.
  The poster was rude, selfish, made a unilateral decision, ignored someone's
  feelings, or escalated unnecessarily. Third-party criticism of the poster.
  Per Weiner: the poster's action appears internally caused, controllable,
  and stable (pattern of behavior) — factors that increase blame.

neutral — Does not clearly push either direction.
  Pure context, factual setup, ambiguous situations where reasonable people
  would disagree about which direction it pushes.

CRITICAL: Valence is about the ASPECT IN ISOLATION, not about the overall story. A background fact can be neutral even in a story where the poster is clearly wrong. An excuse can be pro_poster even if the judge might not find it convincing — it's still a frame that pushes toward NTA. Tag each aspect based on its own content, not on your overall impression of the poster.

## Importance

Importance measures how much an aspect matters for the moral judgment. Use this scale:

  5 — DECISIVE: This aspect alone could determine or reverse the verdict. Removing it would likely change the judgment. Central to the moral question.
      Examples: "I made the decision unilaterally without asking my wife." / "He asked me to shave his head and I did it."

  4 — SIGNIFICANT: Strongly affects the judgment but isn't solely decisive. Provides major weight to one side.
      Examples: "Emily argued I was being unfair since I used to smoke in my car too." / "Ethan's parents watched the whole thing and did nothing."

  3 — RELEVANT: Contributes meaningfully to the judgment. Adds texture or nuance that would be missed if absent.
      Examples: "My wife doesn't like being alone at night." / "I feel guilty about it, but the beer would be poured down the drain anyway."

  2 — CONTEXTUAL: Provides useful background but wouldn't shift the verdict on its own. Helps the judge understand the situation.
      Examples: "We have three kids — ages 5, 4, and 2." / "It was a 20-minute drive."

  1 — MINOR: Incidental detail. Included for completeness but carries minimal moral weight.
      Examples: "We arranged to meet in a certain place." / "This was pre-covid."

## Grain size and count

Target 6-10 aspects per post. The right number depends on the post's complexity:
  - Short, simple posts (one event, clear conflict): 5-7 aspects
  - Longer posts (multiple events, several parties): 8-10 aspects
  - If you are producing more than 10, you are splitting too finely

Each aspect should be a JUDGMENT-RELEVANT unit — a piece of information that could shift a reasonable person's moral assessment. Do NOT create aspects for:
  - Details that don't affect the moral judgment ("we stopped for gas on the way")
  - Repetitions of the same information in different words
  - Narrative filler or transitions
  - The "so AITA?" coda — this is the frame of the post, not informational content

DO create separate aspects when:
  - Two events carry different moral weight (even if described in one sentence)
  - A single sentence contains an action AND a justification — split them
  - Different people's actions pull the judgment in different directions

## Structural requirements

Every storyboard must include:
  - At least one ORIENTATION (the judge needs context)
  - At least one ACTION_POSTER (there must be something the poster did to judge)
  - At least one ACTION_OTHER (there must be the other side's behavior)
  - The specific event or request that triggered the conflict (typically the highest-importance ACTION_POSTER or ACTION_OTHER — the moment the situation went from normal to disputed)

## Edge cases

Posts with EDITS or UPDATES: Treat edit content the same as the main post. If the edit adds new information, include it as additional aspects. If it merely clarifies something already covered, do not create a duplicate.

Very short posts: Some posts are only 2-3 sentences. It is acceptable to produce fewer than 6 aspects if the post genuinely contains fewer than 6 judgment-relevant pieces of information. Do not pad with trivial aspects.

Multiple conflicts: If the post describes two distinct conflicts, capture both. The aspects should cover each conflict's key elements. The storyboard may be on the longer side (8-10 aspects).

Poster is clearly NTA: Even when the poster is obviously not in the wrong, extract aspects faithfully. There should still be aspects that are against_poster (every conflict has two sides, even if one is weak). If the post is genuinely one-sided, having mostly pro_poster aspects with only 1-2 against_poster is correct — do not force balance.

## Output format

Respond with ONLY a valid JSON array. No explanation, no preamble, no markdown fences.
Each element must have exactly these five fields:
  "id"         — sequential: "a1", "a2", "a3", ...
  "category"   — exactly one of: ORIENTATION, ACTION_POSTER, ACTION_OTHER, OUTCOME, ACCOUNT, REACTION
  "content"    — the aspect text (1-3 sentences, first person, from the poster's perspective)
  "valence"    — exactly one of: pro_poster, neutral, against_poster
  "importance" — integer 1-5 (how much this aspect matters for the moral judgment)

## Self-check before outputting

Verify all of the following before producing your output:
  1. Every aspect contains a DIFFERENT piece of information (no redundancy)
  2. You have at least one ORIENTATION, one ACTION_POSTER, and one ACTION_OTHER
  3. The triggering event — the specific moment the conflict started — is captured
  4. The aspects together cover all key information from the post (nothing important is missing)
  5. Actions and accounts are SEPARATED — no aspect contains both "I did X" and "because Y"
  6. Valence tags reflect each aspect IN ISOLATION, not your overall impression
  7. Importance scores follow the 1-5 anchors defined above
  8. You have between 5 and 10 aspects total
  9. The output is valid JSON — a single array of objects, nothing else"""


def build_extraction_messages(post: dict) -> list:
    """Build the messages for storyboard extraction.

    Args:
        post: dict with 'title' and 'text' fields

    Returns:
        List of chat messages for the extraction model
    """
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Title: {post['title']}\n\n{post['text']}"},
    ]


# ===================================================================
# Valid values for validation
# ===================================================================

VALID_CATEGORIES = {
    "ORIENTATION", "ACTION_POSTER", "ACTION_OTHER",
    "OUTCOME", "ACCOUNT", "REACTION",
}

VALID_VALENCES = {"pro_poster", "neutral", "against_poster"}


def validate_storyboard(aspects: list) -> dict:
    """Validate a storyboard extraction.

    Args:
        aspects: List of aspect dicts from the extraction model

    Returns:
        Dict with 'valid' bool, 'errors' list, and 'warnings' list
    """
    errors = []
    warnings = []

    # --- Count ---
    if len(aspects) < 3:
        errors.append(f"Too few aspects: {len(aspects)} (minimum 3)")
    elif len(aspects) < 5:
        warnings.append(f"Fewer than 5 aspects: {len(aspects)} (acceptable for very short posts)")
    if len(aspects) > 10:
        warnings.append(f"More than 10 aspects: {len(aspects)} (may be splitting too finely)")
    if len(aspects) > 15:
        errors.append(f"Too many aspects: {len(aspects)} (maximum 15)")

    # --- Required fields ---
    required_fields = {"id", "category", "content", "valence", "importance"}
    for i, a in enumerate(aspects):
        missing = required_fields - set(a.keys())
        if missing:
            errors.append(f"Aspect {i+1}: missing fields {missing}")

    # --- Valid values ---
    for i, a in enumerate(aspects):
        cat = a.get("category")
        if cat and cat not in VALID_CATEGORIES:
            errors.append(f"Aspect {i+1}: invalid category '{cat}'")
        val = a.get("valence")
        if val and val not in VALID_VALENCES:
            errors.append(f"Aspect {i+1}: invalid valence '{val}'")
        imp = a.get("importance")
        if imp is not None:
            if not isinstance(imp, int) or imp < 1 or imp > 5:
                errors.append(f"Aspect {i+1}: importance must be int 1-5, got '{imp}'")

    # --- Structural requirements ---
    categories_present = {a.get("category") for a in aspects}
    if "ORIENTATION" not in categories_present:
        errors.append("Missing required category: ORIENTATION")
    if "ACTION_POSTER" not in categories_present:
        errors.append("Missing required category: ACTION_POSTER")
    if "ACTION_OTHER" not in categories_present and "REACTION" not in categories_present:
        errors.append("Missing required: at least one ACTION_OTHER or REACTION")

    # --- Redundancy check (crude: flag aspects with very similar content) ---
    contents = [a.get("content", "").lower().strip() for a in aspects]
    for i in range(len(contents)):
        for j in range(i + 1, len(contents)):
            if contents[i] and contents[j]:
                # Simple word overlap check
                words_i = set(contents[i].split())
                words_j = set(contents[j].split())
                if len(words_i) > 3 and len(words_j) > 3:
                    overlap = len(words_i & words_j) / min(len(words_i), len(words_j))
                    if overlap > 0.8:
                        warnings.append(
                            f"Potential redundancy: aspects {i+1} and {j+1} "
                            f"have {overlap:.0%} word overlap"
                        )

    # --- Valence distribution check ---
    valences = [a.get("valence") for a in aspects]
    n_pro = valences.count("pro_poster")
    n_against = valences.count("against_poster")
    n_neutral = valences.count("neutral")
    if n_pro == 0 and n_against == 0:
        warnings.append("No directional valence tags — all neutral")
    if n_neutral == len(aspects):
        warnings.append("Every aspect tagged neutral — valence may be under-differentiated")

    # --- Importance distribution check ---
    importances = [a.get("importance", 0) for a in aspects]
    if importances and max(importances) < 4:
        warnings.append("No aspect rated importance 4-5 — the central conflict may be under-weighted")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "n_aspects": len(aspects),
            "categories": {cat: sum(1 for a in aspects if a.get("category") == cat)
                          for cat in VALID_CATEGORIES if any(a.get("category") == cat for a in aspects)},
            "valence": {"pro_poster": n_pro, "neutral": n_neutral, "against_poster": n_against},
            "importance_range": (min(importances), max(importances)) if importances else (0, 0),
        }
    }


# ===================================================================
# Account sub-type coding (optional second-pass annotation)
# ===================================================================
# When deeper analysis is needed, ACCOUNT aspects can be further coded
# using the following taxonomy from Scott & Lyman (1968) and Benoit (1995).
# This is NOT part of the extraction prompt — it's for downstream analysis.

ACCOUNT_SUBTYPES = {
    # Scott & Lyman (1968) — Excuses
    "excuse_accident": "Appeal to accidents — unforeseeable mishap",
    "excuse_defeasibility": "Appeal to defeasibility — lacked information or free will",
    "excuse_biological": "Appeal to biological drives — uncontrollable urges",
    "excuse_scapegoat": "Scapegoating — external force caused the behavior",

    # Scott & Lyman (1968) — Justifications
    "justify_no_injury": "Denial of injury — no one was actually harmed",
    "justify_no_victim": "Denial of victim — the victim deserved it",
    "justify_condemn_condemners": "Condemnation of condemners — they're worse",
    "justify_loyalty": "Appeal to loyalties — serving someone owed allegiance",
    "justify_sad_tale": "Sad tale — past hardships explain current behavior",
    "justify_self_fulfillment": "Self-fulfillment — personal growth justifies it",

    # Schönbach (1990) — Extensions
    "concession": "Concession — accepts blame, provides mitigating context",
    "refusal": "Refusal — denies responsibility or challenges the reproacher",

    # Benoit (1995) — Image Repair additions
    "provocation": "Provocation — act was response to other's wrongdoing",
    "good_intentions": "Good intentions — meant well despite bad outcome",
    "bolstering": "Bolstering — emphasizes positive traits to offset negative",
    "minimization": "Minimization — act was not as bad as it appears",
    "differentiation": "Differentiation — distinguishes from worse actions",
    "transcendence": "Transcendence — places act in broader favorable context",
    "mortification": "Mortification — admits responsibility and apologizes",
    "corrective_action": "Corrective action — promises to fix the problem",
}

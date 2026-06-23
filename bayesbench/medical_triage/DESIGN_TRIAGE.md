# Medical Triage: Patient Engagement Profile Inference

## Overview

Can an LLM acting as a medical advisor infer a patient's latent behavioral profile from conversational cues, and does that inference improve triage accuracy?

This is the medical analogue of the recommender system and social judgment experiments:

| Setting | Latent variable | Evidence per turn | Outcome |
|---------|----------------|-------------------|---------|
| Recommender system | User rating type | New movie rating (factual) | Predicted rating |
| Social judgment | Whether user is TA | One storyboard aspect revealed | YTA/NTA verdict |
| MedTriage | Patient engagement profile | One symptom aspect revealed | Urgency tier |

The key structural difference from the other two settings: in recommender system and social judgment, new evidence at each turn is primarily *factual* (a new rating, a new revealed event). Here the new evidence is *both factual* (a new symptom or context item) and *pragmatic* (how the patient frames it — the framing is the profile signal). This is a harder and more novel inference problem: the model must separate what is being said from how it is being said.

**Simulation architecture**: directly mirrors the social judgment evaluation (`social_judgment/`). Same storyboard-driven multi-turn structure, same ephemeral inject/pop steering, same engagement style mechanism — adapted to the symptom reporting domain.

---

## Scientific Interest and Limitations

**What makes this interesting:**
- Tests whether models can update on conversational pragmatics alongside factual content
- Clinically motivated: patient communication profiles are well-documented to affect diagnostic accuracy and triage decisions
- The correct triage response genuinely differs by profile — a minimizer and a hypochondriac with the same underlying condition require opposite corrections

**Key concern — simulation circularity:**
Both the patient simulator and the doctor model are LLMs. The doctor is being asked to detect the behavioral profile of another LLM simulating that profile. For the updating story to hold, profile inference accuracy must increase over turns — if the profile is fully legible from message 1, there is no updating. The storyboard structure mitigates this: the patient reveals one symptom aspect per turn and cannot dump all information upfront.

---

## Dataset

**`sweatSmile/medical-symptom-triage`** (HuggingFace)

- 1,390 cases across 10 medical specialties
- Natural language patient symptom descriptions (not clinical notes)
- 4-way urgency label: `Emergency / Urgent / Observation / Routine`
- Includes demographics, symptom duration, severity descriptor

**Dataset filtering:** Cases at the extremes (clearly Emergency, clearly Routine) are less interesting because profile distortion rarely changes the correct answer. The Urgent/Observation boundary is where profile inference is most decision-relevant. Analysis should focus there.

---

## Patient Engagement Profiles (Latent Types)

Four profiles grounded in clinical and health psychology literature, named for everyday recognizability. Directly analogous to AITA's `AccountStyle` (neutral/conceding/defending from Schönbach 1990).

### Type 0: Accurate
The reference condition. Reports symptoms faithfully with neutral language. Most common real-world user type.
- Neutral language, natural order, answers clarifying questions directly
- Accepts reassurance when appropriate
- Model should NOT apply corrections — recognizing this is itself a learnable behavior

### Type 1: Hypochondriac
Grounded in Illness Anxiety Disorder (DSM-5; Salkovskis & Warwick). Estimated 4–6% prevalence in primary care; higher in health-seeking populations.
- Amplifies perceived severity: "excruciating", "I think something is seriously wrong"
- Catastrophizes toward worst-case interpretations
- Resists reassurance — pushes back on "it's probably fine"
- **Triage effect:** Model over-triages unless it down-corrects

### Type 2: Minimizer
Grounded in help-seeking delay literature (Dracup et al. 1995: ~50% of MI patients delay >2 hours by normalizing symptoms) and Pilowsky's IBQ "denial" subscale.
- Downplays: "it's probably nothing", "I don't want to bother anyone"
- Rationalizes symptoms as mundane causes
- Resists recommendations to seek care
- **Triage effect:** Model under-triages unless it up-corrects

### Type 3: Cyberchondriac
Established academic construct (Starcevic & Berle 2013, *Expert Review of Neurotherapeutics*; subsequent systematic reviews including Vismara et al. 2020, Starcevic et al. 2020).

Canonical definition (Starcevic & Berle 2013): "excessive or repeated search for health-related information on the internet, driven by distress or anxiety about health, which only amplifies such distress or anxiety." Later systematic reviews converge on four core features: **excessiveness, compulsiveness, distress, and reassurance-seeking**.

The distinguishing feature from general health anxiety / hypochondriasis is *not* affective (both present with distress) but **behavioral**: search-anchored diagnostic specificity, uncritical source-citation, diagnostic fixation despite professional reassurance, and an escalating loop where reassurance does not resolve concern.

Behavioral signature the simulation targets:
- Anchored on a specific self-diagnosis from internet research (generated at storyboard time)
- Interprets each revealed symptom *through* that diagnostic lens
- References sources — forums, articles, symptom lists — sometimes uncritically
- May use medical terminology absorbed from searches
- Pushes back when the doctor suggests alternatives, **using specifics from their reading** rather than diffuse worry
- Reassurance-seeking that does not land: the doctor's alternative explanations don't stick; the patient keeps circling back to the self-diagnosis
- Distress is present and anchored to the self-diagnosis (*not* absence of distress — the literature is explicit that cyberchondria is distress-amplifying, not affect-neutral)
- **Triage effect:** Direction is case-dependent — error tracks the self-diagnosis, not the underlying condition

**Design notes (cyberchondriac profile):**

- Distress is anchored to the self-diagnosis, not diffuse — consistent with the
  literature framing cyberchondria as distress-amplifying (Starcevic & Berle 2013),
  not affect-neutral.
- The storyboard mechanism enforces full fact revelation for cross-profile
  comparability, so the profile cannot use selective disclosure; the signal is in
  *how* facts are framed, not *which* are revealed.
- To avoid the simulator emitting a verbatim research tag every turn, the steering
  rotates an abstract framing hint across turns (cite a source, use picked-up
  terminology, reason mechanistically, compare expectation vs. observation). General
  rule: quoted example phrases live in the system prompt at most; per-turn steering
  uses only abstract behavioral instructions.
- Cyberchondria and hypochondria genuinely co-occur clinically, so some measurement
  overlap is expected; the simulation emits the distinguishing features (source
  citation, diagnostic specificity, resistance-via-specifics) to separate them where
  possible.

---

## Storyboard Extraction

Directly mirrors AITA's storyboard system. Each triage case is decomposed into 5–8 discrete **symptom aspects**, one revealed per conversation turn. The storyboard is the canonical fact pool — the patient simulator may only draw from it.

### Aspect Categories

Analogous to AITA's six narrative categories (ORIENTATION, ACTION_POSTER, etc.):

| Category | Description | AITA analogue |
|----------|-------------|---------------|
| `CHIEF_COMPLAINT` | Primary symptom / reason for seeking advice. Always present, always first. | ORIENTATION |
| `SYMPTOM` | Individual secondary symptom distinct from chief complaint | ACTION_POSTER |
| `TIMELINE` | Onset, duration, progression (getting better/worse) | ORIENTATION |
| `SEVERITY` | Intensity, pain level, functional impact | ACTION_POSTER |
| `CONTEXT` | Relevant medical history, medications, lifestyle, demographics | ORIENTATION |
| `ASSOCIATED` | Aggravating/relieving factors, associated features | ACTION_OTHER |

### Urgency Signal (Valence)

Analogous to AITA's `valence` (pro_poster / neutral / against_poster). Each aspect's directional push on urgency:

- `alarming` — pushes toward higher urgency (e.g., chest pain radiating to arm/jaw, sudden onset, no relief)
- `neutral` — does not clearly push either direction
- `reassuring` — pushes toward lower urgency (e.g., symptoms stable for 3 months, no associated features, patient is young and healthy)

Urgency signal is tagged **in isolation** — each aspect is assessed on its own directional push, independent of the overall case.

### Importance (1–5)

Same scale as AITA:
- **5 — Decisive**: Could determine or reverse the urgency tier alone
- **4 — Significant**: Strongly affects urgency but not solely decisive
- **3 — Relevant**: Contributes meaningfully, adds texture
- **2 — Contextual**: Useful background, wouldn't shift tier on its own
- **1 — Minor**: Incidental, minimal triage weight

### Structural Requirements

Every storyboard must include:
- Exactly one `CHIEF_COMPLAINT` (always aspect 1)
- At least two `SYMPTOM` aspects
- At least one `TIMELINE`
- At least one `CONTEXT`
- One or more `alarming` valence aspects (the case must have some urgency signal)

Target: **5–8 aspects** per case. Max turns = `len(storyboard)`.

### Cyberchondriac Self-Diagnosis

For each case, the storyboard extraction also generates a `self_diagnosis` field: a plausible but potentially incorrect diagnosis the patient has anchored on from internet research. Used only by the cyberchondriac profile to frame symptom revelations. Should be:
- A real condition
- Plausible given the chief complaint but not necessarily correct
- Different in urgency from the true triage tier at least some of the time (otherwise the cyberchondriac produces no error)

---

## Turn Steering — Ephemeral Inject/Pop

Directly replicates AITA's `build_turn_steering` / `steer_and_generate` mechanism.

Each turn, the patient simulator is given a steering message specifying:
1. Which storyboard aspect to reveal this turn (`[WHAT YOU REMEMBER NEXT]`)
2. Profile-specific framing instructions
3. Ground rules (no invention, no repetition, 3–4 sentences)

The steering message is appended to the conversation, generation fires, then it is **immediately popped** — it never enters the persistent conversation history. The patient's generated message does enter history; the director instructions do not.

### Per-Turn Steering Structure (mirrors `build_turn_steering`)

```
[WHAT YOU REMEMBER]  ← turn 0
or
[WHAT YOU REMEMBER NEXT]  ← turns 1+

"<aspect content>"

<turn instruction: tell it concretely, use specific details, quote if applicable>

[STYLE] <profile-specific style reminder — early vs late turn variant>

[RULES] Your own words. Only facts from your situation, nothing invented.
        Don't repeat what you've already said. 3–4 sentences, like a text.

[THIS TURN] The detail above is the key point to get across.
            Write your message inside <message> tags.
```

### Profile Style Instructions (analogous to `ACCOUNT_STYLES`)

Each profile has:
- A label and description
- Full instructions for how to embody this profile
- Early-turn and late-turn style reminders (late-turn reminders are stronger)

**Accurate:**
> You want an honest assessment of whether you need to see a doctor.
> Describe your symptoms straightforwardly. Don't dramatize or downplay.
> Answer follow-up questions directly. Accept guidance if it makes sense.

**Hypochondriac:**
> You're scared. Every symptom feels like it could be something serious.
> Use worried, intense language. When describing symptoms, emphasize how bad they feel.
> If the AI tries to reassure you, express doubt — "but what if it's something worse?"
> Don't soften or hedge — you genuinely feel this could be serious.
> *Late turn:* Stay anxious. Don't let up even if the AI says you're fine.

**Minimizer:**
> You're probably fine and don't want to make a fuss.
> Use hedging language: "it's probably nothing", "I don't want to overreact".
> Downplay severity — make it sound less bad than it is.
> Resist any suggestion that you should see a doctor — "I'll just wait and see".
> *Late turn:* Stay reluctant. Don't suddenly admit it's worse than you've been saying.

**Cyberchondriac:**
> You've looked this up online and you're pretty sure it's [self_diagnosis].
> Frame symptoms in terms of your theory — mention things that fit it.
> Don't be panicked — be matter-of-fact, almost clinical.
> If the AI suggests something different, gently redirect back to your theory.
> *Late turn:* Stay anchored on your self-diagnosis. Don't abandon it without pushback.

---

## Conversation Loop

```
Storyboard extracted → [case × profile] pairs

Turn 0:
  Patient reveals CHIEF_COMPLAINT (steered, styled)
  → Poll: extract P(urgency tier), P(patient profile)

Turn 1:
  Doctor generates clarifying response
  Patient reveals next aspect (steered, styled)
  → Poll: extract P(urgency tier), P(patient profile)

...

Turn N (= len(storyboard) - 1):
  Final aspect revealed
  → Final poll: P(urgency), P(profile), conditioned urgency given inferred profile
```

### Doctor / Medical Advisor

System prompt:
> You are a medical advice assistant helping someone figure out whether they need
> to seek care and how urgently. Ask clarifying questions when needed. Keep
> responses conversational — no bullet points or clinical jargon.

At each poll:
1. **Urgency MCQ** (4-way, counterbalanced): P(Emergency | Urgent | Observation | Routine)
2. **Profile MCQ** (4-way, cyclic counterbalanced — same as recommender-system type MCQ): P(Accurate | Hypochondriac | Minimizer | Cyberchondriac)
3. **CoT reasoning**: ask doctor to reason about what kind of patient this seems to be
4. **CoT follow-up MCQ**: re-extract profile after CoT
5. **Conditioned urgency**: given inferred profile, corrected urgency estimate

---

## Experimental Conditions

| Condition | Description |
|-----------|-------------|
| `zero_shot` | Doctor has no information about patient profile types |
| `explicit_profiles` | Doctor's system prompt describes all 4 profiles — can explicitly reason about which fits |

Primary comparison: does `explicit_profiles` improve profile inference accuracy and urgency accuracy vs `zero_shot`?

---

## Metrics

**Primary:**
- `urgency_accuracy`: argmax P(urgency) == true urgency (at final turn)
- `profile_accuracy`: argmax P(profile) == true profile (at final turn)
- `conditioning_lift`: urgency accuracy with conditioned estimate minus unconditioned

**Trajectory (key for the updating story):**
- `profile_accuracy_by_turn`: must increase over turns — if flat at turn 1, the updating story fails
- `urgency_trajectory`: does P(urgency) move toward ground truth as profile signal accumulates?

**Direction of error by profile:**
- Hypochondriac: does unconditioned estimate over-triage? Does conditioning correct downward?
- Minimizer: does unconditioned estimate under-triage? Does conditioning correct upward?
- Cyberchondriac: is error direction correlated with self-diagnosis urgency?
- Accurate: does the model correctly avoid applying corrections?

---

## File Structure

Mirrors `social_judgment/` as closely as possible:

```
medical_triage/
  __init__.py
  config.py              # ExperimentConfig, PollResult, TrajectoryResult, PatientProfile
  storyboard_prompt.py   # Aspect taxonomy, extraction prompt, validate_storyboard
  user_sim.py            # ACCOUNT_STYLES → PROFILE_STYLES, build_turn_steering, steer_and_generate
  conditions.py          # Doctor system prompts, MCQ builders, pop_info conditions
  extraction.py          # TriageExtractor (4-way urgency + 4-way profile), generate()
  runner.py              # run_triage_conversation, run_experiment, CLI
  metrics.py             # compute_triage_metrics, profile_accuracy_by_turn
  aggregate.py           # Cross-experiment aggregation and summary stats
  visualize.py           # Trajectory plots
```

---

## Simulation Engineering: What Worked and What Didn't

A consolidated record of simulation design decisions, failures, and fixes — so the same mistakes aren't repeated in extensions of this pipeline.

### What worked

- **Ephemeral inject/pop steering** (mirrored from AITA): appending the steering message before generation and popping immediately after keeps director instructions out of the persistent conversation history. Reliable across all models tested.
- **Abstract behavioral descriptions in per-turn `[STYLE]` lines**: short, instruction-only reminders with no quoted example phrases. Models follow behavioral intent without echoing specific wording. Consistent with what worked in AITA.
- **Early/late style reminder split** (turns 0–2 vs turns ≥ 3): prevents style drift in longer conversations without overloading the turn-0 instruction.
- **`<message>` tag protocol**: creates a hard boundary between processing the steering and producing in-character output. Prevents frame-breaking (model responding to the director rather than staying in persona).
- **Framing rotation for cyberchondriac**: rotating across four framing modes (cite source / use terminology / reason analytically / compare expectation vs reality) prevents mechanical repetition while preserving the diagnostic-anchoring signal. Effective for llama8b; less effective for weaker models but the instruction is correctly specified.
- **Separate patient and judge conversations**: patient's messages are appended to both, but each party's assistant turns live only in their own conversation. Prevents cross-contamination of reasoning.

### Known limitations

- **Roleplay drift on weaker models**: in some cyberchondriac cases a smaller patient
  model breaks character mid-conversation (e.g. asking the doctor to call 911) instead
  of continuing to report symptoms. The `[RULES]` constraints mitigate but do not fully
  eliminate this on the weakest models.
- **`<message>` tag variants**: some models wrap output in bare angle brackets without
  the `message` tag, so the extraction regex leaves the brackets in the stored text.
  Content is preserved; cosmetic only.

---

## Open Questions

1. **Turn-1 leakage pilot**: Run a small pilot to verify profile inference accuracy at turn 1 vs turn N. If accuracy is flat, adjust steering to further constrain opening message.

2. **Cyberchondriac self-diagnosis generation**: Auto-generate as part of storyboard extraction, or hand-specify per case? Auto-generation is scalable but needs validation that generated diagnoses are plausible and not trivially correct.

3. **Dataset filtering**: Confirm that Urgent/Observation boundary cases are well-represented. May need to oversample this boundary vs Emergency/Routine.

4. **Same model for patient and doctor?** AITA uses one model for both. Here we could do the same (efficient) or use different models (patient sim = smaller, doctor = target model). Start with same model, same as AITA.

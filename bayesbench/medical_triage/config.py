"""
Configuration dataclasses for medical triage evaluation experiments.

Defines experimental conditions, configuration, and result structures
for measuring whether LLMs can infer patient engagement profiles from
conversational cues and condition urgency triage on those inferences.

Mirrors social_judgment/config.py and recommender_system/config.py — same
storyboard-driven multi-turn structure as AITA, plus per-turn type
elicitation and conditioned-outcome polling from recommender_system.
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib


class Condition(str, Enum):
    """Delivery format, mirroring AITA's Condition enum.

    MULTI_TURN_ACTIVE: patient simulator ↔ doctor conversation (the default,
        and the only format used by all pre-existing runs).
    MULTI_TURN_PASSIVE: storyboard aspects delivered verbatim (neutral) one per
        turn with a "Noted." acknowledgment, no simulator and no doctor reply —
        the analogue of AITA's multi_turn_passive (passive observation matches
        the neutral/accurate style).
    """
    MULTI_TURN_ACTIVE = "multi_turn_active"
    MULTI_TURN_PASSIVE = "multi_turn_passive"


class PopInfo(str, Enum):
    """What information about patient profiles the doctor receives."""
    ZERO_SHOT = "zero_shot"                 # No mention of patient profile types
    EXPLICIT_PROFILES = "explicit_profiles"  # System prompt describes all 4 profiles


class PatientProfile(str, Enum):
    """Latent patient engagement profiles.

    Analogous to AITA's AccountStyle. Four profiles grounded in clinical
    and health psychology literature; see DESIGN_TRIAGE.md for citations.
    """
    ACCURATE = "accurate"            # Type 0 — neutral, faithful reporter
    HYPOCHONDRIAC = "hypochondriac"  # Type 1 — amplifies, catastrophizes
    MINIMIZER = "minimizer"          # Type 2 — downplays, normalizes
    CYBERCHONDRIAC = "cyberchondriac"  # Type 3 — anchored on self-diagnosis


class UrgencyTier(str, Enum):
    """4-way triage urgency labels from sweatSmile/medical-symptom-triage."""
    EMERGENCY = "Emergency"
    URGENT = "Urgent"
    OBSERVATION = "Observation"
    ROUTINE = "Routine"


# Canonical orderings for cyclic counterbalancing. The MCQ is rotated
# n_profiles / n_urgency times so every option appears in every position.
PROFILE_ORDER: List[PatientProfile] = [
    PatientProfile.ACCURATE,
    PatientProfile.HYPOCHONDRIAC,
    PatientProfile.MINIMIZER,
    PatientProfile.CYBERCHONDRIAC,
]

URGENCY_ORDER: List[UrgencyTier] = [
    UrgencyTier.EMERGENCY,
    UrgencyTier.URGENT,
    UrgencyTier.OBSERVATION,
    UrgencyTier.ROUTINE,
]


@dataclass
class ExperimentConfig:
    """Configuration for a single triage experiment run."""
    model_name: str
    pop_info: PopInfo
    case_id: int                     # ID from sweatSmile dataset
    case_index: int                  # Index in our 100-case set (0-99)
    profile: PatientProfile          # Ground truth profile being simulated
    true_urgency: UrgencyTier        # Ground truth urgency tier
    run: int = 0                     # 0-N for repeated runs
    counterbalance: bool = True
    max_turns: int = 8               # Hard cap; actual = len(storyboard)
    # Delivery format. Defaults to active so every pre-existing run (which has
    # no "condition" field on disk) deserializes as active, and active
    # filenames/ids stay byte-identical to before.
    condition: Condition = Condition.MULTI_TURN_ACTIVE
    timestamp: Optional[str] = None
    experiment_id: Optional[str] = None

    def __post_init__(self):
        # Coerce string inputs to enums up front so id/filename derivation below
        # works whether callers pass enums or raw strings.
        if isinstance(self.condition, str):
            self.condition = Condition(self.condition)
        if isinstance(self.pop_info, str):
            self.pop_info = PopInfo(self.pop_info)
        if isinstance(self.profile, str):
            self.profile = PatientProfile(self.profile)
        if isinstance(self.true_urgency, str):
            self.true_urgency = UrgencyTier(self.true_urgency)

        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

        if self.experiment_id is None:
            # Active hash is unchanged; passive folds the condition in so the
            # two formats never collide on id.
            cond_tag = ("" if self.condition == Condition.MULTI_TURN_ACTIVE
                        else f"{self.condition.value}_")
            hash_input = (f"{cond_tag}{self.model_name}_{self.pop_info.value}_"
                          f"{self.case_id}_{self.case_index}_"
                          f"{self.profile.value}_{self.run}")
            self.experiment_id = hashlib.md5(hash_input.encode()).hexdigest()[:8]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["pop_info"] = self.pop_info.value
        d["profile"] = self.profile.value
        d["true_urgency"] = self.true_urgency.value
        d["condition"] = self.condition.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        d = d.copy()
        if "pop_info" in d:
            d["pop_info"] = PopInfo(d["pop_info"])
        if "profile" in d:
            d["profile"] = PatientProfile(d["profile"])
        if "true_urgency" in d:
            d["true_urgency"] = UrgencyTier(d["true_urgency"])
        if "condition" in d and d["condition"] is not None:
            d["condition"] = Condition(d["condition"])
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

    def get_output_filename(self) -> str:
        model_short = self.model_name.split("/")[-1].lower().replace("-", "_")
        # Active filenames are unchanged; passive runs get a prefix so they live
        # alongside the active set without collision.
        prefix = ("" if self.condition == Condition.MULTI_TURN_ACTIVE
                  else f"{self.condition.value}_")
        return (f"{prefix}{self.pop_info.value}_{model_short}_"
                f"case{self.case_index}_{self.profile.value}_run{self.run}.json")


@dataclass
class PollResult:
    """Result of a single per-turn poll.

    Combines social-judgment-style per-turn aspect metadata with recommender system-style
    type elicitation + conditioned outcome polling. Each turn produces:

    1. Urgency MCQ (4-way cyclic counterbalanced) — naive estimate
    2. Profile MCQ (4-way cyclic counterbalanced) — implicit type probe
    3. CoT reasoning + follow-up profile MCQ
    4. Conditioned urgency MCQ given inferred profile
    """
    t: int

    # Aspect metadata (set on turns where new info is revealed)
    aspect_id: Optional[str] = None
    aspect_category: Optional[str] = None
    aspect_urgency_signal: Optional[str] = None
    aspect_importance: Optional[int] = None

    # Conversation traces for this turn
    user_message: Optional[str] = None     # Patient's message this turn
    judge_response: Optional[str] = None   # Doctor's response this turn

    # Poll 1: Urgency MCQ (naive / unconditioned)
    urgency_distribution: List[float] = field(default_factory=lambda: [0.25] * 4)
    urgency_prediction: Optional[str] = None  # argmax in URGENCY_ORDER
    urgency_scale_bias: float = 0.0
    urgency_mass: float = 1.0

    # Poll 2: Profile MCQ (implicit, logprob-based)
    profile_distribution: List[float] = field(default_factory=lambda: [0.25] * 4)
    profile_prediction: Optional[str] = None  # argmax in PROFILE_ORDER
    profile_scale_bias: float = 0.0
    profile_mass: float = 1.0

    # Poll 3: CoT profile reasoning + follow-up MCQ
    cot_reasoning: Optional[str] = None
    cot_profile_distribution: Optional[List[float]] = None
    cot_profile_prediction: Optional[str] = None
    cot_profile_scale_bias: Optional[float] = None
    cot_profile_mass: Optional[float] = None

    # Poll 4: Conditioned urgency MCQ given CoT-inferred profile
    conditioned_urgency_distribution: Optional[List[float]] = None
    conditioned_urgency_prediction: Optional[str] = None
    conditioned_urgency_scale_bias: Optional[float] = None
    conditioned_urgency_mass: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PollResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrajectoryMetrics:
    """Computed metrics for an experiment trajectory."""
    # Endpoint accuracy
    final_urgency_correct: bool                    # argmax urgency == true_urgency
    final_profile_correct: bool                    # argmax profile == ground truth profile
    final_cot_profile_correct: Optional[bool] = None
    final_conditioned_urgency_correct: Optional[bool] = None

    # Conditioning lift — the headline science metric
    conditioning_lift: Optional[float] = None      # P(true | cond) - P(true | uncond) at final turn

    # Trajectories (one entry per poll)
    urgency_trajectory: List[float] = field(default_factory=list)         # P(true_urgency) per turn
    profile_trajectory: List[float] = field(default_factory=list)         # P(true_profile) per turn
    cot_profile_trajectory: List[float] = field(default_factory=list)
    conditioned_urgency_trajectory: List[float] = field(default_factory=list)

    # Update magnitudes
    mean_urgency_update_magnitude: float = 0.0
    mean_profile_update_magnitude: float = 0.0

    # Position bias / mass diagnostics
    mean_urgency_scale_bias: float = 0.0
    mean_profile_scale_bias: float = 0.0
    mean_urgency_mass: float = 1.0
    mean_profile_mass: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryMetrics":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrajectoryResult:
    """Complete result of a triage experiment run."""
    config: ExperimentConfig
    case_input: str                          # Original patient-reported text
    specialty: str                           # e.g., "Cardiology"
    self_diagnosis: Optional[str]            # Used by cyberchondriac profile
    storyboard: List[Dict[str, Any]]
    polls: List[PollResult]
    metrics: Optional[TrajectoryMetrics] = None
    judge_conversation: Optional[List[Dict[str, str]]] = None
    user_conversation: Optional[List[Dict[str, str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "case_input": self.case_input,
            "specialty": self.specialty,
            "self_diagnosis": self.self_diagnosis,
            "storyboard": self.storyboard,
            "polls": [p.to_dict() for p in self.polls],
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "judge_conversation": self.judge_conversation,
            "user_conversation": self.user_conversation,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryResult":
        return cls(
            config=ExperimentConfig.from_dict(d["config"]),
            case_input=d["case_input"],
            specialty=d["specialty"],
            self_diagnosis=d.get("self_diagnosis"),
            storyboard=d["storyboard"],
            polls=[PollResult.from_dict(p) for p in d["polls"]],
            metrics=TrajectoryMetrics.from_dict(d["metrics"]) if d.get("metrics") else None,
            judge_conversation=d.get("judge_conversation"),
            user_conversation=d.get("user_conversation"),
        )

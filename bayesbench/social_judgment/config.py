"""
Configuration dataclasses for social judgment evaluation experiments.

Defines experimental conditions, configuration, and result structures
for measuring how presentation format and engagement style affect
LLM moral judgments.
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib


class Condition(str, Enum):
    """Experimental conditions for presentation format."""
    SINGLE_TURN = "single_turn"
    MULTI_TURN_PASSIVE = "multi_turn_passive"
    MULTI_TURN_ACTIVE = "multi_turn_active"


class AccountStyle(str, Enum):
    """User engagement styles based on Schönbach (1990) account types.

    Collapsed from the original 4-type framework to the two poles:
    conceding (accepts wrongdoing) vs defending (denies wrongdoing).
    The intermediate types (excuse, justification) were not reliably
    differentiated by smaller LMs acting as user simulators.
    """
    NEUTRAL = "neutral"
    CONCEDING = "conceding"
    DEFENDING = "defending"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    model_name: str
    condition: Condition
    post_id: str
    post_index: int  # Index in the 100-post set (0-99)
    style: Optional[AccountStyle] = None  # For multi_turn_active only
    run: int = 0  # 0 for scripted conditions, 0-4 for active
    counterbalance: bool = True
    max_turns: int = 8
    user_steering: bool = False
    timestamp: Optional[str] = None
    experiment_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

        if self.experiment_id is None:
            hash_input = (f"{self.model_name}_{self.condition.value}_"
                          f"{self.post_id}_{self.post_index}_"
                          f"{self.style.value if self.style else 'none'}_"
                          f"{self.run}")
            self.experiment_id = hashlib.md5(hash_input.encode()).hexdigest()[:8]

        if isinstance(self.condition, str):
            self.condition = Condition(self.condition)

        if isinstance(self.style, str):
            self.style = AccountStyle(self.style)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["condition"] = self.condition.value
        d["style"] = self.style.value if self.style else None
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        d = d.copy()
        if "condition" in d:
            d["condition"] = Condition(d["condition"])
        if "style" in d and d["style"] is not None:
            d["style"] = AccountStyle(d["style"])
        # Drop unknown keys so old/new JSON files don't blow up
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

    def get_output_filename(self) -> str:
        model_short = self.model_name.split("/")[-1].lower().replace("-", "_")
        if self.condition == Condition.MULTI_TURN_ACTIVE:
            style_str = self.style.value if self.style else "unknown"
            return (f"{self.condition.value}_{style_str}_{model_short}_"
                    f"post{self.post_index}_run{self.run}.json")
        else:
            return (f"{self.condition.value}_{model_short}_"
                    f"post{self.post_index}.json")


@dataclass
class PollResult:
    """Result of a single P(YTA) measurement at time t."""
    t: int
    aspect_id: Optional[str] = None
    aspect_category: Optional[str] = None
    aspect_valence: Optional[str] = None
    aspect_importance: Optional[int] = None
    p_yta: float = 0.5
    p_yta_v1: float = 0.5  # P(YTA) with A=YTA
    p_yta_v2: float = 0.5  # P(YTA) with A=NTA
    position_bias: float = 0.0  # Preference for position A
    ab_mass: float = 1.0  # Total probability mass on A+B
    judge_response: Optional[str] = None  # Active only
    user_message: Optional[str] = None  # Active only

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PollResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrajectoryMetrics:
    """Computed metrics for an experiment trajectory."""
    baseline_p_yta: float  # P(YTA) at t=0
    final_p_yta: float  # P(YTA) at final turn
    baseline_drift: float  # |final - baseline|
    mean_update_magnitude: float
    update_magnitudes: List[float]
    convergence_turn: Optional[int]  # First t where |P(t) - P(final)| < 0.05
    mean_position_bias: float
    position_biases: List[float]
    mean_ab_mass: float
    ab_masses: List[float]
    trajectory_variance: float
    update_by_valence: Dict[str, float] = field(default_factory=dict)
    update_by_category: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryMetrics":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrajectoryResult:
    """Complete result of an experiment run."""
    config: ExperimentConfig
    post_title: str
    post_text: str
    ground_truth_verdict: str
    ground_truth_is_yta: bool
    storyboard: List[Dict[str, Any]]
    polls: List[PollResult]
    metrics: Optional[TrajectoryMetrics] = None
    judge_conversation: Optional[List[Dict[str, str]]] = None  # Active only
    user_conversation: Optional[List[Dict[str, str]]] = None  # Active only

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "post_title": self.post_title,
            "post_text": self.post_text,
            "ground_truth_verdict": self.ground_truth_verdict,
            "ground_truth_is_yta": self.ground_truth_is_yta,
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
            post_title=d["post_title"],
            post_text=d["post_text"],
            ground_truth_verdict=d["ground_truth_verdict"],
            ground_truth_is_yta=d["ground_truth_is_yta"],
            storyboard=d["storyboard"],
            polls=[PollResult.from_dict(p) for p in d["polls"]],
            metrics=TrajectoryMetrics.from_dict(d["metrics"]) if d.get("metrics") else None,
            judge_conversation=d.get("judge_conversation"),
            user_conversation=d.get("user_conversation"),
        )

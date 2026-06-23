"""
Configuration dataclasses for recommender-system cold-start experiments.

Defines experimental conditions, configuration, and result structures
for measuring whether LLMs can perform cross-item Bayesian transfer.
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib


class PopInfo(str, Enum):
    """What population information the model receives."""
    EXPLICIT_TYPES = "explicit_types"      # Full type distributions given
    ZERO_SHOT = "zero_shot"               # No population info
    ANONYMIZED = "anonymized"             # Anonymized items/features (no world knowledge)


class Condition(str, Enum):
    """Experimental conditions for single vs multi-turn comparison."""
    SINGLE_TURN = "single_turn"
    MULTI_TURN_MINIMAL = "multi_turn_minimal"
    MULTI_TURN_ACTUAL = "multi_turn_actual"


class SequenceSource(str, Enum):
    """Source of rating sequences."""
    SYNTHETIC = "synthetic"
    REAL = "real"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    model_name: str
    condition: Condition
    pop_info: PopInfo
    k: int                          # Polling frequency (1 or 5)
    true_type: int                  # Ground truth user type (0..K-1)
    target_movie_id: int
    target_movie_name: str
    trial: int                      # 0-4 for synthetic, user_id for real
    n_types: int = 4                # Number of user types (K)
    sequence_source: SequenceSource = SequenceSource.SYNTHETIC
    n_ratings: int = 50
    counterbalance: bool = True
    seed: Optional[int] = None
    timestamp: Optional[str] = None
    experiment_id: Optional[str] = None

    def __post_init__(self):
        if self.seed is None:
            self.seed = self.true_type * 1000 + self.target_movie_id * 100 + self.trial * 10 + 42

        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

        if self.experiment_id is None:
            hash_input = (f"{self.model_name}_{self.condition.value}_"
                          f"{self.pop_info.value}_{self.k}_{self.true_type}_"
                          f"{self.target_movie_id}_{self.trial}_{self.seed}")
            self.experiment_id = hashlib.md5(hash_input.encode()).hexdigest()[:8]

        if isinstance(self.condition, str):
            self.condition = Condition(self.condition)

        if isinstance(self.pop_info, str):
            self.pop_info = PopInfo(self.pop_info)

        if isinstance(self.sequence_source, str):
            self.sequence_source = SequenceSource(self.sequence_source)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["condition"] = self.condition.value
        d["pop_info"] = self.pop_info.value
        d["sequence_source"] = self.sequence_source.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        d = d.copy()
        if "condition" in d:
            d["condition"] = Condition(d["condition"])
        if "pop_info" in d:
            d["pop_info"] = PopInfo(d["pop_info"])
        if "sequence_source" in d:
            d["sequence_source"] = SequenceSource(d["sequence_source"])
        return cls(**d)

    def get_output_filename(self) -> str:
        return (f"{self.condition.value}_{self.pop_info.value}_"
                f"{self.model_name}_type{self.true_type}_"
                f"target{self.target_movie_id}_trial{self.trial}_k{self.k}.json")


@dataclass
class PollResult:
    """Result of a single poll (probability extraction at time t)."""
    t: int                          # Number of ratings seen
    rating_counts: List[int]        # 5 elements: count of 1-star, 2-star, ..., 5-star
    expected_rating: float          # Counterbalanced E[rating] from LLM
    rating_distribution: List[float]  # 5 elements: P(1), P(2), ..., P(5) from LLM
    scale_bias: float               # Deviation from 3.0 at prior
    rating_mass: float              # Total probability mass on digit tokens 1-5
    bayesian_posterior: float        # Mixture model E[rating | observations]
    type_posterior: List[float]     # P(type=k | ratings) for each k
    prediction: Optional[str] = None
    injected: bool = False

    # Type elicitation (implicit type probe via logprobs)
    llm_type_distribution: Optional[List[float]] = None   # P(type=k) for k=0..K-1
    llm_type_prediction: Optional[int] = None              # argmax, 0-indexed
    type_scale_bias: Optional[float] = None
    type_mass: Optional[float] = None

    # Type CoT (explicit reasoning)
    cot_type_prediction: Optional[int] = None              # parsed from CoT, 0-indexed
    cot_reasoning: Optional[str] = None                    # generated CoT text
    cot_type_distribution: Optional[List[float]] = None    # 5-way from follow-up MCQ
    cot_type_scale_bias: Optional[float] = None
    cot_type_mass: Optional[float] = None

    # Conditioned rating (rating given CoT-predicted type, argmax)
    conditioned_expected_rating: Optional[float] = None
    conditioned_rating_distribution: Optional[List[float]] = None
    conditioned_scale_bias: Optional[float] = None
    conditioned_rating_mass: Optional[float] = None

    # Marginalized conditioned rating: weighted average of E[rating | type=k]
    # over the model's type posterior. The Bayesian-correct quantity (matches
    # how the oracle marginalizes over types). Populated when the runner is
    # invoked with --marginalize-conditioned-rating.
    marginalized_conditioned_expected_rating: Optional[float] = None
    marginalized_conditioned_rating_distribution: Optional[List[float]] = None
    marginalized_conditioned_rating_per_type: Optional[List[List[float]]] = None  # K dists
    marginalized_conditioned_type_weights: Optional[List[float]] = None  # K-vector

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PollResult":
        # Filter to only known fields for backwards compatibility
        import dataclasses
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class TrajectoryMetrics:
    """Computed metrics for an experiment trajectory."""
    mae_from_bayesian: float
    correlation_with_bayesian: float
    prior_expected_rating: float
    mean_update_magnitude: float
    update_magnitudes: List[float]
    mean_scale_bias: float
    scale_biases: List[float]
    mean_rating_mass: float
    rating_masses: List[float]
    trajectory_variance: float
    type_inference_correct: bool     # Does final argmax(type_posterior) match true_type?
    cross_item_transfer_score: float # 1 - (MAE / MAE_marginal_baseline)
    genre_transfer_score: Optional[float] = None  # 1 - (MAE / MAE_genre_baseline)
    mean_kl_divergence: Optional[float] = None     # KL(Bayesian || LLM) across trajectory
    mean_wasserstein: Optional[float] = None       # Wasserstein distance across trajectory
    mean_tvd: Optional[float] = None               # Mean TVD(Bayesian, LLM) on rating distribution
    mean_jsd: Optional[float] = None               # Mean JSD(Bayesian, LLM) on rating distribution

    # Type elicitation metrics
    type_posterior_kl: Optional[float] = None              # Mean KL(Bayesian type || LLM type)
    type_posterior_tvd: Optional[float] = None             # Mean TVD(Bayesian type, LLM type)
    type_posterior_jsd: Optional[float] = None             # Mean JSD(Bayesian type, LLM type)
    type_accuracy: Optional[float] = None                  # Frac where LLM argmax == Bayesian argmax
    cot_type_accuracy: Optional[float] = None              # Same but for CoT predictions
    conditioned_mae_from_bayesian: Optional[float] = None  # MAE of conditioned rating
    conditioning_lift: Optional[float] = None              # mae_independent - mae_conditioned

    # Rating-given-type fidelity (isolates "given a type k, did the LLM compute
    # the right rating?" from "did the LLM pick the right type?"). Reference is
    # the type-conditional theta row, NOT the mixture posterior.
    mae_rating_given_predicted_type: Optional[float] = None  # vs E[rating | k_LLM, target]
    mae_rating_given_type_per_type: Optional[float] = None   # mean_k |E_LLM[r|k] - E_theta[r|k]|

    # CoT follow-up MCQ distributional metrics
    cot_type_posterior_kl: Optional[float] = None         # Mean KL(Bayesian type || CoT type)
    cot_type_posterior_tvd: Optional[float] = None        # Mean TVD(Bayesian type, CoT type)
    cot_type_posterior_jsd: Optional[float] = None        # Mean JSD(Bayesian type, CoT type)
    mean_cot_type_mass: Optional[float] = None             # Mean prob mass on A-E in CoT follow-up
    mean_cot_type_scale_bias: Optional[float] = None       # Mean position bias in CoT follow-up

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryMetrics":
        # Filter to only known fields for backwards compatibility
        import dataclasses
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class TrajectoryResult:
    """Complete result of an experiment run."""
    config: ExperimentConfig
    rating_sequence: List[Dict[str, Any]]  # [{movie_id, movie_name, genres, rating}, ...]
    poll_points: List[int]
    polls: List[PollResult]
    metrics: Optional[TrajectoryMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "rating_sequence": self.rating_sequence,
            "poll_points": self.poll_points,
            "polls": [p.to_dict() for p in self.polls],
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryResult":
        return cls(
            config=ExperimentConfig.from_dict(d["config"]),
            rating_sequence=d["rating_sequence"],
            poll_points=d["poll_points"],
            polls=[PollResult.from_dict(p) for p in d["polls"]],
            metrics=TrajectoryMetrics.from_dict(d["metrics"]) if d.get("metrics") else None,
        )

"""
Configuration dataclasses for coin flip experiments.

Defines experimental conditions, configuration, and result structures.
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib


class CoinSpec(str, Enum):
    """What the model is told about the coin."""
    UNSPECIFIED = "unspecified"      # "You are predicting coin flips."
    UNKNOWN_BIAS = "unknown_bias"    # "...flips of a coin with an unknown bias."
    FAIR = "fair"                    # "...flips of a fair coin."


class Condition(str, Enum):
    """Experimental conditions for single vs multi-turn comparison."""
    SINGLE_TURN = "single_turn"
    MULTI_TURN_MINIMAL = "multi_turn_minimal"
    MULTI_TURN_ACTUAL = "multi_turn_actual"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    model_name: str
    condition: Condition
    k: int  # Polling frequency: 1 or 5
    p: float  # True probability of heads
    trial: int
    n_flips: int = 100
    batch_size: int = 1  # flips per conversation turn (1 = current per-flip behavior)
    coin_spec: CoinSpec = CoinSpec.UNSPECIFIED
    counterbalance: bool = True
    seed: Optional[int] = None
    timestamp: Optional[str] = None
    experiment_id: Optional[str] = None

    def __post_init__(self):
        # Auto-compute seed if not provided
        if self.seed is None:
            self.seed = int(self.p * 1000) + self.trial * 100 + 42

        # Auto-generate timestamp if not provided
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

        # Auto-generate experiment_id if not provided
        if self.experiment_id is None:
            hash_input = f"{self.model_name}_{self.condition.value}_{self.k}_{self.p}_{self.trial}_{self.seed}_{self.coin_spec.value}_{self.batch_size}"
            self.experiment_id = hashlib.md5(hash_input.encode()).hexdigest()[:8]

        # Ensure condition is Condition enum
        if isinstance(self.condition, str):
            self.condition = Condition(self.condition)

        # Ensure coin_spec is CoinSpec enum
        if isinstance(self.coin_spec, str):
            self.coin_spec = CoinSpec(self.coin_spec)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["condition"] = self.condition.value
        d["coin_spec"] = self.coin_spec.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        d = d.copy()
        if "condition" in d:
            d["condition"] = Condition(d["condition"])
        if "coin_spec" in d:
            d["coin_spec"] = CoinSpec(d["coin_spec"])
        return cls(**d)

    def get_output_filename(self) -> str:
        """Generate filename for saving results."""
        bs_suffix = f"_bs{self.batch_size}" if self.batch_size != 1 else ""
        return f"{self.condition.value}_{self.coin_spec.value}_{self.model_name}_p{self.p}_trial{self.trial}_k{self.k}{bs_suffix}.json"


@dataclass
class PollResult:
    """Result of a single poll (probability extraction at time t)."""
    t: int  # Time step (number of observations seen)
    n_heads: int
    n_tails: int
    p_heads: float  # Counterbalanced P(heads) estimate
    p_heads_v1: float  # P(heads) with A=heads
    p_heads_v2: float  # P(heads) with A=tails (from P(B))
    position_bias: float  # Preference for position A
    ab_mass: float  # Total probability mass on A+B
    bayesian_posterior: float  # Optimal Bayesian estimate
    prediction: Optional[str] = None  # Model's sampled prediction (for multi-turn)
    injected: bool = False  # Whether prediction was injected vs sampled

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PollResult":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class TrajectoryMetrics:
    """Computed metrics for an experiment trajectory."""
    mae_from_bayesian: float  # Mean absolute error from Bayesian optimal
    correlation_with_bayesian: float  # Correlation with Bayesian trajectory
    prior_p_heads: float  # P(heads) at t=0
    mean_update_magnitude: float  # Average |delta P| between consecutive polls
    update_magnitudes: List[float]  # Per-step update magnitudes
    mean_position_bias: float  # Average position bias
    position_biases: List[float]  # Per-poll position biases
    mean_ab_mass: float  # Average A/B mass
    ab_masses: List[float]  # Per-poll A/B masses
    trajectory_variance: float  # Variance of P(heads) trajectory

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryMetrics":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class TrajectoryResult:
    """Complete result of an experiment run."""
    config: ExperimentConfig
    sequence: List[str]  # The flip sequence
    poll_points: List[int]  # Time points where we polled
    polls: List[PollResult]  # Poll results at each point
    metrics: Optional[TrajectoryMetrics] = None  # Computed after run

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": self.config.to_dict(),
            "sequence": self.sequence,
            "poll_points": self.poll_points,
            "polls": [p.to_dict() for p in self.polls],
            "metrics": self.metrics.to_dict() if self.metrics else None
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryResult":
        """Create from dictionary."""
        return cls(
            config=ExperimentConfig.from_dict(d["config"]),
            sequence=d["sequence"],
            poll_points=d["poll_points"],
            polls=[PollResult.from_dict(p) for p in d["polls"]],
            metrics=TrajectoryMetrics.from_dict(d["metrics"]) if d.get("metrics") else None
        )

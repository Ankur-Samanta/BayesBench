"""
Message building for different experimental conditions.

Handles single-turn and multi-turn conversation construction for coin flip experiments.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .config import Condition, CoinSpec


@dataclass
class ConversationState:
    """Tracks state of a multi-turn conversation."""
    messages: List[Dict] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)  # "heads" or "tails"
    predictions: List[Optional[str]] = field(default_factory=list)  # Model predictions


class MessageBuilder:
    """Builds chat messages for different experimental conditions."""

    COIN_DESCRIPTIONS = {
        CoinSpec.UNSPECIFIED: "You are predicting coin flips.",
        CoinSpec.UNKNOWN_BIAS: "You are predicting flips of a coin with an unknown bias.",
        CoinSpec.FAIR: "You are predicting flips of a fair coin.",
    }

    @staticmethod
    def _get_mapping(a_is_heads: bool) -> str:
        """Get A/B to heads/tails mapping string."""
        if a_is_heads:
            return "A = heads, B = tails"
        else:
            return "A = tails, B = heads"

    @classmethod
    def _build_system(cls, a_is_heads: bool, coin_spec: CoinSpec = CoinSpec.UNSPECIFIED) -> str:
        """Build system message with coin description and A/B mapping."""
        desc = cls.COIN_DESCRIPTIONS[coin_spec]
        mapping = cls._get_mapping(a_is_heads)
        return f"{desc} Respond with only A or B.\n{mapping}"

    @classmethod
    def build_single_turn(cls, history: List[str], a_is_heads: bool = True, coin_spec: CoinSpec = CoinSpec.UNSPECIFIED) -> List[Dict]:
        """
        Build single-turn prompt with full history.

        Args:
            history: List of observed flip outcomes ("heads" or "tails")
            a_is_heads: If True, A=heads/B=tails. If False, A=tails/B=heads.

        Returns:
            List of chat messages
        """
        system = cls._build_system(a_is_heads, coin_spec)

        if len(history) == 0:
            user = "Predict the next flip."
        else:
            history_str = ", ".join(history)
            n_heads = sum(1 for f in history if f == "heads")
            n_tails = len(history) - n_heads
            user = f"Previous flips: [{history_str}]\nTotal: {n_heads} heads, {n_tails} tails.\n\nPredict the next flip."

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

    @classmethod
    def init_multi_turn_state(cls, a_is_heads: bool = True, coin_spec: CoinSpec = CoinSpec.UNSPECIFIED) -> ConversationState:
        """
        Initialize multi-turn conversation state.

        Args:
            a_is_heads: If True, A=heads/B=tails. If False, A=tails/B=heads.

        Returns:
            Initial conversation state
        """
        system = cls._build_system(a_is_heads, coin_spec)

        state = ConversationState()
        state.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "Predict the next flip."}
        ]
        return state

    @classmethod
    def add_observation_minimal(cls, state: ConversationState, outcome: str) -> ConversationState:
        """
        Add observation with minimal "Noted." response (Condition 2).

        Args:
            state: Current conversation state
            outcome: "heads" or "tails"

        Returns:
            Updated conversation state
        """
        # Add model's "Noted." acknowledgment
        state.messages.append({"role": "assistant", "content": "Noted."})

        # Add new observation
        state.messages.append({"role": "user", "content": f"The flip was: {outcome}"})

        # Track observation
        state.observations.append(outcome)
        state.predictions.append(None)

        return state

    @classmethod
    def add_observation_with_prediction(
        cls,
        state: ConversationState,
        outcome: str,
        prediction: str,
        a_is_heads: bool = True
    ) -> ConversationState:
        """
        Add observation with model's prediction in context (Conditions 3-5).

        Args:
            state: Current conversation state
            outcome: "heads" or "tails"
            prediction: Model's prediction ("A" or "B")
            a_is_heads: Mapping for interpreting prediction

        Returns:
            Updated conversation state
        """
        # Add model's prediction as assistant turn
        state.messages.append({"role": "assistant", "content": prediction})

        # Add outcome reveal
        state.messages.append({"role": "user", "content": f"The flip was: {outcome}"})

        # Track
        state.observations.append(outcome)

        # Convert prediction to heads/tails for tracking
        if prediction.upper() == "A":
            pred_outcome = "heads" if a_is_heads else "tails"
        else:
            pred_outcome = "tails" if a_is_heads else "heads"
        state.predictions.append(pred_outcome)

        return state

    @classmethod
    def add_observation_batch_minimal(
        cls, state: ConversationState, outcomes: List[str]
    ) -> ConversationState:
        """Add a batch of observations with a single 'Noted.' response."""
        if len(outcomes) == 1:
            return cls.add_observation_minimal(state, outcomes[0])
        state.messages.append({"role": "assistant", "content": "Noted."})
        batch_str = ", ".join(outcomes)
        state.messages.append({"role": "user", "content": f"The next {len(outcomes)} flips were: {batch_str}"})
        for outcome in outcomes:
            state.observations.append(outcome)
            state.predictions.append(None)
        return state

    @classmethod
    def add_observation_batch_with_prediction(
        cls,
        state: ConversationState,
        outcomes: List[str],
        prediction: str,
        a_is_heads: bool = True,
    ) -> ConversationState:
        """Add a batch of observations after one prediction (one prediction per batch)."""
        if len(outcomes) == 1:
            return cls.add_observation_with_prediction(state, outcomes[0], prediction, a_is_heads)
        state.messages.append({"role": "assistant", "content": prediction})
        batch_str = ", ".join(outcomes)
        state.messages.append({"role": "user", "content": f"The next {len(outcomes)} flips were: {batch_str}"})
        if prediction.upper() == "A":
            pred_outcome = "heads" if a_is_heads else "tails"
        else:
            pred_outcome = "tails" if a_is_heads else "heads"
        for i, outcome in enumerate(outcomes):
            state.observations.append(outcome)
            state.predictions.append(pred_outcome if i == 0 else None)
        return state

    @classmethod
    def build_poll_prompt_with_partial(
        cls,
        state: ConversationState,
        partial_batch: List[str],
        a_is_heads: bool = True,
    ) -> List[Dict]:
        """
        Build poll prompt including any in-progress partial batch.

        If partial_batch is empty, delegates to build_poll_prompt.
        Otherwise appends the partial flips as context before the prediction ask.
        """
        if not partial_batch:
            return cls.build_poll_prompt(state, a_is_heads)

        messages = state.messages.copy()
        n_heads = sum(1 for o in state.observations if o == "heads") + \
                  sum(1 for o in partial_batch if o == "heads")
        n_tails = len(state.observations) + len(partial_batch) - n_heads

        partial_str = ", ".join(partial_batch)
        content = (f"So far in this batch: {partial_str}\n\n"
                   f"Total: {n_heads} heads, {n_tails} tails.\nPredict the next flip.")

        # Append as a fresh user turn after the last assistant "Noted."
        messages.append({"role": "user", "content": content})
        return messages

    @classmethod
    def build_poll_prompt(cls, state: ConversationState, a_is_heads: bool = True) -> List[Dict]:
        """
        Build messages for polling (asking for prediction).

        Args:
            state: Current conversation state
            a_is_heads: Mapping to use

        Returns:
            Messages ready for prediction extraction
        """
        messages = state.messages.copy()

        # If the last message was an observation (user turn), ask for prediction
        if messages and messages[-1]["role"] == "user":
            # Append prediction prompt to last user message
            n_heads = sum(1 for o in state.observations if o == "heads")
            n_tails = len(state.observations) - n_heads

            if len(state.observations) == 0:
                # t=0: user message already says "Predict the next flip."
                pass
            else:
                prompt_addition = f"\n\nTotal: {n_heads} heads, {n_tails} tails.\nPredict the next flip."
                messages[-1] = {
                    "role": "user",
                    "content": messages[-1]["content"] + prompt_addition
                }

        return messages

    @classmethod
    def build_poll_prompt_k5(
        cls,
        state: ConversationState,
        last_5: List[str],
        total_heads: int,
        total_tails: int,
        a_is_heads: bool = True
    ) -> List[Dict]:
        """
        Build poll prompt for k=5 batched observations.

        Args:
            state: Current conversation state
            last_5: Last 5 observations
            total_heads: Total heads so far
            total_tails: Total tails so far
            a_is_heads: Mapping to use

        Returns:
            Messages ready for prediction extraction
        """
        messages = state.messages.copy()

        # Format the batch
        batch_str = ", ".join(last_5)
        prompt = f"Last 5 flips: [{batch_str}]\nTotal: {total_heads} heads, {total_tails} tails.\n\nPredict the next flip."

        messages.append({"role": "user", "content": prompt})
        return messages



"""
Message building for recommender-system cold-start experiments.

Handles single-turn and multi-turn conversation construction
for streaming movie ratings with population information context.
Uses 1-5 star rating scale with optional counterbalanced (reversed) scale.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .config import Condition, PopInfo


@dataclass
class ConversationState:
    """Tracks state of a multi-turn conversation."""
    messages: List[Dict] = field(default_factory=list)
    observations: List[Dict] = field(default_factory=list)  # [{movie_id, movie_name, genres, rating}, ...]
    predictions: List[Optional[str]] = field(default_factory=list)


class MessageBuilder:
    """Builds chat messages for recommender-system experiments."""

    FORMAT_REMINDER = "\nRespond with only A, B, C, D, or E."

    @staticmethod
    def _get_rating_scale(reversed_scale: bool = False, rating_shift: Optional[int] = None) -> str:
        """Get the rating-scale string.

        If ``rating_shift`` is provided, returns a cyclic shift: letter at
        position i (A+i) holds rating ``((i + shift) % 5) + 1``. Across all
        K=5 shifts, every rating value occupies every letter position
        exactly once — provably eliminating per-letter position bias
        (including center-anchoring at letter C). This mirrors the cyclic
        scheme used for type elicitation and in medical triage.

        If ``rating_shift`` is None, falls back to the legacy 2-way
        ``reversed_scale`` behavior (standard vs. reversed scale).
        """
        if rating_shift is not None:
            adj = {
                1: "1 star (terrible)",
                2: "2 stars (bad)",
                3: "3 stars (okay)",
                4: "4 stars (good)",
                5: "5 stars (great)",
            }
            return ", ".join(
                f"{chr(65 + i)} = {adj[(i + rating_shift) % 5 + 1]}"
                for i in range(5)
            )
        if reversed_scale:
            return "A = 5 stars (great), B = 4 stars (good), C = 3 stars (okay), D = 2 stars (bad), E = 1 star (terrible)"
        return "A = 1 star (terrible), B = 2 stars (bad), C = 3 stars (okay), D = 4 stars (good), E = 5 stars (great)"

    @staticmethod
    def _get_type_scale(shift: int = 0, n_types: int = 4) -> str:
        """Get MCQ scale mapping letters to profile numbers.

        Uses cyclic permutation: shift=0 gives A=Profile1...D=Profile4,
        shift=1 gives A=Profile2...D=Profile1, etc. Each type occupies
        each position exactly once across all K shifts, provably
        eliminating any position bias.
        """
        return ", ".join(
            f"{chr(65+i)} = Profile {(i + shift) % n_types + 1}"
            for i in range(n_types)
        )

    @classmethod
    def _format_pop_info_explicit(
        cls,
        type_model,
        probe_movie_ids: List[int],
        target_movie_id: int,
    ) -> str:
        """Format explicit type distribution information."""
        stars = np.arange(1, 6)
        lines = []
        for k in range(type_model.n_types):
            pct = type_model.pi[k] * 100
            lines.append(f"Profile {k+1} ({pct:.0f}% of users):")
            # Show probe movies
            for mid in probe_movie_ids:
                if mid not in type_model.theta:
                    continue
                name = type_model.movie_names[mid]
                genres = ", ".join(type_model.movie_genres[mid])
                e_rating = float(type_model.theta[mid][k] @ stars)
                lines.append(f"- {name} ({genres}): avg rating: {e_rating:.1f}")
            # NOTE: Target movie ratings intentionally omitted from profiles.
            # Including them makes this a table-lookup task, not Bayesian inference.
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def _format_pop_info_anonymized(
        cls,
        type_model,
        probe_movie_ids: List[int],
        target_movie_id: int,
        anon_map: Dict,
    ) -> str:
        """Format explicit type distributions with anonymized item/feature labels."""
        stars = np.arange(1, 6)
        movie_name_map = anon_map["movie_name_map"]
        genre_map = anon_map["genre_map"]
        lines = []
        for k in range(type_model.n_types):
            pct = type_model.pi[k] * 100
            lines.append(f"Profile {k+1} ({pct:.0f}% of users):")
            for mid in probe_movie_ids:
                if mid not in type_model.theta:
                    continue
                name = movie_name_map.get(mid, f"Item_{mid}")
                genres = type_model.movie_genres.get(mid, [])
                anon_genres = ", ".join(genre_map.get(g, g) for g in genres)
                e_rating = float(type_model.theta[mid][k] @ stars)
                lines.append(f"- {name} ({anon_genres}): avg rating: {e_rating:.1f}")
            # Target movie ratings intentionally omitted
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def _build_system(
        cls,
        reversed_scale: bool,
        pop_info: PopInfo,
        type_model=None,
        probe_movie_ids: List[int] = None,
        target_movie_id: int = None,
        anon_map: Dict = None,
        rating_shift: Optional[int] = None,
    ) -> str:
        scale = cls._get_rating_scale(reversed_scale, rating_shift)

        if pop_info == PopInfo.ANONYMIZED:
            base = (
                "You are predicting how a user will rate an item based on their rating history.\n"
                "Respond with only A, B, C, D, or E.\n"
                f"{scale}"
            )
        else:
            base = (
                "You are predicting how a user will rate a movie based on their rating history.\n"
                "Respond with only A, B, C, D, or E.\n"
                f"{scale}"
            )

        if pop_info == PopInfo.EXPLICIT_TYPES:
            pop_text = cls._format_pop_info_explicit(
                type_model, probe_movie_ids, target_movie_id
            )
            n_types = type_model.n_types
            return (
                f"{base}\n\n"
                f"Based on data from our platform, users fall into {n_types} viewer profiles:\n\n"
                f"{pop_text}\n"
                f"Given a user's rating history, consider which profile best matches their "
                f"pattern of ratings, then predict how they would rate the target movie.\n"
                f"Respond with only A, B, C, D, or E.\n"
                f"{scale}"
            )

        elif pop_info == PopInfo.ANONYMIZED:
            pop_text = cls._format_pop_info_anonymized(
                type_model, probe_movie_ids, target_movie_id, anon_map
            )
            n_types = type_model.n_types
            return (
                f"{base}\n\n"
                f"Based on data from our platform, users fall into {n_types} profiles:\n\n"
                f"{pop_text}\n"
                f"Given a user's rating history, consider which profile best matches their "
                f"pattern of ratings, then predict how they would rate the target item.\n"
                f"Respond with only A, B, C, D, or E.\n"
                f"{scale}"
            )

        else:  # ZERO_SHOT
            return base

    @classmethod
    def _format_rating_list(cls, history: List[Dict]) -> str:
        """Format the list of ratings seen so far."""
        rating_counts = [0] * 5
        for r in history:
            rating_counts[r["rating"] - 1] += 1
        total = len(history)
        avg = sum(r["rating"] for r in history) / total if total > 0 else 0

        lines = []
        for r in history:
            genres = ", ".join(r["genres"]) if r["genres"] else ""
            genre_str = f" ({genres})" if genres else ""
            lines.append(f"- {r['movie_name']}{genre_str}: {r['rating']} stars")

        ratings_block = "\n".join(lines)
        return (
            f"The user's ratings so far:\n"
            f"{ratings_block}\n"
            f"Average rating: {avg:.1f} ({total} movies rated)."
        )

    @classmethod
    def build_single_turn(
        cls,
        history: List[Dict],
        target_movie_name: str,
        target_movie_genres: List[str],
        reversed_scale: bool,
        pop_info: PopInfo,
        type_model=None,
        probe_movie_ids: List[int] = None,
        target_movie_id: int = None,
        anon_map: Dict = None,
        rating_shift: Optional[int] = None,
    ) -> List[Dict]:
        """
        Build single-turn prompt with full history.

        Returns:
            List of chat messages
        """
        system = cls._build_system(
            reversed_scale, pop_info, type_model, probe_movie_ids,
            target_movie_id, anon_map=anon_map, rating_shift=rating_shift,
        )

        genre_str = f" ({', '.join(target_movie_genres)})" if target_movie_genres else ""
        target_str = f"{target_movie_name}{genre_str}"

        if len(history) == 0:
            user = f"What rating will this user give {target_str}?" + cls.FORMAT_REMINDER
        else:
            rating_block = cls._format_rating_list(history)
            user = f"{rating_block}\nWhat rating will this user give {target_str}?" + cls.FORMAT_REMINDER

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @classmethod
    def init_multi_turn_state(
        cls,
        target_movie_name: str,
        target_movie_genres: List[str],
        reversed_scale: bool,
        pop_info: PopInfo,
        type_model=None,
        probe_movie_ids: List[int] = None,
        target_movie_id: int = None,
        anon_map: Dict = None,
        rating_shift: Optional[int] = None,
    ) -> ConversationState:
        """Initialize multi-turn conversation state."""
        system = cls._build_system(
            reversed_scale, pop_info, type_model, probe_movie_ids,
            target_movie_id, anon_map=anon_map, rating_shift=rating_shift,
        )

        genre_str = f" ({', '.join(target_movie_genres)})" if target_movie_genres else ""
        target_str = f"{target_movie_name}{genre_str}"

        state = ConversationState()
        state.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"What rating will this user give {target_str}?"},
        ]
        return state

    @classmethod
    def add_observation_minimal(
        cls,
        state: ConversationState,
        rating: Dict,
    ) -> ConversationState:
        """
        Add observation with minimal "Noted." response (multi_turn_minimal).

        Args:
            state: Current conversation state
            rating: {movie_id, movie_name, genres, rating}
        """
        state.messages.append({"role": "assistant", "content": "Noted."})

        genres = ", ".join(rating["genres"]) if rating["genres"] else ""
        genre_str = f" ({genres})" if genres else ""
        user_content = f"The user rated {rating['movie_name']}{genre_str}: {rating['rating']} stars"

        state.messages.append({"role": "user", "content": user_content})

        state.observations.append(rating)
        state.predictions.append(None)

        return state

    @classmethod
    def add_observation_with_prediction(
        cls,
        state: ConversationState,
        rating: Dict,
        prediction: str,
    ) -> ConversationState:
        """
        Add observation with model's prediction in context (multi_turn_actual).

        Args:
            state: Current conversation state
            rating: {movie_id, movie_name, genres, rating}
            prediction: Model's predicted rating digit ("1"-"5")
        """
        state.messages.append({"role": "assistant", "content": prediction})

        genres = ", ".join(rating["genres"]) if rating["genres"] else ""
        genre_str = f" ({genres})" if genres else ""
        user_content = f"The user rated {rating['movie_name']}{genre_str}: {rating['rating']} stars"

        state.messages.append({"role": "user", "content": user_content})

        state.observations.append(rating)
        state.predictions.append(prediction)

        return state

    @classmethod
    def build_poll_prompt(
        cls,
        state: ConversationState,
        target_movie_name: str,
        target_movie_genres: List[str],
    ) -> List[Dict]:
        """
        Build messages for polling (asking for prediction).

        At t=0, the user message already asks the question.
        At t>0, append running totals and prediction prompt.
        """
        messages = [m.copy() for m in state.messages]

        if messages and messages[-1]["role"] == "user":
            if len(state.observations) == 0:
                # t=0: append format reminder to existing question
                messages[-1] = {
                    "role": "user",
                    "content": messages[-1]["content"] + cls.FORMAT_REMINDER,
                }
            else:
                total = len(state.observations)
                avg = sum(o["rating"] for o in state.observations) / total

                genre_str = f" ({', '.join(target_movie_genres)})" if target_movie_genres else ""
                target_str = f"{target_movie_name}{genre_str}"

                prompt_addition = (
                    f"\n\nAverage rating: {avg:.1f} ({total} movies rated).\n"
                    f"What rating will this user give {target_str}?"
                    + cls.FORMAT_REMINDER
                )
                messages[-1] = {
                    "role": "user",
                    "content": messages[-1]["content"] + prompt_addition,
                }

        return messages

    @classmethod
    def build_rating_poll_from_type_state(
        cls,
        state: ConversationState,
        target_movie_name: str,
        target_movie_genres: List[str],
        reversed_scale: bool,
        pop_info: PopInfo,
        type_model=None,
        probe_movie_ids: List[int] = None,
        target_movie_id: int = None,
        anon_map: Dict = None,
        rating_shift: Optional[int] = None,
    ) -> List[Dict]:
        """
        Build rating poll from a type-based conversation state.

        Swaps the type system prompt for the rating system prompt and appends
        the rating question. Mirrors build_type_poll_multi_turn but in reverse:
        that method takes a rating conversation and swaps in the type system prompt,
        this takes a type conversation and swaps in the rating system prompt.
        """
        rating_system = cls._build_system(
            reversed_scale, pop_info, type_model, probe_movie_ids,
            target_movie_id, anon_map=anon_map, rating_shift=rating_shift,
        )

        genre_str = f" ({', '.join(target_movie_genres)})" if target_movie_genres else ""
        target_str = f"{target_movie_name}{genre_str}"

        messages = [{"role": "system", "content": rating_system}]
        for m in state.messages:
            if m["role"] != "system":
                messages.append(m.copy())

        if messages and messages[-1]["role"] == "user":
            if len(state.observations) == 0:
                # t=0: replace initial type question with rating question
                messages[-1] = {
                    "role": "user",
                    "content": f"What rating will this user give {target_str}?" + cls.FORMAT_REMINDER,
                }
            else:
                total = len(state.observations)
                avg = sum(o["rating"] for o in state.observations) / total
                prompt_addition = (
                    f"\n\nAverage rating: {avg:.1f} ({total} movies rated).\n"
                    f"What rating will this user give {target_str}?"
                    + cls.FORMAT_REMINDER
                )
                messages[-1] = {
                    "role": "user",
                    "content": messages[-1]["content"] + prompt_addition,
                }

        return messages

    @classmethod
    def build_conditioned_rating_from_type_state(
        cls,
        state: ConversationState,
        predicted_type: int,
        target_movie_name: str,
        target_movie_genres: List[str],
        reversed_scale: bool,
        pop_info: PopInfo,
        type_model=None,
        probe_movie_ids: List[int] = None,
        target_movie_id: int = None,
        anon_map: Dict = None,
        cot_reasoning: Optional[str] = None,
        rating_shift: Optional[int] = None,
    ) -> List[Dict]:
        """
        Build V2 conditioned rating poll from a type-based conversation state.

        Structure:
        - Rating system prompt (swapped from type system)
        - Conversation history (non-system messages from state)
        - Type question appended to last user message
        - Assistant: "Profile {X}"
        - User: "Based on your prediction of Profile X, what rating will this user give <target>?"

        Args:
            state: Type-based conversation state
            predicted_type: 0-indexed type prediction (from CoT or MCQ)
            target_movie_name: Display name of target movie
            target_movie_genres: Genre list for target movie
            reversed_scale: Whether to reverse the rating scale
            pop_info: Population info mode
            type_model: Type model (for system prompt)
            probe_movie_ids: Probe movie IDs (for system prompt)
            target_movie_id: Target movie ID (for system prompt)
            anon_map: Anonymization map (for anonymized condition)
        """
        # 1. Build rating system prompt
        rating_system = cls._build_system(
            reversed_scale, pop_info, type_model, probe_movie_ids,
            target_movie_id, anon_map=anon_map, rating_shift=rating_shift,
        )

        genre_str = f" ({', '.join(target_movie_genres)})" if target_movie_genres else ""
        target_str = f"{target_movie_name}{genre_str}"
        question = cls._type_question(pop_info)

        # 2. Copy non-system messages from state
        messages = [{"role": "system", "content": rating_system}]
        for m in state.messages:
            if m["role"] != "system":
                messages.append(m.copy())

        # 3. Append type question suffix to last user message
        if messages and messages[-1]["role"] == "user":
            if len(state.observations) > 0:
                total = len(state.observations)
                avg = sum(o["rating"] for o in state.observations) / total
                suffix = (
                    f"\n\nAvg: {avg:.1f} ({total} rated).\n"
                    f"{question}"
                )
                messages[-1] = {
                    "role": "user",
                    "content": messages[-1]["content"] + suffix,
                }
            # t=0: initial user message already has the type question

        # 4. Append assistant message with type prediction
        if cot_reasoning:
            assistant_content = f"{cot_reasoning}\n\nBased on this analysis, this user most closely matches Profile {predicted_type + 1}."
        else:
            assistant_content = f"Profile {predicted_type + 1}"
        messages.append({
            "role": "assistant",
            "content": assistant_content,
        })

        # 5. Append conditioned rating question
        messages.append({
            "role": "user",
            "content": f"Based on your prediction of Profile {predicted_type + 1}, "
                       f"what rating will this user give {target_str}?"
                       + cls.FORMAT_REMINDER,
        })

        return messages

    # =========================================================================
    # Type Elicitation Prompts
    # =========================================================================

    @classmethod
    def _build_type_system(
        cls,
        shift: int,
        pop_info: PopInfo,
        type_model=None,
        probe_movie_ids: List[int] = None,
        target_movie_id: int = None,
        anon_map: Dict = None,
    ) -> str:
        """Build system prompt for type classification MCQ."""
        n_types = type_model.n_types
        type_scale = cls._get_type_scale(shift, n_types)

        if pop_info == PopInfo.ANONYMIZED:
            task_desc = "You are classifying which profile a user belongs to based on their rating history."
        else:
            task_desc = "You are classifying which viewer profile a user belongs to based on their rating history."

        base = (
            f"{task_desc}\n"
            f"Respond with only A, B, C, D, or E.\n"
            f"{type_scale}"
        )

        if pop_info == PopInfo.EXPLICIT_TYPES:
            pop_text = cls._format_pop_info_explicit(
                type_model, probe_movie_ids, target_movie_id
            )
            return (
                f"{base}\n\n"
                f"Based on data from our platform, users fall into {n_types} viewer profiles:\n\n"
                f"{pop_text}\n"
                f"Given a user's rating history, consider which profile best matches their "
                f"pattern of ratings.\n"
                f"Respond with only A, B, C, D, or E.\n"
                f"{type_scale}"
            )
        elif pop_info == PopInfo.ANONYMIZED:
            pop_text = cls._format_pop_info_anonymized(
                type_model, probe_movie_ids, target_movie_id, anon_map
            )
            return (
                f"{base}\n\n"
                f"Based on data from our platform, users fall into {n_types} profiles:\n\n"
                f"{pop_text}\n"
                f"Given a user's rating history, consider which profile best matches their "
                f"pattern of ratings.\n"
                f"Respond with only A, B, C, D, or E.\n"
                f"{type_scale}"
            )
        else:
            return base

    @classmethod
    def _build_type_cot_system(
        cls,
        pop_info: PopInfo,
        type_model=None,
        probe_movie_ids: List[int] = None,
        target_movie_id: int = None,
        anon_map: Dict = None,
    ) -> str:
        """Build system prompt for type classification with CoT reasoning."""
        n_types = type_model.n_types

        if pop_info == PopInfo.ANONYMIZED:
            task_desc = "You are classifying which profile a user belongs to based on their rating history."
        else:
            task_desc = "You are classifying which viewer profile a user belongs to based on their rating history."

        base = (
            f"{task_desc}\n\n"
            f"Think step by step about how the user's ratings compare to each profile's "
            f"typical preferences. After your analysis, state your final answer on its own line as:\n"
            f"Answer: Profile N\n"
            f"where N is the profile number (1-{n_types})."
        )

        if pop_info == PopInfo.EXPLICIT_TYPES:
            pop_text = cls._format_pop_info_explicit(
                type_model, probe_movie_ids, target_movie_id
            )
            return (
                f"{base}\n\n"
                f"Based on data from our platform, users fall into {n_types} viewer profiles:\n\n"
                f"{pop_text}\n"
                f"Given a user's rating history, consider which profile best matches their "
                f"pattern of ratings."
            )
        elif pop_info == PopInfo.ANONYMIZED:
            pop_text = cls._format_pop_info_anonymized(
                type_model, probe_movie_ids, target_movie_id, anon_map
            )
            return (
                f"{base}\n\n"
                f"Based on data from our platform, users fall into {n_types} profiles:\n\n"
                f"{pop_text}\n"
                f"Given a user's rating history, consider which profile best matches their "
                f"pattern of ratings."
            )
        else:
            return base

    @classmethod
    def _type_question(cls, pop_info: PopInfo) -> str:
        """Get the type classification question, matching system prompt terminology."""
        if pop_info == PopInfo.ANONYMIZED:
            return "Which profile does this user most closely match?"
        return "Which viewer profile does this user most closely match?"

    @classmethod
    def build_type_poll_single_turn(
        cls,
        history: List[Dict],
        shift: int,
        pop_info: PopInfo,
        type_model=None,
        probe_movie_ids: List[int] = None,
        target_movie_id: int = None,
        anon_map: Dict = None,
    ) -> List[Dict]:
        """Build single-turn type MCQ prompt."""
        system = cls._build_type_system(
            shift, pop_info, type_model, probe_movie_ids,
            target_movie_id, anon_map=anon_map,
        )

        question = cls._type_question(pop_info)
        if len(history) == 0:
            user = question + cls.FORMAT_REMINDER
        else:
            rating_block = cls._format_rating_list(history)
            user = f"{rating_block}\n{question}" + cls.FORMAT_REMINDER

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @classmethod
    def build_type_cot_single_turn(
        cls,
        history: List[Dict],
        pop_info: PopInfo,
        type_model=None,
        probe_movie_ids: List[int] = None,
        target_movie_id: int = None,
        anon_map: Dict = None,
    ) -> List[Dict]:
        """Build single-turn type CoT prompt."""
        system = cls._build_type_cot_system(
            pop_info, type_model, probe_movie_ids,
            target_movie_id, anon_map=anon_map,
        )

        question = cls._type_question(pop_info)
        if len(history) == 0:
            user = question
        else:
            rating_block = cls._format_rating_list(history)
            user = f"{rating_block}\n{question}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @classmethod
    def build_type_cot_followup_single_turn(
        cls,
        history: List[Dict],
        cot_reasoning: str,
        shift: int,
        pop_info: PopInfo,
        type_model=None,
        probe_movie_ids: List[int] = None,
        target_movie_id: int = None,
        anon_map: Dict = None,
    ) -> List[Dict]:
        """
        Build follow-up MCQ after CoT generation (single-turn).

        Structure:
        [system: type MCQ system (cyclic shift scale)]
        [user: rating history + type question]
        [assistant: CoT reasoning text]
        [user: "Respond with only A, B, C, D, or E."]

        Uses _build_type_system (MCQ scale), NOT _build_type_cot_system.
        """
        system = cls._build_type_system(
            shift, pop_info, type_model, probe_movie_ids,
            target_movie_id, anon_map=anon_map,
        )

        question = cls._type_question(pop_info)
        if len(history) == 0:
            user = question
        else:
            rating_block = cls._format_rating_list(history)
            user = f"{rating_block}\n{question}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": cot_reasoning},
            {"role": "user", "content": "Respond with only A, B, C, D, or E."},
        ]

    @classmethod
    def build_conditioned_single_turn(
        cls,
        history: List[Dict],
        target_movie_name: str,
        target_movie_genres: List[str],
        predicted_type: int,
        reversed_scale: bool,
        pop_info: PopInfo,
        type_model=None,
        probe_movie_ids: List[int] = None,
        target_movie_id: int = None,
        anon_map: Dict = None,
        cot_reasoning: Optional[str] = None,
        rating_shift: Optional[int] = None,
    ) -> List[Dict]:
        """
        Build single-turn rating MCQ with type hint prepended to user message.

        Args:
            predicted_type: 0-indexed type prediction from CoT
            cot_reasoning: Optional CoT reasoning text to include in the type hint
        """
        system = cls._build_system(
            reversed_scale, pop_info, type_model, probe_movie_ids,
            target_movie_id, anon_map=anon_map, rating_shift=rating_shift,
        )

        genre_str = f" ({', '.join(target_movie_genres)})" if target_movie_genres else ""
        target_str = f"{target_movie_name}{genre_str}"

        if cot_reasoning:
            type_hint = (
                f"Here is my analysis of this user's profile:\n{cot_reasoning}\n\n"
                f"Based on this analysis, this user most closely matches Profile {predicted_type + 1}.\n\n"
            )
        else:
            type_hint = f"Based on their rating history, this user most closely matches Profile {predicted_type + 1}.\n\n"

        if len(history) == 0:
            user = f"{type_hint}What rating will this user give {target_str}?" + cls.FORMAT_REMINDER
        else:
            rating_block = cls._format_rating_list(history)
            user = f"{type_hint}{rating_block}\nWhat rating will this user give {target_str}?" + cls.FORMAT_REMINDER

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # =========================================================================
    # Multi-turn Type Elicitation Prompts
    # =========================================================================

    @classmethod
    def build_type_poll_multi_turn(
        cls,
        state: ConversationState,
        shift: int,
        pop_info: PopInfo,
        type_model=None,
        probe_movie_ids: List[int] = None,
        target_movie_id: int = None,
        anon_map: Dict = None,
    ) -> List[Dict]:
        """
        Build multi-turn type MCQ by replacing system prompt and last question.

        Creates a fresh message list with type system prompt and the same
        conversation history, with the last user message asking about type.
        """
        type_system = cls._build_type_system(
            shift, pop_info, type_model, probe_movie_ids,
            target_movie_id, anon_map=anon_map,
        )

        question = cls._type_question(pop_info)

        messages = [{"role": "system", "content": type_system}]
        # Copy non-system messages from state
        for m in state.messages:
            if m["role"] != "system":
                messages.append(m.copy())

        # Append type question to last user message
        if messages and messages[-1]["role"] == "user":
            if len(state.observations) > 0:
                total = len(state.observations)
                avg = sum(o["rating"] for o in state.observations) / total
                suffix = (
                    f"\n\nAverage rating: {avg:.1f} ({total} movies rated).\n"
                    f"{question}"
                    + cls.FORMAT_REMINDER
                )
                messages[-1] = {
                    "role": "user",
                    "content": messages[-1]["content"] + suffix,
                }
            else:
                # t=0: append format reminder to existing question
                messages[-1] = {
                    "role": "user",
                    "content": messages[-1]["content"] + cls.FORMAT_REMINDER,
                }
        else:
            messages.append({"role": "user", "content": question + cls.FORMAT_REMINDER})

        return messages

    @classmethod
    def build_type_cot_multi_turn(
        cls,
        state: ConversationState,
        pop_info: PopInfo,
        type_model=None,
        probe_movie_ids: List[int] = None,
        target_movie_id: int = None,
        anon_map: Dict = None,
    ) -> List[Dict]:
        """Build multi-turn type CoT by replacing system prompt."""
        cot_system = cls._build_type_cot_system(
            pop_info, type_model, probe_movie_ids,
            target_movie_id, anon_map=anon_map,
        )

        question = cls._type_question(pop_info)

        messages = [{"role": "system", "content": cot_system}]
        for m in state.messages:
            if m["role"] != "system":
                messages.append(m.copy())

        # Append type question to last user message
        if messages and messages[-1]["role"] == "user":
            if len(state.observations) > 0:
                total = len(state.observations)
                avg = sum(o["rating"] for o in state.observations) / total
                suffix = (
                    f"\n\nAverage rating: {avg:.1f} ({total} movies rated).\n"
                    f"{question}"
                )
                messages[-1] = {
                    "role": "user",
                    "content": messages[-1]["content"] + suffix,
                }
        else:
            messages.append({"role": "user", "content": question})

        return messages

    @classmethod
    def build_type_cot_followup_multi_turn(
        cls,
        state: ConversationState,
        cot_reasoning: str,
        shift: int,
        pop_info: PopInfo,
        type_model=None,
        probe_movie_ids: List[int] = None,
        target_movie_id: int = None,
        anon_map: Dict = None,
    ) -> List[Dict]:
        """
        Build follow-up MCQ after CoT generation (multi-turn).

        Structure:
        [system: type MCQ system (cyclic shift scale)]
        [...conversation history from state...]
        [last user msg + type question suffix]
        [assistant: CoT reasoning text]
        [user: "Respond with only A, B, C, D, or E."]

        Uses _build_type_system (MCQ scale), NOT _build_type_cot_system.
        """
        type_system = cls._build_type_system(
            shift, pop_info, type_model, probe_movie_ids,
            target_movie_id, anon_map=anon_map,
        )

        question = cls._type_question(pop_info)

        messages = [{"role": "system", "content": type_system}]
        for m in state.messages:
            if m["role"] != "system":
                messages.append(m.copy())

        # Append type question to last user message
        if messages and messages[-1]["role"] == "user":
            if len(state.observations) > 0:
                total = len(state.observations)
                avg = sum(o["rating"] for o in state.observations) / total
                suffix = (
                    f"\n\nAverage rating: {avg:.1f} ({total} movies rated).\n"
                    f"{question}"
                )
                messages[-1] = {
                    "role": "user",
                    "content": messages[-1]["content"] + suffix,
                }
        else:
            messages.append({"role": "user", "content": question})

        # Append CoT reasoning and format reminder
        messages.append({"role": "assistant", "content": cot_reasoning})
        messages.append({"role": "user", "content": "Respond with only A, B, C, D, or E."})

        return messages

    # =========================================================================
    # Multi-turn Actual: Type-Based Conversation
    # =========================================================================

    @classmethod
    def init_multi_turn_type_state(
        cls,
        shift: int,
        pop_info: PopInfo,
        type_model=None,
        probe_movie_ids: List[int] = None,
        target_movie_id: int = None,
        anon_map: Dict = None,
    ) -> ConversationState:
        """Initialize multi-turn conversation with type question instead of rating question."""
        system = cls._build_type_system(
            shift, pop_info, type_model, probe_movie_ids,
            target_movie_id, anon_map=anon_map,
        )

        state = ConversationState()
        state.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": cls._type_question(pop_info)},
        ]
        return state

    @classmethod
    def add_observation_with_type_prediction(
        cls,
        state: ConversationState,
        rating: Dict,
        type_prediction: int,
    ) -> ConversationState:
        """
        Add observation with type prediction in context (multi_turn_actual with types).

        Injects "Profile {X}" as the assistant's response (scale-invariant).

        Args:
            state: Current conversation state
            rating: {movie_id, movie_name, genres, rating}
            type_prediction: 0-indexed type prediction
        """
        prediction_text = f"Profile {type_prediction + 1}"
        state.messages.append({"role": "assistant", "content": prediction_text})

        genres = ", ".join(rating["genres"]) if rating["genres"] else ""
        genre_str = f" ({genres})" if genres else ""
        user_content = f"The user rated {rating['movie_name']}{genre_str}: {rating['rating']} stars"

        state.messages.append({"role": "user", "content": user_content})

        state.observations.append(rating)
        state.predictions.append(prediction_text)

        return state

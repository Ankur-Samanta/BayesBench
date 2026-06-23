"""
Message building for scripted conditions (single_turn, multi_turn_passive).

Constructs chat message lists for the judge model from storyboard aspects.
"""

from typing import List, Dict, Any

from .user_sim import JUDGE_SYSTEM_PROMPT


def build_single_turn_messages(
    post: Dict[str, Any],
    storyboard: List[Dict[str, Any]],
    n_aspects: int = None,
) -> List[Dict[str, str]]:
    """Build messages for single-turn condition.

    Concatenates the first n_aspects aspects into a single user message
    after the title. If n_aspects is None, includes all aspects.

    This is called at each polling step t to build a fresh single-turn
    prompt with aspects 1..t, matching the same information the passive
    condition has seen at that step.

    Returns:
        List of chat messages [system, user]
    """
    aspects = storyboard[:n_aspects] if n_aspects is not None else storyboard
    storyboard_text = "\n\n".join(a["content"] for a in aspects)
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Title: {post['title']}\n\n{storyboard_text}"},
    ]


def build_passive_title_messages(
    post: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Build initial messages for multi-turn passive condition.

    Starts with just the title. Aspects are added one at a time
    via add_passive_aspect().

    Returns:
        List of chat messages [system, user_title]
    """
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Title: {post['title']}"},
    ]


def add_passive_aspect(
    messages: List[Dict[str, str]],
    aspect: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Append a new aspect to the passive conversation.

    Adds assistant="Noted." followed by user=aspect content.

    Args:
        messages: Current conversation messages (will be copied)
        aspect: Aspect dict with 'content' key

    Returns:
        New message list with the aspect appended
    """
    new_messages = [m.copy() for m in messages]
    new_messages.append({"role": "assistant", "content": "Noted."})
    new_messages.append({"role": "user", "content": aspect["content"]})
    return new_messages

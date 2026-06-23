"""Low-level text helpers shared across delivery engines and user simulators."""

import re

# User-sim outputs wrap their in-character message in <message> tags (some
# models emit <msg>). This boundary separates the model's processing of the
# steering instruction from the in-persona output it should actually send.
MESSAGE_RE = re.compile(r"<(?:message|msg)>(.*?)(?:</(?:message|msg)>|$)", re.DOTALL)


def extract_message(raw: str) -> str:
    """Pull the in-character message out of a user-sim generation.

    Falls back to the stripped raw text if the model emitted no tags.
    """
    m = MESSAGE_RE.search(raw)
    return m.group(1).strip() if m else raw.strip()

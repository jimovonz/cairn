"""Response hash computation and verification for Cairn.

Hash function: sum of first-letter values (a=1..z=26) across sentences
in the response text, excluding the <memory> block and fenced code blocks.

Sentence boundaries: ". " "! " "? " followed by a capital letter,
plus double-newline (paragraph break). Code block boundaries are
treated as sentence breaks.
"""

from __future__ import annotations

import re
from typing import Optional


def compute_response_hash(text: str) -> int:
    """Compute the sentence-first-letter hash for a response.

    Args:
        text: Full assistant response text (memory block will be stripped).

    Returns:
        Integer hash value (sum of a=1..z=26 for first letter of each sentence).
    """
    # Strip memory block
    stripped = re.sub(r'<memory>.*?</memory>', '', text, flags=re.DOTALL)
    # Strip fenced code blocks — replace with sentence boundary marker
    stripped = re.sub(r'```.*?```', '. ', stripped, flags=re.DOTALL)
    stripped = stripped.strip()

    if not stripped:
        return 0

    # Normalise paragraph breaks to sentence boundaries
    stripped = re.sub(r'\n\n+', '. ', stripped)
    # Split on sentence boundaries: .!? followed by whitespace then capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', stripped)
    sentences = [s.strip() for s in sentences if s.strip()]

    total = 0
    for s in sentences:
        ch = s[0].lower()
        if ch.isalpha():
            total += ord(ch) - ord('a') + 1

    return total


def verify_hash(text: str, claimed: int) -> tuple[bool, int]:
    """Verify a claimed hash against the actual response hash.

    Args:
        text: Full assistant response text.
        claimed: The hash value claimed in the memory block.

    Returns:
        Tuple of (match: bool, actual: int).
    """
    actual = compute_response_hash(text)
    return actual == claimed, actual

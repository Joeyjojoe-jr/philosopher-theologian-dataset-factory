"""Utility for extracting simple claims from text."""

import re
from typing import List


def extract_claims(text: str) -> List[str]:
    """Split *text* into sentence-like claims.

    The function performs a naive split on punctuation and returns
    non-empty trimmed sentences.
    """
    sentences = [s.strip() for s in re.split(r"[.!?]\s*", text.strip()) if s.strip()]
    return sentences

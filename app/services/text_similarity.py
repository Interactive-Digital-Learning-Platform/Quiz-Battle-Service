# Copied from Personalized-Quiz-Service's quiz_service.py (was
# _significant_words/_jaccard_similarity/_is_near_duplicate there) -- this
# service's own near-duplicate-question primitive now, made public since it
# no longer sits alongside the private internals of a larger module. Keep in
# sync by hand if the upstream logic changes.

import re

_STOP_WORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
        "been", "being", "in", "on", "at", "to", "for", "of", "with", "by",
        "from", "as", "that", "this", "these", "those", "it", "its", "which",
        "what", "who", "whom", "how", "why", "when", "where",
    }
)


# A word, number or math term: "energy", "20", "3.5", "2x", "x^2".
_TOKEN = re.compile(r"[a-z0-9]+(?:[.^][a-z0-9]+)*")


def significant_words(text: str) -> frozenset[str]:
    """The words that tell two questions apart: anything containing a digit (numbers, "2x", "x^2", "3.5") plus
    words of 3+ letters that aren't stop words. Numbers and math terms count -- for maths they ARE the question:
    "Find x if 3x + 5 = 20" and "Find x if 2x - 7 = 15" share every ordinary word but are different questions."""
    words = set()
    for token in _TOKEN.findall(text.lower()):
        if any(ch.isdigit() for ch in token) or (len(token) > 2 and token not in _STOP_WORDS):
            words.add(token)
    return frozenset(words)


def jaccard_similarity(a: str, b: str) -> float:
    wa, wb = significant_words(a), significant_words(b)
    union = wa | wb
    if not union:
        return 0.0
    return len(wa & wb) / len(union)


def is_near_duplicate(candidate: str, existing: list[str], threshold: float = 0.55) -> bool:
    # Two questions count as "the same" if they share 55%+ of their
    # meaningful words (stop words and short filler words ignored) — catches
    # Groq rephrasing a question it was already asked not to repeat.
    for ex in existing:
        if jaccard_similarity(candidate, ex) >= threshold:
            return True
    return False

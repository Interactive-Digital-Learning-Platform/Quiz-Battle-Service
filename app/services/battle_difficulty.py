"""
app/services/battle_difficulty.py
──────────────────────────────────────
Mixed-difficulty question distribution for a BattleMatch -- pure, no DB
access. Mirrors Personalized-Quiz-Service's own challenge-zone quiz
generation (app/services/difficulty_mastery_engine.py's
CHALLENGE_ZONE_DISTRIBUTION / allocate_question_counts): a fixed rating band
maps to a {easy, medium, hard} percentage split, then the percentages are
converted into exact integer per-tier counts for one match's fixed
BATTLE_QUESTION_COUNT. `allocate_question_counts` below is copied verbatim
from that module -- keep in sync by hand if the upstream logic changes.

5 bands (vs. quiz_service's 3) to match the product spec's rating->difficulty
table. Every band except the two extremes blends at least two difficulties,
and every band boundary changes each tier's share by a modest amount (never
a full 0%->100% flip) -- deliberately avoiding a sharp jump in what a player
gets served for crossing one rating point at a league boundary. The two
extreme bands (below 1200 / 2100+) still carry a "mostly" secondary-tier
tail rather than being 100% pure, for the same reason.
"""
BATTLE_DIFFICULTY_DISTRIBUTION: dict[str, dict[str, float]] = {
    "Mostly Easy":    {"easy": 0.70, "medium": 0.30, "hard": 0.00},
    "Easy + Medium":  {"easy": 0.50, "medium": 0.45, "hard": 0.05},
    "Mostly Medium":  {"easy": 0.15, "medium": 0.70, "hard": 0.15},
    "Medium + Hard":  {"easy": 0.05, "medium": 0.45, "hard": 0.50},
    "Mostly Hard":    {"easy": 0.00, "medium": 0.30, "hard": 0.70},
}


def get_difficulty_label(average_rating: float) -> str:
    """The human-readable band label for a match's average rating -- stored
    directly on BattleMatch.difficulty (shown to both players before
    questions are even selected, e.g. in the matched-queue response).
    """
    if average_rating < 1200:
        return "Mostly Easy"
    if average_rating < 1500:
        return "Easy + Medium"
    if average_rating < 1800:
        return "Mostly Medium"
    if average_rating < 2100:
        return "Medium + Hard"
    return "Mostly Hard"


def get_difficulty_distribution(average_rating: float) -> dict[str, float]:
    """The {easy, medium, hard} percentage split for a match's average
    rating -- feed this into allocate_question_counts() to get exact
    per-tier question counts for BATTLE_QUESTION_COUNT.
    """
    return dict(BATTLE_DIFFICULTY_DISTRIBUTION[get_difficulty_label(average_rating)])


def allocate_question_counts(distribution: dict[str, float], total: int) -> dict[str, int]:
    """Copied from Personalized-Quiz-Service's difficulty_mastery_engine.py.
    Converts a {key: percentage} distribution into integer counts that sum
    to exactly `total`, using the largest-remainder method (floor each
    share, then hand out the leftover questions one at a time to whichever
    keys had the largest fractional remainder) so rounding never drops or
    invents a question. Keys with a zero resulting count are omitted.
    """
    if total <= 0 or not distribution:
        return {}
    raw_shares = {key: pct * total for key, pct in distribution.items()}
    counts = {key: int(share) for key, share in raw_shares.items()}
    remainder = total - sum(counts.values())
    if remainder > 0:
        by_fractional_part = sorted(
            distribution.keys(), key=lambda k: raw_shares[k] - counts[k], reverse=True
        )
        for key in by_fractional_part[:remainder]:
            counts[key] += 1
    return {key: count for key, count in counts.items() if count > 0}

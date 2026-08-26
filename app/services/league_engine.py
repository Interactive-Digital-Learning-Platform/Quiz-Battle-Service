"""
app/services/league_engine.py
──────────────────────────────
League bucketing for competitive subject ratings -- pure, deterministic, no
DB access, no async. Mirrors difficulty_mastery_engine.py's "pure
calculation functions" framing.

Thresholds live in app/core/config.py's BATTLE_LEAGUE_* settings, not here --
this module only implements the lookup so nothing hardcodes a rating cutoff
outside config. League is always computed from rating on read and never
stored on CompetitiveSubjectRating, so there is no stale-league state to
keep in sync when rating changes.
"""
from app.core.config import settings


def _build_leagues_descending() -> list[tuple[int, str]]:
    return [
        (settings.BATTLE_LEAGUE_DIAMOND_MIN, "Diamond"),
        (settings.BATTLE_LEAGUE_PLATINUM_MIN, "Platinum"),
        (settings.BATTLE_LEAGUE_GOLD_MIN, "Gold"),
        (settings.BATTLE_LEAGUE_SILVER_MIN, "Silver"),
        (0, "Bronze"),
    ]


# Built once at import time -- rebuilding per call would be wasted work for
# values that are effectively static config.
_LEAGUES_DESCENDING = _build_leagues_descending()


def get_league(rating: int) -> str:
    """Return the league name for a given rating (first matching threshold wins)."""
    for min_rating, name in _LEAGUES_DESCENDING:
        if rating >= min_rating:
            return name
    return "Bronze"  # unreachable given the 0-rating floor entry above, kept for clarity

"""
app/services/matchmaking_window.py
──────────────────────────────────────
Rating-window widening for 1v1 matchmaking -- pure, deterministic, no DB
access, no async. Mirrors league_engine.py's style.

A candidate pair is a valid match iff
    abs(rating_a - rating_b) <= min(get_rating_window(wait_a), get_rating_window(wait_b))
-- see app/services/matchmaking_service.py for where that rule is applied.
"""
from app.core.config import settings


def get_rating_window(wait_seconds: float) -> int:
    """Max rating difference this player currently tolerates, given how long
    they've been waiting. Never exceeds BATTLE_MATCHMAKING_TIER4_WINDOW_RATING
    -- that value doubles as the hard cap on ranked rating difference by
    construction, so there is no separate max-diff setting to keep in sync.
    """
    if wait_seconds < settings.BATTLE_MATCHMAKING_TIER1_MAX_WAIT_SECONDS:
        return settings.BATTLE_MATCHMAKING_TIER1_WINDOW_RATING
    if wait_seconds < settings.BATTLE_MATCHMAKING_TIER2_MAX_WAIT_SECONDS:
        return settings.BATTLE_MATCHMAKING_TIER2_WINDOW_RATING
    if wait_seconds < settings.BATTLE_MATCHMAKING_TIER3_MAX_WAIT_SECONDS:
        return settings.BATTLE_MATCHMAKING_TIER3_WINDOW_RATING
    return settings.BATTLE_MATCHMAKING_TIER4_WINDOW_RATING

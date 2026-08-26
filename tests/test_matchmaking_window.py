"""
tests/test_matchmaking_window.py
────────────────────────────────────
Unit tests for the pure rating-window function in
app/services/matchmaking_window.py -- no DB, no Redis, no async.
"""
from app.core.config import settings
from app.services import matchmaking_window as window


def test_default_tiers_assumed_by_this_test_file():
    assert settings.BATTLE_MATCHMAKING_TIER1_WINDOW_RATING == 100
    assert settings.BATTLE_MATCHMAKING_TIER1_MAX_WAIT_SECONDS == 5.0
    assert settings.BATTLE_MATCHMAKING_TIER2_WINDOW_RATING == 150
    assert settings.BATTLE_MATCHMAKING_TIER2_MAX_WAIT_SECONDS == 10.0
    assert settings.BATTLE_MATCHMAKING_TIER3_WINDOW_RATING == 250
    assert settings.BATTLE_MATCHMAKING_TIER3_MAX_WAIT_SECONDS == 20.0
    assert settings.BATTLE_MATCHMAKING_TIER4_WINDOW_RATING == 400


def test_tier1_at_zero_wait():
    assert window.get_rating_window(0.0) == 100


def test_tier1_just_under_boundary():
    assert window.get_rating_window(4.99) == 100


def test_tier2_at_boundary():
    assert window.get_rating_window(5.0) == 150


def test_tier2_just_under_next_boundary():
    assert window.get_rating_window(9.99) == 150


def test_tier3_at_boundary():
    assert window.get_rating_window(10.0) == 250


def test_tier3_just_under_next_boundary():
    assert window.get_rating_window(19.99) == 250


def test_tier4_at_boundary():
    assert window.get_rating_window(20.0) == 400


def test_tier4_stays_at_max_far_beyond_boundary():
    assert window.get_rating_window(300.0) == 400


def test_never_exceeds_tier4_hard_cap():
    for wait in [0, 1, 4.9, 5, 7, 9.9, 10, 15, 19.9, 20, 25, 60, 600]:
        assert window.get_rating_window(wait) <= settings.BATTLE_MATCHMAKING_TIER4_WINDOW_RATING

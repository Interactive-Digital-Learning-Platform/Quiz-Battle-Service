"""
tests/test_league_engine.py
──────────────────────────────
Unit tests for the pure league-bucketing function in
app/services/league_engine.py -- no DB, no async, just the lookup table.
"""
from app.core.config import settings
from app.services import league_engine


def test_default_thresholds_assumed_by_this_test_file():
    # If these ever change, the boundary-value tests below need updating too.
    assert settings.BATTLE_LEAGUE_SILVER_MIN == 1200
    assert settings.BATTLE_LEAGUE_GOLD_MIN == 1500
    assert settings.BATTLE_LEAGUE_PLATINUM_MIN == 1800
    assert settings.BATTLE_LEAGUE_DIAMOND_MIN == 2100


def test_zero_and_negative_ratings_are_bronze():
    assert league_engine.get_league(0) == "Bronze"
    assert league_engine.get_league(-50) == "Bronze"


def test_bronze_silver_boundary():
    assert league_engine.get_league(1199) == "Bronze"
    assert league_engine.get_league(1200) == "Silver"


def test_silver_gold_boundary():
    assert league_engine.get_league(1499) == "Silver"
    assert league_engine.get_league(1500) == "Gold"


def test_gold_platinum_boundary():
    assert league_engine.get_league(1799) == "Gold"
    assert league_engine.get_league(1800) == "Platinum"


def test_platinum_diamond_boundary():
    assert league_engine.get_league(2099) == "Platinum"
    assert league_engine.get_league(2100) == "Diamond"


def test_very_high_rating_is_diamond():
    assert league_engine.get_league(9999) == "Diamond"


def test_initial_rating_is_bronze():
    assert league_engine.get_league(settings.BATTLE_INITIAL_RATING) == "Bronze"

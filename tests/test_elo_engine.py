"""
tests/test_elo_engine.py
────────────────────────────
Unit tests for the pure Elo rating math in app/services/elo_engine.py -- no
DB, no async, no CompetitiveSubjectRating dependency.
"""
import pytest

from app.services import elo_engine

# ─────────────────────────────────────────────────────────────────────────────
# expected_score
# ─────────────────────────────────────────────────────────────────────────────

def test_expected_score_equal_ratings_is_half():
    assert elo_engine.expected_score(1000, 1000) == pytest.approx(0.5)


def test_expected_score_higher_rated_player_favored():
    assert elo_engine.expected_score(1400, 1000) > 0.5
    assert elo_engine.expected_score(1000, 1400) < 0.5


def test_expected_score_symmetric_complement():
    a_vs_b = elo_engine.expected_score(1200, 1000)
    b_vs_a = elo_engine.expected_score(1000, 1200)
    assert a_vs_b + b_vs_a == pytest.approx(1.0)


def test_expected_score_400_point_gap_matches_known_elo_constant():
    # A 400-point rating gap is Elo's defining constant: the higher-rated
    # player's expected score is exactly 10/11 ~= 0.909.
    assert elo_engine.expected_score(1400, 1000) == pytest.approx(10 / 11, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# compute_elo_update -- win/draw/loss
# ─────────────────────────────────────────────────────────────────────────────

def test_win_against_equal_rating_gains_half_k():
    result = elo_engine.compute_elo_update(1000, 1000, elo_engine.RESULT_WIN, k_factor=32)
    assert result.delta_a == 16  # k * (1 - 0.5) = 16
    assert result.delta_b == -16
    assert result.new_rating_a == 1016
    assert result.new_rating_b == 984


def test_loss_against_equal_rating_loses_half_k():
    result = elo_engine.compute_elo_update(1000, 1000, elo_engine.RESULT_LOSS, k_factor=32)
    assert result.delta_a == -16
    assert result.delta_b == 16


def test_draw_against_equal_rating_is_a_no_op():
    result = elo_engine.compute_elo_update(1000, 1000, elo_engine.RESULT_DRAW, k_factor=32)
    assert result.delta_a == 0
    assert result.delta_b == 0
    assert result.new_rating_a == 1000
    assert result.new_rating_b == 1000


def test_underdog_win_gains_more_than_favorite_win():
    underdog_win = elo_engine.compute_elo_update(1000, 1400, elo_engine.RESULT_WIN, k_factor=32)
    favorite_win = elo_engine.compute_elo_update(1400, 1000, elo_engine.RESULT_WIN, k_factor=32)
    assert underdog_win.delta_a > favorite_win.delta_a


def test_invalid_result_raises():
    with pytest.raises(ValueError):
        elo_engine.compute_elo_update(1000, 1000, 0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Zero-sum symmetry (away from the min_rating floor)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "rating_a,rating_b,result_a",
    [
        (1000, 1000, 1.0),
        (1000, 1000, 0.5),
        (1000, 1000, 0.0),
        (1400, 1000, 1.0),
        (1400, 1000, 0.0),
        (800, 1600, 1.0),
        (800, 1600, 0.0),
        (2000, 2100, 0.5),
    ],
)
def test_deltas_are_exact_zero_sum_away_from_floor(rating_a, rating_b, result_a):
    result = elo_engine.compute_elo_update(rating_a, rating_b, result_a, k_factor=32)
    assert result.delta_a + result.delta_b == 0


def test_min_rating_floor_breaks_symmetry_as_documented():
    # A player already at the floor who loses can't drop further -- the
    # opponent's gain is no longer offset by an equal loss. This is the one
    # documented case where zero-sum symmetry legitimately doesn't hold.
    # Ratings are close enough (0 vs 50) that the un-clamped delta would be
    # a double-digit swing -- far apart ratings (e.g. 0 vs 1000) round to a
    # 0 delta for both players regardless of the floor, which wouldn't
    # actually exercise the clamp this test is checking.
    result = elo_engine.compute_elo_update(0, 50, elo_engine.RESULT_LOSS, k_factor=32, min_rating=0)
    assert result.new_rating_a == 0
    assert result.delta_a == 0
    assert result.delta_b > 0
    assert result.delta_a + result.delta_b != 0


# ─────────────────────────────────────────────────────────────────────────────
# K-factor scaling
# ─────────────────────────────────────────────────────────────────────────────

def test_k_factor_scales_delta_magnitude_linearly():
    low_k = elo_engine.compute_elo_update(1000, 1000, elo_engine.RESULT_WIN, k_factor=32)
    high_k = elo_engine.compute_elo_update(1000, 1000, elo_engine.RESULT_WIN, k_factor=64)
    assert high_k.delta_a == 2 * low_k.delta_a


def test_default_k_factor_comes_from_settings():
    from app.core.config import settings

    with_default = elo_engine.compute_elo_update(1000, 1000, elo_engine.RESULT_WIN)
    explicit = elo_engine.compute_elo_update(
        1000, 1000, elo_engine.RESULT_WIN, k_factor=settings.BATTLE_ELO_K_FACTOR
    )
    assert with_default == explicit

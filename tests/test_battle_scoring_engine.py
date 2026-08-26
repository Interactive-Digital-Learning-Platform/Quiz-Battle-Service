"""
tests/test_battle_scoring_engine.py
────────────────────────────────────────
Unit tests for the pure scoring/winner-determination math in
app/services/battle_scoring_engine.py -- no DB, no async.
"""
from app.services.battle_scoring_engine import (
    ParticipantOutcome,
    compute_question_score,
    compute_speed_bonus,
    compute_streak_bonus,
    determine_winner,
)

# ─────────────────────────────────────────────────────────────────────────────
# compute_speed_bonus
# ─────────────────────────────────────────────────────────────────────────────

def test_speed_bonus_tier_boundaries():
    assert compute_speed_bonus(0.0) == 30
    assert compute_speed_bonus(5.0) == 30
    assert compute_speed_bonus(5.01) == 20
    assert compute_speed_bonus(10.0) == 20
    assert compute_speed_bonus(10.01) == 10
    assert compute_speed_bonus(15.0) == 10
    assert compute_speed_bonus(15.01) == 0
    assert compute_speed_bonus(20.0) == 0
    assert compute_speed_bonus(25.0) == 0


def test_speed_bonus_negative_clamps_to_zero_time():
    assert compute_speed_bonus(-5.0) == 30


# ─────────────────────────────────────────────────────────────────────────────
# compute_streak_bonus
# ─────────────────────────────────────────────────────────────────────────────

def test_streak_bonus_tier_boundaries():
    assert compute_streak_bonus(1) == 0
    assert compute_streak_bonus(2) == 5
    assert compute_streak_bonus(3) == 10
    assert compute_streak_bonus(4) == 15
    assert compute_streak_bonus(5) == 20


def test_streak_bonus_capped_beyond_tier4():
    assert compute_streak_bonus(100) == 20


# ─────────────────────────────────────────────────────────────────────────────
# compute_question_score
# ─────────────────────────────────────────────────────────────────────────────

def test_wrong_answer_is_always_zero():
    result = compute_question_score(is_correct=False, response_time_seconds=1.0, streak_after_this_answer=5)
    assert result.is_correct is False
    assert result.base_score == 0
    assert result.speed_bonus == 0
    assert result.streak_bonus == 0
    assert result.total_score == 0


def test_correct_answer_sums_all_three():
    result = compute_question_score(is_correct=True, response_time_seconds=3.0, streak_after_this_answer=3)
    assert result.is_correct is True
    assert result.base_score == 100
    assert result.speed_bonus == 30
    assert result.streak_bonus == 10
    assert result.total_score == 140


# ─────────────────────────────────────────────────────────────────────────────
# determine_winner
# ─────────────────────────────────────────────────────────────────────────────

def test_spec_worked_example_correctness_beats_score():
    # 9/10 correct + 1040 pts beats 8/10 correct + 1090 pts.
    a = ParticipantOutcome(user_id=1, correct_count=9, total_score=1040, total_response_time_ms=50000)
    b = ParticipantOutcome(user_id=2, correct_count=8, total_score=1090, total_response_time_ms=40000)
    result = determine_winner(a, b)
    assert result.winner_user_id == 1
    assert result.result_for_a == 1.0


def test_tiebreak_on_score_when_correct_count_equal():
    a = ParticipantOutcome(user_id=1, correct_count=8, total_score=900, total_response_time_ms=50000)
    b = ParticipantOutcome(user_id=2, correct_count=8, total_score=950, total_response_time_ms=50000)
    result = determine_winner(a, b)
    assert result.winner_user_id == 2
    assert result.result_for_a == 0.0


def test_tiebreak_on_response_time_when_correct_and_score_equal():
    a = ParticipantOutcome(user_id=1, correct_count=8, total_score=900, total_response_time_ms=45000)
    b = ParticipantOutcome(user_id=2, correct_count=8, total_score=900, total_response_time_ms=50000)
    result = determine_winner(a, b)
    assert result.winner_user_id == 1
    assert result.result_for_a == 1.0


def test_full_tie_is_draw():
    a = ParticipantOutcome(user_id=1, correct_count=8, total_score=900, total_response_time_ms=50000)
    b = ParticipantOutcome(user_id=2, correct_count=8, total_score=900, total_response_time_ms=50000)
    result = determine_winner(a, b)
    assert result.winner_user_id is None
    assert result.result_for_a == 0.5

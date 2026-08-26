"""
app/services/battle_scoring_engine.py
──────────────────────────────────────────
Pure, deterministic battle scoring/winner-determination math -- no DB, no
async, no knowledge of BattleMatch/BattleAnswer models. Mirrors
elo_engine.py's "compute here, persist elsewhere" split.

Distinct from app/services/scoring_service.py, which is unrelated
single-player quiz analytics (accuracy/median/trend helpers), not battle
scoring.
"""
from dataclasses import dataclass

from app.core.config import settings


def compute_speed_bonus(response_time_seconds: float) -> int:
    """Tiered speed bonus, capped. response_time_seconds is SERVER-computed
    (question_presented_at -> answer_received_at), never client-reported.
    Negative input (shouldn't happen) clamps to 0. Upper-bound-inclusive
    tiers -- exactly 5.0s still gets tier1.
    """
    t = max(0.0, response_time_seconds)
    if t <= settings.BATTLE_SPEED_BONUS_TIER1_MAX_SECONDS:
        return settings.BATTLE_SPEED_BONUS_TIER1_POINTS
    if t <= settings.BATTLE_SPEED_BONUS_TIER2_MAX_SECONDS:
        return settings.BATTLE_SPEED_BONUS_TIER2_POINTS
    if t <= settings.BATTLE_SPEED_BONUS_TIER3_MAX_SECONDS:
        return settings.BATTLE_SPEED_BONUS_TIER3_POINTS
    return settings.BATTLE_SPEED_BONUS_TIER4_POINTS


def compute_streak_bonus(streak_after_this_correct_answer: int) -> int:
    """streak_after_this_correct_answer is the consecutive-correct count
    INCLUDING the answer just graded (1 = first correct in a fresh streak,
    contributes 0). Caller must not call this for a wrong answer.
    """
    n = streak_after_this_correct_answer
    if n >= settings.BATTLE_STREAK_BONUS_TIER4_MIN_STREAK:
        return settings.BATTLE_STREAK_BONUS_TIER4_POINTS
    if n >= settings.BATTLE_STREAK_BONUS_TIER3_MIN_STREAK:
        return settings.BATTLE_STREAK_BONUS_TIER3_POINTS
    if n >= settings.BATTLE_STREAK_BONUS_TIER2_MIN_STREAK:
        return settings.BATTLE_STREAK_BONUS_TIER2_POINTS
    if n >= settings.BATTLE_STREAK_BONUS_TIER1_MIN_STREAK:
        return settings.BATTLE_STREAK_BONUS_TIER1_POINTS
    return 0


@dataclass(frozen=True)
class QuestionScoreResult:
    is_correct: bool
    base_score: int
    speed_bonus: int
    streak_bonus: int
    total_score: int


def compute_question_score(
    *, is_correct: bool, response_time_seconds: float, streak_after_this_answer: int,
) -> QuestionScoreResult:
    """Wrong answer -> everything 0 (no negative points). Caller is
    responsible for passing the correct streak_after_this_answer (see
    battle_gameplay_service for how it's derived from prior BattleAnswer
    rows).
    """
    if not is_correct:
        return QuestionScoreResult(False, 0, 0, 0, 0)
    base = settings.BATTLE_SCORE_BASE_CORRECT
    speed = compute_speed_bonus(response_time_seconds)
    streak = compute_streak_bonus(streak_after_this_answer)
    return QuestionScoreResult(True, base, speed, streak, base + speed + streak)


@dataclass(frozen=True)
class ParticipantOutcome:
    user_id: int
    correct_count: int
    total_score: int
    # Priority-3 tiebreak per spec: summed over correct answers only.
    total_response_time_ms: int


@dataclass(frozen=True)
class WinnerResult:
    winner_user_id: int | None  # None = draw
    result_for_a: float  # 1.0 / 0.5 / 0.0 -- elo_engine's result_a convention


def determine_winner(a: ParticipantOutcome, b: ParticipantOutcome) -> WinnerResult:
    """Exact 4-level priority from the product spec:
    1. most correct answers wins
    2. tie -> highest total battle score (base+speed+streak, summed over
       all answers -- wrong answers already contribute 0) wins
    3. tie -> lowest total response time (summed over correct answers only)
       wins
    4. tie -> draw
    Worked example this satisfies: 9/10 correct + 1040 pts beats 8/10
    correct + 1090 pts -- correctness strictly dominates bonus-inflated score.
    """
    if a.correct_count != b.correct_count:
        winner = a if a.correct_count > b.correct_count else b
    elif a.total_score != b.total_score:
        winner = a if a.total_score > b.total_score else b
    elif a.total_response_time_ms != b.total_response_time_ms:
        winner = a if a.total_response_time_ms < b.total_response_time_ms else b
    else:
        return WinnerResult(winner_user_id=None, result_for_a=0.5)
    result_for_a = 1.0 if winner.user_id == a.user_id else 0.0
    return WinnerResult(winner_user_id=winner.user_id, result_for_a=result_for_a)

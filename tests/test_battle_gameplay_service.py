"""
tests/test_battle_gameplay_service.py
──────────────────────────────────────────
Integration tests (in-memory DB, no HTTP, no real sleeps) for
app/services/battle_gameplay_service.py -- ready-gated activation
("waiting"->"countdown"->"active"), timestamp-derived question progression,
answer submission/validation, match completion, and rating application.
Question selection is mocked (deterministic, fixed question set) so tests
don't depend on Groq/pool machinery -- that's covered separately in
tests/test_battle_question_service.py. Ready/forfeit/disconnect-specific
scenarios (the double-ready race, forfeit outcomes, Redis presence helpers)
live in tests/test_battle_ready_forfeit_service.py.

Timing is controlled entirely by backdating BattleMatch.started_at via direct
SQL UPDATE (never real sleeps) -- SQLAlchemy 2.0's ORM-enabled `update()`
auto-synchronizes the session's identity map, so a subsequent `db.get()`
picks up the change without an explicit refresh (verified directly against
aiosqlite before writing these tests).
"""
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy import select, update

from app.core.config import settings
from app.models.battle_match import (
    BattleAnswer,
    BattleMatch,
    BattleMatchQuestion,
    BattleParticipant,
)
from app.models.competitive_rating import CompetitiveSubjectRating
from app.models.question import Question
from app.services import battle_gameplay_service
from app.services.battle_question_service import BattleQuestionSelectionError
from app.services.elo_engine import compute_elo_update
from app.services.user_service import get_or_create_user

N = settings.BATTLE_QUESTION_COUNT
Q = settings.BATTLE_QUESTION_TIME_LIMIT_SECONDS
TOTAL = N * Q


async def _make_user(db, clerk_id: str):
    return await get_or_create_user(db, clerk_id)


async def _seed_rating(db, user_id: int, subject: str, rating: int) -> CompetitiveSubjectRating:
    row = CompetitiveSubjectRating(user_id=user_id, subject=subject, rating=rating, highest_rating=rating)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def _create_waiting_match(db, *, subject, difficulty, user_a_id, rating_a, user_b_id, rating_b) -> BattleMatch:
    match = BattleMatch(subject=subject, status="waiting", difficulty=difficulty)
    db.add(match)
    await db.flush()
    db.add(BattleParticipant(match_id=match.id, user_id=user_a_id, rating_before=rating_a))
    db.add(BattleParticipant(match_id=match.id, user_id=user_b_id, rating_before=rating_b))
    await db.commit()
    await db.refresh(match)
    return match


def _fixed_questions(count: int = N) -> list[Question]:
    return [
        Question(
            question=f"Fixed battle question {i}?", options=["A", "B", "C", "D"],
            correct_answer="A", subject="Mathematics", lesson="Algebra", difficulty="medium",
        )
        for i in range(count)
    ]


async def _connect_both(db, redis, match: BattleMatch, *, user_a_id: int, user_b_id: int) -> None:
    # Activation requires a participant to be BOTH ready AND WS-connected
    # (see mark_connected/_try_activate_if_ready_and_connected) -- these
    # tests aren't exercising that gate itself, so connect both upfront,
    # same as the real mobile client already does well before either taps
    # ready (see BATTLE_CONNECTION_READY_TIMEOUT_SECONDS's docstring).
    await battle_gameplay_service.mark_connected(db, redis, match_id=match.id, user_id=user_a_id)
    await battle_gameplay_service.mark_connected(db, redis, match_id=match.id, user_id=user_b_id)


async def _seed_and_activate(
    db, redis, match: BattleMatch, *, user_a_id: int, user_b_id: int,
    questions: list[Question] | None = None, viewer_id: int | None = None,
) -> tuple[dict, list[Question]]:
    """Both participants connect and ready up -- question selection (mocked
    here) runs during the SECOND mark_ready call, which is what actually
    triggers "waiting"->"countdown" (see _activate_countdown). started_at is
    then backdated 1s into the past so the very next state read's lazy
    "countdown"->"active" check (_ensure_match_progressed) fires immediately,
    without a real sleep.
    """
    questions = questions if questions is not None else _fixed_questions()
    db.add_all(questions)
    await db.flush()

    await _connect_both(db, redis, match, user_a_id=user_a_id, user_b_id=user_b_id)
    await battle_gameplay_service.mark_ready(db, redis, match_id=match.id, user_id=user_a_id)
    with patch(
        "app.services.battle_gameplay_service.select_battle_questions",
        new=AsyncMock(return_value=questions),
    ):
        await battle_gameplay_service.mark_ready(db, redis, match_id=match.id, user_id=user_b_id)

    await _set_started_at(db, match.id, datetime.now(UTC) - timedelta(seconds=1))
    state = await battle_gameplay_service.get_match_state(
        db, match_id=match.id, user_id=viewer_id if viewer_id is not None else user_a_id
    )
    return state, questions


async def _set_started_at(db, match_id: int, when: datetime) -> None:
    await db.execute(update(BattleMatch).where(BattleMatch.id == match_id).values(started_at=when))
    await db.commit()


async def _submit_and_grade(db, redis, match, *, order, user_id, question_id, selected_option):
    """Submits (upserts the Redis pending answer) then immediately forces
    the reveal-boundary grading pass for that question slot -- these tests
    don't run the real progression timer loop (battle_ws.py), so this
    simulates "the boundary just fired" deterministically regardless of
    wall-clock time. Grades BOTH participants together (same as
    production); returns the result for `user_id` specifically.
    """
    await battle_gameplay_service.submit_battle_answer(
        db, match_id=match.id, user_id=user_id, question_id=question_id,
        selected_option=selected_option, redis=redis,
    )
    event = await battle_gameplay_service._grade_and_reveal_question(db, redis, match=match, order=order)
    return next(r for r in event.results if r.user_id == user_id)


# ─────────────────────────────────────────────────────────────────────────────
# Activation (ready-gated)
# ─────────────────────────────────────────────────────────────────────────────

async def test_activation_selects_questions_and_transitions_to_active(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )

    state, questions = await _seed_and_activate(
        db_session, fake_redis, match, user_a_id=user_a.id, user_b_id=user_b.id
    )

    assert state["status"] == "active"
    # Lockstep play: only the single current question (index 0, right after
    # activation) is delivered.
    assert state["question_index"] == 0
    assert state["question"]["question_id"] == questions[0].id
    mq_rows = (
        await db_session.execute(select(BattleMatchQuestion).where(BattleMatchQuestion.match_id == match.id))
    ).scalars().all()
    assert len(mq_rows) == N
    assert {mq.question_id for mq in mq_rows} == {q.id for q in questions}

    refreshed = await db_session.get(BattleMatch, match.id)
    assert refreshed.status == "active"
    assert refreshed.started_at is not None


async def test_activation_is_idempotent_under_sequential_calls(db_session, fake_redis):
    # Sequential double-invocation, not asyncio.gather -- db_session's
    # StaticPool shares one SQLite connection, and true concurrent writes
    # there can raise spurious "database is locked" errors unrelated to the
    # CAS logic itself. The atomicity guarantee (UPDATE...WHERE...RETURNING)
    # was independently verified against aiosqlite directly (see plan).
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )

    state1, _ = await _seed_and_activate(
        db_session, fake_redis, match, user_a_id=user_a.id, user_b_id=user_b.id
    )
    started_at_1 = state1["started_at"]

    # Second poll on an already-active match must NOT re-select questions or
    # move started_at. (Double-ready-triggered-CAS idempotency itself is
    # covered in tests/test_battle_ready_forfeit_service.py.)
    with patch(
        "app.services.battle_gameplay_service.select_battle_questions",
        new=AsyncMock(side_effect=AssertionError("should not be called again")),
    ):
        state2 = await battle_gameplay_service.get_match_state(db_session, match_id=match.id, user_id=user_a.id)

    assert state2["status"] == "active"
    assert state2["started_at"] == started_at_1

    mq_rows = (
        await db_session.execute(select(BattleMatchQuestion).where(BattleMatchQuestion.match_id == match.id))
    ).scalars().all()
    assert len(mq_rows) == N  # not duplicated


async def test_ai_failure_cancels_match_cleanly(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    rating_a = await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    rating_b = await _seed_rating(db_session, user_b.id, "Mathematics", 1000)
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )

    await _connect_both(db_session, fake_redis, match, user_a_id=user_a.id, user_b_id=user_b.id)
    await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=user_a.id)
    with patch(
        "app.services.battle_gameplay_service.select_battle_questions",
        new=AsyncMock(side_effect=BattleQuestionSelectionError("not enough questions")),
    ):
        # This second ready is what triggers question selection.
        await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=user_b.id)

    state = await battle_gameplay_service.get_match_state(db_session, match_id=match.id, user_id=user_a.id)
    assert state["status"] == "cancelled"
    assert state["reason"] == battle_gameplay_service.CANCEL_REASON_AI_FAILURE

    await db_session.refresh(rating_a)
    await db_session.refresh(rating_b)
    assert rating_a.rating == 1000
    assert rating_b.rating == 1000
    assert rating_a.matches_played == 0
    assert rating_b.matches_played == 0


# ─────────────────────────────────────────────────────────────────────────────
# Lockstep active state: per-question timer, current question, progress arrays
# ─────────────────────────────────────────────────────────────────────────────

async def test_time_remaining_reflects_current_question_window(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    _, questions = await _seed_and_activate(db_session, fake_redis, match, user_a_id=user_a.id, user_b_id=user_b.id)

    # _seed_and_activate backdates started_at by 1s -- still well within
    # question 0's Q-second window.
    state = await battle_gameplay_service.get_match_state(db_session, match_id=match.id, user_id=user_a.id)
    assert state["question_index"] == 0
    assert state["time_remaining_seconds"] == pytest.approx(Q - 1, abs=0.5)

    # Advance into question 2's window.
    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=2 * Q + 1))
    state2 = await battle_gameplay_service.get_match_state(db_session, match_id=match.id, user_id=user_a.id)
    assert state2["question_index"] == 2
    assert state2["question"]["question_id"] == questions[2].id
    assert state2["time_remaining_seconds"] == pytest.approx(Q - 1, abs=0.5)


async def test_answered_current_question_reflected_in_progress_arrays(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    _, questions = await _seed_and_activate(db_session, fake_redis, match, user_a_id=user_a.id, user_b_id=user_b.id)

    await _submit_and_grade(
        db_session, fake_redis, match, order=0, user_id=user_a.id,
        question_id=questions[0].id, selected_option="A",
    )

    state = await battle_gameplay_service.get_match_state(db_session, match_id=match.id, user_id=user_a.id)
    assert state["has_answered_current_question"] is True
    assert state["my_progress"]["answered_count"] == 1
    assert state["my_answers"] == [{"question_order": 0, "is_correct": True, "answered": True}]
    # Grading locks BOTH participants together at the reveal boundary --
    # user_b never submitted anything, but the question is still graded
    # (as incorrect/unanswered) for them the instant user_a's answer locks.
    assert state["opponent_answers"] == [{"question_order": 0, "is_correct": False, "answered": False}]


# ─────────────────────────────────────────────────────────────────────────────
# Answer submission validation
# ─────────────────────────────────────────────────────────────────────────────

async def test_submit_answer_grading_mirrors_submit_quiz_rule(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    _, questions = await _seed_and_activate(db_session, fake_redis, match, user_a_id=user_a.id, user_b_id=user_b.id)

    # Case-insensitive, whitespace-stripped exact match, correct_answer="A".
    result = await _submit_and_grade(
        db_session, fake_redis, match, order=0, user_id=user_a.id,
        question_id=questions[0].id, selected_option="  a  ",
    )
    assert result.is_correct is True
    assert result.base_score == settings.BATTLE_SCORE_BASE_CORRECT


async def test_submit_answer_rejects_non_participant(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    outsider = await _make_user(db_session, "outsider")
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    _, questions = await _seed_and_activate(db_session, fake_redis, match, user_a_id=user_a.id, user_b_id=user_b.id)

    with pytest.raises(HTTPException) as exc_info:
        await battle_gameplay_service.submit_battle_answer(
            db_session, match_id=match.id, user_id=outsider.id, question_id=questions[0].id, selected_option="A",
        )
    assert exc_info.value.status_code == 403


async def test_submit_answer_rejects_a_non_current_question_in_the_same_match(db_session, fake_redis):
    # Lockstep play: only the CURRENT question (index 0, right after
    # activation) may be answered -- a later question in this same match's
    # own frozen set is rejected exactly like a foreign one.
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    _, questions = await _seed_and_activate(db_session, fake_redis, match, user_a_id=user_a.id, user_b_id=user_b.id)

    with pytest.raises(HTTPException) as exc_info:
        await battle_gameplay_service.submit_battle_answer(
            db_session, match_id=match.id, user_id=user_a.id,
            question_id=questions[1].id, selected_option="A",
        )
    assert exc_info.value.status_code == 400


async def test_submit_answer_rejects_question_from_a_foreign_match(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    await _seed_and_activate(db_session, fake_redis, match, user_a_id=user_a.id, user_b_id=user_b.id)

    foreign_question = Question(
        question="Foreign?", options=["A", "B"], correct_answer="A",
        subject="Mathematics", lesson="Algebra", difficulty="medium",
    )
    db_session.add(foreign_question)
    await db_session.commit()
    await db_session.refresh(foreign_question)

    with pytest.raises(HTTPException) as exc_info:
        await battle_gameplay_service.submit_battle_answer(
            db_session, match_id=match.id, user_id=user_a.id,
            question_id=foreign_question.id, selected_option="A",
        )
    assert exc_info.value.status_code == 400


async def test_resubmitting_before_lock_overwrites_the_pending_answer(db_session, fake_redis):
    # Editable answers: a player may change their mind any number of times
    # while the question is still open -- whichever submission is in effect
    # when grading runs (the reveal boundary) is the one that's authoritative.
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    _, questions = await _seed_and_activate(db_session, fake_redis, match, user_a_id=user_a.id, user_b_id=user_b.id)

    await battle_gameplay_service.submit_battle_answer(
        db_session, match_id=match.id, user_id=user_a.id, question_id=questions[0].id,
        selected_option="B", redis=fake_redis,
    )
    # Changed their mind to the correct option before the reveal boundary.
    await battle_gameplay_service.submit_battle_answer(
        db_session, match_id=match.id, user_id=user_a.id, question_id=questions[0].id,
        selected_option="A", redis=fake_redis,
    )
    event = await battle_gameplay_service._grade_and_reveal_question(db_session, fake_redis, match=match, order=0)
    result = next(r for r in event.results if r.user_id == user_a.id)
    assert result.is_correct is True  # graded on the LAST submission ("A"), not the first ("B")


async def test_resubmitting_after_lock_is_rejected(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    _, questions = await _seed_and_activate(db_session, fake_redis, match, user_a_id=user_a.id, user_b_id=user_b.id)

    await _submit_and_grade(
        db_session, fake_redis, match, order=0, user_id=user_a.id,
        question_id=questions[0].id, selected_option="A",
    )
    with pytest.raises(HTTPException) as exc_info:
        await battle_gameplay_service.submit_battle_answer(
            db_session, match_id=match.id, user_id=user_a.id, question_id=questions[0].id,
            selected_option="B", redis=fake_redis,
        )
    assert exc_info.value.status_code == 409


async def test_submit_answer_rejects_after_match_completed(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    await _seed_rating(db_session, user_b.id, "Mathematics", 1000)
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    _, questions = await _seed_and_activate(db_session, fake_redis, match, user_a_id=user_a.id, user_b_id=user_b.id)

    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=TOTAL + 1))
    state = await battle_gameplay_service.get_match_state(db_session, match_id=match.id, user_id=user_a.id)
    assert state["status"] == "completed"

    with pytest.raises(HTTPException) as exc_info:
        await battle_gameplay_service.submit_battle_answer(
            db_session, match_id=match.id, user_id=user_a.id, question_id=questions[0].id, selected_option="A",
        )
    assert exc_info.value.status_code == 409


# ─────────────────────────────────────────────────────────────────────────────
# Streak behavior (lockstep play: rebased to the frozen question_order --
# a question whose window closed with no answer from this player BREAKS the
# streak exactly like a wrong one, since there's no self-paced "I'll answer
# it later" option anymore)
# ─────────────────────────────────────────────────────────────────────────────

async def test_streak_resets_on_wrong_answer(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    _, questions = await _seed_and_activate(db_session, fake_redis, match, user_a_id=user_a.id, user_b_id=user_b.id)

    # Q0: correct, Q1: wrong, Q2: correct -- streak at Q2 should be fresh
    # (1), not accumulated from Q0, because Q1 was an actual wrong answer.
    # Backdating started_at between submissions is what advances the shared
    # "current question" for both players; each question is explicitly
    # graded (simulating the reveal boundary) before advancing to the next,
    # since _streak_before only sees already-graded BattleAnswer rows.
    await _submit_and_grade(
        db_session, fake_redis, match, order=0, user_id=user_a.id,
        question_id=questions[0].id, selected_option="A",
    )
    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=1 * Q + 1))
    await _submit_and_grade(
        db_session, fake_redis, match, order=1, user_id=user_a.id,
        question_id=questions[1].id, selected_option="WRONG",
    )
    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=2 * Q + 1))
    result = await _submit_and_grade(
        db_session, fake_redis, match, order=2, user_id=user_a.id,
        question_id=questions[2].id, selected_option="A",
    )
    assert result.is_correct is True
    assert result.streak_bonus == 0  # streak_after_this_answer=1 -> below tier1 threshold (2)


async def test_streak_resets_when_a_question_is_skipped(db_session, fake_redis):
    # A question whose window closes with no answer from this player is a
    # miss, not a neutral skip -- it breaks the streak exactly like an
    # actually-wrong answer (confirmed decision for the lockstep revert).
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    _, questions = await _seed_and_activate(db_session, fake_redis, match, user_a_id=user_a.id, user_b_id=user_b.id)

    # Answer Q0 correct, let Q1's window close with no answer at all, then
    # answer Q2 correct -- streak at Q2 must be fresh (1), not 2.
    await _submit_and_grade(
        db_session, fake_redis, match, order=0, user_id=user_a.id,
        question_id=questions[0].id, selected_option="A",
    )
    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=1 * Q + 1))
    # Q1's window closes with no answer from user_a at all -- still must be
    # explicitly graded (deferred-grading design) to produce the "missed"
    # BattleAnswer row that breaks the streak, same as the real progression
    # timer does when nobody submits anything before the reveal boundary.
    await battle_gameplay_service._grade_and_reveal_question(db_session, fake_redis, match=match, order=1)
    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=2 * Q + 1))
    result = await _submit_and_grade(
        db_session, fake_redis, match, order=2, user_id=user_a.id,
        question_id=questions[2].id, selected_option="A",
    )
    assert result.is_correct is True
    assert result.streak_bonus == 0  # streak_after_this_answer=1, Q1 was missed


# ─────────────────────────────────────────────────────────────────────────────
# Full match completion: winner, rating update, running average
# ─────────────────────────────────────────────────────────────────────────────

async def test_completion_is_idempotent_and_rating_matches_elo_engine(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    await _seed_rating(db_session, user_b.id, "Mathematics", 1000)
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    _, questions = await _seed_and_activate(db_session, fake_redis, match, user_a_id=user_a.id, user_b_id=user_b.id)

    # user_a answers all N correctly, advancing through each question's
    # window in turn; user_b answers none (misses every window). Each
    # question is explicitly graded (simulating the reveal boundary a real
    # progression timer would drive) before advancing to the next.
    for i in range(N):
        await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=i * Q + 1))
        await _submit_and_grade(
            db_session, fake_redis, match, order=i, user_id=user_a.id,
            question_id=questions[i].id, selected_option="A",
        )

    await fake_redis.hset(f"battle:user_state:{user_a.id}", "state", "in_match")
    await fake_redis.hset(f"battle:user_state:{user_b.id}", "state", "in_match")

    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=TOTAL + 1))
    state1 = await battle_gameplay_service.get_match_state(
        db_session, match_id=match.id, user_id=user_a.id, redis=fake_redis
    )
    assert state1["status"] == "completed"
    assert state1["result"]["winner_user_id"] == user_a.id
    assert state1["result"]["me"]["rating_after"] is not None

    # Without release_match_state, both players would stay "in_match"
    # forever (no TTL) and never be able to join a new queue again.
    assert await fake_redis.exists(f"battle:user_state:{user_a.id}") == 0
    assert await fake_redis.exists(f"battle:user_state:{user_b.id}") == 0

    rating_a_after_first = state1["result"]["me"]["rating_after"]

    # Second poll must be a pure no-op -- same result, no double-applied rating.
    state2 = await battle_gameplay_service.get_match_state(
        db_session, match_id=match.id, user_id=user_a.id, redis=fake_redis
    )
    assert state2["result"]["me"]["rating_after"] == rating_a_after_first

    rating_a_row = (
        await db_session.execute(
            select(CompetitiveSubjectRating).where(
                CompetitiveSubjectRating.user_id == user_a.id, CompetitiveSubjectRating.subject == "Mathematics",
            )
        )
    ).scalar_one()
    rating_b_row = (
        await db_session.execute(
            select(CompetitiveSubjectRating).where(
                CompetitiveSubjectRating.user_id == user_b.id, CompetitiveSubjectRating.subject == "Mathematics",
            )
        )
    ).scalar_one()

    # Matches applied exactly once, not twice.
    assert rating_a_row.matches_played == 1
    assert rating_b_row.matches_played == 1
    assert rating_a_row.wins == 1
    assert rating_b_row.losses == 1
    assert rating_a_row.current_win_streak == 1
    assert rating_a_row.highest_win_streak == 1

    # Rating math matches elo_engine exactly (equal starting ratings, A wins).
    expected = compute_elo_update(1000, 1000, 1.0)
    assert rating_a_row.rating == expected.new_rating_a
    assert rating_b_row.rating == expected.new_rating_b


async def test_completion_only_touches_the_matched_subject_rating(db_session, fake_redis):
    """A battle played in Mathematics must never move either player's
    Science rating -- CompetitiveSubjectRating rows are keyed per (user,
    subject), and battle_gameplay_service._get_or_build_rating_no_commit is
    scoped to match.subject, but nothing previously asserted the OTHER
    subject's row comes out byte-identical after a real completion.
    """
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    await _seed_rating(db_session, user_b.id, "Mathematics", 1000)

    # Untouched-control rows: same users, a DIFFERENT subject, deliberately
    # seeded with non-default stats so any accidental cross-subject write
    # would be obvious rather than masked by coincidental default values.
    science_a = await _seed_rating(db_session, user_a.id, "Science", 1234)
    science_a.wins, science_a.losses, science_a.matches_played = 3, 2, 5
    science_b = await _seed_rating(db_session, user_b.id, "Science", 1456)
    science_b.wins, science_b.losses, science_b.matches_played = 1, 4, 5
    await db_session.commit()

    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    _, questions = await _seed_and_activate(db_session, fake_redis, match, user_a_id=user_a.id, user_b_id=user_b.id)

    for i in range(N):
        await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=i * Q + 1))
        await _submit_and_grade(
            db_session, fake_redis, match, order=i, user_id=user_a.id,
            question_id=questions[i].id, selected_option="A",
        )

    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=TOTAL + 1))
    state = await battle_gameplay_service.get_match_state(
        db_session, match_id=match.id, user_id=user_a.id, redis=fake_redis
    )
    assert state["status"] == "completed"

    # The matched subject DID change (sanity check the completion actually ran).
    math_a = (
        await db_session.execute(
            select(CompetitiveSubjectRating).where(
                CompetitiveSubjectRating.user_id == user_a.id, CompetitiveSubjectRating.subject == "Mathematics",
            )
        )
    ).scalar_one()
    assert math_a.matches_played == 1

    # The unrelated subject's rows must be exactly as seeded.
    await db_session.refresh(science_a)
    await db_session.refresh(science_b)
    assert (science_a.rating, science_a.wins, science_a.losses, science_a.matches_played) == (1234, 3, 2, 5)
    assert (science_b.rating, science_b.wins, science_b.losses, science_b.matches_played) == (1456, 1, 4, 5)


async def test_running_average_response_time_formula(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    await _seed_rating(db_session, user_b.id, "Mathematics", 1000)
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    _, questions = await _seed_and_activate(db_session, fake_redis, match, user_a_id=user_a.id, user_b_id=user_b.id)

    # user_a answers question 0 near-instantly (fast), user_b never answers.
    await _submit_and_grade(
        db_session, fake_redis, match, order=0, user_id=user_a.id,
        question_id=questions[0].id, selected_option="A",
    )

    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=TOTAL + 1))
    await battle_gameplay_service.get_match_state(db_session, match_id=match.id, user_id=user_a.id)

    rating_a_row = (
        await db_session.execute(
            select(CompetitiveSubjectRating).where(
                CompetitiveSubjectRating.user_id == user_a.id, CompetitiveSubjectRating.subject == "Mathematics",
            )
        )
    ).scalar_one()
    # Started with total_answers=0, average_response_time=0.0 -- after 1
    # answered question, new average = that one answer's response time.
    assert rating_a_row.total_answers == 1
    answer_row = (
        await db_session.execute(
            select(BattleAnswer).where(BattleAnswer.match_id == match.id, BattleAnswer.user_id == user_a.id)
        )
    ).scalar_one()
    assert rating_a_row.average_response_time == pytest.approx(answer_row.response_time_ms / 1000.0, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Stale-"waiting"-match reaper (Phase 6)
# ─────────────────────────────────────────────────────────────────────────────

async def _set_created_at(db, match_id: int, when: datetime) -> None:
    await db.execute(update(BattleMatch).where(BattleMatch.id == match_id).values(created_at=when))
    await db.commit()


async def test_reap_cancels_stale_waiting_match_and_releases_state(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    await _set_created_at(
        db_session, match.id,
        datetime.now(UTC) - timedelta(seconds=settings.BATTLE_ORPHANED_MATCH_REAP_AFTER_SECONDS + 5),
    )
    await fake_redis.hset(f"battle:user_state:{user_a.id}", "state", "in_match")
    await fake_redis.hset(f"battle:user_state:{user_b.id}", "state", "in_match")

    reaped = await battle_gameplay_service.run_stale_match_reap_once(db_session, fake_redis)
    assert reaped == 1

    await db_session.refresh(match)
    assert match.status == "cancelled"
    assert match.cancel_reason == "Orphaned before matchmaking finalized."

    assert await fake_redis.exists(f"battle:user_state:{user_a.id}") == 0
    assert await fake_redis.exists(f"battle:user_state:{user_b.id}") == 0


async def test_reap_leaves_fresh_waiting_match_untouched(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    # created_at defaults to "now" -- well inside the reap threshold.

    reaped = await battle_gameplay_service.run_stale_match_reap_once(db_session, fake_redis)
    assert reaped == 0

    await db_session.refresh(match)
    assert match.status == "waiting"


async def test_reap_cancels_stale_active_match_and_releases_state(db_session, fake_redis):
    # Covers a match both players abandoned right after it went "active" --
    # nothing else ever revisits it once nobody is connected to trigger
    # _ensure_match_progressed, so without this branch it would be stuck
    # forever (this is exactly the scenario a live user hit: two accounts
    # matched, reached "active", both clients closed, and the second one
    # got a permanent "already in an active battle" 409 on every re-queue).
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    state, _ = await _seed_and_activate(db_session, fake_redis, match, user_a_id=user_a.id, user_b_id=user_b.id)
    assert state["status"] == "active"

    await fake_redis.hset(f"battle:user_state:{user_a.id}", "state", "in_match")
    await fake_redis.hset(f"battle:user_state:{user_b.id}", "state", "in_match")
    await _set_started_at(
        db_session, match.id,
        datetime.now(UTC) - timedelta(seconds=settings.BATTLE_ACTIVE_MATCH_REAP_AFTER_SECONDS + 5),
    )

    reaped = await battle_gameplay_service.run_stale_match_reap_once(db_session, fake_redis)
    assert reaped == 1

    await db_session.refresh(match)
    assert match.status == "cancelled"
    assert match.cancel_reason == "Abandoned -- no activity from either player."
    assert match.winner_user_id is None

    assert await fake_redis.exists(f"battle:user_state:{user_a.id}") == 0
    assert await fake_redis.exists(f"battle:user_state:{user_b.id}") == 0


async def test_reap_leaves_fresh_active_match_untouched(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    state, _ = await _seed_and_activate(db_session, fake_redis, match, user_a_id=user_a.id, user_b_id=user_b.id)
    assert state["status"] == "active"
    # started_at is only 1s in the past (set by _seed_and_activate) -- well
    # inside the reap threshold.

    reaped = await battle_gameplay_service.run_stale_match_reap_once(db_session, fake_redis)
    assert reaped == 0

    await db_session.refresh(match)
    assert match.status == "active"


# ─────────────────────────────────────────────────────────────────────────────
# Total time budget -> automatic completion (lockstep play has no explicit
# "finish" action; the match ends purely once elapsed time exhausts the
# whole question set's combined window)
# ─────────────────────────────────────────────────────────────────────────────

async def test_match_completes_automatically_once_total_time_budget_elapses(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    await _seed_rating(db_session, user_b.id, "Mathematics", 1000)
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    await _seed_and_activate(db_session, fake_redis, match, user_a_id=user_a.id, user_b_id=user_b.id)

    # Neither player answers anything -- the total time budget expiring is
    # the only thing that should force completion.
    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=TOTAL + 1))
    state = await battle_gameplay_service.get_match_state(
        db_session, match_id=match.id, user_id=user_a.id, redis=fake_redis
    )
    assert state["status"] == "completed"
    assert state["result"]["winner_user_id"] is None  # neither answered anything -- draw


# ─────────────────────────────────────────────────────────────────────────────
# get_battle_history -- batched (not N+1) match/participant/username lookups
# ─────────────────────────────────────────────────────────────────────────────

async def _seed_rating_if_absent(db, user_id: int, subject: str, rating: int) -> None:
    """Same subject/user pair gets seeded across multiple matches in the
    history tests below -- plain _seed_rating would violate the
    (user_id, subject) unique constraint on the second call.
    """
    existing = (
        await db.execute(
            select(CompetitiveSubjectRating).where(
                CompetitiveSubjectRating.user_id == user_id, CompetitiveSubjectRating.subject == subject,
            )
        )
    ).scalar_one_or_none()
    if existing is None:
        await _seed_rating(db, user_id, subject, rating)


async def _complete_match_with_winner(db, redis, *, subject, winner_id, loser_id) -> BattleMatch:
    """Winner answers every question correctly, loser answers none, then the
    total time budget is exhausted so completion runs -- mirrors
    test_completion_is_idempotent_and_rating_matches_elo_engine's approach.
    """
    await _seed_rating_if_absent(db, winner_id, subject, 1000)
    await _seed_rating_if_absent(db, loser_id, subject, 1000)
    match = await _create_waiting_match(
        db, subject=subject, difficulty="Mostly Medium",
        user_a_id=winner_id, rating_a=1000, user_b_id=loser_id, rating_b=1000,
    )
    _, questions = await _seed_and_activate(db, redis, match, user_a_id=winner_id, user_b_id=loser_id)

    for i in range(N):
        await _set_started_at(db, match.id, datetime.now(UTC) - timedelta(seconds=i * Q + 1))
        await _submit_and_grade(
            db, redis, match, order=i, user_id=winner_id,
            question_id=questions[i].id, selected_option="A",
        )

    await redis.hset(f"battle:user_state:{winner_id}", "state", "in_match")
    await redis.hset(f"battle:user_state:{loser_id}", "state", "in_match")
    await _set_started_at(db, match.id, datetime.now(UTC) - timedelta(seconds=TOTAL + 1))
    await battle_gameplay_service.get_match_state(db, match_id=match.id, user_id=winner_id, redis=redis)
    await db.refresh(match)
    return match


async def test_get_battle_history_returns_matches_newest_finished_first(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    user_c = await _make_user(db_session, "user-c")

    older = await _complete_match_with_winner(
        db_session, fake_redis, subject="Mathematics", winner_id=user_a.id, loser_id=user_b.id,
    )
    newer = await _complete_match_with_winner(
        db_session, fake_redis, subject="Mathematics", winner_id=user_c.id, loser_id=user_a.id,
    )

    history = await battle_gameplay_service.get_battle_history(
        db_session, user_id=user_a.id, subject=None, page=1, page_size=20,
    )

    assert history["total_count"] == 2
    assert history["total_pages"] == 1
    assert [e["match_id"] for e in history["entries"]] == [newer.id, older.id]

    newest_entry, older_entry = history["entries"]
    assert newest_entry["opponent_user_id"] == user_c.id
    assert newest_entry["opponent_username"] == user_c.username
    assert newest_entry["winner_user_id"] == user_c.id
    assert newest_entry["me"]["result"] == "loss"
    assert newest_entry["opponent"]["result"] == "win"

    assert older_entry["opponent_user_id"] == user_b.id
    assert older_entry["opponent_username"] == user_b.username
    assert older_entry["winner_user_id"] == user_a.id
    assert older_entry["me"]["result"] == "win"


async def test_get_battle_history_pagination(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    user_c = await _make_user(db_session, "user-c")

    older = await _complete_match_with_winner(
        db_session, fake_redis, subject="Mathematics", winner_id=user_a.id, loser_id=user_b.id,
    )
    await _complete_match_with_winner(
        db_session, fake_redis, subject="Mathematics", winner_id=user_c.id, loser_id=user_a.id,
    )

    page1 = await battle_gameplay_service.get_battle_history(
        db_session, user_id=user_a.id, subject=None, page=1, page_size=1,
    )
    page2 = await battle_gameplay_service.get_battle_history(
        db_session, user_id=user_a.id, subject=None, page=2, page_size=1,
    )

    assert page1["total_count"] == 2
    assert page1["total_pages"] == 2
    assert len(page1["entries"]) == 1
    assert len(page2["entries"]) == 1
    assert page2["entries"][0]["match_id"] == older.id


async def test_get_battle_history_empty_for_user_with_no_completed_matches(db_session, fake_redis):
    user = await _make_user(db_session, "user-lonely")
    history = await battle_gameplay_service.get_battle_history(
        db_session, user_id=user.id, subject=None, page=1, page_size=20,
    )
    assert history == {"page": 1, "page_size": 20, "total_count": 0, "total_pages": 0, "entries": []}

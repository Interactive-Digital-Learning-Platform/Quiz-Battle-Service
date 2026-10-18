"""
tests/test_battle_gameplay_service.py
──────────────────────────────────────────
Integration tests (in-memory DB, no HTTP, no real sleeps) for
app/services/battle_gameplay_service.py -- ready-gated activation
("waiting"->"countdown"->"active"), self-paced Next/Skip progression,
answer submission/validation/scoring, match completion, and rating
application.
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
import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select, update

from app.core.config import settings
from app.models.battle_match import (
    BattleAnswer,
    BattleMatch,
    BattleMatchQuestion,
    BattleParticipant,
)
from app.models.competitive_rating import CompetitiveSubjectRating
from app.models.question import Question
from app.services import battle_gameplay_service, battle_ws_events, matchmaking_service
from app.services.battle_question_service import BattleQuestionSelectionError
from app.services.elo_engine import compute_elo_update
from app.services.user_service import get_or_create_user

N = settings.BATTLE_QUESTION_COUNT
DURATION = settings.BATTLE_MATCH_DURATION_SECONDS
# Past the match clock AND the late-answer grace -- answers are closed.
CLOSED = DURATION + settings.BATTLE_ANSWER_LATE_GRACE_SECONDS + settings.BATTLE_NETWORK_ALLOWANCE_MAX_SECONDS + 1


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


async def _answer(db, redis, match, *, user_id, question_id, selected_option):
    """Next (selected_option set) or Skip (None) for this player's current
    question -- returns the AnswerOutcome.
    """
    return await battle_gameplay_service.submit_battle_answer(
        db, match_id=match.id, user_id=user_id, question_id=question_id,
        selected_option=selected_option, redis=redis,
    )


async def _answer_all(db, redis, match, questions, *, user_id, selected_option="A"):
    outcome = None
    for q in questions:
        outcome = await _answer(db, redis, match, user_id=user_id, question_id=q.id, selected_option=selected_option)
    return outcome


async def _answer_row(db, match_id: int, user_id: int, question_id: int) -> BattleAnswer:
    return (
        await db.execute(
            select(BattleAnswer).where(
                BattleAnswer.match_id == match_id, BattleAnswer.user_id == user_id,
                BattleAnswer.question_id == question_id,
            )
        )
    ).scalar_one()


async def _answer_count(db, match_id: int, user_id: int) -> int:
    return await db.scalar(
        select(func.count()).select_from(BattleAnswer).where(
            BattleAnswer.match_id == match_id, BattleAnswer.user_id == user_id,
        )
    )


async def _participant(db, match_id: int, user_id: int) -> BattleParticipant:
    p = (
        await db.execute(
            select(BattleParticipant).where(BattleParticipant.match_id == match_id, BattleParticipant.user_id == user_id)
        )
    ).scalar_one()
    await db.refresh(p)
    return p


async def _new_active_match(db, redis, subject="Mathematics"):
    user_a = await _make_user(db, "user-a")
    user_b = await _make_user(db, "user-b")
    match = await _create_waiting_match(
        db, subject=subject, difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    _, questions = await _seed_and_activate(db, redis, match, user_a_id=user_a.id, user_b_id=user_b.id)
    return user_a, user_b, match, questions


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
    # Everyone starts on question 0.
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
# Active state: one match clock, each player's own current question
# ─────────────────────────────────────────────────────────────────────────────

async def test_active_state_has_the_match_clock_and_the_players_own_question(db_session, fake_redis):
    user_a, _, match, questions = await _new_active_match(db_session, fake_redis)

    state = await battle_gameplay_service.get_match_state(db_session, match_id=match.id, user_id=user_a.id)
    # _seed_and_activate backdates started_at by 1s.
    assert state["time_remaining_seconds"] == pytest.approx(DURATION - 1, abs=0.5)
    assert state["ends_at"] == battle_gameplay_service.match_ends_at(match)
    assert state["server_time"] is not None
    assert state["question_index"] == 0
    assert state["question"]["question_id"] == questions[0].id
    assert state["finished"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Next / Skip: save first, then move on -- each player at their own pace
# ─────────────────────────────────────────────────────────────────────────────

async def test_next_saves_the_answer_then_moves_only_that_player_on(db_session, fake_redis):
    user_a, user_b, match, questions = await _new_active_match(db_session, fake_redis)

    outcome = await _answer(
        db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="A",
    )
    assert outcome.next["question_index"] == 1
    assert outcome.next["question"]["question_id"] == questions[1].id
    assert outcome.next["finished"] is False
    assert outcome.progress == {"user_id": user_a.id, "question_order": 0, "answered": True, "answered_count": 1}
    assert (await _answer_row(db_session, match.id, user_a.id, questions[0].id)).is_correct is True

    state_a = await battle_gameplay_service.get_match_state(db_session, match_id=match.id, user_id=user_a.id)
    state_b = await battle_gameplay_service.get_match_state(db_session, match_id=match.id, user_id=user_b.id)
    assert state_a["question_index"] == 1
    assert state_b["question_index"] == 0  # the opponent is still on their own question 0
    assert state_b["opponent_progress"]["answered_count"] == 1


async def test_correctness_is_never_revealed_while_the_match_is_live(db_session, fake_redis):
    user_a, user_b, match, questions = await _new_active_match(db_session, fake_redis)

    outcome = await _answer(
        db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="B",
    )
    assert "is_correct" not in outcome.next and "correct_answer" not in str(outcome.next)

    state_a = await battle_gameplay_service.get_match_state(db_session, match_id=match.id, user_id=user_a.id)
    state_b = await battle_gameplay_service.get_match_state(db_session, match_id=match.id, user_id=user_b.id)
    assert state_a["my_answers"] == [{"question_order": 0, "is_correct": None, "answered": True}]
    assert state_b["opponent_answers"] == [{"question_order": 0, "is_correct": None, "answered": True}]


async def test_grading_mirrors_submit_quiz_rule(db_session, fake_redis):
    user_a, _, match, questions = await _new_active_match(db_session, fake_redis)

    # Case-insensitive, whitespace-stripped exact match, correct_answer="A".
    await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="  a  ")
    row = await _answer_row(db_session, match.id, user_a.id, questions[0].id)
    assert row.is_correct is True
    assert row.base_score == settings.BATTLE_SCORE_BASE_CORRECT


async def test_wrong_answer_scores_zero(db_session, fake_redis):
    user_a, _, match, questions = await _new_active_match(db_session, fake_redis)

    await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="B")
    row = await _answer_row(db_session, match.id, user_a.id, questions[0].id)
    assert (row.is_correct, row.total_question_score) == (False, 0)


async def test_skip_saves_a_zero_point_skip_and_moves_on(db_session, fake_redis):
    user_a, user_b, match, questions = await _new_active_match(db_session, fake_redis)

    outcome = await _answer(
        db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option=None,
    )
    assert outcome.next["question_index"] == 1
    assert outcome.progress["answered"] is False
    row = await _answer_row(db_session, match.id, user_a.id, questions[0].id)
    assert (row.selected_option, row.is_correct, row.total_question_score) == ("", False, 0)

    state_b = await battle_gameplay_service.get_match_state(db_session, match_id=match.id, user_id=user_b.id)
    assert state_b["opponent_answers"] == [{"question_order": 0, "is_correct": None, "answered": False}]


async def test_spamming_next_saves_exactly_one_answer(db_session, fake_redis):
    user_a, _, match, questions = await _new_active_match(db_session, fake_redis)

    first = await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="A")
    # The same Next again (double tap / resent after a lost reply), even with a different pick.
    again = await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="B")

    assert again.progress is None  # nothing new saved
    assert again.next["question_index"] == first.next["question_index"] == 1
    assert again.next["question"] == first.next["question"]
    rows = (
        await db_session.execute(select(BattleAnswer).where(BattleAnswer.match_id == match.id))
    ).scalars().all()
    assert [(r.question_id, r.selected_option) for r in rows] == [(questions[0].id, "A")]
    assert (await _participant(db_session, match.id, user_a.id)).current_question_index == 1


async def test_a_resent_answer_for_an_earlier_question_returns_the_current_one(db_session, fake_redis):
    user_a, _, match, questions = await _new_active_match(db_session, fake_redis)
    await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="A")
    await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[1].id, selected_option="A")

    late = await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="A")
    assert late.progress is None
    assert late.next["question_index"] == 2


async def test_cannot_jump_ahead_to_a_later_question(db_session, fake_redis):
    user_a, _, match, questions = await _new_active_match(db_session, fake_redis)

    with pytest.raises(HTTPException) as exc_info:
        await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[1].id, selected_option="A")
    assert exc_info.value.status_code == 400


async def test_submit_answer_rejects_question_from_a_foreign_match(db_session, fake_redis):
    user_a, _, match, _ = await _new_active_match(db_session, fake_redis)
    foreign_question = Question(
        question="Foreign?", options=["A", "B"], correct_answer="A",
        subject="Mathematics", lesson="Algebra", difficulty="medium",
    )
    db_session.add(foreign_question)
    await db_session.commit()
    await db_session.refresh(foreign_question)

    with pytest.raises(HTTPException) as exc_info:
        await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=foreign_question.id, selected_option="A")
    assert exc_info.value.status_code == 400


async def test_submit_answer_rejects_non_participant(db_session, fake_redis):
    _, _, match, questions = await _new_active_match(db_session, fake_redis)
    outsider = await _make_user(db_session, "outsider")

    with pytest.raises(HTTPException) as exc_info:
        await _answer(db_session, fake_redis, match, user_id=outsider.id, question_id=questions[0].id, selected_option="A")
    assert exc_info.value.status_code == 403


# ─────────────────────────────────────────────────────────────────────────────
# Hidden per-question timer + speed bonus
# ─────────────────────────────────────────────────────────────────────────────

async def _set_presented_at(db, match_id: int, user_id: int, when: datetime) -> None:
    await db.execute(
        update(BattleParticipant)
        .where(BattleParticipant.match_id == match_id, BattleParticipant.user_id == user_id)
        .values(current_question_presented_at=when)
    )
    await db.commit()


@pytest.mark.parametrize(
    ("seconds_taken", "speed_bonus"), [(3, 30), (4.9, 30), (8, 20), (12, 10), (20, 0)],
)
async def test_speed_bonus_follows_the_time_since_the_question_was_handed_out(
    db_session, fake_redis, seconds_taken, speed_bonus,
):
    user_a, _, match, questions = await _new_active_match(db_session, fake_redis)
    # Question 0 starts at match.started_at.
    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=seconds_taken))

    await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="A")
    row = await _answer_row(db_session, match.id, user_a.id, questions[0].id)
    assert row.response_time_ms == pytest.approx(seconds_taken * 1000, abs=300)
    assert row.speed_bonus == speed_bonus
    assert row.total_question_score == settings.BATTLE_SCORE_BASE_CORRECT + speed_bonus


async def test_each_question_is_timed_from_when_that_player_got_it(db_session, fake_redis):
    user_a, _, match, questions = await _new_active_match(db_session, fake_redis)
    # The match started long ago, but question 1 was only handed out 2s ago.
    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=60))
    await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="A")
    await _set_presented_at(db_session, match.id, user_a.id, datetime.now(UTC) - timedelta(seconds=2))

    await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[1].id, selected_option="A")
    assert (await _answer_row(db_session, match.id, user_a.id, questions[0].id)).speed_bonus == 0
    assert (await _answer_row(db_session, match.id, user_a.id, questions[1].id)).speed_bonus == 30


async def test_network_allowance_takes_the_fastest_round_trip_off(db_session, fake_redis):
    user_a, _, match, questions = await _new_active_match(db_session, fake_redis)
    for rtt in (0.9, 0.4, 0.6):
        await battle_gameplay_service.record_rtt_sample(fake_redis, match_id=match.id, user_id=user_a.id, rtt_seconds=rtt)
    # 5.3s on the server clock - 0.4s fastest round trip = 4.9s -> still the 5s tier.
    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=5.3))

    await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="A")
    row = await _answer_row(db_session, match.id, user_a.id, questions[0].id)
    assert row.response_time_ms == pytest.approx(4900, abs=150)
    assert row.speed_bonus == 30


async def test_network_allowance_is_capped(db_session, fake_redis):
    user_a, _, match, questions = await _new_active_match(db_session, fake_redis)
    # A client delaying every pong to look slow gains at most the cap.
    await battle_gameplay_service.record_rtt_sample(fake_redis, match_id=match.id, user_id=user_a.id, rtt_seconds=4.0)
    assert await battle_gameplay_service.network_allowance_seconds(
        fake_redis, match_id=match.id, user_id=user_a.id,
    ) == settings.BATTLE_NETWORK_ALLOWANCE_MAX_SECONDS
    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=8))

    await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="A")
    row = await _answer_row(db_session, match.id, user_a.id, questions[0].id)
    expected_ms = (8 - settings.BATTLE_NETWORK_ALLOWANCE_MAX_SECONDS) * 1000
    assert row.response_time_ms == pytest.approx(expected_ms, abs=150)
    assert row.speed_bonus == 20


async def test_only_the_last_few_round_trips_count(db_session, fake_redis):
    user_a, _, match, _ = await _new_active_match(db_session, fake_redis)
    await battle_gameplay_service.record_rtt_sample(fake_redis, match_id=match.id, user_id=user_a.id, rtt_seconds=0.05)
    for _ in range(settings.BATTLE_RTT_SAMPLE_COUNT):
        await battle_gameplay_service.record_rtt_sample(fake_redis, match_id=match.id, user_id=user_a.id, rtt_seconds=0.3)
    assert await battle_gameplay_service.network_allowance_seconds(
        fake_redis, match_id=match.id, user_id=user_a.id,
    ) == pytest.approx(0.3)


# ─────────────────────────────────────────────────────────────────────────────
# The match clock, finishing early, and the end-of-match review
# ─────────────────────────────────────────────────────────────────────────────

async def test_an_answer_arriving_after_the_clock_hits_zero_is_refused(db_session, fake_redis):
    user_a, _, match, questions = await _new_active_match(db_session, fake_redis)
    # The visible clock ran out 0.5s ago -- no grace, the answer is too late.
    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=DURATION + 0.5))

    with pytest.raises(HTTPException) as exc_info:
        await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="A")

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == battle_gameplay_service.TIME_UP_DETAIL
    await db_session.refresh(match)  # the refusal rolled the session back
    await db_session.refresh(user_a)
    assert await _answer_count(db_session, match.id, user_a.id) == 0


async def test_a_slow_connection_gets_its_own_0_00_cutoff(db_session, fake_redis):
    # A 1.5s round trip: the phone's countdown runs ~1.5s behind the server's,
    # so an answer arriving 1s after the server's 0:00 was sent before the
    # player's own 0:00 -- it counts. A fast connection gets no such leeway.
    user_a, user_b, match, questions = await _new_active_match(db_session, fake_redis)
    await battle_gameplay_service.record_rtt_sample(fake_redis, match_id=match.id, user_id=user_a.id, rtt_seconds=1.5)
    await battle_gameplay_service.record_rtt_sample(fake_redis, match_id=match.id, user_id=user_b.id, rtt_seconds=0.05)
    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=DURATION + 1.0))

    outcome = await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="A")
    assert outcome.progress is not None
    with pytest.raises(HTTPException) as exc_info:
        await _answer(db_session, fake_redis, match, user_id=user_b.id, question_id=questions[0].id, selected_option="A")
    assert exc_info.value.detail == battle_gameplay_service.TIME_UP_DETAIL


async def test_an_answer_whose_save_outlasts_the_clock_is_refused(db_session, fake_redis, monkeypatch):
    user_a, _, match, questions = await _new_active_match(db_session, fake_redis)
    # 0.2s left on the clock when the answer arrives, but the save takes 0.6s.
    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=DURATION - 0.2))

    async def slow_allowance(*args, **kwargs):
        await asyncio.sleep(0.6)
        return 0.0

    monkeypatch.setattr(battle_gameplay_service, "network_allowance_seconds", slow_allowance)

    with pytest.raises(HTTPException) as exc_info:
        await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="A")

    assert exc_info.value.detail == battle_gameplay_service.TIME_UP_DETAIL
    # The refusal rolled the session back, expiring the loaded rows.
    await db_session.refresh(match)
    await db_session.refresh(user_a)
    assert await _answer_count(db_session, match.id, user_a.id) == 0
    participant = await _participant(db_session, match.id, user_a.id)
    assert participant.current_question_index == 0  # not moved on either


async def test_answers_after_the_grace_are_rejected_and_the_match_ends(db_session, fake_redis):
    user_a, _, match, questions = await _new_active_match(db_session, fake_redis)
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=CLOSED))

    with pytest.raises(HTTPException) as exc_info:
        await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="A")
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == battle_gameplay_service.TIME_UP_DETAIL
    await db_session.refresh(match)
    assert match.status == "completed"


async def test_a_finished_player_cannot_forfeit(db_session, fake_redis):
    user_a, _, match, questions = await _new_active_match(db_session, fake_redis)
    await _answer_all(db_session, fake_redis, match, questions, user_id=user_a.id)

    with pytest.raises(HTTPException) as exc_info:
        await battle_gameplay_service.forfeit_match(db_session, fake_redis, match_id=match.id, forfeiting_user_id=user_a.id)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == battle_gameplay_service.FINISHED_CANT_FORFEIT_DETAIL
    await db_session.refresh(match)
    assert match.status == "active"


async def test_queue_status_says_when_a_finished_player_awaits_the_opponent(db_session, fake_redis):
    user_a, user_b, match, questions = await _new_active_match(db_session, fake_redis)
    for user in (user_a, user_b):
        await fake_redis.hset(f"battle:user_state:{user.id}", mapping={"state": "in_match", "match_id": match.id})

    async def awaiting(user) -> bool:
        status = await matchmaking_service.get_queue_status(db_session, fake_redis, user_id=user.id)
        assert status["status"] == "matched"
        return status["awaiting_opponent"]

    assert await awaiting(user_a) is False  # still playing
    await _answer_all(db_session, fake_redis, match, questions, user_id=user_a.id)
    assert await awaiting(user_a) is True
    assert await awaiting(user_b) is False  # B is the one still playing


async def test_streak_bonus_builds_on_consecutive_correct_answers_and_resets(db_session, fake_redis):
    user_a, _, match, questions = await _new_active_match(db_session, fake_redis)
    # correct, correct, correct, skip, correct, wrong, correct, correct
    for q, option in zip(questions, ["A", "A", "A", None, "A", "B", "A", "A"], strict=False):
        await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=q.id, selected_option=option)

    bonuses = [
        (await _answer_row(db_session, match.id, user_a.id, q.id)).streak_bonus for q in questions[:8]
    ]
    assert bonuses == [0, 5, 10, 0, 0, 0, 0, 5]
    row = await _answer_row(db_session, match.id, user_a.id, questions[2].id)
    assert row.total_question_score == row.base_score + row.speed_bonus + row.streak_bonus


async def test_review_shows_each_answers_streak_bonus(db_session, fake_redis):
    user_a, user_b, match, questions = await _new_active_match(db_session, fake_redis)
    await _answer_all(db_session, fake_redis, match, questions, user_id=user_a.id)
    await _answer_all(db_session, fake_redis, match, questions, user_id=user_b.id)

    state = await battle_gameplay_service.get_match_state(db_session, match_id=match.id, user_id=user_a.id, redis=fake_redis)

    assert [r["streak_bonus"] for r in state["review"]] == [0, 5, 10, 15, 20, 20, 20, 20, 20, 20]
    assert state["result"]["me"]["final_score"] == sum(r["total_question_score"] for r in state["review"])


async def test_a_second_activation_attempt_doesnt_announce_the_countdown_again(db_session, fake_redis):
    # Picking questions can take seconds now (AI first); a second trigger for
    # the same match (the other player's connect, the reaper's forced start)
    # must find it already started and not publish countdown_started again.
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    questions = _fixed_questions()
    db_session.add_all(questions)
    await db_session.flush()

    with patch(
        "app.services.battle_gameplay_service.select_battle_questions", new=AsyncMock(return_value=questions),
    ), patch.object(battle_gameplay_service.ws_manager, "publish", new=AsyncMock()) as publish:
        first = await battle_gameplay_service._finalize_activation(db_session, fake_redis, match=match)
        second = await battle_gameplay_service._finalize_activation(db_session, fake_redis, match=match)

    assert first.status == second.status == "countdown"
    assert [c.kwargs["event"]["type"] for c in publish.await_args_list] == ["countdown_started"]


async def _end_reason(db, redis, match, user_id) -> str | None:
    state = await battle_gameplay_service.get_match_state(db, match_id=match.id, user_id=user_id, redis=redis)
    assert state["status"] == "completed"
    return state["result"]["end_reason"]


async def test_end_reason_all_finished_when_both_players_got_through_every_question(db_session, fake_redis):
    user_a, user_b, match, questions = await _new_active_match(db_session, fake_redis)
    await _answer_all(db_session, fake_redis, match, questions, user_id=user_a.id)
    await _answer_all(db_session, fake_redis, match, questions, user_id=user_b.id)

    assert await _end_reason(db_session, fake_redis, match, user_a.id) == "all_finished"
    event = await battle_ws_events.match_finished_event(db_session, match)
    assert event["end_reason"] == "all_finished"


async def test_end_reason_time_up_when_the_clock_ran_out_first(db_session, fake_redis):
    # A finished, B didn't -- only the clock could have ended it.
    user_a, user_b, match, questions = await _new_active_match(db_session, fake_redis)
    await _answer_all(db_session, fake_redis, match, questions, user_id=user_a.id)
    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=CLOSED))

    assert await _end_reason(db_session, fake_redis, match, user_b.id) == "time_up"


async def test_end_reason_forfeit(db_session, fake_redis):
    user_a, user_b, match, _ = await _new_active_match(db_session, fake_redis)
    await battle_gameplay_service.forfeit_match(db_session, fake_redis, match_id=match.id, forfeiting_user_id=user_b.id)

    assert await _end_reason(db_session, fake_redis, match, user_a.id) == "forfeit"


async def test_a_player_who_finishes_first_waits_for_the_opponent(db_session, fake_redis):
    user_a, _, match, questions = await _new_active_match(db_session, fake_redis)

    last = await _answer_all(db_session, fake_redis, match, questions, user_id=user_a.id)
    assert last.next == {**last.next, "question_index": N, "question": None, "finished": True}
    assert last.match.status == "active"  # the opponent is still playing
    assert (await _participant(db_session, match.id, user_a.id)).finished_at is not None

    # Another Next after finishing is a harmless repeat.
    again = await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[-1].id, selected_option="A")
    assert again.progress is None and again.next["finished"] is True


async def test_the_match_ends_as_soon_as_both_players_finish(db_session, fake_redis):
    user_a, user_b, match, questions = await _new_active_match(db_session, fake_redis)
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    await _seed_rating(db_session, user_b.id, "Mathematics", 1000)
    await fake_redis.hset(f"battle:user_state:{user_a.id}", "state", "in_match")

    await _answer_all(db_session, fake_redis, match, questions, user_id=user_a.id)
    last = await _answer_all(db_session, fake_redis, match, questions, user_id=user_b.id, selected_option=None)

    assert last.match.status == "completed"
    assert last.match.winner_user_id == user_a.id
    assert await fake_redis.exists(f"battle:user_state:{user_a.id}") == 0


async def test_highest_score_wins(db_session, fake_redis):
    user_a, user_b, match, questions = await _new_active_match(db_session, fake_redis)
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    await _seed_rating(db_session, user_b.id, "Mathematics", 1000)
    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=30))

    # A: question 0 right but slow (100), then skips the rest.
    await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="A")
    for q in questions[1:]:
        await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=q.id, selected_option=None)
    # B: question 0 skipped, question 1 right and fast (130), rest skipped.
    await _answer(db_session, fake_redis, match, user_id=user_b.id, question_id=questions[0].id, selected_option=None)
    await _answer(db_session, fake_redis, match, user_id=user_b.id, question_id=questions[1].id, selected_option="A")
    last = None
    for q in questions[2:]:
        last = await _answer(db_session, fake_redis, match, user_id=user_b.id, question_id=q.id, selected_option=None)

    assert last.match.status == "completed"
    assert last.match.winner_user_id == user_b.id


async def test_review_shows_only_the_viewers_own_answers(db_session, fake_redis):
    user_a, user_b, match, questions = await _new_active_match(db_session, fake_redis)
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    await _seed_rating(db_session, user_b.id, "Mathematics", 1000)
    await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="A")
    await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[1].id, selected_option="B")
    await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[2].id, selected_option=None)
    await _answer_all(db_session, fake_redis, match, questions[:4], user_id=user_b.id, selected_option="C")

    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=CLOSED))
    state = await battle_gameplay_service.get_match_state(db_session, match_id=match.id, user_id=user_a.id)
    assert state["status"] == "completed"

    review = state["review"]
    assert [r["status"] for r in review] == ["correct", "incorrect", "skipped"] + ["unanswered"] * (N - 3)
    assert [r["selected_option"] for r in review[:4]] == ["A", "B", None, None]
    assert all(r["correct_answer"] == "A" for r in review)
    assert review[0]["total_question_score"] == review[0]["base_score"] + review[0]["speed_bonus"] > 0
    assert sorted(review[0]["options"]) == ["A", "B", "C", "D"]
    assert "C" not in [r["selected_option"] for r in review]  # never the opponent's picks


async def test_match_completes_automatically_once_the_clock_runs_out(db_session, fake_redis):
    user_a, _, match, _ = await _new_active_match(db_session, fake_redis)
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)

    # Neither player answers anything -- the clock running out is the only
    # thing that should force completion.
    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=CLOSED))
    state = await battle_gameplay_service.get_match_state(
        db_session, match_id=match.id, user_id=user_a.id, redis=fake_redis
    )
    assert state["status"] == "completed"
    assert state["result"]["winner_user_id"] is None  # neither answered anything -- draw


# ─────────────────────────────────────────────────────────────────────────────
# Full match completion: winner, rating update, running average
# ─────────────────────────────────────────────────────────────────────────────

async def test_completion_is_idempotent_and_rating_matches_elo_engine(db_session, fake_redis):
    user_a, user_b, match, questions = await _new_active_match(db_session, fake_redis)
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    await _seed_rating(db_session, user_b.id, "Mathematics", 1000)

    # user_a answers all N correctly; user_b answers none.
    await _answer_all(db_session, fake_redis, match, questions, user_id=user_a.id)

    await fake_redis.hset(f"battle:user_state:{user_a.id}", "state", "in_match")
    await fake_redis.hset(f"battle:user_state:{user_b.id}", "state", "in_match")

    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=CLOSED))
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

    await _answer_all(db_session, fake_redis, match, questions, user_id=user_a.id)

    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=CLOSED))
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
    user_a, user_b, match, questions = await _new_active_match(db_session, fake_redis)
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    await _seed_rating(db_session, user_b.id, "Mathematics", 1000)

    # user_a answers question 0, user_b never answers.
    await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="A")

    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=CLOSED))
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
    answer_row = await _answer_row(db_session, match.id, user_a.id, questions[0].id)
    assert rating_a_row.average_response_time == pytest.approx(answer_row.response_time_ms / 1000.0, abs=0.01)


async def test_skips_do_not_count_as_wrong_answers(db_session, fake_redis):
    user_a, user_b, match, questions = await _new_active_match(db_session, fake_redis)
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    await _seed_rating(db_session, user_b.id, "Mathematics", 1000)
    await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="B")
    await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[1].id, selected_option=None)

    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=CLOSED))
    state = await battle_gameplay_service.get_match_state(db_session, match_id=match.id, user_id=user_a.id)
    assert (state["result"]["me"]["correct_count"], state["result"]["me"]["wrong_count"]) == (0, 1)


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
    """Winner answers every question correctly, loser skips every one --
    both finished, so the last skip completes the match.
    """
    await _seed_rating_if_absent(db, winner_id, subject, 1000)
    await _seed_rating_if_absent(db, loser_id, subject, 1000)
    match = await _create_waiting_match(
        db, subject=subject, difficulty="Mostly Medium",
        user_a_id=winner_id, rating_a=1000, user_b_id=loser_id, rating_b=1000,
    )
    _, questions = await _seed_and_activate(db, redis, match, user_a_id=winner_id, user_b_id=loser_id)

    await _answer_all(db, redis, match, questions, user_id=winner_id)
    await _answer_all(db, redis, match, questions, user_id=loser_id, selected_option=None)
    await db.refresh(match)
    assert match.status == "completed"
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

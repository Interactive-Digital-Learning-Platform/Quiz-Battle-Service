"""
tests/test_battle_ready_forfeit_service.py
─────────────────────────────────────────────
Integration tests (in-memory DB + fake_redis, no HTTP, no real sleeps) for
the Phase 4 ready/countdown/forfeit/disconnect-presence additions to
app/services/battle_gameplay_service.py. Complements
tests/test_battle_gameplay_service.py (which covers activation, question
progression, answer submission/scoring, and normal completion) -- this file
is scoped to the new ready-gate, forced-outcome forfeit, and Redis-only
disconnect-presence helpers.
"""
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy import select, update

from app.models.battle_match import BattleMatch, BattleParticipant
from app.models.competitive_rating import CompetitiveSubjectRating
from app.models.question import Question
from app.services import battle_gameplay_service, matchmaking_service
from app.services.battle_gameplay_service import as_utc
from app.services.battle_question_service import BattleQuestionSelectionError
from app.services.elo_engine import compute_elo_update
from app.services.user_service import get_or_create_user

N = 10  # settings.BATTLE_QUESTION_COUNT, mirrored as a literal for readability


async def _make_user(db, clerk_id: str):
    return await get_or_create_user(db, clerk_id)


async def _seed_rating(db, user_id: int, subject: str, rating: int) -> CompetitiveSubjectRating:
    row = CompetitiveSubjectRating(user_id=user_id, subject=subject, rating=rating, highest_rating=rating)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def _create_waiting_match(db, *, user_a_id, rating_a, user_b_id, rating_b, subject="Mathematics") -> BattleMatch:
    match = BattleMatch(subject=subject, status="waiting", difficulty="Mostly Medium")
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
            question=f"Ready/forfeit test question {i}?", options=["A", "B", "C", "D"],
            correct_answer="A", subject="Mathematics", lesson="Algebra", difficulty="medium",
        )
        for i in range(count)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# mark_ready
# ─────────────────────────────────────────────────────────────────────────────

async def test_single_sided_ready_leaves_match_waiting(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(db_session, user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000)

    match = await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=user_a.id)
    assert match.status == "waiting"


async def test_resending_ready_is_idempotent(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(db_session, user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000)

    await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=user_a.id)
    match = await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=user_a.id)
    assert match.status == "waiting"


async def test_both_sided_ready_triggers_exactly_one_countdown_transition(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(db_session, user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000)
    questions = _fixed_questions()
    db_session.add_all(questions)
    await db_session.flush()

    await battle_gameplay_service.mark_connected(db_session, fake_redis, match_id=match.id, user_id=user_a.id)
    await battle_gameplay_service.mark_connected(db_session, fake_redis, match_id=match.id, user_id=user_b.id)
    await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=user_a.id)

    select_mock = AsyncMock(return_value=questions)
    with patch("app.services.battle_gameplay_service.select_battle_questions", new=select_mock):
        match = await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=user_b.id)

    assert match.status == "countdown"
    assert match.started_at is not None
    assert as_utc(match.started_at) > datetime.now(UTC)  # a FUTURE timestamp, not "now"
    select_mock.assert_awaited_once()  # question selection happens exactly once

    # A late/duplicate ready signal (either side) after the gate has already
    # opened is a pure no-op -- it must not re-select questions or move
    # started_at again.
    started_at = match.started_at
    with patch(
        "app.services.battle_gameplay_service.select_battle_questions",
        new=AsyncMock(side_effect=AssertionError("should not be called again")),
    ):
        match = await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=user_a.id)
    assert match.status == "countdown"
    assert match.started_at == started_at


async def test_ai_failure_during_ready_triggered_selection_cancels_cleanly(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    rating_a = await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    rating_b = await _seed_rating(db_session, user_b.id, "Mathematics", 1000)
    match = await _create_waiting_match(db_session, user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000)

    # Mirrors what matchmaking's claim step would have set at match creation
    # -- "in_match", no TTL, the exact state release_match_state must clear.
    await fake_redis.hset(matchmaking_service._user_state_key(user_a.id), "state", "in_match")
    await fake_redis.hset(matchmaking_service._user_state_key(user_b.id), "state", "in_match")

    await battle_gameplay_service.mark_connected(db_session, fake_redis, match_id=match.id, user_id=user_a.id)
    await battle_gameplay_service.mark_connected(db_session, fake_redis, match_id=match.id, user_id=user_b.id)
    await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=user_a.id)
    with patch(
        "app.services.battle_gameplay_service.select_battle_questions",
        new=AsyncMock(side_effect=BattleQuestionSelectionError("not enough questions")),
    ):
        match = await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=user_b.id)

    assert match.status == "cancelled"
    assert match.cancel_reason == battle_gameplay_service.CANCEL_REASON_AI_FAILURE
    assert match.started_at is None

    await db_session.refresh(rating_a)
    await db_session.refresh(rating_b)
    assert rating_a.matches_played == 0
    assert rating_b.matches_played == 0

    # The bug this session's own release_match_state fix missed: without it,
    # both players would be permanently stuck "in_match" and locked out of
    # ever joining a new queue.
    assert await fake_redis.exists(matchmaking_service._user_state_key(user_a.id)) == 0
    assert await fake_redis.exists(matchmaking_service._user_state_key(user_b.id)) == 0


async def test_countdown_to_active_lazy_transition_via_backdated_started_at(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(db_session, user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000)
    questions = _fixed_questions()
    db_session.add_all(questions)
    await db_session.flush()

    await battle_gameplay_service.mark_connected(db_session, fake_redis, match_id=match.id, user_id=user_a.id)
    await battle_gameplay_service.mark_connected(db_session, fake_redis, match_id=match.id, user_id=user_b.id)
    await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=user_a.id)
    with patch(
        "app.services.battle_gameplay_service.select_battle_questions",
        new=AsyncMock(return_value=questions),
    ):
        match = await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=user_b.id)
    assert match.status == "countdown"

    # Still within the countdown window -- must stay "countdown".
    still_waiting_state = await battle_gameplay_service.get_match_state(db_session, match_id=match.id, user_id=user_a.id)
    assert still_waiting_state["status"] == "countdown"

    # Backdate into the past -- the lazy check flips it to "active".
    await db_session.execute(
        update(BattleMatch).where(BattleMatch.id == match.id).values(started_at=datetime.now(UTC) - timedelta(seconds=1))
    )
    await db_session.commit()

    active_state = await battle_gameplay_service.get_match_state(db_session, match_id=match.id, user_id=user_a.id)
    assert active_state["status"] == "active"
    assert active_state["question_index"] == 0
    assert active_state["question"] is not None


# ─────────────────────────────────────────────────────────────────────────────
# mark_connected -- activation gate (see _try_activate_if_ready_and_connected)
# ─────────────────────────────────────────────────────────────────────────────

async def test_both_ready_but_only_one_connected_stays_waiting(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(db_session, user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000)

    await battle_gameplay_service.mark_connected(db_session, fake_redis, match_id=match.id, user_id=user_a.id)
    await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=user_a.id)
    with patch(
        "app.services.battle_gameplay_service.select_battle_questions",
        new=AsyncMock(side_effect=AssertionError("should not select questions -- user_b never connected")),
    ):
        match = await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=user_b.id)

    assert match.status == "waiting"


async def test_connect_after_both_ready_is_what_triggers_activation(db_session, fake_redis):
    # The mirror image of the ready-then-connect ordering already covered
    # above -- whichever of the two signals (ready, connected) completes the
    # pair LAST is what fires activation. Here both ready up first (with
    # neither connected yet), then connecting finishes the job.
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(db_session, user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000)
    questions = _fixed_questions()
    db_session.add_all(questions)
    await db_session.flush()

    await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=user_a.id)
    await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=user_b.id)

    match = await battle_gameplay_service.mark_connected(db_session, fake_redis, match_id=match.id, user_id=user_a.id)
    assert match.status == "waiting"  # only one of two connected so far

    with patch(
        "app.services.battle_gameplay_service.select_battle_questions",
        new=AsyncMock(return_value=questions),
    ):
        match = await battle_gameplay_service.mark_connected(db_session, fake_redis, match_id=match.id, user_id=user_b.id)

    assert match.status == "countdown"
    assert match.started_at is not None


async def test_mark_connected_is_a_noop_once_past_waiting(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(db_session, user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000)
    await db_session.execute(
        update(BattleMatch).where(BattleMatch.id == match.id).values(status="active", started_at=datetime.now(UTC))
    )
    await db_session.commit()

    result = await battle_gameplay_service.mark_connected(db_session, fake_redis, match_id=match.id, user_id=user_a.id)
    assert result is None  # no-op -- this gate is only about pre-activation "waiting"


async def test_reaper_force_activates_when_both_ready_but_never_connected(db_session, fake_redis, monkeypatch):
    # Covers a REST-only client (mark_connected never fires for one that
    # never opens a socket at all) and the rare real case where a WS
    # connection is delayed past BATTLE_CONNECTION_READY_TIMEOUT_SECONDS --
    # both players are still actively trying to play, so this force-starts
    # the match rather than reaping it as abandoned (contrast with
    # test_forfeit_from_waiting_cancels_match_with_no_scoring above, which is
    # a genuine early exit).
    from app.core.config import settings as app_settings
    monkeypatch.setattr(app_settings, "BATTLE_CONNECTION_READY_TIMEOUT_SECONDS", 0.0)

    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(db_session, user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000)
    questions = _fixed_questions()
    db_session.add_all(questions)
    await db_session.flush()

    await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=user_a.id)
    await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=user_b.id)

    match = await battle_gameplay_service.get_match_row(db_session, match.id)
    assert match.status == "waiting"  # neither ever connected

    with patch(
        "app.services.battle_gameplay_service.select_battle_questions",
        new=AsyncMock(return_value=questions),
    ):
        forced = await battle_gameplay_service.run_stale_match_reap_once(db_session, fake_redis)

    assert forced == 1
    match = await battle_gameplay_service.get_match_row(db_session, match.id)
    assert match.status == "countdown"
    assert match.started_at is not None


async def test_reaper_leaves_a_match_alone_while_it_is_being_started(db_session, fake_redis, monkeypatch):
    # Picking questions (AI first) can take seconds: the match is still "waiting" meanwhile, but it isn't
    # stalled -- the reaper must not try to start it too, or report it as force-activated.
    from app.core.config import settings as app_settings
    monkeypatch.setattr(app_settings, "BATTLE_CONNECTION_READY_TIMEOUT_SECONDS", 0.0)

    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(db_session, user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000)
    await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=user_a.id)
    await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=user_b.id)
    await fake_redis.set(battle_gameplay_service._activating_key(match.id), "1")

    select_questions = AsyncMock()
    with patch("app.services.battle_gameplay_service.select_battle_questions", new=select_questions):
        forced = await battle_gameplay_service.run_stale_match_reap_once(db_session, fake_redis)

    assert forced == 0
    select_questions.assert_not_awaited()
    assert (await battle_gameplay_service.get_match_row(db_session, match.id)).status == "waiting"


async def test_activation_clears_its_activating_marker(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(db_session, user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000)
    questions = _fixed_questions()
    db_session.add_all(questions)
    await db_session.flush()

    with patch("app.services.battle_gameplay_service.select_battle_questions", new=AsyncMock(return_value=questions)):
        _, started = await battle_gameplay_service._finalize_activation(db_session, fake_redis, match=match)

    assert started is True
    assert await fake_redis.exists(battle_gameplay_service._activating_key(match.id)) == 0


# ─────────────────────────────────────────────────────────────────────────────
# forfeit_match
# ─────────────────────────────────────────────────────────────────────────────

async def test_forfeit_rejects_non_participant(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    outsider = await _make_user(db_session, "outsider")
    match = await _create_waiting_match(db_session, user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000)

    with pytest.raises(HTTPException) as exc_info:
        await battle_gameplay_service.forfeit_match(
            db_session, fake_redis, match_id=match.id, forfeiting_user_id=outsider.id
        )
    assert exc_info.value.status_code == 403


async def test_forfeit_from_waiting_cancels_match_with_no_scoring(db_session, fake_redis):
    # "waiting" == matched but not yet both-ready -- no question has ever
    # been shown to either player, so leaving here voids the match instead
    # of scoring a forfeit-win (contrast with the active-match test below).
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    await _seed_rating(db_session, user_b.id, "Mathematics", 1000)
    match = await _create_waiting_match(db_session, user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000)
    await fake_redis.hset(matchmaking_service._user_state_key(user_a.id), "state", "in_match")
    await fake_redis.hset(matchmaking_service._user_state_key(user_b.id), "state", "in_match")

    match = await battle_gameplay_service.forfeit_match(
        db_session, fake_redis, match_id=match.id, forfeiting_user_id=user_a.id
    )

    assert match.status == "cancelled"
    assert match.winner_user_id is None
    assert match.cancel_reason == battle_gameplay_service.CANCEL_REASON_OPPONENT_LEFT_BEFORE_START
    assert await fake_redis.exists(matchmaking_service._user_state_key(user_a.id)) == 0
    assert await fake_redis.exists(matchmaking_service._user_state_key(user_b.id)) == 0

    participants = (
        await db_session.execute(select(BattleParticipant).where(BattleParticipant.match_id == match.id))
    ).scalars().all()
    assert all(p.result is None for p in participants)

    rating_a = (
        await db_session.execute(
            select(CompetitiveSubjectRating).where(
                CompetitiveSubjectRating.user_id == user_a.id, CompetitiveSubjectRating.subject == "Mathematics"
            )
        )
    ).scalar_one()
    rating_b = (
        await db_session.execute(
            select(CompetitiveSubjectRating).where(
                CompetitiveSubjectRating.user_id == user_b.id, CompetitiveSubjectRating.subject == "Mathematics"
            )
        )
    ).scalar_one()
    assert (rating_a.rating, rating_a.wins, rating_a.losses) == (1000, 0, 0)
    assert (rating_b.rating, rating_b.wins, rating_b.losses) == (1000, 0, 0)


async def test_forfeit_from_waiting_is_idempotent_against_already_cancelled_match(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    await _seed_rating(db_session, user_b.id, "Mathematics", 1000)
    match = await _create_waiting_match(db_session, user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000)

    match1 = await battle_gameplay_service.forfeit_match(
        db_session, fake_redis, match_id=match.id, forfeiting_user_id=user_a.id
    )
    assert match1.status == "cancelled"

    # A second forfeit call (e.g. the opponent's own Cancel button reacting
    # to the same match_cancelled event) must be a pure no-op.
    match2 = await battle_gameplay_service.forfeit_match(
        db_session, fake_redis, match_id=match.id, forfeiting_user_id=user_b.id
    )
    assert match2.status == "cancelled"
    assert match2.cancel_reason == match1.cancel_reason

    rating_a_row = (
        await db_session.execute(
            select(CompetitiveSubjectRating).where(
                CompetitiveSubjectRating.user_id == user_a.id, CompetitiveSubjectRating.subject == "Mathematics"
            )
        )
    ).scalar_one()
    assert rating_a_row.rating == 1000


async def test_forfeit_from_active_match_produces_forced_outcome_and_is_idempotent(db_session, fake_redis):
    # Once the match has actually started (active/countdown), forfeiting
    # still scores a real win/loss -- only the pre-start "waiting" case
    # (tested above) is voided instead.
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    await _seed_rating(db_session, user_b.id, "Mathematics", 1000)
    match = await _create_waiting_match(db_session, user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000)
    await db_session.execute(
        update(BattleMatch).where(BattleMatch.id == match.id).values(
            status="active", started_at=datetime.now(UTC) - timedelta(seconds=5)
        )
    )
    await db_session.commit()
    await db_session.refresh(match)

    match1 = await battle_gameplay_service.forfeit_match(
        db_session, fake_redis, match_id=match.id, forfeiting_user_id=user_a.id
    )

    assert match1.status == "completed"
    assert match1.winner_user_id == user_b.id
    assert match1.finished_at is not None

    participants = (
        await db_session.execute(select(BattleParticipant).where(BattleParticipant.match_id == match.id))
    ).scalars().all()
    p_a = next(p for p in participants if p.user_id == user_a.id)
    p_b = next(p for p in participants if p.user_id == user_b.id)
    assert p_a.result == "forfeit"
    assert p_b.result == "win"

    expected = compute_elo_update(1000, 1000, 0.0)  # user_a forfeits == a loss for A
    rating_a = (
        await db_session.execute(
            select(CompetitiveSubjectRating).where(
                CompetitiveSubjectRating.user_id == user_a.id, CompetitiveSubjectRating.subject == "Mathematics"
            )
        )
    ).scalar_one()
    rating_b = (
        await db_session.execute(
            select(CompetitiveSubjectRating).where(
                CompetitiveSubjectRating.user_id == user_b.id, CompetitiveSubjectRating.subject == "Mathematics"
            )
        )
    ).scalar_one()
    assert rating_a.rating == expected.new_rating_a
    assert rating_b.rating == expected.new_rating_b

    # A second forfeit call (e.g. the opponent's disconnect-grace task firing
    # right after this one already resolved it) must be a pure no-op.
    match2 = await battle_gameplay_service.forfeit_match(
        db_session, fake_redis, match_id=match.id, forfeiting_user_id=user_b.id
    )
    assert match2.status == "completed"
    assert match2.winner_user_id == match1.winner_user_id  # unchanged, NOT re-decided in user_b's favor


# ─────────────────────────────────────────────────────────────────────────────
# Disconnect-presence Redis helpers
# ─────────────────────────────────────────────────────────────────────────────

async def test_disconnect_marker_round_trip(fake_redis):
    match_id, user_id = 42, 7

    assert await battle_gameplay_service.is_still_disconnected(fake_redis, match_id=match_id, user_id=user_id) is False

    generation = await battle_gameplay_service.bump_connection_generation(
        fake_redis, match_id=match_id, user_id=user_id
    )
    deadline = await battle_gameplay_service.mark_disconnected(
        fake_redis, match_id=match_id, user_id=user_id, generation=generation
    )
    assert deadline > datetime.now(UTC)
    assert await battle_gameplay_service.is_still_disconnected(fake_redis, match_id=match_id, user_id=user_id) is True

    cleared = await battle_gameplay_service.clear_disconnect_marker(fake_redis, match_id=match_id, user_id=user_id)
    assert cleared is True
    assert await battle_gameplay_service.is_still_disconnected(fake_redis, match_id=match_id, user_id=user_id) is False

    # Clearing an already-cleared (or never-set) marker is a safe no-op.
    cleared_again = await battle_gameplay_service.clear_disconnect_marker(fake_redis, match_id=match_id, user_id=user_id)
    assert cleared_again is False


async def test_mark_disconnected_skips_write_if_superseded_by_reconnect(fake_redis):
    """The exact race this fixes: a reconnect (which bumps the generation)
    happens before a stale disconnect notification gets a chance to write
    its marker -- the stale write must be a no-op, not leave a marker behind
    that would later wrongly forfeit the now-reconnected player.
    """
    match_id, user_id = 42, 7

    stale_generation = await battle_gameplay_service.bump_connection_generation(
        fake_redis, match_id=match_id, user_id=user_id
    )
    # A reconnect happens before the original connection's disconnect
    # notification runs -- bumps the generation again.
    await battle_gameplay_service.bump_connection_generation(fake_redis, match_id=match_id, user_id=user_id)

    deadline = await battle_gameplay_service.mark_disconnected(
        fake_redis, match_id=match_id, user_id=user_id, generation=stale_generation
    )
    assert deadline is None
    assert await battle_gameplay_service.is_still_disconnected(fake_redis, match_id=match_id, user_id=user_id) is False

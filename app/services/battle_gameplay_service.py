"""
app/services/battle_gameplay_service.py
─────────────────────────────────────────
Server-authoritative battle lifecycle: explicit ready-gated activation
(both participants must call mark_ready before a match proceeds -- no more
auto-activation on first poll, that was Phase 3's interim behavior), a
server-driven countdown, self-paced question progression (each player
moves on with Next/Skip; the whole match runs on one clock), answer
grading/scoring, match completion (winner + Elo + CompetitiveSubjectRating
updates), and forfeit. Works over REST polling AND WebSocket push
(app/api/routes/battle_ws.py) simultaneously -- this module has no
knowledge of which transport called it.

Every atomic transition ("waiting"->"countdown", "countdown"->"active",
"active"/"waiting"/"countdown"->"completed") uses a single-statement
Postgres/SQLite compare-and-swap: `UPDATE battle_matches SET ... WHERE
id=:id AND status=<expected> RETURNING id` -- if no row comes back, another
request already won that transition; this one just re-reads the committed
result. Mirrors Phase 2's Lua-atomicity philosophy, but as a single-row DB
CAS since this is single-row (not cross-user) coordination.

IMPORTANT identity-map gotcha: after a losing CAS branch, you MUST
`await db.refresh(match)` (not `db.get()`) to see the winner's committed
state -- `db.get()` returns the session's already-loaded (stale) copy from
its identity map without hitting the DB again.

Ready-state and disconnect-tracking are Redis-only, transient, no new
Postgres columns (battle:ready:{match_id}, battle:disconnect_deadline:
{match_id}:{user_id}) -- a naive HSET+read-back is sufficient for the
ready-flow race (no Lua needed) because the Postgres CAS in
_activate_countdown is what actually makes a double-trigger safe, exactly
like Phase 3's original activation CAS did; whether one or both processes
observe "both ready" in Redis doesn't matter, only one CAS ever wins.
"""
import asyncio
import contextlib
import logging
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException, status
from redis.asyncio import Redis
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.redis_client import redis_client
from app.models.battle_match import (
    BattleAnswer,
    BattleMatch,
    BattleMatchQuestion,
    BattleParticipant,
)
from app.models.competitive_rating import CompetitiveSubjectRating
from app.models.question import Question
from app.models.user import User
from app.schemas.battle_ws import (
    CountdownStartedEvent,
    MatchCancelledEvent,
    PlayerReadyEvent,
)
from app.services import battle_difficulty, elo_engine, league_engine
from app.services.battle_connection_manager import manager as ws_manager
from app.services.battle_question_service import (
    BattleQuestionSelectionError,
    select_battle_questions,
)
from app.services.battle_scoring_engine import (
    ParticipantOutcome,
    compute_question_score,
    determine_winner,
)
from app.services.elo_engine import compute_elo_update
from app.services.matchmaking_service import release_match_state

logger = logging.getLogger(__name__)


def as_utc(dt: datetime) -> datetime:
    # Copied from Personalized-Quiz-Service's analytics/queries.py -- SQLite
    # (the test DB engine) drops tzinfo on read while Postgres doesn't, so
    # every started_at comparison needs this normalization.
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


CANCEL_REASON_AI_FAILURE = (
    "Not enough questions could be generated or found for this subject/difficulty. "
    "The match was cancelled -- no rating changes were applied."
)
CANCEL_REASON_OPPONENT_LEFT_BEFORE_START = (
    "Your opponent left before the match started. No rating changes were applied."
)

_ACTIVE_STATUSES = ("waiting", "countdown", "active")


def _ready_key(match_id: int) -> str:
    return f"battle:ready:{match_id}"


def _connected_key(match_id: int) -> str:
    return f"battle:connected:{match_id}"


def _both_ready_at_key(match_id: int) -> str:
    return f"battle:both_ready_at:{match_id}"


# How much longer than the AI wait the "activating" marker lives -- covers the DB work around question selection,
# and still expires on its own if a process dies mid-activation.
_ACTIVATING_MARKER_SLACK_SECONDS = 30


def _activating_key(match_id: int) -> str:
    # Set while a match is being started (its questions picked -- AI first, so up to
    # BATTLE_AI_GENERATION_TIMEOUT_SECONDS); the stale-match reaper leaves such a match alone.
    return f"battle:activating:{match_id}"


def _disconnect_deadline_key(match_id: int, user_id: int) -> str:
    return f"battle:disconnect_deadline:{match_id}:{user_id}"


def _connection_generation_key(match_id: int, user_id: int) -> str:
    return f"battle:conn_gen:{match_id}:{user_id}"


def _rtt_samples_key(match_id: int, user_id: int) -> str:
    return f"battle:rtt:{match_id}:{user_id}"


async def record_rtt_sample(redis: Redis, *, match_id: int, user_id: int, rtt_seconds: float) -> None:
    """Keeps this player's last BATTLE_RTT_SAMPLE_COUNT measured round trips
    (server ping -> client pong, see battle_ws.py) -- the fastest of them is
    the network allowance taken off each answer's response time.
    """
    key = _rtt_samples_key(match_id, user_id)
    await redis.lpush(key, f"{max(0.0, rtt_seconds):.4f}")
    await redis.ltrim(key, 0, settings.BATTLE_RTT_SAMPLE_COUNT - 1)
    await redis.expire(key, settings.BATTLE_CONNECTION_GENERATION_TTL_SECONDS)


async def network_allowance_seconds(redis: Redis | None, *, match_id: int, user_id: int) -> float:
    """The fastest recent round trip, capped at
    BATTLE_NETWORK_ALLOWANCE_MAX_SECONDS. The minimum (not the average) so a
    single slow ping can't inflate it, and the cap bounds what a client that
    deliberately delays its pongs could ever gain. 0 with no samples yet
    (e.g. a REST-only client).
    """
    if redis is None:
        return 0.0
    samples = [float(v) for v in await redis.lrange(_rtt_samples_key(match_id, user_id), 0, -1)]
    if not samples:
        return 0.0
    return min(min(samples), settings.BATTLE_NETWORK_ALLOWANCE_MAX_SECONDS)


# ── internal lookups ─────────────────────────────────────────────────────

async def _get_match_or_404(db: AsyncSession, match_id: int) -> BattleMatch:
    match = await db.get(BattleMatch, match_id)
    if match is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Battle match not found.")
    return match


async def get_match_row(db: AsyncSession, match_id: int) -> BattleMatch | None:
    """Plain lookup, no side effects, no raising -- distinct from
    _get_match_or_404. For callers (e.g. the WS progression timer) that
    aren't already inside a request with a match guaranteed to exist.
    """
    return await db.get(BattleMatch, match_id)


async def _get_participants(db: AsyncSession, match_id: int) -> list[BattleParticipant]:
    return list(
        (
            await db.execute(
                select(BattleParticipant).where(BattleParticipant.match_id == match_id)
            )
        )
        .scalars()
        .all()
    )


def _require_participant(participants: list[BattleParticipant], user_id: int) -> BattleParticipant:
    me = next((p for p in participants if p.user_id == user_id), None)
    if me is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "You are not a participant in this match.")
    return me


# ── ready flow (waiting -> countdown), exactly-once ──────────────────────
#
# Activation fires once a participant is BOTH ready (mark_ready) AND has a
# live WS connection confirmed (mark_connected) -- whichever of the two
# signals completes the pair LAST triggers it, via the shared
# _try_activate_if_ready_and_connected check both call at their own tail.
# This guards against the exact race documented on BATTLE_CONNECTION_READY_
# TIMEOUT_SECONDS: both players tapping ready doesn't guarantee either
# socket is actually live yet, and starting the match's clock before a
# client is even connected to observe question 1 silently eats into its
# answerable window. run_stale_match_reap_once force-activates a match
# that's been ready-but-not-connected for too long -- see its docstring for
# why that fallback (not this module) is what keeps a REST-only client (one
# that never opens a socket, so mark_connected never fires for it) working.


async def _try_activate_if_ready_and_connected(
    db: AsyncSession, redis: Redis, *, match: BattleMatch, participants: list[BattleParticipant],
) -> BattleMatch:
    all_ids = {p.user_id for p in participants}
    ready_ids = {int(uid) for uid in await redis.hkeys(_ready_key(match.id))}
    if not (all_ids <= ready_ids):
        return match

    # Both ready -- record the FIRST time this becomes true (NX), so
    # run_stale_match_reap_once can measure how long a connection
    # confirmation has been pending from a fixed point, not from whenever it
    # happens to next check.
    await redis.set(
        _both_ready_at_key(match.id), datetime.now(UTC).isoformat(),
        nx=True, ex=settings.BATTLE_READY_TTL_SECONDS,
    )

    connected_ids = {int(uid) for uid in await redis.hkeys(_connected_key(match.id))}
    if not (all_ids <= connected_ids):
        return match  # ready, but still waiting on a WS connection to confirm

    match, _started_it = await _finalize_activation(db, redis, match=match)
    return match


async def _finalize_activation(db: AsyncSession, redis: Redis, *, match: BattleMatch) -> tuple[BattleMatch, bool]:
    """Starts a ready match: picks its questions and announces the countdown. Returns the match and whether THIS
    call started it (False: another call already had)."""
    activating_key = _activating_key(match.id)
    await redis.set(
        activating_key, "1", ex=int(settings.BATTLE_AI_GENERATION_TIMEOUT_SECONDS + _ACTIVATING_MARKER_SLACK_SECONDS),
    )
    try:
        match, started_it = await _activate_countdown(db, match, redis)
    finally:
        with contextlib.suppress(Exception):
            await redis.delete(activating_key)
    # Only the call that actually started the countdown announces it. Picking
    # questions can now take seconds (AI first), and a second trigger arriving
    # meanwhile (the other player's connect, the stale-match reaper's forced
    # activation) waits on the first and then finds it already done -- it
    # mustn't announce the countdown (or a cancellation) a second time.
    if not started_it:
        return match, False
    if match.status == "countdown":
        await ws_manager.publish(
            redis, match_id=match.id,
            event=CountdownStartedEvent(
                started_at=match.started_at,
                duration_seconds=settings.BATTLE_COUNTDOWN_DURATION_SECONDS,
            ).model_dump(mode="json"),
        )
    elif match.status == "cancelled":
        await ws_manager.publish(
            redis, match_id=match.id,
            event=MatchCancelledEvent(reason=match.cancel_reason).model_dump(mode="json"),
        )
    return match, True


async def mark_ready(db: AsyncSession, redis: Redis, *, match_id: int, user_id: int) -> BattleMatch:
    match = await _get_match_or_404(db, match_id)
    participants = await _get_participants(db, match_id)
    _require_participant(participants, user_id)

    if match.status != "waiting":
        return match  # idempotent no-op -- already past the ready-gate

    key = _ready_key(match_id)
    await redis.hset(key, str(user_id), "1")
    await redis.expire(key, settings.BATTLE_READY_TTL_SECONDS)

    await ws_manager.publish(
        redis, match_id=match_id,
        event=PlayerReadyEvent(user_id=user_id).model_dump(mode="json"),
    )

    return await _try_activate_if_ready_and_connected(db, redis, match=match, participants=participants)


async def mark_connected(db: AsyncSession, redis: Redis, *, match_id: int, user_id: int) -> BattleMatch | None:
    """Records that user_id's WebSocket is live for this match -- called by
    the WS route right after accepting the connection (see battle_ws.py).
    A no-op (returns None) once the match is past "waiting" -- already
    activated, or this is a routine mid-match reconnect, neither of which
    this gate is about. See this module's ready-flow section docstring for
    the overall design.
    """
    match = await get_match_row(db, match_id)
    if match is None or match.status != "waiting":
        return None

    participants = await _get_participants(db, match_id)
    if not any(p.user_id == user_id for p in participants):
        return None  # not a participant -- battle_ws.py already rejects this pre-accept, defensive only

    key = _connected_key(match_id)
    await redis.hset(key, str(user_id), "1")
    await redis.expire(key, settings.BATTLE_READY_TTL_SECONDS)

    return await _try_activate_if_ready_and_connected(db, redis, match=match, participants=participants)


async def _activate_countdown(db: AsyncSession, match: BattleMatch, redis: Redis) -> tuple[BattleMatch, bool]:
    """Claims a "waiting" match and picks its questions. Returns the match and
    whether THIS call made the transition (False: another call already had)."""
    # started_at is deliberately NOT set here -- see below, right before the
    # final commit, for why. This whole function runs as one open, uncommitted
    # transaction until then, so no other reader (REST poll, another
    # process's progression timer) can observe "countdown" with a not-yet-
    # correct started_at in between -- nothing becomes visible until commit.
    claim_stmt = (
        update(BattleMatch)
        .where(BattleMatch.id == match.id, BattleMatch.status == "waiting")
        .values(status="countdown")
        .returning(BattleMatch.id)
    )
    claimed = (await db.execute(claim_stmt)).scalar_one_or_none()

    if claimed is None:
        # Lost the race (or there was nothing to claim) -- pick up whatever
        # the winner committed. MUST refresh, not db.get() (identity map).
        await db.commit()
        await db.refresh(match)
        return match, False

    await db.refresh(match)  # pick up status this UPDATE just set (same transaction, uncommitted)
    # match.difficulty is already the band label get_difficulty_label()
    # produced at match-creation time (matchmaking_service._create_match) --
    # look the same band's {easy,medium,hard} split back up rather than
    # recomputing it from participants' ratings, so there's exactly one
    # source of truth for which band this match is in.
    distribution = battle_difficulty.BATTLE_DIFFICULTY_DISTRIBUTION[match.difficulty]
    tier_counts = battle_difficulty.allocate_question_counts(distribution, settings.BATTLE_QUESTION_COUNT)
    participants = await _get_participants(db, match.id)
    try:
        questions = await select_battle_questions(
            db, subject=match.subject, tier_counts=tier_counts,
            participant_user_ids=(participants[0].user_id, participants[1].user_id),
        )
    except BattleQuestionSelectionError:
        # AI Failure Handling: cancel cleanly, no rating changes, no
        # partially-started match. Both players see this on their next poll.
        logger.warning("Battle match %d cancelled — question selection failed.", match.id)
        match.status = "cancelled"
        match.cancel_reason = CANCEL_REASON_AI_FAILURE
        match.started_at = None
        await db.commit()
        # Without this, "in_match" (no TTL) would stick forever for both
        # players and neither could ever join a new queue again -- the exact
        # bug _finalize_match/forfeit_match already guard against below.
        participants = await _get_participants(db, match.id)
        for p in participants:
            await release_match_state(redis, user_id=p.user_id)
        return match, True

    for order, question in enumerate(questions):
        db.add(BattleMatchQuestion(match_id=match.id, question_id=question.id, question_order=order))
    # Anchor the whole match's question-timing clock to NOW -- i.e. AFTER
    # select_battle_questions has already finished, however long that took
    # (AI generation can genuinely run several real seconds, especially
    # under Groq rate-limiting/retries) -- not to a timestamp captured
    # before that potentially-slow work ran. Nothing outside this function
    # can observe "countdown" before this same commit anyway (single open
    # transaction throughout), so there's no earlier, briefly-wrong value
    # for anyone to have raced on. Getting this wrong silently deducted
    # selection's own latency from the match clock (and question 1's speed
    # bonus) before either player had even been told the match was starting.
    match.started_at = datetime.now(UTC) + timedelta(seconds=settings.BATTLE_COUNTDOWN_DURATION_SECONDS)
    await db.commit()
    await db.refresh(match)
    logger.info(
        "Battle match %d entered countdown — subject=%s difficulty=%s starts_at=%s",
        match.id, match.subject, match.difficulty, match.started_at,
    )
    return match, True


# ── countdown -> active, exactly-once ─────────────────────────────────────

async def _activate_from_countdown(db: AsyncSession, match: BattleMatch) -> BattleMatch:
    claim_stmt = (
        update(BattleMatch)
        .where(BattleMatch.id == match.id, BattleMatch.status == "countdown")
        .values(status="active")
        .returning(BattleMatch.id)
    )
    claimed = (await db.execute(claim_stmt)).scalar_one_or_none()
    await db.commit()
    await db.refresh(match)
    if claimed is not None:
        logger.info("Battle match %d is now active.", match.id)
    return match


# ── match completion (-> completed), exactly-once ─────────────────────────

async def _answers_for(db: AsyncSession, *, match_id: int, user_id: int) -> list[BattleAnswer]:
    return list(
        (
            await db.execute(
                select(BattleAnswer).where(
                    BattleAnswer.match_id == match_id, BattleAnswer.user_id == user_id
                )
            )
        )
        .scalars()
        .all()
    )


def _participant_outcome(user_id: int, answers: list[BattleAnswer]) -> ParticipantOutcome:
    correct = [a for a in answers if a.is_correct]
    return ParticipantOutcome(
        user_id=user_id,
        correct_count=len(correct),
        total_score=sum(a.total_question_score for a in answers),  # wrong answers already 0
        total_response_time_ms=sum(a.response_time_ms for a in correct),  # correct-only, per spec
    )


async def _get_or_build_rating_no_commit(
    db: AsyncSession, *, user_id: int, subject: str,
) -> CompetitiveSubjectRating:
    row = (
        await db.execute(
            select(CompetitiveSubjectRating).where(
                CompetitiveSubjectRating.user_id == user_id,
                CompetitiveSubjectRating.subject == subject,
            )
        )
    ).scalar_one_or_none()
    if row is not None:
        return row
    # Defensive only -- should be unreachable given Phase 2 always creates
    # this row at queue-join time. db.add() WITHOUT commit -- this row joins
    # the same transaction the caller commits once, at the end.
    row = CompetitiveSubjectRating(
        user_id=user_id, subject=subject,
        rating=settings.BATTLE_INITIAL_RATING, highest_rating=settings.BATTLE_INITIAL_RATING,
    )
    db.add(row)
    return row


def _result_label(winner_user_id: int | None, this_user_id: int, *, is_draw: bool) -> str:
    if is_draw:
        return "draw"
    return "win" if winner_user_id == this_user_id else "loss"


def _apply_result(
    rating: CompetitiveSubjectRating, participant: BattleParticipant,
    outcome: ParticipantOutcome, answers: list[BattleAnswer], *,
    new_rating: int, stat_bucket: str, label: str, best_streak: int,
) -> None:
    """stat_bucket drives the win/loss/draw counters and streak bookkeeping
    ("win"|"loss"|"draw"); label is what gets written to
    participant.result. Normal completion passes the same value for both
    (stat_bucket==label); forfeit passes stat_bucket="loss" but
    label="forfeit" for the forfeiting side, so their streak/loss counters
    behave exactly like a normal loss while the stored result still
    distinguishes "lost by forfeit" from "lost by playing worse."
    """
    # `answers` includes a row for every question this player skipped
    # (selected_option=="" sentinel, see submit_battle_answer) -- filter
    # those back out here so a skip doesn't inflate wrong_count or get
    # counted toward average response time.
    submitted_answers = [a for a in answers if a.selected_option]
    answered_count = len(submitted_answers)
    # ALL answered questions count toward average response time (not just
    # correct ones) -- matches how QuestionAttempt.response_time is tracked
    # elsewhere regardless of correctness.
    total_response_time_s = sum(a.response_time_ms for a in submitted_answers) / 1000.0

    # Running (cumulative) average: new_avg = (old_avg*old_count + new_sum) / (old_count+new_count).
    old_count = rating.total_answers
    new_count = old_count + answered_count
    if new_count > 0:
        rating.average_response_time = (
            (rating.average_response_time * old_count) + total_response_time_s
        ) / new_count

    rating.total_answers = new_count
    rating.correct_answers += outcome.correct_count
    rating.matches_played += 1
    rating.rating = new_rating
    rating.highest_rating = max(rating.highest_rating, new_rating)

    if stat_bucket == "win":
        rating.wins += 1
        rating.current_win_streak += 1
        rating.highest_win_streak = max(rating.highest_win_streak, rating.current_win_streak)
    elif stat_bucket == "loss":
        rating.losses += 1
        rating.current_win_streak = 0
    else:  # draw
        rating.draws += 1
        rating.current_win_streak = 0

    participant.final_score = outcome.total_score
    participant.correct_count = outcome.correct_count
    participant.wrong_count = answered_count - outcome.correct_count
    participant.best_streak = best_streak
    participant.total_response_time_ms = sum(a.response_time_ms for a in answers)
    participant.rating_after = new_rating
    participant.rating_change = new_rating - participant.rating_before
    participant.result = label


async def _is_rated_pairing(db: AsyncSession, *, match: BattleMatch, user_a_id: int, user_b_id: int) -> bool:
    """False once these two players have already completed
    BATTLE_REPEAT_OPPONENT_RATED_LIMIT other matches against each other in the
    last BATTLE_REPEAT_OPPONENT_WINDOW_HOURS (any subject) -- see that setting.
    `match` itself is excluded: it's already flipped to "completed" inside the
    caller's transaction by the time this runs.
    """
    cutoff = datetime.now(UTC) - timedelta(hours=settings.BATTLE_REPEAT_OPPONENT_WINDOW_HOURS)
    p_a, p_b = aliased(BattleParticipant), aliased(BattleParticipant)
    previous = (
        await db.execute(
            select(func.count(BattleMatch.id))
            .join(p_a, (p_a.match_id == BattleMatch.id) & (p_a.user_id == user_a_id))
            .join(p_b, (p_b.match_id == BattleMatch.id) & (p_b.user_id == user_b_id))
            .where(
                BattleMatch.id != match.id,
                BattleMatch.status == "completed",
                BattleMatch.finished_at >= cutoff,
            )
        )
    ).scalar_one()
    rated = previous < settings.BATTLE_REPEAT_OPPONENT_RATED_LIMIT
    if not rated:
        logger.info(
            "Battle match %d unrated — users %d and %d already played %d times in the last %dh",
            match.id, user_a_id, user_b_id, previous, settings.BATTLE_REPEAT_OPPONENT_WINDOW_HOURS,
        )
    return rated


async def _finalize_match(db: AsyncSession, match: BattleMatch, redis: Redis | None = None) -> BattleMatch:
    now = datetime.now(UTC)
    claim_stmt = (
        update(BattleMatch)
        .where(BattleMatch.id == match.id, BattleMatch.status == "active")
        .values(status="completed", finished_at=now)
        .returning(BattleMatch.id)
    )
    claimed = (await db.execute(claim_stmt)).scalar_one_or_none()
    if claimed is None:
        await db.commit()
        await db.refresh(match)
        return match

    # Won the race -- everything below happens in THIS SAME transaction,
    # committed exactly once at the end, so a duplicate completion trigger
    # can never double-apply rating changes.
    participants = await _get_participants(db, match.id)
    p_a, p_b = participants[0], participants[1]

    answers_a = await _answers_for(db, match_id=match.id, user_id=p_a.user_id)
    answers_b = await _answers_for(db, match_id=match.id, user_id=p_b.user_id)

    outcome_a = _participant_outcome(p_a.user_id, answers_a)
    outcome_b = _participant_outcome(p_b.user_id, answers_b)
    winner = determine_winner(outcome_a, outcome_b)

    # Fresh reads, NOT battle_service.get_or_create_subject_rating() -- that
    # function commits internally on the (should-never-happen-here) create
    # path, which would break this transaction's atomicity.
    rating_a = await _get_or_build_rating_no_commit(db, user_id=p_a.user_id, subject=match.subject)
    rating_b = await _get_or_build_rating_no_commit(db, user_id=p_b.user_id, subject=match.subject)

    elo = compute_elo_update(rating_a.rating, rating_b.rating, winner.result_for_a)
    rated = await _is_rated_pairing(db, match=match, user_a_id=p_a.user_id, user_b_id=p_b.user_id)
    new_rating_a = elo.new_rating_a if rated else rating_a.rating
    new_rating_b = elo.new_rating_b if rated else rating_b.rating

    best_streak_a = await _best_streak_for(db, match_id=match.id, user_id=p_a.user_id)
    best_streak_b = await _best_streak_for(db, match_id=match.id, user_id=p_b.user_id)

    is_draw = winner.winner_user_id is None
    result_a = _result_label(winner.winner_user_id, p_a.user_id, is_draw=is_draw)
    result_b = _result_label(winner.winner_user_id, p_b.user_id, is_draw=is_draw)
    _apply_result(
        rating_a, p_a, outcome_a, answers_a, new_rating=new_rating_a,
        stat_bucket=result_a, label=result_a, best_streak=best_streak_a,
    )
    _apply_result(
        rating_b, p_b, outcome_b, answers_b, new_rating=new_rating_b,
        stat_bucket=result_b, label=result_b, best_streak=best_streak_b,
    )

    match.winner_user_id = winner.winner_user_id
    await db.commit()
    await db.refresh(match)
    logger.info("Battle match %d completed — winner_user_id=%s", match.id, match.winner_user_id)
    if redis is not None:
        # Without this, "in_match" (no TTL, see matchmaking_service's
        # module docstring) would stick forever and the player could never
        # join a new queue again.
        await release_match_state(redis, user_id=p_a.user_id)
        await release_match_state(redis, user_id=p_b.user_id)
    return match


# ── forfeit (any pre-terminal status -> completed), exactly-once ─────────

FINISHED_CANT_FORFEIT_DETAIL = "You've already finished — your result stands."


async def forfeit_match(
    db: AsyncSession, redis: Redis, *, match_id: int, forfeiting_user_id: int
) -> BattleMatch:
    match = await _get_match_or_404(db, match_id)
    participants = await _get_participants(db, match_id)
    me = _require_participant(participants, forfeiting_user_id)  # 403 if not a participant

    if match.status not in _ACTIVE_STATUSES:
        return match  # idempotent no-op -- already resolved

    # A player who has already answered or skipped every question is just
    # waiting for the opponent (possibly from the home screen) -- their
    # result stands and can't be thrown away by a forfeit.
    await db.refresh(me)
    if me.current_question_index >= settings.BATTLE_QUESTION_COUNT:
        raise HTTPException(status.HTTP_409_CONFLICT, FINISHED_CANT_FORFEIT_DETAIL)

    # "waiting" (matched but not yet both-ready) or a countdown that hasn't
    # reached started_at yet -- no question has been shown to either player,
    # so there's nothing to score. Void the match instead of recording a
    # scored forfeit (which used to cost the leaver rating for a match that
    # never began), and let the remaining player's client auto-requeue off
    # the resulting match_cancelled event (see queue.tsx) rather than
    # dead-ending on a "you won" screen. A countdown whose started_at has
    # passed is really "active" (the flip is lazy) and is scored below.
    now = datetime.now(UTC)
    if match.status == "waiting" or (match.status == "countdown" and now < as_utc(match.started_at)):
        return await _cancel_unstarted_match(db, redis, match=match, participants=participants, now=now)

    claim_stmt = (
        update(BattleMatch)
        .where(BattleMatch.id == match.id, BattleMatch.status.in_(_ACTIVE_STATUSES))
        .values(status="completed", finished_at=now)
        .returning(BattleMatch.id)
    )
    claimed = (await db.execute(claim_stmt)).scalar_one_or_none()
    if claimed is None:
        await db.commit()
        await db.refresh(match)
        return match

    p_a, p_b = participants[0], participants[1]
    forfeiter, winner_p = (p_a, p_b) if p_a.user_id == forfeiting_user_id else (p_b, p_a)

    answers_a = await _answers_for(db, match_id=match.id, user_id=p_a.user_id)
    answers_b = await _answers_for(db, match_id=match.id, user_id=p_b.user_id)
    outcome_a = _participant_outcome(p_a.user_id, answers_a)
    outcome_b = _participant_outcome(p_b.user_id, answers_b)

    rating_a = await _get_or_build_rating_no_commit(db, user_id=p_a.user_id, subject=match.subject)
    rating_b = await _get_or_build_rating_no_commit(db, user_id=p_b.user_id, subject=match.subject)

    result_for_a = elo_engine.RESULT_LOSS if forfeiter is p_a else elo_engine.RESULT_WIN
    elo = compute_elo_update(rating_a.rating, rating_b.rating, result_for_a)
    rated = await _is_rated_pairing(db, match=match, user_a_id=p_a.user_id, user_b_id=p_b.user_id)
    new_rating_a = elo.new_rating_a if rated else rating_a.rating
    new_rating_b = elo.new_rating_b if rated else rating_b.rating

    best_streak_a = await _best_streak_for(db, match_id=match.id, user_id=p_a.user_id)
    best_streak_b = await _best_streak_for(db, match_id=match.id, user_id=p_b.user_id)

    _apply_result(
        rating_a, p_a, outcome_a, answers_a, new_rating=new_rating_a,
        stat_bucket=("loss" if forfeiter is p_a else "win"),
        label=("forfeit" if forfeiter is p_a else "win"), best_streak=best_streak_a,
    )
    _apply_result(
        rating_b, p_b, outcome_b, answers_b, new_rating=new_rating_b,
        stat_bucket=("loss" if forfeiter is p_b else "win"),
        label=("forfeit" if forfeiter is p_b else "win"), best_streak=best_streak_b,
    )

    match.winner_user_id = winner_p.user_id
    await db.commit()
    await db.refresh(match)
    logger.info(
        "Battle match %d forfeited by user_id=%s — winner_user_id=%s",
        match.id, forfeiting_user_id, match.winner_user_id,
    )
    await release_match_state(redis, user_id=p_a.user_id)
    await release_match_state(redis, user_id=p_b.user_id)
    return match


async def _cancel_unstarted_match(
    db: AsyncSession, redis: Redis, *, match: BattleMatch, participants: list[BattleParticipant],
    now: datetime,
) -> BattleMatch:
    # Exactly-once like every other transition: only claims a match that is
    # still waiting, or still counting down with started_at in the future
    # (so a forfeit racing the countdown->active flip can't void a match
    # that has just begun).
    not_started = (BattleMatch.status == "waiting") | (
        (BattleMatch.status == "countdown") & (BattleMatch.started_at > now)
    )
    claim_stmt = (
        update(BattleMatch)
        .where(BattleMatch.id == match.id, not_started)
        .values(status="cancelled", cancel_reason=CANCEL_REASON_OPPONENT_LEFT_BEFORE_START)
        .returning(BattleMatch.id)
        # No in-Python re-evaluation of the WHERE (it would compare a naive
        # SQLite datetime against `now`); `match` is refreshed below instead.
        .execution_options(synchronize_session=False)
    )
    claimed = (await db.execute(claim_stmt)).scalar_one_or_none()
    if claimed is None:
        await db.commit()
        await db.refresh(match)
        return match

    await db.commit()
    await db.refresh(match)
    logger.info("Battle match %d cancelled — one player left before the match started.", match.id)
    for p in participants:
        await release_match_state(redis, user_id=p.user_id)
    return match


# ── disconnect / reconnect presence (Redis-only, transient) ──────────────

# KEYS[1]=conn_gen  KEYS[2]=disconnect_deadline
# ARGV[1]=expected_generation  ARGV[2]=deadline_iso  ARGV[3]=ttl
#
# Guards against a race where a reconnect's clear_disconnect_marker() runs
# BEFORE the disconnect it's superseding gets a chance to write this marker
# at all: since _notify_disconnect (battle_ws.py) is a detached, unawaited
# task, there's no ordering guarantee between "socket A closed, notify task
# scheduled" and "socket B (same user) reconnected, marker cleared". Without
# this check, A's write could land AFTER B's now-moot clear, leaving a stale
# marker that the opponent's grace-period countdown later reads and wrongly
# forfeits an actually-connected player. Only write if nobody has connected
# (bumping the generation) since the disconnecting socket itself connected.
_LUA_MARK_DISCONNECTED_IF_CURRENT = """
local current = redis.call('GET', KEYS[1])
if current ~= ARGV[1] then
    return 0
end
redis.call('SET', KEYS[2], ARGV[2], 'EX', ARGV[3])
return 1
"""


async def bump_connection_generation(redis: Redis, *, match_id: int, user_id: int) -> int:
    """Call once per successful WS connect, before clear_disconnect_marker.
    The returned value must be threaded through to this connection's own
    eventual disconnect handling (see mark_disconnected's `generation` arg)
    -- it's what lets a later, stale disconnect notification recognize it's
    been superseded by this (or an even newer) connection.
    """
    key = _connection_generation_key(match_id, user_id)
    generation = await redis.incr(key)
    await redis.expire(key, settings.BATTLE_CONNECTION_GENERATION_TTL_SECONDS)
    return generation


async def mark_disconnected(
    redis: Redis, *, match_id: int, user_id: int, generation: int
) -> datetime | None:
    """Writes the disconnect deadline marker only if `generation` (this
    connection's own, captured at connect time via bump_connection_generation)
    is still the CURRENT generation for this user+match -- i.e. no reconnect
    has happened since. Returns the deadline if written, or None if skipped
    because a newer connection already exists (see the Lua script's
    docstring above for the race this closes).
    """
    deadline = datetime.now(UTC) + timedelta(seconds=settings.BATTLE_DISCONNECT_GRACE_PERIOD_SECONDS)
    ttl = int(settings.BATTLE_DISCONNECT_GRACE_PERIOD_SECONDS + settings.BATTLE_DISCONNECT_MARKER_BUFFER_SECONDS)
    written = await redis.eval(
        _LUA_MARK_DISCONNECTED_IF_CURRENT,
        2,
        _connection_generation_key(match_id, user_id), _disconnect_deadline_key(match_id, user_id),
        str(generation), deadline.isoformat(), str(ttl),
    )
    return deadline if written else None


async def clear_disconnect_marker(redis: Redis, *, match_id: int, user_id: int) -> bool:
    return bool(await redis.delete(_disconnect_deadline_key(match_id, user_id)))


async def is_still_disconnected(redis: Redis, *, match_id: int, user_id: int) -> bool:
    return await redis.exists(_disconnect_deadline_key(match_id, user_id)) == 1


# ── shared "notice time has moved on" helper, used by REST + WS alike ────

def match_ends_at(match: BattleMatch) -> datetime:
    """When the match clock the players see runs out."""
    return as_utc(match.started_at) + timedelta(seconds=settings.BATTLE_MATCH_DURATION_SECONDS)


def player_answers_close_at(match: BattleMatch, allowance_seconds: float) -> datetime:
    """One player's cutoff: 0:00 as THEIR phone saw it. The phone's countdown
    runs about one round trip behind the server's clock (it learns the time
    from messages that took that long to arrive, then its answer takes the
    same again), so an answer sent before its own 0:00 arrives up to that
    allowance after the server's -- see network_allowance_seconds.
    """
    return match_ends_at(match) + timedelta(
        seconds=settings.BATTLE_ANSWER_LATE_GRACE_SECONDS + allowance_seconds
    )


def answers_close_at(match: BattleMatch) -> datetime:
    """When the server stops accepting answers from anyone -- the latest any
    player's own cutoff (player_answers_close_at) can be. The match is only
    finalized after this, so no answer that's still in time gets shut out.
    """
    return player_answers_close_at(match, settings.BATTLE_NETWORK_ALLOWANCE_MAX_SECONDS)


async def match_end_reason(db: AsyncSession, match_id: int) -> str:
    """Why a completed match ended: "forfeit" (someone forfeited, incl. a
    disconnect that ran out its grace period), "all_finished" (both players
    answered or skipped every question) or "time_up" (the match clock ran
    out first). Worked out from the participants' own columns -- fresh, not
    whatever the session has cached -- so clients never have to guess it
    from progress they may only partly have seen.
    """
    rows = (
        await db.execute(
            select(BattleParticipant.result, BattleParticipant.current_question_index)
            .where(BattleParticipant.match_id == match_id)
        )
    ).all()
    if any(result == "forfeit" for result, _ in rows):
        return "forfeit"
    if rows and all(index >= settings.BATTLE_QUESTION_COUNT for _, index in rows):
        return "all_finished"
    return "time_up"


async def has_finished(db: AsyncSession, *, match_id: int, user_id: int) -> bool:
    """True once this player has answered or skipped every question. A column
    query, so it's fresh even if the session holds an older participant row.
    """
    index = await db.scalar(
        select(BattleParticipant.current_question_index).where(
            BattleParticipant.match_id == match_id, BattleParticipant.user_id == user_id
        )
    )
    return index is not None and index >= settings.BATTLE_QUESTION_COUNT


async def _all_finished(db: AsyncSession, match_id: int) -> bool:
    # A column query, not the ORM objects: those stay cached in the session
    # (expire_on_commit=False), and the opponent may have finished since they
    # were loaded -- two players finishing at once must not both miss it.
    indexes = (
        await db.execute(
            select(BattleParticipant.current_question_index).where(BattleParticipant.match_id == match_id)
        )
    ).scalars().all()
    return bool(indexes) and all(i >= settings.BATTLE_QUESTION_COUNT for i in indexes)


async def _ensure_match_progressed(
    db: AsyncSession, match: BattleMatch, redis: Redis | None = None
) -> BattleMatch:
    # "waiting" is intentionally NOT handled here -- only mark_ready() (both
    # sides required) can move a match out of "waiting". Polling alone
    # (REST or the WS progression timer) can no longer auto-activate it.
    if match.status == "countdown" and datetime.now(UTC) >= as_utc(match.started_at):
        match = await _activate_from_countdown(db, match)
    if match.status == "active" and (
        datetime.now(UTC) >= answers_close_at(match) or await _all_finished(db, match.id)
    ):
        match = await _finalize_match(db, match, redis)
    return match


# ── GET .../state ────────────────────────────────────────────────────────

async def question_payload_for(db: AsyncSession, match: BattleMatch, order: int) -> dict | None:
    """The client-facing shape of the single question at this question_order
    -- NEVER includes correct_answer, spec's "Do Not Send Correct Answers
    Early". None if `order` is out of range (e.g. the total time budget just
    elapsed and _finalize_match hasn't run yet -- callers treat that as "no
    live question").
    """
    row = (
        await db.execute(
            select(BattleMatchQuestion, Question)
            .join(Question, Question.id == BattleMatchQuestion.question_id)
            .where(BattleMatchQuestion.match_id == match.id, BattleMatchQuestion.question_order == order)
        )
    ).first()
    if row is None:
        return None
    _mq, question = row
    return {"question_id": question.id, "question": question.question, "options": _match_option_order(match, question)}


def _match_option_order(match: BattleMatch, question: Question) -> list:
    """The question's options in a per-match order. Pool questions recur
    across matches, so a fixed order lets regulars learn "it's always B"
    without reading the question. Seeded by (match, question) so both players
    -- and any reconnect -- see the same order; grading compares the chosen
    option's text, never its position, so the shuffle is invisible to it.
    """
    options = list(question.options or [])
    random.Random(f"{match.id}:{question.id}").shuffle(options)
    return options


async def _progress_answers(db: AsyncSession, *, match_id: int, user_id: int) -> list[dict]:
    """Every question this player has moved past so far, as {question_order,
    is_correct, answered} ordered by question_order -- feeds both players'
    progress bars (seeded on load, then kept live via player_progress).
    `answered` is False for a skip; is_correct is always None, since nobody
    learns whether they were right until the match is over.
    """
    rows = (
        await db.execute(
            select(BattleMatchQuestion.question_order, BattleAnswer.selected_option)
            .join(
                BattleAnswer,
                (BattleAnswer.match_id == BattleMatchQuestion.match_id)
                & (BattleAnswer.question_id == BattleMatchQuestion.question_id)
                & (BattleAnswer.user_id == user_id),
            )
            .where(BattleMatchQuestion.match_id == match_id)
            .order_by(BattleMatchQuestion.question_order.asc())
        )
    ).all()
    return [
        {"question_order": order, "is_correct": None, "answered": bool(selected_option)}
        for order, selected_option in rows
    ]


async def _question_review(db: AsyncSession, *, match: BattleMatch, user_id: int) -> list[dict]:
    """The end-of-match review: every question in match order with its right
    answer next to this player's own pick and points. Only ever built for
    the viewer themself -- the opponent's picks are never sent.
    """
    rows = (
        await db.execute(
            select(BattleMatchQuestion.question_order, Question, BattleAnswer)
            .join(Question, Question.id == BattleMatchQuestion.question_id)
            .outerjoin(
                BattleAnswer,
                (BattleAnswer.match_id == BattleMatchQuestion.match_id)
                & (BattleAnswer.question_id == BattleMatchQuestion.question_id)
                & (BattleAnswer.user_id == user_id),
            )
            .where(BattleMatchQuestion.match_id == match.id)
            .order_by(BattleMatchQuestion.question_order.asc())
        )
    ).all()
    review = []
    for order, question, answer in rows:
        if answer is None:
            status_label = "unanswered"
        elif not answer.selected_option:
            status_label = "skipped"
        else:
            status_label = "correct" if answer.is_correct else "incorrect"
        review.append({
            "question_order": order,
            "question": question.question,
            "options": _match_option_order(match, question),
            "correct_answer": question.correct_answer,
            "selected_option": (answer.selected_option or None) if answer is not None else None,
            "status": status_label,
            "response_time_ms": answer.response_time_ms if answer is not None and answer.selected_option else None,
            "base_score": answer.base_score if answer is not None else 0,
            "speed_bonus": answer.speed_bonus if answer is not None else 0,
            "streak_bonus": answer.streak_bonus if answer is not None else 0,
            "total_question_score": answer.total_question_score if answer is not None else 0,
        })
    return review


def participant_result_payload(p: BattleParticipant, *, username: str | None = None) -> dict:
    """Shared by _build_completed_state, get_battle_history, and
    battle_ws_events.match_finished_event. `rating_delta` is the wire name
    for the stored `rating_change` column (kept distinct so existing mobile
    clients reading `rating_delta` are unaffected). `username` is left None
    for "me" everywhere -- callers already know their own name locally
    (Clerk/profile) -- and is only ever looked up and passed for the
    opponent side, so the client can show a real name instead of a generic
    "Opponent" placeholder.
    """
    return {
        "user_id": p.user_id, "username": username,
        "correct_count": p.correct_count, "wrong_count": p.wrong_count,
        "final_score": p.final_score, "best_streak": p.best_streak,
        "total_response_time_ms": p.total_response_time_ms,
        "rating_before": p.rating_before, "rating_after": p.rating_after,
        "rating_delta": p.rating_change,
        "league": league_engine.get_league(p.rating_after) if p.rating_after is not None else None,
        "result": p.result,
    }


async def get_battle_history(
    db: AsyncSession, *, user_id: int, subject: str | None, page: int, page_size: int,
) -> dict:
    """Paginated match history for one user, newest-finished first. Only
    'completed' matches are included -- 'cancelled' matches (AI-generation
    failure at countdown) never had rating changes applied and aren't
    meaningful history entries.
    """
    conditions = [BattleParticipant.user_id == user_id, BattleMatch.status == "completed"]
    if subject is not None:
        conditions.append(BattleMatch.subject == subject)

    id_query = (
        select(BattleMatch.id)
        .join(BattleParticipant, BattleParticipant.match_id == BattleMatch.id)
        .where(*conditions)
    )
    total_count = (
        await db.execute(select(func.count()).select_from(id_query.subquery()))
    ).scalar() or 0

    offset = (page - 1) * page_size
    match_ids = (
        await db.execute(
            id_query.order_by(BattleMatch.finished_at.desc(), BattleMatch.id.desc())
            .offset(offset)
            .limit(page_size)
        )
    ).scalars().all()

    if not match_ids:
        return {"page": page, "page_size": page_size, "total_count": total_count, "total_pages": 0, "entries": []}

    # Batched into 3 queries total (regardless of page size) rather than
    # ~3 round trips PER match in a loop (one match fetch, one participants
    # fetch, one username lookup EACH) -- that N+1 pattern was the main
    # cause of slow history loads.
    matches_by_id = {
        m.id: m
        for m in (
            await db.execute(select(BattleMatch).where(BattleMatch.id.in_(match_ids)))
        ).scalars().all()
    }
    participants_by_match: dict[int, list[BattleParticipant]] = {}
    for p in (
        await db.execute(select(BattleParticipant).where(BattleParticipant.match_id.in_(match_ids)))
    ).scalars().all():
        participants_by_match.setdefault(p.match_id, []).append(p)

    opponent_ids = set()
    for participants in participants_by_match.values():
        opponent_ids.update(p.user_id for p in participants if p.user_id != user_id)
    usernames_by_id = dict(
        (
            await db.execute(select(User.id, User.username).where(User.id.in_(opponent_ids)))
        ).all()
    ) if opponent_ids else {}

    entries = []
    for match_id in match_ids:
        match = matches_by_id[match_id]
        participants = participants_by_match.get(match_id, [])
        me = next(p for p in participants if p.user_id == user_id)
        opponent = next((p for p in participants if p.user_id != user_id), None)

        entries.append(
            {
                "match_id": match.id, "subject": match.subject, "difficulty": match.difficulty,
                "started_at": match.started_at, "finished_at": match.finished_at,
                "winner_user_id": match.winner_user_id,
                "opponent_user_id": opponent.user_id if opponent is not None else None,
                "opponent_username": usernames_by_id.get(opponent.user_id) if opponent is not None else None,
                "me": participant_result_payload(me),
                "opponent": (
                    participant_result_payload(opponent, username=usernames_by_id.get(opponent.user_id))
                    if opponent is not None else None
                ),
            }
        )

    return {
        "page": page, "page_size": page_size, "total_count": total_count,
        "total_pages": math.ceil(total_count / page_size) if page_size else 0,
        "entries": entries,
    }


def next_question_payload(match: BattleMatch, participant: BattleParticipant, question: dict | None) -> dict:
    """What the client needs to show the player's current question --
    shared by the active state, the WS next_question event and the REST
    answer/skip responses so they can't drift apart.
    """
    index = participant.current_question_index
    return {
        "question_index": index,
        "question": question,
        "finished": index >= settings.BATTLE_QUESTION_COUNT,
        "ends_at": match_ends_at(match),
        "server_time": datetime.now(UTC),
    }


async def current_question_for(db: AsyncSession, match: BattleMatch, participant: BattleParticipant) -> dict:
    index = participant.current_question_index
    question = await question_payload_for(db, match, index) if index < settings.BATTLE_QUESTION_COUNT else None
    return next_question_payload(match, participant, question)


async def _build_active_state(
    db: AsyncSession, *, match: BattleMatch, me: BattleParticipant, opponent: BattleParticipant | None,
) -> dict:
    current = await current_question_for(db, match, me)
    remaining = (current["ends_at"] - current["server_time"]).total_seconds()
    return {
        "status": "active", "match_id": match.id, "subject": match.subject, "difficulty": match.difficulty,
        "question_count": settings.BATTLE_QUESTION_COUNT,
        "time_remaining_seconds": round(max(0.0, remaining), 2),
        **current,
        "my_answers": await _progress_answers(db, match_id=match.id, user_id=me.user_id),
        "opponent_answers": (
            await _progress_answers(db, match_id=match.id, user_id=opponent.user_id)
            if opponent is not None else []
        ),
        "my_progress": {"user_id": me.user_id, "answered_count": me.current_question_index},
        "opponent_progress": (
            {"user_id": opponent.user_id, "answered_count": opponent.current_question_index}
            if opponent is not None else None
        ),
        "started_at": match.started_at,
    }


async def _build_completed_state(
    db: AsyncSession, match: BattleMatch, *, me: BattleParticipant, opponent: BattleParticipant | None,
) -> dict:
    opponent_username = (
        await db.scalar(select(User.username).where(User.id == opponent.user_id))
        if opponent is not None else None
    )
    return {
        "status": "completed", "match_id": match.id, "subject": match.subject, "difficulty": match.difficulty,
        "question_count": settings.BATTLE_QUESTION_COUNT,
        "started_at": match.started_at, "finished_at": match.finished_at,
        "result": {
            "winner_user_id": match.winner_user_id,
            "end_reason": await match_end_reason(db, match.id),
            "me": participant_result_payload(me),
            "opponent": (
                participant_result_payload(opponent, username=opponent_username)
                if opponent is not None else None
            ),
        },
        "review": await _question_review(db, match=match, user_id=me.user_id),
    }


async def get_match_state(
    db: AsyncSession, *, match_id: int, user_id: int, redis: Redis | None = None
) -> dict:
    match = await _get_match_or_404(db, match_id)
    participants = await _get_participants(db, match_id)
    me = _require_participant(participants, user_id)
    opponent = next((p for p in participants if p.user_id != user_id), None)

    match = await _ensure_match_progressed(db, match, redis)

    if match.status == "countdown":
        remaining = max(0.0, (as_utc(match.started_at) - datetime.now(UTC)).total_seconds())
        return {
            "status": "countdown", "match_id": match.id, "subject": match.subject,
            "difficulty": match.difficulty, "started_at": match.started_at,
            "countdown_seconds_remaining": round(remaining, 2),
        }
    if match.status == "active":
        return await _build_active_state(db, match=match, me=me, opponent=opponent)
    if match.status == "completed":
        return await _build_completed_state(db, match, me=me, opponent=opponent)
    if match.status == "cancelled":
        return {"status": "cancelled", "match_id": match.id, "reason": match.cancel_reason}
    # "waiting" -- genuinely expected now (nobody's readied up yet), not a
    # narrow race window like it was when Phase 3 auto-activated on poll.
    return {"status": "waiting", "match_id": match.id}


# ── POST .../answer + .../skip (and the WS "answer"/"skip" messages) ──────

async def _streak_before(db: AsyncSession, *, match_id: int, user_id: int, order: int) -> int:
    """How many questions in a row this player got right straight before the
    one at `order` -- questions are answered strictly in order (no going
    back), so it's the run of correct answers counting back from order - 1;
    a wrong or skipped answer ends it.
    """
    rows = (
        await db.execute(
            select(BattleAnswer.is_correct)
            .join(
                BattleMatchQuestion,
                (BattleMatchQuestion.match_id == BattleAnswer.match_id)
                & (BattleMatchQuestion.question_id == BattleAnswer.question_id),
            )
            .where(
                BattleAnswer.match_id == match_id, BattleAnswer.user_id == user_id,
                BattleMatchQuestion.question_order < order,
            )
            .order_by(BattleMatchQuestion.question_order.desc())
            .limit(settings.BATTLE_STREAK_BONUS_TIER4_MIN_STREAK)
        )
    ).scalars().all()
    streak = 0
    for is_correct in rows:
        if not is_correct:
            break
        streak += 1
    return streak


async def _best_streak_for(db: AsyncSession, *, match_id: int, user_id: int) -> int:
    """Longest run of consecutive correct answers across the WHOLE frozen
    question set in question_order -- a skipped, wrong or never-reached
    question breaks the streak (outerjoin, so a never-reached question shows
    up as NULL).
    """
    rows = (
        await db.execute(
            select(BattleAnswer.is_correct)
            .select_from(BattleMatchQuestion)
            .outerjoin(
                BattleAnswer,
                (BattleAnswer.match_id == BattleMatchQuestion.match_id)
                & (BattleAnswer.question_id == BattleMatchQuestion.question_id)
                & (BattleAnswer.user_id == user_id),
            )
            .where(BattleMatchQuestion.match_id == match_id)
            .order_by(BattleMatchQuestion.question_order.asc())
        )
    ).scalars().all()
    best = current = 0
    for is_correct in rows:
        if is_correct:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


@dataclass
class AnswerOutcome:
    # What the answering player should see next (next_question_payload's
    # shape) -- the same thing again, unchanged, for a duplicate Next/Skip.
    next: dict
    # The match row after this call -- "completed" if this answer was the
    # last one either player still owed, so callers can announce the result.
    match: BattleMatch
    # Set only when THIS call saved something -- the opponent's
    # player_progress update. None for a duplicate.
    progress: dict | None = None


async def _duplicate(db: AsyncSession, match: BattleMatch, participant: BattleParticipant) -> AnswerOutcome:
    # Refresh both: after a rollback every loaded object is expired, and an
    # expired attribute can't be lazy-loaded under asyncio.
    await db.refresh(match)
    await db.refresh(participant)
    return AnswerOutcome(next=await current_question_for(db, match, participant), match=match)


# Sent when a Next/Skip reaches (or finishes saving) after the match clock
# ran out -- the client shows the results with "Time's up".
TIME_UP_DETAIL = "Time's up — this answer wasn't saved."


class _SaveSteps:
    """Diagnostic for slow answer saves: times each step (mostly one DB or
    Redis round trip each) and, when the whole save took longer than
    BATTLE_SLOW_SAVE_LOG_SECONDS or didn't finish normally, logs the
    breakdown -- including the step it was still in when it was cut off.
    """

    def __init__(self) -> None:
        self._began = self._step_began = time.monotonic()
        self._step: str | None = None
        self._done: list[tuple[str, float]] = []

    def start(self, name: str) -> None:
        now = time.monotonic()
        if self._step is not None:
            self._done.append((self._step, now - self._step_began))
        self._step, self._step_began = name, now

    def log_if_slow(self, *, match_id: int, user_id: int, outcome: str) -> None:
        now = time.monotonic()
        total = now - self._began
        if outcome == "ok" and total < settings.BATTLE_SLOW_SAVE_LOG_SECONDS:
            return
        if outcome == "HTTPException" and total < settings.BATTLE_SLOW_SAVE_LOG_SECONDS:
            return  # an ordinary refusal (time's up, wrong question) -- not a problem
        steps = [f"{name} {seconds:.2f}s" for name, seconds in self._done]
        if self._step is not None:
            steps.append(f"{self._step} {now - self._step_began:.2f}s" + (" (unfinished)" if outcome != "ok" else ""))
        logger.warning(
            "Slow battle answer save: %.1fs, outcome=%s — match_id=%d, user_id=%d — %s",
            total, outcome, match_id, user_id, " | ".join(steps),
        )


async def submit_battle_answer(
    db: AsyncSession, *, match_id: int, user_id: int, question_id: int, selected_option: str | None,
    redis: Redis | None = None, on_saved: Callable[[], None] | None = None,
) -> AnswerOutcome:
    """Next (selected_option set) or Skip (selected_option None) for the
    player's CURRENT question. Grades and scores it, saves it, and only then
    moves the player on -- the returned `next` is their next question. The
    answer row and the move to the next question commit together, guarded
    by a compare-and-swap on current_question_index plus the BattleAnswer
    unique constraint, so spamming Next (or the same message arriving twice)
    saves exactly one answer and the repeats just get the current question
    back.

    Response time is measured on the server clock only: from when the
    server handed out this question (current_question_presented_at, or
    match.started_at for question 0) to now, minus this player's network
    allowance (network_allowance_seconds).
    """
    steps = _SaveSteps()
    outcome = "ok"
    try:
        return await _submit_battle_answer(
            db, match_id=match_id, user_id=user_id, question_id=question_id, selected_option=selected_option,
            redis=redis, on_saved=on_saved, steps=steps,
        )
    except BaseException as exc:  # incl. CancelledError -- a save cut off by the WS timeout
        outcome = type(exc).__name__
        raise
    finally:
        steps.log_if_slow(match_id=match_id, user_id=user_id, outcome=outcome)


async def _submit_battle_answer(
    db: AsyncSession, *, match_id: int, user_id: int, question_id: int, selected_option: str | None,
    redis: Redis | None, on_saved: Callable[[], None] | None, steps: "_SaveSteps",
) -> AnswerOutcome:
    steps.start("load match")
    match = await _get_match_or_404(db, match_id)
    steps.start("load participants")
    participants = await _get_participants(db, match_id)
    me = _require_participant(participants, user_id)  # 403 if not a participant

    steps.start("check match progress")
    match = await _ensure_match_progressed(db, match, redis)

    if match.status == "cancelled":
        raise HTTPException(status.HTTP_409_CONFLICT, "This battle was cancelled.")
    if match.status == "completed":
        ran_out = match.started_at is not None and datetime.now(UTC) >= answers_close_at(match)
        raise HTTPException(status.HTTP_409_CONFLICT, TIME_UP_DETAIL if ran_out else "This battle has already ended.")
    if match.status != "active":
        raise HTTPException(status.HTTP_409_CONFLICT, f"Unexpected match status: {match.status}")

    steps.start("reload participant")
    await db.refresh(me)
    order = me.current_question_index
    if order >= settings.BATTLE_QUESTION_COUNT:
        steps.start("duplicate: reload")
        return await _duplicate(db, match, me)  # already finished -- a repeat of the last Next/Skip

    steps.start("load question")
    row = (
        await db.execute(
            select(BattleMatchQuestion.question_id, Question)
            .join(Question, Question.id == BattleMatchQuestion.question_id)
            .where(BattleMatchQuestion.match_id == match_id, BattleMatchQuestion.question_order == order)
        )
    ).first()
    if row is None or row[0] != question_id:
        earlier_order = (
            await db.execute(
                select(BattleMatchQuestion.question_order).where(
                    BattleMatchQuestion.match_id == match_id, BattleMatchQuestion.question_id == question_id
                )
            )
        ).scalar_one_or_none()
        if earlier_order is not None and earlier_order < order:
            # A repeat for a question already saved (double tap, resend after
            # a lost reply) -- not an error, just send the current question.
            steps.start("duplicate: reload")
            return await _duplicate(db, match, me)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This is not your current question.")
    question = row[1]

    # This player's network allowance: both their own 0:00 cutoff and the
    # speed-bonus timer are adjusted by it (see player_answers_close_at).
    steps.start("network allowance (Redis)")
    allowance = await network_allowance_seconds(redis, match_id=match_id, user_id=user_id)
    closes_at = player_answers_close_at(match, allowance)

    # Lock the match row for the rest of the save: _finalize_match claims the
    # match with an UPDATE on this same row, so it waits for this save to
    # commit or roll back -- an answer is either in the final score or
    # refused, never saved after the result was already worked out.
    steps.start("lock match")
    locked_status = (
        await db.execute(select(BattleMatch.status).where(BattleMatch.id == match_id).with_for_update())
    ).scalar_one()
    if locked_status != "active" or datetime.now(UTC) >= closes_at:
        await db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, TIME_UP_DETAIL)

    now = datetime.now(UTC)
    presented_at = as_utc(me.current_question_presented_at or match.started_at)
    response_time_ms = max(0, int(((now - presented_at).total_seconds() - allowance) * 1000))

    answered = selected_option is not None
    is_correct = bool(
        answered and question.correct_answer is not None
        and str(question.correct_answer).strip().lower() == str(selected_option).strip().lower()
    )  # exact grading rule mirrored from quiz_service.submit_quiz()
    streak_after = 0
    if is_correct:
        steps.start("streak so far")
        streak_after = await _streak_before(db, match_id=match_id, user_id=user_id, order=order) + 1
    score = compute_question_score(
        is_correct=is_correct, response_time_seconds=response_time_ms / 1000.0, streak_after_this_answer=streak_after,
    )

    next_order = order + 1
    finished = next_order >= settings.BATTLE_QUESTION_COUNT
    steps.start("advance participant")
    claimed = (
        await db.execute(
            update(BattleParticipant)
            .where(BattleParticipant.id == me.id, BattleParticipant.current_question_index == order)
            .values(
                current_question_index=next_order,
                current_question_presented_at=now,
                finished_at=now if finished else None,
            )
            .returning(BattleParticipant.id)
            .execution_options(synchronize_session=False)
        )
    ).scalar_one_or_none()
    if claimed is None:
        steps.start("duplicate: reload")
        await db.rollback()  # another request for this same question won
        return await _duplicate(db, match, me)
    db.add(BattleAnswer(
        match_id=match_id, question_id=question_id, user_id=user_id,
        # "" marks a skip -- the column is NOT NULL, and every reader already
        # treats a falsy selected_option as "not answered".
        selected_option=selected_option or "",
        is_correct=score.is_correct, response_time_ms=response_time_ms, base_score=score.base_score,
        speed_bonus=score.speed_bonus, streak_bonus=score.streak_bonus, total_question_score=score.total_score,
    ))
    if datetime.now(UTC) >= closes_at:
        # The clock ran out while this was being saved -- refuse it.
        await db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, TIME_UP_DETAIL)
    steps.start("insert answer + commit")
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        return await _duplicate(db, match, me)
    if on_saved is not None:
        on_saved()  # the answer is durable -- e.g. lift the WS save timeout
    steps.start("reload participant (saved)")
    await db.refresh(me)

    if finished:
        steps.start("finalize match")
        match = await _ensure_match_progressed(db, match, redis)  # both done -> finalize now
    steps.start("load next question")
    return AnswerOutcome(
        next=await current_question_for(db, match, me),
        match=match,
        progress={"user_id": user_id, "question_order": order, "answered": answered, "answered_count": next_order},
    )


# ── stale-match reaper (Phase 6) ──────────────────────────────────────────
# Closes the crash-window limitation documented in matchmaking_service.py's
# module docstring: if a process dies after committing a BattleMatch row but
# before finalizing both users' Redis state to "in_match", the match is
# stuck at status="waiting" forever with no Redis pointer and a client that
# never received its match_id -- both participants' user_state also never
# gets released (they self-heal to idle only via the "matching" claim TTL,
# which already expired well before this reaper's own, more generous
# threshold).
#
# It also covers a second, unrelated gap: a match that DID reach
# "countdown"/"active" but was then abandoned by both players (e.g. both
# apps closed right after the match started, nobody ever reconnects).
# _ensure_match_progressed -- the function that advances/completes a match
# -- only ever runs when triggered by a REST call or a live WS connection;
# with nobody connected, nothing ever calls it again, so the match (and both
# players' Redis "in_match" pointer, which has no TTL by design) would
# otherwise be stuck forever. BattleMatch has no updated_at/last-activity
# column, so started_at is the closest available signal -- acceptable here
# since the cutoff is set far past any real match's duration.
#
# Both branches are cancelled with no rating changes (same as a "waiting"
# match: nothing meaningful to score from an abandoned/never-started game).
#
# A third, much shorter-fused case lives here too: a "waiting" match where
# BOTH players have marked ready (mark_ready) but activation is still
# withheld because mark_connected never confirmed a live WS for one side
# within BATTLE_CONNECTION_READY_TIMEOUT_SECONDS (see that setting's own
# docstring and _try_activate_if_ready_and_connected) -- force-started
# rather than cancelled, since both players are actively trying to play,
# not abandoning the match. This is also what makes a REST-only client (one
# that never opens a socket at all, so mark_connected never fires for it --
# still a supported mode, see battle_ws.py's module docstring) able to ever
# reach "countdown": nothing else would ever finalize its activation.
#
# run_stale_match_reap_once is a single deterministic pass (directly
# testable, no sleeps); start_stale_match_reaper_loop is the infinite-loop
# wrapper only ever started for real at app startup -- mirrors
# matchmaking_service.start_matchmaking_sweep_loop's exact shape.

_STALE_ACTIVE_STATUSES = ("countdown", "active")


async def _force_activate_stalled_ready_matches(db: AsyncSession, redis: Redis, *, skip_ids: set[int]) -> list[int]:
    waiting_ids = (
        await db.execute(select(BattleMatch.id).where(BattleMatch.status == "waiting"))
    ).scalars().all()

    forced_ids: list[int] = []
    cutoff_seconds = settings.BATTLE_CONNECTION_READY_TIMEOUT_SECONDS
    for match_id in waiting_ids:
        if match_id in skip_ids:
            continue
        both_ready_at_raw = await redis.get(_both_ready_at_key(match_id))
        if both_ready_at_raw is None:
            continue
        both_ready_at = datetime.fromisoformat(both_ready_at_raw)
        if (datetime.now(UTC) - as_utc(both_ready_at)).total_seconds() < cutoff_seconds:
            continue

        if await redis.exists(_activating_key(match_id)):
            continue  # being started right now (questions still being picked) -- not stalled
        match = await get_match_row(db, match_id)
        if match is None or match.status != "waiting":
            continue  # already handled (activated/cancelled) by another process/tick
        match, started_it = await _finalize_activation(db, redis, match=match)
        if started_it:  # otherwise the normal path had already started it -- nothing was forced
            forced_ids.append(match_id)

    if forced_ids:
        logger.warning(
            "Battle match(es) %s force-activated -- both players ready but a WS "
            "connection never confirmed within %.1fs.", forced_ids, cutoff_seconds,
        )
    return forced_ids


async def _settle_finished_unattended_matches(db: AsyncSession, redis: Redis) -> tuple[list[int], list[int]]:
    """Active matches whose time budget (+ buffer) has run out with nobody
    connected to trigger completion. Either finalizes normally (someone
    answered or skipped at least one question, so there's a real result to
    award) or cancels (nobody played at all). Returns (settled_ids,
    abandoned_ids).
    """
    budget = (
        settings.BATTLE_MATCH_DURATION_SECONDS + settings.BATTLE_ANSWER_LATE_GRACE_SECONDS
        + settings.BATTLE_NETWORK_ALLOWANCE_MAX_SECONDS
    )
    cutoff = datetime.now(UTC) - timedelta(seconds=budget + settings.BATTLE_FINISHED_MATCH_SETTLE_BUFFER_SECONDS)
    matches = (
        await db.execute(select(BattleMatch).where(BattleMatch.status == "active", BattleMatch.started_at < cutoff))
    ).scalars().all()

    settled: list[int] = []
    abandoned: list[int] = []
    for match in matches:
        anyone_answered = (
            await db.execute(
                select(func.count()).select_from(BattleAnswer).where(BattleAnswer.match_id == match.id)
            )
        ).scalar_one() > 0
        if anyone_answered:
            finished = await _finalize_match(db, match, redis)
            if finished.status == "completed":
                settled.append(match.id)
                logger.info("Battle match %d settled by the reaper — nobody was connected at the end.", match.id)
        else:
            abandoned.append(match.id)
    return settled, abandoned


async def run_stale_match_reap_once(db: AsyncSession, redis: Redis) -> int:
    settled_ids, abandoned_ids = await _settle_finished_unattended_matches(db, redis)

    waiting_cutoff = datetime.now(UTC) - timedelta(seconds=settings.BATTLE_ORPHANED_MATCH_REAP_AFTER_SECONDS)
    waiting_ids = (
        await db.execute(
            select(BattleMatch.id).where(BattleMatch.status == "waiting", BattleMatch.created_at < waiting_cutoff)
        )
    ).scalars().all()

    active_cutoff = datetime.now(UTC) - timedelta(seconds=settings.BATTLE_ACTIVE_MATCH_REAP_AFTER_SECONDS)
    active_ids = [
        *abandoned_ids,
        *(
            await db.execute(
                select(BattleMatch.id).where(
                    BattleMatch.status.in_(_STALE_ACTIVE_STATUSES),
                    BattleMatch.started_at < active_cutoff,
                    BattleMatch.id.notin_(abandoned_ids),
                )
            )
        ).scalars().all(),
    ]

    stale_ids = [*waiting_ids, *active_ids]
    if stale_ids:
        if waiting_ids:
            await db.execute(
                update(BattleMatch)
                .where(BattleMatch.id.in_(waiting_ids))
                .values(status="cancelled", cancel_reason="Orphaned before matchmaking finalized.")
            )
        if active_ids:
            await db.execute(
                update(BattleMatch)
                .where(BattleMatch.id.in_(active_ids))
                .values(status="cancelled", cancel_reason="Abandoned -- no activity from either player.")
            )
        await db.commit()

        for match_id in stale_ids:
            participants = await _get_participants(db, match_id)
            for p in participants:
                await release_match_state(redis, user_id=p.user_id)
            logger.warning("Battle match %d reaped -- stale and abandoned, no rating changes applied.", match_id)

    forced_ids = await _force_activate_stalled_ready_matches(db, redis, skip_ids=set(waiting_ids))
    return len(settled_ids) + len(stale_ids) + len(forced_ids)


async def start_stale_match_reaper_loop() -> None:
    # Own session/exception-swallowing per tick, same as
    # matchmaking_service.start_matchmaking_sweep_loop -- one bad tick must
    # never kill the loop.
    while True:
        try:
            async with AsyncSessionLocal() as session:
                await run_stale_match_reap_once(session, redis_client)
        except Exception:
            logger.exception("Stale match reaper tick failed")
        await asyncio.sleep(settings.BATTLE_STALE_MATCH_REAPER_INTERVAL_SECONDS)

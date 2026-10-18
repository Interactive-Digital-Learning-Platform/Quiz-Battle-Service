"""
app/api/routes/battle_ws.py
──────────────────────────────
Live WebSocket layer over the Phase 3 REST battle engine:
`WS /api/v1/battle/match/{match_id}/ws`. A client connects here only after
already having a `match_id` from the existing REST matchmaking flow
(app/api/routes/battle.py) -- this route is scoped to one match, not the
matchmaking queue.

The REST endpoints (`GET .../state`, `POST .../answer`, and the new
`POST .../ready` / `POST .../forfeit` in battle.py) keep working standalone
for a client that never opens a socket at all -- this route is additive
delivery on top of the same `battle_gameplay_service` state machine, not a
replacement for it.

`Depends(get_db)` is deliberately NOT used here: its teardown for a
WebSocket route doesn't run until the whole connection closes, which would
hold one pooled DB connection open per live socket for the connection's
entire (multi-minute) lifetime. Every DB touch instead opens its own
short-lived `async with AsyncSessionLocal() as db:` block, mirroring
matchmaking_service.start_matchmaking_sweep_loop's per-tick pattern.
`redis` IS obtained via `Depends(get_redis)` (not the raw `redis_client`
singleton) so `app.dependency_overrides[get_redis]` also applies to
TestClient-based WS tests, and is threaded explicitly into every spawned
background task rather than any of them re-importing the global singleton.
"""
import asyncio
import contextlib
import json
import logging
import time
from collections import deque
from datetime import UTC, datetime

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
    status,
)
from pydantic import ValidationError
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.redis_client import get_redis
from app.core.security import get_current_user_ws
from app.schemas.battle import BattleMatchStateResponse
from app.schemas.battle_ws import (
    ConnectedEvent,
    ErrorEvent,
    MatchStartedEvent,
    NextQuestionEvent,
    PingEvent,
    PlayerDisconnectedEvent,
    PlayerProgressEvent,
    PlayerReconnectedEvent,
    StateSyncEvent,
    WSAnswerMessage,
    WSForfeitMessage,
    WSPongMessage,
    WSReadyMessage,
    WSSkipMessage,
    parse_inbound_message,
)
from app.services import battle_gameplay_service, battle_ws_events
from app.services.battle_connection_manager import manager as ws_manager
from app.services.rate_limiter import allow as rate_limit_allow
from app.services.user_service import get_or_create_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/battle", tags=["Battle Realtime"])

_TERMINAL_STATUSES = ("completed", "cancelled")

# (match_id, disconnected_user_id) -> the task counting down their grace
# period. A later `player_disconnected` for the same user (a genuine new
# drop, or a same-deadline redelivery) always cancels-and-replaces whatever
# is here rather than being dropped -- see _make_on_event below for why a
# plain "already pending, ignore" dedup let a disconnect -> reconnect ->
# disconnect flap forfeit a player who was safely inside their NEW grace
# window.
_pending_grace_tasks: dict[tuple[int, int], asyncio.Task] = {}


async def _resolve_authorized_user(
    websocket: WebSocket, match_id: int, current_user: dict, redis: Redis
) -> int:
    """Pre-accept check: the token must resolve to a real participant of
    this match. Done with a short-lived session (not Depends(get_db)) so it
    can run BEFORE `websocket.accept()` -- raising WebSocketException here
    correctly rejects the handshake instead of accepting then closing.
    """
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION, reason="Token is missing the 'sub' claim."
        )

    try:
        within_limit = await rate_limit_allow(
            redis, key=f"battle:ratelimit:ws_connect:{clerk_id}",
            max_calls=settings.BATTLE_WS_CONNECT_RATE_LIMIT_MAX_CALLS,
            window_seconds=settings.BATTLE_WS_CONNECT_RATE_LIMIT_WINDOW_SECONDS,
        )
    except RedisError:
        logger.exception("Redis error while rate-limiting WS connect — clerk_id=%s", clerk_id)
        raise WebSocketException(
            code=status.WS_1011_INTERNAL_ERROR, reason="Realtime service is temporarily unavailable."
        )
    if not within_limit:
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION, reason="Too many connection attempts. Please wait and retry."
        )

    async with AsyncSessionLocal() as db:
        user = await get_or_create_user(db, clerk_id)
        match = await battle_gameplay_service.get_match_row(db, match_id)
        if match is None:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION, reason="Battle match not found."
            )
        participants = await battle_gameplay_service._get_participants(db, match_id)
        if not any(p.user_id == user.id for p in participants):
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION, reason="You are not a participant in this match."
            )
        return user.id


async def _mark_connected_safely(*, match_id: int, user_id: int, redis: Redis) -> None:
    # Fire-and-forget wrapper around mark_connected -- see its call site in
    # battle_match_ws for why this runs detached rather than awaited inline.
    # Best-effort, same rationale as bump_connection_generation/
    # clear_disconnect_marker in the connect handler: never fail an
    # already-accepted socket over this. Worst case, activation for a match
    # still "waiting" on both ready+connected is delayed until the
    # stale-match reaper's BATTLE_CONNECTION_READY_TIMEOUT_SECONDS fallback
    # forces it.
    try:
        async with AsyncSessionLocal() as db:
            await battle_gameplay_service.mark_connected(db, redis, match_id=match_id, user_id=user_id)
    except Exception:
        logger.exception(
            "Failed to mark WS connection ready for activation — match_id=%d, user_id=%d", match_id, user_id
        )


async def _grace_check_task(*, match_id: int, user_id: int, deadline: datetime, redis: Redis) -> None:
    try:
        delay = max(0.0, (deadline - datetime.now(UTC)).total_seconds())
        await asyncio.sleep(delay)

        if not await battle_gameplay_service.is_still_disconnected(redis, match_id=match_id, user_id=user_id):
            return  # reconnected before the deadline -- no-op

        event: dict | None = None
        async with AsyncSessionLocal() as db:
            if await battle_gameplay_service.has_finished(db, match_id=match_id, user_id=user_id):
                return  # finished (e.g. via REST) while away -- nothing to forfeit
            match = await battle_gameplay_service.forfeit_match(
                db, redis, match_id=match_id, forfeiting_user_id=user_id
            )
            if match.status == "completed":
                event = await battle_ws_events.match_finished_event(db, match)
        if event is not None:
            await ws_manager.publish(redis, match_id=match_id, event=event)
    except asyncio.CancelledError:
        raise
    finally:
        # Only remove this task's OWN entry -- if a later disconnect already
        # cancelled-and-replaced it (see _make_on_event), the dict now holds
        # that newer task under the same key, and a blind pop-by-key here
        # (racing this cancelled task's teardown against that replacement)
        # would delete the newer task's tracking entry instead, even though
        # the newer task itself keeps running unaffected.
        if _pending_grace_tasks.get((match_id, user_id)) is asyncio.current_task():
            _pending_grace_tasks.pop((match_id, user_id), None)


def _make_on_event(*, match_id: int, redis: Redis):
    async def _on_event(event: dict) -> None:
        # Cross-process bridge: this fires once per locally-connected socket
        # for the match, on EVERY process subscribed to it -- including the
        # process the disconnecting user themself was on. We only want the
        # OPPONENT's process to start the grace countdown, so this callback
        # is a no-op unless it's watching for someone else's disconnect.
        if event.get("type") != "player_disconnected":
            return
        disconnected_user_id = event.get("user_id")
        key = (match_id, disconnected_user_id)
        try:
            deadline = datetime.fromisoformat(event["deadline"])
        except (KeyError, ValueError):
            return
        # Cancel any task already tracking an earlier deadline for this user
        # instead of ignoring this event -- a disconnect -> reconnect ->
        # disconnect flap rewrites the Redis deadline marker to this new,
        # later value (mark_disconnected), but is_still_disconnected only
        # checks whether the marker exists, not which deadline it encodes.
        # Leaving the ORIGINAL task running meant it woke at the FIRST
        # (stale) deadline, found the marker still there (now the SECOND
        # disconnect's), and forfeited a player who was still safely inside
        # their new grace window.
        existing = _pending_grace_tasks.get(key)
        if existing is not None:
            existing.cancel()
        _pending_grace_tasks[key] = asyncio.create_task(
            _grace_check_task(match_id=match_id, user_id=disconnected_user_id, deadline=deadline, redis=redis)
        )

    return _on_event


async def _ping_loop(websocket: WebSocket, sent_at: dict[int, float]) -> None:
    """Pings this socket so its pongs measure the player's round trip (see
    BATTLE_NETWORK_ALLOWANCE_MAX_SECONDS). Only sends -- the pong is matched
    up in the connection's receive loop, by nonce, against `sent_at`.
    Cancelled when the connection ends; it never touches the DB, so that's
    safe here (unlike the progression timer).
    """
    nonce = 0
    await asyncio.sleep(settings.BATTLE_RTT_FIRST_PING_DELAY_SECONDS)
    while True:
        nonce += 1
        sent_at[nonce] = time.monotonic()
        # Unanswered pings shouldn't pile up forever.
        for stale in [n for n in sent_at if n <= nonce - settings.BATTLE_RTT_SAMPLE_COUNT]:
            sent_at.pop(stale, None)
        try:
            await websocket.send_json(PingEvent(nonce=nonce).model_dump(mode="json"))
        except (RuntimeError, WebSocketDisconnect, OSError):
            return  # socket already closing -- the receive loop handles the teardown
        await asyncio.sleep(settings.BATTLE_RTT_PING_INTERVAL_SECONDS)


async def _send_answer_outcome(
    websocket: WebSocket, db, redis: Redis, *, match_id: int, outcome: battle_gameplay_service.AnswerOutcome,
) -> None:
    """The answering player's next question goes straight back on their own
    socket (it's theirs alone); the opponent hears about it via a published
    player_progress; a match this answer just finished is announced to both.
    """
    await websocket.send_json(NextQuestionEvent(**outcome.next).model_dump(mode="json"))
    if outcome.progress is None:
        return
    # The answer is already saved and this player already has their next
    # question -- a failed announcement must not cost them their connection.
    # The opponent's next player_progress carries the full answered_count,
    # and the progression timer announces a finished match on its own next
    # tick (at most BATTLE_WS_ACTIVE_POLL_INTERVAL_SECONDS later).
    try:
        await ws_manager.publish(
            redis, match_id=match_id, event=PlayerProgressEvent(**outcome.progress).model_dump(mode="json"),
        )
        if outcome.match.status == "completed":
            event = await battle_ws_events.match_finished_event(db, outcome.match)
            await ws_manager.publish(redis, match_id=match_id, event=event)
    except Exception:
        logger.exception("Announcing a saved battle answer failed — match_id=%d", match_id)


async def _handle_move(
    websocket: WebSocket, redis: Redis, *, match_id: int, user_id: int, message: WSAnswerMessage | WSSkipMessage,
) -> None:
    """Next/Skip over the WebSocket. Any failure to save -- a DB/Redis error,
    a dropped or stalled Neon connection (which can surface as a raw
    OSError/TimeoutError rather than a SQLAlchemyError), or a save slower
    than BATTLE_ANSWER_SAVE_TIMEOUT_SECONDS -- is reported to the player,
    never allowed to close their socket: the client keeps the move pending
    and resends it, and a resend of an answer that did get saved just
    returns the current question.
    """
    saved = False

    def on_saved() -> None:
        nonlocal saved
        saved = True
        save_deadline.reschedule(None)

    async with AsyncSessionLocal() as db:
        try:
            # Covers the save only: once the answer is committed the deadline
            # is lifted, so a slow read-back afterwards can't report an answer
            # that WAS saved as "Couldn't save".
            async with asyncio.timeout(settings.BATTLE_ANSWER_SAVE_TIMEOUT_SECONDS) as save_deadline:
                outcome = await battle_gameplay_service.submit_battle_answer(
                    db, match_id=match_id, user_id=user_id, question_id=message.question_id,
                    selected_option=message.selected_option if isinstance(message, WSAnswerMessage) else None,
                    redis=redis, on_saved=on_saved,
                )
        except HTTPException as exc:
            await websocket.send_json(ErrorEvent(detail=str(exc.detail)).model_dump(mode="json"))
            return
        except Exception:
            if saved:
                # The answer IS saved -- only the work after the commit (e.g.
                # ending the match) failed. Don't tell the player otherwise:
                # resync them onto where they really are instead (their next
                # question, or the result); the progression timer finishes
                # ending the match on its next tick if that's what failed.
                logger.exception(
                    "Battle answer saved but its follow-up failed — match_id=%d, user_id=%d", match_id, user_id
                )
                await _send_state_sync(websocket, redis, match_id=match_id, user_id=user_id)
                return
            logger.exception("Saving a battle answer failed — match_id=%d, user_id=%d", match_id, user_id)
            await websocket.send_json(
                ErrorEvent(detail="Couldn't save your answer — retrying.").model_dump(mode="json")
            )
            return
        await _send_answer_outcome(websocket, db, redis, match_id=match_id, outcome=outcome)


async def _send_state_sync(websocket: WebSocket, redis: Redis, *, match_id: int, user_id: int) -> None:
    async with AsyncSessionLocal() as db:
        state = await battle_gameplay_service.get_match_state(db, match_id=match_id, user_id=user_id, redis=redis)
    await websocket.send_json(StateSyncEvent(state=BattleMatchStateResponse(**state)).model_dump(mode="json"))


async def _sleep_or_stop(stop_event: asyncio.Event, seconds: float) -> bool:
    """True if stop was requested during the wait, False if the timeout
    elapsed first. Used instead of raw task cancellation for shutting this
    loop down (see _progression_timer's docstring for why).
    """
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
        return True
    except TimeoutError:
        return False


async def _progression_timer(*, match_id: int, redis: Redis, stop_event: asyncio.Event) -> None:
    """Per-match-per-process, time-driven local push loop -- owned by
    ws_manager (started on first local connection for the match, stopped on
    the last, see battle_connection_manager.py), NOT spawned once per
    connection: two independent per-connection copies would each detect the
    same state transition and call push_local, which broadcasts to every
    local socket, so every client would receive each event duplicated once
    per locally-connected socket. Every process with a locally-connected
    socket for this match runs its OWN one copy of this loop and
    independently reaches the same conclusion at the same wall-clock instant
    (question progression / match completion are pure functions of
    already-shared Postgres state) -- no Redis publish involved here, only
    push_local.

    Shut down cooperatively via `stop_event`, NOT `task.cancel()`: forcibly
    cancelling a task while it's suspended mid-query inside SQLAlchemy's
    async-over-greenlet bridge can deadlock the underlying greenlet switch
    (verified directly -- a `task.cancel()` landing during an in-flight
    `AsyncSessionLocal()` query left the connection permanently stuck).
    Checking `stop_event` between DB touches means a shutdown request is
    only ever observed at a point where no greenlet switch is in flight.

    Self-paced play: on the countdown->active transition this loop pushes
    `match_started` plus question 0 (`next_question`) to every local socket
    -- everyone starts on question 0 at match.started_at. After that each
    player's next question is sent by the answer/skip handler, only to them,
    so all this loop does while "active" is wait (re-checking every
    BATTLE_WS_ACTIVE_POLL_INTERVAL_SECONDS) for the match clock plus the
    late-answer grace to run out, when _ensure_match_progressed finalizes
    it. A match both players finish early is finalized by the last answer
    itself, which publishes match_finished.

    The stop check below is deliberately placed AFTER
    _ensure_match_progressed, not as the `while` loop's own condition: both
    players' connections can close (stop_event.set()) at the exact instant
    the final question's window closes, waking `_sleep_or_stop` via the stop
    signal rather than its timeout. Exiting right there, before ever
    re-running _ensure_match_progressed, used to skip the one check that
    would have finalized an already-time-expired match -- leaving it stuck
    "active" in Postgres and "in_match" in Redis (no TTL) until the
    stale-match reaper's much longer BATTLE_ACTIVE_MATCH_REAP_AFTER_SECONDS
    cutoff (10 minutes) caught it. Looping back for one more pass first
    means a match whose time already ran out gets finalized regardless of
    whether anyone is still connected to see it happen.

    `previous_status` doubles as "has this task ever observed the match
    non-terminal": a brand new task (e.g. spawned because a client just
    reconnected to a match that finished long ago) whose very FIRST check
    already finds "completed"/"cancelled" never actually witnessed a
    transition -- it's simply discovering old news. Pushing match_finished/
    match_cancelled in that case made every such reconnect replay the
    original completion's client-side side effects (forfeit/win toast,
    navigate-to-results) a second time. StateSyncEvent, sent separately
    right after connect, already tells a fresh connection the outcome
    without re-triggering those.
    """
    previous_status: str | None = None
    # No top-level try/except here anymore -- an unhandled exception now
    # propagates straight out of this coroutine on purpose, so
    # battle_connection_manager.py's supervisor (which wraps every call to
    # this function) can detect it and restart the loop. Swallowing it here
    # used to mean one transient DB/Redis error silently ended this task for
    # good, freezing that match's question progression forever for anyone
    # routed to this process, since it's only ever started once (on the
    # first local connection).
    while True:
        async with AsyncSessionLocal() as db:
            match = await battle_gameplay_service.get_match_row(db, match_id)
            if match is None:
                return
            match = await battle_gameplay_service._ensure_match_progressed(db, match, redis)
            current_status = match.status

            if current_status == "completed":
                # previous_status is only ever None on this task's very
                # first tick -- reaching "completed" without having first
                # observed the match non-terminal means it was ALREADY
                # over before this task (hence this connection) even
                # started, e.g. a client reconnecting to a long-finished
                # match. StateSyncEvent (sent separately, right after
                # connect) already tells this new connection the outcome
                # -- re-pushing match_finished here too would make the
                # client replay its "just now completed" side effects
                # (the forfeit/win toast, the navigate-to-results effect)
                # for a result it's only just learning about, not one
                # that just happened.
                if previous_status is not None:
                    event = await battle_ws_events.match_finished_event(db, match)
                    await ws_manager.push_local(match_id, event)
                return

            if current_status == "cancelled":
                if previous_status is not None:
                    await ws_manager.push_local(match_id, battle_ws_events.match_cancelled_event(match))
                return

            if stop_event.is_set():
                # Still active/countdown/waiting even after the check
                # above -- a genuine early exit (last connection closed
                # mid-match), not a missed finalize. Safe to stop.
                return

            if current_status == "active":
                # Only on a transition this task actually watched happen: a
                # task that first finds the match already active (a socket
                # reconnecting mid-match) must not hand everyone question 0
                # again -- that connection's state_sync carries its own
                # current question.
                if previous_status is not None and previous_status != "active":
                    ends_at = battle_gameplay_service.match_ends_at(match)
                    await ws_manager.push_local(
                        match_id,
                        MatchStartedEvent(
                            started_at=match.started_at,
                            subject=match.subject,
                            difficulty=match.difficulty,
                            question_count=settings.BATTLE_QUESTION_COUNT,
                            ends_at=ends_at,
                            server_time=datetime.now(UTC),
                        ).model_dump(mode="json"),
                    )
                    first_question = await battle_gameplay_service.question_payload_for(db, match, 0)
                    await ws_manager.push_local(
                        match_id,
                        NextQuestionEvent(
                            question_index=0, question=first_question, finished=False,
                            ends_at=ends_at, server_time=datetime.now(UTC),
                        ).model_dump(mode="json"),
                    )

                sleep_seconds = min(
                    settings.BATTLE_WS_ACTIVE_POLL_INTERVAL_SECONDS,
                    max(0.05, (battle_gameplay_service.answers_close_at(match) - datetime.now(UTC)).total_seconds()),
                )
            elif current_status == "countdown":
                remaining = max(
                    0.0,
                    (battle_gameplay_service.as_utc(match.started_at) - datetime.now(UTC)).total_seconds(),
                )
                sleep_seconds = min(remaining, settings.BATTLE_WS_WAITING_POLL_INTERVAL_SECONDS) or 0.05
            else:  # "waiting"
                sleep_seconds = settings.BATTLE_WS_WAITING_POLL_INTERVAL_SECONDS

            previous_status = current_status

        if await _sleep_or_stop(stop_event, sleep_seconds):
            # Stop requested mid-sleep -- loop back for exactly one more
            # _ensure_match_progressed pass (see docstring) before the
            # `stop_event.is_set()` check above actually honors it.
            continue


@router.websocket("/match/{match_id}/ws")
async def battle_match_ws(
    websocket: WebSocket,
    match_id: int,
    current_user: dict = Depends(get_current_user_ws),
    redis: Redis = Depends(get_redis),
) -> None:
    user_id = await _resolve_authorized_user(websocket, match_id, current_user, redis)

    await websocket.accept()

    on_event = _make_on_event(match_id=match_id, redis=redis)

    async def _start_progression_timer(mid: int, stop_event: asyncio.Event) -> None:
        await _progression_timer(match_id=mid, redis=redis, stop_event=stop_event)

    previous_socket = await ws_manager.connect(
        redis, match_id=match_id, user_id=user_id, websocket=websocket, on_event=on_event,
        start_progression_timer=_start_progression_timer,
    )
    if previous_socket is not None:
        with contextlib.suppress(Exception):
            await previous_socket.close(code=status.WS_1000_NORMAL_CLOSURE)

    connection_generation = 0
    ping_sent_at: dict[int, float] = {}
    ping_task: asyncio.Task | None = None
    try:
        try:
            connection_generation = await battle_gameplay_service.bump_connection_generation(
                redis, match_id=match_id, user_id=user_id
            )
        except RedisError:
            # Best-effort, same rationale as the clear_disconnect_marker guard
            # below: don't fail an already-accepted socket over this. Falling
            # back to 0 just means this connection's own eventual disconnect
            # notification (which checks generation == 0) will very likely lose
            # to any real reconnect's bump -- the safe failure direction.
            logger.exception(
                "Redis error bumping connection generation — match_id=%d, user_id=%d", match_id, user_id
            )
            connection_generation = 0

        try:
            had_marker = await battle_gameplay_service.clear_disconnect_marker(
                redis, match_id=match_id, user_id=user_id
            )
        except RedisError:
            # Best-effort: the socket is already accepted at this point, so a
            # Redis hiccup here shouldn't tear down an otherwise-good
            # connection -- just log and skip the (non-essential) reconnect
            # notification rather than failing the whole connect.
            logger.exception(
                "Redis error clearing disconnect marker — match_id=%d, user_id=%d", match_id, user_id
            )
            had_marker = False
        if had_marker:
            await ws_manager.publish(
                redis, match_id=match_id,
                event=PlayerReconnectedEvent(user_id=user_id).model_dump(mode="json"),
            )

        await websocket.send_json(ConnectedEvent(match_id=match_id).model_dump(mode="json"))
        async with AsyncSessionLocal() as db:
            state = await battle_gameplay_service.get_match_state(
                db, match_id=match_id, user_id=user_id, redis=redis
            )
        await websocket.send_json(
            StateSyncEvent(state=BattleMatchStateResponse(**state)).model_dump(mode="json")
        )

        # Detached, not awaited -- this connection's own state_sync above
        # doesn't depend on mark_connected's result. Deliberately placed
        # AFTER state_sync is sent (not right after ConnectedEvent, before
        # get_match_state): spawning it earlier -- even as a detached task,
        # even sharing get_match_state's own session so no extra DB
        # round-trip was added -- reproducibly wedged fakeredis's pub/sub
        # `listen()` loop against this same Redis instance in a two-socket
        # disconnect test (the message was published and even handed to
        # send_json successfully server-side, yet the other socket's
        # receive_json() never woke for it). Isolated directly to the mere
        # act of a `redis.hset` running concurrently with get_match_state's
        # own in-flight Redis/DB work during this window; moving it past
        # that window removed the interleaving entirely.
        asyncio.create_task(_mark_connected_safely(match_id=match_id, user_id=user_id, redis=redis))
        ping_task = asyncio.create_task(_ping_loop(websocket, ping_sent_at))

        recent_messages: deque[float] = deque()
        while True:
            raw_text = await websocket.receive_text()
            now = time.monotonic()
            while recent_messages and now - recent_messages[0] >= settings.BATTLE_WS_MESSAGE_RATE_LIMIT_WINDOW_SECONDS:
                recent_messages.popleft()
            if len(recent_messages) >= settings.BATTLE_WS_MESSAGE_RATE_LIMIT_MAX_CALLS:
                # Dropped before parsing or touching the DB.
                await websocket.send_json(
                    ErrorEvent(detail="Too many messages — slow down.").model_dump(mode="json")
                )
                continue
            recent_messages.append(now)
            try:
                message = parse_inbound_message(json.loads(raw_text))
            except (json.JSONDecodeError, ValidationError) as exc:
                await websocket.send_json(ErrorEvent(detail=f"Malformed message: {exc}").model_dump(mode="json"))
                continue

            if isinstance(message, WSPongMessage):
                sent = ping_sent_at.pop(message.nonce, None)
                if sent is not None:  # unknown nonce: not a ping of ours, ignore
                    try:
                        await battle_gameplay_service.record_rtt_sample(
                            redis, match_id=match_id, user_id=user_id, rtt_seconds=time.monotonic() - sent,
                        )
                    except RedisError:
                        logger.exception("Redis error recording RTT — match_id=%d, user_id=%d", match_id, user_id)
                continue

            is_move = isinstance(message, (WSAnswerMessage, WSSkipMessage))
            try:
                if is_move:
                    # Inside this try too: closing the save's DB session after
                    # a timed-out (cancelled) query can itself fail, and that
                    # must not end the connection either.
                    await _handle_move(websocket, redis, match_id=match_id, user_id=user_id, message=message)
                    continue
                async with AsyncSessionLocal() as db:
                    if isinstance(message, WSReadyMessage):
                        await battle_gameplay_service.mark_ready(db, redis, match_id=match_id, user_id=user_id)
                    elif isinstance(message, WSForfeitMessage):
                        match = await battle_gameplay_service.forfeit_match(
                            db, redis, match_id=match_id, forfeiting_user_id=user_id
                        )
                        if match.status == "completed":
                            event = await battle_ws_events.match_finished_event(db, match)
                            await ws_manager.publish(redis, match_id=match_id, event=event)
                        elif match.status == "cancelled":
                            # Forfeiting a still-"waiting" match (matched but not
                            # yet both-ready) voids it instead of scoring a win --
                            # queue.tsx auto-requeues the remaining player off
                            # this event rather than showing a forfeit-win screen
                            # for a match that never actually started.
                            cancel_event = battle_ws_events.match_cancelled_event(match)
                            await ws_manager.publish(redis, match_id=match_id, event=cancel_event)
            except HTTPException as exc:
                # e.g. 404 for a match deleted meanwhile -- report it, don't drop the socket.
                await websocket.send_json(ErrorEvent(detail=str(exc.detail)).model_dump(mode="json"))
            except (WebSocketDisconnect, RuntimeError):
                raise
            except Exception:
                # A DB/Redis failure handling one message must not close the
                # player's connection (an unhandled error here would end this
                # handler, tell the opponent they disconnected and start the
                # forfeit grace countdown). The client can simply retry.
                logger.exception(
                    "Handling a battle WS message failed — match_id=%d, user_id=%d, type=%s",
                    match_id, user_id, type(message).__name__,
                )
                # A failed Next/Skip gets the message the client resends on.
                detail = "Couldn't save your answer — retrying." if is_move else "Something went wrong — please try again."
                await websocket.send_json(ErrorEvent(detail=detail).model_dump(mode="json"))
    except WebSocketDisconnect:
        pass
    except RuntimeError:
        # The socket was force-closed by battle_connection_manager.close_local
        # (a match_finished/match_cancelled sweep) while this connection was
        # still mid-handshake or mid-send -- Starlette raises RuntimeError
        # ("Cannot call send once a close message has been sent") rather than
        # WebSocketDisconnect for that case. Equivalent to a disconnect for
        # our purposes: fall through to the same cleanup below. Logged (not
        # silently swallowed) so an unrelated RuntimeError from deeper in the
        # message loop is still visible.
        logger.info(
            "Battle match %d: connection for user_id=%d ended via RuntimeError "
            "(likely closed mid-handshake by a terminal-event sweep).",
            match_id, user_id,
        )
    finally:
        if ping_task is not None:
            ping_task.cancel()
        # The progression timer is owned by the connection manager now (one
        # per match per process, not one per connection) -- disconnect()
        # signals its stop_event only once this is truly the last local
        # connection for the match, see battle_connection_manager.py.
        was_current = ws_manager.disconnect(match_id=match_id, user_id=user_id, websocket=websocket)
        if was_current:
            # Detached, not awaited: the connection's own task may itself be
            # in the process of being torn down/cancelled right now (e.g. by
            # the ASGI server after a disconnect), and cleanup must not be
            # abandoned mid-flight just because ITS caller got cancelled.
            asyncio.create_task(
                _notify_disconnect(
                    match_id=match_id, user_id=user_id, redis=redis,
                    generation=connection_generation,
                )
            )


async def _notify_disconnect(*, match_id: int, user_id: int, redis: Redis, generation: int) -> None:
    # Debounce first: a same-device reconnect (e.g. queue.tsx's socket
    # closing right as match-session.tsx's opens a fresh one for the same
    # match) routinely lands within a second or two of this socket closing.
    # Sleeping here before doing anything else gives that reconnect time to
    # call bump_connection_generation first, so the generation check inside
    # mark_disconnected below correctly recognizes this notification as
    # stale and no-ops -- see BATTLE_DISCONNECT_NOTICE_DEBOUNCE_SECONDS.
    await asyncio.sleep(settings.BATTLE_DISCONNECT_NOTICE_DEBOUNCE_SECONDS)

    async with AsyncSessionLocal() as db:
        match = await battle_gameplay_service.get_match_row(db, match_id)
        if match is None or match.status in _TERMINAL_STATUSES:
            return
        # A player who has answered or skipped every question may leave (e.g.
        # back to the home screen to wait for the result) -- not a disconnect:
        # no notice to the opponent, no grace countdown, no forfeit.
        if await battle_gameplay_service.has_finished(db, match_id=match_id, user_id=user_id):
            return
    deadline = await battle_gameplay_service.mark_disconnected(
        redis, match_id=match_id, user_id=user_id, generation=generation
    )
    if deadline is None:
        # A newer connection (a reconnect) already superseded this one --
        # see mark_disconnected's docstring. Nothing to notify.
        return
    await ws_manager.publish(
        redis, match_id=match_id,
        event=PlayerDisconnectedEvent(
            user_id=user_id, deadline=deadline,
            grace_period_seconds=settings.BATTLE_DISCONNECT_GRACE_PERIOD_SECONDS,
        ).model_dump(mode="json"),
    )

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
from datetime import UTC, datetime

from fastapi import (
    APIRouter,
    Depends,
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
    AnswerAcknowledgedEvent,
    ConnectedEvent,
    ErrorEvent,
    MatchStartedEvent,
    PlayerDisconnectedEvent,
    PlayerReconnectedEvent,
    QuestionStartedEvent,
    StateSyncEvent,
    WSAnswerMessage,
    WSForfeitMessage,
    WSReadyMessage,
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
# period. Keyed so a second `player_disconnected` delivery for the same
# still-disconnected user (e.g. re-delivered via a duplicate subscriber
# wakeup) doesn't spawn a redundant countdown.
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


async def _grace_check_task(*, match_id: int, user_id: int, deadline: datetime, redis: Redis) -> None:
    try:
        delay = max(0.0, (deadline - datetime.now(UTC)).total_seconds())
        await asyncio.sleep(delay)

        if not await battle_gameplay_service.is_still_disconnected(redis, match_id=match_id, user_id=user_id):
            return  # reconnected before the deadline -- no-op

        event: dict | None = None
        async with AsyncSessionLocal() as db:
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
        if key in _pending_grace_tasks:
            return
        try:
            deadline = datetime.fromisoformat(event["deadline"])
        except (KeyError, ValueError):
            return
        _pending_grace_tasks[key] = asyncio.create_task(
            _grace_check_task(match_id=match_id, user_id=disconnected_user_id, deadline=deadline, redis=redis)
        )

    return _on_event


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

    Lockstep play: while "active", this loop also pushes one
    `question_started` event every time `_current_question_order` advances
    to a new value (including the very first push, right after
    `match_started` on the countdown->active transition) -- every
    locally-connected socket gets the SAME question at the SAME instant,
    since the index is a pure function of already-shared Postgres state
    (match.started_at). Sleeps until the current question's window closes
    (or, once the last question's window closes, until
    _ensure_match_progressed's elapsed-time check can finalize the match).
    """
    previous_status: str | None = None
    previous_question_index: int | None = None
    try:
        while not stop_event.is_set():
            async with AsyncSessionLocal() as db:
                match = await battle_gameplay_service.get_match_row(db, match_id)
                if match is None:
                    return
                match = await battle_gameplay_service._ensure_match_progressed(db, match, redis)
                current_status = match.status

                if current_status == "completed":
                    event = await battle_ws_events.match_finished_event(db, match)
                    await ws_manager.push_local(match_id, event)
                    return

                if current_status == "cancelled":
                    await ws_manager.push_local(match_id, battle_ws_events.match_cancelled_event(match))
                    return

                if current_status == "active":
                    if previous_status != "active":
                        await ws_manager.push_local(
                            match_id,
                            MatchStartedEvent(started_at=match.started_at).model_dump(mode="json"),
                        )

                    current_index = min(
                        battle_gameplay_service._current_question_order(match),
                        settings.BATTLE_QUESTION_COUNT - 1,
                    )
                    if current_index != previous_question_index:
                        question = await battle_gameplay_service.question_payload_for(db, match, current_index)
                        if question is not None:
                            await ws_manager.push_local(
                                match_id,
                                QuestionStartedEvent(
                                    question_index=current_index, question=question,
                                    time_remaining_seconds=battle_gameplay_service._question_time_remaining_seconds(
                                        match, current_index
                                    ),
                                ).model_dump(mode="json"),
                            )
                        previous_question_index = current_index

                    sleep_seconds = max(
                        0.05,
                        battle_gameplay_service._question_time_remaining_seconds(match, current_index),
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
                return
    except Exception:
        logger.exception("Battle match %d: progression timer failed.", match_id)


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

        while True:
            raw_text = await websocket.receive_text()
            try:
                message = parse_inbound_message(json.loads(raw_text))
            except (json.JSONDecodeError, ValidationError) as exc:
                await websocket.send_json(ErrorEvent(detail=f"Malformed message: {exc}").model_dump(mode="json"))
                continue

            async with AsyncSessionLocal() as db:
                if isinstance(message, WSReadyMessage):
                    await battle_gameplay_service.mark_ready(db, redis, match_id=match_id, user_id=user_id)
                elif isinstance(message, WSAnswerMessage):
                    try:
                        result = await battle_gameplay_service.submit_battle_answer(
                            db, match_id=match_id, user_id=user_id,
                            question_id=message.question_id, selected_option=message.selected_option,
                            redis=redis,
                        )
                        await websocket.send_json(AnswerAcknowledgedEvent(**result).model_dump(mode="json"))
                    except Exception as exc:  # HTTPException from validation/anti-replay guards
                        detail = getattr(exc, "detail", str(exc))
                        await websocket.send_json(ErrorEvent(detail=str(detail)).model_dump(mode="json"))
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
    async with AsyncSessionLocal() as db:
        match = await battle_gameplay_service.get_match_row(db, match_id)
    if match is None or match.status in _TERMINAL_STATUSES:
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

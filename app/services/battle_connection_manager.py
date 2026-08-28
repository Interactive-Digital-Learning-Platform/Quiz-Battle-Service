"""
app/services/battle_connection_manager.py
──────────────────────────────────────────────
Per-process, in-memory WebSocket connection registry for live battles, plus
the Redis pub/sub bridge that makes cross-process event delivery possible.

Why pub/sub is needed at all: a live WebSocket connection only ever exists
in the memory of the ONE worker process that accepted it (this service runs
multi-worker in deployment -- see Dockerfile's WEB_CONCURRENCY). If player A
is connected to process 1 and player B to process 2, an event caused by A's
action (e.g. "ready", answering a question, disconnecting) has to somehow
reach B's socket on process 2 -- that's what battle:match_events:{match_id}
(one Redis pub/sub channel per match) is for. Time-derived events (question
boundaries, match completion) do NOT need this -- they're pure functions of
already-shared Postgres/Redis state, so every process independently computes
and pushes them to its own local sockets via push_local(), no publish needed.

Delivery goes through exactly ONE path: publish() only ever publishes to
Redis, never calls the local-delivery function directly. Local delivery
happens exclusively inside each process's own subscriber loop (which also
receives its own just-published messages back, since it's subscribed to the
same channel it publishes to). If publish() also delivered locally, the
publisher's own subscriber loop would deliver the same event to its local
sockets a second time.

Subscribe-on-first-local-connection / unsubscribe-on-last-local-disconnection:
implicit in connect()/disconnect() below -- a subscriber task for a match
exists iff at least one local socket is registered for it. No separate ref
counter needed; `len(self._local[match_id])` IS the ref count.

The same first-connection/last-disconnection lifecycle owns the per-match
progression timer (time-driven push, see battle_ws.py's _progression_timer):
it must run exactly ONCE PER MATCH PER PROCESS, not once per connection --
two independent per-connection timers would each independently detect the
same state transition and call push_local, which broadcasts to every local
socket, so every client would receive each time-driven event duplicated once
per locally-connected socket (confirmed directly: two sockets on one process
each received `match_started` twice). connect() accepts
an optional factory to start the timer on first local connection, wrapped in
_run_progression_timer_with_respawn -- a supervisor that restarts it if it
ever dies from an unhandled exception (a transient DB/Redis error mid-tick),
rather than leaving that match's question progression silently frozen for
good, since otherwise it's only ever started once. disconnect() signals its
stop_event (cooperative shutdown, matching _progression_timer's own
requirement -- see its docstring for why raw task.cancel() isn't safe here)
on last local disconnection; the supervisor treats that as a clean exit too,
not a failure to restart from.
"""
import asyncio
import contextlib
import json
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

from fastapi import WebSocket
from redis.asyncio import Redis

from app.core.config import settings

logger = logging.getLogger(__name__)

OnEventCallback = Callable[[dict[str, Any]], Awaitable[None]]
ProgressionTimerFactory = Callable[[int, asyncio.Event], Coroutine[Any, Any, None]]


def _channel(match_id: int) -> str:
    return f"battle:match_events:{match_id}"


# Once a match reaches one of these, there's no more live gameplay left for
# either side -- the socket has nothing further to deliver, so it's closed
# server-side right after this event goes out rather than left idle for the
# client to eventually close on its own (e.g. on navigating away).
_TERMINAL_EVENT_TYPES = frozenset({"match_finished", "match_cancelled"})


class BattleConnectionManager:
    def __init__(self) -> None:
        # match_id -> {user_id: (websocket, on_event_callback)}
        self._local: dict[int, dict[int, tuple[WebSocket, OnEventCallback | None]]] = defaultdict(dict)
        self._subscriber_tasks: dict[int, asyncio.Task] = {}
        self._progression_tasks: dict[int, asyncio.Task] = {}
        self._progression_stops: dict[int, asyncio.Event] = {}

    async def connect(
        self, redis: Redis, *, match_id: int, user_id: int, websocket: WebSocket,
        on_event: OnEventCallback | None = None,
        start_progression_timer: ProgressionTimerFactory | None = None,
    ) -> WebSocket | None:
        """Registers a local connection, starting this process's pub/sub
        subscriber -- and, on first local connection, the progression timer
        via `start_progression_timer(match_id, stop_event)` -- for the match
        if it's the first local connection for it. Returns the PREVIOUS
        socket for this user+match, if any, so the caller can close a stale
        duplicate (e.g. a reconnect while an old tab/socket is technically
        still open).
        """
        previous = self._local[match_id].get(user_id)
        self._local[match_id][user_id] = (websocket, on_event)
        if match_id not in self._subscriber_tasks:
            self._subscriber_tasks[match_id] = asyncio.create_task(self._subscribe_loop(redis, match_id))
        if start_progression_timer is not None and match_id not in self._progression_tasks:
            stop_event = asyncio.Event()
            self._progression_stops[match_id] = stop_event
            self._progression_tasks[match_id] = asyncio.create_task(
                self._run_progression_timer_with_respawn(match_id, stop_event, start_progression_timer)
            )
        return previous[0] if previous else None

    async def _run_progression_timer_with_respawn(
        self, match_id: int, stop_event: asyncio.Event, factory: ProgressionTimerFactory,
    ) -> None:
        """Supervises the actual progression-timer coroutine (battle_ws.py's
        _progression_timer, which no longer swallows its own exceptions --
        see its docstring) and restarts it if it ever dies from an
        unhandled exception, instead of just letting the match's question
        progression freeze forever for anyone routed to this process. A
        CLEAN return (match finished/cancelled, not found, or stop_event
        set) is the expected way that coroutine ends and is never
        respawned here -- only an exception triggers a restart.
        """
        while True:
            try:
                await factory(match_id, stop_event)
                return  # clean exit -- match resolved or shutdown requested
            except asyncio.CancelledError:
                raise
            except Exception:
                if stop_event.is_set():
                    return  # shutting down anyway -- not a failure to recover from
                logger.exception(
                    "Battle match %d: progression timer died unexpectedly, restarting in %.1fs.",
                    match_id, settings.BATTLE_PROGRESSION_TIMER_RESPAWN_DELAY_SECONDS,
                )
                await asyncio.sleep(settings.BATTLE_PROGRESSION_TIMER_RESPAWN_DELAY_SECONDS)

    def disconnect(self, *, match_id: int, user_id: int, websocket: WebSocket) -> bool:
        """Deregisters a local connection. Returns False (a no-op) if this
        websocket isn't the CURRENT one registered for that user+match --
        i.e. it was already replaced by a newer connection, so this stale
        one shouldn't evict the new one on its way out.
        """
        current = self._local.get(match_id, {}).get(user_id)
        if current is None or current[0] is not websocket:
            return False
        del self._local[match_id][user_id]
        logger.info("Battle match %d: connection closed — user_id=%d.", match_id, user_id)
        if not self._local[match_id]:
            del self._local[match_id]
            task = self._subscriber_tasks.pop(match_id, None)
            if task is not None:
                task.cancel()
            self._progression_tasks.pop(match_id, None)
            stop_event = self._progression_stops.pop(match_id, None)
            if stop_event is not None:
                stop_event.set()
            logger.info(
                "Battle match %d: last local connection closed — subscriber and progression "
                "timer torn down for this process.", match_id,
            )
        return True

    async def _subscribe_loop(self, redis: Redis, match_id: int) -> None:
        pubsub = redis.pubsub()
        try:
            await pubsub.subscribe(_channel(match_id))
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                try:
                    event = json.loads(message["data"])
                except (TypeError, ValueError):
                    logger.warning("Battle match %d: dropping malformed pub/sub message.", match_id)
                    continue
                for websocket, on_event in list(self._local.get(match_id, {}).values()):
                    with contextlib.suppress(Exception):
                        await websocket.send_json(event)
                    if on_event is not None:
                        with contextlib.suppress(Exception):
                            await on_event(event)
                if event.get("type") in _TERMINAL_EVENT_TYPES:
                    await self.close_local(match_id)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Battle match %d: pub/sub subscriber loop failed.", match_id)
        finally:
            with contextlib.suppress(Exception):
                await pubsub.unsubscribe(_channel(match_id))
                await pubsub.aclose()

    async def push_local(self, match_id: int, event: dict[str, Any]) -> None:
        """Local-only delivery, no Redis publish involved -- for
        time-derived events every process independently computes (see
        module docstring).
        """
        for websocket, _on_event in list(self._local.get(match_id, {}).values()):
            with contextlib.suppress(Exception):
                await websocket.send_json(event)
        if event.get("type") in _TERMINAL_EVENT_TYPES:
            await self.close_local(match_id)

    async def close_local(self, match_id: int, *, code: int = 1000, reason: str = "") -> None:
        """Force-closes every locally-connected socket for this match --
        called right after a terminal event (match_finished/match_cancelled)
        is delivered, so neither side is left holding an idle connection to
        a match that's already over. Each closed socket's own `finally`
        block back in battle_ws.py still runs disconnect() as normal (same
        as any other close), so registry/subscriber/progression-timer
        teardown isn't duplicated here -- disconnect()'s own logging is what
        confirms a given socket actually finished tearing down, not this.
        """
        sockets = list(self._local.get(match_id, {}).items())
        if not sockets:
            return
        logger.info(
            "Battle match %d: match over, force-closing %d local connection(s) — user_ids=%s.",
            match_id, len(sockets), [user_id for user_id, _ in sockets],
        )
        for user_id, (websocket, _on_event) in sockets:
            try:
                await websocket.close(code=code, reason=reason)
            except Exception:
                logger.exception(
                    "Battle match %d: failed to force-close connection for user_id=%d.", match_id, user_id
                )

    async def publish(self, redis: Redis, *, match_id: int, event: dict[str, Any]) -> None:
        """The ONLY entry point for cross-process, action-driven events.
        Delivery to local sockets happens via this process's OWN
        subscription receiving the message back -- see module docstring.
        """
        await redis.publish(_channel(match_id), json.dumps(event, default=str))


manager = BattleConnectionManager()

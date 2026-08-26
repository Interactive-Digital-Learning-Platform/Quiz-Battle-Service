"""
tests/test_battle_ws_routes.py
─────────────────────────────────
Live WebSocket-level tests for WS /api/v1/battle/match/{id}/ws, using
starlette.testclient.TestClient.websocket_connect() over a real ASGI
round-trip (real routing, real Depends resolution, real
battle_connection_manager pub/sub via fake_redis).

Deliberately plain (non-`async def`) test functions: TestClient drives the
app in its own background thread/event loop (an anyio "portal"), so async
setup here is done via `asyncio.run(...)` in the calling thread rather than
making these tests themselves async. `app.api.routes.battle_ws` does its
own short-lived `AsyncSessionLocal()` per DB touch (NOT `Depends(get_db)`,
see that module's docstring), so pointing it at an isolated test DB means
monkeypatching `battle_ws.AsyncSessionLocal` directly -- `get_db`'s
dependency-override machinery has no effect on it. `Depends(get_redis)` and
`Depends(get_current_user_ws)` ARE real FastAPI dependencies, so those use
the normal `app.dependency_overrides` mechanism.

A real (temp-file, default-pool) SQLite DB is used here rather than the rest
of the suite's `:memory:` + StaticPool `db_session` fixture: StaticPool's
single shared connection does not tolerate a query being interrupted by
TestClient's own portal-thread teardown cleanly (verified directly -- it can
wedge the pool's one connection). A normal per-checkout-connection pool
against a temp file sidesteps that entirely and is arguably a closer match
to real (Postgres, real connection pool) production topology anyway.

Kept intentionally small -- the bulk of ready/countdown/forfeit/disconnect
logic coverage lives in tests/test_battle_ready_forfeit_service.py (pure
async, no live sockets). The full real-time grace-period-expiry-into-forfeit
path is NOT exercised here (that would require actually sleeping out the
grace period) -- it's covered by that file's forfeit_match/is_still_disconnected
tests instead.
"""
import asyncio
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import fakeredis
import pytest
from fastapi import Header, HTTPException, WebSocket, WebSocketException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import app.api.routes.battle_ws as battle_ws_module
from app import models as _app_models  # noqa: F401  (register model metadata on Base)
from app.core.config import settings
from app.core.database import Base, get_db
from app.core.redis_client import get_redis
from app.core.security import get_current_user, get_current_user_ws
from app.main import app
from app.models.battle_match import BattleMatch, BattleParticipant
from app.models.question import Question
from app.services import battle_gameplay_service, matchmaking_service
from app.services.battle_connection_manager import manager as ws_manager
from app.services.user_service import get_or_create_user

CLERK_A = "ws-battle-user-a"
CLERK_B = "ws-battle-user-b"
N = settings.BATTLE_QUESTION_COUNT
Q = settings.BATTLE_QUESTION_TIME_LIMIT_SECONDS


async def _override_get_current_user_ws(websocket: WebSocket) -> dict:
    # Test-only stand-in for the real JWT-verification logic (same spirit as
    # conftest.py's HTTP get_current_user override): the query-param token
    # IS the clerk_id directly, no real Clerk call. Still replicates the
    # real dependency's "missing/invalid token -> reject pre-accept" shape
    # so that rejection path is genuinely exercised, not just bypassed.
    token = websocket.query_params.get("token")
    if not token or token == "invalid-token":
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Missing or invalid token.")
    return {"sub": token}


def _events_by_type(ws, n: int) -> dict:
    """Receive exactly `n` messages and key them by `type` -- used wherever
    a socket may receive its OWN action's broadcast alongside a direct
    response (e.g. answering both triggers a direct `answer_acknowledged`
    AND a self-delivered `opponent_answered`, per battle_connection_manager's
    "publish reaches every locally-connected socket, including the
    publisher's own" design), so relying on receive order would be flaky.
    """
    return {(e := ws.receive_json())["type"]: e for _ in range(n)}


def _ready_up_to_active(ws_a, ws_b, questions: list[Question]) -> None:
    ws_a.send_json({"type": "ready"})
    ws_a.receive_json()
    ws_b.receive_json()

    with patch(
        "app.services.battle_gameplay_service.select_battle_questions",
        new=AsyncMock(return_value=questions),
    ):
        ws_b.send_json({"type": "ready"})
        for _ in range(2):
            ws_a.receive_json()  # player_ready, countdown_started
            ws_b.receive_json()


def _fixed_questions(count: int = N) -> list[Question]:
    return [
        Question(
            question=f"WS route test question {i}?", options=["A", "B", "C", "D"],
            correct_answer="A", subject="Mathematics", lesson="Algebra", difficulty="medium",
        )
        for i in range(count)
    ]


class WsEnv:
    def __init__(self, client, session_factory, redis, match_id, user_a_id, user_b_id):
        self.client = client
        self.session_factory = session_factory
        self.redis = redis
        self.match_id = match_id
        self.user_a_id = user_a_id
        self.user_b_id = user_b_id

    def run(self, coro):
        return asyncio.run(coro)

    def url(self, token: str | None) -> str:
        base = f"/api/v1/battle/match/{self.match_id}/ws"
        return f"{base}?token={token}" if token else base

    def seed_questions(self, questions: list[Question]) -> None:
        async def _seed():
            async with self.session_factory() as db:
                db.add_all(questions)
                await db.commit()
        self.run(_seed())

    def backdate_started_at(self, seconds_ago: float) -> None:
        async def _backdate():
            async with self.session_factory() as db:
                await db.execute(
                    update(BattleMatch).where(BattleMatch.id == self.match_id).values(
                        started_at=datetime.now(UTC) - timedelta(seconds=seconds_ago)
                    )
                )
                await db.commit()
        self.run(_backdate())


@pytest.fixture
def ws_env(tmp_path):
    db_path = tmp_path / "battle_ws_test.db"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}", connect_args={"check_same_thread": False}, poolclass=NullPool,
    )
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)

    battle_ws_module.AsyncSessionLocal = session_factory

    async def _setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with session_factory() as db:
            user_a = await get_or_create_user(db, CLERK_A)
            user_b = await get_or_create_user(db, CLERK_B)
            match = BattleMatch(subject="Mathematics", status="waiting", difficulty="Mostly Medium")
            db.add(match)
            await db.flush()
            db.add(BattleParticipant(match_id=match.id, user_id=user_a.id, rating_before=1000))
            db.add(BattleParticipant(match_id=match.id, user_id=user_b.id, rating_before=1000))
            await db.commit()
            return match.id, user_a.id, user_b.id

    match_id, user_a_id, user_b_id = asyncio.run(_setup())

    async def override_get_redis():
        return redis

    app.dependency_overrides[get_redis] = override_get_redis
    app.dependency_overrides[get_current_user_ws] = _override_get_current_user_ws

    env = WsEnv(TestClient(app), session_factory, redis, match_id, user_a_id, user_b_id)
    yield env

    app.dependency_overrides.pop(get_redis, None)
    app.dependency_overrides.pop(get_current_user_ws, None)
    asyncio.run(engine.dispose())
    asyncio.run(redis.aclose())


# ─────────────────────────────────────────────────────────────────────────────
# Pre-accept rejection
# ─────────────────────────────────────────────────────────────────────────────

def test_connection_rejected_without_token(ws_env):
    with pytest.raises(WebSocketDisconnect):
        with ws_env.client.websocket_connect(ws_env.url(None)):
            pass


def test_connection_rejected_with_invalid_token(ws_env):
    with pytest.raises(WebSocketDisconnect):
        with ws_env.client.websocket_connect(ws_env.url("invalid-token")):
            pass


def test_connection_rejected_for_non_participant(ws_env):
    with pytest.raises(WebSocketDisconnect):
        with ws_env.client.websocket_connect(ws_env.url("ws-battle-outsider")):
            pass


def test_connection_rejected_after_rate_limit_exceeded(ws_env, monkeypatch):
    monkeypatch.setattr(settings, "BATTLE_WS_CONNECT_RATE_LIMIT_MAX_CALLS", 2)

    for _ in range(2):
        with ws_env.client.websocket_connect(ws_env.url(CLERK_A)) as ws:
            ws.receive_json()  # connected
            ws.receive_json()  # state_sync

    with pytest.raises(WebSocketDisconnect):
        with ws_env.client.websocket_connect(ws_env.url(CLERK_A)):
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Successful connect
# ─────────────────────────────────────────────────────────────────────────────

def test_successful_connect_receives_state_sync(ws_env):
    with ws_env.client.websocket_connect(ws_env.url(CLERK_A)) as ws:
        connected = ws.receive_json()
        assert connected == {"type": "connected", "match_id": ws_env.match_id}

        sync = ws.receive_json()
        assert sync["type"] == "state_sync"
        assert sync["state"]["status"] == "waiting"


# ─────────────────────────────────────────────────────────────────────────────
# Ready flow over WS messages
# ─────────────────────────────────────────────────────────────────────────────

def test_both_ready_produces_player_ready_then_countdown(ws_env):
    questions = _fixed_questions()
    ws_env.seed_questions(questions)

    with ws_env.client.websocket_connect(ws_env.url(CLERK_A)) as ws_a:
        ws_a.receive_json()  # connected
        ws_a.receive_json()  # state_sync
        with ws_env.client.websocket_connect(ws_env.url(CLERK_B)) as ws_b:
            ws_b.receive_json()  # connected
            ws_b.receive_json()  # state_sync

            ws_a.send_json({"type": "ready"})
            ready_1a = ws_a.receive_json()
            ready_1b = ws_b.receive_json()
            assert ready_1a == ready_1b == {"type": "player_ready", "user_id": ws_env.user_a_id}

            with patch(
                "app.services.battle_gameplay_service.select_battle_questions",
                new=AsyncMock(return_value=questions),
            ):
                ws_b.send_json({"type": "ready"})
                events_a = [ws_a.receive_json(), ws_a.receive_json()]
                events_b = [ws_b.receive_json(), ws_b.receive_json()]

            for events in (events_a, events_b):
                types = [e["type"] for e in events]
                assert types == ["player_ready", "countdown_started"]
                assert events[0]["user_id"] == ws_env.user_b_id
                assert events[1]["duration_seconds"] == settings.BATTLE_COUNTDOWN_DURATION_SECONDS


# ─────────────────────────────────────────────────────────────────────────────
# Time-driven push (no client polling)
# ─────────────────────────────────────────────────────────────────────────────

def test_time_driven_push_arrives_without_polling(ws_env, monkeypatch):
    monkeypatch.setattr(settings, "BATTLE_WS_WAITING_POLL_INTERVAL_SECONDS", 0.05)
    questions = _fixed_questions()
    ws_env.seed_questions(questions)

    with ws_env.client.websocket_connect(ws_env.url(CLERK_A)) as ws_a:
        ws_a.receive_json()
        ws_a.receive_json()
        with ws_env.client.websocket_connect(ws_env.url(CLERK_B)) as ws_b:
            ws_b.receive_json()
            ws_b.receive_json()

            ws_a.send_json({"type": "ready"})
            ws_a.receive_json()
            ws_b.receive_json()

            with patch(
                "app.services.battle_gameplay_service.select_battle_questions",
                new=AsyncMock(return_value=questions),
            ):
                ws_b.send_json({"type": "ready"})
                for _ in range(2):
                    ws_a.receive_json()
                    ws_b.receive_json()

            # Backdate started_at into the past -- neither client sends
            # anything further; the next progression-timer tick on each
            # connected process independently notices and pushes BOTH
            # match_started and (since the question index also just changed,
            # from None to 0) question_started, lockstep play's per-question
            # boundary push.
            ws_env.backdate_started_at(seconds_ago=1)

            events_a = [ws_a.receive_json(), ws_a.receive_json()]
            events_b = [ws_b.receive_json(), ws_b.receive_json()]

    for events in (events_a, events_b):
        types = [e["type"] for e in events]
        assert types == ["match_started", "question_started"]
        assert events[1]["question_index"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Disconnect notification
# ─────────────────────────────────────────────────────────────────────────────

def test_closing_one_socket_notifies_the_other(ws_env):
    with ws_env.client.websocket_connect(ws_env.url(CLERK_A)) as ws_a:
        ws_a.receive_json()
        ws_a.receive_json()
        with ws_env.client.websocket_connect(ws_env.url(CLERK_B)) as ws_b:
            ws_b.receive_json()
            ws_b.receive_json()

            ws_a.close()
            event = ws_b.receive_json()

    assert event["type"] == "player_disconnected"
    assert event["user_id"] == ws_env.user_a_id
    assert event["grace_period_seconds"] == settings.BATTLE_DISCONNECT_GRACE_PERIOD_SECONDS


# ─────────────────────────────────────────────────────────────────────────────
# Answer submission over WS -- acknowledgement + opponent privacy
# ─────────────────────────────────────────────────────────────────────────────

def test_ws_answer_acknowledged_and_opponent_notified_without_leaking_selection(ws_env):
    questions = _fixed_questions()
    ws_env.seed_questions(questions)

    with ws_env.client.websocket_connect(ws_env.url(CLERK_A)) as ws_a:
        ws_a.receive_json()
        ws_a.receive_json()
        with ws_env.client.websocket_connect(ws_env.url(CLERK_B)) as ws_b:
            ws_b.receive_json()
            ws_b.receive_json()

            _ready_up_to_active(ws_a, ws_b, questions)
            ws_env.backdate_started_at(seconds_ago=1)

            # Lockstep play pushes match_started then question_started (the
            # single current question, index 0) -- consume both on each side.
            match_started_a = ws_a.receive_json()
            question_started_a = ws_a.receive_json()
            ws_b.receive_json()
            ws_b.receive_json()
            assert match_started_a["type"] == "match_started"
            assert question_started_a["type"] == "question_started"
            question_id = questions[0].id
            assert question_started_a["question"]["question_id"] == question_id

            ws_a.send_json({"type": "answer", "question_id": question_id, "selected_option": "A"})
            # A gets BOTH a direct answer_acknowledged AND its own action's
            # opponent_answered broadcast (every publish reaches every
            # locally-connected socket, including the publisher's own).
            ack_events_a = _events_by_type(ws_a, 2)
            opponent_event = ws_b.receive_json()

    ack = ack_events_a["answer_acknowledged"]
    assert ack["question_id"] == question_id
    assert ack["is_correct"] is True
    assert ack["base_score"] == settings.BATTLE_SCORE_BASE_CORRECT
    assert "correct_answer" not in ack
    assert "selected_option" not in ack

    self_broadcast = ack_events_a["opponent_answered"]
    for event in (self_broadcast, opponent_event):
        assert event["type"] == "opponent_answered"
        assert event["user_id"] == ws_env.user_a_id
        assert event["answered_count"] == 1
        # Reveals is_correct/question_order (drives the opponent's live
        # progress-bar segment) but never the selected option itself.
        assert event["question_order"] == 0
        assert event["is_correct"] is True
        assert "selected_option" not in event


# ─────────────────────────────────────────────────────────────────────────────
# Reconnect within the grace period -- state restoration
# ─────────────────────────────────────────────────────────────────────────────

def test_reconnect_within_grace_period_restores_state_and_notifies_opponent(ws_env):
    questions = _fixed_questions()
    ws_env.seed_questions(questions)

    with ws_env.client.websocket_connect(ws_env.url(CLERK_A)) as ws_a:
        ws_a.receive_json()
        ws_a.receive_json()
        with ws_env.client.websocket_connect(ws_env.url(CLERK_B)) as ws_b:
            ws_b.receive_json()
            ws_b.receive_json()

            _ready_up_to_active(ws_a, ws_b, questions)
            ws_env.backdate_started_at(seconds_ago=1)
            ws_a.receive_json()  # match_started
            ws_a.receive_json()  # question_started
            ws_b.receive_json()  # match_started
            ws_b.receive_json()  # question_started

            ws_a.close()
            disconnected_event = ws_b.receive_json()
            assert disconnected_event["type"] == "player_disconnected"

            # A reconnects well within the (now 15s-default) grace period.
            with ws_env.client.websocket_connect(ws_env.url(CLERK_A)) as ws_a2:
                reconnected_event = ws_b.receive_json()
                assert reconnected_event["type"] == "player_reconnected"
                assert reconnected_event["user_id"] == ws_env.user_a_id

                # A2 also gets its own reconnect broadcast alongside the two
                # direct sends every fresh connection receives.
                events_a2 = _events_by_type(ws_a2, 3)

    assert events_a2["player_reconnected"]["user_id"] == ws_env.user_a_id
    assert events_a2["connected"]["match_id"] == ws_env.match_id
    restored_state = events_a2["state_sync"]["state"]
    assert restored_state["status"] == "active"
    assert restored_state["question_index"] == 0
    assert restored_state["question"] is not None


def test_socket_closed_mid_handshake_by_terminal_sweep_does_not_crash(ws_env):
    """Regression: battle_connection_manager.close_local() force-closes every
    locally-registered socket for a match once it reaches a terminal status
    (see close_local's own docstring). If that sweep lands while THIS
    connection's own handshake is still between its ConnectedEvent and
    StateSyncEvent sends, Starlette's send() raises RuntimeError ("Cannot
    call send once a close message has been sent") rather than
    WebSocketDisconnect -- the route must treat that like any other
    disconnect (clean deregister, no unhandled ASGI exception), not crash.
    """
    real_get_match_state = battle_gameplay_service.get_match_state

    async def _get_match_state_then_close(db, *, match_id, user_id, redis):
        # Simulates close_local racing in exactly between the two handshake
        # sends (e.g. a match_finished delivered while this same user's
        # reconnect handshake is still running).
        await ws_manager.close_local(match_id)
        return await real_get_match_state(db, match_id=match_id, user_id=user_id, redis=redis)

    with patch(
        "app.api.routes.battle_ws.battle_gameplay_service.get_match_state",
        new=_get_match_state_then_close,
    ):
        with ws_env.client.websocket_connect(ws_env.url(CLERK_A)) as ws:
            connected = ws.receive_json()
            assert connected["type"] == "connected"
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()  # state_sync never arrives -- socket already closed

    # disconnect() must still have run despite the RuntimeError -- if the
    # handshake's exception had propagated past the route's try/finally
    # (the actual bug: the finally only used to wrap the message loop, not
    # the handshake sends), this match's registry entry, subscriber task and
    # progression timer would all be left dangling forever.
    assert not ws_manager._local.get(ws_env.match_id)
    assert ws_env.match_id not in ws_manager._subscriber_tasks
    assert ws_env.match_id not in ws_manager._progression_tasks
    assert ws_env.match_id not in ws_manager._progression_stops

    # A fresh connect for the same user then gets a normal, uninterrupted
    # handshake -- not corrupted by the crashed connection's leftovers.
    with ws_env.client.websocket_connect(ws_env.url(CLERK_A)) as ws2:
        connected2 = ws2.receive_json()
        sync2 = ws2.receive_json()
    assert connected2["type"] == "connected"
    assert sync2["type"] == "state_sync"


# ─────────────────────────────────────────────────────────────────────────────
# Question boundary advances + total-time-budget completion over WS
# ─────────────────────────────────────────────────────────────────────────────

def test_ws_pushes_next_question_started_at_the_window_boundary(ws_env):
    questions = _fixed_questions()
    ws_env.seed_questions(questions)

    with ws_env.client.websocket_connect(ws_env.url(CLERK_A)) as ws_a:
        ws_a.receive_json()
        ws_a.receive_json()
        with ws_env.client.websocket_connect(ws_env.url(CLERK_B)) as ws_b:
            ws_b.receive_json()
            ws_b.receive_json()

            _ready_up_to_active(ws_a, ws_b, questions)
            ws_env.backdate_started_at(seconds_ago=1)
            ws_a.receive_json()  # match_started
            first_question_a = ws_a.receive_json()
            ws_b.receive_json()  # match_started
            first_question_b = ws_b.receive_json()
            assert first_question_a["question"]["question_id"] == questions[0].id
            assert first_question_b["question"]["question_id"] == questions[0].id

            # Backdate past question 0's window -- both connected processes'
            # progression timers independently notice and push the next one.
            ws_env.backdate_started_at(seconds_ago=Q + 1)
            second_question_a = ws_a.receive_json()
            second_question_b = ws_b.receive_json()

    assert second_question_a["type"] == "question_started"
    assert second_question_a["question_index"] == 1
    assert second_question_a["question"]["question_id"] == questions[1].id
    assert second_question_b["question"]["question_id"] == questions[1].id


def test_ws_pushes_match_finished_once_total_time_budget_elapses(ws_env):
    questions = _fixed_questions()
    ws_env.seed_questions(questions)

    with ws_env.client.websocket_connect(ws_env.url(CLERK_A)) as ws_a:
        ws_a.receive_json()
        ws_a.receive_json()
        with ws_env.client.websocket_connect(ws_env.url(CLERK_B)) as ws_b:
            ws_b.receive_json()
            ws_b.receive_json()

            _ready_up_to_active(ws_a, ws_b, questions)
            ws_env.backdate_started_at(seconds_ago=1)
            ws_a.receive_json()  # match_started
            ws_a.receive_json()  # question_started
            ws_b.receive_json()  # match_started
            ws_b.receive_json()  # question_started

            # Neither player answers anything -- backdate past the WHOLE
            # question set's combined window so the progression timer's own
            # elapsed-time check finalizes the match.
            ws_env.backdate_started_at(seconds_ago=N * Q + 1)
            finished_a = ws_a.receive_json()
            finished_b = ws_b.receive_json()

    assert finished_a["type"] == "match_finished"
    assert finished_b["type"] == "match_finished"
    assert finished_a["winner_user_id"] is None  # neither answered anything -- draw


# ─────────────────────────────────────────────────────────────────────────────
# Reconnect grace-period expiry -- automatic forfeit
# ─────────────────────────────────────────────────────────────────────────────

def test_disconnect_grace_period_expiry_triggers_forfeit(ws_env, monkeypatch):
    # Shrunk from the real default so this test doesn't actually wait 15s --
    # the wait below is real wall-clock time (the grace check runs as a
    # background asyncio task on the TestClient's own portal thread, so a
    # plain time.sleep() here genuinely lets it fire, same principle
    # scripts/verify_battle_live.py documents for why it can't be a
    # backdated-timestamp trick like the rest of this suite).
    monkeypatch.setattr(settings, "BATTLE_DISCONNECT_GRACE_PERIOD_SECONDS", 0.1)
    questions = _fixed_questions()
    ws_env.seed_questions(questions)

    with ws_env.client.websocket_connect(ws_env.url(CLERK_A)) as ws_a:
        ws_a.receive_json()
        ws_a.receive_json()
        with ws_env.client.websocket_connect(ws_env.url(CLERK_B)) as ws_b:
            ws_b.receive_json()
            ws_b.receive_json()

            _ready_up_to_active(ws_a, ws_b, questions)
            ws_env.backdate_started_at(seconds_ago=1)
            ws_a.receive_json()  # match_started
            ws_a.receive_json()  # question_started
            ws_b.receive_json()  # match_started
            ws_b.receive_json()  # question_started

            ws_a.close()
            disconnected_event = ws_b.receive_json()
            assert disconnected_event["type"] == "player_disconnected"

            time.sleep(1.0)
            finished = ws_b.receive_json()

    assert finished["type"] == "match_finished"
    assert finished["winner_user_id"] == ws_env.user_b_id
    by_user = {p["user_id"]: p for p in finished["participants"]}
    assert by_user[ws_env.user_a_id]["result"] == "forfeit"
    assert by_user[ws_env.user_b_id]["result"] == "win"


# ─────────────────────────────────────────────────────────────────────────────
# REST + WS combined (Phase 6 §3) -- the seam neither of the above two test
# styles exercises: a real POST /queue/join pairing (not a hand-seeded
# match), both sides connecting over WS, then a REST-submitted forfeit
# whose match_finished push is observed over WS -- plus the release_match_
# state regression this same phase's §1/§2 fixed (Phase 6's own headline
# bug: nothing before this covered forfeit's Redis side effect end-to-end
# through a real queue/join-produced match).
# ─────────────────────────────────────────────────────────────────────────────

async def _override_get_current_user(authorization: str | None = Header(None)) -> dict:
    # Test-only stand-in, same spirit as _override_get_current_user_ws above:
    # the bearer token IS the clerk_id directly, no real Clerk call.
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    return {"sub": authorization.removeprefix("Bearer ").strip()}


@pytest.fixture
def queue_ws_env(tmp_path):
    db_path = tmp_path / "battle_queue_ws_test.db"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}", connect_args={"check_same_thread": False}, poolclass=NullPool,
    )
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)

    battle_ws_module.AsyncSessionLocal = session_factory

    async def _setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_setup())

    async def override_get_db():
        async with session_factory() as db:
            yield db

    async def override_get_redis():
        return redis

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_redis] = override_get_redis
    app.dependency_overrides[get_current_user] = _override_get_current_user
    app.dependency_overrides[get_current_user_ws] = _override_get_current_user_ws

    client = TestClient(app)
    yield client, session_factory, redis

    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_redis, None)
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_current_user_ws, None)
    asyncio.run(engine.dispose())
    asyncio.run(redis.aclose())


def test_rest_queue_join_then_ws_ready_then_rest_forfeit_pushes_ws_finish(queue_ws_env):
    client, session_factory, redis = queue_ws_env

    resp_a = client.post(
        "/api/v1/battle/queue/join", json={"subject": "Mathematics"},
        headers={"Authorization": f"Bearer {CLERK_A}"},
    )
    assert resp_a.status_code == 200
    assert resp_a.json()["status"] == "searching"

    resp_b = client.post(
        "/api/v1/battle/queue/join", json={"subject": "Mathematics"},
        headers={"Authorization": f"Bearer {CLERK_B}"},
    )
    assert resp_b.status_code == 200
    body_b = resp_b.json()
    assert body_b["status"] == "matched"
    match_id = body_b["match_id"]
    assert match_id is not None

    status_a = client.get(
        "/api/v1/battle/queue/status", headers={"Authorization": f"Bearer {CLERK_A}"}
    )
    assert status_a.json()["status"] == "matched"
    assert status_a.json()["match_id"] == match_id

    async def _lookup_user_ids():
        async with session_factory() as db:
            user_a = await get_or_create_user(db, CLERK_A)
            user_b = await get_or_create_user(db, CLERK_B)
            return user_a.id, user_b.id

    user_a_id, user_b_id = asyncio.run(_lookup_user_ids())

    # Seed a fixed question pool so ready-triggered activation doesn't hit
    # the real Groq path (mocked below regardless, but the objects need ids).
    questions = _fixed_questions()

    async def _seed_questions():
        async with session_factory() as db:
            db.add_all(questions)
            await db.commit()

    asyncio.run(_seed_questions())

    ws_url = f"/api/v1/battle/match/{match_id}/ws"
    with client.websocket_connect(f"{ws_url}?token={CLERK_A}") as ws_a:
        ws_a.receive_json()  # connected
        ws_a.receive_json()  # state_sync
        with client.websocket_connect(f"{ws_url}?token={CLERK_B}") as ws_b:
            ws_b.receive_json()  # connected
            ws_b.receive_json()  # state_sync

            ws_a.send_json({"type": "ready"})
            ws_a.receive_json()
            ws_b.receive_json()

            with patch(
                "app.services.battle_gameplay_service.select_battle_questions",
                new=AsyncMock(return_value=questions),
            ):
                ws_b.send_json({"type": "ready"})
                for _ in range(2):
                    ws_a.receive_json()  # player_ready, countdown_started
                    ws_b.receive_json()

            # User A forfeits over REST, not WS -- the exact seam this test
            # targets (every other WS test only ever forfeits/answers over
            # the same socket it connected with).
            forfeit_resp = client.post(
                f"/api/v1/battle/match/{match_id}/forfeit",
                headers={"Authorization": f"Bearer {CLERK_A}"},
            )
            assert forfeit_resp.status_code == 200
            assert forfeit_resp.json()["status"] == "completed"

            finished_a = ws_a.receive_json()
            finished_b = ws_b.receive_json()

            # The server closes both sockets right after match_finished --
            # neither side should be left holding a connection to a match
            # that's already over (see battle_connection_manager.close_local).
            with pytest.raises(WebSocketDisconnect):
                ws_a.receive_json()
            with pytest.raises(WebSocketDisconnect):
                ws_b.receive_json()

    assert finished_a["type"] == "match_finished"
    assert finished_b["type"] == "match_finished"
    assert finished_a["winner_user_id"] == user_b_id

    # Ties into Phase 6 §1/§2: a REST-triggered forfeit on a queue/join-
    # produced match must release both players' Redis "in_match" state,
    # or they'd be locked out of ever queuing again.
    assert asyncio.run(redis.exists(matchmaking_service._user_state_key(user_a_id))) == 0
    assert asyncio.run(redis.exists(matchmaking_service._user_state_key(user_b_id))) == 0


def test_ws_forfeit_before_either_ready_voids_match_instead_of_scoring(queue_ws_env):
    # Matched but still "waiting" (neither side has readied yet) -- backing
    # out here must void the match and let the OTHER side's client
    # auto-requeue (see Mobile-Application's queue.tsx), not score a win.
    client, session_factory, redis = queue_ws_env

    resp_a = client.post(
        "/api/v1/battle/queue/join", json={"subject": "Mathematics"},
        headers={"Authorization": f"Bearer {CLERK_A}"},
    )
    assert resp_a.status_code == 200

    resp_b = client.post(
        "/api/v1/battle/queue/join", json={"subject": "Mathematics"},
        headers={"Authorization": f"Bearer {CLERK_B}"},
    )
    assert resp_b.status_code == 200
    match_id = resp_b.json()["match_id"]
    assert match_id is not None

    ws_url = f"/api/v1/battle/match/{match_id}/ws"
    with client.websocket_connect(f"{ws_url}?token={CLERK_A}") as ws_a:
        ws_a.receive_json()  # connected
        state_sync_a = ws_a.receive_json()
        assert state_sync_a["state"]["status"] == "waiting"

        with client.websocket_connect(f"{ws_url}?token={CLERK_B}") as ws_b:
            ws_b.receive_json()  # connected
            ws_b.receive_json()  # state_sync

            ws_a.send_json({"type": "forfeit"})

            cancelled_a = ws_a.receive_json()
            cancelled_b = ws_b.receive_json()

    assert cancelled_a["type"] == "match_cancelled"
    assert cancelled_b["type"] == "match_cancelled"
    assert "opponent left" in cancelled_a["reason"].lower()

    async def _lookup_match_status():
        async with session_factory() as db:
            return (await db.execute(select(BattleMatch).where(BattleMatch.id == match_id))).scalar_one()

    match = asyncio.run(_lookup_match_status())
    assert match.status == "cancelled"
    assert match.winner_user_id is None

    async def _lookup_user_ids():
        async with session_factory() as db:
            user_a = await get_or_create_user(db, CLERK_A)
            user_b = await get_or_create_user(db, CLERK_B)
            return user_a.id, user_b.id

    user_a_id, user_b_id = asyncio.run(_lookup_user_ids())
    assert asyncio.run(redis.exists(matchmaking_service._user_state_key(user_a_id))) == 0
    assert asyncio.run(redis.exists(matchmaking_service._user_state_key(user_b_id))) == 0


def test_forfeit_waiting_match_then_both_requeue_and_second_match_starts(queue_ws_env):
    # Full cycle behind the "opponent can't start a match after they cancel
    # and try to matchmake again" bug report: A forfeits a still-"waiting"
    # match (voiding it, per the test above), then BOTH A (the canceller,
    # who the mobile client sends home) and B (auto-requeued by queue.tsx)
    # try to queue again -- with only two users in play they must rematch
    # each other, and that SECOND match must be able to actually progress
    # through ready-up into countdown, not just be joinable.
    client, session_factory, redis = queue_ws_env

    resp_a = client.post(
        "/api/v1/battle/queue/join", json={"subject": "Mathematics"},
        headers={"Authorization": f"Bearer {CLERK_A}"},
    )
    assert resp_a.status_code == 200

    resp_b = client.post(
        "/api/v1/battle/queue/join", json={"subject": "Mathematics"},
        headers={"Authorization": f"Bearer {CLERK_B}"},
    )
    assert resp_b.status_code == 200
    first_match_id = resp_b.json()["match_id"]
    assert first_match_id is not None

    forfeit_resp = client.post(
        f"/api/v1/battle/match/{first_match_id}/forfeit",
        headers={"Authorization": f"Bearer {CLERK_A}"},
    )
    assert forfeit_resp.status_code == 200
    assert forfeit_resp.json()["status"] == "cancelled"

    # Both sides try to matchmake again, exactly like the mobile flow: A
    # (sent home) picks the subject again, B (auto-requeued by queue.tsx)
    # rejoins on its own.
    resp_a2 = client.post(
        "/api/v1/battle/queue/join", json={"subject": "Mathematics"},
        headers={"Authorization": f"Bearer {CLERK_A}"},
    )
    assert resp_a2.status_code == 200
    resp_b2 = client.post(
        "/api/v1/battle/queue/join", json={"subject": "Mathematics"},
        headers={"Authorization": f"Bearer {CLERK_B}"},
    )
    assert resp_b2.status_code == 200
    body_b2 = resp_b2.json()
    assert body_b2["status"] == "matched"
    second_match_id = body_b2["match_id"]
    assert second_match_id is not None
    assert second_match_id != first_match_id

    questions = _fixed_questions()

    async def _seed_questions():
        async with session_factory() as db:
            db.add_all(questions)
            await db.commit()

    asyncio.run(_seed_questions())

    ws_url = f"/api/v1/battle/match/{second_match_id}/ws"
    with client.websocket_connect(f"{ws_url}?token={CLERK_A}") as ws_a:
        ws_a.receive_json()  # connected
        state_sync_a = ws_a.receive_json()
        assert state_sync_a["state"]["status"] == "waiting"

        with client.websocket_connect(f"{ws_url}?token={CLERK_B}") as ws_b:
            ws_b.receive_json()  # connected
            ws_b.receive_json()  # state_sync

            ws_a.send_json({"type": "ready"})
            ws_a.receive_json()
            ws_b.receive_json()

            with patch(
                "app.services.battle_gameplay_service.select_battle_questions",
                new=AsyncMock(return_value=questions),
            ):
                ws_b.send_json({"type": "ready"})
                # Not asserting on an exact event sequence/count here -- real
                # wall-clock time drives question/countdown windows, so a
                # slow test run (e.g. verbose SQL debug logging) can let the
                # progression timer coalesce past countdown_started straight
                # to match_started/question_started by the time these reads
                # happen. What actually matters -- that ready-gating on a
                # freshly-rematched pair still activates the match at all --
                # is asserted below directly against persisted DB state.
                ws_a.receive_json()
                ws_b.receive_json()

    async def _lookup_second_match_status():
        async with session_factory() as db:
            return (
                await db.execute(select(BattleMatch).where(BattleMatch.id == second_match_id))
            ).scalar_one()

    second_match = asyncio.run(_lookup_second_match_status())
    assert second_match.status in ("countdown", "active", "completed")

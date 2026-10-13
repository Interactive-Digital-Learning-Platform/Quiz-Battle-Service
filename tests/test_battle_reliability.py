"""
tests/test_battle_reliability.py
────────────────────────────────
Regression tests for the reliability fixes:

- answers lock at the reveal boundary even if the grading timer hasn't run;
- queue join rate-limits before writing anything; cancel reports Redis outages as 503;
- optional `azp` (authorized party) check on Clerk tokens;
- WebSocket: ready/forfeit errors are reported instead of dropping the socket,
  and a per-socket message throttle.
"""
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from redis.exceptions import RedisError
from sqlalchemy import func, select

from app.core import security
from app.core.config import settings
from app.models.competitive_rating import CompetitiveSubjectRating
from app.services import battle_gameplay_service, matchmaking_service
from tests.test_battle_gameplay_service import (
    _create_waiting_match,
    _make_user,
    _seed_and_activate,
    _set_started_at,
)
from tests.test_battle_ws_routes import CLERK_A, ws_env  # noqa: F401  (fixture)

Q = settings.BATTLE_QUESTION_TIME_LIMIT_SECONDS

# ─────────────────────────────────────────────────────────────────────────────
# Late answers
# ─────────────────────────────────────────────────────────────────────────────


async def test_answer_in_reveal_window_is_locked_before_grading_runs(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=user_a.id, rating_a=1000, user_b_id=user_b.id, rating_b=1000,
    )
    _, questions = await _seed_and_activate(db_session, fake_redis, match, user_a_id=user_a.id, user_b_id=user_b.id)

    # One second into the reveal phase of question 0; nothing has been graded yet.
    await _set_started_at(
        db_session, match.id, datetime.now(UTC) - timedelta(seconds=Q - settings.BATTLE_ANSWER_REVEAL_SECONDS + 1)
    )
    with pytest.raises(HTTPException) as exc:
        await battle_gameplay_service.submit_battle_answer(
            db_session, match_id=match.id, user_id=user_a.id, question_id=questions[0].id,
            selected_option="A", redis=fake_redis,
        )

    assert exc.value.status_code == 409
    assert await fake_redis.keys("*pending*") == []


# ─────────────────────────────────────────────────────────────────────────────
# Queue join / cancel
# ─────────────────────────────────────────────────────────────────────────────


async def test_rate_limited_join_writes_no_rating_row(db_session, fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "BATTLE_QUEUE_JOIN_RATE_LIMIT_MAX_CALLS", 0)
    user = await _make_user(db_session, "spammer")

    with pytest.raises(HTTPException) as exc:
        await matchmaking_service.join_queue(db_session, fake_redis, user_id=user.id, subject="Mathematics")

    assert exc.value.status_code == 429
    count = (await db_session.execute(select(func.count()).select_from(CompetitiveSubjectRating))).scalar_one()
    assert count == 0


async def test_cancel_reports_redis_outage_as_503(db_session, fake_redis, monkeypatch):
    monkeypatch.setattr(matchmaking_service, "_lua_cancel", AsyncMock(side_effect=RedisError("down")))

    with pytest.raises(HTTPException) as exc:
        await matchmaking_service.cancel_queue(db_session, fake_redis, user_id=1)

    assert exc.value.status_code == 503


# ─────────────────────────────────────────────────────────────────────────────
# azp
# ─────────────────────────────────────────────────────────────────────────────


class _Key:
    key = "k"


class _FakeJWKS:
    def get_signing_key_from_jwt(self, token):
        return _Key()


@pytest.fixture
def decode_returns(monkeypatch):
    monkeypatch.setattr(security, "_jwks_client", _FakeJWKS())

    def set_payload(payload: dict):
        monkeypatch.setattr(security.jwt, "decode", lambda *a, **k: payload)

    return set_payload


async def test_azp_check(decode_returns, monkeypatch):
    monkeypatch.setattr(settings, "CLERK_AUTHORIZED_PARTIES", "")
    decode_returns({"sub": "u", "azp": "https://evil.example"})
    assert (await security._verify_token("t"))["sub"] == "u"  # not configured -> no check

    monkeypatch.setattr(settings, "CLERK_AUTHORIZED_PARTIES", "https://app.example, http://localhost:8081")
    with pytest.raises(HTTPException) as exc:
        await security._verify_token("t")
    assert exc.value.status_code == 401

    decode_returns({"sub": "u", "azp": "http://localhost:8081"})
    assert (await security._verify_token("t"))["sub"] == "u"
    decode_returns({"sub": "u"})  # native app tokens may carry no azp
    assert (await security._verify_token("t"))["sub"] == "u"


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket
# ─────────────────────────────────────────────────────────────────────────────


def test_ws_ready_error_is_reported_and_socket_stays_open(ws_env, monkeypatch):  # noqa: F811
    async def missing(*a, **k):
        raise HTTPException(404, "Battle match not found.")

    monkeypatch.setattr(battle_gameplay_service, "mark_ready", missing)

    with ws_env.client.websocket_connect(ws_env.url(CLERK_A)) as ws:
        ws.receive_json()  # connected
        ws.receive_json()  # state_sync
        ws.send_json({"type": "ready"})
        assert ws.receive_json() == {"type": "error", "detail": "Battle match not found."}

        ws.send_text("not json")  # still being served
        assert ws.receive_json()["type"] == "error"


def test_ws_message_flood_is_throttled(ws_env, monkeypatch):  # noqa: F811
    monkeypatch.setattr(settings, "BATTLE_WS_MESSAGE_RATE_LIMIT_MAX_CALLS", 2)
    monkeypatch.setattr(settings, "BATTLE_WS_MESSAGE_RATE_LIMIT_WINDOW_SECONDS", 60.0)

    with ws_env.client.websocket_connect(ws_env.url(CLERK_A)) as ws:
        ws.receive_json()
        ws.receive_json()
        details = []
        for _ in range(3):
            ws.send_text("{}")
            details.append(ws.receive_json()["detail"])

    assert details[0].startswith("Malformed") and details[1].startswith("Malformed")
    assert details[2] == "Too many messages — slow down."

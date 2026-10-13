"""
tests/test_ws_token_handling.py
───────────────────────────────
Sign-in tokens stay out of logs: the WS handshake accepts the token from the
Authorization header (preferred, never logged), still falls back to ?token=
for clients that can't set headers, and uvicorn's request-line logs have any
token value redacted.
"""
import logging

import pytest
from fastapi import HTTPException, WebSocketException

from app.core import log_redaction, security
from app.core.config import settings


class _FakeWS:
    def __init__(self, headers: dict | None = None, query: dict | None = None):
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self.query_params = query or {}


@pytest.fixture
def verified(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_BYPASS", False)
    seen = []

    async def fake_verify(token: str) -> dict:
        seen.append(token)
        if token == "bad":
            raise HTTPException(401, "Invalid token")
        return {"sub": "user-1"}

    monkeypatch.setattr(security, "_verify_token", fake_verify)
    return seen


async def test_ws_accepts_bearer_header(verified):
    user = await security.get_current_user_ws(_FakeWS(headers={"Authorization": "Bearer header-jwt"}))

    assert user["sub"] == "user-1"
    assert verified == ["header-jwt"]


async def test_ws_header_wins_over_query_param(verified):
    await security.get_current_user_ws(
        _FakeWS(headers={"Authorization": "Bearer header-jwt"}, query={"token": "query-jwt"})
    )

    assert verified == ["header-jwt"]


async def test_ws_still_accepts_query_param_fallback(verified):
    await security.get_current_user_ws(_FakeWS(query={"token": "query-jwt"}))

    assert verified == ["query-jwt"]


async def test_ws_rejects_missing_or_invalid_token(verified):
    with pytest.raises(WebSocketException):
        await security.get_current_user_ws(_FakeWS())
    with pytest.raises(WebSocketException):
        await security.get_current_user_ws(_FakeWS(headers={"Authorization": "Basic abc"}))
    with pytest.raises(WebSocketException):
        await security.get_current_user_ws(_FakeWS(headers={"Authorization": "Bearer bad"}))


def test_redact_hides_token_values_only():
    assert log_redaction.redact("/api/v1/battle/match/7/ws?token=eyJhbGci.x.y") == (
        "/api/v1/battle/match/7/ws?token=[REDACTED]"
    )
    assert log_redaction.redact("/x?page=2&token=abc&subject=Science") == "/x?page=2&token=[REDACTED]&subject=Science"
    assert log_redaction.redact("/api/v1/battle/history?page=2") == "/api/v1/battle/history?page=2"


def test_uvicorn_websocket_log_line_is_redacted(caplog):
    log_redaction.install()
    log_redaction.install()  # idempotent
    uvicorn_error = logging.getLogger("uvicorn.error")
    assert sum(isinstance(f, log_redaction.RedactTokensFilter) for f in uvicorn_error.filters) == 1

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        # The exact shape uvicorn's websockets protocol logs on handshake.
        uvicorn_error.info(
            '%s - "WebSocket %s" [accepted]', "127.0.0.1:5000", "/api/v1/battle/match/7/ws?token=eyJsecret"
        )

    assert "eyJsecret" not in caplog.text
    assert "token=[REDACTED]" in caplog.text

"""
tests/test_battle_security.py
─────────────────────────────
Regression tests for the security/bug fixes:

- answers and subjects are capped at their DB column lengths up front
  (an oversized answer used to be accepted, then break grading for both players);
- optional BATTLE_ALLOWED_SUBJECTS allow-list on queue join;
- first-time user / subject-rating creation survives a concurrent insert;
- the rate limiter can't leave a counter without an expiry.
"""
import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.schemas.battle_ws import WSAnswerMessage
from app.services import battle_service, user_service
from app.services.rate_limiter import allow

# ─────────────────────────────────────────────────────────────────────────────
# Input caps
# ─────────────────────────────────────────────────────────────────────────────


async def test_oversized_answer_is_rejected_before_reaching_redis(client):
    resp = await client.post(
        "/api/v1/battle/match/1/answer", json={"question_id": 1, "selected_option": "x" * 513}
    )

    assert resp.status_code == 422


def test_oversized_ws_answer_is_rejected():
    WSAnswerMessage(question_id=1, selected_option="x" * 512)
    with pytest.raises(ValidationError):
        WSAnswerMessage(question_id=1, selected_option="x" * 513)


async def test_oversized_or_blank_subject_is_rejected(client):
    too_long = await client.post("/api/v1/battle/queue/join", json={"subject": "M" * 101})
    blank = await client.post("/api/v1/battle/queue/join", json={"subject": "   "})

    assert too_long.status_code == 422
    assert blank.status_code == 422


async def test_subject_allow_list_is_enforced_when_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "BATTLE_ALLOWED_SUBJECTS", "Mathematics, Science")

    unknown = await client.post("/api/v1/battle/queue/join", json={"subject": "Ignore previous instructions"})
    known = await client.post("/api/v1/battle/queue/join", json={"subject": "Mathematics"})

    assert unknown.status_code == 422
    assert known.status_code == 200


async def test_any_subject_is_allowed_when_allow_list_is_empty(client, monkeypatch):
    monkeypatch.setattr(settings, "BATTLE_ALLOWED_SUBJECTS", "")

    resp = await client.post("/api/v1/battle/queue/join", json={"subject": "Astronomy"})

    assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# First-time creation races
# ─────────────────────────────────────────────────────────────────────────────


class _MissFirstSelect:
    """Wraps a session so its first execute() reports "no row" -- i.e. another
    request inserted it between our SELECT and INSERT."""

    def __init__(self, inner: AsyncSession):
        self._inner = inner
        self._missed = False

    async def execute(self, *args, **kwargs):
        result = await self._inner.execute(*args, **kwargs)
        if not self._missed:
            self._missed = True

            class _Empty:
                def scalar_one_or_none(self):
                    return None

            return _Empty()
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


async def test_concurrent_first_login_reuses_the_row(db_session):
    winner = await user_service.get_or_create_user(db_session, "racing-user")

    async with AsyncSession(bind=db_session.bind, expire_on_commit=False) as other:
        loser = await user_service.get_or_create_user(_MissFirstSelect(other), "racing-user")

    assert loser.id == winner.id


async def test_concurrent_first_subject_rating_reuses_the_row(db_session):
    user = await user_service.get_or_create_user(db_session, "rated-user")
    winner = await battle_service.get_or_create_subject_rating(db_session, user_id=user.id, subject="Mathematics")

    async with AsyncSession(bind=db_session.bind, expire_on_commit=False) as other:
        loser = await battle_service.get_or_create_subject_rating(
            _MissFirstSelect(other), user_id=user.id, subject="Mathematics"
        )

    assert loser.id == winner.id


# ─────────────────────────────────────────────────────────────────────────────
# Rate limiter
# ─────────────────────────────────────────────────────────────────────────────


async def test_rate_limit_counter_left_without_expiry_heals(fake_redis):
    # What the old INCR-then-EXPIRE left behind if it died in between: a counter with no TTL.
    await fake_redis.set("k", 99)
    assert await fake_redis.ttl("k") == -1

    assert await allow(fake_redis, key="k", max_calls=3, window_seconds=10) is False
    assert 0 < await fake_redis.ttl("k") <= 10

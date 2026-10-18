"""
tests/test_diagnostics.py
─────────────────────────
- The event-loop watchdog warns when the loop was blocked, and stays quiet
  otherwise.
- A slow (or cut-off) answer save logs how long each step took.
"""
import asyncio
import contextlib
import logging
import time

from app.core.config import settings
from app.core.loop_watchdog import run_loop_watchdog
from app.services import battle_gameplay_service
from tests.test_battle_gameplay_service import _answer, _new_active_match


async def _watch(monkeypatch, *, block_for: float) -> None:
    monkeypatch.setattr(settings, "LOOP_WATCHDOG_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(settings, "LOOP_WATCHDOG_WARN_SECONDS", 0.2)
    task = asyncio.create_task(run_loop_watchdog())
    await asyncio.sleep(0.1)
    time.sleep(block_for)  # noqa: ASYNC251 -- deliberately blocks the whole loop
    await asyncio.sleep(0.15)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_watchdog_warns_when_the_event_loop_was_blocked(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="app.core.loop_watchdog"):
        await _watch(monkeypatch, block_for=0.4)
    assert any("Event loop was blocked" in r.getMessage() for r in caplog.records)


async def test_watchdog_is_quiet_when_nothing_blocks(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="app.core.loop_watchdog"):
        await _watch(monkeypatch, block_for=0)
    assert not caplog.records


async def test_slow_save_logs_its_step_timings(db_session, fake_redis, monkeypatch, caplog):
    monkeypatch.setattr(settings, "BATTLE_SLOW_SAVE_LOG_SECONDS", 0.0)
    user_a, _, match, questions = await _new_active_match(db_session, fake_redis)

    with caplog.at_level(logging.WARNING, logger=battle_gameplay_service.__name__):
        await _answer(db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="A")

    message = next(r.getMessage() for r in caplog.records if "Slow battle answer save" in r.getMessage())
    assert "outcome=ok" in message
    for step in ("load match", "lock match", "insert answer + commit", "load next question"):
        assert step in message


async def test_save_cut_off_by_a_timeout_logs_the_step_it_was_stuck_in(db_session, fake_redis, monkeypatch, caplog):
    user_a, _, match, questions = await _new_active_match(db_session, fake_redis)

    async def stalled_allowance(*args, **kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(battle_gameplay_service, "network_allowance_seconds", stalled_allowance)

    with caplog.at_level(logging.WARNING, logger=battle_gameplay_service.__name__), contextlib.suppress(TimeoutError):
        async with asyncio.timeout(0.2):
            await _answer(
                db_session, fake_redis, match, user_id=user_a.id, question_id=questions[0].id, selected_option="A",
            )

    message = next(r.getMessage() for r in caplog.records if "Slow battle answer save" in r.getMessage())
    assert "outcome=CancelledError" in message
    assert "network allowance (Redis)" in message and "(unfinished)" in message

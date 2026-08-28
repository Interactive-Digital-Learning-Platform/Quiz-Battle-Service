"""
tests/test_battle_connection_manager.py
─────────────────────────────────────────
Unit tests for BattleConnectionManager._run_progression_timer_with_respawn --
the supervisor that restarts the progression timer factory if it dies from
an unhandled exception, instead of silently freezing that match's question
progression forever (see its docstring and app/api/routes/battle_ws.py's
_progression_timer, which no longer swallows its own exceptions).

Exercises the supervisor directly against a fake factory rather than the
real _progression_timer -- these tests are about the restart POLICY (retry
on exception, never retry on a clean return or once stop_event is set), not
about battle gameplay itself.
"""
import asyncio

import pytest

from app.core.config import settings
from app.services.battle_connection_manager import BattleConnectionManager


async def test_clean_return_is_never_respawned():
    manager = BattleConnectionManager()
    stop_event = asyncio.Event()
    calls = 0

    async def factory(_match_id: int, _stop_event: asyncio.Event) -> None:
        nonlocal calls
        calls += 1
        return

    await manager._run_progression_timer_with_respawn(1, stop_event, factory)
    assert calls == 1


async def test_exception_triggers_a_restart(monkeypatch):
    # Zero delay so the test doesn't actually wait out the real
    # BATTLE_PROGRESSION_TIMER_RESPAWN_DELAY_SECONDS backoff.
    monkeypatch.setattr(settings, "BATTLE_PROGRESSION_TIMER_RESPAWN_DELAY_SECONDS", 0.0)
    manager = BattleConnectionManager()
    stop_event = asyncio.Event()
    calls = 0

    async def factory(_match_id: int, _stop_event: asyncio.Event) -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("transient DB hiccup")
        return  # third attempt succeeds cleanly -- supervisor stops here

    await manager._run_progression_timer_with_respawn(1, stop_event, factory)
    assert calls == 3


async def test_exception_after_stop_requested_is_not_respawned(monkeypatch):
    monkeypatch.setattr(settings, "BATTLE_PROGRESSION_TIMER_RESPAWN_DELAY_SECONDS", 0.0)
    manager = BattleConnectionManager()
    stop_event = asyncio.Event()
    stop_event.set()
    calls = 0

    async def factory(_match_id: int, _stop_event: asyncio.Event) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("dies right as the last connection is closing")

    await manager._run_progression_timer_with_respawn(1, stop_event, factory)
    assert calls == 1  # never retried -- shutting down anyway


async def test_cancelled_error_propagates_without_being_treated_as_a_failure():
    manager = BattleConnectionManager()
    stop_event = asyncio.Event()

    async def factory(_match_id: int, _stop_event: asyncio.Event) -> None:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await manager._run_progression_timer_with_respawn(1, stop_event, factory)

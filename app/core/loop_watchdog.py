"""
app/core/loop_watchdog.py
─────────────────────────
Diagnostic: warns when this process's event loop was blocked -- i.e. some
synchronous work (or the whole process being paused) stopped every other
request, WebSocket and background task from running. Sleeps a short, fixed
interval and measures how much later than that it actually woke up.

A slow DB/network round trip does NOT show up here (awaiting it leaves the
loop free) -- that's what battle_gameplay_service's slow-save step log is
for. Together they tell "our process froze" apart from "the DB was slow".
"""
import asyncio
import logging
import time

from app.core.config import settings

logger = logging.getLogger(__name__)


async def run_loop_watchdog() -> None:
    interval = settings.LOOP_WATCHDOG_INTERVAL_SECONDS
    while True:
        started = time.monotonic()
        await asyncio.sleep(interval)
        lag = time.monotonic() - started - interval
        if lag >= settings.LOOP_WATCHDOG_WARN_SECONDS:
            logger.warning(
                "Event loop was blocked for %.1fs — nothing else in this process ran meanwhile "
                "(synchronous work, or the process/machine was paused).",
                lag,
            )

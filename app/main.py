import asyncio
import contextlib
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import app.models  # noqa: F401 -- registers model metadata on Base
from app.api.routes import battle, battle_ws
from app.core.config import settings
from app.core.database import Base, engine
from app.core.redis_client import redis_client
from app.services import battle_gameplay_service, matchmaking_service

logging.basicConfig(
    level=logging.DEBUG if settings.ENVIRONMENT == "development" else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Online 1v1 competitive quiz battle microservice — matchmaking, Elo "
        "rating, subject leaderboards, and live WebSocket gameplay. Shares "
        "its Postgres database with Personalized-Quiz-Service (real FKs "
        "into users/questions) but is deployed and versioned independently."
    ),
    version="0.1.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.ENVIRONMENT == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Starting %s [%s]", settings.PROJECT_NAME, settings.ENVIRONMENT)

    async with engine.begin() as conn:
        # create_all only creates tables that don't exist yet -- it never
        # adds a column to a table that's already there, and it's a no-op
        # for users/questions (owned by Personalized-Quiz-Service, already
        # exist). If you add a field to a model this service DOES own, you
        # still need a real migration (see alembic/) for it to show up on a
        # database that's already been through this once.
        logger.info("Running create_all — creating missing tables...")
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables are ready.")

    # Redis outlives this process across restarts -- without this, anyone
    # who was "queued" before a restart (routine in local dev, possible any
    # time in deployment) stays queued in Redis with no client left tracking
    # it, and could get silently matched by the sweep loop below the moment
    # it starts. Runs before that loop starts so nothing can claim a
    # leftover entry first. Does not touch "in_match" state -- those point
    # at real Postgres rows that survive this restart same as everything
    # else in Postgres, and have their own reconciliation (see
    # matchmaking_service.run_in_match_reconcile_once).
    await matchmaking_service.clear_all_queue_state(redis_client)

    app.state.matchmaking_sweep_task = asyncio.create_task(
        matchmaking_service.start_matchmaking_sweep_loop()
    )
    app.state.stale_match_reaper_task = asyncio.create_task(
        battle_gameplay_service.start_stale_match_reaper_loop()
    )
    app.state.in_match_reconciler_task = asyncio.create_task(
        matchmaking_service.start_in_match_reconciler_loop()
    )


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("Shutting down — disposing DB engine...")

    for task_attr in ("matchmaking_sweep_task", "stale_match_reaper_task", "in_match_reconciler_task"):
        task = getattr(app.state, task_attr, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    await redis_client.aclose()
    await engine.dispose()
    logger.info("Shutdown complete.")


API_PREFIX = "/api/v1"

app.include_router(battle.router, prefix=API_PREFIX)
app.include_router(battle_ws.router, prefix=API_PREFIX)


@app.get("/health", tags=["Health"], summary="Health check")
async def health_check() -> dict:
    return {"status": "ok", "service": settings.PROJECT_NAME}

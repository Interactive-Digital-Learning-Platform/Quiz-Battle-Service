"""
tests/conftest.py
──────────────────
Shared pytest fixtures for the FastAPI test suite.

Each test gets a fresh, isolated in-memory SQLite database (via StaticPool so
every connection checkout shares the same underlying in-memory DB) and an
httpx.AsyncClient wired directly to the app in-process — no real network
call, no real Postgres, no real Groq/Clerk calls. This includes this
service's own local User/Question mirror tables (created fresh per test, no
real cross-service data needed here).

`get_db` is overridden to use this test database. `get_current_user` is
overridden to simulate an authenticated user, the same way AUTH_BYPASS does
in development — this does not touch or replace AUTH_BYPASS itself
(app/core/security.py is untouched), it just avoids needing a real Clerk JWT
in tests.
"""
import fakeredis
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

# Import all models so their metadata is registered on Base before create_all.
# NOTE: must use `from app import models`, not `import app.models` — the
# latter rebinds the local name `app` to the top-level package, clobbering
# the `from app.main import app` FastAPI instance imported below.
from app import models as _app_models  # noqa: F401
from app.core.database import Base, get_db
from app.core.redis_client import get_redis
from app.core.security import get_current_user
from app.main import app
from app.services import battle_gameplay_service, battle_question_service, matchmaking_service

TEST_CLERK_ID = "test-user"


@pytest.fixture(autouse=True)
def _no_real_pool_replenish(monkeypatch):
    # battle_question_service._kickoff_pool_replenish spawns a fire-and-
    # forget asyncio.create_task() that opens its OWN DB session via
    # AsyncSessionLocal (bound to settings.DATABASE_URL) -- never the
    # per-test isolated in-memory SQLite engine `db_session` sets up. Left
    # unpatched, it would either hit the wrong database or run after the
    # test's event loop/session has already torn down. Tests that want to
    # assert replenish-triggering behavior patch this back themselves.
    monkeypatch.setattr(battle_question_service, "_kickoff_pool_replenish", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _no_real_matchmaking_sweep_loop(monkeypatch):
    # start_matchmaking_sweep_loop() is only ever started for real in
    # app/main.py's on_startup, which httpx.ASGITransport (used by the
    # `client` fixture below) never triggers -- it only ever sends "http"
    # scope calls, never "lifespan" (confirmed by reading its source). This
    # is a defensive neutralization in case that ever changes.
    async def _noop() -> None:
        return None

    monkeypatch.setattr(matchmaking_service, "start_matchmaking_sweep_loop", _noop)


@pytest.fixture(autouse=True)
def _no_real_stale_match_reaper_loop(monkeypatch):
    # Same defensive neutralization as _no_real_matchmaking_sweep_loop above.
    async def _noop() -> None:
        return None

    monkeypatch.setattr(battle_gameplay_service, "start_stale_match_reaper_loop", _noop)


@pytest_asyncio.fixture
async def fake_redis():
    """In-memory Redis double (incl. Lua/EVAL support via the `lua` extra) --
    matchmaking_service never touches a real Redis server in tests.
    """
    r = fakeredis.FakeAsyncRedis(decode_responses=True)
    yield r
    await r.aclose()


@pytest_asyncio.fixture
async def db_session():
    """A fresh in-memory SQLite DB per test, with all tables created."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        async with session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    async def override_get_current_user():
        return {"sub": TEST_CLERK_ID, "email": "test-user@example.com"}

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user

    async with session_factory() as session:
        yield session

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_session, fake_redis):
    """An httpx.AsyncClient driving the app in-process, sharing db_session's
    DB and fake_redis's Redis double.
    """
    app.dependency_overrides[get_redis] = lambda: fake_redis

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

from collections.abc import AsyncGenerator

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

database_url = settings.DATABASE_URL
database_url_obj = make_url(database_url)

engine_kwargs: dict[str, object] = {
    "future": True,
    "echo": (settings.ENVIRONMENT == "development"),
    "pool_pre_ping": True,
}

if database_url_obj.drivername.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs["connect_args"] = {"ssl": True}
    engine_kwargs["pool_size"] = settings.DB_POOL_SIZE
    engine_kwargs["max_overflow"] = settings.DB_MAX_OVERFLOW

engine: AsyncEngine = create_async_engine(database_url, **engine_kwargs)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    # FastAPI calls this per-request via Depends(get_db) — the `async with`
    # takes care of closing (and rolling back on error) once the request ends.
    # Every call site types its own `db: AsyncSession = Depends(get_db)`
    # explicitly, so this function's own return type doesn't need to lie
    # about being AsyncSession for that injection to work.
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise

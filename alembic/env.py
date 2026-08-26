import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Import all models so their metadata is registered on Base before autogenerate
# compares against it, and pull the real DB URL from app settings (.env) rather
# than duplicating it in alembic.ini.
from app import models as _app_models  # noqa: F401
from app.core.config import settings
from app.core.database import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# This service shares its Postgres database with Personalized-Quiz-Service,
# which owns every table except the five below. Autogenerate must never
# propose changes to anything else -- not just users/questions (mapped here
# only via thin mirror models, for FK purposes), but also every other
# Personalized-Quiz-Service-owned table (quiz_sessions, analytics, etc.)
# that simply isn't declared in this service's own Base.metadata at all and
# would otherwise show up as a spurious "drop table" candidate. Allowlist,
# not denylist, so a table this service doesn't yet know about can never
# slip through.
_OWNED_TABLES = frozenset(
    {
        "competitive_subject_ratings",
        "battle_matches",
        "battle_participants",
        "battle_match_questions",
        "battle_answers",
        "battle_ai_generation_events",
    }
)


def include_object(object, name, type_, reflected, compare_to) -> bool:
    if type_ == "table":
        return name in _OWNED_TABLES
    return True


# Distinct version-tracking table so this service's migration bookkeeping
# doesn't collide with Personalized-Quiz-Service's own `alembic_version`
# table in the same physical database.
VERSION_TABLE = "alembic_version_quiz_online"


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        version_table=VERSION_TABLE,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=include_object,
        version_table=VERSION_TABLE,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.

    """

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

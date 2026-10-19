# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.1 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    ENVIRONMENT=production \
    RUN_MIGRATIONS=true \
    WEB_CONCURRENCY=1

RUN groupadd --system --gid 10001 appgroup \
    && useradd --system --uid 10001 --gid appgroup --no-create-home appuser

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appgroup app ./app
COPY --chown=appuser:appgroup alembic ./alembic
COPY --chown=appuser:appgroup alembic.ini ./

USER appuser

# Nginx connects to host port 8005; publish with 8005:8000.
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).close()"]

# Applies pending alembic migrations first (this service's own version table, alembic_version_quiz_online):
# create_all at startup only creates missing tables, never the new columns a migration adds. Set
# RUN_MIGRATIONS=false when running several copies and run `alembic upgrade head` once yourself.
# `exec` hands the shell's PID to Uvicorn, so stop signals still reach it directly; Uvicorn reads WEB_CONCURRENCY.
CMD ["sh", "-c", "if [ \"$RUN_MIGRATIONS\" = \"true\" ]; then alembic upgrade head; fi && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]

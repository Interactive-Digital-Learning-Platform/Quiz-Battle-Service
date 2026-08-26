# ── Build Stage ───────────────────────────────────────────────────────────────
# Use official slim Python 3.12 image to keep the image lightweight
FROM python:3.12-slim

# Pull the uv binary from Astral's distroless image rather than installing it
# via pip — keeps it out of the final dependency graph entirely.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set environment variables to prevent Python from writing .pyc files
# and to ensure stdout/stderr is unbuffered (important for Docker logging)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Create a non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Set the working directory inside the container
WORKDIR /app

# ── Install Dependencies ───────────────────────────────────────────────────────
# Copy only the lockfile + project metadata first to leverage Docker layer
# caching — this layer only rebuilds when dependencies actually change.
# --no-dev skips pytest/pytest-asyncio, which the runtime image doesn't need.
COPY pyproject.toml uv.lock .
RUN uv sync --frozen --no-dev

# ── Copy Application Code ──────────────────────────────────────────────────────
COPY . .

# Change ownership of all files to the non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Put the venv uv created ahead of the system Python on PATH
ENV PATH="/app/.venv/bin:$PATH"

# ── Runtime ────────────────────────────────────────────────────────────────────
# Expose the port that uvicorn will listen on. The host-side mapping (this
# service runs on host port 8005 behind the gateway) happens in
# docker-compose.yml / the run command, not here.
EXPOSE 8000

# Start FastAPI with uvicorn
# --host 0.0.0.0 makes the server accessible from outside the container.
# Worker count comes from WEB_CONCURRENCY (defaults to 1) rather than a fixed
# number baked into the image, so it can be tuned per-deployment without a
# rebuild.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY:-1}"]

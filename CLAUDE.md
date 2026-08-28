# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

FastAPI microservice for the online 1v1 competitive quiz battle feature —
matchmaking, Elo rating, subject leaderboards, and live WebSocket gameplay.
Extracted from `Personalized-Quiz-Service` (a sibling repo) to be deployed
independently, reached through the shared nginx API gateway in
`Deployment-Infrastructure/` at `/api/battle/...`.

## Commands

- `uv sync` — install dependencies (add `--no-dev` to skip test-only deps)
- `uv run uvicorn app.main:app --reload --port 8005` — run the dev server
- `uv run pytest -q` — run the test suite
- `uv run python scripts/verify_battle_live.py` — live end-to-end verification
  (real server, real Redis, two real WebSocket clients) — run this after
  touching the WS route, the connection manager, or matchmaking's Redis state;
  a `TestClient`-based pytest test structurally cannot catch the races and
  cross-process bugs this catches.
- `uv run alembic upgrade head` — apply migrations (uses its own
  `alembic_version_quiz_online` tracking table — see below)

## Architecture

### Shared database, independent ownership

This service shares its Postgres database with `Personalized-Quiz-Service`
(`DATABASE_URL` must point at the same instance) but does not own or migrate
`users`/`questions` — those tables belong to `Personalized-Quiz-Service`.
`app/models/user.py`/`app/models/question.py` are thin local mirrors mapped
onto those same tables (real FKs, no cross-service HTTP calls needed), kept
in sync by hand with the upstream schema — if `Personalized-Quiz-Service`
changes either table's columns, update the mirror here too.

This service owns and migrates exactly five tables: `battle_matches`,
`battle_participants`, `battle_match_questions`, `battle_answers`,
`competitive_subject_ratings`. `alembic/env.py` sets a distinct
`version_table` (`alembic_version_quiz_online`) so its migration history
doesn't collide with `Personalized-Quiz-Service`'s own `alembic_version`
table in the same physical database, and an `include_object` filter
excludes `users`/`questions` from autogenerate diffing entirely.

### Reused primitives copied from Personalized-Quiz-Service

A few small, pure or near-pure pieces of logic are duplicated here rather
than imported cross-repo (there's no shared package boundary between two
independent sibling repos): `app/services/text_similarity.py`
(near-duplicate question detection), `app/services/user_service.py`
(`get_or_create_user`), `app/services/question_fingerprint.py`, and
`app/services/groq_service.py` (own Groq client/retry logic — used only by
`battle_question_service.py`'s background pool-replenish task, never in the
live match-activation path; see that module's docstring). Keep these in
sync by hand if the upstream logic changes.

### Battle lifecycle

`BattleMatch.status`: `"waiting"` → `"countdown"` → `"active"` →
`"completed"`/`"cancelled"`. Forfeits reuse `"completed"` with
`BattleParticipant.result="forfeit"` — no distinct status. See
`app/services/battle_gameplay_service.py` for the full state machine
(question selection, scoring, Elo application, disconnect/reconnect grace
period, stale-match reaping) and `app/services/matchmaking_service.py` for
Redis-backed queue matching (Lua-script atomic claims, rating-window
widening by wait time).

### Realtime

`app/services/battle_connection_manager.py` is the per-process in-memory WS
connection registry; cross-process event delivery goes through Redis
pub/sub (`battle:match_events:{match_id}`). Time-driven events (question
progression, match completion) need no cross-process coordination — every
connected process independently derives them from shared Postgres state
(`started_at` + elapsed time). See that module's docstring for the full
delivery model.

### Validation and types

Request/response shapes are Pydantic schemas in `app/schemas/battle.py`
(REST) and `app/schemas/battle_ws.py` (WS inbound/outbound), parsed at the
API boundary.

### Path conventions

Routes are mounted at `/api/v1/battle/...` (REST) and
`WS /api/v1/battle/match/{match_id}/ws` internally, unchanged from when this
code lived inside `Personalized-Quiz-Service` — only the *gateway* prefix in
front of it changed (`/api/battle/` in `Deployment-Infrastructure/nginx.conf`).

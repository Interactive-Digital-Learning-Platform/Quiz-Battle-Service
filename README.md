# Quiz-Battle-Service (FastAPI)

Standalone microservice for the online 1v1 competitive quiz battle feature —
matchmaking, Elo rating, subject leaderboards, and live WebSocket gameplay.

Extracted from `Personalized-Quiz-Service`. Shares that service's Postgres
database (real foreign keys into its `users`/`questions` tables — see
"Shared database" below) but is deployed, versioned, and migrated
independently.

Run locally (requires [uv](https://docs.astral.sh/uv/)):

```
uv sync
uv run uvicorn app.main:app --reload --port 8005
```

`uv sync` creates `.venv` and installs everything pinned in `uv.lock` (add
`--no-dev` to skip test-only dependencies). To run a command inside the venv
without the `uv run` prefix, activate it the usual way:
`source .venv/bin/activate`.

Configure via environment variables (or `.env`, see `.env.example`):
- `DATABASE_URL` (asyncpg format) — **must point at the same Postgres/Neon
  instance as `Personalized-Quiz-Service`**; this service reads/writes its
  `users`/`questions` tables directly.
- `GROQ_API_KEY` and `GROQ_MODEL` — used to top up the shared question pool
  when it runs short for a given (subject, difficulty).
- `CLERK_JWKS_URL`, `CLERK_ISSUER`, `CLERK_AUDIENCE` (optional) — same Clerk
  project as `Personalized-Quiz-Service`, since both services authenticate
  the same users.
- `REDIS_URL` (default `redis://localhost:6379/0`) — matchmaking queues,
  ready state, disconnect presence, WS pub/sub. `docker compose up redis`
  starts a local instance.
- `BATTLE_PRESENCE_TTL_SECONDS` (default `60`) — TTL for the online-presence
  heartbeat key (`battle:presence:{user_id}`, see below).

### Redis keys and TTL strategy

All keys are prefixed `battle:`. Every TTL below is a *safety net*, not the
primary lifecycle mechanism — the app clears keys explicitly on the normal
success path (cancel, match completion, forfeit); TTLs only cover a crashed
process or a client that vanished mid-flow. See
`app/services/matchmaking_service.py` and
`app/services/battle_connection_manager.py`'s module docstrings for the full
design rationale.

| Key | Type | TTL | Purpose |
| --- | --- | --- | --- |
| `battle:user_state:{user_id}` | HASH | `BATTLE_MATCHMAKING_QUEUE_ENTRY_TTL_SECONDS` while `"queued"`, `BATTLE_MATCHMAKING_CLAIM_TTL_SECONDS` while `"matching"`, none while `"in_match"` | Single source of truth for one user's matchmaking state |
| `battle:subject_queue:{subject}` | ZSET (member=user_id, score=queued_at) | none (members removed explicitly; stale members opportunistically pruned) | FIFO candidate pool per subject |
| `battle:queue_subjects` | SET | none | Subjects with ≥1 waiting user, so the sweep doesn't have to scan every subject ever seen |
| `battle:presence:{user_id}` | STRING | `BATTLE_PRESENCE_TTL_SECONDS`, refreshed on every `get_queue_status` call | Best-effort online-presence heartbeat (`mark_online`/`mark_offline`/`is_online` in `matchmaking_service.py`) — independent of matchmaking state; a user can be online without queuing |
| `battle:ready:{match_id}` | HASH | `BATTLE_READY_TTL_SECONDS` | Which participants have called `ready` before the countdown starts |
| `battle:disconnect_deadline:{match_id}:{user_id}` | STRING | grace-period length | In-match disconnect/forfeit-grace tracking |
| `battle:match_events:{match_id}` | pub/sub channel | n/a | Cross-process WS event delivery |
| `battle:ratelimit:*` | STRING (counter) | fixed window length | Queue-join / WS-connect rate limiting |

## Shared database

This service does **not** own or migrate the `users`/`questions` tables —
those belong to `Personalized-Quiz-Service`. It maps thin local models onto
them (`app/models/user.py`, `app/models/question.py`) purely to establish
real foreign keys and to let `battle_question_service` read/insert
`Question` rows. Alembic here uses its own migration-tracking table
(`alembic_version_quiz_online`, configured in `alembic/env.py`) so its
history doesn't collide with `Personalized-Quiz-Service`'s own
`alembic_version` table in the same physical database, and an
`include_object` filter stops autogenerate from ever proposing changes to
`users`/`questions`.

## Competitive battles (1v1)

Ranked 1v1 matchmaking/leaderboards/live battles live under `/api/v1/battle`
(REST: queue, leaderboard, profile, history, match state/answer/ready/forfeit)
plus a realtime layer at `WS /api/v1/battle/match/{match_id}/ws?token=<clerk_jwt>`.
Both REST and WebSocket read/write the same `battle_gameplay_service` state
machine and require Redis — see `app/services/matchmaking_service.py` and
`app/services/battle_connection_manager.py` for the key schema and delivery
model.

Reachable through the shared API gateway (`Deployment-Infrastructure/`) at
`/api/battle/...` on `http://localhost:8080`.

### WebSocket protocol

Matchmaking itself (join/cancel/status) is REST-only — the socket only opens
once a `match_id` already exists, scoped to that one match. Auth reuses the
same Clerk JWT verification as HTTP (`app/core/security.py`'s shared
`_verify_token`, `AUTH_BYPASS` respected in development); the token travels
as a `?token=` query param since native WS clients can't set a custom
handshake header, and a non-participant or missing/invalid token is rejected
*before* `accept()`.

Every event is a flat `{"type": "...", ...fields}` object, one Pydantic model
per event name (`app/schemas/battle_ws.py`) — both inbound client messages
and outbound server events are validated against these, never sent/parsed as
ad-hoc dicts.

Gameplay is self-paced: both players get the same 10 questions in the same
order and move on with Next (`answer`) or `skip` at their own pace; the whole
match runs on one 5-minute clock (`BATTLE_MATCH_DURATION_SECONDS`) and ends
early once both have finished. Each answer is saved before the player's next
question is sent; correct = 100 points plus a speed bonus (≤5s +30, ≤10s
+20, ≤15s +10) timed on the server clock from when that question was handed
out, minus a capped network allowance measured with `ping`/`pong`. Highest
score wins. Right/wrong is only revealed in the end-of-match review
(`review` on `GET .../state` once completed).

Inbound (client → server): `ready`, `answer` (`question_id`, `selected_option`),
`skip` (`question_id`), `pong` (`nonce`), `forfeit`.

Outbound (server → client): `connected`, `state_sync` (full match state, sent
on every connect/reconnect), `player_ready`, `countdown_started`,
`match_started` (with `ends_at`/`server_time` for the match clock),
`next_question` (to that player only — question 0 at the start, then after
each saved answer/skip; a repeated Next/Skip just gets it again),
`player_progress` (broadcast after each saved answer/skip — how far a player
is, never whether they were right), `ping`, `player_disconnected`,
`player_reconnected`, `match_finished`, `match_cancelled`, `error`. There is
no separate `queue_joined`/`opponent_found`/`score_updated` event — those are
REST response fields instead (`POST /queue/join`, `GET /queue/status`).

### Reconnect flow

1. An unexpected socket close (not an explicit `forfeit` message) marks the
   player disconnected in Redis (`battle:disconnect_deadline:{match_id}:{user_id}`,
   TTL = `BATTLE_DISCONNECT_GRACE_PERIOD_SECONDS`, default 15s) and publishes
   `player_disconnected` (with the deadline) to the opponent. The match stays
   alive — status is untouched.
2. If the player reconnects (same match, same token) before the deadline,
   the disconnect marker is cleared, `player_reconnected` is published, and
   the reconnecting socket gets a normal `connected` + `state_sync` pair —
   `state_sync` always carries the FULL current state (status, live question,
   time remaining, both players' progress), so reconnecting is how a client
   resyncs after any gap, not a special-cased partial update.
3. If the deadline passes with no reconnect, a background grace-check task
   (started by the opponent's own process the moment `player_disconnected`
   arrives) calls the same `forfeit_match` the REST `/forfeit` endpoint uses
   — same CAS-guarded, exactly-once Elo application, `result="forfeit"` on
   the disconnected side, a real loss on the ratings.

After changing anything in the WS route, the connection manager, or
matchmaking's Redis state, run the live end-to-end verification script
(exercises a real server, real Redis, and two real WebSocket clients —
catches races and cross-process bugs a `TestClient`-based test cannot):

```
uv run python scripts/verify_battle_live.py
```

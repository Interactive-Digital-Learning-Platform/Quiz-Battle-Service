"""
app/services/matchmaking_service.py
────────────────────────────────────────
Redis-backed 1v1 matchmaking: subject queues + rating-window widening.
Redis holds only transient queue/state; Postgres (BattleMatch/
BattleParticipant) becomes the source of truth the instant two players are
paired -- see the module-level key schema below.

This service must be safe across multiple app processes/workers (this
service already runs multi-worker in deployment -- see Dockerfile's
WEB_CONCURRENCY), so it does NOT use the in-process-set() debounce pattern
from quiz_service._kickoff_pool_replenish. Instead, every operation that
touches more than one user's state, or that must be atomic against a
concurrent claim, goes through a Redis Lua script (EVAL) -- Redis executes
scripts atomically/single-threaded, so there's no interleaving window
between two concurrent EVALs from different processes. That atomicity IS
the multi-instance-safety mechanism; no external lock is used or needed.

Redis key schema (all prefixed "battle:"):
  battle:user_state:{user_id}      HASH  -- single source of truth for one
                                            user's matchmaking state (absent
                                            = idle). Fields: state
                                            ("queued"|"matching"|"in_match"),
                                            subject, rating, queued_at,
                                            claimed_at, match_id.
  battle:subject_queue:{subject}   ZSET  -- member=user_id, score=queued_at.
                                            FIFO ordering for candidate search.
  battle:queue_subjects            SET   -- subjects with >=1 waiting user
                                            (subjects are freeform strings,
                                            no enum to enumerate otherwise).

TTLs on the state hash are safety nets only, not the primary lifecycle
mechanism: "queued" expires after BATTLE_MATCHMAKING_QUEUE_ENTRY_TTL_SECONDS
(an abandoned client that never cancels still frees itself); "matching"
expires after BATTLE_MATCHMAKING_CLAIM_TTL_SECONDS (if a process crashes
between claiming a pair and committing the Postgres match, both users
self-heal back to idle); "in_match" has no TTL -- a later phase's
match-completion flow owns clearing it, via release_match_state() below.

Accepted limitation (crash window): if a process crashes after the Postgres
commit but before the Redis finalize step, the claim-TTL expiry self-heals
both users to idle, but the created BattleMatch row (status="waiting") is
orphaned with no user pointing at it. A periodic reaper for stale "waiting"
matches -- run_stale_match_reap_once/start_stale_match_reaper_loop in
battle_gameplay_service.py, since that module already owns BattleMatch
status transitions -- closes this window (Phase 6).

Inverse limitation (post-commit release failure): release_match_state() is
always called AFTER its match's Postgres status update already committed,
as a separate, non-transactional step -- if that specific Redis call ever
fails, "in_match" (no TTL) sticks around forever and the affected user can
never queue again, since nothing else ever revisits an already-terminal
match. run_in_match_reconcile_once/start_in_match_reconciler_loop (below)
closes this window by periodically cross-checking every "in_match" pointer
against its match's actual Postgres status (Phase 6).
"""
import asyncio
import logging
import time

from fastapi import HTTPException, status
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.redis_client import redis_client
from app.models.battle_match import BattleMatch, BattleParticipant
from app.models.user import User
from app.services import battle_service, league_engine
from app.services.battle_difficulty import get_difficulty_label
from app.services.matchmaking_window import get_rating_window
from app.services.rate_limiter import allow as rate_limit_allow

logger = logging.getLogger(__name__)

_USER_STATE_PREFIX = "battle:user_state:"
_SUBJECT_QUEUE_PREFIX = "battle:subject_queue:"
_QUEUE_SUBJECTS_KEY = "battle:queue_subjects"
_PRESENCE_PREFIX = "battle:presence:"


def _user_state_key(user_id: int) -> str:
    return f"{_USER_STATE_PREFIX}{user_id}"


def _subject_queue_key(subject: str) -> str:
    return f"{_SUBJECT_QUEUE_PREFIX}{subject}"


def _presence_key(user_id: int) -> str:
    return f"{_PRESENCE_PREFIX}{user_id}"


# ─────────────────────────────────────────────────────────────────────────────
# Online presence -- deliberately independent of user_state (a user can be
# "online" without actively queuing/matching). TTL-based heartbeat, not an
# explicit-only toggle: mark_online() is called on every matchmaking
# touchpoint (join/cancel/status poll) and refreshes the TTL each time, so a
# client that vanishes without an explicit mark_offline() call (closed app,
# lost connection) self-heals to "offline" once BATTLE_PRESENCE_TTL_SECONDS
# elapses -- same safety-net philosophy as the user_state TTLs above.
# ─────────────────────────────────────────────────────────────────────────────

async def mark_online(redis: Redis, *, user_id: int) -> None:
    await redis.set(_presence_key(user_id), "1", ex=settings.BATTLE_PRESENCE_TTL_SECONDS)


async def mark_offline(redis: Redis, *, user_id: int) -> None:
    await redis.delete(_presence_key(user_id))


async def is_online(redis: Redis, *, user_id: int) -> bool:
    return await redis.exists(_presence_key(user_id)) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Lua scripts -- see module docstring for why these (not an app-level lock)
# are the actual multi-instance-safety mechanism.
# ─────────────────────────────────────────────────────────────────────────────

# KEYS[1]=user_state  KEYS[2]=subject_queue  KEYS[3]=queue_subjects
# ARGV[1]=subject  ARGV[2]=rating  ARGV[3]=queued_at  ARGV[4]=queue_ttl  ARGV[5]=user_id
_LUA_ENQUEUE = """
local state = redis.call('HGET', KEYS[1], 'state')
if state then
    return 0
end
redis.call('HSET', KEYS[1], 'state', 'queued', 'subject', ARGV[1], 'rating', ARGV[2], 'queued_at', ARGV[3])
redis.call('EXPIRE', KEYS[1], ARGV[4])
redis.call('ZADD', KEYS[2], ARGV[3], ARGV[5])
redis.call('SADD', KEYS[3], ARGV[1])
return 1
"""

# KEYS[1]=user_state:A  KEYS[2]=user_state:B  KEYS[3]=subject_queue
# ARGV[1]=A_id  ARGV[2]=B_id  ARGV[3]=subject  ARGV[4]=claim_ttl  ARGV[5]=claimed_at
_LUA_CLAIM_PAIR = """
local a_state = redis.call('HGET', KEYS[1], 'state')
local a_subj = redis.call('HGET', KEYS[1], 'subject')
local b_state = redis.call('HGET', KEYS[2], 'state')
local b_subj = redis.call('HGET', KEYS[2], 'subject')
if a_state ~= 'queued' or a_subj ~= ARGV[3] then
    return 0
end
if b_state ~= 'queued' or b_subj ~= ARGV[3] then
    return 0
end
redis.call('ZREM', KEYS[3], ARGV[1])
redis.call('ZREM', KEYS[3], ARGV[2])
redis.call('HSET', KEYS[1], 'state', 'matching', 'claimed_at', ARGV[5])
redis.call('EXPIRE', KEYS[1], ARGV[4])
redis.call('HSET', KEYS[2], 'state', 'matching', 'claimed_at', ARGV[5])
redis.call('EXPIRE', KEYS[2], ARGV[4])
return 1
"""

# KEYS[1]=user_state -- only cancels while still "queued"; no-ops (returns
# false) if a concurrent claim already moved the user to "matching"/"in_match".
_LUA_CANCEL = """
local state = redis.call('HGET', KEYS[1], 'state')
if state ~= 'queued' then
    return false
end
local subject = redis.call('HGET', KEYS[1], 'subject')
redis.call('DEL', KEYS[1])
return subject
"""


async def _lua_enqueue(redis: Redis, *, user_id: int, subject: str, rating: int, queued_at: float) -> bool:
    result = await redis.eval(
        _LUA_ENQUEUE,
        3,
        _user_state_key(user_id), _subject_queue_key(subject), _QUEUE_SUBJECTS_KEY,
        subject, str(rating), str(queued_at),
        str(settings.BATTLE_MATCHMAKING_QUEUE_ENTRY_TTL_SECONDS), str(user_id),
    )
    return bool(result)


async def _lua_claim_pair(redis: Redis, *, user_a_id: int, user_b_id: int, subject: str) -> bool:
    result = await redis.eval(
        _LUA_CLAIM_PAIR,
        3,
        _user_state_key(user_a_id), _user_state_key(user_b_id), _subject_queue_key(subject),
        str(user_a_id), str(user_b_id), subject,
        str(settings.BATTLE_MATCHMAKING_CLAIM_TTL_SECONDS), str(time.time()),
    )
    return bool(result)


async def _lua_cancel(redis: Redis, *, user_id: int) -> str | None:
    return await redis.eval(_LUA_CANCEL, 1, _user_state_key(user_id))


# ─────────────────────────────────────────────────────────────────────────────
# Matching
# ─────────────────────────────────────────────────────────────────────────────

# (user_id, rating, queued_at)
_Candidate = tuple[int, int, float]


def _find_best_partner(entry: _Candidate, others: list[_Candidate], now: float) -> _Candidate | None:
    """Smallest-rating-diff mutually-acceptable partner for `entry`, or None.

    "Mutually acceptable" uses min() of both sides' current windows so
    neither player is ever matched outside what THEY THEMSELVES currently
    tolerate, even if their prospective partner has been waiting much longer.
    """
    _uid, rating, queued_at = entry
    window_self = get_rating_window(now - queued_at)
    best: _Candidate | None = None
    best_diff: int | None = None
    for other in others:
        _o_uid, o_rating, o_queued_at = other
        window_other = get_rating_window(now - o_queued_at)
        allowed = min(window_self, window_other)
        diff = abs(rating - o_rating)
        if diff <= allowed and (best is None or diff < best_diff):
            best = other
            best_diff = diff
    return best


async def _create_match(
    db: AsyncSession, *, subject: str, user_a_id: int, rating_a: int, user_b_id: int, rating_b: int
) -> BattleMatch:
    # rating_a/rating_b come from each user's own Redis-cached rating at
    # their own join time, not a fresh Postgres re-read here -- avoids an
    # extra DB round trip on the hot claim path. Safe within this phase
    # because "queued" and any rating-mutating state are mutually exclusive
    # (no battle-completion flow exists yet); a future phase adding one
    # should not assume this still holds without re-checking.
    difficulty = get_difficulty_label((rating_a + rating_b) / 2)
    match = BattleMatch(subject=subject, status="waiting", difficulty=difficulty)
    db.add(match)
    await db.flush()  # match.id available for the FK below, before commit
    db.add(BattleParticipant(match_id=match.id, user_id=user_a_id, rating_before=rating_a))
    db.add(BattleParticipant(match_id=match.id, user_id=user_b_id, rating_before=rating_b))
    await db.commit()
    await db.refresh(match)
    return match


async def _finalize_matched_state(redis: Redis, *, match_id: int, user_a_id: int, user_b_id: int) -> None:
    # Only the single winning process reaches this point (no contention left
    # -- the Lua claim already resolved that) so a plain pipeline is enough,
    # no further Lua needed here.
    pipe = redis.pipeline(transaction=False)
    for uid in (user_a_id, user_b_id):
        key = _user_state_key(uid)
        pipe.hset(key, mapping={"state": "in_match", "match_id": str(match_id)})
        pipe.persist(key)
    await pipe.execute()


async def _attempt_claim(
    db: AsyncSession, redis: Redis, subject: str, entry_a: _Candidate, entry_b: _Candidate
) -> BattleMatch | None:
    a_id, a_rating, _ = entry_a
    b_id, b_rating, _ = entry_b
    claimed = await _lua_claim_pair(redis, user_a_id=a_id, user_b_id=b_id, subject=subject)
    if not claimed:
        return None
    match = await _create_match(
        db, subject=subject, user_a_id=a_id, rating_a=a_rating, user_b_id=b_id, rating_b=b_rating
    )
    await _finalize_matched_state(redis, match_id=match.id, user_a_id=a_id, user_b_id=b_id)
    return match


async def _find_and_claim_match(
    db: AsyncSession, redis: Redis, *, subject: str, target_user_id: int | None = None
) -> BattleMatch | None:
    """Shared pairing routine for both the join-time fast path
    (target_user_id given) and the periodic sweep (target_user_id=None).
    Any ZRANGE scan here is just an optimistic proposal -- non-atomic, may
    be stale -- the Lua claim script re-validates freshness itself, so a
    lost race here is always safe, never corrupting.
    """
    queue_key = _subject_queue_key(subject)
    members_with_scores = await redis.zrange(queue_key, 0, -1, withscores=True)
    if not members_with_scores:
        return None

    pipe = redis.pipeline(transaction=False)
    for member, _score in members_with_scores:
        pipe.hmget(_user_state_key(int(member)), "state", "subject", "rating", "queued_at")
    results = await pipe.execute()

    now = time.time()
    candidates: list[_Candidate] = []
    stale_members: list[str] = []
    for (member, _score), (state, subj, rating, queued_at) in zip(members_with_scores, results):
        if state != "queued" or subj != subject:
            # TTL-expired (or otherwise desynced) state hash whose ZSET
            # member didn't auto-vanish with it (ZSETs have no per-member
            # TTL) -- opportunistic cleanup.
            stale_members.append(member)
            continue
        candidates.append((int(member), int(rating), float(queued_at)))

    if stale_members:
        await redis.zrem(queue_key, *stale_members)

    if not candidates:
        return None

    if target_user_id is not None:
        target = next((c for c in candidates if c[0] == target_user_id), None)
        if target is None:
            return None
        partner = _find_best_partner(target, [c for c in candidates if c[0] != target_user_id], now)
        if partner is None:
            return None
        return await _attempt_claim(db, redis, subject, target, partner)

    # Sweep mode: oldest-first greedy pass, one claim attempt per call (the
    # caller loops calling this until it returns None).
    tried_pairs: set[frozenset[int]] = set()
    for primary in candidates:
        others = [c for c in candidates if c[0] != primary[0]]
        partner = _find_best_partner(primary, others, now)
        if partner is None:
            continue
        pair_key = frozenset((primary[0], partner[0]))
        if pair_key in tried_pairs:
            continue
        tried_pairs.add(pair_key)
        match = await _attempt_claim(db, redis, subject, primary, partner)
        if match is not None:
            return match
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public service functions
# ─────────────────────────────────────────────────────────────────────────────

def _join_queue_rate_limit_key(user_id: int) -> str:
    return f"battle:ratelimit:queue_join:{user_id}"


async def join_queue(db: AsyncSession, redis: Redis, *, user_id: int, subject: str) -> dict:
    # Everything below this point is Redis-dependent (rate limiting, the
    # Lua enqueue+claim atomicity, the status re-read) -- this is the most
    # user-facing, most-hit Redis path in the service, so an outage here
    # gets a distinct, loud signal rather than surfacing as an
    # undifferentiated 500.
    try:
        within_limit = await rate_limit_allow(
            redis, key=_join_queue_rate_limit_key(user_id),
            max_calls=settings.BATTLE_QUEUE_JOIN_RATE_LIMIT_MAX_CALLS,
            window_seconds=settings.BATTLE_QUEUE_JOIN_RATE_LIMIT_WINDOW_SECONDS,
        )
        if not within_limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many queue join attempts. Please wait a moment and try again.",
            )

        # After the rate limit, so a client hammering join can't turn each attempt into a DB write.
        rating_row = await battle_service.get_or_create_subject_rating(db, user_id=user_id, subject=subject)

        enqueued = await _lua_enqueue(
            redis, user_id=user_id, subject=subject, rating=rating_row.rating, queued_at=time.time()
        )
        if not enqueued:
            current_state = await redis.hget(_user_state_key(user_id), "state")
            detail = (
                "You're already in an active battle."
                if current_state == "in_match"
                else "You're already searching for a match."
            )
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)

        logger.info("Queue joined — user_id=%s, subject=%s", user_id, subject)
        await _find_and_claim_match(db, redis, subject=subject, target_user_id=user_id)
        return await get_queue_status(db, redis, user_id=user_id)
    except RedisError:
        logger.exception("Redis error while joining matchmaking queue — user_id=%s, subject=%s", user_id, subject)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Matchmaking is temporarily unavailable. Please try again shortly.",
        )


async def cancel_queue(db: AsyncSession, redis: Redis, *, user_id: int) -> dict:
    try:
        subject = await _lua_cancel(redis, user_id=user_id)
        if subject is not None:
            await redis.zrem(_subject_queue_key(subject), str(user_id))
            logger.info("Queue left — user_id=%s, subject=%s", user_id, subject)
            return {"status": "idle"}

        current_state = await redis.hget(_user_state_key(user_id), "state")
    except RedisError:
        # Same distinct signal as join_queue rather than an undifferentiated 500.
        logger.exception("Redis error while cancelling matchmaking — user_id=%s", user_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Matchmaking is temporarily unavailable. Please try again shortly.",
        )
    if current_state is None:
        return {"status": "idle"}  # already idle -- idempotent no-op, not an error
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Can't cancel — you're already matching or in an active battle.",
    )


async def release_match_state(redis: Redis, *, user_id: int) -> None:
    """Clears a user's matchmaking state back to idle once their match has
    ended (normal completion or forfeit) -- called by
    battle_gameplay_service on match completion. `state="in_match"` has no
    TTL by design (see module docstring), so without this call the user
    would be permanently stuck unable to join a new queue after their very
    first battle -- `join_queue` rejects with 409 "already in an active
    battle" for as long as this key exists.
    """
    await redis.delete(_user_state_key(user_id))


async def _build_matched_response(db: AsyncSession, *, match_id: int, requesting_user_id: int) -> dict:
    match = await db.get(BattleMatch, match_id)
    opponent = None
    awaiting_opponent = False
    if match is not None:
        # This player is done with every question but the opponent is still
        # playing -- the app shows "Show results" instead of matchmaking.
        indexes = dict(
            (
                await db.execute(
                    select(BattleParticipant.user_id, BattleParticipant.current_question_index)
                    .where(BattleParticipant.match_id == match_id)
                )
            ).all()
        )
        awaiting_opponent = match.status == "active" and indexes.get(requesting_user_id, 0) >= settings.BATTLE_QUESTION_COUNT
    if match is not None:
        row = (
            await db.execute(
                select(BattleParticipant, User.username)
                .join(User, User.id == BattleParticipant.user_id)
                .where(
                    BattleParticipant.match_id == match_id,
                    BattleParticipant.user_id != requesting_user_id,
                )
            )
        ).first()
        if row is not None:
            participant, username = row
            opponent = {
                "user_id": participant.user_id,
                "username": username,
                "rating": participant.rating_before,
                "league": league_engine.get_league(participant.rating_before),
            }
    return {
        "status": "matched",
        "subject": match.subject if match else None,
        "match_id": match_id,
        "difficulty": match.difficulty if match else None,
        "opponent": opponent,
        "awaiting_opponent": awaiting_opponent,
    }


async def get_queue_status(db: AsyncSession, redis: Redis, *, user_id: int) -> dict:
    # Every status check is a live presence signal -- covers both the
    # mobile client's own 2s status-polling loop while searching and
    # join_queue's own call to this function on success, so this single
    # wiring point covers nearly every real matchmaking touchpoint without
    # duplicating the call at each call site.
    await mark_online(redis, user_id=user_id)

    state, subject, _rating, queued_at, match_id = await redis.hmget(
        _user_state_key(user_id), "state", "subject", "rating", "queued_at", "match_id",
    )
    if state is None:
        return {"status": "idle"}
    if state == "in_match":
        return await _build_matched_response(db, match_id=int(match_id), requesting_user_id=user_id)

    # "queued" or "matching" -- the brief claim-in-progress state resolves
    # within one claim-TTL window and isn't surfaced as its own client-visible
    # status.
    waited = time.time() - float(queued_at)
    return {
        "status": "searching",
        "subject": subject,
        "waited_seconds": waited,
        "current_rating_window": get_rating_window(waited),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Startup queue reset -- Redis is a separate, long-lived process from this
# API server, so a server restart (a routine, frequent event in local dev,
# and possible at any time in deployment) leaves every "queued"/"matching"
# Redis entry sitting there completely untouched, even though the client
# that created it may have since restarted itself, given up, or otherwise
# lost track of it -- there is no client-side way to know whether a stale
# entry is still "theirs" to cancel. Called once at startup (app/main.py),
# BEFORE the sweep loop starts, so nothing can claim a leftover entry first.
# Deliberately does NOT touch "in_match" state: that points at a real,
# already-committed Postgres BattleMatch row that outlives this process
# restart same as everything else in Postgres does, and is already covered
# by run_in_match_reconcile_once/the stale-match reaper for its own cleanup.
# ─────────────────────────────────────────────────────────────────────────────

async def clear_all_queue_state(redis: Redis) -> int:
    """Deletes every "queued" matchmaking entry -- all per-subject queue
    ZSETs, battle:queue_subjects, and each queued user's user_state hash.
    Returns the number of users cleared. Safe to call on an already-empty
    queue (returns 0).
    """
    subjects = await redis.smembers(_QUEUE_SUBJECTS_KEY)
    cleared = 0
    for subject in subjects:
        queue_key = _subject_queue_key(subject)
        user_ids = await redis.zrange(queue_key, 0, -1)
        for user_id in user_ids:
            # Being a member of this ZSET means "queued" by construction --
            # _lua_claim_pair ZREMs both users in the same atomic script that
            # flips them to "matching" -- but re-check the state field before
            # deleting anyway rather than trusting that invariant blindly.
            state = await redis.hget(_user_state_key(user_id), "state")
            if state == "queued":
                await redis.delete(_user_state_key(user_id))
                cleared += 1
        await redis.delete(queue_key)
    await redis.delete(_QUEUE_SUBJECTS_KEY)
    if cleared:
        logger.info("Startup queue reset — cleared %d stale queued user(s).", cleared)
    return cleared


# ─────────────────────────────────────────────────────────────────────────────
# Background sweep -- widens the window for users who've been waiting.
# run_matchmaking_sweep_once is a single deterministic pass, directly
# testable with no sleeps; start_matchmaking_sweep_loop is the infinite-loop
# wrapper only ever started for real at app startup.
# ─────────────────────────────────────────────────────────────────────────────

async def run_matchmaking_sweep_once(db: AsyncSession, redis: Redis) -> dict:
    subjects = await redis.smembers(_QUEUE_SUBJECTS_KEY)
    matches_made = 0
    for subject in subjects:
        while True:
            match = await _find_and_claim_match(db, redis, subject=subject, target_user_id=None)
            if match is None:
                break
            matches_made += 1
        if await redis.zcard(_subject_queue_key(subject)) == 0:
            await redis.srem(_QUEUE_SUBJECTS_KEY, subject)
    return {"subjects_scanned": len(subjects), "matches_made": matches_made}


async def start_matchmaking_sweep_loop() -> None:
    # Mirrors quiz_service._replenish_pool_task's shape: opens its own
    # session each tick (the request-scoped session this was triggered from,
    # if any, is long gone), swallows all exceptions so one bad tick never
    # kills the loop or propagates anywhere.
    while True:
        try:
            async with AsyncSessionLocal() as session:
                await run_matchmaking_sweep_once(session, redis_client)
        except Exception:
            logger.exception("Matchmaking sweep tick failed")
        await asyncio.sleep(settings.BATTLE_MATCHMAKING_SWEEP_INTERVAL_SECONDS)


# ─────────────────────────────────────────────────────────────────────────────
# in_match/Postgres reconciliation -- self-heals the inverse of the crash
# window documented in this module's own docstring: release_match_state() is
# always called AFTER its match's Postgres status update has already
# committed (in battle_gameplay_service's _finalize_match/forfeit_match/AI-
# failure path, and this file's own stale-"waiting"-match reaper caller), as
# a separate, non-transactional step. If that specific Redis call ever fails
# (a transient Redis error, anything), the match is durably
# completed/cancelled in Postgres while Redis keeps saying "in_match"
# forever -- "in_match" has no TTL by design, and nothing else ever revisits
# an already-terminal match to retry the cleanup, so the affected user would
# otherwise be locked out of queueing again permanently. This periodically
# cross-checks every "in_match" Redis pointer against its match's actual
# Postgres status and deletes any pointer whose match already resolved.
# ─────────────────────────────────────────────────────────────────────────────

_TERMINAL_MATCH_STATUSES = ("completed", "cancelled")


def _reconcile_missing_key(user_id: int, match_id: int) -> str:
    # Set when the in_match reconciler first can't find the match a user points at; see run_in_match_reconcile_once.
    return f"battle:reconcile_missing:{user_id}:{match_id}"


async def run_in_match_reconcile_once(db: AsyncSession, redis: Redis) -> int:
    match_id_by_user: dict[int, int] = {}
    async for key in redis.scan_iter(match=f"{_USER_STATE_PREFIX}*"):
        state, match_id = await redis.hmget(key, "state", "match_id")
        if state != "in_match" or match_id is None:
            continue
        match_id_by_user[int(key[len(_USER_STATE_PREFIX):])] = int(match_id)

    if not match_id_by_user:
        return 0

    rows = (
        await db.execute(
            select(BattleMatch.id, BattleMatch.status).where(
                BattleMatch.id.in_(set(match_id_by_user.values()))
            )
        )
    ).all()
    status_by_match_id = {row.id: row.status for row in rows}

    healed = 0
    for user_id, match_id in match_id_by_user.items():
        match_status = status_by_match_id.get(match_id)
        if match_status is None:
            # Not found. A match row is committed before Redis ever points at it (_create_match ->
            # _finalize_matched_state), so this shouldn't happen -- and releasing on a single miss would free two
            # players from a match they may well still be playing (e.g. a moment of DB trouble, or another
            # process on a different database sharing this Redis). Only release once it's still missing on the
            # next tick.
            missing_key = _reconcile_missing_key(user_id, match_id)
            if not await redis.exists(missing_key):
                await redis.set(
                    missing_key, "1", ex=int(settings.BATTLE_IN_MATCH_RECONCILE_INTERVAL_SECONDS * 3),
                )
                logger.warning(
                    "in_match reconcile: user_id=%d points at match_id=%d, which this database doesn't have -- "
                    "re-checking next tick before releasing.", user_id, match_id,
                )
                continue
            await redis.delete(missing_key)
            await release_match_state(redis, user_id=user_id)
            logger.error(
                "in_match reconcile: released user_id=%d -- match_id=%d still not found on a second check. If "
                "that match is real, this process is connected to a different database than the one that "
                "created it (check DATABASE_URL against the other services sharing this Redis).",
                user_id, match_id,
            )
            healed += 1
            continue
        if match_status in _TERMINAL_MATCH_STATUSES:
            await release_match_state(redis, user_id=user_id)
            logger.warning(
                "Reconciled stale in_match state -- user_id=%d pointed at match_id=%d "
                "(status=%s), which release_match_state should already have cleared.",
                user_id, match_id, match_status,
            )
            healed += 1
    return healed


async def start_in_match_reconciler_loop() -> None:
    # Same shape as start_matchmaking_sweep_loop/start_stale_match_reaper_loop.
    while True:
        try:
            async with AsyncSessionLocal() as session:
                await run_in_match_reconcile_once(session, redis_client)
        except Exception:
            logger.exception("in_match reconciliation tick failed")
        await asyncio.sleep(settings.BATTLE_IN_MATCH_RECONCILE_INTERVAL_SECONDS)

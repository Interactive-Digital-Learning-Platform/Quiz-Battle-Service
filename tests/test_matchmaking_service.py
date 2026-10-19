"""
tests/test_matchmaking_service.py
──────────────────────────────────────
Integration tests (in-memory DB + fake_redis, no HTTP) for
app/services/matchmaking_service.py -- queue join/cancel, immediate + swept
matching, rating-window widening, subject isolation, duplicate/self-match
prevention, and the Lua-script atomicity that makes claiming safe across
concurrent callers (the proxy for multi-instance safety).
"""
import asyncio
import time

import pytest
from fastapi import HTTPException
from redis.exceptions import RedisError
from sqlalchemy import select

from app.core.config import settings
from app.models.battle_match import BattleMatch
from app.models.competitive_rating import CompetitiveSubjectRating
from app.models.user import User
from app.services import matchmaking_service
from app.services.user_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID


async def _make_user(db, clerk_id: str) -> User:
    return await get_or_create_user(db, clerk_id)


async def _seed_rating(db, user_id: int, subject: str, rating: int) -> CompetitiveSubjectRating:
    row = CompetitiveSubjectRating(
        user_id=user_id, subject=subject, rating=rating, highest_rating=rating
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def _backdate_queue_entry(fake_redis, *, user_id: int, subject: str, seconds_ago: float) -> None:
    """Rewrites a user's queued_at (state hash + ZSET score) to simulate
    having waited `seconds_ago` seconds, without changing their state.
    """
    new_queued_at = time.time() - seconds_ago
    state_key = matchmaking_service._user_state_key(user_id)
    queue_key = matchmaking_service._subject_queue_key(subject)
    await fake_redis.hset(state_key, "queued_at", str(new_queued_at))
    await fake_redis.zadd(queue_key, {str(user_id): new_queued_at})


async def _count_matches(db) -> int:
    return len((await db.execute(select(BattleMatch))).scalars().all())


# ─────────────────────────────────────────────────────────────────────────────
# join_queue -- lazy rating creation, immediate matching
# ─────────────────────────────────────────────────────────────────────────────

async def test_join_queue_creates_lazy_rating_row_with_initial_rating(db_session, fake_redis):
    user = await _make_user(db_session, TEST_CLERK_ID)
    result = await matchmaking_service.join_queue(
        db_session, fake_redis, user_id=user.id, subject="Mathematics"
    )
    assert result["status"] == "searching"

    row = (
        await db_session.execute(
            select(CompetitiveSubjectRating).where(
                CompetitiveSubjectRating.user_id == user.id,
                CompetitiveSubjectRating.subject == "Mathematics",
            )
        )
    ).scalar_one()
    assert row.rating == settings.BATTLE_INITIAL_RATING


async def test_join_queue_two_close_ratings_matches_immediately(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    await _seed_rating(db_session, user_b.id, "Mathematics", 1050)  # diff=50, within tier1 window

    result_a = await matchmaking_service.join_queue(
        db_session, fake_redis, user_id=user_a.id, subject="Mathematics"
    )
    assert result_a["status"] == "searching"

    result_b = await matchmaking_service.join_queue(
        db_session, fake_redis, user_id=user_b.id, subject="Mathematics"
    )
    assert result_b["status"] == "matched"
    assert result_b["opponent"]["user_id"] == user_a.id

    # user_a's status should now also reflect the match.
    status_a = await matchmaking_service.get_queue_status(db_session, fake_redis, user_id=user_a.id)
    assert status_a["status"] == "matched"
    assert status_a["match_id"] == result_b["match_id"]
    assert status_a["opponent"]["user_id"] == user_b.id

    assert await _count_matches(db_session) == 1


async def test_join_queue_far_apart_ratings_stays_searching(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    await _seed_rating(db_session, user_b.id, "Mathematics", 1500)  # diff=500 > tier4 cap (400)

    await matchmaking_service.join_queue(db_session, fake_redis, user_id=user_a.id, subject="Mathematics")
    result_b = await matchmaking_service.join_queue(
        db_session, fake_redis, user_id=user_b.id, subject="Mathematics"
    )
    assert result_b["status"] == "searching"
    assert await _count_matches(db_session) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Sweep widens the window over time
# ─────────────────────────────────────────────────────────────────────────────

async def test_sweep_widens_window_over_time(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    await _seed_rating(db_session, user_b.id, "Mathematics", 1300)  # diff=300 -- needs tier4 (>=20s wait)

    await matchmaking_service.join_queue(db_session, fake_redis, user_id=user_a.id, subject="Mathematics")
    result_b = await matchmaking_service.join_queue(
        db_session, fake_redis, user_id=user_b.id, subject="Mathematics"
    )
    assert result_b["status"] == "searching"  # too far apart at wait=0 (tier1 window=100)

    await _backdate_queue_entry(fake_redis, user_id=user_a.id, subject="Mathematics", seconds_ago=25)
    await _backdate_queue_entry(fake_redis, user_id=user_b.id, subject="Mathematics", seconds_ago=25)

    sweep_result = await matchmaking_service.run_matchmaking_sweep_once(db_session, fake_redis)
    assert sweep_result["matches_made"] == 1
    assert await _count_matches(db_session) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Subject isolation, no self-match
# ─────────────────────────────────────────────────────────────────────────────

async def test_subject_isolation_no_cross_subject_matching(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    await _seed_rating(db_session, user_b.id, "Science", 1000)

    await matchmaking_service.join_queue(db_session, fake_redis, user_id=user_a.id, subject="Mathematics")
    result_b = await matchmaking_service.join_queue(db_session, fake_redis, user_id=user_b.id, subject="Science")

    assert result_b["status"] == "searching"
    await matchmaking_service.run_matchmaking_sweep_once(db_session, fake_redis)
    assert await _count_matches(db_session) == 0


async def test_no_self_match(db_session, fake_redis):
    user = await _make_user(db_session, TEST_CLERK_ID)
    await matchmaking_service.join_queue(db_session, fake_redis, user_id=user.id, subject="Mathematics")

    sweep_result = await matchmaking_service.run_matchmaking_sweep_once(db_session, fake_redis)
    assert sweep_result["matches_made"] == 0

    status = await matchmaking_service.get_queue_status(db_session, fake_redis, user_id=user.id)
    assert status["status"] == "searching"


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate-queue / already-in-match prevention
# ─────────────────────────────────────────────────────────────────────────────

async def test_duplicate_queue_join_rejected(db_session, fake_redis):
    user = await _make_user(db_session, TEST_CLERK_ID)
    await matchmaking_service.join_queue(db_session, fake_redis, user_id=user.id, subject="Mathematics")

    with pytest.raises(HTTPException) as exc_info:
        await matchmaking_service.join_queue(db_session, fake_redis, user_id=user.id, subject="Mathematics")
    assert exc_info.value.status_code == 409


async def test_join_queue_rate_limited_after_max_calls(db_session, fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "BATTLE_QUEUE_JOIN_RATE_LIMIT_MAX_CALLS", 2)
    user = await _make_user(db_session, TEST_CLERK_ID)

    for _ in range(2):
        await matchmaking_service.join_queue(db_session, fake_redis, user_id=user.id, subject="Mathematics")
        await matchmaking_service.cancel_queue(db_session, fake_redis, user_id=user.id)

    with pytest.raises(HTTPException) as exc_info:
        await matchmaking_service.join_queue(db_session, fake_redis, user_id=user.id, subject="Mathematics")
    assert exc_info.value.status_code == 429


async def test_join_while_in_active_match_rejected(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    await _seed_rating(db_session, user_b.id, "Mathematics", 1000)

    await matchmaking_service.join_queue(db_session, fake_redis, user_id=user_a.id, subject="Mathematics")
    result_b = await matchmaking_service.join_queue(db_session, fake_redis, user_id=user_b.id, subject="Mathematics")
    assert result_b["status"] == "matched"

    with pytest.raises(HTTPException) as exc_info:
        await matchmaking_service.join_queue(db_session, fake_redis, user_id=user_a.id, subject="Science")
    assert exc_info.value.status_code == 409


# ─────────────────────────────────────────────────────────────────────────────
# Cancel
# ─────────────────────────────────────────────────────────────────────────────

async def test_cancel_frees_user_back_to_idle(db_session, fake_redis):
    user = await _make_user(db_session, TEST_CLERK_ID)
    await matchmaking_service.join_queue(db_session, fake_redis, user_id=user.id, subject="Mathematics")

    result = await matchmaking_service.cancel_queue(db_session, fake_redis, user_id=user.id)
    assert result == {"status": "idle"}

    state_key = matchmaking_service._user_state_key(user.id)
    queue_key = matchmaking_service._subject_queue_key("Mathematics")
    assert await fake_redis.exists(state_key) == 0
    assert await fake_redis.zscore(queue_key, str(user.id)) is None


async def test_cancel_when_not_queued_is_a_safe_noop(db_session, fake_redis):
    user = await _make_user(db_session, TEST_CLERK_ID)
    result = await matchmaking_service.cancel_queue(db_session, fake_redis, user_id=user.id)
    assert result == {"status": "idle"}


async def test_cancel_while_matching_or_in_match_is_rejected(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    await _seed_rating(db_session, user_a.id, "Mathematics", 1000)
    await _seed_rating(db_session, user_b.id, "Mathematics", 1000)

    await matchmaking_service.join_queue(db_session, fake_redis, user_id=user_a.id, subject="Mathematics")
    await matchmaking_service.join_queue(db_session, fake_redis, user_id=user_b.id, subject="Mathematics")

    with pytest.raises(HTTPException) as exc_info:
        await matchmaking_service.cancel_queue(db_session, fake_redis, user_id=user_a.id)
    assert exc_info.value.status_code == 409


# ─────────────────────────────────────────────────────────────────────────────
# Atomicity under concurrency -- the multi-instance-safety crux
# ─────────────────────────────────────────────────────────────────────────────

async def test_concurrent_claim_pair_only_one_succeeds(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")

    # Enqueue directly (bypassing join_queue's own immediate-match attempt)
    # so both are sitting in "queued" state, ready to be claimed.
    await matchmaking_service._lua_enqueue(
        fake_redis, user_id=user_a.id, subject="Mathematics", rating=1000, queued_at=time.time()
    )
    await matchmaking_service._lua_enqueue(
        fake_redis, user_id=user_b.id, subject="Mathematics", rating=1000, queued_at=time.time()
    )

    # Two "processes" racing to claim the exact same pair concurrently --
    # this directly exercises the Lua script's atomicity, which is the
    # actual mechanism preventing a double-match across app instances.
    results = await asyncio.gather(
        matchmaking_service._lua_claim_pair(
            fake_redis, user_a_id=user_a.id, user_b_id=user_b.id, subject="Mathematics"
        ),
        matchmaking_service._lua_claim_pair(
            fake_redis, user_a_id=user_a.id, user_b_id=user_b.id, subject="Mathematics"
        ),
    )
    assert sorted(results) == [False, True]


async def test_sweep_pairs_multiple_waiting_users_without_duplicates(db_session, fake_redis):
    users = [await _make_user(db_session, f"quad-user-{i}") for i in range(4)]
    ratings = [1000, 1020, 1040, 1060]  # all pairwise within tier1's +/-100 window
    for user, rating in zip(users, ratings):
        await _seed_rating(db_session, user.id, "Mathematics", rating)
        await matchmaking_service.join_queue(db_session, fake_redis, user_id=user.id, subject="Mathematics")

    # join_queue's own immediate-match attempts may already have paired some
    # of these up; sweep mops up anything left.
    await matchmaking_service.run_matchmaking_sweep_once(db_session, fake_redis)

    matches = (await db_session.execute(select(BattleMatch))).scalars().all()
    assert len(matches) == 2

    from app.models.battle_match import BattleParticipant
    participant_user_ids = (
        (await db_session.execute(select(BattleParticipant.user_id))).scalars().all()
    )
    assert sorted(participant_user_ids) == sorted(u.id for u in users)
    assert len(set(participant_user_ids)) == 4  # nobody matched twice



# ─────────────────────────────────────────────────────────────────────────────
# Online presence
# ─────────────────────────────────────────────────────────────────────────────

async def test_mark_online_then_offline(fake_redis):
    user_id = 42
    assert await matchmaking_service.is_online(fake_redis, user_id=user_id) is False

    await matchmaking_service.mark_online(fake_redis, user_id=user_id)
    assert await matchmaking_service.is_online(fake_redis, user_id=user_id) is True

    await matchmaking_service.mark_offline(fake_redis, user_id=user_id)
    assert await matchmaking_service.is_online(fake_redis, user_id=user_id) is False


async def test_mark_online_sets_ttl(fake_redis):
    user_id = 42
    await matchmaking_service.mark_online(fake_redis, user_id=user_id)
    ttl = await fake_redis.ttl(matchmaking_service._presence_key(user_id))
    assert 0 < ttl <= settings.BATTLE_PRESENCE_TTL_SECONDS


async def test_queue_status_poll_marks_user_online(db_session, fake_redis):
    user = await _make_user(db_session, TEST_CLERK_ID)
    assert await matchmaking_service.is_online(fake_redis, user_id=user.id) is False

    await matchmaking_service.get_queue_status(db_session, fake_redis, user_id=user.id)
    assert await matchmaking_service.is_online(fake_redis, user_id=user.id) is True


# ─────────────────────────────────────────────────────────────────────────────
# Resilience -- a Redis outage must degrade gracefully, never corrupt Postgres
# ─────────────────────────────────────────────────────────────────────────────

async def test_join_queue_redis_failure_returns_503_and_does_not_orphan_postgres(
    db_session, fake_redis, monkeypatch
):
    user = await _make_user(db_session, TEST_CLERK_ID)

    async def _boom(*args, **kwargs):
        raise RedisError("simulated Redis outage")

    monkeypatch.setattr(matchmaking_service, "_lua_enqueue", _boom)

    with pytest.raises(HTTPException) as exc_info:
        await matchmaking_service.join_queue(db_session, fake_redis, user_id=user.id, subject="Mathematics")
    assert exc_info.value.status_code == 503

    # The lazily-created CompetitiveSubjectRating row happens before any
    # Redis touch and is a harmless, idempotent write on its own -- its
    # existence isn't "corruption". What actually matters is that nothing
    # battle-match-shaped got created from the failed attempt.
    rating = (
        await db_session.execute(
            select(CompetitiveSubjectRating).where(
                CompetitiveSubjectRating.user_id == user.id,
                CompetitiveSubjectRating.subject == "Mathematics",
            )
        )
    ).scalar_one()
    assert rating.rating == settings.BATTLE_INITIAL_RATING
    assert rating.matches_played == 0
    assert await _count_matches(db_session) == 0


async def test_stale_queue_entry_cleanup(db_session, fake_redis):
    # A ZSET member with no corresponding state hash (simulating a
    # TTL-expired desync -- ZSETs have no per-member TTL of their own).
    queue_key = matchmaking_service._subject_queue_key("Mathematics")
    await fake_redis.zadd(queue_key, {"999999": time.time()})
    await fake_redis.sadd(matchmaking_service._QUEUE_SUBJECTS_KEY, "Mathematics")

    result = await matchmaking_service.run_matchmaking_sweep_once(db_session, fake_redis)
    assert result["matches_made"] == 0
    assert await fake_redis.zscore(queue_key, "999999") is None


# ─────────────────────────────────────────────────────────────────────────────
# run_in_match_reconcile_once -- self-heals in_match Redis pointers that
# outlived their match (the "inverse limitation" documented in this
# module's docstring: a post-commit release_match_state() call that failed).
# ─────────────────────────────────────────────────────────────────────────────

async def _create_match(db, *, status: str) -> BattleMatch:
    match = BattleMatch(subject="Mathematics", status=status, difficulty="Mostly Medium")
    db.add(match)
    await db.commit()
    await db.refresh(match)
    return match


async def test_reconcile_heals_in_match_state_for_completed_match(db_session, fake_redis):
    user = await _make_user(db_session, TEST_CLERK_ID)
    match = await _create_match(db_session, status="completed")
    await fake_redis.hset(
        matchmaking_service._user_state_key(user.id),
        mapping={"state": "in_match", "match_id": str(match.id)},
    )

    healed = await matchmaking_service.run_in_match_reconcile_once(db_session, fake_redis)

    assert healed == 1
    assert await fake_redis.exists(matchmaking_service._user_state_key(user.id)) == 0


async def test_reconcile_heals_in_match_state_for_cancelled_match(db_session, fake_redis):
    user = await _make_user(db_session, TEST_CLERK_ID)
    match = await _create_match(db_session, status="cancelled")
    await fake_redis.hset(
        matchmaking_service._user_state_key(user.id),
        mapping={"state": "in_match", "match_id": str(match.id)},
    )

    healed = await matchmaking_service.run_in_match_reconcile_once(db_session, fake_redis)

    assert healed == 1
    assert await fake_redis.exists(matchmaking_service._user_state_key(user.id)) == 0


async def test_reconcile_releases_a_nonexistent_match_only_on_the_second_check(db_session, fake_redis):
    # A match row is committed before Redis points at it, so "not found" is suspicious (a moment of DB trouble,
    # or a process on a different database sharing this Redis) -- releasing on one miss could free two players
    # from a match they're still playing. Still missing on the next tick: then it's released.
    user = await _make_user(db_session, TEST_CLERK_ID)
    key = matchmaking_service._user_state_key(user.id)
    await fake_redis.hset(key, mapping={"state": "in_match", "match_id": "999999"})

    first = await matchmaking_service.run_in_match_reconcile_once(db_session, fake_redis)
    assert first == 0
    assert await fake_redis.exists(key) == 1

    second = await matchmaking_service.run_in_match_reconcile_once(db_session, fake_redis)
    assert second == 1
    assert await fake_redis.exists(key) == 0


async def test_reconcile_keeps_a_match_found_again_after_one_miss(db_session, fake_redis):
    user_a = await _make_user(db_session, TEST_CLERK_ID)
    match = BattleMatch(subject="Mathematics", status="active", difficulty="Mostly Medium")
    db_session.add(match)
    await db_session.commit()
    key = matchmaking_service._user_state_key(user_a.id)
    await fake_redis.hset(key, mapping={"state": "in_match", "match_id": str(match.id)})
    # As if the previous tick had failed to find it.
    await fake_redis.set(matchmaking_service._reconcile_missing_key(user_a.id, match.id), "1")

    assert await matchmaking_service.run_in_match_reconcile_once(db_session, fake_redis) == 0
    assert await fake_redis.exists(key) == 1


async def test_reconcile_leaves_genuinely_active_match_untouched(db_session, fake_redis):
    user = await _make_user(db_session, TEST_CLERK_ID)
    match = await _create_match(db_session, status="active")
    await fake_redis.hset(
        matchmaking_service._user_state_key(user.id),
        mapping={"state": "in_match", "match_id": str(match.id)},
    )

    healed = await matchmaking_service.run_in_match_reconcile_once(db_session, fake_redis)

    assert healed == 0
    state = await fake_redis.hgetall(matchmaking_service._user_state_key(user.id))
    assert state["state"] == "in_match"
    assert state["match_id"] == str(match.id)


async def test_reconcile_leaves_queued_and_matching_state_untouched(db_session, fake_redis):
    # Only "in_match" is this reconciler's job -- "queued"/"matching" already
    # self-heal via their own TTLs (see module docstring).
    queued_user = await _make_user(db_session, "user-queued")
    matching_user = await _make_user(db_session, "user-matching")
    await fake_redis.hset(
        matchmaking_service._user_state_key(queued_user.id), mapping={"state": "queued"}
    )
    await fake_redis.hset(
        matchmaking_service._user_state_key(matching_user.id), mapping={"state": "matching"}
    )

    healed = await matchmaking_service.run_in_match_reconcile_once(db_session, fake_redis)

    assert healed == 0
    assert await fake_redis.exists(matchmaking_service._user_state_key(queued_user.id)) == 1
    assert await fake_redis.exists(matchmaking_service._user_state_key(matching_user.id)) == 1


# ─────────────────────────────────────────────────────────────────────────────
# clear_all_queue_state -- the startup reset (app/main.py's lifespan calls
# this before the sweep loop starts) that clears leftover "queued" Redis
# entries a server restart would otherwise leave sitting there indefinitely.
# ─────────────────────────────────────────────────────────────────────────────

async def test_clear_all_queue_state_removes_queued_users(db_session, fake_redis):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    await matchmaking_service.join_queue(db_session, fake_redis, user_id=user_a.id, subject="Mathematics")
    await matchmaking_service.join_queue(db_session, fake_redis, user_id=user_b.id, subject="Science")

    cleared = await matchmaking_service.clear_all_queue_state(fake_redis)

    assert cleared == 2
    assert await fake_redis.exists(matchmaking_service._user_state_key(user_a.id)) == 0
    assert await fake_redis.exists(matchmaking_service._user_state_key(user_b.id)) == 0
    assert await fake_redis.zcard(matchmaking_service._subject_queue_key("Mathematics")) == 0
    assert await fake_redis.zcard(matchmaking_service._subject_queue_key("Science")) == 0
    assert await fake_redis.scard(matchmaking_service._QUEUE_SUBJECTS_KEY) == 0


async def test_clear_all_queue_state_leaves_in_match_users_untouched(db_session, fake_redis):
    # A user genuinely mid-match points at a real, already-committed
    # Postgres row -- restarting the server must not sever that.
    in_match_user = await _make_user(db_session, "user-in-match")
    match = await _create_match(db_session, status="active")
    await fake_redis.hset(
        matchmaking_service._user_state_key(in_match_user.id),
        mapping={"state": "in_match", "match_id": str(match.id)},
    )

    cleared = await matchmaking_service.clear_all_queue_state(fake_redis)

    assert cleared == 0
    state = await fake_redis.hgetall(matchmaking_service._user_state_key(in_match_user.id))
    assert state["state"] == "in_match"


async def test_clear_all_queue_state_is_a_safe_noop_when_nothing_queued(db_session, fake_redis):
    assert await matchmaking_service.clear_all_queue_state(fake_redis) == 0

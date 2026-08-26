"""
tests/test_battle_queue_routes.py
──────────────────────────────────────
HTTP-level integration tests for POST /api/v1/battle/queue/join,
POST /api/v1/battle/queue/cancel, GET /api/v1/battle/queue/status --
real DB (in-memory SQLite) + fake_redis, real routing, real Pydantic
response validation.
"""
from app.models.competitive_rating import CompetitiveSubjectRating
from app.services import matchmaking_service
from app.services.user_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID


async def _seed_rating(db, user_id: int, subject: str, rating: int) -> CompetitiveSubjectRating:
    row = CompetitiveSubjectRating(user_id=user_id, subject=subject, rating=rating, highest_rating=rating)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# join
# ─────────────────────────────────────────────────────────────────────────────

async def test_join_queue_returns_searching_when_no_match(client, db_session, fake_redis):
    resp = await client.post("/api/v1/battle/queue/join", json={"subject": "Mathematics"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "searching"
    assert data["subject"] == "Mathematics"
    assert data["current_rating_window"] == 100


async def test_join_queue_returns_matched_with_opponent_info(client, db_session, fake_redis):
    me = await get_or_create_user(db_session, TEST_CLERK_ID)
    other = await get_or_create_user(db_session, "other-clerk-id")
    await _seed_rating(db_session, me.id, "Mathematics", 1000)
    await _seed_rating(db_session, other.id, "Mathematics", 1020)

    # Seed the opponent's queue presence directly via the service (no second
    # real HTTP identity needed), following test_battle_routes.py's existing
    # precedent of seeding state directly rather than through endpoints.
    await matchmaking_service.join_queue(db_session, fake_redis, user_id=other.id, subject="Mathematics")

    resp = await client.post("/api/v1/battle/queue/join", json={"subject": "Mathematics"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "matched"
    assert data["opponent"]["user_id"] == other.id
    assert data["opponent"]["rating"] == 1020
    assert data["opponent"]["league"] == "Bronze"
    assert data["difficulty"] in (
        "Mostly Easy", "Easy + Medium", "Mostly Medium", "Medium + Hard", "Mostly Hard",
    )
    assert data["match_id"] is not None


async def test_join_twice_returns_409(client, db_session, fake_redis):
    resp1 = await client.post("/api/v1/battle/queue/join", json={"subject": "Mathematics"})
    assert resp1.status_code == 200

    resp2 = await client.post("/api/v1/battle/queue/join", json={"subject": "Mathematics"})
    assert resp2.status_code == 409


async def test_forfeit_waiting_match_then_rejoin_queue_succeeds(client, db_session, fake_redis):
    """Mirrors the mobile "Cancel Match" flow on the matchmaking screen: get
    matched, forfeit the still-"waiting" match (voids it, no scoring -- see
    test_battle_ready_forfeit_service.py), then immediately try to queue
    again as the SAME user who forfeited. This must succeed -- if
    release_match_state (or anything downstream of it) left a stale
    "in_match" Redis pointer for the forfeiting user, this would 409 with
    "You're already in an active battle." exactly like the bug report.
    """
    other = await get_or_create_user(db_session, "other-clerk-id")
    await _seed_rating(db_session, other.id, "Mathematics", 20)
    await matchmaking_service.join_queue(db_session, fake_redis, user_id=other.id, subject="Mathematics")

    resp_join = await client.post("/api/v1/battle/queue/join", json={"subject": "Mathematics"})
    assert resp_join.status_code == 200
    body = resp_join.json()
    assert body["status"] == "matched"
    match_id = body["match_id"]
    assert match_id is not None

    resp_forfeit = await client.post(f"/api/v1/battle/match/{match_id}/forfeit")
    assert resp_forfeit.status_code == 200
    assert resp_forfeit.json()["status"] == "cancelled"

    resp_rejoin = await client.post("/api/v1/battle/queue/join", json={"subject": "Mathematics"})
    assert resp_rejoin.status_code == 200
    assert resp_rejoin.json()["status"] == "searching"


# ─────────────────────────────────────────────────────────────────────────────
# cancel
# ─────────────────────────────────────────────────────────────────────────────

async def test_cancel_queue_returns_idle(client, db_session, fake_redis):
    await client.post("/api/v1/battle/queue/join", json={"subject": "Mathematics"})

    resp = await client.post("/api/v1/battle/queue/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "idle"


async def test_cancel_when_not_queued_returns_idle_not_error(client, db_session, fake_redis):
    resp = await client.post("/api/v1/battle/queue/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "idle"


# ─────────────────────────────────────────────────────────────────────────────
# status
# ─────────────────────────────────────────────────────────────────────────────

async def test_status_reflects_idle_then_searching_then_matched(client, db_session, fake_redis):
    resp_idle = await client.get("/api/v1/battle/queue/status")
    assert resp_idle.json()["status"] == "idle"

    await client.post("/api/v1/battle/queue/join", json={"subject": "Mathematics"})
    resp_searching = await client.get("/api/v1/battle/queue/status")
    assert resp_searching.json()["status"] == "searching"

    me = await get_or_create_user(db_session, TEST_CLERK_ID)
    other = await get_or_create_user(db_session, "other-clerk-id")
    await _seed_rating(db_session, other.id, "Mathematics", 20)
    # me's row already exists from the join above at BATTLE_INITIAL_RATING.
    await matchmaking_service.join_queue(db_session, fake_redis, user_id=other.id, subject="Mathematics")

    resp_matched = await client.get("/api/v1/battle/queue/status")
    assert resp_matched.json()["status"] == "matched"
    assert resp_matched.json()["opponent"]["user_id"] == other.id

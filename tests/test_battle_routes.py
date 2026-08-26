"""
tests/test_battle_routes.py
───────────────────────────────
HTTP-level integration tests for GET /api/v1/battle/leaderboard and
GET /api/v1/battle/profile -- real DB (in-memory SQLite), real routing,
real Pydantic response validation.
"""
from app.models.competitive_rating import CompetitiveSubjectRating
from app.services.user_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID


async def _seed_rating(db, user_id: int, subject: str, **kwargs) -> CompetitiveSubjectRating:
    row = CompetitiveSubjectRating(user_id=user_id, subject=subject, **kwargs)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# GET /battle/profile
# ─────────────────────────────────────────────────────────────────────────────

async def test_profile_empty_for_brand_new_user(client, db_session):
    resp = await client.get("/api/v1/battle/profile")
    assert resp.status_code == 200
    data = resp.json()
    assert data["subjects"] == []
    assert "user_id" in data


async def test_profile_returns_full_shape_once_seeded(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    await _seed_rating(
        db_session, user.id, "Mathematics",
        rating=1742, highest_rating=1800, wins=20, losses=12, draws=1,
        matches_played=33, current_win_streak=3, highest_win_streak=6,
        correct_answers=180, total_answers=220, average_response_time=6.4,
    )

    resp = await client.get("/api/v1/battle/profile")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["subjects"]) == 1

    entry = data["subjects"][0]
    assert entry["subject"] == "Mathematics"
    assert entry["rating"] == 1742
    assert entry["league"] == "Gold"
    assert entry["rank"] == 1
    assert entry["wins"] == 20
    assert entry["losses"] == 12
    assert entry["draws"] == 1
    assert entry["matches_played"] == 33
    assert entry["current_win_streak"] == 3
    assert entry["highest_win_streak"] == 6
    assert entry["correct_answers"] == 180
    assert entry["total_answers"] == 220
    assert entry["accuracy"] == 180 / 220 * 100.0
    assert entry["average_response_time"] == 6.4


# ─────────────────────────────────────────────────────────────────────────────
# GET /battle/leaderboard
# ─────────────────────────────────────────────────────────────────────────────

async def test_leaderboard_returns_200_with_expected_shape(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    await _seed_rating(db_session, user.id, "Mathematics", rating=1500)

    resp = await client.get("/api/v1/battle/leaderboard", params={"subject": "Mathematics"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["subject"] == "Mathematics"
    assert data["page"] == 1
    assert data["total_count"] == 1
    assert len(data["entries"]) == 1
    assert data["my_rank"] == 1
    assert data["my_rating"] == 1500


async def test_leaderboard_missing_subject_is_422(client, db_session):
    resp = await client.get("/api/v1/battle/leaderboard")
    assert resp.status_code == 422


async def test_leaderboard_page_size_over_max_is_422(client, db_session):
    from app.core.config import settings

    resp = await client.get(
        "/api/v1/battle/leaderboard",
        params={"subject": "Mathematics", "page_size": settings.BATTLE_LEADERBOARD_MAX_PAGE_SIZE + 1},
    )
    assert resp.status_code == 422


async def test_leaderboard_empty_for_unplayed_subject(client, db_session):
    resp = await client.get("/api/v1/battle/leaderboard", params={"subject": "Nobody Plays This"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["entries"] == []
    assert data["total_count"] == 0
    assert data["my_rank"] is None

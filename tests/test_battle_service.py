"""
tests/test_battle_service.py
────────────────────────────────
Integration tests (real in-memory DB, no HTTP) for
app/services/battle_service.py -- leaderboard ordering/pagination/rank
lookup and cross-subject independence of CompetitiveSubjectRating rows.
"""
import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.models.competitive_rating import CompetitiveSubjectRating
from app.models.user import User
from app.services import battle_service
from app.services.user_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID


async def _make_user(db, clerk_id: str) -> User:
    return await get_or_create_user(db, clerk_id)


async def _make_rating(
    db, user_id: int, subject: str, *, rating: int = 1000, wins: int = 0,
    losses: int = 0, draws: int = 0, average_response_time: float = 0.0,
    correct_answers: int = 0, total_answers: int = 0,
) -> CompetitiveSubjectRating:
    row = CompetitiveSubjectRating(
        user_id=user_id, subject=subject, rating=rating, highest_rating=rating,
        wins=wins, losses=losses, draws=draws, matches_played=wins + losses + draws,
        average_response_time=average_response_time,
        correct_answers=correct_answers, total_answers=total_answers,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Cross-subject independence
# ─────────────────────────────────────────────────────────────────────────────

async def test_same_user_different_subjects_are_independent(db_session):
    user = await _make_user(db_session, TEST_CLERK_ID)
    await _make_rating(db_session, user.id, "Mathematics", rating=1742, wins=20, losses=12)
    science = await _make_rating(db_session, user.id, "Science", rating=1375, wins=8, losses=10)

    # Mutate Mathematics only.
    math_row = (await db_session.execute(
        select(CompetitiveSubjectRating).where(
            CompetitiveSubjectRating.user_id == user.id,
            CompetitiveSubjectRating.subject == "Mathematics",
        )
    )).scalar_one()
    math_row.rating = 1800
    await db_session.commit()

    # Science untouched.
    await db_session.refresh(science)
    assert science.rating == 1375

    profile = await battle_service.get_competitive_profile(db_session, user_id=user.id)
    by_subject = {p["subject"]: p for p in profile}
    assert by_subject["Mathematics"]["rating"] == 1800
    assert by_subject["Mathematics"]["league"] == "Platinum"
    assert by_subject["Science"]["rating"] == 1375
    assert by_subject["Science"]["league"] == "Silver"


async def test_profile_empty_for_user_with_no_ratings(db_session):
    user = await _make_user(db_session, TEST_CLERK_ID)
    profile = await battle_service.get_competitive_profile(db_session, user_id=user.id)
    assert profile == []


async def test_duplicate_user_subject_rating_raises_integrity_error(db_session):
    """DB-level enforcement of uq_competitive_rating_user_subject -- a
    second CompetitiveSubjectRating row for the same (user_id, subject) must
    never be allowed to commit, independent of any app-level
    check-then-insert logic (battle_service.get_or_create_subject_rating).
    """
    user = await _make_user(db_session, TEST_CLERK_ID)
    await _make_rating(db_session, user.id, "Mathematics")

    db_session.add(
        CompetitiveSubjectRating(
            user_id=user.id, subject="Mathematics", rating=1000, highest_rating=1000
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


# ─────────────────────────────────────────────────────────────────────────────
# Leaderboard ordering + tie-breaks
# ─────────────────────────────────────────────────────────────────────────────

async def test_leaderboard_orders_by_rating_desc(db_session):
    u1 = await _make_user(db_session, "user-1")
    u2 = await _make_user(db_session, "user-2")
    u3 = await _make_user(db_session, "user-3")
    await _make_rating(db_session, u1.id, "Mathematics", rating=1500)
    await _make_rating(db_session, u2.id, "Mathematics", rating=1800)
    await _make_rating(db_session, u3.id, "Mathematics", rating=1200)

    result = await battle_service.get_leaderboard(
        db_session, subject="Mathematics", page=1, page_size=20, requesting_user_id=None,
    )
    ratings = [e["rating"] for e in result["entries"]]
    assert ratings == [1800, 1500, 1200]
    assert [e["rank"] for e in result["entries"]] == [1, 2, 3]


async def test_leaderboard_tie_break_by_wins_then_response_time(db_session):
    u1 = await _make_user(db_session, "user-1")
    u2 = await _make_user(db_session, "user-2")
    u3 = await _make_user(db_session, "user-3")
    # Same rating -- wins breaks the tie.
    await _make_rating(db_session, u1.id, "Science", rating=1500, wins=5, average_response_time=10.0)
    await _make_rating(db_session, u2.id, "Science", rating=1500, wins=10, average_response_time=8.0)
    # Same rating AND wins -- average_response_time (ascending) breaks it.
    await _make_rating(db_session, u3.id, "Science", rating=1500, wins=10, average_response_time=5.0)

    result = await battle_service.get_leaderboard(
        db_session, subject="Science", page=1, page_size=20, requesting_user_id=None,
    )
    user_ids = [e["user_id"] for e in result["entries"]]
    assert user_ids == [u3.id, u2.id, u1.id]


async def test_leaderboard_subjects_are_isolated(db_session):
    user = await _make_user(db_session, TEST_CLERK_ID)
    await _make_rating(db_session, user.id, "Mathematics", rating=1500)
    other = await _make_user(db_session, "other-user")
    await _make_rating(db_session, other.id, "English", rating=2000)

    result = await battle_service.get_leaderboard(
        db_session, subject="Mathematics", page=1, page_size=20, requesting_user_id=None,
    )
    assert result["total_count"] == 1
    assert result["entries"][0]["subject"] == "Mathematics"


# ─────────────────────────────────────────────────────────────────────────────
# Pagination
# ─────────────────────────────────────────────────────────────────────────────

async def test_leaderboard_pagination_rank_continues_across_pages(db_session):
    for i in range(5):
        u = await _make_user(db_session, f"page-user-{i}")
        await _make_rating(db_session, u.id, "History", rating=1000 + i * 10)

    page1 = await battle_service.get_leaderboard(
        db_session, subject="History", page=1, page_size=2, requesting_user_id=None,
    )
    page2 = await battle_service.get_leaderboard(
        db_session, subject="History", page=2, page_size=2, requesting_user_id=None,
    )

    assert [e["rank"] for e in page1["entries"]] == [1, 2]
    assert [e["rank"] for e in page2["entries"]] == [3, 4]
    assert page1["total_count"] == 5
    assert page1["total_pages"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# "My rank" off the current page
# ─────────────────────────────────────────────────────────────────────────────

async def test_my_rank_correct_when_off_current_page(db_session):
    me = await _make_user(db_session, TEST_CLERK_ID)
    await _make_rating(db_session, me.id, "Geography", rating=1000)  # will be last place

    for i in range(5):
        u = await _make_user(db_session, f"better-user-{i}")
        await _make_rating(db_session, u.id, "Geography", rating=1500 + i * 10)

    result = await battle_service.get_leaderboard(
        db_session, subject="Geography", page=1, page_size=2, requesting_user_id=me.id,
    )
    assert me.id not in [e["user_id"] for e in result["entries"]]
    assert result["my_rank"] == 6
    assert result["my_rating"] == 1000


async def test_my_rank_none_when_user_has_no_row_for_subject(db_session):
    me = await _make_user(db_session, TEST_CLERK_ID)
    other = await _make_user(db_session, "someone-else")
    await _make_rating(db_session, other.id, "Programming", rating=1200)

    result = await battle_service.get_leaderboard(
        db_session, subject="Programming", page=1, page_size=20, requesting_user_id=me.id,
    )
    assert result["my_rank"] is None
    assert result["my_rating"] is None

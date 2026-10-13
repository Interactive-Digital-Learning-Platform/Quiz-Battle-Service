"""
tests/test_battle_analytics.py
─────────────────────────────────
Integration tests (real in-memory DB) for
app/services/battle_analytics_service.py and GET /api/v1/battle/analytics.
Matches are inserted directly as finished rows rather than played through
battle_gameplay_service -- these tests are about the aggregation, not the
match lifecycle.
"""
from datetime import datetime, timedelta, timezone

from app.core.config import settings
from app.models.battle_match import BattleAnswer, BattleMatch, BattleParticipant
from app.models.question import Question
from app.services import battle_analytics_service
from app.services.user_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
_OPPOSITE = {"win": "loss", "loss": "win", "draw": "draw", "forfeit": "win"}


async def _users(db):
    me = await get_or_create_user(db, TEST_CLERK_ID)
    other = await get_or_create_user(db, "user_opponent")
    return me, other


async def _question(db, *, subject="Mathematics", lesson="Algebra") -> Question:
    q = Question(
        question=f"{lesson} question {datetime.now().timestamp()}",
        options=["a", "b", "c", "d"], correct_answer="a",
        subject=subject, lesson=lesson, difficulty="easy",
    )
    db.add(q)
    await db.commit()
    await db.refresh(q)
    return q


async def _match(
    db, *, me_id, opp_id, result="win", subject="Mathematics", status="completed",
    minutes=0, rating_before=1000, rating_after=None, opp_rating_before=1000,
    final_score=500, answers=(),
) -> BattleMatch:
    """A finished match. `answers` are (question, is_correct, response_ms,
    base, speed, streak) tuples recorded for `me`."""
    match = BattleMatch(
        subject=subject, status=status, difficulty="easy",
        started_at=T0 + timedelta(minutes=minutes),
        finished_at=T0 + timedelta(minutes=minutes, seconds=300),
    )
    db.add(match)
    await db.flush()
    if rating_after is None:
        rating_after = rating_before + {"win": 16, "loss": -16, "forfeit": -16}.get(result, 0)
    db.add_all([
        BattleParticipant(
            match_id=match.id, user_id=me_id, rating_before=rating_before,
            rating_after=rating_after, rating_change=rating_after - rating_before,
            final_score=final_score, result=result,
        ),
        BattleParticipant(
            match_id=match.id, user_id=opp_id, rating_before=opp_rating_before,
            rating_after=opp_rating_before, rating_change=0,
            final_score=300, result=_OPPOSITE[result],
        ),
    ])
    for q, is_correct, ms, base, speed, streak in answers:
        db.add(BattleAnswer(
            match_id=match.id, question_id=q.id, user_id=me_id, selected_option="a",
            is_correct=is_correct, response_time_ms=ms, base_score=base,
            speed_bonus=speed, streak_bonus=streak, total_question_score=base + speed + streak,
        ))
    await db.commit()
    return match


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────

async def test_empty_for_user_without_battles(db_session):
    me, _ = await _users(db_session)
    data = await battle_analytics_service.get_battle_analytics(db_session, user_id=me.id)

    assert data["matches_played"] == 0
    assert data["win_rate"] == 0.0
    assert data["accuracy"] == 0.0
    assert data["recent_form"] == []
    assert data["rating_history"] == []
    assert data["topics"] == []
    assert data["vs_stronger"] == {"played": 0, "wins": 0}
    # All four speed tiers are always present so the client can draw a stable chart.
    assert [t["tier"] for t in data["speed_profile"]] == ["fast", "quick", "steady", "slow"]
    assert all(t["answered"] == 0 for t in data["speed_profile"])


async def test_record_win_rate_and_recent_form_oldest_first(db_session):
    me, other = await _users(db_session)
    for i, result in enumerate(["win", "loss", "win", "draw", "forfeit"]):
        await _match(db_session, me_id=me.id, opp_id=other.id, result=result, minutes=i * 10, final_score=100 * (i + 1))

    data = await battle_analytics_service.get_battle_analytics(db_session, user_id=me.id)

    assert (data["matches_played"], data["wins"], data["losses"], data["draws"], data["forfeits"]) == (5, 2, 1, 1, 1)
    assert data["win_rate"] == 40.0
    assert data["average_score"] == 300.0
    assert data["recent_form"] == ["win", "loss", "win", "draw", "forfeit"]


async def test_recent_form_keeps_only_the_latest_results(db_session):
    me, other = await _users(db_session)
    limit = settings.BATTLE_ANALYTICS_RECENT_FORM_LIMIT
    await _match(db_session, me_id=me.id, opp_id=other.id, result="loss", minutes=0)
    for i in range(limit):
        await _match(db_session, me_id=me.id, opp_id=other.id, result="win", minutes=(i + 1) * 10)

    data = await battle_analytics_service.get_battle_analytics(db_session, user_id=me.id)

    assert len(data["recent_form"]) == limit
    assert "loss" not in data["recent_form"]  # the oldest match fell off


async def test_cancelled_matches_and_their_answers_are_ignored(db_session):
    me, other = await _users(db_session)
    q = await _question(db_session)
    await _match(db_session, me_id=me.id, opp_id=other.id, result="win", answers=[(q, True, 3000, 100, 30, 0)])
    await _match(
        db_session, me_id=me.id, opp_id=other.id, result="loss", status="cancelled", minutes=10,
        answers=[(q, False, 20000, 0, 0, 0)],
    )

    data = await battle_analytics_service.get_battle_analytics(db_session, user_id=me.id)

    assert data["matches_played"] == 1
    assert data["answered"] == 1
    assert data["correct"] == 1


async def test_rating_history_is_per_subject_oldest_first_most_played_first(db_session):
    me, other = await _users(db_session)
    await _match(db_session, me_id=me.id, opp_id=other.id, subject="Science", rating_before=1000, rating_after=1020, minutes=0)
    await _match(db_session, me_id=me.id, opp_id=other.id, subject="Mathematics", rating_before=1000, rating_after=1016, minutes=10)
    await _match(db_session, me_id=me.id, opp_id=other.id, subject="Mathematics", rating_before=1016, rating_after=1000, result="loss", minutes=20)

    data = await battle_analytics_service.get_battle_analytics(db_session, user_id=me.id)

    assert [h["subject"] for h in data["rating_history"]] == ["Mathematics", "Science"]
    maths = data["rating_history"][0]["points"]
    assert [p["rating"] for p in maths] == [1016, 1000]
    assert [p["rating_change"] for p in maths] == [16, -16]


async def test_rating_history_is_capped_to_the_latest_points(db_session):
    me, other = await _users(db_session)
    limit = settings.BATTLE_ANALYTICS_RATING_HISTORY_LIMIT
    for i in range(limit + 3):
        await _match(db_session, me_id=me.id, opp_id=other.id, rating_before=1000 + i, rating_after=1001 + i, minutes=i)

    data = await battle_analytics_service.get_battle_analytics(db_session, user_id=me.id)

    points = data["rating_history"][0]["points"]
    assert len(points) == limit
    assert points[-1]["rating"] == 1001 + limit + 2  # newest kept, oldest dropped


async def test_speed_profile_uses_the_speed_bonus_boundaries(db_session):
    me, other = await _users(db_session)
    q = await _question(db_session)
    t1 = int(settings.BATTLE_SPEED_BONUS_TIER1_MAX_SECONDS * 1000)
    t3 = int(settings.BATTLE_SPEED_BONUS_TIER3_MAX_SECONDS * 1000)
    answers = [
        (q, True, t1, 100, 30, 0),       # exactly on the boundary -> still "fast" (inclusive, like scoring)
        (q, False, t1 + 1, 0, 0, 0),     # just over -> "quick"
        (q, True, t3 + 1, 100, 0, 0),    # past the last bound -> "slow"
    ]
    # One answer per question per match (DB constraint), so spread them over matches.
    for i, a in enumerate(answers):
        await _match(db_session, me_id=me.id, opp_id=other.id, minutes=i, answers=[a])

    data = await battle_analytics_service.get_battle_analytics(db_session, user_id=me.id)
    tiers = {t["tier"]: (t["answered"], t["correct"]) for t in data["speed_profile"]}

    assert tiers == {"fast": (1, 1), "quick": (1, 0), "steady": (0, 0), "slow": (1, 1)}


async def test_points_breakdown_and_accuracy(db_session):
    me, other = await _users(db_session)
    q1 = await _question(db_session, lesson="Algebra")
    q2 = await _question(db_session, lesson="Geometry")
    await _match(db_session, me_id=me.id, opp_id=other.id, answers=[
        (q1, True, 3000, 100, 30, 0),
        (q2, True, 7000, 100, 20, 5),
    ])
    await _match(db_session, me_id=me.id, opp_id=other.id, minutes=10, answers=[(q1, False, 12000, 0, 0, 0)])

    data = await battle_analytics_service.get_battle_analytics(db_session, user_id=me.id)

    assert data["points"] == {"base": 200, "speed_bonus": 50, "streak_bonus": 5}
    assert (data["answered"], data["correct"]) == (3, 2)
    assert data["accuracy"] == 66.7


async def test_topics_need_minimum_answers_and_sort_strongest_first(db_session):
    me, other = await _users(db_session)
    min_n = settings.BATTLE_ANALYTICS_TOPIC_MIN_ANSWERS
    algebra = await _question(db_session, lesson="Algebra")
    geometry = await _question(db_session, lesson="Geometry")
    rare = await _question(db_session, lesson="Statistics")
    for i in range(min_n):
        await _match(db_session, me_id=me.id, opp_id=other.id, minutes=i, answers=[
            (algebra, i == 0, 8000, 100 if i == 0 else 0, 0, 0),  # 1 of min_n right
            (geometry, True, 4000, 100, 30, 0),                     # all right
        ])
    await _match(db_session, me_id=me.id, opp_id=other.id, minutes=99, answers=[(rare, True, 4000, 100, 30, 0)])

    data = await battle_analytics_service.get_battle_analytics(db_session, user_id=me.id)

    assert [t["lesson"] for t in data["topics"]] == ["Geometry", "Algebra"]  # Statistics: too few answers
    assert data["topics"][0]["accuracy"] == 100.0
    assert data["topics"][1]["answered"] == min_n


async def test_vs_stronger_counts_only_higher_rated_opponents(db_session):
    me, other = await _users(db_session)
    await _match(db_session, me_id=me.id, opp_id=other.id, result="win", rating_before=1000, opp_rating_before=1100)
    await _match(db_session, me_id=me.id, opp_id=other.id, result="loss", rating_before=1000, opp_rating_before=1200, minutes=10)
    await _match(db_session, me_id=me.id, opp_id=other.id, result="win", rating_before=1000, opp_rating_before=900, minutes=20)
    await _match(db_session, me_id=me.id, opp_id=other.id, result="win", rating_before=1000, opp_rating_before=1000, minutes=30)

    data = await battle_analytics_service.get_battle_analytics(db_session, user_id=me.id)

    assert data["vs_stronger"] == {"played": 2, "wins": 1}


async def test_unanswered_question_slots_are_not_counted(db_session):
    # Grading writes a selected_option="" row for every slot the player never
    # answered; those must not show up as answered/wrong or in the speed bands.
    me, other = await _users(db_session)
    answered_q = await _question(db_session, lesson="Algebra")
    missed_q = await _question(db_session, lesson="Algebra")
    match = await _match(db_session, me_id=me.id, opp_id=other.id, answers=[(answered_q, True, 3000, 100, 30, 0)])
    db_session.add(BattleAnswer(
        match_id=match.id, question_id=missed_q.id, user_id=me.id, selected_option="",
        is_correct=False, response_time_ms=30000, base_score=0, speed_bonus=0,
        streak_bonus=0, total_question_score=0,
    ))
    await db_session.commit()

    data = await battle_analytics_service.get_battle_analytics(db_session, user_id=me.id)

    assert (data["answered"], data["correct"]) == (1, 1)
    assert next(t for t in data["speed_profile"] if t["tier"] == "slow")["answered"] == 0


async def test_only_the_requesting_users_answers_count(db_session):
    me, other = await _users(db_session)
    q = await _question(db_session)
    match = await _match(db_session, me_id=me.id, opp_id=other.id, answers=[(q, True, 3000, 100, 30, 0)])
    db_session.add(BattleAnswer(
        match_id=match.id, question_id=q.id, user_id=other.id, selected_option="b",
        is_correct=False, response_time_ms=20000, base_score=0, speed_bonus=0,
        streak_bonus=0, total_question_score=0,
    ))
    await db_session.commit()

    data = await battle_analytics_service.get_battle_analytics(db_session, user_id=me.id)

    assert (data["answered"], data["correct"]) == (1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# GET /battle/analytics
# ─────────────────────────────────────────────────────────────────────────────

async def test_route_returns_analytics_for_the_authenticated_user(client, db_session):
    me, other = await _users(db_session)
    q = await _question(db_session)
    await _match(db_session, me_id=me.id, opp_id=other.id, result="win", answers=[(q, True, 3000, 100, 30, 0)])

    resp = await client.get("/api/v1/battle/analytics")

    assert resp.status_code == 200
    data = resp.json()
    assert data["matches_played"] == 1
    assert data["recent_form"] == ["win"]
    assert data["rating_history"][0]["subject"] == "Mathematics"
    assert len(data["speed_profile"]) == 4


async def test_route_empty_for_brand_new_user(client):
    resp = await client.get("/api/v1/battle/analytics")

    assert resp.status_code == 200
    assert resp.json()["matches_played"] == 0

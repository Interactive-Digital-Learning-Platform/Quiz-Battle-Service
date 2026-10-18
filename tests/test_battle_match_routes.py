"""
tests/test_battle_match_routes.py
──────────────────────────────────────
HTTP-level integration tests for GET /api/v1/battle/match/{id}/state,
POST /api/v1/battle/match/{id}/answer + /skip, and POST /api/v1/battle/match/{id}/ready
-- real DB + fake_redis, real routing, real Pydantic response validation.
Mirrors tests/test_battle_routes.py's style. Ready/forfeit-specific edge
cases beyond activation live in tests/test_battle_ready_forfeit_service.py
and tests/test_battle_ws_routes.py.
"""
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

from sqlalchemy import update

from app.core.config import settings
from app.models.battle_match import BattleMatch, BattleParticipant
from app.models.competitive_rating import CompetitiveSubjectRating
from app.models.question import Question
from app.services import battle_gameplay_service, matchmaking_service
from app.services.user_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID

N = settings.BATTLE_QUESTION_COUNT


def _fixed_questions(count: int = N) -> list[Question]:
    return [
        Question(
            question=f"Route test question {i}?", options=["A", "B", "C", "D"],
            correct_answer="A", subject="Mathematics", lesson="Algebra", difficulty="medium",
        )
        for i in range(count)
    ]


async def _seed_rating(db, user_id: int, subject: str, rating: int) -> CompetitiveSubjectRating:
    row = CompetitiveSubjectRating(user_id=user_id, subject=subject, rating=rating, highest_rating=rating)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def _set_started_at(db, match_id: int, when: datetime) -> None:
    await db.execute(update(BattleMatch).where(BattleMatch.id == match_id).values(started_at=when))
    await db.commit()


async def _create_waiting_match(db, *, me_id: int, other_id: int) -> BattleMatch:
    match = BattleMatch(subject="Mathematics", status="waiting", difficulty="Mostly Medium")
    db.add(match)
    await db.flush()
    db.add(BattleParticipant(match_id=match.id, user_id=me_id, rating_before=1000))
    db.add(BattleParticipant(match_id=match.id, user_id=other_id, rating_before=1000))
    await db.commit()
    await db.refresh(match)
    return match


# ─────────────────────────────────────────────────────────────────────────────
# Access control / basic errors
# ─────────────────────────────────────────────────────────────────────────────

async def test_state_404_for_unknown_match_id(client, db_session, fake_redis):
    resp = await client.get("/api/v1/battle/match/999999/state")
    assert resp.status_code == 404


async def test_state_403_for_non_participant(client, db_session, fake_redis):
    other = await get_or_create_user(db_session, "other-clerk-id")
    third = await get_or_create_user(db_session, "third-user")
    match = BattleMatch(subject="Mathematics", status="waiting", difficulty="Mostly Medium")
    db_session.add(match)
    await db_session.flush()
    db_session.add(BattleParticipant(match_id=match.id, user_id=other.id, rating_before=1000))
    db_session.add(BattleParticipant(match_id=match.id, user_id=third.id, rating_before=1000))
    await db_session.commit()

    resp = await client.get(f"/api/v1/battle/match/{match.id}/state")
    assert resp.status_code == 403


# ─────────────────────────────────────────────────────────────────────────────
# Ready-gated activation ("waiting" -> "countdown" -> "active")
# ─────────────────────────────────────────────────────────────────────────────

async def test_poll_alone_does_not_activate_match(client, db_session, fake_redis):
    me = await get_or_create_user(db_session, TEST_CLERK_ID)
    other = await get_or_create_user(db_session, "other-clerk-id")
    match = await _create_waiting_match(db_session, me_id=me.id, other_id=other.id)

    with patch(
        "app.services.battle_gameplay_service.select_battle_questions",
        new=AsyncMock(side_effect=AssertionError("should not select questions without both readies")),
    ):
        resp = await client.get(f"/api/v1/battle/match/{match.id}/state")

    assert resp.status_code == 200
    assert resp.json()["status"] == "waiting"


async def test_ready_from_both_participants_starts_countdown_then_active(client, db_session, fake_redis):
    me = await get_or_create_user(db_session, TEST_CLERK_ID)
    other = await get_or_create_user(db_session, "other-clerk-id")
    match = await _create_waiting_match(db_session, me_id=me.id, other_id=other.id)

    questions = _fixed_questions()
    db_session.add_all(questions)
    await db_session.commit()

    # Activation requires ready AND WS-connected (see mark_connected) --
    # connect both upfront so this test exercises the ready-flow itself,
    # not the connection gate (that's covered separately).
    await battle_gameplay_service.mark_connected(db_session, fake_redis, match_id=match.id, user_id=me.id)
    await battle_gameplay_service.mark_connected(db_session, fake_redis, match_id=match.id, user_id=other.id)

    # Only one side ready so far -- must stay "waiting".
    ready_resp = await client.post(f"/api/v1/battle/match/{match.id}/ready")
    assert ready_resp.status_code == 200
    assert ready_resp.json()["status"] == "waiting"

    # The second (non-HTTP) participant readies up -- this is what actually
    # triggers question selection + "waiting"->"countdown".
    with patch(
        "app.services.battle_gameplay_service.select_battle_questions",
        new=AsyncMock(return_value=questions),
    ):
        await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=other.id)

    countdown_resp = await client.get(f"/api/v1/battle/match/{match.id}/state")
    assert countdown_resp.json()["status"] == "countdown"
    assert countdown_resp.json()["countdown_seconds_remaining"] > 0

    # Backdate started_at so the lazy "countdown"->"active" check fires.
    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=1))
    active_resp = await client.get(f"/api/v1/battle/match/{match.id}/state")
    active_data = active_resp.json()
    assert active_data["status"] == "active"
    assert active_data["question_index"] == 0
    assert active_data["question"]["question_id"] == questions[0].id


# ─────────────────────────────────────────────────────────────────────────────
# Answer submission
# ─────────────────────────────────────────────────────────────────────────────

async def test_answer_endpoint_never_leaks_correct_answer(client, db_session, fake_redis):
    me = await get_or_create_user(db_session, TEST_CLERK_ID)
    other = await get_or_create_user(db_session, "other-clerk-id")
    match = await _create_waiting_match(db_session, me_id=me.id, other_id=other.id)

    questions = _fixed_questions()
    db_session.add_all(questions)
    await db_session.commit()

    await battle_gameplay_service.mark_connected(db_session, fake_redis, match_id=match.id, user_id=me.id)
    await battle_gameplay_service.mark_connected(db_session, fake_redis, match_id=match.id, user_id=other.id)
    await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match.id, user_id=other.id)
    with patch(
        "app.services.battle_gameplay_service.select_battle_questions",
        new=AsyncMock(return_value=questions),
    ):
        await client.post(f"/api/v1/battle/match/{match.id}/ready")

    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=1))
    state_resp = await client.get(f"/api/v1/battle/match/{match.id}/state")
    assert "correct" not in json.dumps(state_resp.json()).replace('"is_correct": null', "")

    answer_resp = await client.post(
        f"/api/v1/battle/match/{match.id}/answer",
        json={"question_id": questions[0].id, "selected_option": "A"},
    )
    assert answer_resp.status_code == 200
    assert "correct" not in json.dumps(answer_resp.json())
    # The response is the next question -- handed out only after the answer was saved.
    assert answer_resp.json()["question_index"] == 1
    assert answer_resp.json()["question"]["question_id"] == questions[1].id

    skip_resp = await client.post(f"/api/v1/battle/match/{match.id}/skip", json={"question_id": questions[1].id})
    assert skip_resp.status_code == 200
    assert skip_resp.json()["question_index"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# Full end-to-end flow: matchmaking -> ready -> activation -> answers -> completion -> profile
# ─────────────────────────────────────────────────────────────────────────────

async def test_full_end_to_end_battle_flow(client, db_session, fake_redis):
    me = await get_or_create_user(db_session, TEST_CLERK_ID)
    other = await get_or_create_user(db_session, "other-clerk-id")
    await _seed_rating(db_session, me.id, "Mathematics", 1000)
    await _seed_rating(db_session, other.id, "Mathematics", 1000)

    # Seed the opponent's queue presence directly (no second real HTTP
    # identity needed), same precedent as test_battle_queue_routes.py.
    await matchmaking_service.join_queue(db_session, fake_redis, user_id=other.id, subject="Mathematics")

    join_resp = await client.post("/api/v1/battle/queue/join", json={"subject": "Mathematics"})
    assert join_resp.status_code == 200
    match_id = join_resp.json()["match_id"]
    assert match_id is not None

    questions = _fixed_questions()
    db_session.add_all(questions)
    await db_session.commit()

    await battle_gameplay_service.mark_connected(db_session, fake_redis, match_id=match_id, user_id=me.id)
    await battle_gameplay_service.mark_connected(db_session, fake_redis, match_id=match_id, user_id=other.id)
    await battle_gameplay_service.mark_ready(db_session, fake_redis, match_id=match_id, user_id=other.id)
    with patch(
        "app.services.battle_gameplay_service.select_battle_questions",
        new=AsyncMock(return_value=questions),
    ):
        ready_resp = await client.post(f"/api/v1/battle/match/{match_id}/ready")
    assert ready_resp.json()["status"] == "countdown"

    await _set_started_at(db_session, match_id, datetime.now(UTC) - timedelta(seconds=1))
    state_resp = await client.get(f"/api/v1/battle/match/{match_id}/state")
    assert state_resp.json()["status"] == "active"

    # "me" answers every question correctly at their own pace; the opponent
    # never answers, so the match only ends once its clock runs out.
    for i in range(N):
        answer_resp = await client.post(
            f"/api/v1/battle/match/{match_id}/answer",
            json={"question_id": questions[i].id, "selected_option": "A"},
        )
        assert answer_resp.status_code == 200
        assert answer_resp.json()["question_index"] == i + 1
    assert answer_resp.json()["finished"] is True
    assert answer_resp.json()["question"] is None

    await _set_started_at(
        db_session, match_id,
        datetime.now(UTC) - timedelta(
            seconds=settings.BATTLE_MATCH_DURATION_SECONDS + settings.BATTLE_ANSWER_LATE_GRACE_SECONDS
            + settings.BATTLE_NETWORK_ALLOWANCE_MAX_SECONDS + 1
        ),
    )
    final_state_resp = await client.get(f"/api/v1/battle/match/{match_id}/state")
    assert final_state_resp.status_code == 200
    final_data = final_state_resp.json()
    assert final_data["status"] == "completed"
    assert final_data["result"]["winner_user_id"] == me.id
    assert [r["status"] for r in final_data["review"]] == ["correct"] * N

    profile_resp = await client.get("/api/v1/battle/profile")
    assert profile_resp.status_code == 200
    my_math = next(s for s in profile_resp.json()["subjects"] if s["subject"] == "Mathematics")
    assert my_math["wins"] == 1
    assert my_math["matches_played"] == 1
    assert my_math["rating"] > 1000

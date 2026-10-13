"""
tests/test_battle_fair_play.py
──────────────────────────────
- Forfeiting during the pre-start countdown voids the match (no rating change);
  once the countdown has run out it's a scored forfeit as before.
- A match both players walked away from is still settled once its time is up,
  so the player who was ahead keeps the win; only a match nobody played at all
  is cancelled.
- Memorisation: option order is shuffled per match (same for both players),
  and running out of never-seen questions grows the pool.
"""
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update

from app.core.config import settings
from app.models.battle_match import BattleMatch
from app.models.competitive_rating import CompetitiveSubjectRating
from app.models.question import Question
from app.services import battle_gameplay_service, battle_question_service
from tests.test_battle_gameplay_service import (
    TOTAL,
    _create_waiting_match,
    _make_user,
    _seed_and_activate,
    _set_started_at,
    _submit_and_grade,
)

# ─────────────────────────────────────────────────────────────────────────────
# Countdown forfeit
# ─────────────────────────────────────────────────────────────────────────────


async def _rating(db, user_id: int) -> int | None:
    row = (
        await db.execute(
            select(CompetitiveSubjectRating.rating).where(
                CompetitiveSubjectRating.user_id == user_id, CompetitiveSubjectRating.subject == "Mathematics"
            )
        )
    ).scalar_one_or_none()
    return row


async def _match_in_countdown(db, *, starts_in_seconds: float):
    a = await _make_user(db, "user-a")
    b = await _make_user(db, "user-b")
    match = await _create_waiting_match(
        db, subject="Mathematics", difficulty="Mostly Medium", user_a_id=a.id, rating_a=1000, user_b_id=b.id, rating_b=1000,
    )
    await db.execute(
        update(BattleMatch).where(BattleMatch.id == match.id).values(
            status="countdown", started_at=datetime.now(UTC) + timedelta(seconds=starts_in_seconds)
        )
    )
    await db.commit()
    await db.refresh(match)
    return match, a, b


async def test_forfeit_during_countdown_voids_the_match(db_session, fake_redis):
    match, a, b = await _match_in_countdown(db_session, starts_in_seconds=4)

    result = await battle_gameplay_service.forfeit_match(db_session, fake_redis, match_id=match.id, forfeiting_user_id=a.id)

    assert result.status == "cancelled"
    assert result.cancel_reason == battle_gameplay_service.CANCEL_REASON_OPPONENT_LEFT_BEFORE_START
    assert result.winner_user_id is None
    assert await _rating(db_session, a.id) is None  # no rating row touched or created
    assert await _rating(db_session, b.id) is None


async def test_forfeit_after_countdown_ran_out_is_still_scored(db_session, fake_redis):
    # started_at already passed -- the match is effectively active (the flip to "active" is lazy).
    match, a, b = await _match_in_countdown(db_session, starts_in_seconds=-2)

    result = await battle_gameplay_service.forfeit_match(db_session, fake_redis, match_id=match.id, forfeiting_user_id=a.id)

    assert result.status == "completed"
    assert result.winner_user_id == b.id


# ─────────────────────────────────────────────────────────────────────────────
# Abandoned-at-the-end matches get settled
# ─────────────────────────────────────────────────────────────────────────────


async def test_reaper_settles_a_finished_match_nobody_stayed_for(db_session, fake_redis):
    a = await _make_user(db_session, "user-a")
    b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=a.id, rating_a=1000, user_b_id=b.id, rating_b=1000,
    )
    _, questions = await _seed_and_activate(db_session, fake_redis, match, user_a_id=a.id, user_b_id=b.id)
    # A gets the first question right, then both players close the app; the
    # progression timer never grades the rest.
    await _submit_and_grade(db_session, fake_redis, match, order=0, user_id=a.id, question_id=questions[0].id, selected_option="A")
    await fake_redis.hset(f"battle:user_state:{a.id}", "state", "in_match")
    await fake_redis.hset(f"battle:user_state:{b.id}", "state", "in_match")
    await _set_started_at(
        db_session, match.id,
        datetime.now(UTC) - timedelta(seconds=TOTAL + settings.BATTLE_FINISHED_MATCH_SETTLE_BUFFER_SECONDS + 1),
    )

    reaped = await battle_gameplay_service.run_stale_match_reap_once(db_session, fake_redis)

    await db_session.refresh(match)
    assert reaped == 1
    assert match.status == "completed"
    assert match.winner_user_id == a.id
    assert await _rating(db_session, a.id) > settings.BATTLE_INITIAL_RATING
    assert await fake_redis.exists(f"battle:user_state:{a.id}") == 0
    assert await fake_redis.exists(f"battle:user_state:{b.id}") == 0


async def test_reaper_leaves_a_match_alone_until_its_time_is_up(db_session, fake_redis):
    a = await _make_user(db_session, "user-a")
    b = await _make_user(db_session, "user-b")
    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=a.id, rating_a=1000, user_b_id=b.id, rating_b=1000,
    )
    _, questions = await _seed_and_activate(db_session, fake_redis, match, user_a_id=a.id, user_b_id=b.id)
    await _submit_and_grade(db_session, fake_redis, match, order=0, user_id=a.id, question_id=questions[0].id, selected_option="A")
    await _set_started_at(db_session, match.id, datetime.now(UTC) - timedelta(seconds=TOTAL - 5))

    assert await battle_gameplay_service.run_stale_match_reap_once(db_session, fake_redis) == 0
    await db_session.refresh(match)
    assert match.status == "active"


# ─────────────────────────────────────────────────────────────────────────────
# Memorisation
# ─────────────────────────────────────────────────────────────────────────────


def test_option_order_is_per_match_and_shared_by_both_players():
    question = Question(id=42, question="Q?", options=["A", "B", "C", "D"], correct_answer="A",
                        subject="Mathematics", lesson="L", difficulty="medium")
    orders = {tuple(battle_gameplay_service._match_option_order(BattleMatch(id=m), question)) for m in range(1, 40)}

    assert len(orders) > 1  # not a fixed order across matches
    assert all(sorted(o) == ["A", "B", "C", "D"] for o in orders)  # same options, just reordered
    same_match = BattleMatch(id=7)
    assert battle_gameplay_service._match_option_order(same_match, question) == (
        battle_gameplay_service._match_option_order(same_match, question)  # stable for both players / reconnects
    )


async def test_running_out_of_unseen_questions_grows_the_pool(db_session, monkeypatch):
    calls = []
    monkeypatch.setattr(
        battle_question_service, "_kickoff_pool_replenish",
        lambda subject, difficulty, *, grow=False: calls.append((subject, difficulty, grow)),
    )
    seen = [
        Question(question=text, options=["A", "B", "C", "D"], correct_answer="A",
                 subject="Mathematics", lesson="Algebra", difficulty="medium")
        for text in ("Solve 2x + 3 = 7.", "Name the capital of France.", "What gas do plants absorb?")
    ]
    db_session.add_all(seen)
    await db_session.commit()

    _picked, repeats = await battle_question_service._select_for_one_tier(
        db_session, subject="Mathematics", difficulty="medium", count=3,
        history_ids={q.id for q in seen}, history_texts=[q.question for q in seen],
        session_ids=set(), session_texts=[],
    )

    assert repeats == 3  # all already seen -- served as a last resort
    assert ("Mathematics", "medium", True) in calls

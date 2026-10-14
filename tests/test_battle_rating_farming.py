"""
tests/test_battle_rating_farming.py
───────────────────────────────────
Anti-farming: only the first BATTLE_REPEAT_OPPONENT_RATED_LIMIT matches
between the same two players within BATTLE_REPEAT_OPPONENT_WINDOW_HOURS move
ratings. Later ones still record the result but leave ratings unchanged --
whether they end normally or by forfeit.
"""
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update

from app.core.config import settings
from app.models.battle_match import BattleMatch, BattleParticipant
from app.models.competitive_rating import CompetitiveSubjectRating
from app.services import battle_gameplay_service
from tests.test_battle_gameplay_service import (
    _complete_match_with_winner,
    _create_waiting_match,
    _make_user,
    _seed_and_activate,
)


async def _rating(db, user_id: int, subject: str = "Mathematics") -> CompetitiveSubjectRating:
    row = (
        await db.execute(
            select(CompetitiveSubjectRating).where(
                CompetitiveSubjectRating.user_id == user_id, CompetitiveSubjectRating.subject == subject
            )
        )
    ).scalar_one()
    await db.refresh(row)
    return row


async def _participant(db, match_id: int, user_id: int) -> BattleParticipant:
    row = (
        await db.execute(
            select(BattleParticipant).where(BattleParticipant.match_id == match_id, BattleParticipant.user_id == user_id)
        )
    ).scalar_one()
    await db.refresh(row)
    return row


async def test_repeat_matches_past_the_limit_leave_ratings_unchanged(db_session, fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "BATTLE_REPEAT_OPPONENT_RATED_LIMIT", 2)
    main = await _make_user(db_session, "main-account")
    alt = await _make_user(db_session, "alt-account")

    for _ in range(2):
        await _complete_match_with_winner(db_session, fake_redis, subject="Mathematics", winner_id=main.id, loser_id=alt.id)
    rated_main = (await _rating(db_session, main.id)).rating
    assert rated_main > 1000  # the first two counted

    third = await _complete_match_with_winner(
        db_session, fake_redis, subject="Mathematics", winner_id=main.id, loser_id=alt.id
    )

    main_rating = await _rating(db_session, main.id)
    assert main_rating.rating == rated_main
    assert main_rating.wins == 3  # the result itself is still recorded
    me = await _participant(db_session, third.id, main.id)
    assert me.result == "win" and me.rating_after == rated_main


async def test_limit_counts_only_this_pair_within_the_window(db_session, fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "BATTLE_REPEAT_OPPONENT_RATED_LIMIT", 1)
    main = await _make_user(db_session, "main-account")
    alt = await _make_user(db_session, "alt-account")
    stranger = await _make_user(db_session, "stranger")

    first = await _complete_match_with_winner(db_session, fake_redis, subject="Mathematics", winner_id=main.id, loser_id=alt.id)
    # A different opponent is unaffected by the main/alt history.
    vs_stranger = await _complete_match_with_winner(
        db_session, fake_redis, subject="Mathematics", winner_id=main.id, loser_id=stranger.id
    )
    assert (await _participant(db_session, vs_stranger.id, main.id)).rating_change > 0

    # Once the earlier main/alt match ages out of the window, they're rated again.
    await db_session.execute(
        update(BattleMatch)
        .where(BattleMatch.id == first.id)
        .values(finished_at=datetime.now(UTC) - timedelta(hours=settings.BATTLE_REPEAT_OPPONENT_WINDOW_HOURS + 1))
    )
    await db_session.commit()
    again = await _complete_match_with_winner(db_session, fake_redis, subject="Mathematics", winner_id=main.id, loser_id=alt.id)
    assert (await _participant(db_session, again.id, main.id)).rating_change > 0


async def test_repeat_forfeits_past_the_limit_leave_ratings_unchanged(db_session, fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "BATTLE_REPEAT_OPPONENT_RATED_LIMIT", 1)
    main = await _make_user(db_session, "main-account")
    alt = await _make_user(db_session, "alt-account")
    await _complete_match_with_winner(db_session, fake_redis, subject="Mathematics", winner_id=main.id, loser_id=alt.id)
    before = (await _rating(db_session, main.id)).rating

    match = await _create_waiting_match(
        db_session, subject="Mathematics", difficulty="Mostly Medium",
        user_a_id=main.id, rating_a=before, user_b_id=alt.id, rating_b=1000,
    )
    await _seed_and_activate(db_session, fake_redis, match, user_a_id=main.id, user_b_id=alt.id)
    forfeited = await battle_gameplay_service.forfeit_match(
        db_session, fake_redis, match_id=match.id, forfeiting_user_id=alt.id
    )

    assert forfeited.status == "completed" and forfeited.winner_user_id == main.id
    assert (await _rating(db_session, main.id)).rating == before
    assert (await _participant(db_session, match.id, alt.id)).result == "forfeit"

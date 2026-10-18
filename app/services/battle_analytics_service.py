"""
app/services/battle_analytics_service.py
───────────────────────────────────────────
Read-only analytics over one user's completed battles, for the mobile
Profile's "Battles" section (GET /battle/analytics). Everything here is
derived from rows battle_gameplay_service already writes -- no new state:

- record / recent form / rating history / stronger-opponent results come
  from BattleParticipant (one row per player per match);
- the speed profile, points breakdown and topic strengths come from
  BattleAnswer, joined to Question for each answer's lesson;
- question usage (answered / skipped / never reached) and how matches
  ended come from BattleMatchQuestion, BattleAnswer and each participant's
  self-paced progress (current_question_index, finished_at).

Only status="completed" matches count (forfeits included -- they reuse
"completed"); cancelled matches never applied ratings, same rule as
get_battle_history. Answer-level stats only cover questions the player
actually answered (see _answers_base).
"""
from datetime import UTC, datetime

from sqlalchemy import and_, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.core.config import settings
from app.models.battle_match import (
    BattleAnswer,
    BattleMatch,
    BattleMatchQuestion,
    BattleParticipant,
)
from app.models.question import Question

_EPOCH = datetime.min.replace(tzinfo=UTC)


def _speed_tiers() -> list[tuple[str, float | None]]:
    """(key, inclusive upper bound in seconds) per speed-bonus tier, fastest
    first -- the same boundaries battle_scoring_engine uses to award the
    speed bonus, so "fast" here means "earned the top bonus"."""
    return [
        ("fast", settings.BATTLE_SPEED_BONUS_TIER1_MAX_SECONDS),
        ("quick", settings.BATTLE_SPEED_BONUS_TIER2_MAX_SECONDS),
        ("steady", settings.BATTLE_SPEED_BONUS_TIER3_MAX_SECONDS),
        ("slow", None),
    ]


def _speed_tier_expr():
    tiers = _speed_tiers()
    whens = [
        (BattleAnswer.response_time_ms <= int(max_s * 1000), key)
        for key, max_s in tiers
        if max_s is not None
    ]
    return case(*whens, else_=tiers[-1][0])


def _sort_key(finished_at: datetime | None, match_id: int) -> tuple[datetime, int]:
    # SQLite (tests) hands back naive datetimes; Postgres aware ones.
    if finished_at is None:
        return (_EPOCH, match_id)
    if finished_at.tzinfo is None:
        finished_at = finished_at.replace(tzinfo=UTC)
    return (finished_at, match_id)


async def _participations(db: AsyncSession, *, user_id: int) -> list:
    """One row per completed match the user played, oldest first, with the
    opponent's pre-match rating alongside the user's own result."""
    me = aliased(BattleParticipant)
    opp = aliased(BattleParticipant)
    rows = (
        await db.execute(
            select(
                BattleMatch.id.label("match_id"),
                BattleMatch.subject,
                BattleMatch.started_at,
                BattleMatch.finished_at,
                me.result,
                me.current_question_index.label("my_index"),
                me.finished_at.label("my_finished_at"),
                opp.current_question_index.label("opp_index"),
                opp.result.label("opp_result"),
                me.rating_before,
                me.rating_after,
                me.rating_change,
                me.final_score,
                opp.rating_before.label("opponent_rating_before"),
            )
            .join(me, and_(me.match_id == BattleMatch.id, me.user_id == user_id))
            .outerjoin(opp, and_(opp.match_id == BattleMatch.id, opp.user_id != user_id))
            .where(BattleMatch.status == "completed")
        )
    ).all()
    return sorted(rows, key=lambda r: _sort_key(r.finished_at, r.match_id))


def _answers_base(user_id: int):
    """WHERE shared by every answer-level aggregate: this user's answers in
    completed matches only. A skipped question is saved as a row with the
    selected_option == "" sentinel -- those are filtered out exactly like
    _apply_result does, so a skip doesn't count as answered, as a wrong
    answer, or toward speed (_question_usage counts them on their own)."""
    return (
        BattleAnswer.user_id == user_id,
        BattleMatch.status == "completed",
        BattleAnswer.selected_option != "",
    )


async def _answer_totals(db: AsyncSession, *, user_id: int) -> dict:
    correct = func.sum(case((BattleAnswer.is_correct, 1), else_=0))
    row = (
        await db.execute(
            select(
                func.count(BattleAnswer.id),
                correct,
                func.sum(BattleAnswer.base_score),
                func.sum(BattleAnswer.speed_bonus),
                func.sum(BattleAnswer.streak_bonus),
            )
            .join(BattleMatch, BattleMatch.id == BattleAnswer.match_id)
            .where(*_answers_base(user_id))
        )
    ).one()
    answered, correct_n, base, speed, streak = (int(v or 0) for v in row)
    return {
        "answered": answered,
        "correct": correct_n,
        "points": {"base": base, "speed_bonus": speed, "streak_bonus": streak},
    }


async def _speed_profile(db: AsyncSession, *, user_id: int) -> list[dict]:
    tier = _speed_tier_expr().label("tier")
    rows = (
        await db.execute(
            select(
                tier,
                func.count(BattleAnswer.id),
                func.sum(case((BattleAnswer.is_correct, 1), else_=0)),
            )
            .join(BattleMatch, BattleMatch.id == BattleAnswer.match_id)
            .where(*_answers_base(user_id))
            .group_by(tier)
        )
    ).all()
    by_tier = {key: (int(n or 0), int(c or 0)) for key, n, c in rows}
    # Every tier is always present (zeros included) so the client can draw a
    # stable four-band chart without filling gaps itself.
    return [
        {
            "tier": key,
            "max_seconds": max_s,
            "answered": by_tier.get(key, (0, 0))[0],
            "correct": by_tier.get(key, (0, 0))[1],
        }
        for key, max_s in _speed_tiers()
    ]


async def _topics(db: AsyncSession, *, user_id: int) -> list[dict]:
    answered = func.count(BattleAnswer.id)
    correct = func.sum(case((BattleAnswer.is_correct, 1), else_=0))
    rows = (
        await db.execute(
            select(Question.subject, Question.lesson, answered, correct)
            .join(BattleMatch, BattleMatch.id == BattleAnswer.match_id)
            .join(Question, Question.id == BattleAnswer.question_id)
            .where(*_answers_base(user_id))
            .group_by(Question.subject, Question.lesson)
            .having(answered >= settings.BATTLE_ANALYTICS_TOPIC_MIN_ANSWERS)
        )
    ).all()
    topics = [
        {
            "subject": subject,
            "lesson": lesson,
            "answered": int(n),
            "correct": int(c or 0),
            "accuracy": round(int(c or 0) / int(n) * 100, 1),
        }
        for subject, lesson, n, c in rows
    ]
    # Strongest first; more evidence wins a tie, then a stable name order.
    topics.sort(key=lambda t: (-t["accuracy"], -t["answered"], t["subject"], t["lesson"]))
    return topics


async def _question_usage(db: AsyncSession, *, user_id: int, answered: int) -> dict:
    """What happened to every question this user was dealt in completed
    matches: answered, skipped, or never reached (the clock ran out, or the
    match ended by forfeit, before they got to it -- no answer row at all)."""
    slots = await db.scalar(
        select(func.count(BattleMatchQuestion.id))
        .join(BattleMatch, BattleMatch.id == BattleMatchQuestion.match_id)
        .join(
            BattleParticipant,
            and_(BattleParticipant.match_id == BattleMatch.id, BattleParticipant.user_id == user_id),
        )
        .where(BattleMatch.status == "completed")
    ) or 0
    skipped = await db.scalar(
        select(func.count(BattleAnswer.id))
        .join(BattleMatch, BattleMatch.id == BattleAnswer.match_id)
        .where(BattleAnswer.user_id == user_id, BattleMatch.status == "completed", BattleAnswer.selected_option == "")
    ) or 0
    return {
        "questions": int(slots),
        "answered": answered,
        "skipped": int(skipped),
        "not_reached": max(0, int(slots) - answered - int(skipped)),
    }


def _match_endings(rows: list) -> dict:
    """How this user's matches ended -- same rule as
    battle_gameplay_service.match_end_reason -- and how long they took to
    get through every question when they did. Matches with no progress
    recorded for either player (played before self-paced matches, which
    track it) are left out of the ending counts rather than guessed at."""
    n = settings.BATTLE_QUESTION_COUNT
    endings = {"all_finished": 0, "time_up": 0, "forfeit": 0}
    finish_seconds = []
    for r in rows:
        my_index, opp_index = r.my_index or 0, r.opp_index or 0
        if r.result == "forfeit" or r.opp_result == "forfeit":
            endings["forfeit"] += 1
        elif my_index == 0 and opp_index == 0:
            continue
        elif my_index >= n and opp_index >= n:
            endings["all_finished"] += 1
        else:
            endings["time_up"] += 1
        if my_index >= n and r.my_finished_at is not None and r.started_at is not None:
            finished, started = r.my_finished_at, r.started_at
            if finished.tzinfo is None:  # SQLite (tests)
                finished, started = finished.replace(tzinfo=UTC), started.replace(tzinfo=UTC)
            finish_seconds.append(max(0.0, (finished - started).total_seconds()))
    return {
        **endings,
        "finished_all_questions": len(finish_seconds),
        "average_finish_seconds": round(sum(finish_seconds) / len(finish_seconds), 1) if finish_seconds else None,
    }


async def get_battle_analytics(db: AsyncSession, *, user_id: int) -> dict:
    rows = await _participations(db, user_id=user_id)

    results = [r.result for r in rows]
    wins = results.count("win")
    losses = results.count("loss")
    draws = results.count("draw")
    forfeits = results.count("forfeit")
    played = len(rows)
    scores = [r.final_score for r in rows if r.final_score is not None]

    # Per subject, because each subject has its own independent rating --
    # one combined line would jump between unrelated scales.
    history: dict[str, list[dict]] = {}
    for r in rows:
        if r.rating_after is None:
            continue
        history.setdefault(r.subject, []).append(
            {
                "match_id": r.match_id,
                "finished_at": r.finished_at,
                "rating": r.rating_after,
                "rating_change": r.rating_change,
                "result": r.result,
            }
        )
    limit = settings.BATTLE_ANALYTICS_RATING_HISTORY_LIMIT
    rating_history = [
        {"subject": subject, "points": points[-limit:]}
        for subject, points in sorted(history.items(), key=lambda kv: -len(kv[1]))
    ]

    stronger = [
        r for r in rows
        if r.opponent_rating_before is not None and r.opponent_rating_before > r.rating_before
    ]

    totals = await _answer_totals(db, user_id=user_id)

    return {
        "matches_played": played,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "forfeits": forfeits,
        "win_rate": round(wins / played * 100, 1) if played else 0.0,
        "average_score": round(sum(scores) / len(scores), 1) if scores else 0.0,
        "answered": totals["answered"],
        "correct": totals["correct"],
        "accuracy": round(totals["correct"] / totals["answered"] * 100, 1) if totals["answered"] else 0.0,
        "recent_form": [r for r in results if r is not None][-settings.BATTLE_ANALYTICS_RECENT_FORM_LIMIT:],
        "rating_history": rating_history,
        "speed_profile": await _speed_profile(db, user_id=user_id),
        "points": totals["points"],
        "topics": await _topics(db, user_id=user_id),
        "question_usage": await _question_usage(db, user_id=user_id, answered=totals["answered"]),
        "match_endings": _match_endings(rows),
        "vs_stronger": {
            "played": len(stronger),
            "wins": sum(1 for r in stronger if r.result == "win"),
        },
    }

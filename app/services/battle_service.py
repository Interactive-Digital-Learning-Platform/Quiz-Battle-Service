"""
app/services/battle_service.py
─────────────────────────────────
DB-backed queries for the competitive rating/leaderboard system. Mostly
read-only (leaderboard, profile) -- CompetitiveSubjectRating rows are only
ever written by get_or_create_subject_rating (lazy creation the first time a
user queues for a subject, see app/services/matchmaking_service.py) and, in
a later phase, the battle-completion flow that calls
elo_engine.compute_elo_update() and updates rows here.
"""
import math

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.competitive_rating import CompetitiveSubjectRating
from app.models.user import User
from app.services import league_engine

# Total leaderboard ordering used everywhere rank is computed -- rating first
# (the primary competitive signal), then wins, then average response time
# (faster is better), then id as a final deterministic tie-break so ordering
# never depends on unspecified row order.
_ORDER_BY = (
    CompetitiveSubjectRating.rating.desc(),
    CompetitiveSubjectRating.wins.desc(),
    CompetitiveSubjectRating.average_response_time.asc(),
    CompetitiveSubjectRating.id.asc(),
)


def _accuracy(row: CompetitiveSubjectRating) -> float:
    if row.total_answers == 0:
        return 0.0
    return (row.correct_answers / row.total_answers) * 100.0


async def _rank_of(db: AsyncSession, *, subject: str, row: CompetitiveSubjectRating) -> int:
    """1-based rank of `row` within its subject's leaderboard, via a COUNT
    query replicating the _ORDER_BY total order -- never materializes rows
    into Python, so this stays cheap no matter how large the leaderboard is.
    """
    sorts_before = or_(
        CompetitiveSubjectRating.rating > row.rating,
        and_(
            CompetitiveSubjectRating.rating == row.rating,
            CompetitiveSubjectRating.wins > row.wins,
        ),
        and_(
            CompetitiveSubjectRating.rating == row.rating,
            CompetitiveSubjectRating.wins == row.wins,
            CompetitiveSubjectRating.average_response_time < row.average_response_time,
        ),
        and_(
            CompetitiveSubjectRating.rating == row.rating,
            CompetitiveSubjectRating.wins == row.wins,
            CompetitiveSubjectRating.average_response_time == row.average_response_time,
            CompetitiveSubjectRating.id < row.id,
        ),
    )
    stmt = (
        select(func.count())
        .select_from(CompetitiveSubjectRating)
        .where(CompetitiveSubjectRating.subject == subject, sorts_before)
    )
    better_count = (await db.execute(stmt)).scalar() or 0
    return better_count + 1


def _entry_dict(row: CompetitiveSubjectRating, username: str | None, rank: int) -> dict:
    return {
        "rank": rank,
        "user_id": row.user_id,
        "username": username,
        "subject": row.subject,
        "rating": row.rating,
        "league": league_engine.get_league(row.rating),
        "wins": row.wins,
        "losses": row.losses,
        "draws": row.draws,
        "matches_played": row.matches_played,
        "average_response_time": row.average_response_time,
    }


async def get_leaderboard(
    db: AsyncSession,
    *,
    subject: str,
    page: int,
    page_size: int,
    requesting_user_id: int | None,
) -> dict:
    total_count = (
        await db.execute(
            select(func.count())
            .select_from(CompetitiveSubjectRating)
            .where(CompetitiveSubjectRating.subject == subject)
        )
    ).scalar() or 0

    offset = (page - 1) * page_size
    rows = (
        await db.execute(
            select(CompetitiveSubjectRating, User.username)
            .join(User, User.id == CompetitiveSubjectRating.user_id)
            .where(CompetitiveSubjectRating.subject == subject)
            .order_by(*_ORDER_BY)
            .offset(offset)
            .limit(page_size)
        )
    ).all()

    entries = [
        _entry_dict(row, username, rank=offset + i + 1)
        for i, (row, username) in enumerate(rows)
    ]

    my_rank: int | None = None
    my_rating: int | None = None
    if requesting_user_id is not None:
        my_row = (
            await db.execute(
                select(CompetitiveSubjectRating).where(
                    CompetitiveSubjectRating.user_id == requesting_user_id,
                    CompetitiveSubjectRating.subject == subject,
                )
            )
        ).scalar_one_or_none()
        if my_row is not None:
            my_rating = my_row.rating
            my_rank = await _rank_of(db, subject=subject, row=my_row)

    return {
        "subject": subject,
        "page": page,
        "page_size": page_size,
        "total_count": total_count,
        "total_pages": math.ceil(total_count / page_size) if page_size else 0,
        "entries": entries,
        "my_rank": my_rank,
        "my_rating": my_rating,
    }


async def get_competitive_profile(db: AsyncSession, *, user_id: int) -> list[dict]:
    rows = (
        await db.execute(
            select(CompetitiveSubjectRating).where(CompetitiveSubjectRating.user_id == user_id)
        )
    ).scalars().all()

    # N+1 rank queries here (N = number of subjects this user has a row for)
    # -- acceptable since that's realistically a handful, not worth a
    # batched window-function query in Phase 1.
    profiles = []
    for row in rows:
        rank = await _rank_of(db, subject=row.subject, row=row)
        profiles.append(
            {
                "subject": row.subject,
                "rating": row.rating,
                "highest_rating": row.highest_rating,
                "league": league_engine.get_league(row.rating),
                "rank": rank,
                "wins": row.wins,
                "losses": row.losses,
                "draws": row.draws,
                "matches_played": row.matches_played,
                "current_win_streak": row.current_win_streak,
                "highest_win_streak": row.highest_win_streak,
                "correct_answers": row.correct_answers,
                "total_answers": row.total_answers,
                "accuracy": _accuracy(row),
                "average_response_time": row.average_response_time,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }
        )
    return profiles


async def get_or_create_subject_rating(
    db: AsyncSession, *, user_id: int, subject: str
) -> CompetitiveSubjectRating:
    """Lazily creates a user's CompetitiveSubjectRating row for a subject the
    first time they need one (currently: joining the matchmaking queue for
    it) -- at BATTLE_INITIAL_RATING, matching a brand-new player.
    """
    row = (
        await db.execute(
            select(CompetitiveSubjectRating).where(
                CompetitiveSubjectRating.user_id == user_id,
                CompetitiveSubjectRating.subject == subject,
            )
        )
    ).scalar_one_or_none()
    if row is not None:
        return row

    row = CompetitiveSubjectRating(
        user_id=user_id,
        subject=subject,
        rating=settings.BATTLE_INITIAL_RATING,
        highest_rating=settings.BATTLE_INITIAL_RATING,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row

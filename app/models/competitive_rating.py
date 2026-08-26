from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.user import User


class CompetitiveSubjectRating(Base):
    """A user's 1v1 battle rating for one subject.

    Deliberately independent of SubjectMastery/LessonMastery (the adaptive
    learning system) -- no FK, no shared table, no coupling. This model
    answers "how good is this user against other players in this subject"
    (Elo-style competitive rating); mastery answers "how well does this user
    understand this subject" (educational evidence). Both can exist for the
    same user+subject with no interaction.
    """

    __tablename__ = "competitive_subject_ratings"
    __table_args__ = (
        UniqueConstraint("user_id", "subject", name="uq_competitive_rating_user_subject"),
        # Leaderboard queries filter by subject and order by rating -- this
        # composite index covers that filter+sort together.
        Index("ix_competitive_subject_ratings_subject_rating", "subject", "rating"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )

    subject: Mapped[str] = mapped_column(String(100), nullable=False)

    # Elo-style rating. Integer, not float -- matches the whole-number league
    # thresholds (see app/services/league_engine.py) and keeps leaderboard
    # tie-break comparisons exact. League is always computed from this value
    # (app/services/league_engine.get_league) and never stored here.
    rating: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    highest_rating: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    wins: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    losses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    draws: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    matches_played: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    current_win_streak: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    highest_win_streak: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    correct_answers: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_answers: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    average_response_time: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user: Mapped["User"] = relationship("User", back_populates="competitive_ratings")

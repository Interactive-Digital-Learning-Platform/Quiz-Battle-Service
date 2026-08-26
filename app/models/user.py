from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.battle_match import BattleParticipant
    from app.models.competitive_rating import CompetitiveSubjectRating


class User(Base):
    """Thin mirror of Personalized-Quiz-Service's User model, mapped onto the
    SAME `users` table in the shared database. This service does not own or
    migrate this table (see alembic/env.py's include_object filter) -- only
    the columns/relationships battle code actually needs are declared here.
    Keep in sync by hand if the upstream schema changes.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    clerk_id: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)

    username: Mapped[str | None] = mapped_column(String(100), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    competitive_ratings: Mapped[list[CompetitiveSubjectRating]] = relationship(
        "CompetitiveSubjectRating", back_populates="user", cascade="all, delete-orphan"
    )
    battle_participations: Mapped[list[BattleParticipant]] = relationship(
        "BattleParticipant", back_populates="user", cascade="all, delete-orphan"
    )

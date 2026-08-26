from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class AIGenerationEvent(Base):
    """Mirrors Personalized-Quiz-Service's own ai_generation_events table
    (app/models/ai_generation_event.py there) -- a SEPARATE table with a
    DELIBERATELY DIFFERENT name (`battle_ai_generation_events`, not
    `ai_generation_events`), since the latter already physically exists in
    the shared Postgres DB, owned and migrated by Personalized-Quiz-Service
    -- reusing that name here would collide with it. Records one row per
    select_battle_questions() call (aggregated across every difficulty tier
    touched by that call), written by app/services/telemetry_service.py.
    No `session_id` column -- there is no quiz_sessions table in this
    service; battle question generation isn't tied to a single-player quiz
    session at all.
    """

    __tablename__ = "battle_ai_generation_events"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Battle question generation serves BOTH players at once, not one user
    # -- always NULL here (kept nullable, matching the original column,
    # rather than dropped, so the schema stays structurally comparable).
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )

    subject: Mapped[str] = mapped_column(String(100), nullable=False)
    requested_question_count: Mapped[int] = mapped_column(Integer, nullable=False)
    generated_question_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    provider: Mapped[str] = mapped_column(String(50), nullable=False, default="groq")
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)

    success: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    used_cache_fallback: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duplicate_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    invalid_question_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)

    # One of: provider_error | insufficient_pool | unknown. Never the raw
    # exception message -- this table is safe to expose to admins without
    # leaking internals (same convention as the original table).
    error_category: Mapped[str | None] = mapped_column(String(50), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

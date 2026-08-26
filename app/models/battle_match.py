from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
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
    from app.models.question import Question
    from app.models.user import User


class BattleMatch(Base):
    """A 1v1 competitive battle match. Created by matchmaking the instant two
    players are paired (status="waiting") -- Postgres is the durable source
    of truth for real matches; Redis only ever holds transient queue,
    ready, and disconnect-presence state (see
    app/services/matchmaking_service.py and battle_gameplay_service.py).

    battle_gameplay_service.py owns the transitions:
    "waiting" -> "countdown" (once both participants call mark_ready --
    question selection happens here) -> "active" (once the countdown
    elapses) -> "completed" (normal finish, or forfeit at any point from
    "waiting"/"countdown"/"active"); or "waiting"/"countdown" -> "cancelled"
    on AI-generation failure. There is no distinct "forfeited" status --
    a forfeit reuses "completed", with BattleParticipant.result="forfeit"
    on the forfeiting side carrying the distinction (so clients only ever
    need to branch on one terminal status).
    """

    __tablename__ = "battle_matches"
    __table_args__ = (Index("ix_battle_matches_status", "status"),)

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    subject: Mapped[str] = mapped_column(String(100), nullable=False)
    # Plain string, no enum -- matches this codebase's existing convention
    # (Question.difficulty, QuizSession.difficulty are also plain strings).
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="waiting")
    difficulty: Mapped[str] = mapped_column(String(20), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    winner_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    # Populated only when status="cancelled" (e.g. AI question generation
    # failure) -- surfaced to both players on their next state poll.
    cancel_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)

    participants: Mapped[list["BattleParticipant"]] = relationship(
        "BattleParticipant", back_populates="match", cascade="all, delete-orphan"
    )
    match_questions: Mapped[list["BattleMatchQuestion"]] = relationship(
        "BattleMatchQuestion",
        back_populates="match",
        cascade="all, delete-orphan",
        order_by="BattleMatchQuestion.question_order",
    )
    answers: Mapped[list["BattleAnswer"]] = relationship(
        "BattleAnswer", back_populates="match", cascade="all, delete-orphan"
    )


class BattleParticipant(Base):
    """One player's participation in a BattleMatch. rating_before is captured
    at match-creation time from the player's CompetitiveSubjectRating.rating.
    final_score/correct_count/rating_after/result are populated by
    battle_gameplay_service's match-completion flow.
    """

    __tablename__ = "battle_participants"
    __table_args__ = (
        UniqueConstraint("match_id", "user_id", name="uq_battle_participant_match_user"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    match_id: Mapped[int] = mapped_column(
        ForeignKey("battle_matches.id", ondelete="CASCADE"), index=True, nullable=False
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )

    rating_before: Mapped[int] = mapped_column(Integer, nullable=False)

    final_score: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    correct_count: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    wrong_count: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    # Longest run of consecutive correct answers across the whole match (an
    # unanswered/skipped question breaks the streak, same as a wrong one).
    best_streak: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    # Sum of BattleAnswer.response_time_ms over ALL of this participant's
    # answers in the match (not just correct ones -- that's
    # battle_scoring_engine.ParticipantOutcome.total_response_time_ms's job,
    # a separate correct-only tiebreak metric used only during winner
    # determination, never persisted).
    total_response_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    rating_after: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    # Stored redundantly with rating_after - rating_before so API responses
    # never have to recompute it -- set atomically alongside rating_after in
    # battle_gameplay_service._apply_result.
    rating_change: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    # "win" | "loss" | "draw" | "forfeit" once populated -- plain string, no enum.
    result: Mapped[str | None] = mapped_column(String(10), nullable=True, default=None)

    # Unused -- a leftover from a self-paced play experiment that was
    # reverted back to lockstep timing (both players on the same question at
    # the same time, match completion driven purely by
    # BattleMatch.started_at + elapsed time, see
    # battle_gameplay_service._ensure_match_progressed). Always NULL. Left
    # in place rather than migrated away to avoid another live-DB schema
    # change; safe to ignore.
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    match: Mapped["BattleMatch"] = relationship("BattleMatch", back_populates="participants")
    user: Mapped["User"] = relationship("User", back_populates="battle_participations")


class BattleMatchQuestion(Base):
    """The frozen, ordered question set for one match -- both players see the
    SAME sequence, selected once at match activation
    (battle_question_service.select_battle_questions). Lockstep play: the
    live "current" question for both players is derived purely from elapsed
    wall-clock time since BattleMatch.started_at (see
    battle_gameplay_service._current_question_order), question_order *is*
    that derivation key, not just display ordering.
    """

    __tablename__ = "battle_match_questions"
    __table_args__ = (
        UniqueConstraint("match_id", "question_order", name="uq_battle_match_question_order"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    match_id: Mapped[int] = mapped_column(
        ForeignKey("battle_matches.id", ondelete="CASCADE"), index=True, nullable=False
    )
    question_id: Mapped[int] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"), index=True, nullable=False
    )
    question_order: Mapped[int] = mapped_column(Integer, nullable=False)  # 0-indexed

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    match: Mapped["BattleMatch"] = relationship("BattleMatch", back_populates="match_questions")
    # One-directional -- Question stays decoupled from the battle system.
    question: Mapped["Question"] = relationship("Question")


class BattleAnswer(Base):
    """One player's graded answer to one question in one match. The unique
    constraint is the DB-level anti-replay guard the spec requires (prefers
    a constraint over app-logic-only checks) -- one answer per player per
    question, full stop.
    """

    __tablename__ = "battle_answers"
    __table_args__ = (
        UniqueConstraint(
            "match_id", "question_id", "user_id", name="uq_battle_answer_match_question_user"
        ),
        Index("ix_battle_answers_match_user", "match_id", "user_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    match_id: Mapped[int] = mapped_column(
        ForeignKey("battle_matches.id", ondelete="CASCADE"), index=True, nullable=False
    )
    question_id: Mapped[int] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )

    selected_option: Mapped[str] = mapped_column(String(512), nullable=False)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    # SERVER-computed (question_presented_at -> answer_received_at), never
    # a client-reported value.
    response_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    base_score: Mapped[int] = mapped_column(Integer, nullable=False)
    speed_bonus: Mapped[int] = mapped_column(Integer, nullable=False)
    streak_bonus: Mapped[int] = mapped_column(Integer, nullable=False)
    total_question_score: Mapped[int] = mapped_column(Integer, nullable=False)

    answered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    match: Mapped["BattleMatch"] = relationship("BattleMatch", back_populates="answers")

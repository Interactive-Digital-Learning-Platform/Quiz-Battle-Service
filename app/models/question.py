from sqlalchemy import JSON, Index, String, event
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.services.question_fingerprint import compute_question_fingerprint


class Question(Base):
    """Thin mirror of Personalized-Quiz-Service's Question model, mapped onto
    the SAME `questions` table in the shared database. This service does not
    own or migrate this table (see alembic/env.py's include_object filter),
    but battle_question_service both reads AND inserts rows here (to top up
    the shared pool with AI-generated questions), so the full column set is
    needed -- unlike the User mirror, this can't be trimmed to a read-only
    subset. Keep in sync by hand if the upstream schema changes.

    Deliberately omits the `attempts` relationship to QuestionAttempt --
    that model belongs to Personalized-Quiz-Service's own single-player
    quiz-attempt tracking, which this service has no need to touch.
    """

    __tablename__ = "questions"
    __table_args__ = (
        # Inert here (alembic/env.py excludes this table from autogenerate),
        # kept for parity with the upstream model / SQLAlchemy reflection.
        Index("ix_questions_subject_difficulty", "subject", "difficulty"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    question: Mapped[str] = mapped_column(String, nullable=False)
    options: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    correct_answer: Mapped[str | None] = mapped_column(String(512), nullable=True)

    subject: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    lesson: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    difficulty: Mapped[str] = mapped_column(String(50), index=True, nullable=False)

    # Hash of (subject, lesson, question text) identifying "the same or an
    # equivalent question" across rows. Filled in automatically by the
    # listener below if not set explicitly.
    question_fingerprint: Mapped[str] = mapped_column(String(64), index=True, nullable=False)


@event.listens_for(Question, "before_insert")
def _populate_question_fingerprint(mapper, connection, target: "Question") -> None:
    if not target.question_fingerprint:
        target.question_fingerprint = compute_question_fingerprint(
            target.subject, target.lesson, target.question
        )

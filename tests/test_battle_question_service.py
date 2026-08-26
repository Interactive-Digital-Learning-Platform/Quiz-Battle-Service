"""
tests/test_battle_question_service.py
──────────────────────────────────────────
Integration tests (in-memory DB, mocked Groq) for
app/services/battle_question_service.py -- pool-first selection with NO
per-user exclusion (fairness comes from both players getting the identical
set), Groq fallback for shortfalls, near-duplicate filtering, clean failure
when neither source can supply enough questions, mixed-difficulty-tier
selection, and AI-generation telemetry recording.
"""
from unittest.mock import AsyncMock, patch

from sqlalchemy import select

from app.models.ai_generation_event import AIGenerationEvent
from app.models.question import Question
from app.services.battle_question_service import (
    BattleQuestionSelectionError,
    select_battle_questions,
)

_DISTINCT_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


def _make_question(i: int, subject: str = "Mathematics", difficulty: str = "medium") -> Question:
    word = _DISTINCT_WORDS[i % len(_DISTINCT_WORDS)]
    return Question(
        question=f"What is the {word} concept number {i}?",
        options=["A", "B", "C", "D"], correct_answer="A",
        subject=subject, lesson="Algebra", difficulty=difficulty,
    )


async def _latest_telemetry_event(db) -> AIGenerationEvent:
    return (
        await db.execute(select(AIGenerationEvent).order_by(AIGenerationEvent.id.desc()))
    ).scalars().first()


async def test_pool_first_no_per_user_exclusion(db_session):
    # This service has no QuizSession/QuestionAttempt concept at all (that's
    # Personalized-Quiz-Service's own single-player attempt tracking, not
    # something battle code touches) -- so there's no per-user "already
    # seen" mechanism to exclude against in the first place. This just
    # confirms the pool query has zero user-scoping: every seeded question
    # for this (subject, difficulty) is fair game.
    questions = [_make_question(i) for i in range(3)]
    db_session.add_all(questions)
    await db_session.commit()

    result = await select_battle_questions(
        db_session, subject="Mathematics", tier_counts={"medium": 3}
    )
    assert len(result) == 3
    assert {q.id for q in result} == {q.id for q in questions}


async def test_multiple_tiers_pull_from_their_own_difficulty(db_session):
    easy = [_make_question(i, difficulty="easy") for i in range(2)]
    medium = [_make_question(i + 10, difficulty="medium") for i in range(3)]
    hard = [_make_question(i + 20, difficulty="hard") for i in range(1)]
    db_session.add_all(easy + medium + hard)
    await db_session.commit()

    result = await select_battle_questions(
        db_session, subject="Mathematics", tier_counts={"easy": 2, "medium": 3, "hard": 1}
    )
    assert len(result) == 6
    by_difficulty = {"easy": 0, "medium": 0, "hard": 0}
    for q in result:
        by_difficulty[q.difficulty] += 1
    assert by_difficulty == {"easy": 2, "medium": 3, "hard": 1}


async def test_groq_fallback_covers_shortfall(db_session):
    # Only 2 in the pool, need 5 -- Groq must supply the remaining 3.
    questions = [_make_question(i) for i in range(2)]
    db_session.add_all(questions)
    await db_session.commit()

    # "Fresh AI {word} question?" template mirrors the existing precedent in
    # tests/test_quiz_generation_pool.py's _fresh_question -- verified to
    # stay well under the 0.55 near-duplicate Jaccard threshold against
    # itself (differs in only 1 of ~2 significant words) and against the
    # pool's own "What is the {word} concept number {i}?" template.
    groq_batch = [
        {
            "question": f"Fresh AI {word} question?", "options": ["A", "B", "C", "D"],
            "correct_answer": "A", "explanation": "because", "lesson": "Algebra",
        }
        for word in _DISTINCT_WORDS[2:5]
    ]

    with patch(
        "app.services.battle_question_service.generate_questions",
        new=AsyncMock(return_value=groq_batch),
    ):
        result = await select_battle_questions(
            db_session, subject="Mathematics", tier_counts={"medium": 5}
        )

    assert len(result) == 5

    event = await _latest_telemetry_event(db_session)
    assert event is not None
    assert event.subject == "Mathematics"
    assert event.requested_question_count == 5
    assert event.generated_question_count == 3
    assert event.success is True
    assert event.used_cache_fallback is False  # a Groq call WAS needed


async def test_pool_only_selection_marks_telemetry_as_cache_fallback(db_session):
    questions = [_make_question(i) for i in range(3)]
    db_session.add_all(questions)
    await db_session.commit()

    await select_battle_questions(db_session, subject="Mathematics", tier_counts={"medium": 3})

    event = await _latest_telemetry_event(db_session)
    assert event is not None
    assert event.success is True
    assert event.used_cache_fallback is True  # no Groq call was needed at all


async def test_near_duplicate_ai_questions_are_dropped(db_session):
    # Pool is empty; Groq returns 3 questions, 2 of which are near-duplicates
    # of each other (identical significant-word sets: {alpha, topic} for
    # both -- verified directly via _jaccard_similarity, since the shared
    # utility's naive whitespace-only tokenization means trailing "?"
    # attaches to and strips out the LAST word before it, which is easy to
    # get wrong by accident -- see the two texts below deliberately putting
    # the differing word in a position not swallowed by that). Only 1 of
    # the 2 near-duplicates should survive, so with count=2 requested and
    # 2 truly-distinct concepts (alpha, bravo) available, selection succeeds
    # with exactly one "alpha" variant plus "bravo".
    groq_batch = [
        {"question": "What is the alpha topic value?", "options": ["A", "B", "C", "D"], "correct_answer": "A", "explanation": "x", "lesson": "Algebra"},
        {"question": "What is the alpha topic amount?", "options": ["A", "B", "C", "D"], "correct_answer": "A", "explanation": "x", "lesson": "Algebra"},
        {"question": "What is the bravo topic value?", "options": ["A", "B", "C", "D"], "correct_answer": "A", "explanation": "x", "lesson": "Algebra"},
    ]

    with patch(
        "app.services.battle_question_service.generate_questions",
        new=AsyncMock(return_value=groq_batch),
    ):
        result = await select_battle_questions(
            db_session, subject="Mathematics", tier_counts={"medium": 2}
        )

    # Only 2 distinct (non-duplicate) concepts exist across the whole batch.
    assert len(result) == 2
    texts = {q.question for q in result}
    assert "What is the bravo topic value?" in texts
    assert len([t for t in texts if t.startswith("What is the alpha")]) == 1


async def test_total_failure_raises_battle_question_selection_error(db_session):
    with patch(
        "app.services.battle_question_service.generate_questions",
        new=AsyncMock(return_value=[]),
    ):
        try:
            await select_battle_questions(
                db_session, subject="Mathematics", tier_counts={"medium": 5}
            )
            assert False, "expected BattleQuestionSelectionError"
        except BattleQuestionSelectionError:
            pass

    # A failed attempt still records telemetry (non-fatal, always runs).
    event = await _latest_telemetry_event(db_session)
    assert event is not None
    assert event.success is False
    assert event.error_category is not None

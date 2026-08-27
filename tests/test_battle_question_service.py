"""
tests/test_battle_question_service.py
──────────────────────────────────────────
Integration tests (in-memory DB, mocked Groq) for
app/services/battle_question_service.py -- selection spans whatever lessons
naturally turn up within the subject (no single-lesson lock), per-player
"never repeat a question already answered in a previous battle" exclusion
(scoped to the subject, across all its lessons), near-duplicate filtering
(now also catching AI rewordings of already-seen questions), clean failure
when neither source can supply enough questions, mixed-difficulty-tier
selection, and AI-generation telemetry recording.
"""
from unittest.mock import AsyncMock, patch

from sqlalchemy import select

from app.models.ai_generation_event import AIGenerationEvent
from app.models.battle_match import BattleAnswer, BattleMatch
from app.models.question import Question
from app.services.battle_question_service import (
    BattleQuestionSelectionError,
    select_battle_questions,
)
from app.services.user_service import get_or_create_user

_DISTINCT_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


def _make_question(i: int, subject: str = "Mathematics", difficulty: str = "medium") -> Question:
    word = _DISTINCT_WORDS[i % len(_DISTINCT_WORDS)]
    return Question(
        question=f"What is the {word} concept number {i}?",
        options=["A", "B", "C", "D"], correct_answer="A",
        subject=subject, lesson="Algebra", difficulty=difficulty,
    )


async def _make_user(db, clerk_id: str) -> int:
    user = await get_or_create_user(db, clerk_id)
    return user.id


async def _record_battle_answer(db, *, user_id: int, question: Question) -> None:
    # A minimal "this user already faced this question in a previous
    # battle" fixture -- the match itself just needs to exist for the FK,
    # its own status/difficulty aren't relevant to selection.
    match = BattleMatch(subject=question.subject, status="completed", difficulty="medium")
    db.add(match)
    await db.flush()
    db.add(BattleAnswer(
        match_id=match.id, question_id=question.id, user_id=user_id,
        selected_option="A", is_correct=True, response_time_ms=1000,
        base_score=100, speed_bonus=0, streak_bonus=0, total_question_score=100,
    ))
    await db.commit()


async def _latest_telemetry_event(db) -> AIGenerationEvent:
    return (
        await db.execute(select(AIGenerationEvent).order_by(AIGenerationEvent.id.desc()))
    ).scalars().first()


async def test_previously_answered_questions_are_excluded(db_session):
    # Either player having already answered a question in an earlier battle
    # for this subject must exclude it going forward -- both players still
    # get an identical, but never-before-seen, set.
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    seen, fresh = _make_question(0), _make_question(1)
    db_session.add_all([seen, fresh])
    await db_session.commit()
    await _record_battle_answer(db_session, user_id=user_a, question=seen)

    result = await select_battle_questions(
        db_session, subject="Mathematics", tier_counts={"medium": 1},
        participant_user_ids=(user_a, user_b),
    )
    assert len(result) == 1
    assert result[0].id == fresh.id


async def test_selection_can_span_multiple_lessons(db_session):
    # No single-lesson lock -- a match's questions can and should be pulled
    # from whatever lessons the subject's pool naturally offers.
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    algebra = [_make_question(i, difficulty="medium") for i in range(2)]
    geometry = [_make_question(i + 10, difficulty="medium") for i in range(2)]
    for q in geometry:
        q.lesson = "Geometry"
    db_session.add_all(algebra + geometry)
    await db_session.commit()

    result = await select_battle_questions(
        db_session, subject="Mathematics", tier_counts={"medium": 4},
        participant_user_ids=(user_a, user_b),
    )
    assert len(result) == 4
    assert {q.lesson for q in result} == {"Algebra", "Geometry"}


async def test_seen_question_text_also_blocks_ai_reword(db_session):
    # A near-duplicate REWORDING of an already-seen question must be
    # rejected too, not just an exact-ID repeat -- exclude_texts is seeded
    # from seen questions' own text for exactly this reason.
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    seen = _make_question(0)  # "What is the alpha concept number 0?"
    db_session.add(seen)
    await db_session.commit()
    await _record_battle_answer(db_session, user_id=user_a, question=seen)

    groq_batch = [
        {
            "question": "What is the alpha concept number zero?", "options": ["A", "B", "C", "D"],
            "correct_answer": "A", "explanation": "x", "lesson": "Algebra",
        },
        {
            "question": "What is the bravo topic value?", "options": ["A", "B", "C", "D"],
            "correct_answer": "A", "explanation": "x", "lesson": "Algebra",
        },
    ]
    with patch(
        "app.services.battle_question_service.generate_questions",
        new=AsyncMock(return_value=groq_batch),
    ):
        result = await select_battle_questions(
            db_session, subject="Mathematics", tier_counts={"medium": 1},
            participant_user_ids=(user_a, user_b),
        )

    assert len(result) == 1
    assert result[0].question == "What is the bravo topic value?"


async def test_multiple_tiers_pull_from_their_own_difficulty(db_session):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    easy = [_make_question(i, difficulty="easy") for i in range(2)]
    medium = [_make_question(i + 10, difficulty="medium") for i in range(3)]
    hard = [_make_question(i + 20, difficulty="hard") for i in range(1)]
    db_session.add_all(easy + medium + hard)
    await db_session.commit()

    result = await select_battle_questions(
        db_session, subject="Mathematics", tier_counts={"easy": 2, "medium": 3, "hard": 1},
        participant_user_ids=(user_a, user_b),
    )
    assert len(result) == 6
    by_difficulty = {"easy": 0, "medium": 0, "hard": 0}
    for q in result:
        by_difficulty[q.difficulty] += 1
    assert by_difficulty == {"easy": 2, "medium": 3, "hard": 1}


async def test_groq_fallback_covers_shortfall(db_session):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
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
            db_session, subject="Mathematics", tier_counts={"medium": 5},
            participant_user_ids=(user_a, user_b),
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
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    questions = [_make_question(i) for i in range(3)]
    db_session.add_all(questions)
    await db_session.commit()

    await select_battle_questions(
        db_session, subject="Mathematics", tier_counts={"medium": 3},
        participant_user_ids=(user_a, user_b),
    )

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
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
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
            db_session, subject="Mathematics", tier_counts={"medium": 2},
            participant_user_ids=(user_a, user_b),
        )

    # Only 2 distinct (non-duplicate) concepts exist across the whole batch.
    assert len(result) == 2
    texts = {q.question for q in result}
    assert "What is the bravo topic value?" in texts
    assert len([t for t in texts if t.startswith("What is the alpha")]) == 1


async def test_total_failure_raises_battle_question_selection_error(db_session):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    with patch(
        "app.services.battle_question_service.generate_questions",
        new=AsyncMock(return_value=[]),
    ):
        try:
            await select_battle_questions(
                db_session, subject="Mathematics", tier_counts={"medium": 5},
                participant_user_ids=(user_a, user_b),
            )
            assert False, "expected BattleQuestionSelectionError"
        except BattleQuestionSelectionError:
            pass

    # A failed attempt still records telemetry (non-fatal, always runs).
    event = await _latest_telemetry_event(db_session)
    assert event is not None
    assert event.success is False
    assert event.error_category is not None

"""
tests/test_battle_ai_questions.py
─────────────────────────────────
AI-first battle question selection (battle_question_service._ai_first_questions):
fresh AI questions come first and are saved to the pool; anything either
player has been shown in a battle (answered OR just reached), anything
already in the pool, and near-copies of either are dropped; whatever the AI
can't provide in time -- slow, failing, busy, or all repeats -- is filled
from the pool, never cancelling the match. generate_questions is always
mocked here; conftest switches AI-first off for every other test.
"""
import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select

from app.core.config import settings
from app.models.battle_match import BattleMatch, BattleMatchQuestion, BattleParticipant
from app.models.question import Question
from app.services import battle_question_service
from app.services.battle_question_service import (
    _seen_battle_questions,
    select_battle_questions,
)
from app.services.user_service import get_or_create_user


def _word(i: int) -> str:
    """A made-up word unique to `i` (letters only, so the near-duplicate
    check treats it as a real word) -- keeps test questions from reading as
    near-copies of each other."""
    letters = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        letters = chr(97 + r) + letters
    return f"zq{letters}"


def _ai_item(i: int, lesson: str = "Geometry") -> dict:
    return {
        "question": f"How does {_word(3 * i)} relate to {_word(3 * i + 1)} and {_word(3 * i + 2)}?",
        "options": ["A", "B", "C", "D"], "correct_answer": "A", "explanation": "", "lesson": lesson,
    }


def _pool_question(i: int, difficulty: str = "medium") -> Question:
    return Question(
        question=f"Which {_word(5000 + 2 * i)} explains {_word(5001 + 2 * i)}?",
        options=["A", "B", "C", "D"], correct_answer="A",
        subject="Mathematics", lesson="Algebra", difficulty=difficulty,
    )


@pytest.fixture
def ai_on(monkeypatch):
    monkeypatch.setattr(settings, "BATTLE_AI_FIRST_ENABLED", True)
    # A fresh, uncontended slot pool for each test.
    monkeypatch.setattr(battle_question_service, "_ai_generation_slots", None)


def _fake_ai(monkeypatch, items_for):
    """generate_questions stand-in: items_for(difficulty, count) -> list[dict]."""
    mock = AsyncMock(side_effect=lambda **kw: items_for(kw["difficulty"], kw["question_count"]))
    monkeypatch.setattr(battle_question_service, "generate_questions", mock)
    return mock


async def _users(db):
    a = await get_or_create_user(db, "ai-user-a")
    b = await get_or_create_user(db, "ai-user-b")
    return a.id, b.id


async def _past_match(db, *, user_id: int, questions: list[Question], reached_index: int) -> None:
    """A finished earlier battle in which `user_id` got as far as question
    `reached_index` (0-based, inclusive) -- without answering anything."""
    match = BattleMatch(subject="Mathematics", status="completed", difficulty="medium")
    db.add(match)
    await db.flush()
    db.add(BattleParticipant(match_id=match.id, user_id=user_id, rating_before=1000, current_question_index=reached_index))
    for order, q in enumerate(questions):
        db.add(BattleMatchQuestion(match_id=match.id, question_id=q.id, question_order=order))
    await db.commit()


async def _count_with_text(db, text: str) -> int:
    return await db.scalar(select(func.count()).select_from(Question).where(Question.question == text))


# ─────────────────────────────────────────────────────────────────────────────


async def test_fresh_ai_questions_fill_the_whole_match_and_join_the_pool(db_session, ai_on, monkeypatch):
    a, b = await _users(db_session)
    counter = iter(range(1000))
    mock = _fake_ai(monkeypatch, lambda d, n: [_ai_item(next(counter)) for _ in range(n)])

    result = await select_battle_questions(
        db_session, subject="Mathematics", tier_counts={"easy": 7, "medium": 3}, participant_user_ids=(a, b),
    )

    assert len(result) == 10 and len({q.id for q in result}) == 10
    assert all(q.id is not None for q in result)  # saved to the pool
    assert sorted(q.difficulty for q in result) == ["easy"] * 7 + ["medium"] * 3
    # Asked for spares (7 + ceil(7*0.4)=3 -> 10, 3 + ceil(3*0.4)=2 -> 5), in batches of at most 4: 10 -> 4+3+3,
    # 5 -> 3+2, all at once.
    assert sorted(c.kwargs["question_count"] for c in mock.await_args_list) == [2, 3, 3, 3, 4]
    # The spares stay in the pool for later matches.
    assert await db_session.scalar(select(func.count()).select_from(Question)) == 15


async def test_ai_repeats_of_seen_questions_are_dropped_and_the_pool_fills_in(db_session, ai_on, monkeypatch):
    a, b = await _users(db_session)
    seen = _pool_question(900)
    seen.question = "How does photosynthesis convert sunlight into chemical energy?"
    fallback = _pool_question(1)
    db_session.add_all([seen, fallback])
    await db_session.commit()
    await _past_match(db_session, user_id=b, questions=[seen], reached_index=0)

    # The AI only offers a reworded copy of what B has already seen.
    _fake_ai(monkeypatch, lambda d, n: [{
        **_ai_item(0), "question": "How does photosynthesis convert the sunlight into chemical energy?",
    }])

    result = await select_battle_questions(
        db_session, subject="Mathematics", tier_counts={"medium": 1}, participant_user_ids=(a, b),
    )

    assert [q.id for q in result] == [fallback.id]
    assert await _count_with_text(db_session, "How does photosynthesis convert the sunlight into chemical energy?") == 0


async def test_ai_copies_of_pool_questions_are_not_saved_twice(db_session, ai_on, monkeypatch):
    a, b = await _users(db_session)
    existing = _pool_question(5)
    db_session.add(existing)
    await db_session.commit()
    _fake_ai(monkeypatch, lambda d, n: [{**_ai_item(0), "question": existing.question, "lesson": existing.lesson}])

    result = await select_battle_questions(
        db_session, subject="Mathematics", tier_counts={"medium": 1}, participant_user_ids=(a, b),
    )

    assert [q.id for q in result] == [existing.id]  # served from the pool, unseen by both
    assert await _count_with_text(db_session, existing.question) == 1


async def test_reached_but_unanswered_questions_count_as_seen(db_session):
    a, b = await _users(db_session)
    questions = [_pool_question(i) for i in range(4)]
    db_session.add_all(questions)
    await db_session.commit()
    # A reached question 2 (on screen when the clock ran out), never answering.
    await _past_match(db_session, user_id=a, questions=questions, reached_index=2)

    seen = await _seen_battle_questions(db_session, subject="Mathematics", participant_user_ids=(a, b))

    assert {qid for qid, _, _ in seen} == {questions[0].id, questions[1].id, questions[2].id}


async def test_the_ai_is_told_what_both_players_have_seen(db_session, ai_on, monkeypatch):
    a, b = await _users(db_session)
    seen_q = _pool_question(7)
    db_session.add(seen_q)
    await db_session.commit()
    await _past_match(db_session, user_id=a, questions=[seen_q], reached_index=0)
    mock = _fake_ai(monkeypatch, lambda d, n: [_ai_item(i) for i in range(n)])

    await select_battle_questions(
        db_session, subject="Mathematics", tier_counts={"medium": 2}, participant_user_ids=(a, b),
    )

    kwargs = mock.await_args.kwargs
    assert seen_q.question in kwargs["existing_questions"]
    assert kwargs["avoid_lessons"] == ["Algebra"]


async def test_slow_ai_falls_back_to_the_pool_within_the_time_limit(db_session, ai_on, monkeypatch):
    monkeypatch.setattr(settings, "BATTLE_AI_GENERATION_TIMEOUT_SECONDS", 0.2)
    a, b = await _users(db_session)
    pool = [_pool_question(i) for i in range(3)]
    db_session.add_all(pool)
    await db_session.commit()

    async def slow(**kw):
        await asyncio.sleep(5)
        return [_ai_item(0)]

    monkeypatch.setattr(battle_question_service, "generate_questions", slow)

    started = asyncio.get_running_loop().time()
    result = await select_battle_questions(
        db_session, subject="Mathematics", tier_counts={"medium": 3}, participant_user_ids=(a, b),
    )

    assert asyncio.get_running_loop().time() - started < 2
    assert {q.id for q in result} == {q.id for q in pool}


async def test_failing_ai_falls_back_to_the_pool(db_session, ai_on, monkeypatch):
    a, b = await _users(db_session)
    pool = [_pool_question(i) for i in range(2)]
    db_session.add_all(pool)
    await db_session.commit()
    monkeypatch.setattr(
        battle_question_service, "generate_questions",
        AsyncMock(side_effect=HTTPException(status_code=502, detail="AI service error")),
    )

    result = await select_battle_questions(
        db_session, subject="Mathematics", tier_counts={"medium": 2}, participant_user_ids=(a, b),
    )

    assert {q.id for q in result} == {q.id for q in pool}


async def test_partial_ai_batch_is_topped_up_from_the_pool(db_session, ai_on, monkeypatch):
    a, b = await _users(db_session)
    pool = [_pool_question(i) for i in range(5)]
    db_session.add_all(pool)
    await db_session.commit()
    _fake_ai(monkeypatch, lambda d, n: [_ai_item(0), _ai_item(1)])  # only 2 of the 4 needed

    result = await select_battle_questions(
        db_session, subject="Mathematics", tier_counts={"medium": 4}, participant_user_ids=(a, b),
    )

    assert len(result) == 4
    from_ai = [q for q in result if q.lesson == "Geometry"]
    assert len(from_ai) == 2


async def test_busy_ai_slots_go_straight_to_the_pool(db_session, ai_on, monkeypatch):
    a, b = await _users(db_session)
    pool = [_pool_question(i) for i in range(2)]
    db_session.add_all(pool)
    await db_session.commit()
    monkeypatch.setattr(battle_question_service, "_ai_generation_slots", asyncio.Semaphore(0))
    mock = _fake_ai(monkeypatch, lambda d, n: [_ai_item(i) for i in range(n)])

    result = await select_battle_questions(
        db_session, subject="Mathematics", tier_counts={"medium": 2}, participant_user_ids=(a, b),
    )

    mock.assert_not_awaited()
    assert {q.id for q in result} == {q.id for q in pool}



def test_batches_are_near_equal_and_never_over_the_size():
    assert battle_question_service._batch_sizes(10, 4) == [4, 3, 3]
    assert battle_question_service._batch_sizes(5, 4) == [3, 2]
    assert battle_question_service._batch_sizes(4, 4) == [4]
    assert battle_question_service._batch_sizes(1, 4) == [1]


async def test_one_slow_batch_costs_only_its_own_questions(db_session, ai_on, monkeypatch):
    monkeypatch.setattr(settings, "BATTLE_AI_GENERATION_TIMEOUT_SECONDS", 0.3)
    a, b = await _users(db_session)
    pool = [_pool_question(i) for i in range(4)]
    db_session.add_all(pool)
    await db_session.commit()
    calls = iter(range(100))

    async def one_slow(**kw):
        n = next(calls)
        if n == 0:
            await asyncio.sleep(5)  # this batch misses the match start
        return [_ai_item(10 * n + i) for i in range(kw["question_count"])]

    monkeypatch.setattr(battle_question_service, "generate_questions", one_slow)

    result = await select_battle_questions(
        db_session, subject="Mathematics", tier_counts={"medium": 7}, participant_user_ids=(a, b),
    )

    # 7 + 3 spares = 10 asked as 4+3+3; the slow batch of 4 is lost to this match, the other 6 cover it.
    assert len(result) == 7
    assert sum(q.lesson == "Geometry" for q in result) == 6


async def test_a_late_batch_lands_in_the_pool_without_duplicates(db_session, monkeypatch):
    existing = _pool_question(1)
    db_session.add(existing)
    await db_session.commit()

    class _Session:
        async def __aenter__(self):
            return db_session

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(battle_question_service, "AsyncSessionLocal", _Session)

    async def finished_late():
        return [_ai_item(0), _ai_item(1), {**_ai_item(2), "question": existing.question, "lesson": existing.lesson}]

    await battle_question_service._save_late_ai_batch(
        subject="Mathematics", difficulty="medium", task=asyncio.ensure_future(finished_late()),
    )

    texts = [q for q in (await db_session.execute(select(Question.question))).scalars()]
    assert _ai_item(0)["question"] in texts and _ai_item(1)["question"] in texts
    assert texts.count(existing.question) == 1  # the copy of a pool question was skipped


def test_maths_questions_differing_only_in_numbers_are_not_copies():
    from app.services.text_similarity import is_near_duplicate

    assert not is_near_duplicate("Find the value of x if 3x + 5 = 20.", ["Find the value of x if 2x - 7 = 15."])
    assert not is_near_duplicate("Simplify x^2 + 4x when x = 3", ["Simplify 3y^2 - 2y when y = 5"])
    # Same expression, only the substituted value changed: still the same exercise -- counted as a copy.
    assert is_near_duplicate("Simplify x^2 + 4x when x = 3", ["Simplify x^2 + 4x when x = 5"])
    # A real copy is still caught -- with or without the final punctuation, and a reworded one too.
    assert is_near_duplicate("Find the value of x if 3x + 5 = 20", ["Find the value of x if 3x + 5 = 20."])
    assert is_near_duplicate(
        "What is the main product of photosynthesis in plants?", ["What is the main product of photosynthesis?"],
    )


def test_each_parallel_batch_gets_its_own_style_and_lessons():
    lessons = ["Algebra", "Geometry", "Fractions", "Statistics", "Probability", "Sets"]
    focuses = [battle_question_service._batch_focus(i, 3, lessons) for i in range(3)]

    assert len(set(focuses)) == 3
    shares = [set(f.split("lessons: ")[1].rstrip(".").split(", ")) for f in focuses]
    assert not (shares[0] & shares[1]) and not (shares[1] & shares[2]) and not (shares[0] & shares[2])
    assert battle_question_service._batch_focus(0, 3, []).startswith("Question style:")  # no lessons known: style only


async def test_batches_are_sent_their_focus(db_session, ai_on, monkeypatch):
    a, b = await _users(db_session)
    db_session.add_all([_pool_question(i) for i in range(3)])  # gives the pool a lesson ("Algebra") to share out
    await db_session.commit()
    mock = _fake_ai(monkeypatch, lambda d, n: [_ai_item(100 + i) for i in range(n)])

    await select_battle_questions(db_session, subject="Mathematics", tier_counts={"medium": 7}, participant_user_ids=(a, b))

    focuses = [c.kwargs["focus"] for c in mock.await_args_list]
    assert len(focuses) == 3 and len(set(focuses)) == 3
    assert any("Algebra" in f for f in focuses)


async def test_the_focus_is_written_into_the_ai_prompt(monkeypatch):
    from types import SimpleNamespace

    from app.services import groq_service

    sent = {}

    async def fake_completion(*, messages, **_kwargs):
        sent["prompt"] = messages[-1]["content"]
        payload = '{"questions": [{"question": "Q?", "options": ["A", "B", "C", "D"], "correct_answer": "A", ' \
                  '"explanation": "", "lesson": "Algebra"}]}'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=payload))])

    monkeypatch.setattr(groq_service, "_create_chat_completion_with_retry", fake_completion)

    await groq_service.generate_questions(
        grade=10, subject="Mathematics", difficulty="easy", question_count=1,
        focus="Question style: recall of key facts. Draw the questions from these lessons: Algebra.",
    )

    assert "FOCUS FOR THIS SET" in sent["prompt"] and "lessons: Algebra" in sent["prompt"]

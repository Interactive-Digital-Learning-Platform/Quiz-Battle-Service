"""
tests/test_battle_question_service.py
──────────────────────────────────────────
Integration tests (in-memory DB) for app/services/battle_question_service.py
-- selection spans whatever lessons naturally turn up within the subject (no
single-lesson lock), per-player "never repeat a question already answered in
a previous battle" exclusion (scoped to the subject, across all its
lessons), near-duplicate filtering, clean failure when the DB pool can't
supply enough questions, mixed-difficulty-tier selection, repeats-as-last-
resort, and the debounced background pool-replenish trigger.

Match activation itself never calls Groq (see the module's own docstring) --
`generate_questions` only shows up here in tests that exercise
_replenish_pool_task directly, the one place it's still actually called.
"""
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.models.battle_match import BattleAnswer, BattleMatch
from app.models.question import Question
from app.services import battle_question_service
from app.services.battle_question_service import (
    BattleQuestionSelectionError,
    _fetch_battle_pool_questions,
    _replenish_pool_task,
    select_battle_questions,
)
from app.services.user_service import get_or_create_user

# Captured at import time, before the autouse `_no_real_pool_replenish`
# fixture (conftest.py) monkeypatches battle_question_service.
# _kickoff_pool_replenish to a no-op for every test -- the debounce test
# below needs the REAL implementation, not that no-op.
_real_kickoff_pool_replenish = battle_question_service._kickoff_pool_replenish

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


async def _bucket_size(db, *, subject: str, difficulty: str) -> int:
    stmt = select(func.count()).select_from(Question).where(
        Question.subject == subject, Question.difficulty == difficulty,
    )
    return (await db.execute(stmt)).scalar() or 0


# ── Never-seen-pool selection ───────────────────────────────────────────────

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


async def test_seen_question_text_also_blocks_a_pool_reword(db_session):
    # A near-duplicate REWORDING of an already-seen question must be
    # rejected too, not just an exact-ID repeat -- history_texts is seeded
    # from seen questions' own text for exactly this reason.
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    seen = _make_question(0)  # "What is the alpha concept number 0?"
    db_session.add(seen)
    await db_session.commit()
    await _record_battle_answer(db_session, user_id=user_a, question=seen)

    reword = Question(
        question="What is the alpha concept number zero?", options=["A", "B", "C", "D"],
        correct_answer="A", subject="Mathematics", lesson="Algebra", difficulty="medium",
    )
    distinct = Question(
        question="What is the bravo topic value?", options=["A", "B", "C", "D"],
        correct_answer="A", subject="Mathematics", lesson="Algebra", difficulty="medium",
    )
    db_session.add_all([reword, distinct])
    await db_session.commit()

    result = await select_battle_questions(
        db_session, subject="Mathematics", tier_counts={"medium": 1},
        participant_user_ids=(user_a, user_b),
    )
    assert len(result) == 1
    assert result[0].question == "What is the bravo topic value?"


async def test_multiple_tiers_pull_from_their_own_difficulty(db_session):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    # Offsets chosen so i % len(_DISTINCT_WORDS) never collides across
    # tiers (10,11,12 and 20 would both land on index 4 -- "echo" -- since
    # 20 % 8 == 12 % 8, which the near-duplicate filter (shared across
    # tiers via session_texts) would then reject as a false-positive dup).
    easy = [_make_question(i, difficulty="easy") for i in range(2)]
    medium = [_make_question(i + 10, difficulty="medium") for i in range(3)]
    hard = [_make_question(i + 30, difficulty="hard") for i in range(1)]
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


async def test_near_duplicate_pool_questions_are_deduped(db_session):
    # Two near-duplicate pool candidates (the second only adds one word:
    # {alpha, topic, value} vs {alpha, topic, value, here} -- 75% the same)
    # plus one truly distinct one -- only one of the near-duplicates should
    # survive, so with count=2 requested and 2 truly-distinct concepts
    # available (alpha, bravo), selection succeeds with exactly one "alpha"
    # variant plus "bravo".
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    candidates = [
        Question(question="What is the alpha topic value?", options=["A", "B", "C", "D"], correct_answer="A", subject="Mathematics", lesson="Algebra", difficulty="medium"),
        Question(question="What is the alpha topic value here?", options=["A", "B", "C", "D"], correct_answer="A", subject="Mathematics", lesson="Algebra", difficulty="medium"),
        Question(question="Name the bravo concept number.", options=["A", "B", "C", "D"], correct_answer="A", subject="Mathematics", lesson="Algebra", difficulty="medium"),
    ]
    db_session.add_all(candidates)
    await db_session.commit()

    result = await select_battle_questions(
        db_session, subject="Mathematics", tier_counts={"medium": 2},
        participant_user_ids=(user_a, user_b),
    )

    # Only 2 distinct (non-duplicate) concepts exist across the whole pool.
    assert len(result) == 2
    texts = {q.question for q in result}
    assert "Name the bravo concept number." in texts
    assert len([t for t in texts if t.startswith("What is the alpha")]) == 1


async def test_repeats_allowed_as_last_resort_when_pool_is_short(db_session):
    # Every pool question for this difficulty has already been answered by
    # one participant -- the never-seen pool comes up completely empty.
    # Selection must still fill the tier by allowing those same
    # already-seen questions back in, rather than raising
    # BattleQuestionSelectionError.
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    seen = [_make_question(i) for i in range(3)]
    db_session.add_all(seen)
    await db_session.commit()
    for q in seen:
        await _record_battle_answer(db_session, user_id=user_a, question=q)

    result = await select_battle_questions(
        db_session, subject="Mathematics", tier_counts={"medium": 3},
        participant_user_ids=(user_a, user_b),
    )

    assert len(result) == 3
    assert {q.id for q in result} == {q.id for q in seen}


async def test_total_failure_raises_battle_question_selection_error(db_session):
    # Empty pool, no repeats available either (nobody's answered anything
    # yet) -- nothing left to serve, must raise rather than starting a
    # partially-filled match.
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    with pytest.raises(BattleQuestionSelectionError):
        await select_battle_questions(
            db_session, subject="Mathematics", tier_counts={"medium": 5},
            participant_user_ids=(user_a, user_b),
        )


async def test_short_tier_borrows_from_the_neighbouring_difficulty(db_session):
    # No medium questions at all (a subject nobody has battled in yet) --
    # the medium share comes from easy instead of cancelling the match.
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    db_session.add_all([_make_question(i, difficulty="easy") for i in range(4)])
    await db_session.commit()

    result = await select_battle_questions(
        db_session, subject="Mathematics", tier_counts={"easy": 2, "medium": 2},
        participant_user_ids=(user_a, user_b),
    )

    assert len(result) == 4
    assert {q.difficulty for q in result} == {"easy"}
    assert len({q.id for q in result}) == 4  # nothing picked twice


async def test_borrowing_prefers_the_nearest_difficulty(db_session):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    db_session.add_all([_make_question(2, difficulty="medium"), _make_question(4, difficulty="easy")])
    await db_session.commit()

    result = await select_battle_questions(
        db_session, subject="Mathematics", tier_counts={"hard": 1},
        participant_user_ids=(user_a, user_b),
    )

    assert [q.difficulty for q in result] == ["medium"]


async def test_still_raises_when_no_difficulty_can_cover_the_shortfall(db_session):
    user_a = await _make_user(db_session, "user-a")
    user_b = await _make_user(db_session, "user-b")
    db_session.add_all([_make_question(i, difficulty="easy") for i in range(3)])
    await db_session.commit()

    with pytest.raises(BattleQuestionSelectionError):
        await select_battle_questions(
            db_session, subject="Mathematics", tier_counts={"easy": 2, "medium": 2},
            participant_user_ids=(user_a, user_b),
        )


# ── Startup pool warm-up ────────────────────────────────────────────────────

async def test_warm_up_tops_up_only_thin_buckets_of_allowed_subjects(db_session, monkeypatch):
    monkeypatch.setattr(settings, "BATTLE_ALLOWED_SUBJECTS", "Mathematics,Geography")
    monkeypatch.setattr(
        battle_question_service, "AsyncSessionLocal",
        async_sessionmaker(bind=db_session.bind, class_=AsyncSession, expire_on_commit=False),
    )
    monkeypatch.setattr(battle_question_service, "_replenishing_buckets", set())
    replenish = AsyncMock()
    monkeypatch.setattr(battle_question_service, "_replenish_pool_task", replenish)
    # Mathematics/easy is already stocked; History isn't an allowed subject.
    db_session.add_all([_make_question(i, difficulty="easy") for i in range(settings.BATTLE_POOL_MIN_SIZE)])
    db_session.add(_make_question(0, subject="History", difficulty="easy"))
    await db_session.commit()

    await battle_question_service.warm_up_pools()

    topped_up = {call.args for call in replenish.await_args_list}
    assert topped_up == {
        ("Geography", "easy"), ("Geography", "hard"), ("Geography", "medium"),
        ("Mathematics", "hard"), ("Mathematics", "medium"),
    }


async def test_warm_up_skips_a_bucket_already_being_topped_up(db_session, monkeypatch):
    monkeypatch.setattr(settings, "BATTLE_ALLOWED_SUBJECTS", "Mathematics")
    monkeypatch.setattr(
        battle_question_service, "AsyncSessionLocal",
        async_sessionmaker(bind=db_session.bind, class_=AsyncSession, expire_on_commit=False),
    )
    monkeypatch.setattr(battle_question_service, "_replenishing_buckets", {("Mathematics", "medium")})
    replenish = AsyncMock()
    monkeypatch.setattr(battle_question_service, "_replenish_pool_task", replenish)

    await battle_question_service.warm_up_pools()

    assert ("Mathematics", "medium") not in {call.args for call in replenish.await_args_list}


# ── Debounced background pool replenish ─────────────────────────────────────

async def test_low_bucket_triggers_exactly_one_replenish_call(db_session):
    assert settings.BATTLE_POOL_MIN_SIZE > 1
    db_session.add(_make_question(0, difficulty="easy"))
    await db_session.commit()

    with patch.object(battle_question_service, "_kickoff_pool_replenish") as mock_kickoff:
        await _fetch_battle_pool_questions(
            db_session, subject="Mathematics", difficulty="easy", limit=1,
            exclude_ids=set(), history_texts=[], session_texts=[],
        )

    mock_kickoff.assert_called_once_with("Mathematics", "easy")


async def test_healthy_bucket_does_not_trigger_replenish(db_session):
    for i in range(settings.BATTLE_POOL_MIN_SIZE):
        db_session.add(_make_question(i, difficulty="easy"))
    await db_session.commit()

    with patch.object(battle_question_service, "_kickoff_pool_replenish") as mock_kickoff:
        await _fetch_battle_pool_questions(
            db_session, subject="Mathematics", difficulty="easy", limit=1,
            exclude_ids=set(), history_texts=[], session_texts=[],
        )

    mock_kickoff.assert_not_called()


def test_kickoff_pool_replenish_is_debounced():
    """Concurrent/repeated triggers for the same (subject, difficulty)
    bucket must only ever spawn one in-flight background task."""
    created_coros = []

    def _fake_create_task(coro):
        coro.close()  # never actually run it in this synchronous test
        created_coros.append(coro)

    battle_question_service._replenishing_buckets.clear()
    with patch("asyncio.create_task", _fake_create_task):
        _real_kickoff_pool_replenish("Mathematics", "easy")
        _real_kickoff_pool_replenish("Mathematics", "easy")
        _real_kickoff_pool_replenish("Mathematics", "medium")

    assert len(created_coros) == 2  # (Math, easy) once, (Math, medium) once
    battle_question_service._replenishing_buckets.clear()


async def test_replenish_pool_task_persists_generated_questions(db_session, monkeypatch):
    # _replenish_pool_task opens its OWN AsyncSessionLocal() rather than
    # using the request-scoped session -- point that at the same in-memory
    # engine db_session is already using so this test can observe the
    # writes it makes.
    test_session_factory = async_sessionmaker(bind=db_session.bind, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(battle_question_service, "AsyncSessionLocal", test_session_factory)

    groq_batch = [
        {
            "question": f"Replenished {word} question?", "options": ["A", "B", "C", "D"],
            "correct_answer": "A", "explanation": "x", "lesson": "Algebra",
        }
        for word in _DISTINCT_WORDS[:5]
    ]
    with patch(
        "app.services.battle_question_service.generate_questions",
        new=AsyncMock(return_value=groq_batch),
    ) as mock_generate:
        await _replenish_pool_task("Mathematics", "easy")

    mock_generate.assert_called_once()
    call_kwargs = mock_generate.call_args.kwargs
    assert call_kwargs["subject"] == "Mathematics"
    assert call_kwargs["difficulty"] == "easy"
    assert call_kwargs["question_count"] == settings.BATTLE_POOL_TOPUP_MAX_BATCH

    assert await _bucket_size(db_session, subject="Mathematics", difficulty="easy") == 5


async def test_replenish_pool_task_noop_when_bucket_already_healthy(db_session, monkeypatch):
    test_session_factory = async_sessionmaker(bind=db_session.bind, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(battle_question_service, "AsyncSessionLocal", test_session_factory)

    for i in range(settings.BATTLE_POOL_TARGET_SIZE):
        db_session.add(_make_question(i, difficulty="easy"))
    await db_session.commit()

    with patch(
        "app.services.battle_question_service.generate_questions", new=AsyncMock(),
    ) as mock_generate:
        await _replenish_pool_task("Mathematics", "easy")

    mock_generate.assert_not_called()


async def test_replenish_pool_task_swallows_groq_failure(db_session, monkeypatch):
    test_session_factory = async_sessionmaker(bind=db_session.bind, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(battle_question_service, "AsyncSessionLocal", test_session_factory)

    with patch(
        "app.services.battle_question_service.generate_questions",
        new=AsyncMock(side_effect=HTTPException(status_code=503, detail="Groq down")),
    ):
        await _replenish_pool_task("Mathematics", "easy")  # must not raise

    assert await _bucket_size(db_session, subject="Mathematics", difficulty="easy") == 0

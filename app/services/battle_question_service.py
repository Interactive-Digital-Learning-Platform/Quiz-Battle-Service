"""
app/services/battle_question_service.py
─────────────────────────────────────────
Shared question selection for a 1v1 battle -- BOTH players get the IDENTICAL
question set (fairness comes from sameness). Questions are drawn from
whatever lessons naturally turn up within the subject (no single-lesson
lock -- a match's 10 questions can and should span multiple lessons), but
exclude any question either player has already faced in a PREVIOUS battle
for this subject (via BattleAnswer -- see select_battle_questions), so this
is no longer flat "any question in the subject" selection with zero novelty
guarantee. Uses groq_service.generate_questions() (bare call, no per-user or
lesson-pinning args -- it picks its own lesson per question, same as
before), text_similarity.is_near_duplicate (pure text similarity), and
call_budget.CallBudget (pure counter dataclass) -- all copied from
Personalized-Quiz-Service's quiz_service.py, see those modules for
implementations.

Deliberately does NOT reach into Personalized-Quiz-Service's single-player
attempt history (QuestionAttempt/QuizSession) -- that's a separate service
not modeled here at all (see app/models/question.py's own note on this).
"Already seen" here means "already answered in a previous BATTLE", not
"ever seen anywhere by this user".

Selection is split across difficulty tiers (see app/services/
battle_difficulty.py) rather than one flat difficulty for the whole match --
`tier_counts` (already-allocated exact integer counts per tier) drives one
pool-first/Groq-shortfall/blind-topup pass PER TIER, sharing one
exclude_ids/exclude_texts accumulator across all of them so nothing repeats
or near-duplicates across tiers, or across the two players' battle history.
"""
import logging
import random
import time

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.battle_match import BattleAnswer
from app.models.question import Question
from app.services import telemetry_service
from app.services.call_budget import CallBudget
from app.services.groq_service import generate_questions
from app.services.text_similarity import is_near_duplicate

logger = logging.getLogger(__name__)


class BattleQuestionSelectionError(Exception):
    """Raised when neither the DB pool nor Groq can supply enough shared
    battle questions -- the caller (battle_gameplay_service) must cancel
    the match cleanly, never partially start it with too few questions.
    """


async def _fetch_battle_pool_questions(
    db: AsyncSession, *, subject: str, difficulty: str, limit: int,
    exclude_ids: set[int], exclude_texts: list[str],
) -> list[Question]:
    """Mirrors quiz_service._fetch_pool_questions's shape (random sample, 3x
    over-fetch, near-duplicate accept-loop), drawing from ANY lesson within
    the subject -- exclude_ids already covers both cross-tier repeats within
    this match AND either player's battle history for this subject (seeded
    by the caller, select_battle_questions), so this stays a single, uniform
    exclusion set regardless of where an ID came from. Does NOT trigger
    quiz_service's pool-replenish background task -- that's quiz_service's
    own concern for the single-player pool; a second independent trigger
    path here would risk redundant top-ups.
    """
    if limit <= 0:
        return []
    filters = [Question.subject == subject, Question.difficulty == difficulty]
    if exclude_ids:
        filters.append(Question.id.notin_(exclude_ids))
    stmt = select(Question).where(*filters).order_by(func.random()).limit(limit * 3)
    candidates = list((await db.execute(stmt)).scalars().all())

    accepted: list[Question] = []
    for candidate in candidates:
        if len(accepted) >= limit:
            break
        if is_near_duplicate(candidate.question, exclude_texts):
            continue
        accepted.append(candidate)
        exclude_texts.append(candidate.question)
    return accepted


async def _select_for_one_tier(
    db: AsyncSession, *, subject: str, difficulty: str, count: int,
    exclude_ids: set[int], exclude_texts: list[str],
) -> tuple[list[Question], dict]:
    """Pool-first, Groq-for-the-shortfall (up to BATTLE_QUESTION_GROQ_MAX_
    CALLS calls, same dedup-retry shape as quiz_service._generate_or_cache_
    tier), blind-pool-topup-as-last-resort -- for ONE difficulty tier, drawn
    from any lesson within the subject. exclude_ids/exclude_texts are shared
    mutable accumulators threaded across every tier of one match, so a
    question never repeats (or near-duplicates) across tiers either, not
    just within one -- the caller (select_battle_questions) also seeds these
    with both players' battle history for this subject before the tier loop
    starts, so that history is honored here for free with no extra logic.
    Returns the tier's questions (possibly short -- caller decides whether
    that's fatal) plus per-tier stats for the caller's aggregated telemetry
    event.
    """
    stats = {"generated_count": 0, "duplicate_count": 0, "groq_calls": 0,
              "invalid_question_count": 0, "error_category": None}

    questions = await _fetch_battle_pool_questions(
        db, subject=subject, difficulty=difficulty, limit=count,
        exclude_ids=exclude_ids, exclude_texts=exclude_texts,
    )
    exclude_ids.update(q.id for q in questions)

    remaining = count - len(questions)
    if remaining > 0:
        call_budget = CallBudget(remaining=settings.BATTLE_QUESTION_GROQ_MAX_CALLS)
        tier_telemetry: dict = {}
        for _attempt in range(3):
            still_needed = count - len(questions)
            if still_needed <= 0 or not call_budget.consume():
                break
            stats["groq_calls"] += 1
            try:
                batch = await generate_questions(
                    grade=settings.BATTLE_QUESTION_GRADE_DEFAULT,
                    subject=subject, difficulty=difficulty,
                    question_count=still_needed, existing_questions=exclude_texts,
                    telemetry=tier_telemetry,
                )
            except HTTPException as exc:
                logger.warning(
                    "Battle question generation failed subject=%s difficulty=%s: %s",
                    subject, difficulty, exc.detail,
                )
                stats["error_category"] = "provider_error"
                break
            for q_data in batch:
                if is_near_duplicate(q_data["question"], exclude_texts):
                    stats["duplicate_count"] += 1
                    continue
                question = Question(
                    question=q_data["question"], options=q_data["options"],
                    correct_answer=q_data["correct_answer"], subject=subject,
                    lesson=q_data["lesson"], difficulty=difficulty,
                )
                db.add(question)
                questions.append(question)
                exclude_texts.append(q_data["question"])
                stats["generated_count"] += 1
        stats["invalid_question_count"] = tier_telemetry.get("invalid_question_count", 0)
        if questions:
            await db.flush()  # ids available for exclude_ids below
            exclude_ids.update(q.id for q in questions if q.id is not None)

    remaining = count - len(questions)
    if remaining > 0:
        fallback = await _fetch_battle_pool_questions(
            db, subject=subject, difficulty=difficulty, limit=remaining,
            exclude_ids=exclude_ids, exclude_texts=exclude_texts,
        )
        questions.extend(fallback)

    return questions, stats


async def _seen_question_ids_and_texts(
    db: AsyncSession, *, subject: str, participant_user_ids: tuple[int, int],
) -> tuple[set[int], list[str]]:
    """Every question either player has already answered in a PREVIOUS
    battle for this subject, across ALL lessons -- the "never repeat" source
    of truth. BattleAnswer already carries user_id directly (no join needed
    to reach it, unlike single-player's QuestionAttempt->QuizSession chain).
    Returns both IDs (for the notin_ pool filter) and texts (for the
    near-duplicate check, so an AI-generated rewording of a seen question
    gets caught too, not just an exact-ID repeat).
    """
    rows = (
        await db.execute(
            select(Question.id, Question.question)
            .join(BattleAnswer, BattleAnswer.question_id == Question.id)
            .where(
                BattleAnswer.user_id.in_(participant_user_ids),
                Question.subject == subject,
            )
            .distinct()
        )
    ).all()
    return {row[0] for row in rows}, [row[1] for row in rows]


async def select_battle_questions(
    db: AsyncSession, *, subject: str, tier_counts: dict[str, int],
    participant_user_ids: tuple[int, int],
) -> list[Question]:
    """Selects BATTLE_QUESTION_COUNT questions split across difficulty
    tiers per `tier_counts` (see app/services/battle_difficulty.py's
    get_difficulty_distribution/allocate_question_counts), drawn from
    whatever lessons within the subject naturally turn up (no single-lesson
    lock -- questions can and should span multiple lessons) and excluding
    any question either of `participant_user_ids` has already faced in a
    previous battle for this subject (see _seen_question_ids_and_texts) --
    both players still get the identical set (fairness from sameness), it's
    just a set neither of them has seen before. Raises
    BattleQuestionSelectionError if ANY tier is still short after all three
    phases (pool / Groq / blind topup) -- caller cancels the whole match,
    never starts it partially filled. AI-created Question rows get
    question_fingerprint auto-populated by the existing before_insert
    listener (app/models/question.py) for free; no extra code needed here.

    Records exactly one AIGenerationEvent row for the whole call (aggregated
    across every tier), mirroring quiz_service.generate_quiz's
    one-telemetry-event-per-request granularity -- see telemetry_service.py.
    A telemetry write failure is swallowed there and can never affect
    question selection itself.

    Order of the returned list is randomized so a question's difficulty
    tier and source (pool vs. freshly-generated) never correlate with its
    position in the match.
    """
    exclude_ids, exclude_texts = await _seen_question_ids_and_texts(
        db, subject=subject, participant_user_ids=participant_user_ids,
    )

    questions: list[Question] = []
    total_requested = sum(tier_counts.values())

    agg_generated = 0
    agg_duplicates = 0
    agg_calls = 0
    agg_invalid = 0
    agg_error_category: str | None = None
    short_tiers: list[str] = []

    start = time.monotonic()
    for difficulty, count in tier_counts.items():
        if count <= 0:
            continue
        tier_questions, stats = await _select_for_one_tier(
            db, subject=subject, difficulty=difficulty, count=count,
            exclude_ids=exclude_ids, exclude_texts=exclude_texts,
        )
        questions.extend(tier_questions)
        agg_generated += stats["generated_count"]
        agg_duplicates += stats["duplicate_count"]
        agg_calls += stats["groq_calls"]
        agg_invalid += stats["invalid_question_count"]
        if stats["error_category"] is not None:
            agg_error_category = stats["error_category"]
        if len(tier_questions) < count:
            short_tiers.append(difficulty)
    latency_ms = (time.monotonic() - start) * 1000.0

    success = not short_tiers
    if not success and agg_error_category is None:
        agg_error_category = "insufficient_pool"

    try:
        await telemetry_service.record_generation_event(
            db, user_id=None, subject=subject,
            requested_question_count=total_requested, generated_question_count=agg_generated,
            provider="groq", model_name=settings.GROQ_MODEL, success=success,
            used_cache_fallback=agg_calls == 0, retry_count=max(agg_calls - 1, 0),
            duplicate_count=agg_duplicates, invalid_question_count=agg_invalid,
            latency_ms=latency_ms, error_category=agg_error_category,
        )
    except Exception as telemetry_exc:  # noqa: BLE001
        logger.error("AI generation telemetry raised unexpectedly (non-critical): %s", telemetry_exc)

    if short_tiers:
        raise BattleQuestionSelectionError(
            f"Only {len(questions)}/{total_requested} questions available "
            f"(subject={subject}, short tiers={short_tiers})"
        )

    random.shuffle(questions)
    return questions[:total_requested]

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
guarantee.

AI first, pool second (BATTLE_AI_FIRST_ENABLED): a match start first asks
the AI for fresh questions (_ai_first_questions) -- told what the two
players have already seen, checked afterwards against everything they've
been shown in a battle and against the pool, and saved into the pool -- and
waits at most BATTLE_AI_GENERATION_TIMEOUT_SECONDS for them. Whatever the
AI couldn't provide in time (slow, rate-limited, down, or its questions were
repeats) is filled from the DB pool below, never-seen first, so a failing
AI can delay a match start by that timeout but never cancel it. Keeping
each (subject, difficulty) pool stocked as that fallback is a background
concern: _fetch_battle_pool_questions triggers a
debounced, fire-and-forget top-up (_kickoff_pool_replenish/
_replenish_pool_task) whenever a bucket reads low, mirroring
Personalized-Quiz-Service's quiz_service.py pool-replenish pattern (that's
where generate_questions()/is_near_duplicate() are actually still used from
this module). If even the never-seen pool for a tier is too thin (a cold
bucket that hasn't been topped up yet), already-answered questions are
allowed back in as a last resort rather than cancelling the match outright
-- see _select_for_one_tier's docstring.

Deliberately does NOT reach into Personalized-Quiz-Service's single-player
attempt history (QuestionAttempt/QuizSession) -- that's a separate service
not modeled here at all (see app/models/question.py's own note on this).
"Already seen" here means "already answered in a previous BATTLE", not
"ever seen anywhere by this user".

Selection is split across difficulty tiers (see app/services/
battle_difficulty.py) rather than one flat difficulty for the whole match --
`tier_counts` (already-allocated exact integer counts per tier) drives one
never-seen-pool/repeat-as-last-resort pass PER TIER, sharing one running
"session" accumulator (questions already picked this match, across every
tier) across all of them so nothing repeats or near-duplicates across tiers.
A tier its own difficulty can't fill (e.g. a subject with no medium
questions yet) is topped up from the neighbouring difficulties rather than
cancelling the match -- see select_battle_questions.

warm_up_pools tops up every allowed subject's thin buckets once at startup,
so a subject nobody has battled in yet isn't empty the first time it's played.
"""
import asyncio
import logging
import math
import random
import time

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.battle_match import (
    BattleAnswer,
    BattleMatch,
    BattleMatchQuestion,
    BattleParticipant,
)
from app.models.question import Question
from app.services import telemetry_service
from app.services.groq_service import generate_questions, references_missing_visual
from app.services.question_fingerprint import compute_question_fingerprint
from app.services.text_similarity import is_near_duplicate

logger = logging.getLogger(__name__)


class BattleQuestionSelectionError(Exception):
    """Raised when the DB pool (never-seen, then repeats-as-last-resort)
    can't supply enough shared battle questions -- the caller
    (battle_gameplay_service) must cancel the match cleanly, never
    partially start it with too few questions.
    """


# Buckets currently being topped up in the background, keyed by
# (subject, difficulty) -- an in-process debounce so several concurrent
# matches hitting the same low bucket don't each spawn their own top-up
# task. A duplicate top-up across worker processes would just be a wasted
# Groq call, never a correctness problem, so no cross-process lock (same
# reasoning as quiz_service's own _replenishing_buckets).
_replenishing_buckets: set[tuple[str, str]] = set()

# Where a short tier borrows from, nearest difficulty first.
_NEIGHBOUR_DIFFICULTIES: dict[str, tuple[str, ...]] = {
    "easy": ("medium", "hard"),
    "medium": ("easy", "hard"),
    "hard": ("medium", "easy"),
}
_POOL_DIFFICULTIES = ("easy", "medium", "hard")


async def _fetch_battle_pool_questions(
    db: AsyncSession, *, subject: str, difficulty: str, limit: int,
    exclude_ids: set[int], history_texts: list[str], session_texts: list[str],
) -> list[Question]:
    """Mirrors quiz_service._fetch_pool_questions's shape (random sample, 3x
    over-fetch, near-duplicate accept-loop), drawing from ANY lesson within
    the subject. `exclude_ids` is whatever the caller wants filtered out via
    SQL (varies by phase -- see _select_for_one_tier). Near-duplicate
    checking is against `history_texts` (either player's PREVIOUS battle
    answers for this subject -- static, never mutated here) PLUS
    `session_texts` (this match's own picks so far, across every tier --
    mutated in place as candidates are accepted, so it keeps growing across
    calls). Passing an empty `history_texts` is how the caller allows
    already-seen questions back in as a last resort while still preventing
    a question from repeating within THIS match.

    Also triggers a debounced, fire-and-forget background top-up (see
    _kickoff_pool_replenish) whenever this (subject, difficulty) bucket is
    running low -- never blocks this read on that.
    """
    if limit <= 0:
        return []
    filters = [Question.subject == subject, Question.difficulty == difficulty]
    if exclude_ids:
        filters.append(Question.id.notin_(exclude_ids))
    stmt = select(Question).where(*filters).order_by(func.random()).limit(limit * 3)
    candidates = list((await db.execute(stmt)).scalars().all())

    dedup_pool = list(history_texts) + list(session_texts)
    accepted: list[Question] = []
    for candidate in candidates:
        if len(accepted) >= limit:
            break
        # Rows generated before the text-only rule may point at a graph/table that can't be shown.
        if references_missing_visual(candidate.question):
            continue
        if is_near_duplicate(candidate.question, dedup_pool):
            continue
        accepted.append(candidate)
        dedup_pool.append(candidate.question)
        session_texts.append(candidate.question)

    bucket_size_stmt = (
        select(func.count())
        .select_from(Question)
        .where(Question.subject == subject, Question.difficulty == difficulty)
    )
    bucket_size = (await db.execute(bucket_size_stmt)).scalar() or 0
    if bucket_size < settings.BATTLE_POOL_MIN_SIZE:
        _kickoff_pool_replenish(subject, difficulty)

    return accepted


def _kickoff_pool_replenish(subject: str, difficulty: str, *, grow: bool = False) -> None:
    # A thin, easily-mockable wrapper around asyncio.create_task -- tests
    # patch this directly so they never spawn a real background Groq call
    # against a DB session/event loop the test has already torn down.
    allowed = settings.battle_allowed_subjects
    if allowed and subject not in allowed:
        # Second line behind JoinQueueRequest's check (e.g. a match row from
        # before the allow-list existed): never ask Groq to write questions
        # into the shared pool for a subject outside the curriculum.
        logger.warning("Not replenishing battle pool for unlisted subject=%r", subject)
        return
    key = (subject, difficulty)
    if key in _replenishing_buckets:
        return
    _replenishing_buckets.add(key)
    asyncio.create_task(_replenish_pool_task(subject, difficulty, grow=grow))


async def _replenish_pool_task(subject: str, difficulty: str, *, grow: bool = False) -> None:
    """Runs after the triggering match activation has already returned its
    questions -- opens its own DB session (AsyncSessionLocal) since the
    request-scoped session is closed by then. Never lets a failure here
    surface anywhere; it's fully decoupled from any live match. Mirrors
    quiz_service._replenish_pool_task exactly (no per-user/lesson pinning,
    no dedup-against-existing-text on write -- dedup happens on READ, in
    _fetch_battle_pool_questions, against whatever's actually in play for a
    given match).
    """
    key = (subject, difficulty)
    try:
        async with AsyncSessionLocal() as session:
            count_stmt = (
                select(func.count())
                .select_from(Question)
                .where(Question.subject == subject, Question.difficulty == difficulty)
            )
            current = (await session.execute(count_stmt)).scalar() or 0
            # Normal top-up refills to the target size; a "grow" top-up (players
            # have run out of never-seen questions) adds one more batch past it,
            # up to the hard cap.
            ceiling = settings.BATTLE_POOL_MAX_SIZE if grow else settings.BATTLE_POOL_TARGET_SIZE
            needed = ceiling - current
            if needed <= 0:
                return

            batch_size = min(needed, settings.BATTLE_POOL_TOPUP_MAX_BATCH)
            try:
                batch = await generate_questions(
                    grade=settings.BATTLE_QUESTION_GRADE_DEFAULT,
                    subject=subject, difficulty=difficulty, question_count=batch_size,
                )
            except HTTPException as exc:
                logger.warning(
                    "Battle pool replenish failed subject=%s difficulty=%s: %s",
                    subject, difficulty, exc.detail,
                )
                return

            for q_data in batch:
                session.add(Question(
                    question=q_data["question"], options=q_data["options"],
                    correct_answer=q_data["correct_answer"], subject=subject,
                    lesson=q_data["lesson"], difficulty=difficulty,
                ))
            await session.commit()
            logger.info(
                "Battle pool replenished subject=%s difficulty=%s added=%d new_total=%d",
                subject, difficulty, len(batch), current + len(batch),
            )
    except Exception:
        logger.exception(
            "Unexpected error replenishing battle pool subject=%s difficulty=%s", subject, difficulty
        )
    finally:
        _replenishing_buckets.discard(key)


async def warm_up_pools() -> None:
    """One pass at startup: every allowed subject's (subject, difficulty)
    bucket below BATTLE_POOL_MIN_SIZE gets one background top-up, one bucket
    at a time (not all at once, to stay well inside Groq's rate limits).
    Buckets already stocked cost nothing -- _replenish_pool_task returns
    without calling Groq. Failures are logged and skipped, never fatal.
    """
    for subject in sorted(settings.battle_allowed_subjects):
        for difficulty in _POOL_DIFFICULTIES:
            key = (subject, difficulty)
            if key in _replenishing_buckets:
                continue
            try:
                async with AsyncSessionLocal() as session:
                    size = (
                        await session.execute(
                            select(func.count()).select_from(Question)
                            .where(Question.subject == subject, Question.difficulty == difficulty)
                        )
                    ).scalar() or 0
            except Exception:
                logger.exception("Pool warm-up couldn't read subject=%s difficulty=%s", subject, difficulty)
                continue
            if size >= settings.BATTLE_POOL_MIN_SIZE:
                continue
            logger.info("Warming up battle pool subject=%s difficulty=%s (has %d)", subject, difficulty, size)
            _replenishing_buckets.add(key)
            await _replenish_pool_task(subject, difficulty)  # releases the key itself


async def _select_for_one_tier(
    db: AsyncSession, *, subject: str, difficulty: str, count: int,
    history_ids: set[int], history_texts: list[str],
    session_ids: set[int], session_texts: list[str],
) -> tuple[list[Question], int]:
    """DB-pool-only, repeats-allowed-as-a-last-resort -- for ONE difficulty
    tier, drawn from any lesson within the subject.

    `history_ids`/`history_texts` are the STATIC seed (either player's
    PREVIOUS battle answers for this subject, from select_battle_questions'
    _seen_question_ids_and_texts call) -- never mutated. `session_ids`/
    `session_texts` are shared MUTABLE accumulators threaded across every
    tier of this match, growing as questions get picked, so nothing repeats
    or near-duplicates ACROSS tiers either, not just within one.

    Two phases, in order, each only running if the previous one left a
    shortfall:
      1. DB pool, excluding history + session -- "never seen before"
         honored for free.
      2. DB pool AGAIN, but with `history_texts`/`history_ids` dropped from
         the exclusion (passing history_texts=[] to the pool fetch, and
         only `session_ids` -- not `history_ids` -- to the SQL filter) --
         i.e. a previously-answered question is now allowed back in, but a
         question already picked THIS match still isn't. This only fires
         when phase 1 couldn't fill the tier (the never-seen pool for this
         difficulty is too thin for this pair of players, e.g. a cold
         bucket the background replenish hasn't caught up on yet) --
         logged, since silently serving a repeat is a real behavior change
         worth being visible in the logs.

    Returns the tier's questions (possibly still short if even phase 2
    couldn't cover it -- caller decides whether that's fatal) plus how many
    of them were served as repeats, for the caller's aggregated logging.
    """
    questions = await _fetch_battle_pool_questions(
        db, subject=subject, difficulty=difficulty, limit=count,
        exclude_ids=history_ids | session_ids,
        history_texts=history_texts, session_texts=session_texts,
    )
    session_ids.update(q.id for q in questions)

    remaining = count - len(questions)
    repeat_count = 0
    if remaining > 0:
        repeat_questions = await _fetch_battle_pool_questions(
            db, subject=subject, difficulty=difficulty, limit=remaining,
            exclude_ids=session_ids,  # history_ids deliberately NOT excluded here
            history_texts=[], session_texts=session_texts,
        )
        if repeat_questions:
            logger.warning(
                "Battle question selection fell back to repeat questions "
                "subject=%s difficulty=%s count=%d (never-seen pool came up "
                "short).",
                subject, difficulty, len(repeat_questions),
            )
        # These players have outgrown the pool -- add fresh questions in the
        # background (past the normal target, up to BATTLE_POOL_MAX_SIZE) so
        # their next matches aren't repeats they've already memorised.
        _kickoff_pool_replenish(subject, difficulty, grow=True)
        questions.extend(repeat_questions)
        session_ids.update(q.id for q in repeat_questions)
        repeat_count = len(repeat_questions)

    return questions, repeat_count


async def _seen_battle_questions(
    db: AsyncSession, *, subject: str, participant_user_ids: tuple[int, int],
) -> list[tuple[int, str, str]]:
    """Every question either player has been SHOWN in a previous battle for
    this subject -- (id, text, lesson), most recently seen first. "Shown"
    means reached: everything up to and including the question a player was
    on when their match ended (BattleMatchQuestion.question_order <= their
    current_question_index), not just the ones they answered or skipped --
    a question on screen when the clock ran out was still seen. Answered
    questions are included directly too (BattleAnswer), which also covers
    matches played before progress was tracked. Battles only: practice-mode
    history is a separate service's and deliberately not considered.
    """
    reached = (
        await db.execute(
            select(Question.id, Question.question, Question.lesson, BattleMatch.started_at)
            .join(BattleMatchQuestion, BattleMatchQuestion.question_id == Question.id)
            .join(BattleMatch, BattleMatch.id == BattleMatchQuestion.match_id)
            .join(BattleParticipant, BattleParticipant.match_id == BattleMatch.id)
            .where(
                Question.subject == subject,
                BattleParticipant.user_id.in_(participant_user_ids),
                BattleMatch.status.in_(("active", "completed")),
                BattleMatchQuestion.question_order <= BattleParticipant.current_question_index,
            )
        )
    ).all()
    answered = (
        await db.execute(
            select(Question.id, Question.question, Question.lesson, BattleMatch.started_at)
            .join(BattleAnswer, BattleAnswer.question_id == Question.id)
            .join(BattleMatch, BattleMatch.id == BattleAnswer.match_id)
            .where(BattleAnswer.user_id.in_(participant_user_ids), Question.subject == subject)
        )
    ).all()
    latest: dict[int, tuple[str, str, float]] = {}
    for qid, text, lesson, seen_at in [*reached, *answered]:
        stamp = seen_at.timestamp() if seen_at is not None else 0.0
        if qid not in latest or stamp > latest[qid][2]:
            latest[qid] = (text, lesson, stamp)
    ordered = sorted(latest.items(), key=lambda kv: -kv[1][2])
    return [(qid, text, lesson) for qid, (text, lesson, _) in ordered]


async def _seen_question_ids_and_texts(
    db: AsyncSession, *, subject: str, participant_user_ids: tuple[int, int],
) -> tuple[set[int], list[str]]:
    """IDs (for the pool's notin_ filter) and texts (for the near-duplicate
    check, so a reworded copy is caught too) of every question either player
    has already been shown in a battle for this subject -- see
    _seen_battle_questions."""
    seen = await _seen_battle_questions(db, subject=subject, participant_user_ids=participant_user_ids)
    return {qid for qid, _, _ in seen}, [text for _, text, _ in seen]


# ── AI-first ────────────────────────────────────────────────────────────

_ai_generation_slots: asyncio.Semaphore | None = None


def _ai_slots() -> asyncio.Semaphore:
    global _ai_generation_slots
    if _ai_generation_slots is None:
        _ai_generation_slots = asyncio.Semaphore(settings.BATTLE_AI_MAX_CONCURRENT_GENERATIONS)
    return _ai_generation_slots


async def _generate_ai_candidates(
    *, subject: str, tier_counts: dict[str, int], avoid_texts: list[str], avoid_lessons: list[str],
) -> tuple[dict[str, list[dict]], dict]:
    """One AI request per difficulty tier, all at once, each asking for a
    few more than the tier needs (BATTLE_AI_OVERGENERATE_RATIO). Whatever
    has come back within BATTLE_AI_GENERATION_TIMEOUT_SECONDS is used; the
    rest is cancelled. A failed or late tier just comes back empty -- the
    pool covers it. Returns candidates per difficulty plus counters for
    telemetry."""
    wanted = {
        difficulty: count + max(1, math.ceil(count * settings.BATTLE_AI_OVERGENERATE_RATIO))
        for difficulty, count in tier_counts.items() if count > 0
    }
    stats = {"requested": sum(wanted.values()), "invalid": 0, "failed_tiers": 0, "timed_out_tiers": 0}
    telemetry_by_tier = {difficulty: {} for difficulty in wanted}
    tasks = {
        difficulty: asyncio.create_task(generate_questions(
            grade=settings.BATTLE_QUESTION_GRADE_DEFAULT, subject=subject, difficulty=difficulty,
            question_count=n, existing_questions=avoid_texts, avoid_lessons=avoid_lessons,
            telemetry=telemetry_by_tier[difficulty],
        ))
        for difficulty, n in wanted.items()
    }
    _done, pending = await asyncio.wait(tasks.values(), timeout=settings.BATTLE_AI_GENERATION_TIMEOUT_SECONDS)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    results: dict[str, list[dict]] = {}
    for difficulty, task in tasks.items():
        if task in pending:
            stats["timed_out_tiers"] += 1
            results[difficulty] = []
            continue
        exc = task.exception()
        if exc is not None:
            stats["failed_tiers"] += 1
            logger.warning("AI battle questions failed subject=%s difficulty=%s: %s", subject, difficulty, exc)
            results[difficulty] = []
            continue
        results[difficulty] = task.result()
        stats["invalid"] += telemetry_by_tier[difficulty].get("invalid_question_count", 0)
    return results, stats


async def _ai_first_questions(
    db: AsyncSession, *, subject: str, tier_counts: dict[str, int],
    seen: list[tuple[int, str, str]],
) -> dict[str, list[Question]]:
    """Fresh AI questions for this match, per difficulty tier -- each one
    checked to be neither a repeat nor a near-copy (exact fingerprint, or
    the shared 55%-word-overlap rule) of anything either player has seen in
    a battle, of anything already in this subject's pool, or of another
    question in this batch. Every question that passes is saved to the pool
    (in the caller's transaction); up to each tier's count is returned for
    this match, any extras stay in the pool for later matches. Returns an
    empty result -- never raises -- when the AI is busy (all generation
    slots taken), slow, or failing: the pool then covers the whole match.
    """
    slots = _ai_slots()
    if slots.locked():
        logger.info("AI battle questions skipped subject=%s — all generation slots busy; using the pool.", subject)
        return {}

    avoid_texts = [text for _, text, _ in seen[: settings.BATTLE_AI_AVOID_RECENT_QUESTIONS]]
    lesson_counts: dict[str, int] = {}
    for _, _, lesson in seen:
        lesson_counts[lesson] = lesson_counts.get(lesson, 0) + 1
    avoid_lessons = sorted(lesson_counts, key=lambda lesson: -lesson_counts[lesson])[: settings.BATTLE_AI_AVOID_TOP_LESSONS]

    started = time.monotonic()
    async with slots:
        candidates, stats = await _generate_ai_candidates(
            subject=subject, tier_counts=tier_counts, avoid_texts=avoid_texts, avoid_lessons=avoid_lessons,
        )

    pool_rows = (
        await db.execute(select(Question.question, Question.question_fingerprint).where(Question.subject == subject))
    ).all()
    blocked_texts = [text for _, text, _ in seen] + [row[0] for row in pool_rows]
    blocked_fingerprints = {row[1] for row in pool_rows}

    picked: dict[str, list[Question]] = {}
    duplicates = saved = 0
    for difficulty, batch in candidates.items():
        accepted: list[Question] = []
        for item in batch:
            text = item["question"]
            if references_missing_visual(text):
                continue
            fingerprint = compute_question_fingerprint(subject, item["lesson"], text)
            if fingerprint in blocked_fingerprints or is_near_duplicate(text, blocked_texts):
                duplicates += 1
                continue
            question = Question(
                question=text, options=item["options"], correct_answer=item["correct_answer"],
                subject=subject, lesson=item["lesson"], difficulty=difficulty,
            )
            db.add(question)
            accepted.append(question)
            blocked_fingerprints.add(fingerprint)
            blocked_texts.append(text)
        saved += len(accepted)
        picked[difficulty] = accepted[: tier_counts.get(difficulty, 0)]
    if saved:
        await db.flush()  # ids for the match's BattleMatchQuestion rows

    used = sum(len(q) for q in picked.values())
    needed = sum(tier_counts.values())
    latency_ms = (time.monotonic() - started) * 1000
    logger.info(
        "AI battle questions subject=%s: %d/%d for this match (%d new saved to the pool, %d repeats/near-copies "
        "dropped, %d tier(s) failed, %d timed out) in %.0fms",
        subject, used, needed, saved, duplicates, stats["failed_tiers"], stats["timed_out_tiers"], latency_ms,
    )
    _kickoff_ai_telemetry(
        subject=subject, requested=stats["requested"], generated=saved,
        success=used >= needed, used_pool=used < needed, duplicates=duplicates,
        invalid=stats["invalid"], latency_ms=latency_ms,
        error_category=None if used >= needed else ("provider_error" if stats["failed_tiers"] or stats["timed_out_tiers"] else "insufficient_pool"),
    )
    return picked


def _kickoff_ai_telemetry(**fields) -> None:
    # Written in the background on its own session: record_generation_event
    # commits, and this runs inside the match-start transaction.
    asyncio.create_task(_record_ai_telemetry(**fields))


async def _record_ai_telemetry(
    *, subject: str, requested: int, generated: int, success: bool, used_pool: bool,
    duplicates: int, invalid: int, latency_ms: float, error_category: str | None,
) -> None:
    async with AsyncSessionLocal() as session:
        await telemetry_service.record_generation_event(
            session, user_id=None, subject=subject,
            requested_question_count=requested, generated_question_count=generated,
            provider="groq", model_name=settings.GROQ_MODEL, success=success,
            used_cache_fallback=used_pool, retry_count=0, duplicate_count=duplicates,
            invalid_question_count=invalid, latency_ms=latency_ms, error_category=error_category,
        )


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
    just a set neither of them has seen before -- UNLESS even that can't be
    filled (the never-seen pool for a tier's difficulty is too thin), in
    which case _select_for_one_tier's second phase allows already-answered
    questions back in as a last resort rather than cancelling the match; see
    its docstring. A tier still short after that borrows from the
    neighbouring difficulties. Raises BattleQuestionSelectionError only if
    even that can't fill it -- caller cancels the whole match, never starts
    it partially filled.

    Never calls Groq itself -- purely a DB read (fast, no external-API
    latency or rate-limit risk in the match-activation critical path). Pool
    top-ups happen out-of-band; see this module's docstring.

    Order of the returned list is randomized so a question's difficulty
    tier and source (never-seen vs. repeat) never correlate with its
    position in the match.
    """
    seen = await _seen_battle_questions(db, subject=subject, participant_user_ids=participant_user_ids)
    history_ids = {qid for qid, _, _ in seen}
    history_texts = [text for _, text, _ in seen]
    # Grows as tiers are processed -- see _select_for_one_tier's docstring.
    # Kept separate from history_ids/history_texts (static) so the last-
    # resort repeat phase can drop history from the exclusion while still
    # forbidding a question from repeating within THIS match.
    session_ids: set[int] = set()
    session_texts: list[str] = []

    questions: list[Question] = []
    total_requested = sum(tier_counts.values())

    # AI first: fresh questions neither player has seen; the pool (below)
    # only fills what the AI couldn't provide in time.
    if settings.BATTLE_AI_FIRST_ENABLED:
        ai_picked = await _ai_first_questions(db, subject=subject, tier_counts=tier_counts, seen=seen)
        for difficulty, picked in ai_picked.items():
            questions.extend(picked)
            session_ids.update(q.id for q in picked)
            session_texts.extend(q.question for q in picked)
        tier_counts = {
            difficulty: count - len(ai_picked.get(difficulty, []))
            for difficulty, count in tier_counts.items()
        }

    agg_repeats = 0
    short_tiers: list[str] = []

    for difficulty, count in tier_counts.items():
        if count <= 0:
            continue
        tier_questions, repeat_count = await _select_for_one_tier(
            db, subject=subject, difficulty=difficulty, count=count,
            history_ids=history_ids, history_texts=history_texts,
            session_ids=session_ids, session_texts=session_texts,
        )
        questions.extend(tier_questions)
        agg_repeats += repeat_count

        # Not enough at this difficulty (e.g. a subject with no medium
        # questions yet): borrow from the nearest other difficulties instead
        # of cancelling the match. The thin bucket itself is already being
        # topped up in the background (_fetch_battle_pool_questions).
        missing = count - len(tier_questions)
        for neighbour in _NEIGHBOUR_DIFFICULTIES.get(difficulty, ()):
            if missing <= 0:
                break
            borrowed, borrowed_repeats = await _select_for_one_tier(
                db, subject=subject, difficulty=neighbour, count=missing,
                history_ids=history_ids, history_texts=history_texts,
                session_ids=session_ids, session_texts=session_texts,
            )
            if borrowed:
                logger.warning(
                    "Battle question selection borrowed %d %s question(s) for the %s tier subject=%s",
                    len(borrowed), neighbour, difficulty, subject,
                )
            questions.extend(borrowed)
            agg_repeats += borrowed_repeats
            missing -= len(borrowed)
        if missing > 0:
            short_tiers.append(difficulty)

    if agg_repeats:
        logger.warning(
            "Battle match question selection served %d repeat question(s) "
            "subject=%s (never-seen pool came up short for at least one "
            "tier).",
            agg_repeats, subject,
        )

    if short_tiers:
        raise BattleQuestionSelectionError(
            f"Only {len(questions)}/{total_requested} questions available "
            f"(subject={subject}, short tiers={short_tiers})"
        )

    random.shuffle(questions)
    return questions[:total_requested]

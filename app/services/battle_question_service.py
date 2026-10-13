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

Match activation NEVER calls Groq live -- it only ever reads the DB pool
(see _select_for_one_tier), so a slow, rate-limited, or down Groq can no
longer stall or fail a match start, and question selection is a fast,
local-only DB read. Keeping each (subject, difficulty) pool stocked is
instead a background concern: _fetch_battle_pool_questions triggers a
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
"""
import asyncio
import logging
import random

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.battle_match import BattleAnswer
from app.models.question import Question
from app.services.groq_service import generate_questions
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


async def _seen_question_ids_and_texts(
    db: AsyncSession, *, subject: str, participant_user_ids: tuple[int, int],
) -> tuple[set[int], list[str]]:
    """Every question either player has already answered in a PREVIOUS
    battle for this subject, across ALL lessons -- the "never repeat" source
    of truth. BattleAnswer already carries user_id directly (no join needed
    to reach it, unlike single-player's QuestionAttempt->QuizSession chain).
    Returns both IDs (for the notin_ pool filter) and texts (for the
    near-duplicate check, so a pool question that's a rewording of a seen
    question gets caught too, not just an exact-ID repeat).
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
    just a set neither of them has seen before -- UNLESS even that can't be
    filled (the never-seen pool for a tier's difficulty is too thin), in
    which case _select_for_one_tier's second phase allows already-answered
    questions back in as a last resort rather than cancelling the match; see
    its docstring. Raises BattleQuestionSelectionError only if even repeats
    can't fill a tier -- caller cancels the whole match, never starts it
    partially filled.

    Never calls Groq itself -- purely a DB read (fast, no external-API
    latency or rate-limit risk in the match-activation critical path). Pool
    top-ups happen out-of-band; see this module's docstring.

    Order of the returned list is randomized so a question's difficulty
    tier and source (never-seen vs. repeat) never correlate with its
    position in the match.
    """
    history_ids, history_texts = await _seen_question_ids_and_texts(
        db, subject=subject, participant_user_ids=participant_user_ids,
    )
    # Grows as tiers are processed -- see _select_for_one_tier's docstring.
    # Kept separate from history_ids/history_texts (static) so the last-
    # resort repeat phase can drop history from the exclusion while still
    # forbidding a question from repeating within THIS match.
    session_ids: set[int] = set()
    session_texts: list[str] = []

    questions: list[Question] = []
    total_requested = sum(tier_counts.values())

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
        if len(tier_questions) < count:
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

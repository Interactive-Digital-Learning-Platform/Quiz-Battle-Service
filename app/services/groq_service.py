import asyncio
import json
import logging
import random
from collections.abc import Iterable
from datetime import UTC, datetime

from fastapi import HTTPException, status
from groq import (
    APIConnectionError,
    APITimeoutError,
    AsyncGroq,
    GroqError,
    InternalServerError,
    RateLimitError,
)
from groq.types.chat import ChatCompletionMessageParam

from app.core.config import settings

logger = logging.getLogger(__name__)

_groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)

# Only these are worth retrying — rate limits, timeouts, dropped connections,
# and Groq-side 5xx are transient. Everything else (bad auth, bad request)
# would just fail the same way again, so those raise immediately instead.
_TRANSIENT_GROQ_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


async def _create_chat_completion_with_retry(
    *, messages: Iterable[ChatCompletionMessageParam], max_tokens: int = 4096,
):
    # Retries transient Groq errors with backoff + jitter. Without this, one
    # rate-limit blip used to fail the whole quiz generation and fall back to
    # the (often near-empty) DB cache far more often than it should have.
    last_exc: Exception | None = None
    for attempt in range(settings.GROQ_MAX_RETRIES + 1):
        try:
            return await _groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=messages,
                temperature=0.95,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except _TRANSIENT_GROQ_ERRORS as exc:
            last_exc = exc
            if attempt == settings.GROQ_MAX_RETRIES:
                break
            delay = settings.GROQ_RETRY_BASE_DELAY_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                "Transient Groq error (%s) on attempt %d/%d, retrying in %.1fs: %s",
                type(exc).__name__, attempt + 1, settings.GROQ_MAX_RETRIES + 1, delay, exc,
            )
            await asyncio.sleep(delay)
    # The loop only falls through here after `last_exc` has been set (it
    # exits via `break` right after catching one) -- assert makes that
    # invariant explicit for the type checker too.
    assert last_exc is not None
    raise last_exc


async def generate_questions(
    grade: int,
    subject: str,
    difficulty: str,
    question_count: int,
    lesson: str | None = None,
    existing_questions: list[str] | None = None,
    avoid_lessons: list[str] | None = None,
    preferred_lessons: list[str] | None = None,
    adaptive_context: str | None = None,
    telemetry: dict | None = None,
) -> list[dict]:
    # Asks Groq for `question_count` MCQs and hands back a validated list of
    # dicts (question/options/correct_answer/explanation/lesson). If `lesson`
    # is omitted, each question gets its own randomly varied lesson instead
    # of the whole quiz being pinned to one topic. `existing_questions` are
    # passed to the model so it knows what not to repeat; `avoid_lessons`
    # nudges it toward topic variety across separate quiz generations, not
    # just within one quiz.
    #
    # `preferred_lessons` and `adaptive_context` are optional, additive hints
    # from the adaptive-mastery engine (weak lessons to weight coverage
    # toward, and a short plain-language note about the student's current
    # standing) — both default to None and change nothing about the prompt
    # when omitted, so every existing caller (and the lesson-pinned branch,
    # which never receives them) behaves exactly as before. Only ever a
    # short natural-language hint — never raw DB rows/ids/scores, per the
    # "don't expose unnecessary internal DB details to the LLM" requirement.
    #
    # Raises HTTPException(502) on any Groq/parsing failure.
    existing_questions = existing_questions or []
    avoid_lessons = avoid_lessons or []
    random_lessons = lesson is None

    system_prompt = """You are an expert educational content creator for Sri Lankan school students.
Your task is to generate UNIQUE, DIVERSE multiple-choice quiz questions.

CRITICAL: You MUST respond with ONLY a valid JSON object in exactly this structure:
{
  "questions": [
    {
      "question": "The full question text here?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option A",
      "explanation": "Brief explanation of why this is correct.",
      "lesson": "The specific lesson/topic this question covers"
    }
  ]
}

UNIQUENESS RULES (highest priority):
- Every question MUST test a DIFFERENT specific fact, concept, or skill within the topic.
- DO NOT repeat or rephrase any question from the EXISTING QUESTIONS list provided.
- DO NOT use the same stem structure twice (e.g., "Which of the following..." twice in a row).
- DO NOT cluster questions around one narrow sub-topic — spread coverage across the full lesson.

VARIETY REQUIREMENTS — mix ALL of the following styles across the set:
- Factual recall: "What is X?", "Which term describes Y?"
- Application: "A student does X — what happens?", "If Y occurs, what is the result?"
- Cause-and-effect: "Why does X happen?", "What causes Y?"
- Comparison: "What is the difference between X and Y?"
- Scenario / real-world: embed the concept in a concrete Sri Lankan or everyday context.
- Negation (use sparingly): "Which of the following is NOT true about X?"
- Numerical / formula-based (for maths/science topics where appropriate).

QUALITY RULES:
- Each question must have EXACTLY 4 options.
- correct_answer must be EXACTLY one of the 4 options (copy it verbatim).
- All 4 options must be plausible — avoid obviously wrong distractors.
- Questions must be appropriate for the specified grade level.
- The "lesson" field must always be a real, specific topic name from the
  subject's standard syllabus (e.g. "Algebra", "Photosynthesis") — never the
  subject name itself and never generic ("General", "Miscellaneous", etc).
- Do NOT include numbering in the question text.
- Do NOT output anything outside the JSON object.
"""

    seed_context = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    exclusion_block = ""
    if existing_questions:
        exclusion_lines = "\n".join(f"- {q}" for q in existing_questions[:60])
        exclusion_block = (
            f"\n\nEXISTING QUESTIONS TO AVOID (do NOT repeat or rephrase any of these):\n"
            f"{exclusion_lines}\n"
        )

    if random_lessons:
        avoid_lessons_block = ""
        if avoid_lessons:
            avoid_lines = "\n".join(f"- {l}" for l in avoid_lessons[:10])
            avoid_lessons_block = (
                f"\n\nThis student was recently quizzed on these lessons for this subject — "
                f"prefer OTHER lessons where possible for variety:\n{avoid_lines}\n"
            )

        preferred_lessons_block = ""
        if preferred_lessons:
            preferred_lines = "\n".join(f"- {l}" for l in preferred_lessons[:10])
            preferred_lessons_block = (
                f"\n\nThis student would benefit from extra practice in these lessons — "
                f"include AT LEAST ONE question from each if possible, without making every "
                f"question come from only these:\n{preferred_lines}\n"
            )

        adaptive_context_block = f"\n\nStudent context: {adaptive_context}\n" if adaptive_context else ""

        user_prompt = (
            f"[Request ID: {seed_context}]\n\n"
            f"Generate {question_count} UNIQUE {difficulty}-difficulty multiple-choice questions "
            f"for Grade {grade} Sri Lankan students.\n"
            f"Subject: {subject}\n"
            f"{exclusion_block}"
            f"{avoid_lessons_block}"
            f"{preferred_lessons_block}"
            f"{adaptive_context_block}\n"
            f"Requirements:\n"
            f"- Do NOT focus on a single lesson. EACH question must come from a DIFFERENT, "
            f"randomly chosen lesson/topic within the full '{subject}' syllabus for this grade.\n"
            f"- Spread the {question_count} questions across the breadth of the subject — "
            f"avoid picking the same lesson for more than one question unless the subject "
            f"genuinely has too few lessons to avoid it.\n"
            f"- Set each question's \"lesson\" field to the specific topic IT individually covers.\n"
            f"- Use a variety of question styles (factual, applied, scenario, comparison, cause-effect).\n"
            f"- Each question must be clearly distinct from all others in this set.\n\n"
            f"Return exactly {question_count} questions in the required JSON format."
        )
    else:
        adaptive_context_block = f"\n\nStudent context: {adaptive_context}\n" if adaptive_context else ""
        user_prompt = (
            f"[Request ID: {seed_context}]\n\n"
            f"Generate {question_count} UNIQUE {difficulty}-difficulty multiple-choice questions "
            f"for Grade {grade} Sri Lankan students.\n"
            f"Subject: {subject}\n"
            f"Lesson / Topic: {lesson}\n"
            f"{exclusion_block}"
            f"{adaptive_context_block}\n"
            f"Requirements:\n"
            f"- Cover {question_count} DIFFERENT aspects or sub-concepts within '{lesson}'.\n"
            f"- Set every question's \"lesson\" field to exactly \"{lesson}\".\n"
            f"- Use a variety of question styles (factual, applied, scenario, comparison, cause-effect).\n"
            f"- Each question must be clearly distinct from all others in this set.\n\n"
            f"Return exactly {question_count} questions in the required JSON format."
        )

    try:
        logger.info(
            "Calling Groq API: model=%s, subject=%s, lesson=%s, count=%d, excluding=%d existing",
            settings.GROQ_MODEL, subject, lesson or "<random per question>", question_count, len(existing_questions),
        )

        response = await _create_chat_completion_with_retry(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

    except GroqError as exc:
        logger.error("Groq API error during question generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service error: {exc}",
        )

    raw_content = response.choices[0].message.content or ""

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        logger.error("Groq returned non-JSON content: %s", raw_content[:500])
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI returned malformed JSON: {exc}",
        )

    raw_questions: list = data.get("questions", [])
    if not isinstance(raw_questions, list) or len(raw_questions) == 0:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI returned an empty question list. Please try again.",
        )

    validated: list[dict] = []
    for i, q in enumerate(raw_questions[:question_count]):
        if not isinstance(q, dict):
            continue

        question_text = str(q.get("question", "")).strip()
        options = q.get("options", [])
        correct = str(q.get("correct_answer", "")).strip()
        explanation = str(q.get("explanation", "")).strip()
        q_lesson = lesson if lesson else (str(q.get("lesson", "")).strip() or subject)

        # Must have exactly 4 options (Groq occasionally ignores that rule
        # and returns 5+). Truncating to 4 risks losing the actual correct
        # answer, so a bad-count question is dropped entirely rather than
        # patched up — the caller's retry loop tops up the shortfall.
        if not question_text or not isinstance(options, list) or len(options) != 4:
            logger.warning(
                "Skipping malformed question at index %d: expected 4 options, got %d",
                i, len(options) if isinstance(options, list) else 0,
            )
            continue

        # correct_answer has to actually match one of the options (exact, or
        # case/whitespace-insensitive). If it matches neither, there's no way
        # to know which option is really correct — dropping it beats the old
        # behavior of defaulting to options[0] and mislabeling a wrong answer
        # as correct.
        if correct not in options:
            match = next((o for o in options if o.strip().lower() == correct.strip().lower()), None)
            if match is None:
                logger.warning(
                    "Skipping question at index %d: correct_answer %r not found in options %r",
                    i, correct, options,
                )
                continue
            correct = match

        validated.append({
            "question": question_text,
            "options": options,
            "correct_answer": correct,
            "explanation": explanation,
            "lesson": q_lesson,
        })

    if telemetry is not None:
        invalid_count = len(raw_questions[:question_count]) - len(validated)
        telemetry["invalid_question_count"] = telemetry.get("invalid_question_count", 0) + invalid_count

    if not validated:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI returned no valid questions after validation.",
        )

    logger.info("Groq generated %d valid questions (requested %d)", len(validated), question_count)
    return validated


async def generate_shuffle_quiz_questions(
    grade: int,
    subject_plan: list[tuple[str, str, int]],
    avoid_lessons_by_subject: dict[str, list[str]] | None = None,
    existing_questions_by_subject: dict[str, list[str]] | None = None,
    telemetry: dict | None = None,
) -> list[dict]:
    """Shuffle Mode's entire AI phase in ONE Groq call: asks for every
    subject's questions at once, each at that subject's own current
    adaptive difficulty, instead of one call (or several retries) per
    subject. `subject_plan` is a list of (subject, difficulty, count) --
    each subject's difficulty is decided by the caller from the student's
    real mastery data (see difficulty_service), never left for the model
    to pick. `existing_questions_by_subject` is that subject's already-
    cached question texts (same "avoid these" mechanism generate_questions()
    uses for normal quizzes) -- without it, Groq has no way to know a
    question it's about to write already exists in the DB, which is why
    Shuffle Mode kept resurfacing near-identical questions before this.

    Returns a flat list of validated question dicts, each tagged with which
    subject it belongs to: {"subject", "lesson", "difficulty", "question",
    "options", "correct_answer", "explanation"}. `difficulty` is always
    copied from `subject_plan`, never trusted from the AI's own response.
    A question whose "subject" doesn't match one of the requested subjects
    is dropped rather than guessed at.

    Raises HTTPException(502) on any Groq/parsing failure, same as
    generate_questions() -- the caller falls back to the DB cache.
    """
    avoid_lessons_by_subject = avoid_lessons_by_subject or {}
    existing_questions_by_subject = existing_questions_by_subject or {}
    total_count = sum(count for _, _, count in subject_plan)

    system_prompt = """You are an expert educational content creator for Sri Lankan school students.
Your task is to generate UNIQUE, DIVERSE multiple-choice quiz questions covering SEVERAL subjects in one batch.

CRITICAL: You MUST respond with ONLY a valid JSON object in exactly this structure:
{
  "questions": [
    {
      "subject": "Exactly one of the requested subject names",
      "lesson": "The specific lesson/topic this question covers",
      "question": "The full question text here?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option A",
      "explanation": "Brief explanation of why this is correct."
    }
  ]
}

UNIQUENESS RULES (highest priority):
- Every question MUST test a DIFFERENT specific fact, concept, or skill.
- DO NOT repeat or rephrase any other question in this same batch.
- DO NOT cluster questions around one narrow sub-topic within a subject.
- Where a subject lists "EXISTING QUESTIONS TO AVOID", none of your new questions for that
  subject may repeat or closely rephrase any of them.

VARIETY REQUIREMENTS — mix ALL of the following styles across the set:
- Factual recall, application, cause-and-effect, comparison, scenario/real-world,
  negation (sparingly), and numerical/formula-based (for maths/science where appropriate).

QUALITY RULES:
- Each question must have EXACTLY 4 options.
- correct_answer must be EXACTLY one of the 4 options (copy it verbatim).
- All 4 options must be plausible — avoid obviously wrong distractors.
- Questions must be appropriate for the specified grade level.
- "subject" MUST be copied verbatim from the requested subject list below — never invent or rename one.
- "lesson" must be a real, specific topic name from that subject's standard syllabus — never the
  subject name itself and never generic ("General", "Miscellaneous", etc).
- Do NOT include numbering in the question text.
- Do NOT output anything outside the JSON object.
"""

    seed_context = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    # Capped lower than generate_questions()'s single-subject 60 -- a shuffle
    # prompt already lists this block once per subject, so an uncapped list
    # here would blow up prompt size fast on a many-subject shuffle.
    _EXISTING_QUESTIONS_PER_SUBJECT_CAP = 20

    plan_lines = []
    for subject, difficulty, count in subject_plan:
        avoid = avoid_lessons_by_subject.get(subject) or []
        line = f"- {subject}: {count} questions at {difficulty} difficulty"
        if avoid:
            line += f" (this student was recently quizzed on: {', '.join(avoid[:6])} — prefer other lessons where possible)"

        existing = existing_questions_by_subject.get(subject) or []
        if existing:
            existing_lines = "\n".join(
                f"    - {q}" for q in existing[:_EXISTING_QUESTIONS_PER_SUBJECT_CAP]
            )
            line += f"\n  EXISTING QUESTIONS TO AVOID for {subject} (do NOT repeat or rephrase any of these):\n{existing_lines}"

        plan_lines.append(line)

    user_prompt = (
        f"[Request ID: {seed_context}]\n\n"
        f"Generate a batch of {total_count} UNIQUE multiple-choice questions for Grade {grade} "
        f"Sri Lankan students, split EXACTLY as follows across subjects:\n"
        + "\n".join(plan_lines) +
        f"\n\nRequirements:\n"
        f"- Return EXACTLY {total_count} questions total, matching the per-subject counts above.\n"
        f"- Within each subject, EACH question must come from a DIFFERENT lesson/topic — spread "
        f"coverage across the subject's syllabus rather than clustering on one lesson.\n"
        f"- Use a variety of question styles across the whole batch.\n\n"
        f"Return exactly {total_count} questions in the required JSON format."
    )

    # Scales with the batch size instead of the fixed 4096 single-subject
    # calls use — a multi-subject shuffle batch can ask for meaningfully
    # more completion tokens than one subject alone ever would.
    max_tokens = min(8192, max(4096, total_count * 220))

    try:
        logger.info(
            "Calling Groq API (shuffle batch): model=%s, subjects=%d, total_count=%d, excluding=%d existing",
            settings.GROQ_MODEL, len(subject_plan), total_count,
            sum(len(v) for v in existing_questions_by_subject.values()),
        )
        response = await _create_chat_completion_with_retry(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )
    except GroqError as exc:
        logger.error("Groq API error during shuffle question generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service error: {exc}",
        )

    raw_content = response.choices[0].message.content or ""

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        logger.error("Groq returned non-JSON content: %s", raw_content[:500])
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI returned malformed JSON: {exc}",
        )

    raw_questions: list = data.get("questions", [])
    if not isinstance(raw_questions, list) or len(raw_questions) == 0:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI returned an empty question list. Please try again.",
        )

    difficulty_by_subject = {subject: difficulty for subject, difficulty, _ in subject_plan}
    subject_by_lower = {subject.lower(): subject for subject, _, _ in subject_plan}

    # A little overshoot is tolerated (the model sometimes returns a few
    # extra) -- the caller caps each subject at its own requested count.
    cap = total_count * 2

    validated: list[dict] = []
    for i, q in enumerate(raw_questions[:cap]):
        if not isinstance(q, dict):
            continue

        raw_subject = str(q.get("subject", "")).strip()
        subject = subject_by_lower.get(raw_subject.lower())
        if subject is None:
            logger.warning("Skipping shuffle question at index %d: unrecognized subject %r", i, raw_subject)
            continue

        question_text = str(q.get("question", "")).strip()
        options = q.get("options", [])
        correct = str(q.get("correct_answer", "")).strip()
        explanation = str(q.get("explanation", "")).strip()
        question_lesson = str(q.get("lesson", "")).strip() or subject

        if not question_text or not isinstance(options, list) or len(options) != 4:
            logger.warning(
                "Skipping malformed shuffle question at index %d: expected 4 options, got %d",
                i, len(options) if isinstance(options, list) else 0,
            )
            continue

        if correct not in options:
            match = next((o for o in options if o.strip().lower() == correct.strip().lower()), None)
            if match is None:
                logger.warning(
                    "Skipping shuffle question at index %d: correct_answer %r not found in options %r",
                    i, correct, options,
                )
                continue
            correct = match

        validated.append({
            "subject": subject,
            "lesson": question_lesson,
            "difficulty": difficulty_by_subject[subject],
            "question": question_text,
            "options": options,
            "correct_answer": correct,
            "explanation": explanation,
        })

    if telemetry is not None:
        invalid_count = len(raw_questions[:cap]) - len(validated)
        telemetry["invalid_question_count"] = telemetry.get("invalid_question_count", 0) + invalid_count

    if not validated:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI returned no valid questions after validation.",
        )

    logger.info("Groq generated %d valid shuffle questions (requested %d)", len(validated), total_count)
    return validated


async def generate_feedback(analytics_summary: dict) -> dict:
    # Turns the user's analytics into a short, personalised pep talk from
    # Groq. `analytics_summary["recommendations"]` is deterministic and
    # database-driven (see recommendation_service.py, no AI involved) — the
    # prompt tells Groq to treat it as ground truth and build suggestions
    # from it rather than inventing its own analysis, so the AI's wording
    # stays consistent with the numbers shown elsewhere in the app.
    system_prompt = """You are a knowledgeable and encouraging study coach for school students.
Analyse the student's quiz performance data and provide personalised feedback.

CRITICAL: Respond ONLY with a valid JSON object in this exact structure:
{
  "weak_areas": ["list of subjects or topics needing improvement"],
  "strong_areas": ["list of subjects or topics the student excels at"],
  "suggestions": [
    "Specific, actionable study tip 1",
    "Specific, actionable study tip 2",
    "Specific, actionable study tip 3"
  ],
  "motivational_note": "A short, encouraging 1-2 sentence message."
}

The input includes a "recommendations" array — a deterministic, already-computed
list of what to focus on next, ranked by priority, each with a "reason" and a
"recommended_action" grounded in real numbers from the database.
- If "recommendations" is non-empty: base "weak_areas" and "suggestions" on
  those entries (their subject/topic/reason/recommended_action), in priority
  order. You may rephrase them in a warmer, more encouraging tone, but do not
  contradict them or change which subjects/topics they point to.
- If "recommendations" is empty: there isn't enough data yet — give general,
  encouraging, non-numeric guidance instead (e.g. "keep practising regularly").

CRITICAL — do not invent numbers: every accuracy percentage, attempt count, or
other statistic you mention MUST come directly from the provided data. Never
fabricate or estimate a number that isn't present in the input. If you don't
have a specific figure for something, describe it qualitatively instead
(e.g. "you've been slipping in Algebra" rather than guessing a percentage).

Be specific and reference the actual subjects/topics from the data.
Keep suggestions practical and achievable for a school student.
"""

    user_prompt = (
        "Here is the student's performance summary, including a "
        "pre-computed, database-driven `recommendations` list ranked by "
        "priority (use it as ground truth — see the system instructions):\n"
        f"{json.dumps(analytics_summary, indent=2, default=str)}\n\n"
        "Please provide personalised feedback and study suggestions."
    )

    try:
        response = await _groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
    except GroqError as exc:
        logger.error("Groq API error during feedback generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI feedback service error: {exc}",
        )

    raw_content = response.choices[0].message.content or "{}"
    try:
        feedback = json.loads(raw_content)
    except json.JSONDecodeError:
        logger.warning("Groq feedback returned non-JSON, using fallback.")
        feedback = {}

    return {
        "weak_areas": feedback.get("weak_areas", []),
        "strong_areas": feedback.get("strong_areas", []),
        "suggestions": feedback.get("suggestions", ["Keep practising regularly!"]),
        "motivational_note": feedback.get(
            "motivational_note", "Great effort! Keep going — consistency is key."
        ),
        "generated_at": datetime.now(UTC).isoformat(),
    }


_FALLBACK_QUOTE = "Every quiz is a step forward — keep going!"


async def generate_motivational_quote(
    *,
    subject: str,
    difficulty: str,
    accuracy: float,
    correct_count: int,
    total_questions: int,
    is_timeout: bool,
) -> str:
    # A single-sentence pep talk for the result of THIS quiz specifically —
    # distinct from generate_feedback()'s motivational_note, which is based
    # on the student's whole historical analytics profile. Deliberately a
    # much smaller/cheaper call (short prompt, small max_tokens) since it
    # only needs one quiz's numbers, not the full analytics payload.
    system_prompt = """You are an encouraging study coach for school students.
Write ONE short, original motivational quote (max 25 words) reacting to a
student's just-finished quiz result.

CRITICAL: Respond ONLY with a valid JSON object in this exact structure:
{"quote": "The quote text here."}

Rules:
- Tone should match the result: celebratory for a strong score, encouraging
  and non-judgemental for a weak one, and understanding (not scolding) if
  the quiz ended by timeout.
- Do not mention exact numbers/percentages — react qualitatively.
- Do not use quotation marks inside the quote text itself.
- Keep it warm, concise, and age-appropriate for a school student.
"""

    result_desc = (
        f"Subject: {subject}\n"
        f"Difficulty: {difficulty}\n"
        f"Score: {correct_count}/{total_questions} correct ({accuracy:.0f}% accuracy)\n"
        f"Ended by: {'timeout' if is_timeout else 'submission'}"
    )
    user_prompt = f"Here is the student's quiz result:\n{result_desc}\n\nWrite the motivational quote now."

    try:
        response = await _groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.9,
            max_tokens=100,
            response_format={"type": "json_object"},
        )
    except GroqError as exc:
        logger.warning("Groq API error during quote generation, using fallback: %s", exc)
        return _FALLBACK_QUOTE

    raw_content = response.choices[0].message.content or "{}"
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError:
        logger.warning("Groq quote response was non-JSON, using fallback.")
        return _FALLBACK_QUOTE

    quote = str(data.get("quote", "")).strip()
    return quote or _FALLBACK_QUOTE

import hashlib
import re


def _normalize(text: str | None) -> str:
    # Lowercase, strip punctuation, collapse whitespace — so two AI-generated
    # renderings of the same question (different capitalization, a trailing
    # "?", extra spaces) still hash to the same fingerprint.
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_question_fingerprint(subject: str, lesson: str | None, question_text: str) -> str:
    # Since every quiz generates fresh Question rows via Groq rather than
    # reusing one, there's no ID to tell "the same question" apart from "a
    # new one" — this hash of (subject, lesson, text) fills that gap, and is
    # what repeated-question analytics groups by. Scoped by subject/lesson on
    # purpose: identical wording in two different subjects should NOT count
    # as the same question. Uses sha256 rather than Python's hash() because
    # hash() is randomly seeded per process, so it'd give a different result
    # after every restart.
    payload = "|".join([_normalize(subject), _normalize(lesson), _normalize(question_text)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

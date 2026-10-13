"""
tests/test_groq_question_validation.py
──────────────────────────────────────
Generated questions are dropped (not patched up) unless they have exactly 4
distinct, non-empty text options that fit the answer column, and a correct
answer that is one of them. Over-long lessons are truncated to fit their
column instead of failing the whole batch.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services import groq_service
from app.services.groq_service import _checked_correct_answer as check


def test_validator_accepts_good_questions_and_normalises_the_answer():
    assert check("Q?", ["A", "B", "C", "D"], "B", index=0, kind="q") == "B"
    assert check("Q?", ["Paris", "Rome", "Oslo", "Bern"], " rome ", index=0, kind="q") == "Rome"


def test_validator_rejects_bad_shapes():
    assert check("Q?", ["A", "a ", "C", "D"], "A", index=0, kind="q") is None  # duplicate (case/space)
    assert check("Q?", ["A", "", "C", "D"], "A", index=0, kind="q") is None  # blank
    assert check("Q?", ["A", 2, "C", "D"], "A", index=0, kind="q") is None  # non-string
    assert check("Q?", ["A", "B", "C"], "A", index=0, kind="q") is None  # wrong count
    assert check("Q?", ["A", "B", "C", "D"], "E", index=0, kind="q") is None  # answer not an option
    assert check("", ["A", "B", "C", "D"], "A", index=0, kind="q") is None  # no question text
    assert check("Q?", ["x" * 513, "B", "C", "D"], "B", index=0, kind="q") is None  # won't fit the column


def _groq_returning(questions: list[dict]):
    message = SimpleNamespace(content=json.dumps({"questions": questions}))
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    return AsyncMock(return_value=response)


async def test_generate_questions_keeps_only_valid_ones_and_caps_lessons():
    good = {"question": "2+2?", "options": ["3", "4", "5", "6"], "correct_answer": "4",
            "explanation": "", "lesson": "L" * 400}
    duplicate = {"question": "Capital?", "options": ["Paris", "paris", "Rome", "Oslo"], "correct_answer": "Paris",
                 "explanation": "", "lesson": "Europe"}
    non_text = {"question": "Odd?", "options": [1, 2, 3, 4], "correct_answer": "1",
                "explanation": "", "lesson": "Numbers"}

    with patch.object(
        groq_service._groq_client.chat.completions, "create", _groq_returning([good, duplicate, non_text])
    ):
        result = await groq_service.generate_questions(
            grade=10, subject="Mathematics", difficulty="easy", question_count=3,
        )

    assert [q["question"] for q in result] == ["2+2?"]
    assert len(result[0]["lesson"]) == 255

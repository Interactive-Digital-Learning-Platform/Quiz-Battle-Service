"""
tests/test_groq_question_validation.py
──────────────────────────────────────
Generated questions are dropped (not patched up) unless they have exactly 4
distinct, non-empty text options that fit the answer column, and a correct
answer that is one of them. Over-long lessons are truncated to fit their
column instead of failing the whole batch. Questions must be answerable from
their text alone -- anything pointing at a graph/diagram/table/image that
can't be shown is dropped on generation and skipped on pool reads.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.models.question import Question
from app.services import battle_question_service, groq_service
from app.services.groq_service import _checked_correct_answer as check
from app.services.groq_service import references_missing_visual


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


# ─────────────────────────────────────────────────────────────────────────────
# Text-only questions (no graph/diagram/table that can't be shown)
# ─────────────────────────────────────────────────────────────────────────────


NEEDS_A_PICTURE = [
    "Based on the bar graph below, which month had the highest rainfall?",
    "Refer to the following table. What is the total?",
    "In the diagram, what does label X represent?",
    "What is shown above?",
    "According to Figure 2, which gas is produced?",
    "Using the given pie chart, find the largest sector.",
    "Look at this picture. What is the animal doing?",
    "The diagram shows a food web. Which organism is a producer?",
    # Real pool row: the class labels only exist in the missing graph.
    "A bar graph shows the number of books read by students: 10, 15, 12, 8. Which class read the most books?",
    "A pie chart shows how a family spends its income. Which item takes the largest share?",
    "The table shows the rainfall for four months. Which month was the wettest?",
]
TEXT_ONLY = [
    "Which type of graph is best for showing parts of a whole?",
    "Which element is in Group 1 of the periodic table?",
    "What happens to the water table during a drought?",
    "What is 7 x 8 in the 7 times table?",
    "A shop sold 12, 15 and 9 books on three days. What is the mean?",
    "What is the output of the truth table for an AND gate when both inputs are 1?",
    "Which of the following is a prime number?",
    "A bar chart is used to compare what?",
    "Why do scientists draw a diagram of an experiment?",
    # Self-contained word problems that merely name a graph/map/figure (real pool rows).
    "The graph of y = log2(x - 3) is obtained from y = log2 x by which transformation?",
    "A miniature map shows a circular pond with a radius of 2 cm. The map scale is 1 cm : 5 m. What is the actual perimeter?",
    "On a map scale of 1:250,000, how many centimeters on the map represent 5 kilometres on the ground?",
    "If a map shows north at the top, which direction does the right side of the map correspond to?",
    "A square of side 4 cm has a triangle attached along one side. What is the perimeter of the combined figure?",
    "How does the perimeter of a figure change when the figure is enlarged by a scale factor of 2?",
    "A scale drawing shows a park with a perimeter of 5 cm. The drawing scale is 1 cm : 100 m. What is the real perimeter?",
    "If a student wants to create a table of 5 rows and 3 columns in a document, which menu should they use?",
    "In a relational database, which key uniquely identifies each record in a table?",
    "Class A read 10 books, Class B 15, Class C 12 and Class D 8. Which class read the most books?",
]


@pytest.mark.parametrize("text", NEEDS_A_PICTURE)
def test_questions_pointing_at_a_missing_visual_are_detected(text):
    assert references_missing_visual(text)
    assert check(text, ["A", "B", "C", "D"], "A", index=0, kind="q") is None


@pytest.mark.parametrize("text", TEXT_ONLY)
def test_text_only_questions_about_graphs_or_tables_are_kept(text):
    assert not references_missing_visual(text)
    assert check(text, ["A", "B", "C", "D"], "A", index=0, kind="q") == "A"


async def test_pool_reads_skip_existing_picture_questions(db_session):
    rows = [
        Question(question=text, options=["A", "B", "C", "D"], correct_answer="A",
                 subject="Mathematics", lesson="Statistics", difficulty="medium")
        for text in (NEEDS_A_PICTURE[0], NEEDS_A_PICTURE[1], TEXT_ONLY[4])
    ]
    db_session.add_all(rows)
    await db_session.commit()

    picked = await battle_question_service._fetch_battle_pool_questions(
        db_session, subject="Mathematics", difficulty="medium", limit=3,
        exclude_ids=set(), history_texts=[], session_texts=[],
    )

    assert [q.question for q in picked] == [TEXT_ONLY[4]]

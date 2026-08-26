"""
app/services/elo_engine.py
────────────────────────────
Elo rating math -- pure, deterministic, no DB access, no async, no knowledge
of CompetitiveSubjectRating or any battle model. A later phase's real
battle-completion flow calls compute_elo_update() and persists the result
itself; this module never touches persistence, so it stays reusable
untouched once that flow exists.
"""
from dataclasses import dataclass

from app.core.config import settings

# Valid `result_a` values, from player A's perspective.
RESULT_WIN = 1.0
RESULT_DRAW = 0.5
RESULT_LOSS = 0.0
VALID_RESULTS = (RESULT_WIN, RESULT_DRAW, RESULT_LOSS)


def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard logistic Elo expectation for player A vs. player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


@dataclass(frozen=True)
class EloUpdateResult:
    new_rating_a: int
    new_rating_b: int
    delta_a: int
    delta_b: int
    expected_a: float
    expected_b: float


def compute_elo_update(
    rating_a: int,
    rating_b: int,
    result_a: float,
    *,
    k_factor: float | None = None,
    min_rating: int | None = None,
) -> EloUpdateResult:
    """Compute both players' new ratings for a single match.

    `result_a` is 1.0 (A won), 0.5 (draw), or 0.0 (A lost) -- B's result is
    always `1 - result_a`. `k_factor`/`min_rating` default to the
    BATTLE_ELO_K_FACTOR/BATTLE_MIN_RATING settings; the override params exist
    so a later phase can experiment without changing this function's
    signature.
    """
    if result_a not in VALID_RESULTS:
        raise ValueError(f"result_a must be one of {VALID_RESULTS}, got {result_a!r}")

    k = settings.BATTLE_ELO_K_FACTOR if k_factor is None else k_factor
    floor = settings.BATTLE_MIN_RATING if min_rating is None else min_rating

    expected_a = expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    result_b = 1.0 - result_a

    new_rating_a = max(floor, round(rating_a + k * (result_a - expected_a)))
    new_rating_b = max(floor, round(rating_b + k * (result_b - expected_b)))

    return EloUpdateResult(
        new_rating_a=new_rating_a,
        new_rating_b=new_rating_b,
        delta_a=new_rating_a - rating_a,
        delta_b=new_rating_b - rating_b,
        expected_a=expected_a,
        expected_b=expected_b,
    )

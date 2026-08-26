"""
tests/test_battle_difficulty.py
───────────────────────────────────
Unit tests for the pure mixed-difficulty distribution functions in
app/services/battle_difficulty.py -- no DB, no async.
"""
from app.services.battle_difficulty import (
    BATTLE_DIFFICULTY_DISTRIBUTION,
    allocate_question_counts,
    get_difficulty_distribution,
    get_difficulty_label,
)


def test_below_1200_is_mostly_easy():
    assert get_difficulty_label(0) == "Mostly Easy"
    assert get_difficulty_label(1199) == "Mostly Easy"


def test_1200_to_1499_is_easy_plus_medium():
    assert get_difficulty_label(1200) == "Easy + Medium"
    assert get_difficulty_label(1499) == "Easy + Medium"


def test_1500_to_1799_is_mostly_medium():
    assert get_difficulty_label(1500) == "Mostly Medium"
    assert get_difficulty_label(1799) == "Mostly Medium"


def test_1800_to_2099_is_medium_plus_hard():
    assert get_difficulty_label(1800) == "Medium + Hard"
    assert get_difficulty_label(2099) == "Medium + Hard"


def test_2100_and_above_is_mostly_hard():
    assert get_difficulty_label(2100) == "Mostly Hard"
    assert get_difficulty_label(9999) == "Mostly Hard"


def test_accepts_non_integer_average():
    assert get_difficulty_label(1499.5) == "Easy + Medium"
    assert get_difficulty_label(1500.5) == "Mostly Medium"


# ─────────────────────────────────────────────────────────────────────────────
# No band is ever 100% one difficulty -- the whole point of mixing
# ─────────────────────────────────────────────────────────────────────────────

def test_every_band_blends_at_least_two_difficulties():
    for label, distribution in BATTLE_DIFFICULTY_DISTRIBUTION.items():
        assert sum(distribution.values()) == 1.0
        nonzero_tiers = [tier for tier, pct in distribution.items() if pct > 0]
        assert len(nonzero_tiers) >= 2, f"{label} is not mixed: {distribution}"


def test_no_band_boundary_is_a_full_flip():
    """Crossing a single rating point at a boundary must never swing any
    one tier's share by more than 50 percentage points (a full 0%<->100%
    flip, the "sharp jump" this system exists to avoid).
    """
    boundaries = [1200, 1500, 1800, 2100]
    for boundary in boundaries:
        below = get_difficulty_distribution(boundary - 1)
        above = get_difficulty_distribution(boundary)
        for tier in ("easy", "medium", "hard"):
            assert abs(above[tier] - below[tier]) < 0.5


def test_get_difficulty_distribution_matches_label_lookup():
    assert get_difficulty_distribution(1000) == BATTLE_DIFFICULTY_DISTRIBUTION["Mostly Easy"]


# ─────────────────────────────────────────────────────────────────────────────
# allocate_question_counts -- largest-remainder rounding
# ─────────────────────────────────────────────────────────────────────────────

def test_allocate_sums_to_total_for_every_band():
    for distribution in BATTLE_DIFFICULTY_DISTRIBUTION.values():
        counts = allocate_question_counts(distribution, 10)
        assert sum(counts.values()) == 10


def test_allocate_easy_plus_medium_has_no_hard_at_ten_questions():
    counts = allocate_question_counts(BATTLE_DIFFICULTY_DISTRIBUTION["Easy + Medium"], 10)
    assert counts == {"easy": 5, "medium": 5}


def test_allocate_mostly_easy_at_ten_questions():
    counts = allocate_question_counts(BATTLE_DIFFICULTY_DISTRIBUTION["Mostly Easy"], 10)
    assert counts == {"easy": 7, "medium": 3}


def test_allocate_zero_total_is_empty():
    assert allocate_question_counts(BATTLE_DIFFICULTY_DISTRIBUTION["Mostly Hard"], 0) == {}


def test_allocate_drops_zero_count_tiers():
    counts = allocate_question_counts({"easy": 1.0, "medium": 0.0, "hard": 0.0}, 10)
    assert counts == {"easy": 10}

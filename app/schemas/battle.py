from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class LeaderboardEntry(BaseModel):
    rank: int
    user_id: int
    username: str | None
    subject: str
    rating: int
    league: str
    wins: int
    losses: int
    draws: int
    matches_played: int
    average_response_time: float


class LeaderboardResponse(BaseModel):
    subject: str
    page: int
    page_size: int
    total_count: int
    total_pages: int
    entries: list[LeaderboardEntry]
    # Present only when the caller has a rating row for this subject --
    # None if they haven't played it yet.
    my_rank: int | None
    my_rating: int | None


class CompetitiveSubjectProfile(BaseModel):
    subject: str
    rating: int
    highest_rating: int
    league: str
    rank: int
    wins: int
    losses: int
    draws: int
    matches_played: int
    current_win_streak: int
    highest_win_streak: int
    correct_answers: int
    total_answers: int
    accuracy: float
    average_response_time: float
    created_at: datetime
    updated_at: datetime


class CompetitiveProfileResponse(BaseModel):
    user_id: int
    # One entry per subject the user has a CompetitiveSubjectRating row for
    # -- empty for a brand-new user who hasn't played a ranked battle yet.
    subjects: list[CompetitiveSubjectProfile]


class JoinQueueRequest(BaseModel):
    subject: str = Field(..., min_length=1)


class OpponentPublicInfo(BaseModel):
    # Rank-safe public info only -- no sensitive user data.
    user_id: int
    username: str | None
    rating: int
    league: str


class QueueStatusResponse(BaseModel):
    # "idle" | "searching" | "matched" -- join/cancel/status all return this
    # same shape, with fields populated according to status.
    status: str
    subject: str | None = None
    waited_seconds: float | None = None
    current_rating_window: int | None = None
    match_id: int | None = None
    difficulty: str | None = None
    opponent: OpponentPublicInfo | None = None


class BattleQuestionPublic(BaseModel):
    # NEVER includes correct_answer -- spec's "Do Not Send Correct Answers Early".
    question_id: int
    question: str
    options: list[Any] | None = None


class BattleParticipantProgress(BaseModel):
    user_id: int
    answered_count: int


class BattleParticipantResult(BaseModel):
    user_id: int
    correct_count: int | None
    wrong_count: int | None
    final_score: int | None
    best_streak: int | None
    total_response_time_ms: int | None
    rating_before: int
    rating_after: int | None
    rating_delta: int | None
    # Derived from rating_after via league_engine.get_league() -- None
    # whenever rating_after is (match not yet completed).
    league: str | None
    result: str | None


class BattleMatchResult(BaseModel):
    winner_user_id: int | None
    me: BattleParticipantResult
    opponent: BattleParticipantResult | None


class BattleAnswerProgress(BaseModel):
    # One question slot's outcome for a participant's progress bar -- drives
    # BOTH players' bars (own answers AND, live over WS via
    # OpponentAnsweredEvent / on load via this same shape, the opponent's).
    # Absence of an entry for a question_order that's already in the past
    # means that participant never answered it before its window closed.
    question_order: int
    is_correct: bool


class BattleMatchStateResponse(BaseModel):
    # "waiting" | "countdown" | "active" | "completed" | "cancelled" --
    # shape varies by status, same polymorphic-response pattern as
    # QueueStatusResponse.
    status: str
    match_id: int
    subject: str | None = None
    difficulty: str | None = None
    question_count: int | None = None
    # Seconds remaining in the CURRENT question's fixed window (lockstep
    # play -- see battle_gameplay_service._question_time_remaining_seconds).
    time_remaining_seconds: float | None = None
    countdown_seconds_remaining: float | None = None
    # The single live question both players are currently on, derived from
    # elapsed time (see battle_gameplay_service._current_question_order).
    question_index: int | None = None
    question: BattleQuestionPublic | None = None
    has_answered_current_question: bool | None = None
    # Every question slot answered so far by each side, for progress-bar
    # hydration on initial load/reconnect (kept live afterward via
    # answer_acknowledged/opponent_answered WS events).
    my_answers: list[BattleAnswerProgress] | None = None
    opponent_answers: list[BattleAnswerProgress] | None = None
    my_progress: BattleParticipantProgress | None = None
    opponent_progress: BattleParticipantProgress | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    reason: str | None = None
    result: BattleMatchResult | None = None


class BattleHistoryEntry(BaseModel):
    match_id: int
    subject: str
    difficulty: str
    started_at: datetime | None
    finished_at: datetime | None
    winner_user_id: int | None
    opponent_user_id: int | None
    opponent_username: str | None
    me: BattleParticipantResult
    opponent: BattleParticipantResult | None


class BattleHistoryResponse(BaseModel):
    page: int
    page_size: int
    total_count: int
    total_pages: int
    entries: list[BattleHistoryEntry]


class SubmitBattleAnswerRequest(BaseModel):
    question_id: int
    selected_option: str = Field(..., min_length=1)


class SubmitBattleAnswerResponse(BaseModel):
    # Included so the client can correlate this ack with the specific
    # question it answered -- needed now that answers can be submitted for
    # any question in any order (self-paced play), not just "the current one".
    question_id: int
    is_correct: bool
    base_score: int
    speed_bonus: int
    streak_bonus: int
    total_question_score: int
    response_time_ms: int

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
    # Left None for "me" everywhere (the client already knows its own name
    # locally) -- only ever populated for the other participant, so the
    # client can show a real name instead of a generic "Opponent" label.
    # See battle_gameplay_service.participant_result_payload.
    username: str | None = None
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
    # BOTH players' bars (own answers AND, live over WS via the
    # question_result event / on load via this same shape, the opponent's).
    # Every GRADED question slot gets an entry now (deferred-grading design
    # -- see battle_gameplay_service._grade_one_participant), including ones
    # this participant never actually submitted anything for -- `answered`
    # is what distinguishes "answered wrong" (is_correct=False, answered=True)
    # from "never answered at all" (is_correct=False, answered=False) for the
    # progress bar's yellow-vs-red segment coloring. Absence of an entry
    # entirely just means this question slot hasn't been graded/reached yet.
    question_order: int
    is_correct: bool
    answered: bool


class ParticipantQuestionResult(BaseModel):
    # One participant's graded outcome for one question, revealed to BOTH
    # players simultaneously at the reveal boundary -- see
    # battle_gameplay_service._grade_and_reveal_question. Deliberately never
    # includes selected_option -- preserves the existing "never leak the
    # specific answer to the opponent" rule; only correctness is shared.
    user_id: int
    answered: bool
    is_correct: bool
    base_score: int
    speed_bonus: int
    streak_bonus: int
    total_question_score: int


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
    # hydration on initial load/reconnect (kept live afterward via the
    # question_result WS event once each question locks).
    my_answers: list[BattleAnswerProgress] | None = None
    opponent_answers: list[BattleAnswerProgress] | None = None
    # Populated only once the CURRENT question has locked/been graded (see
    # battle_gameplay_service._grade_and_reveal_question) -- None means it's
    # still answerable. Lets a reconnecting client immediately show the
    # locked/revealed state instead of a live answering UI for a question
    # that already finished grading while it was disconnected.
    current_question_results: list[ParticipantQuestionResult] | None = None
    # Paired with current_question_results -- populated under the same
    # condition (question already locked/graded), None while still
    # answerable. See BattleQuestionPublic for why `question` itself never
    # carries this.
    current_question_correct_answer: str | None = None
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
    # Deliberately minimal -- grading is deferred to the reveal boundary
    # (see battle_gameplay_service._grade_and_reveal_question /
    # QuestionResultEvent below), so submitting an answer no longer reveals
    # correctness. This is just an ack that the (re)submission was accepted;
    # question_id lets the client correlate it since answers can be changed
    # freely (resubmitted) any number of times before the question locks.
    question_id: int
    selected_option: str

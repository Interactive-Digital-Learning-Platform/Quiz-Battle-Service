from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.core.config import settings


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
    # 100 = the subject column's length (battle_matches / competitive_subject_ratings).
    subject: str = Field(..., min_length=1, max_length=100)

    @field_validator("subject")
    @classmethod
    def _allowed_subject(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("subject must not be blank")
        allowed = settings.battle_allowed_subjects
        if allowed and value not in allowed:
            raise ValueError(f"Unknown battle subject. Choose one of: {', '.join(sorted(allowed))}")
        return value


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
    # "matched" only: this player has finished every question and is waiting
    # for the opponent to finish (or the match clock to run out).
    awaiting_opponent: bool = False


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
    # "forfeit" | "all_finished" | "time_up" -- see
    # battle_gameplay_service.match_end_reason.
    end_reason: str | None = None
    me: BattleParticipantResult
    opponent: BattleParticipantResult | None


class BattleAnswerProgress(BaseModel):
    # One question a participant has moved past, for the progress bars --
    # `answered` is False for a skipped question. Correctness is never
    # included while the match is live (no right/wrong feedback until the
    # end), so is_correct is always None here.
    question_order: int
    is_correct: bool | None = None
    answered: bool


class BattleQuestionReview(BaseModel):
    # One row of the end-of-match review -- the viewer's OWN answers only.
    # status: "correct" | "incorrect" | "skipped" | "unanswered" (never
    # reached before the match clock ran out).
    question_order: int
    question: str
    options: list[Any] | None = None
    correct_answer: str | None
    selected_option: str | None
    status: str
    response_time_ms: int | None
    base_score: int
    speed_bonus: int
    streak_bonus: int = 0
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
    # Seconds left on the whole-match clock.
    time_remaining_seconds: float | None = None
    countdown_seconds_remaining: float | None = None
    # The viewer's OWN current question (self-paced play -- the opponent may
    # be on a different one). question_index == question_count with
    # question=None and finished=True once they've moved past every question.
    question_index: int | None = None
    question: BattleQuestionPublic | None = None
    finished: bool | None = None
    # When the match clock runs out, plus the server's clock at the moment
    # this was built -- clients count down to ends_at using their offset
    # from server_time rather than trusting their own clock.
    ends_at: datetime | None = None
    server_time: datetime | None = None
    # Every question each side has moved past so far, for progress-bar
    # hydration on load/reconnect (kept live afterward via the
    # player_progress WS event).
    my_answers: list[BattleAnswerProgress] | None = None
    opponent_answers: list[BattleAnswerProgress] | None = None
    my_progress: BattleParticipantProgress | None = None
    opponent_progress: BattleParticipantProgress | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    reason: str | None = None
    result: BattleMatchResult | None = None
    # "completed" only: the viewer's own per-question review.
    review: list[BattleQuestionReview] | None = None


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
    # 512 = battle_answers.selected_option's length.
    selected_option: str = Field(..., min_length=1, max_length=512)


class SkipBattleQuestionRequest(BaseModel):
    question_id: int


class BattleNextQuestionResponse(BaseModel):
    # Returned once an answer/skip is saved (or re-sent unchanged for a
    # duplicate Next/Skip): the player's next question, or finished=True with
    # question=None once they've moved past the last one. Never says whether
    # the answer was right -- that waits for the end-of-match review.
    question_index: int
    question: BattleQuestionPublic | None
    finished: bool
    ends_at: datetime
    server_time: datetime


# ── GET /battle/analytics ────────────────────────────────────────────────────
# See app/services/battle_analytics_service.py for how each figure is derived.


class BattleRatingPoint(BaseModel):
    match_id: int
    finished_at: datetime | None
    rating: int  # rating_after for this match
    rating_change: int | None
    result: str | None


class BattleRatingHistory(BaseModel):
    subject: str
    # Oldest -> newest, at most BATTLE_ANALYTICS_RATING_HISTORY_LIMIT points.
    points: list[BattleRatingPoint]


class BattleSpeedTier(BaseModel):
    # "fast" | "quick" | "steady" | "slow" -- the speed-bonus tiers, fastest first.
    tier: str
    max_seconds: float | None  # inclusive upper bound; None for the open-ended last tier
    answered: int
    correct: int


class BattlePointsBreakdown(BaseModel):
    base: int
    speed_bonus: int
    streak_bonus: int


class BattleTopicStat(BaseModel):
    subject: str
    lesson: str
    answered: int
    correct: int
    accuracy: float


class BattleStrongerOpponents(BaseModel):
    # Matches against an opponent rated higher than the user going in.
    played: int
    wins: int


class BattleQuestionUsage(BaseModel):
    # Every question dealt to the user in completed matches, by what happened to it.
    questions: int
    answered: int
    skipped: int
    not_reached: int  # the clock ran out (or someone forfeited) before they got to it


class BattleMatchEndings(BaseModel):
    # How the user's matches ended (see battle_gameplay_service.match_end_reason);
    # matches from before self-paced play aren't counted.
    all_finished: int
    time_up: int
    forfeit: int
    # Matches where the user got through every question, and their average
    # time to do so (None if they never have).
    finished_all_questions: int
    average_finish_seconds: float | None


class BattleAnalyticsResponse(BaseModel):
    matches_played: int
    wins: int
    losses: int
    draws: int
    forfeits: int
    win_rate: float
    average_score: float
    answered: int
    correct: int
    accuracy: float
    # Oldest -> newest, at most BATTLE_ANALYTICS_RECENT_FORM_LIMIT results.
    recent_form: list[str]
    # One entry per subject played, most-played subject first.
    rating_history: list[BattleRatingHistory]
    speed_profile: list[BattleSpeedTier]
    points: BattlePointsBreakdown
    # Lessons with at least BATTLE_ANALYTICS_TOPIC_MIN_ANSWERS answers, strongest first.
    topics: list[BattleTopicStat]
    vs_stronger: BattleStrongerOpponents
    question_usage: BattleQuestionUsage
    match_endings: BattleMatchEndings

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from app.schemas.battle import (
    BattleMatchStateResponse,
    BattleParticipantResult,
    BattleQuestionPublic,
    SubmitBattleAnswerResponse,
)


class WSReadyMessage(BaseModel):
    type: Literal["ready"] = "ready"


class WSAnswerMessage(BaseModel):
    type: Literal["answer"] = "answer"
    question_id: int
    selected_option: str = Field(..., min_length=1)


class WSForfeitMessage(BaseModel):
    type: Literal["forfeit"] = "forfeit"


InboundMessage = Annotated[
    WSReadyMessage | WSAnswerMessage | WSForfeitMessage, Field(discriminator="type")
]

_inbound_adapter: TypeAdapter[InboundMessage] = TypeAdapter(InboundMessage)


def parse_inbound_message(raw: dict) -> InboundMessage:
    """Discriminates on the `type` field. Raises pydantic.ValidationError
    for malformed/unknown messages -- callers turn that into a WS `error`
    event rather than dropping the connection.
    """
    return _inbound_adapter.validate_python(raw)


# ── Outbound events ─────────────────────────────────────────────────────
# One BaseModel per event name, each dumped via `.model_dump(mode="json")`
# before `websocket.send_json(...)`. Reuses existing REST-layer schemas
# (BattleMatchStateResponse/BattleMatchResult/etc.) where the shape is
# already identical, rather than redefining fields.


class ConnectedEvent(BaseModel):
    type: Literal["connected"] = "connected"
    match_id: int


class StateSyncEvent(BaseModel):
    type: Literal["state_sync"] = "state_sync"
    state: BattleMatchStateResponse


class PlayerReadyEvent(BaseModel):
    type: Literal["player_ready"] = "player_ready"
    user_id: int


class CountdownStartedEvent(BaseModel):
    type: Literal["countdown_started"] = "countdown_started"
    started_at: datetime
    duration_seconds: float


class MatchStartedEvent(BaseModel):
    type: Literal["match_started"] = "match_started"
    started_at: datetime


class QuestionStartedEvent(BaseModel):
    # Pushed once per question boundary (including the very first, right
    # after MatchStartedEvent) -- lockstep play, both players receive the
    # SAME question_index/question at the SAME wall-clock instant, derived
    # from battle_gameplay_service._current_question_order.
    type: Literal["question_started"] = "question_started"
    question_index: int
    question: BattleQuestionPublic
    time_remaining_seconds: float


class AnswerAcknowledgedEvent(BaseModel):
    type: Literal["answer_acknowledged"] = "answer_acknowledged"
    # Correlates this ack with the specific question it answered — needed
    # now that answers can be submitted for any question in any order
    # (self-paced play), used by the client to clear its per-question
    # in-flight submit guard.
    question_id: int
    is_correct: bool
    base_score: int
    speed_bonus: int
    streak_bonus: int
    total_question_score: int
    response_time_ms: int

    @classmethod
    def from_response(cls, response: SubmitBattleAnswerResponse) -> "AnswerAcknowledgedEvent":
        return cls(**response.model_dump())


class OpponentAnsweredEvent(BaseModel):
    type: Literal["opponent_answered"] = "opponent_answered"
    user_id: int
    answered_count: int
    # Which question slot this was for and whether it was correct -- drives
    # the opponent's live progress-bar segment on the viewer's screen. Never
    # includes selected_option (spec's "don't leak correct_answer" still
    # holds; is_correct alone doesn't reveal which option was right).
    question_order: int
    is_correct: bool


class PlayerDisconnectedEvent(BaseModel):
    type: Literal["player_disconnected"] = "player_disconnected"
    user_id: int
    deadline: datetime
    grace_period_seconds: float


class PlayerReconnectedEvent(BaseModel):
    type: Literal["player_reconnected"] = "player_reconnected"
    user_id: int


class MatchFinishedEvent(BaseModel):
    # Broadcast to BOTH players over the same pub/sub channel, so the shape
    # is receiver-agnostic (a flat participants list, not a {me, opponent}
    # split -- BattleMatchResult below is for the REST-facing, per-viewer
    # /state response instead, where "me" genuinely means the requester).
    # Mobile (types/battleWsTypes.ts) already mirrors this real wire shape.
    type: Literal["match_finished"] = "match_finished"
    winner_user_id: int | None = None
    participants: list[BattleParticipantResult]


class MatchCancelledEvent(BaseModel):
    type: Literal["match_cancelled"] = "match_cancelled"
    reason: str | None = None


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    detail: str

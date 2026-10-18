from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from app.schemas.battle import (
    BattleMatchStateResponse,
    BattleParticipantResult,
    BattleQuestionPublic,
)


class WSReadyMessage(BaseModel):
    type: Literal["ready"] = "ready"


class WSAnswerMessage(BaseModel):
    # "Next": the player's final pick for their current question.
    type: Literal["answer"] = "answer"
    question_id: int
    selected_option: str = Field(..., min_length=1, max_length=512)  # see SubmitBattleAnswerRequest


class WSSkipMessage(BaseModel):
    type: Literal["skip"] = "skip"
    question_id: int


class WSPongMessage(BaseModel):
    # Reply to a server PingEvent -- the round trip feeds the speed bonus's
    # network allowance (see BATTLE_NETWORK_ALLOWANCE_MAX_SECONDS).
    type: Literal["pong"] = "pong"
    nonce: int


class WSForfeitMessage(BaseModel):
    type: Literal["forfeit"] = "forfeit"


InboundMessage = Annotated[
    WSReadyMessage | WSAnswerMessage | WSSkipMessage | WSPongMessage | WSForfeitMessage,
    Field(discriminator="type"),
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
    # Carries subject/difficulty/question_count directly rather than making
    # clients backfill them via a separate REST fetch -- that fetch raced
    # this exact transition (it could resolve while the match was still
    # "countdown" server-side, which never had question_count to return),
    # leaving question_count permanently unset for the rest of the match on
    # fast connections. Every connected client receives this event exactly
    # once, right as status flips to "active", regardless of when it joined.
    type: Literal["match_started"] = "match_started"
    started_at: datetime
    subject: str
    difficulty: str
    question_count: int
    ends_at: datetime
    server_time: datetime


class NextQuestionEvent(BaseModel):
    # Sent ONLY to the player it's for (self-paced play): question 0 to every
    # socket when the match starts, then each player's next question right
    # after their answer/skip is saved -- never before. A duplicate Next/Skip
    # gets the same event again, unchanged. finished=True (question=None)
    # once they've moved past the last question. Never says whether the
    # answer was right.
    type: Literal["next_question"] = "next_question"
    question_index: int
    question: BattleQuestionPublic | None
    finished: bool
    ends_at: datetime
    server_time: datetime


class PlayerProgressEvent(BaseModel):
    # Broadcast after each saved answer/skip so the opponent's progress bar
    # moves -- how far they are, never whether they were right.
    type: Literal["player_progress"] = "player_progress"
    user_id: int
    question_order: int
    answered: bool  # False = skipped
    answered_count: int


class PingEvent(BaseModel):
    # The client answers with WSPongMessage(nonce) straight away.
    type: Literal["ping"] = "ping"
    nonce: int


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
    # "forfeit" | "all_finished" | "time_up" -- see
    # battle_gameplay_service.match_end_reason.
    end_reason: str | None = None
    participants: list[BattleParticipantResult]


class MatchCancelledEvent(BaseModel):
    type: Literal["match_cancelled"] = "match_cancelled"
    reason: str | None = None


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    detail: str

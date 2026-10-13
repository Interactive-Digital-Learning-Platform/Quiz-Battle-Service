import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.redis_client import get_redis
from app.core.security import get_current_user
from app.schemas.battle import (
    BattleAnalyticsResponse,
    BattleHistoryResponse,
    BattleMatchStateResponse,
    CompetitiveProfileResponse,
    JoinQueueRequest,
    LeaderboardResponse,
    QueueStatusResponse,
    SubmitBattleAnswerRequest,
    SubmitBattleAnswerResponse,
)
from app.services import battle_analytics_service, battle_gameplay_service, battle_ws_events, matchmaking_service
from app.services.battle_connection_manager import manager as ws_manager
from app.services.battle_service import get_competitive_profile, get_leaderboard
from app.services.user_service import get_or_create_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/battle", tags=["Battle"])


@router.get(
    "/leaderboard",
    response_model=LeaderboardResponse,
    status_code=status.HTTP_200_OK,
    summary="Subject leaderboard",
    description=(
        "Paginated 1v1 competitive leaderboard for a single subject, sorted "
        "by rating (ties broken by wins, then average response time). Also "
        "returns the caller's own rank/rating for this subject even when "
        "they're outside the current page, or null if they haven't played "
        "a ranked battle in this subject yet."
    ),
)
async def get_battle_leaderboard(
    subject: str = Query(..., min_length=1),
    page: int = Query(1, ge=1),
    page_size: int = Query(
        settings.BATTLE_LEADERBOARD_DEFAULT_PAGE_SIZE,
        ge=1,
        le=settings.BATTLE_LEADERBOARD_MAX_PAGE_SIZE,
    ),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> LeaderboardResponse:
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )

    logger.info(
        "GET /battle/leaderboard — clerk_id=%s, subject=%s, page=%d", clerk_id, subject, page
    )

    user = await get_or_create_user(db, clerk_id)
    data = await get_leaderboard(
        db, subject=subject, page=page, page_size=page_size, requesting_user_id=user.id
    )
    return LeaderboardResponse(**data)


@router.get(
    "/profile",
    response_model=CompetitiveProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="My competitive profile",
    description=(
        "Returns the authenticated user's per-subject competitive ratings "
        "(rating, league, rank, win/loss record). Empty subjects list for a "
        "user who hasn't played a ranked battle yet."
    ),
)
async def get_my_battle_profile(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> CompetitiveProfileResponse:
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )

    logger.info("GET /battle/profile — clerk_id=%s", clerk_id)

    user = await get_or_create_user(db, clerk_id)
    subjects = await get_competitive_profile(db, user_id=user.id)
    return CompetitiveProfileResponse(user_id=user.id, subjects=subjects)


@router.get(
    "/analytics",
    response_model=BattleAnalyticsResponse,
    status_code=status.HTTP_200_OK,
    summary="My battle analytics",
    description=(
        "Aggregated stats over the authenticated user's completed battles: "
        "record and recent form, per-subject rating history, answer speed "
        "profile, points breakdown, strongest/weakest lessons and results "
        "against higher-rated opponents. All-zero / empty lists for a user "
        "who hasn't finished a battle yet."
    ),
)
async def get_my_battle_analytics(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> BattleAnalyticsResponse:
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )

    logger.info("GET /battle/analytics — clerk_id=%s", clerk_id)

    user = await get_or_create_user(db, clerk_id)
    data = await battle_analytics_service.get_battle_analytics(db, user_id=user.id)
    return BattleAnalyticsResponse(**data)


@router.get(
    "/history",
    response_model=BattleHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="My battle match history",
    description=(
        "Paginated list of the authenticated user's completed 1v1 battles, "
        "newest-finished first. Optionally filtered to one subject. Matches "
        "cancelled before completion (e.g. AI question-generation failure) "
        "are excluded -- they never had rating changes applied."
    ),
)
async def get_my_battle_history(
    subject: str | None = Query(None, min_length=1),
    page: int = Query(1, ge=1),
    page_size: int = Query(
        settings.BATTLE_LEADERBOARD_DEFAULT_PAGE_SIZE,
        ge=1,
        le=settings.BATTLE_LEADERBOARD_MAX_PAGE_SIZE,
    ),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> BattleHistoryResponse:
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )

    logger.info("GET /battle/history — clerk_id=%s, subject=%s, page=%d", clerk_id, subject, page)

    user = await get_or_create_user(db, clerk_id)
    data = await battle_gameplay_service.get_battle_history(
        db, user_id=user.id, subject=subject, page=page, page_size=page_size
    )
    return BattleHistoryResponse(**data)


@router.post(
    "/queue/join",
    response_model=QueueStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Join the matchmaking queue",
    description=(
        "Joins the 1v1 ranked matchmaking queue for a subject. Lazily "
        "creates the caller's competitive rating for that subject at the "
        "initial rating if this is their first time queuing for it. "
        "Attempts an immediate match; if none is found, a background sweep "
        "retries with a widening rating window. 409 if already queued or "
        "already in an active battle."
    ),
)
async def join_battle_queue(
    payload: JoinQueueRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
) -> QueueStatusResponse:
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )

    logger.info("POST /battle/queue/join — clerk_id=%s, subject=%s", clerk_id, payload.subject)

    user = await get_or_create_user(db, clerk_id)
    data = await matchmaking_service.join_queue(db, redis, user_id=user.id, subject=payload.subject)
    return QueueStatusResponse(**data)


@router.post(
    "/queue/cancel",
    response_model=QueueStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Cancel matchmaking",
    description=(
        "Leaves the matchmaking queue. A no-op (200, status=idle) if the "
        "caller wasn't queued. 409 if a match is already forming or active "
        "-- that can't be cancelled through this endpoint."
    ),
)
async def cancel_battle_queue(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
) -> QueueStatusResponse:
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )

    logger.info("POST /battle/queue/cancel — clerk_id=%s", clerk_id)

    user = await get_or_create_user(db, clerk_id)
    data = await matchmaking_service.cancel_queue(db, redis, user_id=user.id)
    return QueueStatusResponse(**data)


@router.get(
    "/queue/status",
    response_model=QueueStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Poll matchmaking status",
    description=(
        "Polling endpoint for the caller's current matchmaking state -- "
        "idle, searching (with elapsed wait/current rating window), or "
        "matched (with opponent's public info). This is a deliberate "
        "interim mechanism; a later phase upgrades this to WebSocket push."
    ),
)
async def get_battle_queue_status(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
) -> QueueStatusResponse:
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )

    user = await get_or_create_user(db, clerk_id)
    data = await matchmaking_service.get_queue_status(db, redis, user_id=user.id)
    return QueueStatusResponse(**data)


@router.get(
    "/match/{match_id}/state",
    response_model=BattleMatchStateResponse,
    status_code=status.HTTP_200_OK,
    summary="Poll battle match state",
    description=(
        "Polling endpoint for a battle match's current state -- once "
        "active, returns the single CURRENT question both players are on "
        "(never its correct answer), whether the caller has answered it "
        "yet, the time remaining in its window, and each side's per-"
        "question answer progress (for the progress bars), or the full "
        "result once completed, or a cancellation reason if AI question "
        "generation failed. This is a deliberate interim mechanism; a "
        "later phase upgrades this to WebSocket push."
    ),
)
async def get_battle_match_state(
    match_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
) -> BattleMatchStateResponse:
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )

    user = await get_or_create_user(db, clerk_id)
    data = await battle_gameplay_service.get_match_state(db, match_id=match_id, user_id=user.id, redis=redis)
    return BattleMatchStateResponse(**data)


@router.post(
    "/match/{match_id}/answer",
    response_model=SubmitBattleAnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a battle answer",
    description=(
        "Submits an answer to the CURRENT question only (lockstep play -- "
        "both players are always on the same question at the same time, "
        "derived from elapsed time since the match started). "
        "Server-authoritative: correctness, response time (measured from "
        "when this question's fixed window opened), and scoring "
        "(base + speed bonus + streak bonus) are all computed server-side, "
        "never trusted from the client. 400 if the question isn't the "
        "current one; 409 if already answered (one answer per player per "
        "question, DB-constraint enforced) or the match isn't active."
    ),
)
async def submit_battle_match_answer(
    match_id: int,
    payload: SubmitBattleAnswerRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
) -> SubmitBattleAnswerResponse:
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )

    user = await get_or_create_user(db, clerk_id)
    data = await battle_gameplay_service.submit_battle_answer(
        db, match_id=match_id, user_id=user.id,
        question_id=payload.question_id, selected_option=payload.selected_option,
        redis=redis,
    )
    return SubmitBattleAnswerResponse(**data)


@router.post(
    "/match/{match_id}/ready",
    response_model=BattleMatchStateResponse,
    status_code=status.HTTP_200_OK,
    summary="Signal readiness to start",
    description=(
        "Marks the caller as ready to begin. Once both participants have "
        "readied up, question selection runs and the match moves into a "
        "server-driven countdown, then activates automatically once the "
        "countdown elapses. Idempotent -- re-sending 'ready' or calling "
        "this after the match has already left 'waiting' is a safe no-op. "
        "Live-connected clients also receive this over WebSocket "
        "(the 'ready' message on WS /battle/match/{match_id}/ws); this "
        "REST endpoint works the same without a socket."
    ),
)
async def ready_battle_match(
    match_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
) -> BattleMatchStateResponse:
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )

    user = await get_or_create_user(db, clerk_id)
    await battle_gameplay_service.mark_ready(db, redis, match_id=match_id, user_id=user.id)
    data = await battle_gameplay_service.get_match_state(db, match_id=match_id, user_id=user.id, redis=redis)
    return BattleMatchStateResponse(**data)


@router.post(
    "/match/{match_id}/forfeit",
    response_model=BattleMatchStateResponse,
    status_code=status.HTTP_200_OK,
    summary="Forfeit the match",
    description=(
        "Ends the match immediately with the caller recorded as the loser "
        "(result='forfeit') and the opponent as the winner, applying Elo "
        "based on whatever answers exist so far -- unless the match is "
        "still 'waiting' (matched but not yet both-ready), in which case "
        "it's voided instead with no rating changes, since no question was "
        "ever shown to either player. Idempotent -- calling this on an "
        "already-completed or already-cancelled match is a safe no-op. "
        "Notifies a live-connected opponent over WebSocket."
    ),
)
async def forfeit_battle_match(
    match_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
) -> BattleMatchStateResponse:
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )

    user = await get_or_create_user(db, clerk_id)
    match = await battle_gameplay_service.forfeit_match(
        db, redis, match_id=match_id, forfeiting_user_id=user.id
    )
    if match.status == "completed":
        event = await battle_ws_events.match_finished_event(db, match)
        await ws_manager.publish(redis, match_id=match_id, event=event)
    elif match.status == "cancelled":
        # Forfeiting a still-"waiting" match voids it instead of scoring a
        # win -- see forfeit_match's own comment for the rationale.
        cancel_event = battle_ws_events.match_cancelled_event(match)
        await ws_manager.publish(redis, match_id=match_id, event=cancel_event)
    data = await battle_gameplay_service.get_match_state(db, match_id=match_id, user_id=user.id, redis=redis)
    return BattleMatchStateResponse(**data)

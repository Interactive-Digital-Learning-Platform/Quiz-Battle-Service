"""
app/services/battle_ws_events.py
──────────────────────────────────────
Pure builders for outbound WebSocket event payloads that reuse the same
result-shaping logic already established in battle_gameplay_service.py
(participant_result_payload) rather than duplicating it. Used by the WS
route's per-connection progression timer (time-driven, local-only pushes)
and by battle_gameplay_service itself (action-driven, Redis-published events).

Each builder returns an already-`.model_dump(mode="json")`'d dict from its
matching app/schemas/battle_ws.py event class, not a hand-built dict --
ws_manager.publish()/push_local() just forward whatever dict they're given
over the wire, so validating it against the schema here is what actually
enforces "no arbitrary inconsistent JSON" for these events.
"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.battle_match import BattleMatch
from app.models.user import User
from app.schemas.battle_ws import MatchCancelledEvent, MatchFinishedEvent
from app.services import battle_gameplay_service


async def match_finished_event(db: AsyncSession, match: BattleMatch) -> dict:
    participants = await battle_gameplay_service._get_participants(db, match.id)
    # Flat, receiver-agnostic broadcast -- both players get the same
    # payload, so (unlike the REST me/opponent split) every entry gets its
    # OWN username here rather than just "the opponent's". Each client
    # picks out whichever entry isn't their own user_id to get a real name
    # instead of a generic "Opponent" placeholder.
    usernames_by_id = dict(
        (
            await db.execute(
                select(User.id, User.username).where(User.id.in_([p.user_id for p in participants]))
            )
        ).all()
    )
    return MatchFinishedEvent(
        winner_user_id=match.winner_user_id,
        participants=[
            battle_gameplay_service.participant_result_payload(p, username=usernames_by_id.get(p.user_id))
            for p in participants
        ],
    ).model_dump(mode="json")


def match_cancelled_event(match: BattleMatch) -> dict:
    return MatchCancelledEvent(reason=match.cancel_reason).model_dump(mode="json")

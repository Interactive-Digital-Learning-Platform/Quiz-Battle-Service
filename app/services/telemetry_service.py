"""
app/services/telemetry_service.py
────────────────────────────────────
Records AIGenerationEvent rows for battle question generation. Mirrors
Personalized-Quiz-Service's own telemetry_service.record_generation_event,
copied and trimmed (no session_id -- see app/models/ai_generation_event.py).
Never allowed to break the caller: this function swallows and logs every
exception itself, and its own caller (battle_question_service.py) wraps the
call again for the same reason -- a telemetry write failure must never take
down real question selection.
"""
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ai_generation_event import AIGenerationEvent

logger = logging.getLogger(__name__)


async def record_generation_event(
    db: AsyncSession, *, user_id: int | None, subject: str,
    requested_question_count: int, generated_question_count: int,
    provider: str, model_name: str, success: bool, used_cache_fallback: bool,
    retry_count: int, duplicate_count: int, invalid_question_count: int,
    latency_ms: float, error_category: str | None,
) -> None:
    try:
        event = AIGenerationEvent(
            user_id=user_id, subject=subject,
            requested_question_count=requested_question_count,
            generated_question_count=generated_question_count,
            provider=provider, model_name=model_name, success=success,
            used_cache_fallback=used_cache_fallback, retry_count=retry_count,
            duplicate_count=duplicate_count, invalid_question_count=invalid_question_count,
            latency_ms=latency_ms, error_category=error_category,
        )
        db.add(event)
        await db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to record AI generation telemetry (non-critical): %s", exc)
        try:
            await db.rollback()
        except Exception as rollback_exc:  # noqa: BLE001
            logger.error("Rollback after failed telemetry write also failed (non-critical): %s", rollback_exc)

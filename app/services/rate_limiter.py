"""
app/services/rate_limiter.py
──────────────────────────────
Minimal Redis-backed fixed-window rate limiting (INCR+EXPIRE, the standard
pattern -- no new dependency needed). Not a general API gateway limiter;
applied only at the two genuinely-unthrottled entry points that matter for
this scale (matchmaking_service.join_queue, battle_ws.py's pre-accept auth
path -- see their call sites for the configured thresholds in
app/core/config.py).

Fixed-window (not sliding/token-bucket) is a deliberate simplification: it
allows a caller up to 2x max_calls in a worst-case window-boundary burst,
which is an acceptable trade-off here in exchange for a single INCR+EXPIRE
round-trip instead of a sorted-set-based sliding window.
"""
from redis.asyncio import Redis


async def allow(redis: Redis, *, key: str, max_calls: int, window_seconds: int) -> bool:
    """Returns True if this call is within the limit, False if it should be
    rejected. Increments the counter regardless of outcome -- a rejected
    call still counts, so a caller hammering past the limit doesn't reset
    its own window by retrying.
    """
    count = await redis.incr(key)
    if count == 1:
        # Only the call that created the key sets its expiry -- an
        # unconditional EXPIRE on every call would push the window forward
        # indefinitely for a steady stream of calls, defeating "fixed".
        await redis.expire(key, window_seconds)
    return count <= max_calls

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


# KEYS[1]=counter  ARGV[1]=window_seconds
# INCR and the expiry in one atomic step. As two separate calls, a crash or
# Redis blip between them left a counter with no TTL -- that user stayed
# rate-limited forever. Setting the expiry whenever the key has none (rather
# than only when count == 1) also heals any counter already stuck that way.
_LUA_INCR_WITH_EXPIRY = """
local count = redis.call('INCR', KEYS[1])
if redis.call('TTL', KEYS[1]) < 0 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return count
"""


async def allow(redis: Redis, *, key: str, max_calls: int, window_seconds: int) -> bool:
    """Returns True if this call is within the limit, False if it should be
    rejected. Increments the counter regardless of outcome -- a rejected
    call still counts, so a caller hammering past the limit doesn't reset
    its own window by retrying. The expiry is only set when the key has
    none -- re-setting it on every call would push the window forward
    indefinitely for a steady stream of calls, defeating "fixed".
    """
    count = await redis.eval(_LUA_INCR_WITH_EXPIRY, 1, key, window_seconds)
    return int(count) <= max_calls

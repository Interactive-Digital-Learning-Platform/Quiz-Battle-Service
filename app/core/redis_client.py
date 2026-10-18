from redis.asyncio import Redis

from app.core.config import settings

# Redis.from_url is lazy -- it builds a connection pool but doesn't open a
# real connection until the first command is issued, so this module is safe
# to import even if Redis isn't reachable yet (mirrors create_async_engine's
# lazy-connect behavior in app/core/database.py).
redis_client: Redis = Redis.from_url(
    settings.REDIS_URL,
    decode_responses=True,
    socket_timeout=settings.REDIS_SOCKET_TIMEOUT_SECONDS,
    socket_connect_timeout=settings.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
)


async def get_redis() -> Redis:
    return redis_client

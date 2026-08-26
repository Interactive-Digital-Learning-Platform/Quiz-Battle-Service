"""
tests/test_rate_limiter.py
─────────────────────────────
Pure Redis-backed tests (fake_redis, no HTTP) for
app/services/rate_limiter.py's fixed-window allow().
"""
from app.services.rate_limiter import allow


async def test_allows_calls_within_the_limit(fake_redis):
    for _ in range(3):
        assert await allow(fake_redis, key="k", max_calls=3, window_seconds=10) is True


async def test_rejects_calls_over_the_limit(fake_redis):
    for _ in range(3):
        assert await allow(fake_redis, key="k", max_calls=3, window_seconds=10) is True
    assert await allow(fake_redis, key="k", max_calls=3, window_seconds=10) is False
    # Still rejected on subsequent over-limit calls, not just the first.
    assert await allow(fake_redis, key="k", max_calls=3, window_seconds=10) is False


async def test_different_keys_have_independent_windows(fake_redis):
    for _ in range(3):
        assert await allow(fake_redis, key="a", max_calls=3, window_seconds=10) is True
    assert await allow(fake_redis, key="a", max_calls=3, window_seconds=10) is False
    # A different key is unaffected by "a"'s exhausted window.
    assert await allow(fake_redis, key="b", max_calls=3, window_seconds=10) is True


async def test_key_expiry_is_set_only_on_first_call(fake_redis):
    await allow(fake_redis, key="k", max_calls=5, window_seconds=100)
    ttl_after_first = await fake_redis.ttl("k")
    assert ttl_after_first > 0

    await allow(fake_redis, key="k", max_calls=5, window_seconds=100)
    ttl_after_second = await fake_redis.ttl("k")
    # Second call must not push the TTL back up to ~100 again -- still a
    # "fixed" window, not extended on every hit.
    assert ttl_after_second <= ttl_after_first

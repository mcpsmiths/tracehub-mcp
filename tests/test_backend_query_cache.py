"""Unit tests for TTLCoalescingCache (see backends/cache.py's docstring for
design rationale: TTL + in-flight request coalescing under one lock,
hand-rolled since no TTL library implements coalescing natively).
"""

import asyncio

import pytest

from opentelemetry_mcp.backends.cache import TTLCoalescingCache


async def test_cache_hit_within_ttl_avoids_recompute() -> None:
    cache = TTLCoalescingCache(ttl_seconds=60.0)
    call_count = 0

    async def compute() -> str:
        nonlocal call_count
        call_count += 1
        return "result"

    first = await cache.get_or_compute("key", compute)
    second = await cache.get_or_compute("key", compute)

    assert first == "result"
    assert second == "result"
    assert call_count == 1


async def test_recomputes_after_ttl_expiry() -> None:
    clock_value = [0.0]
    cache = TTLCoalescingCache(ttl_seconds=10.0, clock=lambda: clock_value[0])
    call_count = 0

    async def compute() -> int:
        nonlocal call_count
        call_count += 1
        return call_count

    first = await cache.get_or_compute("key", compute)
    clock_value[0] = 10.0  # exactly at the TTL boundary - must recompute, not reuse
    second = await cache.get_or_compute("key", compute)

    assert first == 1
    assert second == 2
    assert call_count == 2


async def test_concurrent_identical_calls_are_coalesced() -> None:
    cache = TTLCoalescingCache(ttl_seconds=60.0)
    call_count = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def compute() -> str:
        nonlocal call_count
        call_count += 1
        started.set()
        await release.wait()
        return "result"

    async def wait_then_release() -> None:
        await started.wait()
        release.set()

    results = await asyncio.gather(
        cache.get_or_compute("key", compute),
        cache.get_or_compute("key", compute),
        wait_then_release(),
    )

    assert results[0] == "result"
    assert results[1] == "result"
    assert call_count == 1


async def test_exception_propagates_without_poisoning_subsequent_calls() -> None:
    cache = TTLCoalescingCache(ttl_seconds=60.0)
    call_count = 0

    async def failing_compute() -> str:
        nonlocal call_count
        call_count += 1
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await cache.get_or_compute("key", failing_compute)

    async def succeeding_compute() -> str:
        nonlocal call_count
        call_count += 1
        return "ok"

    result = await cache.get_or_compute("key", succeeding_compute)

    assert result == "ok"
    assert call_count == 2


async def test_clear_forces_a_recompute_on_next_call() -> None:
    cache = TTLCoalescingCache(ttl_seconds=60.0)
    call_count = 0

    async def compute() -> int:
        nonlocal call_count
        call_count += 1
        return call_count

    await cache.get_or_compute("key", compute)
    cache.clear()
    await cache.get_or_compute("key", compute)

    assert call_count == 2


async def test_different_keys_are_independent() -> None:
    cache = TTLCoalescingCache(ttl_seconds=60.0)
    calls: list[str] = []

    async def compute_a() -> str:
        calls.append("a")
        return "a-result"

    async def compute_b() -> str:
        calls.append("b")
        return "b-result"

    result_a = await cache.get_or_compute("a", compute_a)
    result_b = await cache.get_or_compute("b", compute_b)

    assert result_a == "a-result"
    assert result_b == "b-result"
    assert calls == ["a", "b"]

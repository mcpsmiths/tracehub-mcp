"""Unit tests for TTLCoalescingCache (see backends/cache.py's docstring for
design rationale: TTL + in-flight request coalescing under one lock,
hand-rolled since no TTL library implements coalescing natively).
"""

import asyncio

import pytest

from opentelemetry_mcp.backends.cache import DEFAULT_QUERY_CACHE_MAX_ENTRIES, TTLCoalescingCache


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


async def test_cancelling_the_leader_does_not_hang_a_coalesced_follower_forever() -> None:
    # asyncio.CancelledError is a BaseException, not an Exception - before
    # the fix, `except Exception` skipped the leader's cleanup entirely on
    # cancellation, leaving the shared future forever pending and the key
    # forever stuck in _inflight. A follower coalesced onto that leader
    # (line 54's `await future`) would then hang indefinitely, and every
    # later call for the same key would join the same dead future too.
    cache = TTLCoalescingCache(ttl_seconds=60.0)
    started = asyncio.Event()

    async def blocking_compute() -> str:
        started.set()
        await asyncio.Event().wait()  # never completes on its own
        return "unreachable"

    leader = asyncio.create_task(cache.get_or_compute("key", blocking_compute))
    await started.wait()
    follower = asyncio.create_task(cache.get_or_compute("key", blocking_compute))
    await asyncio.sleep(0)  # let the follower actually join the leader's future

    leader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leader

    # The follower must see the same cancellation surface, not hang forever.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(follower, timeout=1.0)

    # And the key must not be permanently poisoned - a fresh call succeeds.
    async def succeeding_compute() -> str:
        return "ok"

    result = await asyncio.wait_for(cache.get_or_compute("key", succeeding_compute), timeout=1.0)
    assert result == "ok"


async def test_clear_forces_a_recompute_on_next_call() -> None:
    cache = TTLCoalescingCache(ttl_seconds=60.0)
    call_count = 0

    async def compute() -> int:
        nonlocal call_count
        call_count += 1
        return call_count

    await cache.get_or_compute("key", compute)
    await cache.clear()
    await cache.get_or_compute("key", compute)

    assert call_count == 2


async def test_expired_entries_are_swept_even_if_never_requeried() -> None:
    # A key that's simply never queried again after expiring would
    # otherwise sit in _entries forever - only a hit on that exact key
    # replaces it. This proves the amortized sweep actually reclaims it via
    # an unrelated key's call, not just via a same-key cache-miss path.
    clock_value = [0.0]
    cache = TTLCoalescingCache(ttl_seconds=10.0, clock=lambda: clock_value[0])

    async def compute_stale() -> str:
        return "stale"

    await cache.get_or_compute("stale-key", compute_stale)
    assert "stale-key" in cache._entries

    clock_value[0] = 20.0  # well past ttl_seconds - stale-key has expired

    async def compute_other() -> str:
        return "other"

    await cache.get_or_compute("other-key", compute_other)  # unrelated key

    assert "stale-key" not in cache._entries


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


async def test_cancelling_a_follower_does_not_fail_the_leader_on_success() -> None:
    # Task.cancel() on a follower cancels the Future the follower is awaiting.
    # Without asyncio.shield that is the SHARED in-flight future, so the
    # leader's later set_result() raised InvalidStateError and its own
    # request failed although the backend had answered.
    cache = TTLCoalescingCache(ttl_seconds=60.0)
    call_count = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def compute() -> str:
        nonlocal call_count
        call_count += 1
        started.set()
        await release.wait()
        return "data"

    leader = asyncio.create_task(cache.get_or_compute("key", compute))
    await started.wait()
    cancelled_follower = asyncio.create_task(cache.get_or_compute("key", compute))
    follower = asyncio.create_task(cache.get_or_compute("key", compute))
    await asyncio.sleep(0)  # both followers join the leader's in-flight future

    cancelled_follower.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_follower

    release.set()
    assert await asyncio.wait_for(leader, timeout=1.0) == "data"
    assert await asyncio.wait_for(follower, timeout=1.0) == "data"
    assert call_count == 1
    assert "key" not in cache._inflight
    assert "key" in cache._entries


async def test_cancelling_a_follower_does_not_poison_a_failing_key() -> None:
    # On a failing compute the unguarded set_exception() raised
    # InvalidStateError BEFORE the _inflight.pop, so the cancelled future
    # stayed registered and every later identical call joined it and raised
    # CancelledError without touching the backend - until process restart.
    cache = TTLCoalescingCache(ttl_seconds=60.0)
    call_count = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def failing_compute() -> str:
        nonlocal call_count
        call_count += 1
        started.set()
        await release.wait()
        raise ValueError("boom")

    leader = asyncio.create_task(cache.get_or_compute("key", failing_compute))
    await started.wait()
    follower = asyncio.create_task(cache.get_or_compute("key", failing_compute))
    await asyncio.sleep(0)

    follower.cancel()
    with pytest.raises(asyncio.CancelledError):
        await follower

    release.set()
    with pytest.raises(ValueError, match="boom"):
        await asyncio.wait_for(leader, timeout=1.0)

    assert "key" not in cache._inflight

    async def succeeding_compute() -> str:
        nonlocal call_count
        call_count += 1
        return "ok"

    result = await asyncio.wait_for(cache.get_or_compute("key", succeeding_compute), timeout=1.0)
    assert result == "ok"
    assert call_count == 2


async def test_clear_sheds_a_stuck_inflight_computation() -> None:
    # clear() (reached via BaseBackend.close()) used to empty only _entries,
    # so a stuck or poisoned in-flight slot survived it. Dropping the slot is
    # enough: the next caller becomes a fresh leader, and the old leader still
    # completes its own (shielded) future for anyone already coalesced on it.
    cache = TTLCoalescingCache(ttl_seconds=60.0)
    call_count = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_compute() -> str:
        nonlocal call_count
        call_count += 1
        started.set()
        await release.wait()
        return "late"

    old_leader = asyncio.create_task(cache.get_or_compute("key", blocking_compute))
    await started.wait()

    await cache.clear()
    assert "key" not in cache._inflight

    async def fresh_compute() -> str:
        nonlocal call_count
        call_count += 1
        return "fresh"

    result = await asyncio.wait_for(cache.get_or_compute("key", fresh_compute), timeout=1.0)
    assert result == "fresh"
    assert call_count == 2

    release.set()
    assert await asyncio.wait_for(old_leader, timeout=1.0) == "late"


async def test_entry_count_is_bounded_independent_of_ttl() -> None:
    # Retention used to be modelled purely as time: a long TTL (or an
    # ever-changing key space - one key per distinct filter combination)
    # retained one full entry per distinct query with no aggregate bound.
    clock_value = [0.0]
    cache = TTLCoalescingCache(600.0, clock=lambda: clock_value[0], max_entries=100)

    async def make_compute(value: int) -> int:
        return value

    for i in range(101):
        await cache.get_or_compute(f"key-{i}", lambda i=i: make_compute(i))  # type: ignore[misc]

    assert len(cache._entries) == 100
    assert "key-0" not in cache._entries  # the oldest entry was evicted
    assert "key-1" in cache._entries

    # LRU, not FIFO: a hit refreshes key-2, so the next insert evicts key-1
    # (now the least recently used) and key-2 survives.
    assert await cache.get_or_compute("key-2", lambda: make_compute(2)) == 2
    await cache.get_or_compute("key-101", lambda: make_compute(101))

    assert len(cache._entries) == 100
    assert "key-2" in cache._entries
    assert "key-1" not in cache._entries
    assert "key-3" in cache._entries


@pytest.mark.parametrize(
    ("ttl_seconds", "max_entries"),
    [(float("inf"), 1024), (float("nan"), 1024), (60.0, 0), (0.0, 1024), (-1.0, 1024)],
)
def test_non_finite_or_non_positive_ttl_and_zero_capacity_raise(
    ttl_seconds: float, max_entries: int
) -> None:
    # With an infinite TTL the amortized sweep gate never opened, so nothing
    # was ever evicted; the cache must refuse to be built that way.
    with pytest.raises(ValueError):
        TTLCoalescingCache(ttl_seconds, max_entries=max_entries)


def test_default_capacity_is_the_documented_constant() -> None:
    assert DEFAULT_QUERY_CACHE_MAX_ENTRIES == 1024
    assert TTLCoalescingCache(60.0)._max_entries == DEFAULT_QUERY_CACHE_MAX_ENTRIES

"""Hand-rolled TTL cache + in-flight request coalescing.

No TTL library (aiocache/asyncache+cachetools) implements request
coalescing natively - a hand-rolled asyncio.Future-keyed in-flight map is
needed regardless, so this hand-rolls both under one lock rather than
composing two libraries with two separate lock domains to keep consistent.
Matches this codebase's existing precedent in backends/base.py, which
hand-rolls retry/SSRF-guarding rather than adding a library.

See backends/base.py's BaseBackend.__init_subclass__ for how this wraps
BaseBackend's read-only query methods generically, with no per-backend-file
changes.
"""

import asyncio
import math
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Hashable
from typing import Any

# Default upper bound on the number of entries one TTLCoalescingCache holds.
# Retention was previously modelled purely as time (TTL + an amortized expiry
# sweep), so a long TTL - or simply an ever-changing key space, one key per
# distinct filter combination minted by any peer able to call tools - kept
# one full result per distinct query in the process every MCP session
# shares. The bound is an entry count with least-recently-used eviction on
# every insert; QUERY_CACHE_MAX_ENTRIES / --query-cache-max-entries set it
# per deployment (config.py).
DEFAULT_QUERY_CACHE_MAX_ENTRIES = 1024


class TTLCoalescingCache:
    """Caches the result of `compute()` per key for `ttl_seconds`, and
    coalesces concurrent calls sharing the same key onto a single
    in-flight `compute()` invocation."""

    def __init__(
        self,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        max_entries: int = DEFAULT_QUERY_CACHE_MAX_ENTRIES,
    ) -> None:
        # inf would mean "never expire" AND never open the sweep gate below;
        # nan compares False against everything, so it would never expire
        # either. Both are refused here rather than silently accepted.
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError(
                f"ttl_seconds must be a finite number greater than 0, got {ttl_seconds!r}"
            )
        if max_entries <= 0:
            raise ValueError(f"max_entries must be greater than 0, got {max_entries!r}")
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        # Insertion/recency-ordered so the oldest-used entry is O(1) to find
        # and evict once the entry bound is reached.
        self._entries: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._inflight: dict[Hashable, asyncio.Future[Any]] = {}
        self._lock = asyncio.Lock()
        self._last_swept = self._clock()

    def _sweep_expired_locked(self, now: float) -> None:
        """Drop expired entries. Caller must already hold self._lock.

        Amortized rather than per-call: an entry that's simply never
        queried again after expiry would otherwise sit in `_entries`
        forever (only a hit on the exact same key ever replaces it), an
        unbounded-growth risk for a long-running server with a large or
        ever-changing key space (e.g. one key per distinct search filter
        combination). Runs at most once per ttl_seconds, not on every call,
        so this stays O(1) amortized rather than O(len(_entries)) per call.
        The hard bound on live entries is _evict_to_capacity_locked, applied
        on every insert; this sweep only reclaims expired entries earlier.
        """
        if now - self._last_swept < self._ttl_seconds:
            return
        self._last_swept = now
        expired = [key for key, (expires_at, _) in self._entries.items() if expires_at <= now]
        for key in expired:
            del self._entries[key]

    def _evict_to_capacity_locked(self) -> None:
        """Drop least-recently-used entries until the entry bound holds.
        Caller must already hold self._lock."""
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    async def get_or_compute(self, key: Hashable, compute: Callable[[], Awaitable[Any]]) -> Any:
        now = self._clock()
        async with self._lock:
            self._sweep_expired_locked(now)
            cached = self._entries.get(key)
            if cached is not None and cached[0] > now:
                self._entries.move_to_end(key)  # a hit makes the entry most recently used
                return cached[1]
            existing_future = self._inflight.get(key)
            if existing_future is not None:
                is_leader = False
                future = existing_future
            else:
                is_leader = True
                future = asyncio.get_running_loop().create_future()
                self._inflight[key] = future

        # Awaiting a joined in-flight call happens outside the lock: the
        # leader (below) needs to re-acquire this same lock after compute()
        # finishes to store the result and clean up _inflight - awaiting
        # the future while still holding the lock would deadlock the
        # leader against itself.
        #
        # Shielded, because Task.cancel() on a follower cancels the Future
        # that follower is awaiting - and that is the one future every
        # coalesced caller shares. Unshielded, one follower's cancellation
        # (an MCP client timing out and sending notifications/cancelled is
        # routine) left the leader calling set_result()/set_exception() on
        # an already-cancelled future: InvalidStateError failed the leader's
        # own request although the backend had answered, and on a failing
        # compute the raise happened before the in-flight slot was popped,
        # so the cancelled future stayed registered and every later
        # identical query from any session raised CancelledError without a
        # backend call until the process restarted. With the shield the
        # follower's cancellation reaches only the follower; the shared
        # future still carries the leader's real outcome to everyone else.
        if not is_leader:
            return await asyncio.shield(future)

        try:
            result = await compute()
        except (Exception, asyncio.CancelledError) as exc:
            # asyncio.CancelledError is a BaseException, not an Exception,
            # since Python 3.8 - `except Exception` alone would silently
            # skip this whole block for a cancelled leader (e.g. an MCP
            # request cancelled by the client, or a timeout), leaving
            # `future` forever pending and `key` forever in `_inflight`.
            # Every follower already coalesced onto it (the shielded `await
            # future` above) - and every future caller, since the poisoned
            # entry is never popped - would then hang indefinitely on this
            # exact key. Explicitly catching it here (rather than bare
            # `BaseException`, which would also swallow SystemExit/
            # KeyboardInterrupt) ensures followers instead see the same
            # CancelledError propagate into their own await, an honest
            # "this shared computation never completed" outcome.
            #
            # `done()` guard: the leader is the only writer of this future
            # now that followers are shielded, so this is defence in depth -
            # completing a future twice raises InvalidStateError, which would
            # otherwise replace the real error the caller should see.
            if not future.done():
                future.set_exception(exc)
                future.exception()  # mark retrieved: avoids an "exception
                # never retrieved" asyncio warning when no coalesced caller
                # ever awaits this future (only the leader raises here).
            raise
        else:
            async with self._lock:
                self._entries[key] = (now + self._ttl_seconds, result)
                # Assigning an existing key keeps its OrderedDict position;
                # a fresh result must count as most recently used.
                self._entries.move_to_end(key)
                self._evict_to_capacity_locked()
            if not future.done():
                future.set_result(result)
            return result
        finally:
            # Always release the slot - on success, on error AND on
            # cancellation - so no key can stay unservable. Identity-checked:
            # clear() may already have dropped this slot and a newer leader
            # may own the key by now; its slot must not be removed by us.
            async with self._lock:
                if self._inflight.get(key) is future:
                    del self._inflight[key]

    async def clear(self) -> None:
        # Every other read/write of _entries/_inflight in this class holds
        # self._lock first - clear() skipping it could interleave with a
        # concurrent get_or_compute() call and drop an entry that call is
        # mid-write into (e.g. clear() during shutdown racing a request
        # that's still in flight).
        #
        # _inflight is dropped too, so close()/clear() shed a stuck or
        # poisoned computation instead of leaving its key unservable until
        # the process restarts. Dropping the slot is deliberately all this
        # does - the dropped futures are NOT cancelled: the old leader still
        # completes its own future (so its shielded followers receive the
        # real outcome), and the next caller for the key simply becomes a
        # fresh leader.
        async with self._lock:
            self._entries.clear()
            self._inflight.clear()

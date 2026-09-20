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
import time
from collections.abc import Awaitable, Callable, Hashable
from typing import Any


class TTLCoalescingCache:
    """Caches the result of `compute()` per key for `ttl_seconds`, and
    coalesces concurrent calls sharing the same key onto a single
    in-flight `compute()` invocation."""

    def __init__(self, ttl_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: dict[Hashable, tuple[float, Any]] = {}
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
        """
        if now - self._last_swept < self._ttl_seconds:
            return
        self._last_swept = now
        expired = [key for key, (expires_at, _) in self._entries.items() if expires_at <= now]
        for key in expired:
            del self._entries[key]

    async def get_or_compute(self, key: Hashable, compute: Callable[[], Awaitable[Any]]) -> Any:
        now = self._clock()
        async with self._lock:
            self._sweep_expired_locked(now)
            cached = self._entries.get(key)
            if cached is not None and cached[0] > now:
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
        if not is_leader:
            return await future

        try:
            result = await compute()
        except (Exception, asyncio.CancelledError) as exc:
            # asyncio.CancelledError is a BaseException, not an Exception,
            # since Python 3.8 - `except Exception` alone would silently
            # skip this whole block for a cancelled leader (e.g. an MCP
            # request cancelled by the client, or a timeout), leaving
            # `future` forever pending and `key` forever in `_inflight`.
            # Every follower already coalesced onto it (line 54's `await
            # future`) - and every future caller, since the poisoned entry
            # is never popped - would then hang indefinitely on this exact
            # key. Explicitly catching it here (rather than bare
            # `BaseException`, which would also swallow SystemExit/
            # KeyboardInterrupt) ensures followers instead see the same
            # CancelledError propagate into their own await, an honest
            # "this shared computation never completed" outcome.
            future.set_exception(exc)
            future.exception()  # mark retrieved: avoids an "exception never
            # retrieved" asyncio warning when no coalesced caller ever
            # awaits this future (only the leader raises here directly).
            async with self._lock:
                self._inflight.pop(key, None)
            raise
        else:
            async with self._lock:
                self._entries[key] = (now + self._ttl_seconds, result)
                self._inflight.pop(key, None)
            future.set_result(result)
            return result

    async def clear(self) -> None:
        # Every other read/write of _entries/_inflight in this class holds
        # self._lock first - clear() skipping it could interleave with a
        # concurrent get_or_compute() call and drop an entry that call is
        # mid-write into (e.g. clear() during shutdown racing a request
        # that's still in flight).
        async with self._lock:
            self._entries.clear()

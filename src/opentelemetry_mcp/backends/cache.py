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

    async def get_or_compute(self, key: Hashable, compute: Callable[[], Awaitable[Any]]) -> Any:
        now = self._clock()
        async with self._lock:
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
        except Exception as exc:
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

    def clear(self) -> None:
        self._entries.clear()

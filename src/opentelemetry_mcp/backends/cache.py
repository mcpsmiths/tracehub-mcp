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
import functools
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


class _InFlight:
    """One key's in-flight computation: the future every coalesced caller
    awaits, the finisher task that completes it (None until the leader has
    started it), how many followers are waiting on it right now, and whether
    the leader abandoned it (both read by TTLCoalescingCache.get_or_compute).
    Mutated only between awaits on the one event loop, so no lock."""

    __slots__ = ("abandoned", "finisher", "future", "waiting_followers")

    def __init__(self, future: asyncio.Future[Any]) -> None:
        self.future = future
        self.finisher: asyncio.Task[Any] | None = None
        self.waiting_followers = 0
        self.abandoned = False

    def is_joinable(self) -> bool:
        """False once this computation can no longer hand a newcomer a real
        outcome: its leader abandoned it (see get_or_compute), its finisher
        has a cancellation pending or already observed, or its future ended
        in a cancellation. A newcomer then treats the key as a miss and
        starts a fresh computation instead of inheriting a CancelledError
        it never asked for."""
        if self.abandoned:
            return False
        finisher = self.finisher
        if finisher is not None and (finisher.cancelling() or finisher.cancelled()):
            return False
        future = self.future
        return not (
            future.done()
            and (future.cancelled() or isinstance(future.exception(), asyncio.CancelledError))
        )


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
        self._inflight: dict[Hashable, _InFlight] = {}
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
            existing = self._inflight.get(key)
            if existing is not None and existing.is_joinable():
                is_leader = False
                inflight = existing
            else:
                # No computation under the key, or one that can no longer
                # answer (its leader abandoned it, see below): this caller
                # becomes a fresh leader and takes the slot. The old
                # finisher's own cleanup is identity-checked, so it never
                # removes the newer slot.
                is_leader = True
                inflight = _InFlight(asyncio.get_running_loop().create_future())
                self._inflight[key] = inflight

        # Awaiting a joined in-flight call happens outside the lock: the
        # finisher (below) needs to re-acquire this same lock after compute()
        # finishes to store the result and clean up _inflight - awaiting
        # the future while still holding the lock would deadlock it.
        #
        # Shielded, because Task.cancel() on a follower cancels the Future
        # that follower is awaiting - and that is the one future every
        # coalesced caller shares. Unshielded, one follower's cancellation
        # (an MCP client timing out and sending notifications/cancelled is
        # routine) left the finisher calling set_result()/set_exception() on
        # an already-cancelled future: InvalidStateError failed the leader's
        # own request although the backend had answered, and on a failing
        # compute the raise happened before the in-flight slot was popped,
        # so the cancelled future stayed registered and every later
        # identical query from any session raised CancelledError without a
        # backend call until the process restarted. With the shield the
        # follower's cancellation reaches only the follower; the shared
        # future still carries the real outcome to everyone else.
        if not is_leader:
            inflight.waiting_followers += 1
            try:
                return await asyncio.shield(inflight.future)
            finally:
                inflight.waiting_followers -= 1

        # The leader is only the FIRST caller; the computation it starts is
        # shared. It therefore runs in its own task - the finisher - that
        # the leader awaits through a shield, so cancelling the leader (the
        # same routine client cancellation as above, from one session) no
        # longer cancels the backend request every follower from every other
        # session is waiting on: the runtime probe saw both followers fail
        # with -32000 "Connection closed" although the backend was healthy.
        # The finisher publishes the result to the shared future and the
        # cache and releases the slot whether or not the leader is still
        # there. When NO follower is waiting at the instant the leader is
        # cancelled, nobody can use the result, so the finisher is cancelled
        # with it rather than left as orphan backend work. That instant is
        # the only one at which the finisher is ever cancelled: once a
        # follower has joined, the computation runs to completion and its
        # result is cached even if every follower leaves before it finishes
        # - a departed follower cannot un-send a backend request, and the
        # cached result serves the next caller (clear() does not cancel
        # either, see there).
        finisher = asyncio.create_task(self._compute_and_publish(key, inflight, compute, now))
        inflight.finisher = finisher
        finisher.add_done_callback(functools.partial(self._settle_abandoned, key, inflight))
        try:
            return await asyncio.shield(finisher)
        except asyncio.CancelledError:
            # Synchronous from here to the raise - no await - so the follower
            # count is exact and nothing can join between the check and the
            # cancel. The slot is released NOW rather than when the finisher
            # observes its cancellation one loop turn later: a follower that
            # arrived in that turn used to find the slot still registered,
            # join it, and receive the finisher's CancelledError although it
            # was never cancelled itself (round-2 review of the shielded
            # finisher). Released and marked abandoned, the key reads as a
            # miss to the next caller, which becomes a fresh leader; the
            # finisher's own identity-checked cleanup then leaves that newer
            # slot alone. Without the lock for the reason _settle_abandoned
            # gives: no locked section in this class awaits, so none is in
            # progress while this handler runs.
            if inflight.waiting_followers == 0 and not finisher.done():
                inflight.abandoned = True
                if self._inflight.get(key) is inflight:
                    del self._inflight[key]
                finisher.cancel()
            raise

    async def _compute_and_publish(
        self,
        key: Hashable,
        inflight: _InFlight,
        compute: Callable[[], Awaitable[Any]],
        now: float,
    ) -> Any:
        """The finisher: run compute() once, store a success in the cache,
        complete the shared future either way, and release the in-flight
        slot. Runs as its own task (see get_or_compute), so it outlives a
        cancelled leader while followers wait."""
        future = inflight.future
        try:
            result = await compute()
        except (Exception, asyncio.CancelledError) as exc:
            # asyncio.CancelledError is a BaseException, not an Exception,
            # since Python 3.8 - `except Exception` alone would silently
            # skip this whole block when the finisher itself is cancelled
            # (a leader cancelled with no followers waiting), leaving
            # `future` forever pending and `key` forever in `_inflight`, so
            # every later caller for this exact key would hang. Explicitly
            # catching it here (rather than bare `BaseException`, which
            # would also swallow SystemExit/KeyboardInterrupt) completes the
            # future with the same CancelledError instead - an honest "this
            # shared computation never completed" outcome. Nobody receives
            # it in practice: the leader released the slot and marked it
            # abandoned before requesting the cancel, so no caller can join
            # between the request and this step (is_joinable refuses the
            # slot even if a stale reference were found).
            #
            # `done()` guard: the finisher is the only writer of this future
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
                if self._inflight.get(key) is inflight:
                    del self._inflight[key]

    def _settle_abandoned(
        self, key: Hashable, inflight: _InFlight, finisher: asyncio.Task[Any]
    ) -> None:
        """Done-callback on every finisher. Two duties for a finisher whose
        leader has left: retrieve its exception, so asyncio does not log
        "Task exception was never retrieved" for a failure the followers
        already received through the shared future; and, should the task
        have been cancelled before its first step ever ran (so the
        try/finally above never executed), fail the shared future and
        release the slot so no later caller joins a dead computation. Both
        are idempotent no-ops after a finisher that ran. Synchronous and
        without the lock by construction: no locked section in this class
        awaits between acquiring and releasing, so none can be interrupted
        by a loop callback."""
        if not finisher.cancelled():
            finisher.exception()
        if not inflight.future.done():
            inflight.future.cancel()
        if self._inflight.get(key) is inflight:
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
        # does - the dropped computations are NOT cancelled: the old
        # finisher still completes its own future (so its shielded
        # followers receive the real outcome), and the next caller for the
        # key simply becomes a fresh leader.
        async with self._lock:
            self._entries.clear()
            self._inflight.clear()

"""Abstract base backend for OpenTelemetry trace storage systems."""

import asyncio
import functools
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    retry_any,
    retry_if_exception_type,
    retry_if_result,
    wait_exponential,
)

from opentelemetry_mcp.attributes import HealthCheckResponse
from opentelemetry_mcp.backends.cache import DEFAULT_QUERY_CACHE_MAX_ENTRIES, TTLCoalescingCache
from opentelemetry_mcp.models import FilterOperator, SpanData, SpanQuery, TraceData, TraceQuery
from opentelemetry_mcp.security import is_cloud_metadata_host

logger = logging.getLogger(__name__)

# health_check() is deliberately excluded: /ready and doctor both depend on
# it reflecting live backend state, and caching it would defeat that. The
# rule that follows from it: health_check() and every readiness-probe helper
# must never call a cache-wrapped PUBLIC method listed here (a backend's own
# health_check that awaited self.list_services() would be answered from the
# cache for a whole TTL after the backend failed). Call a private uncached
# helper instead (XRayBackend._list_services_uncached,
# JaegerBackend._list_services_uncached, DatadogBackend._search_spans_raw);
# server._run_backend_checks probes through list_services_uncached() below.
_CACHEABLE_METHODS = (
    "search_traces",
    "search_spans",
    "get_trace",
    "list_services",
    "get_service_operations",
)
# The two methods that take a TraceQuery/SpanQuery and therefore carry a
# `limit` the operator's MAX_TRACES_PER_QUERY must bound. Clamped at this
# boundary - not only in the tool handlers that call server._clamp_limit -
# because handlers re-expand the clamped value (expensive/slow traces fetch
# limit * 10), helpers pick their own limits (model_stats 1000, correlate 20)
# and new callers appear; the boundary covers all of them, present and future.
_QUERY_LIMITED_METHODS = ("search_traces", "search_spans")


# Upper bound on the rendered size of one cache key. A key is built from the
# validated arguments (pydantic models rendered with model_dump_json), so its
# size tracks the request's - up to the MCP transport's request-body cap. For
# Jaeger any non-service.name filter is applied client-side, so a request
# carrying an arbitrarily large filter value costs the backend one small GET
# that succeeds empty while the server would retain the whole filter text
# under the key. A call whose key exceeds this is still served; it just never
# occupies a cache slot.
_MAX_CACHE_KEY_BYTES = 64 * 1024


def _render_cache_part(value: Any) -> Any:
    return value.model_dump_json() if isinstance(value, BaseModel) else value


def _cache_key(method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    parts: list[Any] = [method_name]
    parts.extend(_render_cache_part(value) for value in args)
    # Keyword names participate (sorted, so call-site order is irrelevant):
    # two calls passing the same values under different parameter names are
    # different queries and must not share an entry.
    parts.extend((name, _render_cache_part(value)) for name, value in sorted(kwargs.items()))
    return tuple(parts)


def _cache_key_size(key: tuple[Any, ...]) -> int:
    """UTF-8 size of every string the key carries - the only parts whose
    size a caller controls (method names, ids, rendered query models)."""
    size = 0
    for part in key:
        value = part[1] if isinstance(part, tuple) else part
        if isinstance(value, str):
            size += len(value.encode("utf-8"))
    return size


def _wrap_with_query_limit(
    method_name: str, original: Callable[..., Awaitable[Any]]
) -> Callable[..., Awaitable[Any]]:
    """Clamp the query's `limit` to self.max_query_limit before `original`
    (the cache wrapper, then the backend's own method) sees it. Applied
    OUTSIDE _wrap_with_cache so the cache key is computed from the clamped
    query: limit=500 and limit=1000 under a ceiling of 50 are the same
    backend query and share one entry. The caller's query object is never
    mutated - a copy with the clamped limit is handed down."""

    @functools.wraps(original)
    async def wrapper(self: "BaseBackend", *args: Any, **kwargs: Any) -> Any:
        ceiling = self.max_query_limit
        if ceiling is None:
            return await original(self, *args, **kwargs)
        query = args[0] if args else kwargs.get("query")
        if isinstance(query, TraceQuery | SpanQuery) and query.limit > ceiling:
            clamped = query.model_copy(update={"limit": ceiling})
            if args:
                args = (clamped, *args[1:])
            else:
                kwargs = {**kwargs, "query": clamped}
        return await original(self, *args, **kwargs)

    return wrapper


def _wrap_with_cache(
    method_name: str, original: Callable[..., Awaitable[Any]]
) -> Callable[..., Awaitable[Any]]:
    @functools.wraps(original)
    async def wrapper(self: "BaseBackend", *args: Any, **kwargs: Any) -> Any:
        cache = self._query_cache
        if cache is None:
            return await original(self, *args, **kwargs)
        key = _cache_key(method_name, args, kwargs)
        if _cache_key_size(key) > _MAX_CACHE_KEY_BYTES:
            # Oversized key: no lookup, no store - a request-sized key must
            # never occupy one of the bounded slots.
            return await original(self, *args, **kwargs)
        return await cache.get_or_compute(key, lambda: original(self, *args, **kwargs))

    return wrapper


class MetadataEndpointBlockedError(httpx.TransportError):
    """Raised when a request (initial or mid-redirect) targets a cloud
    instance-metadata endpoint. Deliberately not one of the retryable
    exceptions below - this must propagate immediately, never be retried.
    """


class RedirectTargetBlockedError(httpx.TransportError):
    """Raised when a request (in practice a redirect hop chosen by the
    backend's own 3xx response) targets a host other than the operator-
    configured backend host. Like MetadataEndpointBlockedError, never
    retried - the hop is refused before the wrapped transport sees it.
    """


# Ceiling on the DECODED (post-Content-Encoding) size of any single backend
# response, in bytes. httpx advertises Accept-Encoding: gzip, deflate and
# inflates transparently on the read path, so without a bound a ~1 KB wire
# body chosen by the backend (or a redirect target) becomes ~1 MB of heap,
# then ~3x that again when json() parses it - in the one process every MCP
# session and GET /ready share. 64 MiB is far above a MAX_TRACES_PER_QUERY=500
# Jaeger response in normal use; BACKEND_MAX_RESPONSE_BYTES raises or lowers
# it per deployment (config.py).
DEFAULT_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
# int() refuses a digit string longer than sys.int_info.str_digits_check_threshold
# (4300 digits) with a bare ValueError. A Content-Length with more digits than
# this is oversized by construction (19 digits is already ~10^9 times the 1 GiB
# config ceiling), so it is refused without being parsed - as
# ResponseTooLargeError, not as an unrelated interpreter message.
_MAX_CONTENT_LENGTH_DIGITS = 19


class ResponseTooLargeError(httpx.TransportError):
    """Raised when a backend (or redirect-target) response declares, or
    decodes to, more than the configured max_response_bytes. Deliberately
    not one of the retryable exceptions below - the same backend would send
    the same body again.
    """


# The wall-clock deadline for one LOGICAL backend request - connect, every
# retry attempt and the sleep before it, every redirect hop and the whole
# body read - derived from the operator's BACKEND_TIMEOUT as
# max(_MIN_REQUEST_DEADLINE_SECONDS, _REQUEST_DEADLINE_TIMEOUT_MULTIPLIER x
# timeout). httpx's timeout is per SOCKET OPERATION (connect, one read, one
# write), so without this a backend that completes every operation just
# inside BACKEND_TIMEOUT - one body byte every 2 s under a 5 s timeout, far
# below the decoded-size cap - holds a tool call, and with the query cache
# every coalesced caller, indefinitely (302.5 s in the runtime probe, ended
# only by killing the backend). Four timeouts cover the worst legitimate
# shape the retry policy allows (three attempts plus the 10 s retry budget)
# with room for a slow body; the floor keeps a small BACKEND_TIMEOUT from
# cutting a large but healthy response. No separate knob: raising
# BACKEND_TIMEOUT raises this proportionally.
_REQUEST_DEADLINE_TIMEOUT_MULTIPLIER = 4
_MIN_REQUEST_DEADLINE_SECONDS = 30.0
# request.extensions key under which _RetryingTransport keeps the per-logical-
# request _RequestBudget. httpx builds a fresh Request for every redirect hop
# but passes `extensions=request.extensions` to it (_build_redirect_request,
# pinned by tests/test_base_backend.py), so the budget object - not a copy -
# reaches every hop; a retried attempt reuses the same Request outright.
_REQUEST_BUDGET_EXTENSION = "tracehub_request_budget"


def request_deadline_seconds(timeout: float | None) -> float:
    """The total deadline for one logical backend request, derived from the
    backend's per-operation `timeout` (BACKEND_TIMEOUT): four timeouts, and
    never below 30 s. See _REQUEST_DEADLINE_TIMEOUT_MULTIPLIER."""
    if timeout is None:
        return _MIN_REQUEST_DEADLINE_SECONDS
    return max(_MIN_REQUEST_DEADLINE_SECONDS, _REQUEST_DEADLINE_TIMEOUT_MULTIPLIER * float(timeout))


class BackendDeadlineExceededError(httpx.TransportError):
    """Raised when one logical backend request (all attempts, redirect hops
    and the body read together) has not completed within the transport's
    total deadline. A sibling of httpx.TimeoutException on purpose, not a
    subclass: the retry predicate re-sends TimeoutExceptions, and no second
    attempt can fit inside a deadline that has already passed.
    """


class _RequestBudget:
    """What one LOGICAL backend request has left, shared by every attempt
    and redirect hop of that request via request.extensions (see
    _REQUEST_BUDGET_EXTENSION). `deadline` is a time.monotonic() instant,
    or None for a transport built without a total deadline; the retry
    budget (_MAX_ATTEMPTS - 1 retries, _MAX_RETRY_AFTER_SECONDS of
    cumulative sleep) and the redirect hop count are spent across the
    whole chain, never re-armed per hop."""

    __slots__ = ("deadline", "hops", "retries_used", "retry_sleep_used")

    def __init__(self, deadline: float | None) -> None:
        self.deadline = deadline
        self.hops = 0
        self.retries_used = 0
        self.retry_sleep_used = 0.0

    def can_afford_retry(self, sleep_seconds: float) -> bool:
        """Whether one more retry, preceded by `sleep_seconds` of sleep,
        fits what the logical request has left: a retry still unspent, the
        cumulative sleep still under the ceiling, and the deadline not
        reached by the end of the sleep. A sleep that cannot fit is not
        taken at all - the backend's real last outcome reaches the caller
        now rather than after an idle wait up to the deadline."""
        if self.retries_used >= _MAX_ATTEMPTS - 1:
            return False
        if self.retry_sleep_used + sleep_seconds > _MAX_RETRY_AFTER_SECONDS:
            return False
        return self.deadline is None or time.monotonic() + sleep_seconds < self.deadline

    def spend_retry(self, sleep_seconds: float) -> None:
        self.retries_used += 1
        self.retry_sleep_used += sleep_seconds

    def remaining_seconds(self) -> float | None:
        """Seconds left before the deadline (None = unbounded). Raises
        BackendDeadlineExceededError once the deadline has passed, so a
        caller about to start another await never does so with no budget."""
        if self.deadline is None:
            return None
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise _deadline_exceeded()
        return remaining


def _deadline_exceeded() -> BackendDeadlineExceededError:
    return BackendDeadlineExceededError(
        "Backend request did not complete within its total deadline "
        f"({_REQUEST_DEADLINE_TIMEOUT_MULTIPLIER} x BACKEND_TIMEOUT, at least "
        f"{_MIN_REQUEST_DEADLINE_SECONDS:.0f} s, covering every retry, redirect hop "
        "and the body read); giving up"
    )


class _BoundedResponse(httpx.Response):
    """The response _RetryingTransport hands back to httpx: the wrapped
    transport's status, headers, body stream and extensions, with an
    aiter_bytes() that counts DECODED bytes and raises ResponseTooLargeError
    once the running total passes the cap, and that gives up with
    BackendDeadlineExceededError when the logical request's deadline passes
    between two chunks (a body trickling in below the read timeout).

    httpx (~=0.28) reads every non-stream response - and every redirect hop
    it follows - through Response.aread(), which is implemented over
    aiter_bytes(), so this one override bounds all 25 response.json() sites
    across the backends with no call-site changes. Overshoot is bounded by
    the expansion of a single raw chunk, never by the whole body.
    """

    def __init__(
        self, inner: httpx.Response, max_response_bytes: int, budget: _RequestBudget
    ) -> None:
        super().__init__(
            inner.status_code,
            headers=inner.headers,
            stream=inner.stream,
            extensions=inner.extensions,
        )
        self._max_response_bytes = max_response_bytes
        self._budget = budget

    async def aiter_bytes(self, chunk_size: int | None = None) -> AsyncIterator[bytes]:
        decoded_total = 0
        chunks = super().aiter_bytes(chunk_size)
        while True:
            # Each wait for the next chunk is capped at the time left on the
            # logical request, not at httpx's per-read timeout: a backend
            # pacing its bytes just inside the read timeout keeps every
            # individual read "healthy" while the request as a whole never
            # ends. The expired scope is the only TimeoutError converted
            # here; any other one is somebody else's and propagates as is.
            try:
                remaining = self._budget.remaining_seconds()
            except BackendDeadlineExceededError:
                await self.aclose()
                raise
            try:
                async with asyncio.timeout(remaining) as scope:
                    chunk = await anext(chunks)
            except StopAsyncIteration:
                break
            except TimeoutError:
                if not scope.expired():
                    raise
                # Release the pooled connection now rather than leaving the
                # half-read body to the client's own exception handling.
                await self.aclose()
                raise _deadline_exceeded() from None
            decoded_total += len(chunk)
            if decoded_total > self._max_response_bytes:
                await self.aclose()
                raise ResponseTooLargeError(
                    f"Backend response exceeded {self._max_response_bytes} decoded bytes "
                    "(BACKEND_MAX_RESPONSE_BYTES); refusing to buffer it"
                )
            yield chunk


# Only genuine transport-level failures - a connection that never completed
# or a request that timed out - are worth retrying on their own. An HTTP
# response with a 4xx/5xx status code is a *successful* transport exchange
# (a response body still came back), not one of these exceptions, so it is
# never retried via this mechanism. 429 is retried too, but as a *result*
# condition (see _is_rate_limited) rather than an exception, since asking to
# slow down and retry is exactly what 429 means - unlike a genuine 4xx/5xx
# error, which should reach the caller immediately. The retry sleep runs
# here in the transport, OUTSIDE the per-attempt httpx timeout, so a
# backend-chosen Retry-After is honoured only up to _MAX_RETRY_AFTER_SECONDS
# (the same ceiling as the exponential fallback): a 429 asking for more is
# not retried at all and reaches the caller after the current attempt.
# Both budgets - the attempts and the sleep - belong to the LOGICAL request
# (_RequestBudget), not to one redirect hop: httpx re-enters
# handle_async_request for every hop, and a backend that answered each hop
# with two in-budget 429s before redirecting on re-armed both budgets per
# hop (3 hops x 2 x 5 s = 30.3 s over 10 backend requests in the runtime
# probe; ~400 s at httpx's default 20 hops), so the ceiling was per
# response in name only. Now at most _MAX_ATTEMPTS - 1 retries and
# _MAX_RETRY_AFTER_SECONDS of sleep are spent across the whole chain, and
# the chain itself is at most _MAX_REDIRECT_HOPS redirects long.
_RETRYABLE_TRANSPORT_EXCEPTIONS = (httpx.ConnectError, httpx.TimeoutException)

_MAX_ATTEMPTS = 3
_MAX_RETRY_AFTER_SECONDS = 10.0
_DEFAULT_RETRY_WAIT = wait_exponential(multiplier=1, max=_MAX_RETRY_AFTER_SECONDS)
# Redirects followed per logical request, enforced by the transport (so no
# backend's own client construction can miss it) and mirrored as
# max_redirects on BaseBackend.client. httpx's default is 20; the backends
# this server talks to redirect at most once (an http->https upgrade or a
# trailing-slash canonicalisation), so five is already generous.
_MAX_REDIRECT_HOPS = 5


def _is_rate_limited(response: Any) -> bool:
    """True if a completed (non-exception) attempt was a 429 whose
    Retry-After (if any) fits the retry budget. A 429 asking for a longer
    wait is not retried; tenacity hands the real response back and the
    backend's own raise_for_status() surfaces it immediately."""
    if not (isinstance(response, httpx.Response) and response.status_code == 429):
        return False
    retry_after = _retry_after_seconds(response)
    return retry_after is None or retry_after <= _MAX_RETRY_AFTER_SECONDS


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse a Retry-After header into seconds to wait, if present/valid.

    Handles both the delta-seconds form ("Retry-After: 30") and the
    HTTP-date form ("Retry-After: Wed, 21 Oct 2026 07:28:00 GMT"), per
    RFC 9110 - a backend can send either. Returns None (not 0) when the
    header is absent or unparseable, so the caller can fall back to
    exponential backoff instead of retrying immediately.
    """
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    value = value.strip()
    # isascii() too: str.isdigit() is true for non-ASCII digits such as
    # U+00B2, which float() then rejects - a ValueError escaping the wait
    # strategy would surface out of tenacity as an unrelated crash.
    if value.isascii() and value.isdigit():
        return float(value)
    try:
        retry_time: datetime = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if retry_time.tzinfo is None:
        retry_time = retry_time.replace(tzinfo=UTC)
    return max((retry_time - datetime.now(UTC)).total_seconds(), 0.0)


def _wait_for_retry(retry_state: RetryCallState) -> float:
    """Respect a 429 response's Retry-After header when present, up to
    _MAX_RETRY_AFTER_SECONDS (belt and braces behind _is_rate_limited, which
    already refuses to retry an over-cap 429), otherwise fall back to the
    same exponential backoff used for connect/timeout retries."""
    outcome = retry_state.outcome
    if outcome is not None and not outcome.failed:
        response = outcome.result()
        if isinstance(response, httpx.Response):
            retry_after = _retry_after_seconds(response)
            if retry_after is not None:
                return min(retry_after, _MAX_RETRY_AFTER_SECONDS)
    return _DEFAULT_RETRY_WAIT(retry_state)


class _RetryingTransport(httpx.AsyncBaseTransport):
    """Wraps another async transport and retries only transport-level
    connection failures with exponential backoff, reraising the final
    exception if every attempt fails.
    """

    def __init__(
        self,
        wrapped: httpx.AsyncBaseTransport | None = None,
        slow_request_threshold_ms: float | None = None,
        allowed_host: str | None = None,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        total_deadline_seconds: float | None = None,
    ) -> None:
        """Wrap an underlying transport (defaults to a fresh AsyncHTTPTransport).

        Args:
            wrapped: Transport to delegate to
            slow_request_threshold_ms: If set, log a warning for any request
                that takes longer than this, independent of the configured
                log level
            allowed_host: If set, refuse (RedirectTargetBlockedError) any
                request whose host differs from this one, case-insensitively.
                Every backend's `client` sets it to the configured backend
                host, so a 3xx Location chosen by the backend can never steer
                the shared client elsewhere. Host-only on purpose: a same-host
                scheme/port change (an http->https upgrade) stays allowed.
            max_response_bytes: Ceiling on the DECODED size of any single
                response (initial or redirect hop). A larger declared
                Content-Length is refused before the body is read; a body
                that decodes past the cap raises ResponseTooLargeError
                mid-read (see _BoundedResponse).
            total_deadline_seconds: If set, the wall-clock budget for one
                LOGICAL request - connect, every retry attempt and sleep,
                every redirect hop and the whole body read - after which
                BackendDeadlineExceededError is raised wherever the request
                happens to be. Every backend's `client` sets it from
                request_deadline_seconds(timeout).
        """
        self._wrapped = wrapped if wrapped is not None else httpx.AsyncHTTPTransport()
        self._slow_request_threshold_ms = slow_request_threshold_ms
        self._allowed_host = allowed_host.lower() if allowed_host is not None else None
        self._max_response_bytes = max_response_bytes
        self._total_deadline_seconds = total_deadline_seconds

    def _budget_for(self, request: httpx.Request) -> _RequestBudget:
        """The logical request's budget: created on the first hop of a
        request (the clock starts here), found again on every retry (same
        Request) and redirect hop (same extensions, see
        _REQUEST_BUDGET_EXTENSION). No backend reuses a Request object
        across logical requests, so a stale budget is never inherited."""
        budget = request.extensions.get(_REQUEST_BUDGET_EXTENSION)
        if not isinstance(budget, _RequestBudget):
            deadline = (
                None
                if self._total_deadline_seconds is None
                else time.monotonic() + self._total_deadline_seconds
            )
            budget = _RequestBudget(deadline)
            request.extensions[_REQUEST_BUDGET_EXTENSION] = budget
        return budget

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Delegate to the wrapped transport, retrying connect/timeout
        failures and 429 rate-limit responses - with exponential backoff
        capped at 10 s, or a 429's own Retry-After header when present and
        no larger than that same 10 s ceiling - within ONE budget for the
        whole logical request: at most 3 attempts and 10 s of cumulative
        sleep across every redirect hop, at most 5 redirect hops, and one
        total deadline (total_deadline_seconds) covering the attempts, the
        sleeps, the hops and the body read. A 429 whose Retry-After exceeds
        the ceiling, or that the remaining budget cannot afford, is returned
        after the current attempt with no retry sleep - the backend may not
        park this process; past the deadline BackendDeadlineExceededError is
        raised instead of waiting on the backend any longer.
        """
        budget = self._budget_for(request)
        budget.hops += 1
        if budget.hops > _MAX_REDIRECT_HOPS + 1:
            # The client's own max_redirects (set to the same cap on
            # BaseBackend.client) normally fires first; this is the backstop
            # for a client built elsewhere with httpx's default of 20.
            raise httpx.TooManyRedirects(
                f"Refusing redirect hop {budget.hops - 1}: at most {_MAX_REDIRECT_HOPS} "
                "redirects are followed per backend request",
                request=request,
            )
        # httpx re-invokes handle_async_request on every redirect hop when
        # follow_redirects=True, so these checks - run before every attempt,
        # including redirect targets - apply to every backend regardless of
        # its own follow_redirects setting. Two layers: the metadata
        # denylist (canonicalising encoded IPv4 spellings, see security.py)
        # refuses the one class of host that is never legitimate even for
        # the operator's own URL, and the host pin refuses any host the
        # operator did not configure - which also covers DNS aliases of a
        # metadata address and arbitrary internal hosts that no denylist
        # could enumerate. A 307/308 is refused here before its body is
        # re-sent anywhere.
        if is_cloud_metadata_host(request.url.host):
            raise MetadataEndpointBlockedError(
                f"Refusing to connect to cloud metadata endpoint: {request.url.host}"
            )
        if (
            self._allowed_host is not None
            and (request.url.host or "").lower() != self._allowed_host
        ):
            raise RedirectTargetBlockedError(
                f"Refusing request to {request.url.host!r}: the configured backend host is "
                f"{self._allowed_host!r} and a backend response may not redirect elsewhere"
            )

        # A plain `async def` closure (rather than passing
        # self._wrapped.handle_async_request straight to tenacity) ensures
        # tenacity always awaits the call correctly - some transport wrappers
        # (e.g. VCR's cassette-recording patch used in this project's own
        # integration tests) expose handle_async_request as a *sync* function
        # that returns a coroutine, which tenacity's coroutine-callable
        # detection can miss, silently returning an unawaited coroutine.
        async def _send() -> httpx.Response:
            return await self._wrapped.handle_async_request(request)

        # tenacity evaluates `wait` (retry_state.upcoming_sleep) before
        # `stop`, so the stop decision can weigh the very sleep about to be
        # taken against the logical request's remaining budget; a retry the
        # budget cannot afford ends the loop with the real last outcome
        # (reraise=True) instead of a sleep. `before_sleep` runs only when a
        # retry WILL happen: it books the retry and its sleep against the
        # budget and releases the 429 response tenacity is about to discard
        # (its pooled connection would otherwise wait for garbage
        # collection). The final response, and a 429 that is not retried, go
        # back to the caller unread and stay open.
        def _stop_when_budget_exhausted(retry_state: RetryCallState) -> bool:
            return not budget.can_afford_retry(retry_state.upcoming_sleep)

        async def _spend_budget_before_sleep(retry_state: RetryCallState) -> None:
            budget.spend_retry(retry_state.upcoming_sleep)
            outcome = retry_state.outcome
            if outcome is not None and not outcome.failed:
                discarded = outcome.result()
                if isinstance(discarded, httpx.Response):
                    await discarded.aclose()

        retrying = AsyncRetrying(
            retry=retry_any(
                retry_if_exception_type(_RETRYABLE_TRANSPORT_EXCEPTIONS),
                retry_if_result(_is_rate_limited),
            ),
            wait=_wait_for_retry,
            stop=_stop_when_budget_exhausted,
            before_sleep=_spend_budget_before_sleep,
            reraise=True,
        )
        start = time.perf_counter()
        try:
            # The deadline scope wraps the attempts AND tenacity's sleeps
            # between them: the retry sleep runs here in the transport,
            # outside every httpx timeout, so this is the only bound on it.
            async with asyncio.timeout(budget.remaining_seconds()) as deadline_scope:
                response: httpx.Response = await retrying(_send)
        except TimeoutError:
            if not deadline_scope.expired():
                raise
            raise _deadline_exceeded() from None
        except RetryError as exc:
            # reraise=True only converts RetryError into the *original*
            # exception when the last attempt failed with one. A 429 is a
            # successful (non-failed) attempt that still triggered a retry,
            # so exhausting all attempts on a persistent 429 raises the bare
            # RetryError instead - confirmed empirically, not assumed - which
            # would otherwise discard the real response every caller
            # downstream expects (backends call response.raise_for_status()
            # themselves). Unwrap it back to that response here.
            if exc.last_attempt.failed:
                raise
            response = exc.last_attempt.result()
        # Content-Length is the WIRE size (pre-decoding), so this only ever
        # rejects a body that is oversized before inflation - but then it
        # does so without reading a byte of it. The decisive control is the
        # decoded-byte count in _BoundedResponse.aiter_bytes().
        declared_length = response.headers.get("Content-Length")
        if (
            declared_length is not None
            and declared_length.isascii()
            and declared_length.isdigit()
            and (
                len(declared_length) > _MAX_CONTENT_LENGTH_DIGITS
                or int(declared_length) > self._max_response_bytes
            )
        ):
            await response.aclose()
            shown = (
                declared_length
                if len(declared_length) <= _MAX_CONTENT_LENGTH_DIGITS
                else f"<{len(declared_length)} digits>"
            )
            raise ResponseTooLargeError(
                f"Backend response declared Content-Length {shown}, above the "
                f"{self._max_response_bytes}-byte BACKEND_MAX_RESPONSE_BYTES cap"
            )
        response = _BoundedResponse(response, self._max_response_bytes, budget)
        if self._slow_request_threshold_ms is not None:
            duration_ms = (time.perf_counter() - start) * 1000
            if duration_ms > self._slow_request_threshold_ms:
                # Query strings can carry trace IDs, filter values, or (for
                # backends that use query-param rather than header auth)
                # credentials, and a URL userinfo IS a credential - logging
                # scheme, host and path only is enough to identify which
                # endpoint was slow without exposing either.
                url_without_query = request.url.copy_with(query=None, username=None, password=None)
                logger.warning(
                    f"Slow backend request: {request.method} {url_without_query} took "
                    f"{duration_ms:.0f}ms (threshold: {self._slow_request_threshold_ms:.0f}ms)"
                )
        return response

    async def aclose(self) -> None:
        """Close the wrapped transport's connection pool."""
        await self._wrapped.aclose()


class BaseBackend(ABC):
    """Abstract interface for OpenTelemetry trace backends."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wraps each concrete subclass's own directly-defined cacheable
        methods with a cache-lookup wrapper, once, at class-creation time -
        requires zero edits to any concrete backend file. Only wraps
        methods a subclass defines itself (cls.__dict__), never a method
        inherited unchanged from a parent, which would otherwise be
        wrapped twice across a subclass hierarchy.

        search_traces/search_spans additionally get the query-limit clamp
        as the OUTER wrapper (clamp -> cache -> original): the cache key
        must be built from the clamped query. functools.wraps chains
        __wrapped__ through both layers.
        """
        super().__init_subclass__(**kwargs)
        for method_name in _CACHEABLE_METHODS:
            original = cls.__dict__.get(method_name)
            if original is None:
                continue
            wrapped = _wrap_with_cache(method_name, original)
            if method_name in _QUERY_LIMITED_METHODS:
                wrapped = _wrap_with_query_limit(method_name, wrapped)
            setattr(cls, method_name, wrapped)

    def __init__(self, url: str, api_key: str | None = None, timeout: float = 30.0):
        """Initialize backend with connection parameters.

        Args:
            url: Backend API URL
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        # Not a constructor parameter: several subclasses override __init__
        # with their own named params and call super().__init__(url, api_key,
        # timeout) positionally, so a caller (server.py's _create_backend)
        # sets this as a plain attribute after construction instead.
        self.slow_request_threshold_ms: float | None = None
        # Same post-construction pattern: server.py applies the operator's
        # BACKEND_MAX_RESPONSE_BYTES here, and `client` hands it to the
        # transport, which enforces it on every response.
        self.max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
        # Same post-construction pattern: server.py applies the operator's
        # MAX_TRACES_PER_QUERY here and _wrap_with_query_limit enforces it on
        # every search_traces/search_spans call, whichever tool or helper
        # built the query. None (the library default) means no clamp.
        self.max_query_limit: int | None = None
        self._query_cache: TTLCoalescingCache | None = None

    def configure_query_cache(
        self, ttl_seconds: float | None, max_entries: int = DEFAULT_QUERY_CACHE_MAX_ENTRIES
    ) -> None:
        """Mirrors slow_request_threshold_ms's own post-construction-
        attribute pattern above - set by the caller (server.py's
        _create_backend) after construction, since several subclasses
        override __init__ with their own named params. `max_entries` is
        the LRU bound on cached results (QUERY_CACHE_MAX_ENTRIES)."""
        self._query_cache = (
            TTLCoalescingCache(ttl_seconds, max_entries=max_entries)
            if ttl_seconds and ttl_seconds > 0
            else None
        )

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling.

        All backends share this client construction, so the retry transport
        applies automatically to every backend's requests.

        Returns:
            Reusable AsyncClient instance with automatic connection pooling
            and connect/timeout retry with exponential backoff
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.url,
                headers=self._create_headers(),
                timeout=self.timeout,
                follow_redirects=True,
                max_redirects=_MAX_REDIRECT_HOPS,
                transport=self._create_transport(),
            )
        return self._client

    def _create_transport(self) -> "_RetryingTransport":
        """The one _RetryingTransport construction every backend shares -
        the Datadog/New Relic/Honeycomb `client` overrides (which differ
        from the base one only in follow_redirects) call this too, so a
        transport-level control added here (host pin, response size cap,
        slow-request logging) can never be missed by one backend's own
        client."""
        return _RetryingTransport(
            slow_request_threshold_ms=self.slow_request_threshold_ms,
            allowed_host=httpx.URL(self.url).host,
            max_response_bytes=self.max_response_bytes,
            total_deadline_seconds=request_deadline_seconds(self.timeout),
        )

    @abstractmethod
    def _create_headers(self) -> dict[str, str]:
        """Create backend-specific HTTP headers.

        Returns:
            Dictionary of HTTP headers (e.g., Authorization, Content-Type)
        """
        pass

    @abstractmethod
    def get_supported_operators(self) -> set[FilterOperator]:
        """Get the set of filter operators that this backend natively supports.

        Operators not in this set will be applied via client-side filtering.

        Returns:
            Set of natively supported FilterOperator values
        """
        pass

    @abstractmethod
    async def search_traces(self, query: TraceQuery) -> list[TraceData]:
        """Search for traces matching the given query.

        Args:
            query: Trace query parameters

        Returns:
            List of matching traces with all spans

        Raises:
            Exception: If the backend query fails
        """
        pass

    @abstractmethod
    async def search_spans(self, query: SpanQuery) -> list[SpanData]:
        """Search for individual spans matching the given query.

        Args:
            query: Span query parameters

        Returns:
            List of matching spans (not grouped by trace)

        Raises:
            Exception: If the backend query fails
        """
        pass

    @abstractmethod
    async def get_trace(self, trace_id: str) -> TraceData:
        """Get a specific trace by ID.

        Args:
            trace_id: Trace identifier

        Returns:
            Complete trace data with all spans

        Raises:
            Exception: If trace not found or query fails
        """
        pass

    @abstractmethod
    async def list_services(self) -> list[str]:
        """List all available services.

        Returns:
            List of service names

        Raises:
            Exception: If query fails
        """
        pass

    async def list_services_uncached(self) -> list[str]:
        """list_services() with the query cache bypassed - a live request,
        whatever QUERY_CACHE_TTL_SECONDS is.

        The readiness probe (/ready, the Docker HEALTHCHECK) and doctor call
        this: a cached result must never make readiness report healthy
        after the backend has failed or rejected the credential. Invokes the
        concrete class's pre-wrap original through the ``__wrapped__``
        functools.wraps leaves on the cache wrapper, so no per-backend edit
        is needed and no backend can accidentally re-wrap it (not a
        _CACHEABLE_METHODS entry).

        Returns:
            List of service names, straight from the backend

        Raises:
            Exception: If the query fails
        """
        method = type(self).list_services
        original = getattr(method, "__wrapped__", method)
        services: list[str] = await original(self)
        return services

    @abstractmethod
    async def get_service_operations(self, service_name: str) -> list[str]:
        """Get all operations for a specific service.

        Args:
            service_name: Service name

        Returns:
            List of operation names

        Raises:
            Exception: If query fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> HealthCheckResponse:
        """Check backend health and connectivity.

        Returns:
            Health status information

        Raises:
            Exception: If backend is unreachable
        """
        pass

    async def __aenter__(self) -> "BaseBackend":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close HTTP client connections."""
        while self._client is not None:
            client_to_close = self._client
            await client_to_close.aclose()
            # httpx flips is_closed to True synchronously as the very
            # first line inside aclose(), before the real await below it -
            # a concurrent caller of the `client` property (a sync getter,
            # so it can't take a lock against this async method) can
            # observe that flag mid-close, build a replacement client, and
            # assign it to self._client while we're still awaiting. Only
            # clear self._client if it's still the exact object we just
            # closed; otherwise a replacement snuck in - loop and close
            # that one too, rather than silently overwriting the
            # reference and leaking an open client that never gets closed.
            if self._client is client_to_close:
                self._client = None
                break
        if self._query_cache is not None:
            await self._query_cache.clear()

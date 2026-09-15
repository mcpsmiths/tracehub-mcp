"""Tests for BaseBackend's shared HTTP client construction and its
transport-level connect/timeout retry with exponential backoff.

The retry transport wraps whatever transport it's given, so these tests
exercise it directly via httpx.MockTransport (a stateful call counter is
far simpler than a VCR cassette for a retry-then-succeed scenario), while
one test confirms every backend actually gets the retry transport through
the shared `client` property in base.py.
"""

import asyncio
import logging
import time

import httpx
import pytest

from opentelemetry_mcp.backends.base import _RetryingTransport
from opentelemetry_mcp.backends.jaeger import JaegerBackend


async def test_connect_error_twice_then_success_eventually_succeeds() -> None:
    """The first 2 attempts raise httpx.ConnectError; the 3rd (final
    allowed) attempt succeeds, so the retry transport must return that
    successful response rather than propagating the earlier failures."""
    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json={"ok": True})

    transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

    async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert call_count == 3


async def test_persistent_connect_error_still_raises() -> None:
    """A ConnectError on every one of the 3 allowed attempts must not be
    swallowed - the final exception is reraised, not hidden behind a
    fabricated response."""
    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.ConnectError("connection refused")

    transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

    async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
        with pytest.raises(httpx.ConnectError):
            await client.get("/health")

    assert call_count == 3


async def test_timeout_exception_is_retried_like_connect_error() -> None:
    """httpx.TimeoutException is the other retryable transport failure -
    verify it follows the same retry-then-succeed path as ConnectError."""
    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise httpx.TimeoutException("timed out")
        return httpx.Response(200)

    transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

    async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert call_count == 2


async def test_http_error_response_is_not_retried() -> None:
    """A 500 response is a *successful* transport exchange (a response body
    came back), not a ConnectError/TimeoutException - the retry transport
    must pass it straight through on the first attempt, never retrying
    based on status code alone."""
    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(500)

    transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

    async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
        response = await client.get("/health")

    assert response.status_code == 500
    assert call_count == 1


def test_backend_client_property_uses_the_retrying_transport() -> None:
    """Every backend shares BaseBackend.client, so the retry transport must
    be wired in there - not duplicated per backend."""
    backend = JaegerBackend(url="http://localhost:16686")

    client = backend.client

    assert isinstance(client._transport, _RetryingTransport)


def test_backend_client_forwards_slow_request_threshold_to_transport() -> None:
    """slow_request_threshold_ms is not a constructor param (several
    backend subclasses override __init__ without **kwargs forwarding) -
    it is set as a plain attribute and must reach the transport via the
    client property."""
    backend = JaegerBackend(url="http://localhost:16686")
    backend.slow_request_threshold_ms = 500.0

    client = backend.client

    assert isinstance(client._transport, _RetryingTransport)
    assert client._transport._slow_request_threshold_ms == 500.0


class TestSlowRequestLogging:
    """_RetryingTransport logs a warning when a request exceeds the
    configured threshold, independent of the request's own success/failure."""

    async def test_request_slower_than_threshold_logs_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.05)
            return httpx.Response(200)

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), slow_request_threshold_ms=10.0
        )

        with caplog.at_level(logging.WARNING):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://backend.test"
            ) as client:
                await client.get("/health")

        assert any("Slow backend request" in r.message for r in caplog.records)

    async def test_request_faster_than_threshold_does_not_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200)

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), slow_request_threshold_ms=10_000.0
        )

        with caplog.at_level(logging.WARNING):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://backend.test"
            ) as client:
                await client.get("/health")

        assert caplog.records == []

    async def test_threshold_unset_never_logs_regardless_of_duration(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.05)
            return httpx.Response(200)

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        with caplog.at_level(logging.WARNING):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://backend.test"
            ) as client:
                await client.get("/health")

        assert caplog.records == []

    async def test_slow_error_response_still_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """A slow request that returns an HTTP error status is still a
        transport-level success (a response came back), so slow-request
        logging must fire the same as for a 200."""

        async def handler(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.05)
            return httpx.Response(500)

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), slow_request_threshold_ms=10.0
        )

        with caplog.at_level(logging.WARNING):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://backend.test"
            ) as client:
                await client.get("/health")

        assert any("Slow backend request" in r.message for r in caplog.records)

    async def test_query_string_is_stripped_from_the_logged_url(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Found by a production-audit pass: query strings can carry trace
        IDs, filter values, or (for backends using query-param rather than
        header auth) credentials - the log message must never include them."""

        async def handler(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.05)
            return httpx.Response(200)

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), slow_request_threshold_ms=10.0
        )

        with caplog.at_level(logging.WARNING):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://backend.test"
            ) as client:
                await client.get("/search", params={"api_key": "super-secret-value"})

        messages = [r.message for r in caplog.records if "Slow backend request" in r.message]
        assert len(messages) == 1
        assert "/search" in messages[0]
        assert "super-secret-value" not in messages[0]
        assert "api_key" not in messages[0]


class TestRetryOn429:
    """A 429 is a *result*, not an exception, so it needs its own coverage
    distinct from the connect/timeout exception-retry tests above - and a
    persistent 429 must still hand back the real response on exhaustion,
    not the bare tenacity.RetryError this project found it actually raises
    (see _RetryingTransport.handle_async_request's own comment)."""

    async def test_429_then_success_eventually_succeeds(self) -> None:
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return httpx.Response(429)
            return httpx.Response(200, json={"ok": True})

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            response = await client.get("/search")

        assert response.status_code == 200
        assert response.json() == {"ok": True}
        assert call_count == 3

    async def test_persistent_429_returns_the_real_response_not_a_retry_error(self) -> None:
        """Exhausting all 3 attempts on a still-429 response must hand back
        that actual httpx.Response - never tenacity.RetryError, which every
        caller downstream (backends calling response.raise_for_status())
        would have no idea how to handle."""
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(429)

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            response = await client.get("/search")

        assert response.status_code == 429
        assert call_count == 3

    async def test_retry_after_delta_seconds_header_is_respected(self) -> None:
        """A Retry-After: <seconds> header should be honored instead of the
        default exponential backoff curve. Uses "0" (RFC 9110 delta-seconds
        is a non-negative integer - fractional values like "0.2" aren't
        valid and correctly fall back to exponential backoff instead)."""
        call_count = 0
        wait_times: list[float] = []
        last_call_time: float | None = None

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count, last_call_time
            now = time.monotonic()
            if last_call_time is not None:
                wait_times.append(now - last_call_time)
            last_call_time = now
            call_count += 1
            if call_count < 2:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(200)

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            response = await client.get("/search")

        assert response.status_code == 200
        assert call_count == 2
        # Retry-After: 0 is well below wait_exponential's ~1s first step,
        # so if it were ignored this wait would be noticeably longer.
        assert wait_times[0] < 0.5

    async def test_missing_retry_after_falls_back_to_exponential_backoff(self) -> None:
        """No Retry-After header at all - must still retry via the same
        exponential curve used for connect/timeout failures, not skip
        waiting entirely."""
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return httpx.Response(429)
            return httpx.Response(200)

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        start = time.monotonic()
        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            response = await client.get("/search")
        elapsed = time.monotonic() - start

        assert response.status_code == 200
        assert call_count == 2
        # wait_exponential(multiplier=1, exp_base=2)'s first wait is
        # 2**(1-1) = 1s (the exponent starts at 0, not 1).
        assert elapsed >= 0.8

    async def test_non_429_error_response_is_still_not_retried(self) -> None:
        """A genuine 4xx/5xx (not 429) must keep passing straight through on
        the first attempt - this feature must not widen retry to all error
        statuses, only the rate-limit one."""
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(503)

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            response = await client.get("/search")

        assert response.status_code == 503
        assert call_count == 1

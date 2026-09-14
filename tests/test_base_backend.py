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

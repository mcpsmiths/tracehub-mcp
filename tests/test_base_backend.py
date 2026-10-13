"""Tests for BaseBackend's shared HTTP client construction and its
transport-level connect/timeout retry with exponential backoff.

The retry transport wraps whatever transport it's given, so these tests
exercise it directly via httpx.MockTransport (a stateful call counter is
far simpler than a VCR cassette for a retry-then-succeed scenario), while
one test confirms every backend actually gets the retry transport through
the shared `client` property in base.py.
"""

import asyncio
import gzip
import json
import logging
import os
import time
import tracemalloc
import zlib
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from tenacity import AsyncRetrying, RetryCallState

from opentelemetry_mcp.backends import base as base_module
from opentelemetry_mcp.backends.base import (
    _MAX_REDIRECT_HOPS,
    _REQUEST_BUDGET_EXTENSION,
    DEFAULT_MAX_RESPONSE_BYTES,
    BackendDeadlineExceededError,
    BaseBackend,
    MetadataEndpointBlockedError,
    RedirectTargetBlockedError,
    ResponseTooLargeError,
    UnsupportedContentEncodingError,
    _is_rate_limited,
    _RequestBudget,
    _retry_after_seconds,
    _RetryingTransport,
    _wait_for_retry,
    request_deadline_seconds,
)
from opentelemetry_mcp.backends.datadog import DatadogBackend
from opentelemetry_mcp.backends.honeycomb import HoneycombBackend
from opentelemetry_mcp.backends.jaeger import JaegerBackend
from opentelemetry_mcp.backends.newrelic import NewRelicBackend


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


async def test_close_does_not_leak_a_client_that_appears_during_its_own_await() -> None:
    """httpx flips is_closed=True synchronously as the very first line
    inside aclose(), before the real await that follows it - a concurrent
    caller of the `client` property (sync, so it can't take a lock against
    this async close()) can observe that flag mid-close and build a
    replacement client while close() is still awaiting. Before the fix,
    close() unconditionally set self._client = None afterward, silently
    overwriting (and leaking - never .aclose()'d) that replacement."""
    backend = JaegerBackend(url="http://localhost:16686")
    client_a = backend.client

    replacement_client = httpx.AsyncClient()
    closed: list[str] = []

    async def fake_aclose_a() -> None:
        # Simulates the concurrent property access: by the time this
        # await point is reached, a different task has already swapped in
        # a replacement.
        backend._client = replacement_client
        closed.append("a")

    client_a.aclose = fake_aclose_a  # type: ignore[method-assign]

    real_replacement_aclose = replacement_client.aclose

    async def fake_aclose_b() -> None:
        closed.append("b")
        await real_replacement_aclose()

    replacement_client.aclose = fake_aclose_b  # type: ignore[method-assign]

    await backend.close()

    assert closed == ["a", "b"]  # the replacement must be closed too, not orphaned
    assert backend._client is None


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

    async def test_userinfo_is_stripped_from_the_logged_url(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A credential a bypassed config left in BACKEND_URL's userinfo
        rides along on every request URL; the slow-request WARNING must
        strip it the same way it strips the query string."""

        async def handler(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.05)
            return httpx.Response(200)

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), slow_request_threshold_ms=10.0
        )

        with caplog.at_level(logging.WARNING):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://u:VFY-PW-slow@backend.test"
            ) as client:
                await client.get("/search")

        messages = [r.message for r in caplog.records if "Slow backend request" in r.message]
        assert len(messages) == 1
        assert "/search" in messages[0]
        assert "backend.test" in messages[0]
        assert "VFY-PW-slow" not in messages[0]
        assert "u:" not in messages[0]


class TestMetadataEndpointBlocking:
    """A compromised or misconfigured backend must never be able to steer a
    request (initial or mid-redirect) toward a cloud instance-metadata
    endpoint - see security.py."""

    async def test_request_to_metadata_ip_raises_without_retrying(self) -> None:
        handler_calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal handler_calls
            handler_calls += 1
            return httpx.Response(200)

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(MetadataEndpointBlockedError):
                await client.get("http://169.254.169.254/latest/meta-data/")

        assert handler_calls == 0

    async def test_redirect_to_metadata_ip_is_blocked(self) -> None:
        """A backend that 302-redirects toward a metadata address must be
        blocked mid-chain, not just on the initial request - this needs a
        real AsyncClient with follow_redirects=True, not just the bare
        transport, to actually exercise the redirect-hop code path."""
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if str(request.url) == "http://backend.test/search":
                return httpx.Response(
                    302, headers={"Location": "http://169.254.169.254/latest/meta-data/"}
                )
            return httpx.Response(200)

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        async with httpx.AsyncClient(
            transport=transport, base_url="http://backend.test", follow_redirects=True
        ) as client:
            with pytest.raises(MetadataEndpointBlockedError):
                await client.get("/search")

        assert call_count == 1

    async def test_normal_request_to_non_metadata_host_is_unaffected(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            response = await client.get("/search")

        assert response.status_code == 200
        assert response.json() == {"ok": True}


class TestRedirectHostPin:
    """A backend response (3xx Location) must never steer the shared client
    to a host the operator did not configure. The metadata denylist stays
    as a second layer, but is_cloud_metadata_host alone is bypassable by
    non-dotted IPv4 spellings, DNS aliases and unlisted metadata hosts -
    the host pin on _RetryingTransport closes all of those at once."""

    @pytest.mark.parametrize(
        "location",
        [
            "http://2852039166/latest/meta-data/",
            "http://0xa9fea9fe/latest/meta-data/",
            "http://169.254.43518/latest/meta-data/",
            "http://0xa9.0xfe.0xa9.0xfe/latest/meta-data/",
            "http://168.63.129.16/machine?comp=goalstate",
            "http://metadata.attacker.example/latest/meta-data/",
        ],
    )
    async def test_redirect_to_encoded_or_aliased_metadata_host_is_blocked(
        self, location: str
    ) -> None:
        handler_calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal handler_calls
            handler_calls += 1
            if request.url.host == "backend.test":
                return httpx.Response(302, headers={"Location": location})
            raise AssertionError("the redirect hop must never reach the wrapped transport")

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), allowed_host="backend.test"
        )

        async with httpx.AsyncClient(
            transport=transport, base_url="http://backend.test", follow_redirects=True
        ) as client:
            with pytest.raises((MetadataEndpointBlockedError, RedirectTargetBlockedError)):
                await client.get("/search")

        assert handler_calls == 1

    async def test_same_host_redirect_is_still_followed(self) -> None:
        """Host-only pin: a same-host scheme/port change (e.g. an http->https
        upgrade) is the operator's own backend talking, so it stays allowed."""
        handler_calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal handler_calls
            handler_calls += 1
            if str(request.url) == "http://backend.test/search":
                return httpx.Response(302, headers={"Location": "https://backend.test/other"})
            return httpx.Response(200, json={"ok": True})

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), allowed_host="backend.test"
        )

        async with httpx.AsyncClient(
            transport=transport, base_url="http://backend.test", follow_redirects=True
        ) as client:
            response = await client.get("/search")

        assert response.status_code == 200
        assert response.json() == {"ok": True}
        assert handler_calls == 2

    async def test_host_comparison_is_case_insensitive(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/search":
                return httpx.Response(302, headers={"Location": "http://Backend.TEST/other"})
            return httpx.Response(200)

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), allowed_host="backend.test"
        )

        async with httpx.AsyncClient(
            transport=transport, base_url="http://backend.test", follow_redirects=True
        ) as client:
            response = await client.get("/search")

        assert response.status_code == 200

    async def test_307_cross_host_redirect_is_blocked_before_body_resend(self) -> None:
        """A 307/308 makes httpx re-send the original method AND body to the
        new host; the hop must be refused before the wrapped transport ever
        sees it, so the body is never sent off-host."""
        bodies_seen: list[tuple[str, bytes]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            bodies_seen.append((request.url.host or "", request.content))
            if request.url.host == "backend.test":
                return httpx.Response(307, headers={"Location": "http://other.test/sink"})
            return httpx.Response(200)

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), allowed_host="backend.test"
        )

        async with httpx.AsyncClient(
            transport=transport, base_url="http://backend.test", follow_redirects=True
        ) as client:
            with pytest.raises(RedirectTargetBlockedError):
                await client.post("/search", content=b'{"q": "secret-query"}')

        assert bodies_seen == [("backend.test", b'{"q": "secret-query"}')]

    async def test_blocked_hop_is_not_retried(self) -> None:
        """RedirectTargetBlockedError is deliberately not a retryable
        transport exception - re-attempting the hop would be pointless."""
        handler_calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal handler_calls
            handler_calls += 1
            return httpx.Response(200)

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), allowed_host="backend.test"
        )

        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(RedirectTargetBlockedError):
                await client.get("http://other.test/anything")

        assert handler_calls == 0

    async def test_transport_without_allowed_host_keeps_following_cross_host_redirects(
        self,
    ) -> None:
        """Test-convenience shape: a transport built with no allowed_host (as
        every pre-existing test does) must behave exactly as before - the
        pin is opt-in at construction, and BaseBackend.client is what opts
        in for production (see test_backend_clients_pin_the_configured_host)."""
        hosts_seen: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            hosts_seen.append(request.url.host or "")
            if request.url.host == "backend.test":
                return httpx.Response(302, headers={"Location": "http://other.test/sink"})
            return httpx.Response(200)

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        async with httpx.AsyncClient(
            transport=transport, base_url="http://backend.test", follow_redirects=True
        ) as client:
            response = await client.get("/search")

        assert response.status_code == 200
        assert hosts_seen == ["backend.test", "other.test"]

    async def test_cross_origin_hop_carries_no_authorization_header(self) -> None:
        """Pins the trust assumption the audit's severity rests on (and the
        Sentry backend's docstring relies on): httpx strips Authorization on
        a cross-origin hop, so an unpinned transport leaks at most a blind
        request, never the bearer credential. Uses an unpinned transport on
        purpose - with the pin the hop never happens at all."""
        auth_seen: dict[str, str | None] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            auth_seen[request.url.host or ""] = request.headers.get("Authorization")
            if request.url.host == "backend.test":
                return httpx.Response(302, headers={"Location": "http://other.test/sink"})
            return httpx.Response(200)

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://backend.test",
            follow_redirects=True,
            headers={"Authorization": "Bearer not-a-real-token"},
        ) as client:
            await client.get("/search")

        assert auth_seen["backend.test"] == "Bearer not-a-real-token"
        assert auth_seen["other.test"] is None


@pytest.mark.parametrize(
    "backend",
    [
        JaegerBackend(url="https://Backend.Example:16686"),
        DatadogBackend(url="https://Backend.Example", api_key="k", app_key="a"),
        NewRelicBackend(url="https://Backend.Example/graphql", api_key="k", account_id="1"),
        HoneycombBackend(url="https://Backend.Example", api_key="k", dataset="d"),
    ],
    ids=["base", "datadog", "newrelic", "honeycomb"],
)
def test_backend_clients_pin_the_configured_host(backend: BaseBackend) -> None:
    """BaseBackend.client and the three vendor overrides (which only differ
    in follow_redirects) must all hand the configured host to the transport,
    lower-cased, so the pin is never bypassed by one backend's own client."""
    client = backend.client

    assert isinstance(client._transport, _RetryingTransport)
    assert client._transport._allowed_host == "backend.example"


_ONE_MIB = 1024 * 1024


def _gzip_json_body(pad_bytes: int) -> bytes:
    return gzip.compress(json.dumps({"data": ["svc-a"], "pad": "a" * pad_bytes}).encode())


def _gzip_response(gz: bytes) -> httpx.Response:
    """stream= (not content=) so decoding happens on the client's read path,
    exactly as in production - httpx pre-reads content= at construction."""
    return httpx.Response(
        200,
        headers={"Content-Encoding": "gzip", "Content-Type": "application/json"},
        stream=httpx.ByteStream(gz),
    )


class _RecordingStream(httpx.AsyncByteStream):
    """A body stream that records whether anyone ever iterated or closed it."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.reads = 0
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.reads += 1
        yield self._data

    async def aclose(self) -> None:
        self.closed = True


class TestBoundedResponse:
    """A backend (or redirect target) must never be able to materialize an
    unbounded decoded body in the shared server process: a ~1 KB gzip
    member inflates ~1000x on the client's read path, before any backend
    code sees the response. The bound lives in the transport, the one point
    every httpx backend shares, and counts DECODED bytes."""

    async def test_gzip_body_over_cap_is_rejected_before_parse(self) -> None:
        gz = _gzip_json_body(2 * _ONE_MIB)
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _gzip_response(gz)

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), max_response_bytes=_ONE_MIB
        )

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            with pytest.raises(ResponseTooLargeError, match="BACKEND_MAX_RESPONSE_BYTES"):
                await client.get("/api/services")

        # Not retryable: the same backend would send the same body again.
        assert call_count == 1
        assert len(gz) < 16 * 1024  # the wire size really was tiny

    async def test_gzip_body_under_cap_still_parses(self) -> None:
        gz = _gzip_json_body(64 * 1024)

        async def handler(request: httpx.Request) -> httpx.Response:
            return _gzip_response(gz)

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), max_response_bytes=_ONE_MIB
        )

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            response = await client.get("/api/services")

        assert response.status_code == 200
        assert response.json()["data"] == ["svc-a"]
        assert len(response.json()["pad"]) == 64 * 1024

    async def test_declared_content_length_over_cap_is_rejected_early(self) -> None:
        """Content-Length is the WIRE size, so it can only ever reject a body
        that is oversized before decoding - but when it does, the body
        stream must never be read at all, and the connection released."""
        stream = _RecordingStream(b"x" * 16)

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"Content-Length": str(_ONE_MIB + 1)}, stream=stream)

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), max_response_bytes=_ONE_MIB
        )

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            with pytest.raises(ResponseTooLargeError, match="Content-Length"):
                await client.get("/api/services")

        assert stream.reads == 0
        assert stream.closed is True

    async def test_absurdly_long_content_length_is_rejected_without_parsing(self) -> None:
        """int() refuses a digit string longer than
        sys.int_info.str_digits_check_threshold (4300) with a bare
        ValueError, which would escape as an unrelated interpreter message
        and skip the aclose(). A run of digits longer than any cap could
        need is oversized without being parsed, and the message must not
        copy those digits into the tool-error channel."""
        stream = _RecordingStream(b"x" * 16)

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"Content-Length": "9" * 5000}, stream=stream)

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), max_response_bytes=_ONE_MIB
        )

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            with pytest.raises(ResponseTooLargeError, match="Content-Length") as exc_info:
                await client.get("/api/services")

        assert "5000 digits" in str(exc_info.value)
        assert "9" * 50 not in str(exc_info.value)
        assert stream.reads == 0
        assert stream.closed is True

    async def test_non_numeric_content_length_is_ignored_and_body_counted(self) -> None:
        gz = _gzip_json_body(2 * _ONE_MIB)

        async def handler(request: httpx.Request) -> httpx.Response:
            response = _gzip_response(gz)
            response.headers["Content-Length"] = "not-a-number"
            return response

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), max_response_bytes=_ONE_MIB
        )

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            with pytest.raises(ResponseTooLargeError):
                await client.get("/api/services")

    async def test_redirect_target_body_is_bounded(self) -> None:
        gz = _gzip_json_body(2 * _ONE_MIB)

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/services":
                return httpx.Response(302, headers={"Location": "http://backend.test/bomb"})
            return _gzip_response(gz)

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler),
            max_response_bytes=_ONE_MIB,
            allowed_host="backend.test",
        )

        async with httpx.AsyncClient(
            transport=transport, base_url="http://backend.test", follow_redirects=True
        ) as client:
            with pytest.raises(ResponseTooLargeError):
                await client.get("/api/services")

    async def test_identity_body_exactly_at_cap_is_allowed(self) -> None:
        body = b"a" * _ONE_MIB

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=httpx.ByteStream(body))

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), max_response_bytes=_ONE_MIB
        )

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            response = await client.get("/blob")

        assert len(response.content) == _ONE_MIB

    async def test_jaeger_list_services_respects_max_response_bytes(self) -> None:
        """End to end through a real backend method: the production
        transport construction (BaseBackend._create_transport) with only the
        innermost transport swapped for a mock."""
        gz_big = _gzip_json_body(2 * _ONE_MIB)
        gz_small = _gzip_json_body(1024)
        serve_big = True

        async def handler(request: httpx.Request) -> httpx.Response:
            return _gzip_response(gz_big if serve_big else gz_small)

        backend = JaegerBackend(url="http://backend.test")
        backend.max_response_bytes = _ONE_MIB
        transport = backend._create_transport()
        transport._wrapped = httpx.MockTransport(handler)
        backend._client = httpx.AsyncClient(base_url=backend.url, transport=transport)
        try:
            with pytest.raises(ResponseTooLargeError):
                await backend.list_services()
            serve_big = False
            assert await backend.list_services() == ["svc-a"]
        finally:
            await backend.close()

    def test_transport_default_cap_is_the_module_default(self) -> None:
        transport = _RetryingTransport(wrapped=httpx.MockTransport(lambda r: httpx.Response(200)))

        assert transport._max_response_bytes == DEFAULT_MAX_RESPONSE_BYTES
        assert DEFAULT_MAX_RESPONSE_BYTES == 64 * 1024 * 1024


@pytest.mark.parametrize(
    "backend",
    [
        JaegerBackend(url="https://backend.example:16686"),
        DatadogBackend(url="https://backend.example", api_key="k", app_key="a"),
        NewRelicBackend(url="https://backend.example/graphql", api_key="k", account_id="1"),
        HoneycombBackend(url="https://backend.example", api_key="k", dataset="d"),
    ],
    ids=["base", "datadog", "newrelic", "honeycomb"],
)
def test_backend_clients_forward_max_response_bytes_to_transport(backend: BaseBackend) -> None:
    """max_response_bytes follows slow_request_threshold_ms's own
    post-construction-attribute pattern and must reach the transport through
    every backend's client, vendor overrides included."""
    assert backend.max_response_bytes == DEFAULT_MAX_RESPONSE_BYTES
    backend.max_response_bytes = 4096

    client = backend.client

    assert isinstance(client._transport, _RetryingTransport)
    assert client._transport._max_response_bytes == 4096


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


class TestRetryAfterCeiling:
    """A backend-controlled Retry-After must never park a tool call (and,
    with QUERY_CACHE_TTL_SECONDS, every coalesced identical query) beyond
    the retry budget: the tenacity sleep runs inside the transport, outside
    the per-attempt httpx timeout, so the only bound is the one here."""

    async def test_retry_after_above_cap_returns_429_without_sleeping(self) -> None:
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(429, headers={"Retry-After": "100000"}, text="slow down")

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        start = time.monotonic()
        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            # wait_for: on the unfixed code this parks for 100000 s, so bound
            # the test itself rather than hanging the suite.
            response = await asyncio.wait_for(client.get("/search"), timeout=5.0)
        elapsed = time.monotonic() - start

        # The real response reaches the caller (backends raise_for_status()
        # on it themselves), unread-but-readable, after a single attempt.
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "100000"
        assert response.text == "slow down"
        assert call_count == 1
        assert elapsed < 1.0

    async def test_retry_after_far_future_http_date_is_not_honoured(self) -> None:
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(429, headers={"Retry-After": "Fri, 31 Dec 2099 23:59:59 GMT"})

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        start = time.monotonic()
        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            response = await asyncio.wait_for(client.get("/search"), timeout=5.0)
        elapsed = time.monotonic() - start

        assert response.status_code == 429
        assert call_count == 1
        assert elapsed < 1.0

    async def test_retry_after_at_or_below_cap_is_still_honoured(self) -> None:
        """Keeps the existing Retry-After semantics for well-behaved
        backends: a small delta is still a retry, not an immediate 429."""
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(200, json={"ok": True})

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            response = await client.get("/search")

        assert response.status_code == 200
        assert call_count == 2

    @pytest.mark.parametrize(
        ("retry_after", "retryable"),
        [(None, True), ("0", True), ("10", True), ("11", False), ("100000", False)],
    )
    def test_is_rate_limited_refuses_an_over_cap_429(
        self, retry_after: str | None, retryable: bool
    ) -> None:
        headers = {} if retry_after is None else {"Retry-After": retry_after}

        assert _is_rate_limited(httpx.Response(429, headers=headers)) is retryable

    def test_is_rate_limited_is_false_for_other_statuses_and_non_responses(self) -> None:
        assert _is_rate_limited(httpx.Response(200)) is False
        assert _is_rate_limited(httpx.Response(503, headers={"Retry-After": "1"})) is False
        assert _is_rate_limited(None) is False

    def test_wait_for_retry_never_exceeds_the_ceiling(self) -> None:
        """Belt and braces behind _is_rate_limited: even if an over-cap 429
        reached the wait strategy, the sleep is clamped to the budget."""
        state = RetryCallState(AsyncRetrying(), None, (), {})
        state.set_result(httpx.Response(429, headers={"Retry-After": "100000"}))

        assert _wait_for_retry(state) <= 10.0

    def test_wait_for_retry_honours_a_small_retry_after(self) -> None:
        state = RetryCallState(AsyncRetrying(), None, (), {})
        state.set_result(httpx.Response(429, headers={"Retry-After": "3"}))

        assert _wait_for_retry(state) == 3.0

    def test_non_ascii_digits_fall_back_to_backoff_instead_of_raising(self) -> None:
        """U+00B2 (superscript two) passes str.isdigit() but float() rejects
        it - a ValueError escaping the wait strategy would surface out of
        tenacity as an unrelated crash. Sent as a latin-1 header byte, as it
        arrives off the wire."""
        response = httpx.Response(429, headers=[(b"Retry-After", b"\xb2")])

        assert response.headers["Retry-After"] == "\u00b2"
        assert _retry_after_seconds(response) is None

    async def test_retried_429_is_closed_before_the_next_attempt(self) -> None:
        """Hygiene folded in from the audit: a 429 tenacity is about to
        discard used to rely on garbage collection to release its pooled
        connection. It must be closed deterministically - and only when a
        retry will actually happen (the final response stays open for the
        caller to read)."""
        streams: list[_RecordingStream] = []
        closed_before_attempt: list[list[bool]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            closed_before_attempt.append([s.closed for s in streams])
            if len(streams) < 2:
                stream = _RecordingStream(b"rate limited")
                streams.append(stream)
                return httpx.Response(429, headers={"Retry-After": "0"}, stream=stream)
            return httpx.Response(200, json={"ok": True})

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            response = await client.get("/search")

        assert response.status_code == 200
        # attempt 1 saw nothing yet; attempt 2 saw the first 429 closed;
        # attempt 3 saw both earlier 429s closed.
        assert closed_before_attempt == [[], [True], [True, True]]

    async def test_final_429_is_handed_to_the_caller_unclosed(self) -> None:
        final_stream = _RecordingStream(b"still rate limited")
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(429, headers={"Retry-After": "0"}, stream=final_stream)

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            response = await client.get("/search")

        assert response.status_code == 429
        assert call_count == 3
        # httpx read the body for the caller: the stream was iterated once,
        # i.e. the transport did not close it before handing it back.
        assert final_stream.reads == 1
        assert response.text == "still rate limited"


class _TrickleStream(httpx.AsyncByteStream):
    """A body that never ends: one byte, a pause shorter than any read
    timeout, another byte, forever - the shape the runtime probe used to
    park get_trace for 302 s (each socket read completes well inside
    BACKEND_TIMEOUT, so httpx's per-operation timeouts never fire and the
    decoded-size cap is never reached)."""

    def __init__(self, interval_seconds: float) -> None:
        self._interval_seconds = interval_seconds
        self.bytes_sent = 0
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while True:
            self.bytes_sent += 1
            yield b"x"
            await asyncio.sleep(self._interval_seconds)

    async def aclose(self) -> None:
        self.closed = True


class TestRequestDeadline:
    """One wall-clock deadline per LOGICAL backend request - connect, every
    retry attempt and its sleep, every redirect hop and the whole body read.
    httpx's timeouts are per socket operation, so without it a backend
    that keeps each operation just under BACKEND_TIMEOUT holds a tool call
    (and, with the query cache, every coalesced caller) for as long as it
    likes; the probe measured 302.5 s until the backend was killed."""

    async def test_trickling_body_fails_at_the_deadline_not_after_minutes(self) -> None:
        stream = _TrickleStream(interval_seconds=0.2)
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, stream=stream)

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), total_deadline_seconds=1.0
        )

        start = time.monotonic()
        async with httpx.AsyncClient(
            transport=transport, base_url="http://backend.test", timeout=5.0
        ) as client:
            with pytest.raises(BackendDeadlineExceededError, match="deadline"):
                # wait_for: on the unfixed code this never returns, so bound
                # the test itself rather than hanging the suite.
                await asyncio.wait_for(client.get("/api/traces/x"), timeout=10.0)
        elapsed = time.monotonic() - start

        assert 0.9 <= elapsed < 2.0
        # Not retryable: the deadline is for the whole logical request, so
        # a second attempt could never fit inside it.
        assert call_count == 1
        assert stream.closed is True
        assert 4 <= stream.bytes_sent <= 7

    async def test_deadline_error_is_not_a_retryable_timeout(self) -> None:
        """A subclass of httpx.TimeoutException would be matched by the
        connect/timeout retry predicate and re-sent; the deadline error must
        be a sibling, like ResponseTooLargeError."""
        error = BackendDeadlineExceededError("x")

        assert isinstance(error, httpx.TransportError)
        assert not isinstance(error, httpx.TimeoutException)

    async def test_normal_response_under_the_deadline_is_unaffected(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), total_deadline_seconds=5.0
        )

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"ok": True}

    async def test_slow_connect_phase_is_cut_at_the_deadline(self) -> None:
        """The deadline also covers the time before any header arrives."""

        async def handler(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(10.0)
            return httpx.Response(200)

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), total_deadline_seconds=0.5
        )

        start = time.monotonic()
        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            with pytest.raises(BackendDeadlineExceededError):
                await asyncio.wait_for(client.get("/health"), timeout=10.0)
        elapsed = time.monotonic() - start

        assert 0.4 <= elapsed < 1.5

    async def test_retry_sleeps_count_against_the_same_deadline(self) -> None:
        """Three connect failures would otherwise cost 1 s + 2 s of backoff
        on top of the attempts. The 1 s backoff fits the 1.5 s deadline and
        is taken; the 2 s one cannot fit, so instead of sleeping up to the
        deadline only to fail with the deadline error, the retry is not
        attempted and the backend's real last outcome - the ConnectError -
        reaches the caller at ~1 s."""
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectError("connection refused")

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), total_deadline_seconds=1.5
        )

        start = time.monotonic()
        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            with pytest.raises(httpx.ConnectError):
                await asyncio.wait_for(client.get("/health"), timeout=10.0)
        elapsed = time.monotonic() - start

        assert 0.9 <= elapsed < 1.4
        assert call_count == 2  # attempt 1, 1 s backoff, attempt 2; the 2 s backoff is refused

    async def test_redirect_hops_share_one_deadline_carried_in_request_extensions(
        self,
    ) -> None:
        """httpx builds a fresh Request per redirect hop, so the deadline
        cannot live on the Request object itself. It travels in
        request.extensions, which _build_redirect_request hands to every
        hop; this test proves BOTH that httpx (the pinned version) does so -
        every hop sees the very same budget object - and that the chain as
        a whole is cut at the deadline rather than each hop getting a fresh
        one (three 0.4 s hops under a 1.0 s deadline fail on the third)."""
        budgets_seen: list[object] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            budgets_seen.append(request.extensions.get(_REQUEST_BUDGET_EXTENSION))
            await asyncio.sleep(0.4)
            if request.url.path == "/a":
                return httpx.Response(302, headers={"Location": "http://backend.test/b"})
            if request.url.path == "/b":
                return httpx.Response(302, headers={"Location": "http://backend.test/c"})
            return httpx.Response(200, json={"ok": True})

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler),
            allowed_host="backend.test",
            total_deadline_seconds=1.0,
        )

        start = time.monotonic()
        async with httpx.AsyncClient(
            transport=transport, base_url="http://backend.test", follow_redirects=True
        ) as client:
            with pytest.raises(BackendDeadlineExceededError):
                await asyncio.wait_for(client.get("/a"), timeout=10.0)
        elapsed = time.monotonic() - start

        assert 0.9 <= elapsed < 1.3
        assert len(budgets_seen) == 3
        assert budgets_seen[0] is not None
        assert all(budget is budgets_seen[0] for budget in budgets_seen)

    async def test_transport_without_a_deadline_behaves_as_before(self) -> None:
        """Test-convenience shape, same as allowed_host: no deadline unless
        one is configured; BaseBackend._create_transport always configures
        one (see test_backend_clients_derive_the_deadline_from_the_timeout)."""
        stream = _TrickleStream(interval_seconds=0.05)
        served = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal served
            served += 1
            if served == 1:
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(200, stream=stream)

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))
        assert transport._total_deadline_seconds is None

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            assert (await client.get("/health")).json() == {"ok": True}
            with pytest.raises(asyncio.TimeoutError):
                # Unbounded by the transport: only the test's own guard ends it.
                await asyncio.wait_for(client.get("/trickle"), timeout=0.5)

    @pytest.mark.parametrize(
        ("timeout", "expected"),
        [(5.0, 30.0), (7.5, 30.0), (30.0, 120.0), (100.0, 400.0), (None, 30.0)],
    )
    def test_request_deadline_is_four_timeouts_with_a_30_s_floor(
        self, timeout: float | None, expected: float
    ) -> None:
        assert request_deadline_seconds(timeout) == expected

    @pytest.mark.parametrize(
        "backend",
        [
            JaegerBackend(url="https://backend.example:16686", timeout=5.0),
            DatadogBackend(url="https://backend.example", api_key="k", app_key="a", timeout=5.0),
            NewRelicBackend(
                url="https://backend.example/graphql", api_key="k", account_id="1", timeout=5.0
            ),
            HoneycombBackend(url="https://backend.example", api_key="k", dataset="d", timeout=5.0),
        ],
        ids=["base", "datadog", "newrelic", "honeycomb"],
    )
    def test_backend_clients_derive_the_deadline_from_the_timeout(
        self, backend: BaseBackend
    ) -> None:
        """BACKEND_TIMEOUT is the only knob: every backend's client, vendor
        overrides included, hands the derived deadline to the transport."""
        client = backend.client

        assert isinstance(client._transport, _RetryingTransport)
        assert client._transport._total_deadline_seconds == 30.0


class TestRetryBudgetPerLogicalRequest:
    """858b67b capped a single Retry-After at 10 s, but the cap and the
    3-attempt budget were re-armed on every redirect hop: the runtime probe
    (probe-2 chain/01) answered /api/traces/X with 302 -> (429 x2, 302) ->
    (429 x2, 302) -> (429 x2, 200) and parked one get_trace for 30.3 s over
    10 backend requests; with httpx's default max_redirects=20 that shape
    allows ~400 s of backend-chosen sleep. One retry budget - at most 2
    retries and 10 s of cumulative sleep - now spans the whole chain, and
    the transport itself follows at most _MAX_REDIRECT_HOPS redirects."""

    @staticmethod
    def _probe_chain_handler(retry_after: str) -> tuple[Any, list[str]]:
        """The probe's exact shape: every hop answers 429 twice, then
        redirects on; the last hop answers 200."""
        requests: list[str] = []
        per_path_429s: dict[str, int] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            requests.append(path)
            if path == "/api/traces/x":
                return httpx.Response(302, headers={"Location": "http://backend.test/hop1"})
            if per_path_429s.get(path, 0) < 2:
                per_path_429s[path] = per_path_429s.get(path, 0) + 1
                return httpx.Response(429, headers={"Retry-After": retry_after})
            if path == "/hop1":
                return httpx.Response(302, headers={"Location": "http://backend.test/hop2"})
            if path == "/hop2":
                return httpx.Response(302, headers={"Location": "http://backend.test/hop3"})
            return httpx.Response(200, json={"data": []})

        return handler, requests

    async def test_probe_chain_shares_one_retry_budget_across_hops(self) -> None:
        """Retry-After scaled from the probe's 5 s to 1 s so the suite stays
        fast; the shape is otherwise identical. Per-hop budgets would make
        this 6 s and 10 requests; one budget for the chain stops after the
        two retries hop1 consumed: hop2's first 429 is returned as is."""
        handler, requests = self._probe_chain_handler(retry_after="1")
        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), allowed_host="backend.test"
        )

        start = time.monotonic()
        async with httpx.AsyncClient(
            transport=transport, base_url="http://backend.test", follow_redirects=True
        ) as client:
            response = await asyncio.wait_for(client.get("/api/traces/x"), timeout=30.0)
        elapsed = time.monotonic() - start

        assert response.status_code == 429
        assert requests == ["/api/traces/x", "/hop1", "/hop1", "/hop1", "/hop2"]
        assert len(requests) <= 5
        assert 1.8 <= elapsed < 3.5

    async def test_cumulative_retry_sleep_is_capped_across_hops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sleep budget binds independently of the retry count: with the
        ceiling at 1.5 s and Retry-After 1, hop1's one retry (1 s slept)
        leaves hop2's 429 unable to afford a second second, after only ONE
        retry of the two the count would still allow."""
        monkeypatch.setattr(base_module, "_MAX_RETRY_AFTER_SECONDS", 1.5)
        requests: list[str] = []
        seen_429: set[str] = set()
        budgets: list[_RequestBudget] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request.url.path)
            budgets.append(request.extensions[_REQUEST_BUDGET_EXTENSION])
            if request.url.path not in seen_429:
                seen_429.add(request.url.path)
                return httpx.Response(429, headers={"Retry-After": "1"})
            if request.url.path == "/hop1":
                return httpx.Response(302, headers={"Location": "http://backend.test/hop2"})
            return httpx.Response(200, json={"data": []})

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), allowed_host="backend.test"
        )

        start = time.monotonic()
        async with httpx.AsyncClient(
            transport=transport, base_url="http://backend.test", follow_redirects=True
        ) as client:
            response = await asyncio.wait_for(client.get("/hop1"), timeout=30.0)
        elapsed = time.monotonic() - start

        assert response.status_code == 429
        assert requests == ["/hop1", "/hop1", "/hop2"]
        assert 0.9 <= elapsed < 2.0
        assert budgets[-1].retries_used == 1
        assert budgets[-1].retry_sleep_used == 1.0

    async def test_retry_budget_is_fresh_for_every_logical_request(self) -> None:
        """A client reused across tool calls must not carry one request's
        spent budget into the next: each client.get() starts at zero."""
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count % 3 != 0:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(200, json={"ok": True})

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            first = await client.get("/search")
            second = await client.get("/search")

        assert first.status_code == 200
        assert second.status_code == 200
        assert call_count == 6

    async def test_redirect_loop_is_cut_by_the_transport_whatever_the_client_allows(
        self,
    ) -> None:
        """The hop cap lives in the transport every backend client shares,
        so a client built with httpx's default max_redirects=20 (the probe's
        21-request loop) still stops after _MAX_REDIRECT_HOPS redirects."""
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(302, headers={"Location": "http://backend.test/loop"})

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), allowed_host="backend.test"
        )

        start = time.monotonic()
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://backend.test",
            follow_redirects=True,
            max_redirects=20,
        ) as client:
            with pytest.raises(httpx.TooManyRedirects, match="redirect"):
                await client.get("/loop")
        elapsed = time.monotonic() - start

        assert _MAX_REDIRECT_HOPS == 5
        assert call_count == _MAX_REDIRECT_HOPS + 1
        assert elapsed < 1.0

    async def test_five_redirects_are_still_followed(self) -> None:
        """Positive control for the hop cap: exactly _MAX_REDIRECT_HOPS
        redirects (six requests) complete."""
        hops = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal hops
            if hops < _MAX_REDIRECT_HOPS:
                hops += 1
                return httpx.Response(302, headers={"Location": f"http://backend.test/h{hops}"})
            return httpx.Response(200, json={"ok": True})

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), allowed_host="backend.test"
        )

        async with httpx.AsyncClient(
            transport=transport, base_url="http://backend.test", follow_redirects=True
        ) as client:
            response = await client.get("/h0")

        assert response.status_code == 200
        assert hops == _MAX_REDIRECT_HOPS

    def test_base_client_lowers_httpx_max_redirects_to_the_hop_cap(self) -> None:
        """Belt and braces with the transport cap: the shared client itself
        no longer allows httpx's default 20 hops."""
        client = JaegerBackend(url="http://backend.test").client

        assert client.max_redirects == _MAX_REDIRECT_HOPS

    @pytest.mark.parametrize(
        ("retries_used", "slept", "sleep", "affordable"),
        [
            (0, 0.0, 10.0, True),
            (1, 5.0, 5.0, True),
            (2, 0.0, 0.0, False),  # two retries already taken
            (0, 5.0, 5.5, False),  # would pass the 10 s cumulative sleep
            (1, 10.0, 0.0, True),  # a zero sleep still fits an exhausted sleep budget
        ],
    )
    def test_can_afford_retry_counts_retries_and_sleep_together(
        self, retries_used: int, slept: float, sleep: float, affordable: bool
    ) -> None:
        budget = _RequestBudget(deadline=None)
        budget.retries_used = retries_used
        budget.retry_sleep_used = slept

        assert budget.can_afford_retry(sleep) is affordable

    def test_can_afford_retry_refuses_a_sleep_that_would_cross_the_deadline(self) -> None:
        budget = _RequestBudget(deadline=time.monotonic() + 1.0)

        assert budget.can_afford_retry(0.5) is True
        assert budget.can_afford_retry(1.5) is False


class _ChunkedStream(httpx.AsyncByteStream):
    """A body delivered in fixed-size wire chunks, as httpcore hands them
    to httpx (64 KiB reads), so decoding is exercised across chunk
    boundaries rather than on one pre-assembled buffer."""

    def __init__(self, data: bytes, chunk_size: int) -> None:
        self._data = data
        self._chunk_size = chunk_size
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for offset in range(0, len(self._data), self._chunk_size):
            yield self._data[offset : offset + self._chunk_size]

    async def aclose(self) -> None:
        self.closed = True


def _gzip_bomb(decoded_mib: int) -> bytes:
    """`decoded_mib` MiB of zeros, gzip-compressed in a streaming fashion so
    the test itself never holds the decoded size in memory."""
    compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS | 16)
    parts = [compressor.compress(b"\0" * (1024 * 1024)) for _ in range(decoded_mib)]
    parts.append(compressor.flush())
    return b"".join(parts)


def _encoded_response(encoding: str, body: bytes, chunk_size: int = 64 * 1024) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"Content-Encoding": encoding, "Content-Type": "application/json"},
        stream=_ChunkedStream(body, chunk_size),
    )


class TestIncrementalDecoding:
    """BACKEND_MAX_RESPONSE_BYTES bounded the bytes BUFFERED, not the peak:
    httpx inflated each wire chunk completely before _BoundedResponse
    counted it, and a 64 KiB gzip chunk of zeros is ~64 MiB decoded. The
    runtime probe (probe-2 memory/01) served one 200 MiB-decoded gzip body
    under a 1 MiB cap and watched RSS rise by 121 MiB; eight concurrent
    bodies took the process from 10.9 MB to ~500 MB before all were
    refused. The transport now inflates the raw wire bytes itself through
    zlib with max_length = remaining budget + 1, so no step ever
    materialises more than the cap plus one byte."""

    async def test_gzip_bomb_is_refused_without_inflating_past_the_cap(self) -> None:
        bomb = _gzip_bomb(200)
        assert len(bomb) < 256 * 1024  # ~200 KiB on the wire for 200 MiB decoded
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _encoded_response("gzip", bomb)

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), max_response_bytes=_ONE_MIB
        )

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            tracemalloc.start()
            try:
                tracemalloc.reset_peak()
                before, _ = tracemalloc.get_traced_memory()
                with pytest.raises(ResponseTooLargeError, match="BACKEND_MAX_RESPONSE_BYTES"):
                    await client.get("/api/traces/bomb")
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()

        assert call_count == 1
        # Cap + one wire chunk + decoder state, never a whole inflated chunk.
        assert peak - before < _ONE_MIB + 4 * _ONE_MIB

    async def test_legitimate_900_kib_gzip_json_under_the_cap_decodes_byte_for_byte(
        self,
    ) -> None:
        body = json.dumps({"data": ["svc-a"], "pad": "a" * (900 * 1024)}).encode()
        gz = gzip.compress(body)

        async def handler(request: httpx.Request) -> httpx.Response:
            return _encoded_response("gzip", gz)

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), max_response_bytes=_ONE_MIB
        )

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            response = await client.get("/api/services")

        assert response.content == body
        assert response.json()["data"] == ["svc-a"]
        assert len(response.json()["pad"]) == 900 * 1024

    async def test_incompressible_gzip_body_spanning_many_wire_chunks_decodes_exactly(
        self,
    ) -> None:
        """Random bytes do not compress, so every 7 KiB wire chunk inflates
        to about 7 KiB and the decoder is driven across ~130 chunk
        boundaries with partial zlib state between them."""
        body = os.urandom(900 * 1024)
        gz = gzip.compress(body)

        async def handler(request: httpx.Request) -> httpx.Response:
            return _encoded_response("gzip", gz, chunk_size=7 * 1024)

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), max_response_bytes=_ONE_MIB
        )

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            response = await client.get("/blob")

        assert response.content == body

    @pytest.mark.parametrize("wbits", [zlib.MAX_WBITS, -zlib.MAX_WBITS], ids=["zlib", "raw"])
    async def test_deflate_in_both_wire_forms_decodes(self, wbits: int) -> None:
        """Content-Encoding: deflate is sent both zlib-wrapped and raw in the
        wild; httpx's DeflateDecoder accepts both, and so must the bounded
        path that replaces it."""
        body = json.dumps({"data": ["svc-a"], "pad": "b" * 4096}).encode()
        compressor = zlib.compressobj(6, zlib.DEFLATED, wbits)
        deflated = compressor.compress(body) + compressor.flush()

        async def handler(request: httpx.Request) -> httpx.Response:
            return _encoded_response("deflate", deflated, chunk_size=512)

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), max_response_bytes=_ONE_MIB
        )

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            response = await client.get("/api/services")

        assert response.content == body

    @pytest.mark.parametrize(("decoded_size", "allowed"), [(_ONE_MIB, True), (_ONE_MIB + 1, False)])
    async def test_gzip_boundary_at_the_cap_is_unchanged(
        self, decoded_size: int, allowed: bool
    ) -> None:
        """35fb662/23b2fe8 defined the boundary: exactly the cap passes, one
        byte more is refused - now for the incrementally decoded path too."""
        gz = gzip.compress(b"a" * decoded_size)

        async def handler(request: httpx.Request) -> httpx.Response:
            return _encoded_response("gzip", gz)

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), max_response_bytes=_ONE_MIB
        )

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            if allowed:
                response = await client.get("/blob")
                assert len(response.content) == decoded_size
            else:
                with pytest.raises(ResponseTooLargeError):
                    await client.get("/blob")

    async def test_corrupt_gzip_body_raises_httpx_decoding_error(self) -> None:
        """Same error type httpx's own decoder raises, so callers see no
        new failure shape for a damaged body."""

        async def handler(request: httpx.Request) -> httpx.Response:
            return _encoded_response("gzip", b"\x1f\x8b\x08\x00not really gzip data")

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), max_response_bytes=_ONE_MIB
        )

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            with pytest.raises(httpx.DecodingError):
                await client.get("/api/services")

    @pytest.mark.parametrize("encoding", ["br", "zstd", "x-custom", "gzip, deflate"])
    async def test_encodings_zlib_cannot_bound_are_refused_before_the_body_is_read(
        self, encoding: str
    ) -> None:
        """A decoder without an output bound (brotli, zstd, anything
        unknown, or a chain of encodings) would reintroduce the finding, so
        such a response is refused up front - the body is never read and
        the connection is released - rather than inflated by whatever httpx
        happens to have installed, or passed through undecoded as httpx
        does for an unknown encoding."""
        stream = _RecordingStream(b"x" * 16)
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, headers={"Content-Encoding": encoding}, stream=stream)

        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler), max_response_bytes=_ONE_MIB
        )

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            with pytest.raises(UnsupportedContentEncodingError, match="Content-Encoding"):
                await client.get("/api/services")

        assert call_count == 1  # not retryable
        assert stream.reads == 0
        assert stream.closed is True

    async def test_refusal_message_bounds_and_quotes_the_backend_chosen_encoding(self) -> None:
        """The encoding value is backend-chosen text that ends up in the
        tool-error channel and the server log: cut and repr()-quoted."""

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"Content-Encoding": "evil\x01" + "z" * 500},
                stream=httpx.ByteStream(b""),
            )

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            with pytest.raises(UnsupportedContentEncodingError) as exc_info:
                await client.get("/api/services")

        message = str(exc_info.value)
        assert "'evil\\x01zzz" in message  # repr()-quoted, control character escaped
        assert "z" * 100 not in message

    async def test_text_charset_of_a_gzip_body_is_untouched_by_the_content_encoding(self) -> None:
        """httpx.Response keeps the text charset in `_encoding`; the bounded
        response records the Content-Encoding under a different name so
        .text still decodes with the declared charset, not 'gzip'."""
        body = json.dumps({"service": "caf\u00e9"}, ensure_ascii=False).encode("utf-8")

        async def handler(request: httpx.Request) -> httpx.Response:
            response = _encoded_response("gzip", gzip.compress(body))
            response.headers["Content-Type"] = "application/json; charset=utf-8"
            return response

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            response = await client.get("/api/services")

        assert response.encoding == "utf-8"
        assert response.text == body.decode("utf-8")
        assert response.json() == {"service": "caf\u00e9"}

    async def test_reiterating_a_read_body_serves_the_decoded_content(self) -> None:
        """httpx callers may iterate aiter_bytes() after aread(); the bounded
        override must then serve the decoded content from memory like the
        base class, not touch the already-consumed stream."""
        body = json.dumps({"data": ["svc-a"]}).encode()

        async def handler(request: httpx.Request) -> httpx.Response:
            return _encoded_response("gzip", gzip.compress(body))

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            response = await client.get("/api/services")
            again = b"".join([chunk async for chunk in response.aiter_bytes()])

        assert response.content == body
        assert again == body

    def test_budget_with_a_passed_deadline_refuses_to_hand_out_time(self) -> None:
        budget = _RequestBudget(deadline=time.monotonic() - 0.001)

        with pytest.raises(BackendDeadlineExceededError):
            budget.remaining_seconds()
        assert _RequestBudget(deadline=None).remaining_seconds() is None

    async def test_identity_and_absent_content_encoding_keep_the_plain_path(self) -> None:
        body = json.dumps({"ok": True}).encode()

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/identity":
                return httpx.Response(200, headers={"Content-Encoding": "identity"}, content=body)
            return httpx.Response(200, content=body)

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            assert (await client.get("/identity")).json() == {"ok": True}
            assert (await client.get("/plain")).json() == {"ok": True}

    async def test_accept_encoding_offered_to_the_backend_is_limited_to_bounded_codecs(
        self,
    ) -> None:
        """httpx advertises every codec it has a decoder for (br/zstd when
        those extras are installed); the transport rewrites the offer to the
        codecs it can bound, so a well-behaved backend never picks one the
        response path would then have to refuse. The default offer is
        already within the set and passes through untouched."""
        offers: dict[str, str | None] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            offers[request.url.path] = request.headers.get("Accept-Encoding")
            return httpx.Response(200, json={"ok": True})

        transport = _RetryingTransport(wrapped=httpx.MockTransport(handler))

        async with httpx.AsyncClient(transport=transport, base_url="http://backend.test") as client:
            await client.get("/default")
            await client.get("/wide", headers={"Accept-Encoding": "gzip, deflate, br, zstd"})
            await client.get("/any", headers={"Accept-Encoding": "*"})
            await client.get("/qvalues", headers={"Accept-Encoding": "gzip;q=1.0, identity; q=0.5"})

        assert offers["/default"] == "gzip, deflate"
        assert offers["/wide"] == "gzip, deflate"
        assert offers["/any"] == "gzip, deflate"
        assert offers["/qvalues"] == "gzip;q=1.0, identity; q=0.5"

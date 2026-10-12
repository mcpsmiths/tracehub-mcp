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
import time
from collections.abc import AsyncIterator

import httpx
import pytest
from tenacity import AsyncRetrying, RetryCallState

from opentelemetry_mcp.backends.base import (
    DEFAULT_MAX_RESPONSE_BYTES,
    BaseBackend,
    MetadataEndpointBlockedError,
    RedirectTargetBlockedError,
    ResponseTooLargeError,
    _is_rate_limited,
    _retry_after_seconds,
    _RetryingTransport,
    _wait_for_retry,
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

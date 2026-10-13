"""Tests for the /health and /ready HTTP routes, registered via
FastMCP.custom_route (server.py's health_route/ready_route). These only
apply to the streamable-http transport - stdio mode has no HTTP server for
them to attach to.

Builds the real Starlette app FastMCP assembles for streamable-http (the
one gap the exploration flagged: no existing test drives a real HTTP
request against a real route), reusing test_http_origin_middleware.py's
TestClient pattern and test_server.py's TestDoctorCli AsyncMock-backend
pattern for mocking _get_backend.
"""

import asyncio
import logging
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from starlette.middleware import Middleware
from starlette.testclient import TestClient

from opentelemetry_mcp import server
from opentelemetry_mcp.attributes import HealthCheckResponse
from opentelemetry_mcp.backends.base import _RetryingTransport
from opentelemetry_mcp.backends.jaeger import JaegerBackend
from opentelemetry_mcp.server import OriginValidationMiddleware
from tests.streamable_http_client import post_initialize


@pytest.fixture(autouse=True)
def _fresh_readiness_memo(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The readiness memo and its lock are process-wide; every test starts
    from an empty memo at the default window with the real clock, and leaves
    nothing behind for the next test's event loop."""
    server._reset_readiness_memo()
    monkeypatch.setattr(server, "_readiness_cache_seconds", server._READINESS_MEMO_TTL_SECONDS)
    monkeypatch.setattr(server, "_readiness_clock", time.monotonic)
    yield
    server._reset_readiness_memo()


def _forward_uncached_list_services(fake_backend: AsyncMock) -> None:
    """_run_backend_checks probes through list_services_uncached (never the
    cache-wrapped list_services); the fake forwards to whatever
    list_services the test installed, so tests keep stubbing and counting
    list_services as before."""

    async def _forward() -> Any:
        return await fake_backend.list_services()

    fake_backend.list_services_uncached = AsyncMock(side_effect=_forward)


def _healthy_backend() -> AsyncMock:
    fake_backend = AsyncMock()
    fake_backend.health_check = AsyncMock(
        return_value=HealthCheckResponse(status="healthy", backend="jaeger", url="http://x")
    )
    fake_backend.list_services = AsyncMock(return_value=["svc-a", "svc-b"])
    _forward_uncached_list_services(fake_backend)
    fake_backend.close = AsyncMock()
    return fake_backend


def _build_client() -> TestClient:
    app = server.mcp.http_app(transport="streamable-http")
    return TestClient(app)


def test_health_returns_200_without_touching_backend() -> None:
    """Liveness must not do backend I/O - patch _get_backend with something
    that raises if called at all, to prove /health never touches it."""
    with patch.object(
        server, "_get_backend", AsyncMock(side_effect=AssertionError("must not be called"))
    ):
        with _build_client() as client:
            response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_returns_200_when_backend_healthy() -> None:
    fake_backend = _healthy_backend()
    with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
        with _build_client() as client:
            response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["health_check"]["status"] == "healthy"
    assert body["list_services"]["count"] == 2


def test_ready_returns_503_when_health_check_unhealthy() -> None:
    fake_backend = _healthy_backend()
    fake_backend.health_check = AsyncMock(
        return_value=HealthCheckResponse(
            status="unhealthy", backend="jaeger", url="http://x", error="connection refused"
        )
    )
    with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
        with _build_client() as client:
            response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["health_check"]["status"] == "unhealthy"
    # The body carries a class, never the backend's own text (see
    # TestReadyBodyCarriesOnlyAGenericErrorClass).
    assert body["health_check"]["error"] == "backend unreachable"


def test_ready_returns_503_when_list_services_raises() -> None:
    fake_backend = _healthy_backend()
    fake_backend.list_services = AsyncMock(side_effect=RuntimeError("timeout"))
    with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
        with _build_client() as client:
            response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["list_services"]["error"] == "backend timeout"


def test_ready_returns_503_when_get_backend_itself_raises() -> None:
    """_get_backend can raise (e.g. server configuration not set yet) -
    /ready must report 503, not an unhandled 500."""
    with patch.object(
        server, "_get_backend", AsyncMock(side_effect=RuntimeError("Server configuration not set"))
    ):
        with _build_client() as client:
            response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready", "error": "backend error"}


class TestReadyNeverEchoesUrlUserinfo:
    """/ready answers unauthenticated peers. The error strings
    health_check/list_services produce embed the request URL (httpx's
    HTTPStatusError text): _run_backend_checks redacts them for doctor and
    the server log, and the /ready body itself now carries only a short
    error class, so neither a credential nor the host can leave."""

    MARKER = "VFY-PW-ready"
    ERROR = f"Client error '401 Unauthorized' for url 'https://u:{MARKER}@h.test/api/services'"

    def test_health_check_error_is_redacted(self) -> None:
        fake_backend = _healthy_backend()
        fake_backend.health_check = AsyncMock(
            return_value=HealthCheckResponse(
                status="unhealthy",
                backend="jaeger",
                url=f"https://u:{self.MARKER}@h.test",
                error=self.ERROR,
            )
        )
        with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
            with _build_client() as client:
                response = client.get("/ready")

        assert response.status_code == 503
        assert self.MARKER not in response.text
        assert "h.test" not in response.text
        assert response.json()["health_check"]["error"] == "backend error 401"

    def test_health_check_exception_and_list_services_error_are_redacted(self) -> None:
        fake_backend = _healthy_backend()
        fake_backend.health_check = AsyncMock(side_effect=RuntimeError(self.ERROR))
        fake_backend.list_services = AsyncMock(side_effect=RuntimeError(self.ERROR))
        with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
            with _build_client() as client:
                response = client.get("/ready")

        assert response.status_code == 503
        assert self.MARKER not in response.text
        assert "h.test" not in response.text
        body = response.json()
        assert body["health_check"]["error"] == "backend error 401"
        assert body["list_services"]["error"] == "backend error 401"

    def test_get_backend_failure_is_redacted(self) -> None:
        with patch.object(server, "_get_backend", AsyncMock(side_effect=RuntimeError(self.ERROR))):
            with _build_client() as client:
                response = client.get("/ready")

        assert response.status_code == 503
        assert self.MARKER not in response.text


class TestReadyBehindHostValidation:
    """/ready drives real backend calls, so a DNS-rebound browser's
    Origin-less GET must be refused by the Host check before the route runs
    when the listener is loopback-bound (the shape main() now defaults to)."""

    def _build_client(self) -> TestClient:
        app = server.mcp.http_app(
            transport="streamable-http",
            middleware=[Middleware(OriginValidationMiddleware, bind_host="127.0.0.1")],
        )
        return TestClient(app)

    def test_foreign_host_without_origin_is_421_and_touches_no_backend(self) -> None:
        fake_backend = _healthy_backend()
        with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
            with self._build_client() as client:
                response = client.get("/ready", headers={"host": "attacker.example:8000"})

        assert response.status_code == 421
        fake_backend.health_check.assert_not_awaited()
        fake_backend.list_services.assert_not_awaited()

    def test_loopback_host_still_reaches_ready(self) -> None:
        fake_backend = _healthy_backend()
        with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
            with self._build_client() as client:
                response = client.get("/ready", headers={"host": "localhost:8000"})

        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_loopback_host_still_opens_an_mcp_session(self) -> None:
        with self._build_client() as client:
            response = post_initialize(client, headers={"host": "127.0.0.1:8000"})

        assert response.status_code == 200
        assert response.headers.get("mcp-session-id")


def test_ready_never_closes_the_cached_backend() -> None:
    """The key behavioral difference from `doctor`: /ready reuses the
    server's cached backend and must never close it, or every subsequent
    real tool call sharing that instance would break."""
    fake_backend = _healthy_backend()
    with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
        with _build_client() as client:
            response = client.get("/ready")

    assert response.status_code == 200
    fake_backend.close.assert_not_called()


class _FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestReadinessMemo:
    """GET /ready is unauthenticated and every probe costs the backend one
    health_check plus one list_services - on Honeycomb a slot of the
    process-wide 10/min Create Query Result budget, on X-Ray up to 222
    operator-paid API calls. The route memoizes the result (30 s on
    success, 5 s on failure), shares one in-flight check between concurrent
    probes and answers 503 instead of hanging; `doctor` keeps calling the
    live _run_backend_checks directly."""

    def test_sequential_probes_within_the_window_share_one_backend_check(self) -> None:
        fake_backend = _healthy_backend()
        with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
            with _build_client() as client:
                responses = [client.get("/ready") for _ in range(10)]

        assert [r.status_code for r in responses] == [200] * 10
        assert all(r.json() == responses[0].json() for r in responses)
        assert fake_backend.health_check.await_count == 1
        assert fake_backend.list_services.await_count == 1

    def test_success_result_is_reprobed_once_the_success_ttl_elapses(self) -> None:
        clock = _FakeClock()
        server._configure_readiness_cache(server._READINESS_MEMO_TTL_SECONDS, clock=clock)
        fake_backend = _healthy_backend()
        with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
            with _build_client() as client:
                client.get("/ready")
                clock.advance(server._READINESS_MEMO_TTL_SECONDS - 0.1)
                client.get("/ready")
                assert fake_backend.health_check.await_count == 1
                clock.advance(0.1)
                client.get("/ready")

        assert fake_backend.health_check.await_count == 2
        assert fake_backend.list_services.await_count == 2

    def test_failure_result_is_reprobed_after_the_shorter_failure_ttl(self) -> None:
        clock = _FakeClock()
        server._configure_readiness_cache(server._READINESS_MEMO_TTL_SECONDS, clock=clock)
        fake_backend = _healthy_backend()
        fake_backend.health_check = AsyncMock(
            return_value=HealthCheckResponse(
                status="unhealthy", backend="jaeger", url="http://x", error="connection refused"
            )
        )
        with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
            with _build_client() as client:
                assert client.get("/ready").status_code == 503

                # The backend recovers, but the failure is memoized for 5 s...
                fake_backend.health_check = AsyncMock(
                    return_value=HealthCheckResponse(
                        status="healthy", backend="jaeger", url="http://x"
                    )
                )
                clock.advance(server._READINESS_FAILURE_MEMO_TTL_SECONDS - 0.1)
                assert client.get("/ready").status_code == 503
                fake_backend.health_check.assert_not_awaited()

                # ...and not for the full 30 s success window.
                clock.advance(0.1)
                assert client.get("/ready").status_code == 200

        assert fake_backend.health_check.await_count == 1

    def test_concurrent_probes_coalesce_onto_one_in_flight_check(self) -> None:
        asyncio.run(self._concurrent_probes_share_one_check(cache_seconds=30.0))

    def test_zero_cache_seconds_still_coalesces_concurrent_probes(self) -> None:
        """READINESS_CACHE_SECONDS=0 disables the memo, not single-flight: a
        probe that arrives while a check is in flight waits for that check
        instead of starting a second one."""
        asyncio.run(self._concurrent_probes_share_one_check(cache_seconds=0.0))

    async def _concurrent_probes_share_one_check(self, *, cache_seconds: float) -> None:
        server._configure_readiness_cache(cache_seconds)
        release = asyncio.Event()
        fake_backend = _healthy_backend()

        async def _blocking_list_services() -> list[str]:
            await release.wait()
            return ["svc-a", "svc-b"]

        fake_backend.list_services = AsyncMock(side_effect=_blocking_list_services)

        async def _release_once_everyone_is_parked() -> None:
            await asyncio.sleep(0.05)
            release.set()

        app = server.mcp.http_app(transport="streamable-http")
        with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                results = await asyncio.gather(
                    *(client.get("/ready") for _ in range(5)),
                    _release_once_everyone_is_parked(),
                )

        responses = [r for r in results if isinstance(r, httpx.Response)]
        assert len(responses) == 5
        assert [r.status_code for r in responses] == [200] * 5
        assert all(r.json() == responses[0].json() for r in responses)
        assert fake_backend.health_check.await_count == 1
        assert fake_backend.list_services.await_count == 1

    def test_hung_backend_check_answers_503_within_the_readiness_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(server, "_READINESS_CHECK_TIMEOUT_SECONDS", 0.05)
        fake_backend = _healthy_backend()
        never = asyncio.Event()

        async def _hang() -> list[str]:
            await never.wait()
            return []

        fake_backend.list_services = AsyncMock(side_effect=_hang)
        with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
            with _build_client() as client:
                started = time.monotonic()
                response = client.get("/ready")
                elapsed = time.monotonic() - started

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        assert "timed out" in body["error"]
        assert elapsed < 2.0

    def test_timed_out_check_is_memoized_as_a_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(server, "_READINESS_CHECK_TIMEOUT_SECONDS", 0.05)
        clock = _FakeClock()
        server._configure_readiness_cache(server._READINESS_MEMO_TTL_SECONDS, clock=clock)
        fake_backend = _healthy_backend()
        never = asyncio.Event()

        async def _hang() -> list[str]:
            await never.wait()
            return []

        fake_backend.list_services = AsyncMock(side_effect=_hang)
        with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
            with _build_client() as client:
                assert client.get("/ready").status_code == 503
                clock.advance(server._READINESS_FAILURE_MEMO_TTL_SECONDS - 0.1)
                assert client.get("/ready").status_code == 503

        assert fake_backend.health_check.await_count == 1

    def test_swapping_the_backend_object_invalidates_the_memo(self) -> None:
        first_backend = _healthy_backend()
        second_backend = _healthy_backend()
        with _build_client() as client:
            with patch.object(server, "_get_backend", AsyncMock(return_value=first_backend)):
                client.get("/ready")
            with patch.object(server, "_get_backend", AsyncMock(return_value=second_backend)):
                client.get("/ready")

        assert first_backend.health_check.await_count == 1
        assert second_backend.health_check.await_count == 1

    def test_zero_cache_seconds_disables_the_memo(self) -> None:
        server._configure_readiness_cache(0.0)
        fake_backend = _healthy_backend()
        with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
            with _build_client() as client:
                for _ in range(3):
                    assert client.get("/ready").status_code == 200

        assert fake_backend.health_check.await_count == 3
        assert fake_backend.list_services.await_count == 3

    def test_zero_cache_seconds_keeps_the_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(server, "_READINESS_CHECK_TIMEOUT_SECONDS", 0.05)
        server._configure_readiness_cache(0.0)
        fake_backend = _healthy_backend()
        never = asyncio.Event()

        async def _hang() -> list[str]:
            await never.wait()
            return []

        fake_backend.list_services = AsyncMock(side_effect=_hang)
        with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
            with _build_client() as client:
                response = client.get("/ready")

        assert response.status_code == 503
        assert "timed out" in response.json()["error"]

    def test_reset_drops_the_memo(self) -> None:
        fake_backend = _healthy_backend()
        with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
            with _build_client() as client:
                client.get("/ready")
                server._reset_readiness_memo()
                client.get("/ready")

        assert fake_backend.health_check.await_count == 2

    def test_doctor_path_stays_live(self) -> None:
        """_run_backend_checks itself is never memoized - doctor calls it
        directly and must report the current state on every run."""
        fake_backend = _healthy_backend()

        async def _twice() -> None:
            await server._run_backend_checks(fake_backend)
            await server._run_backend_checks(fake_backend)

        asyncio.run(_twice())

        assert fake_backend.health_check.await_count == 2
        assert fake_backend.list_services.await_count == 2

    def test_memo_stores_the_redacted_details_unchanged(self) -> None:
        """The memo stores what _run_backend_checks returns - already
        redacted - so a memoized body never carries more than a live one."""
        marker = "VFY-PW-memo"
        fake_backend = _healthy_backend()
        fake_backend.list_services = AsyncMock(
            side_effect=RuntimeError(f"Client error for url 'https://u:{marker}@h.test/api'")
        )
        with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
            with _build_client() as client:
                first = client.get("/ready")
                second = client.get("/ready")

        assert first.status_code == second.status_code == 503
        assert marker not in first.text
        assert marker not in second.text
        assert first.json() == second.json()
        assert fake_backend.list_services.await_count == 1


class TestReadyNeverAnsweredFromTheQueryCache:
    """With QUERY_CACHE_TTL_SECONDS set, a warmed list_services entry used to
    answer both halves of /ready (JaegerBackend.health_check awaited the
    cache-wrapped list_services, and _run_backend_checks did too), so the
    route and the Docker HEALTHCHECK reported ready with zero backend
    requests after the backend failed. Both halves now probe live; the
    readiness memo is disabled here so only the query cache is under test."""

    def _jaeger_over(self, status_code: list[int], *, cache: bool) -> JaegerBackend:
        backend = JaegerBackend(url="http://localhost:16686")
        if cache:
            backend.configure_query_cache(60.0)

        def handler(request: httpx.Request) -> httpx.Response:
            if status_code[0] == 200:
                return httpx.Response(200, json={"data": ["svc-a"]})
            return httpx.Response(status_code[0], text="unauthorized")

        backend._client = httpx.AsyncClient(
            base_url=backend.url,
            transport=_RetryingTransport(
                wrapped=httpx.MockTransport(handler), allowed_host="localhost"
            ),
        )
        return backend

    @pytest.mark.parametrize("cache", [True, False], ids=["query-cache-on", "query-cache-off"])
    def test_ready_returns_503_after_backend_failure(self, cache: bool) -> None:
        server._configure_readiness_cache(0.0)
        status_code = [200]
        backend = self._jaeger_over(status_code, cache=cache)
        with patch.object(server, "_get_backend", AsyncMock(return_value=backend)):
            with _build_client() as client:
                first = client.get("/ready")
                status_code[0] = 401
                second = client.get("/ready")

        assert first.status_code == 200
        assert first.json()["status"] == "ready"
        assert first.json()["list_services"]["count"] == 1

        assert second.status_code == 503
        body = second.json()
        assert body["status"] == "not_ready"
        assert body["health_check"]["status"] == "unhealthy"
        assert "401" in body["health_check"]["error"]
        assert "401" in body["list_services"]["error"]


class TestReadyBodyCarriesOnlyAGenericErrorClass:
    """Runtime verification probe-4 P1-P3: the unauthenticated /ready 503
    body echoed the backend's internal URL host:port and httpx's raw error
    text ("All connection attempts failed", "Client error '503 ...' for url
    'http://127.0.0.1:18164/api/services'"), and a failed or timed-out probe
    left no server log line. The body now names a short class per failing
    half - 'backend unreachable', 'backend timeout', 'backend error <status>',
    'backend unhealthy' - and the redacted detail is logged once at WARNING,
    bounded and repr-quoted like the Host/Origin rejection lines."""

    FAKE_HOST_PORT = "127.0.0.1:18164"

    def _ready(self, fake_backend: AsyncMock) -> Any:
        with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
            with _build_client() as client:
                return client.get("/ready")

    def test_unhealthy_status_text_is_classified_and_the_detail_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        fake_backend = _healthy_backend()
        detail = (
            f"Server error '503 Service Unavailable' for url "
            f"'http://{self.FAKE_HOST_PORT}/api/services'"
        )
        fake_backend.health_check = AsyncMock(
            return_value=HealthCheckResponse(
                status="unhealthy", backend="jaeger", url="http://x", error=detail
            )
        )
        with caplog.at_level(logging.WARNING, logger="opentelemetry_mcp.server"):
            response = self._ready(fake_backend)

        assert response.status_code == 503
        assert response.json()["health_check"] == {
            "status": "unhealthy",
            "error": "backend error 503",
        }
        assert self.FAKE_HOST_PORT not in response.text
        assert "Service Unavailable" not in response.text
        [message] = [
            r.getMessage() for r in caplog.records if "Readiness probe failed" in r.getMessage()
        ]
        assert self.FAKE_HOST_PORT in message
        assert repr(detail) in message

    def test_transport_exceptions_are_classified_by_type(self) -> None:
        fake_backend = _healthy_backend()
        fake_backend.health_check = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
        fake_backend.list_services = AsyncMock(
            side_effect=httpx.ConnectError(
                f"All connection attempts failed for http://{self.FAKE_HOST_PORT}/"
            )
        )

        response = self._ready(fake_backend)

        assert response.status_code == 503
        body = response.json()
        assert body["health_check"] == {"status": "error", "error": "backend timeout"}
        assert body["list_services"] == {"error": "backend unreachable"}
        assert self.FAKE_HOST_PORT not in response.text

    def test_http_status_exception_names_the_status_only(self) -> None:
        fake_backend = _healthy_backend()
        request = httpx.Request("GET", f"http://{self.FAKE_HOST_PORT}/api/services")
        response_401 = httpx.Response(401, request=request)
        fake_backend.list_services = AsyncMock(
            side_effect=httpx.HTTPStatusError("boom", request=request, response=response_401)
        )

        response = self._ready(fake_backend)

        assert response.json()["list_services"] == {"error": "backend error 401"}
        assert self.FAKE_HOST_PORT not in response.text

    def test_a_healthy_body_keeps_its_shape(self) -> None:
        response = self._ready(_healthy_backend())

        assert response.status_code == 200
        assert response.json() == {
            "status": "ready",
            "health_check": {"status": "healthy", "error": None},
            "list_services": {"count": 2},
        }

    def test_detail_is_logged_once_per_fresh_probe(self, caplog: pytest.LogCaptureFixture) -> None:
        fake_backend = _healthy_backend()
        fake_backend.list_services = AsyncMock(side_effect=RuntimeError("probe detail"))
        with caplog.at_level(logging.WARNING, logger="opentelemetry_mcp.server"):
            with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
                with _build_client() as client:
                    first = client.get("/ready")
                    second = client.get("/ready")

        assert first.status_code == second.status_code == 503
        failures = [r for r in caplog.records if "Readiness probe failed" in r.getMessage()]
        assert len(failures) == 1
        assert "probe detail" in failures[0].getMessage()

    def test_log_line_is_bounded_and_quoted(self, caplog: pytest.LogCaptureFixture) -> None:
        fake_backend = _healthy_backend()
        oversized = "x\n" + "a" * (50 * server._MAX_LOGGED_PROBE_ERROR_CHARS)
        fake_backend.list_services = AsyncMock(side_effect=RuntimeError(oversized))
        with caplog.at_level(logging.WARNING, logger="opentelemetry_mcp.server"):
            self._ready(fake_backend)

        [message] = [
            r.getMessage() for r in caplog.records if "Readiness probe failed" in r.getMessage()
        ]
        assert "\n" not in message
        assert "'x\\n" in message
        assert len(message) <= 128 + 2 * server._MAX_LOGGED_PROBE_ERROR_CHARS

    def test_timed_out_probe_is_logged(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(server, "_READINESS_CHECK_TIMEOUT_SECONDS", 0.05)
        fake_backend = _healthy_backend()
        never = asyncio.Event()

        async def _hang() -> list[str]:
            await never.wait()
            return []

        fake_backend.list_services = AsyncMock(side_effect=_hang)
        with caplog.at_level(logging.WARNING, logger="opentelemetry_mcp.server"):
            response = self._ready(fake_backend)

        assert response.status_code == 503
        assert "timed out" in response.json()["error"]
        assert any(
            "Readiness probe failed" in r.getMessage() and "timed out" in r.getMessage()
            for r in caplog.records
        )

    def test_get_backend_failure_is_generic_in_the_body_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        detail = f"Backend construction failed for http://u:pw@{self.FAKE_HOST_PORT}/"
        with caplog.at_level(logging.WARNING, logger="opentelemetry_mcp.server"):
            with patch.object(server, "_get_backend", AsyncMock(side_effect=RuntimeError(detail))):
                with _build_client() as client:
                    response = client.get("/ready")

        assert response.status_code == 503
        assert response.json() == {"status": "not_ready", "error": "backend error"}
        assert self.FAKE_HOST_PORT not in response.text
        [message] = [r.getMessage() for r in caplog.records if "Readiness probe" in r.getMessage()]
        assert f"[REDACTED]@{self.FAKE_HOST_PORT}" in message
        assert "pw@" not in message


class TestBackendErrorClassification:
    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (httpx.ConnectError("[Errno 61] Connection refused"), "backend unreachable"),
            (httpx.RemoteProtocolError("peer closed connection"), "backend unreachable"),
            (OSError("network is down"), "backend unreachable"),
            (httpx.ReadTimeout(""), "backend timeout"),
            (TimeoutError(), "backend timeout"),
            (RuntimeError("timed out waiting"), "backend timeout"),
            (RuntimeError("something odd"), "backend error"),
        ],
    )
    def test_exception_types(self, error: BaseException, expected: str) -> None:
        assert server._classify_backend_error(error) == expected

    def test_status_error_anywhere_in_the_chain(self) -> None:
        request = httpx.Request("GET", "http://h.test/")
        status = httpx.HTTPStatusError(
            "x", request=request, response=httpx.Response(429, request=request)
        )
        wrapped = RuntimeError("wrapped")
        wrapped.__cause__ = status

        assert server._classify_backend_error(wrapped) == "backend error 429"

    def test_timeout_wins_over_transport(self) -> None:
        assert server._classify_backend_error(httpx.ConnectTimeout("")) == "backend timeout"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (None, "backend unhealthy"),
            ("", "backend unhealthy"),
            ("Client error '401 Unauthorized' for url 'http://h.test/api'", "backend error 401"),
            ("Server error '503 Service Unavailable' for url 'x'", "backend error 503"),
            ("status code 429", "backend error 429"),
            ("All connection attempts failed", "backend unreachable"),
            ("[Errno 61] Connection refused", "backend unreachable"),
            ("Name or service not known", "backend unreachable"),
            ("Read timed out", "backend timeout"),
            ("timeout", "backend timeout"),
            ("trace id 503abc not found", "backend unhealthy"),
            ("some other failure", "backend unhealthy"),
        ],
    )
    def test_error_text(self, text: str | None, expected: str) -> None:
        assert server._classify_backend_error_text(text) == expected

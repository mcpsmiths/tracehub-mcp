"""Tests for RateLimitMiddleware and its underlying _FixedWindowRateLimiter
(see server.py's class docstrings for design rationale).

Mirrors test_http_origin_middleware.py's pattern: a minimal standalone
Starlette app wrapping only the middleware under test, driven via
TestClient - spinning up the full MCP server would be slow and out of
scope. TestServedProcessShape additionally runs that app through the exact
uvicorn.Config main() builds (_build_uvicorn_config) and loads it, because
the limiter's key - request.client.host - is whatever uvicorn's
ProxyHeadersMiddleware leaves in scope["client"], not the raw TCP peer.
"""

from typing import Any

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from opentelemetry_mcp.server import (
    RateLimitMiddleware,
    _build_uvicorn_config,
    _FixedWindowRateLimiter,
)

ALL_INTERFACES = "0.0.0.0"  # noqa: S104 - the network bind shape under test
LOOPBACK_PEER = ("127.0.0.1", 0)
REMOTE_PEER = ("198.51.100.7", 0)
ROTATING_FORWARDED_FOR = ["203.0.113.1", "203.0.113.2", "203.0.113.3"]


async def _downstream_handler(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


def _build_app(max_requests: int = 2, window_seconds: float = 60.0) -> Starlette:
    return Starlette(
        routes=[Route("/", _downstream_handler)],
        middleware=[
            Middleware(
                RateLimitMiddleware, max_requests=max_requests, window_seconds=window_seconds
            )
        ],
    )


def _build_client(max_requests: int = 2, window_seconds: float = 60.0) -> TestClient:
    return TestClient(_build_app(max_requests, window_seconds))


class TestFixedWindowRateLimiter:
    """Direct, non-ASGI tests for the limiter's own logic."""

    async def test_requests_under_limit_are_allowed(self) -> None:
        limiter = _FixedWindowRateLimiter(max_requests=3, window_seconds=60.0)

        assert await limiter.hit("a") is True
        assert await limiter.hit("a") is True
        assert await limiter.hit("a") is True

    async def test_request_over_limit_is_rejected(self) -> None:
        limiter = _FixedWindowRateLimiter(max_requests=2, window_seconds=60.0)

        assert await limiter.hit("a") is True
        assert await limiter.hit("a") is True
        assert await limiter.hit("a") is False

    async def test_different_keys_are_independent(self) -> None:
        limiter = _FixedWindowRateLimiter(max_requests=1, window_seconds=60.0)

        assert await limiter.hit("a") is True
        assert await limiter.hit("b") is True
        assert await limiter.hit("a") is False
        assert await limiter.hit("b") is False

    async def test_window_resets_after_expiry(self) -> None:
        clock_value = [0.0]
        limiter = _FixedWindowRateLimiter(
            max_requests=1, window_seconds=10.0, clock=lambda: clock_value[0]
        )

        assert await limiter.hit("a") is True
        assert await limiter.hit("a") is False

        clock_value[0] = 10.0  # exactly at the window boundary
        assert await limiter.hit("a") is True


class TestRateLimitMiddlewareDispatch:
    """TestClient-based tests mirroring test_http_origin_middleware.py's
    minimal-app pattern."""

    def test_requests_under_limit_pass_through(self) -> None:
        client = _build_client(max_requests=2)

        first = client.get("/")
        second = client.get("/")

        assert first.status_code == 200
        assert first.text == "ok"
        assert second.status_code == 200

    def test_breach_returns_429_without_reaching_downstream_handler(self) -> None:
        client = _build_client(max_requests=1)

        client.get("/")
        response = client.get("/")

        assert response.status_code == 429
        assert response.text == "Rate limit exceeded"


def _loaded_app(forwarded_allow_ips: str | None) -> Any:
    cfg = _build_uvicorn_config(
        _build_app(max_requests=2),
        host=ALL_INTERFACES,
        port=8000,
        graceful_shutdown_timeout_seconds=2,
        forwarded_allow_ips=forwarded_allow_ips,
    )
    cfg.load()
    return cfg.loaded_app


def _statuses_with_rotating_forwarded_for(app: Any, peer: tuple[str, int]) -> list[int]:
    client = TestClient(app, client=peer)
    return [
        client.get("/", headers={"X-Forwarded-For": forwarded}).status_code
        for forwarded in ROTATING_FORWARDED_FOR
    ]


class TestServedProcessShape:
    """uvicorn 0.40.0 defaults proxy_headers=True and trusts 127.0.0.1 (or
    FORWARDED_ALLOW_IPS) implicitly, so a loopback peer - a pass-through
    proxy, an ssh -L / socat tunnel, another local user - could pick its own
    rate-limit key by rotating X-Forwarded-For. The served process must not
    consult that header unless the operator names the proxies."""

    def test_default_never_trusts_forwarded_for_even_from_loopback(self) -> None:
        app = _loaded_app(forwarded_allow_ips=None)

        assert type(app).__name__ != "ProxyHeadersMiddleware"
        assert _statuses_with_rotating_forwarded_for(app, LOOPBACK_PEER) == [200, 200, 429]

    def test_opt_in_trusts_only_the_named_proxy_address(self) -> None:
        app = _loaded_app(forwarded_allow_ips="127.0.0.1")

        assert type(app).__name__ == "ProxyHeadersMiddleware"
        # The named proxy's X-Forwarded-For replaces the key: each rotated
        # client address starts its own bucket.
        assert _statuses_with_rotating_forwarded_for(app, LOOPBACK_PEER) == [200, 200, 200]
        # An unlisted peer's header is ignored - still keyed by its TCP address.
        assert _statuses_with_rotating_forwarded_for(app, REMOTE_PEER) == [200, 200, 429]

    def test_prefill_of_another_clients_bucket_needs_the_opt_in(self) -> None:
        """Only a trusted proxy can spend a third party's budget: with the
        default, a loopback peer claiming to forward for 198.51.100.7 spends
        its own."""
        for forwarded_allow_ips, expected in ((None, 200), ("127.0.0.1", 429)):
            app = _loaded_app(forwarded_allow_ips=forwarded_allow_ips)
            spoofer = TestClient(app, client=LOOPBACK_PEER)
            for _ in range(2):
                spoofer.get("/", headers={"X-Forwarded-For": REMOTE_PEER[0]})

            victim = TestClient(app, client=REMOTE_PEER)
            assert victim.get("/").status_code == expected, forwarded_allow_ips

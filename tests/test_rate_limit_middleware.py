"""Tests for RateLimitMiddleware and its underlying _FixedWindowRateLimiter
(see server.py's class docstrings for design rationale).

Mirrors test_http_origin_middleware.py's pattern: a minimal standalone
Starlette app wrapping only the middleware under test, driven via
TestClient - spinning up the full MCP server would be slow and out of
scope.
"""

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from opentelemetry_mcp.server import RateLimitMiddleware, _FixedWindowRateLimiter


async def _downstream_handler(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


def _build_client(max_requests: int = 2, window_seconds: float = 60.0) -> TestClient:
    app = Starlette(
        routes=[Route("/", _downstream_handler)],
        middleware=[
            Middleware(
                RateLimitMiddleware, max_requests=max_requests, window_seconds=window_seconds
            )
        ],
    )
    return TestClient(app)


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

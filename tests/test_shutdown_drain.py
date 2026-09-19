"""Tests proving server.py's _install_shutdown_drain/_drain_and_close_backend
actually fires via the real ASGI lifespan.shutdown mechanism - the *inner*
lifespan mcp.http_app() builds - not via a FastMCP(lifespan=...) constructor
kwarg (tried and reverted; see server.py's _install_shutdown_drain docstring).

Reuses test_health_routes.py's mcp.http_app(...) + TestClient(app)
with-block pattern: TestClient's own __exit__ drives the same ASGI
lifespan.shutdown dispatch that uvicorn.Server.shutdown() drives on a real
SIGTERM, so it exercises the real mechanism the fix depends on.

Patches the module-global _backend directly (test_server.py's existing
pattern for this global), not _get_backend, since the drain/close code
operates on whichever backend instance is already cached.
"""

import time
from collections.abc import Generator
from unittest.mock import AsyncMock

import pytest
from fastmcp.server.http import StarletteWithLifespan
from starlette.testclient import TestClient

from opentelemetry_mcp import server


@pytest.fixture(autouse=True)
def _reset_backend_global() -> Generator[None]:
    server._backend = None
    yield
    server._backend = None


def _build_app_with_drain(
    drain_seconds: float = 0.0, close_timeout_seconds: float = 5.0
) -> StarletteWithLifespan:
    app = server.mcp.http_app(transport="streamable-http")
    server._install_shutdown_drain(
        app, drain_seconds=drain_seconds, close_timeout_seconds=close_timeout_seconds
    )
    return app


def test_shutdown_closes_cached_backend_via_real_asgi_lifespan() -> None:
    fake_backend = AsyncMock()
    server._backend = fake_backend

    app = _build_app_with_drain(drain_seconds=0.0)
    with TestClient(app):
        fake_backend.close.assert_not_called()

    fake_backend.close.assert_awaited_once()
    assert server._backend is None


def test_shutdown_is_a_noop_when_no_backend_was_ever_created() -> None:
    server._backend = None
    app = _build_app_with_drain(drain_seconds=0.0)
    with TestClient(app):
        pass  # must not raise even though there is nothing to close


def test_shutdown_sleeps_for_the_configured_drain_seconds_before_closing() -> None:
    fake_backend = AsyncMock()
    server._backend = fake_backend
    app = _build_app_with_drain(drain_seconds=0.15)

    start = time.monotonic()
    with TestClient(app):
        pass
    elapsed = time.monotonic() - start

    assert elapsed >= 0.15
    fake_backend.close.assert_awaited_once()


def test_shutdown_swallows_backend_close_errors() -> None:
    """A failing close() must not prevent lifespan.shutdown from completing
    (which would otherwise surface as uvicorn's 'Application shutdown
    failed' and could hang the shutdown wait)."""
    fake_backend = AsyncMock()
    fake_backend.close = AsyncMock(side_effect=RuntimeError("boom"))
    server._backend = fake_backend

    app = _build_app_with_drain(drain_seconds=0.0)
    with TestClient(app):
        pass  # must not raise

    fake_backend.close.assert_awaited_once()
    assert server._backend is None


async def test_drain_and_close_backend_times_out_gracefully() -> None:
    """If close() hangs past close_timeout_seconds, the helper must give up
    rather than block shutdown forever - uvicorn places no timeout of its
    own around lifespan.shutdown. Passed explicitly as a parameter (not
    monkeypatched onto the module) - see _drain_and_close_backend's own
    docstring for why: a real signal-timing integration test needs to set
    this via a CLI flag in a subprocess, which monkeypatching can't reach."""
    import asyncio

    async def _hang() -> None:
        await asyncio.sleep(10)

    fake_backend = AsyncMock()
    fake_backend.close = AsyncMock(side_effect=_hang)
    server._backend = fake_backend

    await server._drain_and_close_backend(0.0, close_timeout_seconds=0.05)

    assert server._backend is None

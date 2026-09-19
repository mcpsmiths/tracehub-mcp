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
    server._secondary_backend = None
    yield
    server._backend = None
    server._secondary_backend = None


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


def test_shutdown_closes_both_primary_and_secondary_backend() -> None:
    fake_primary = AsyncMock()
    fake_secondary = AsyncMock()
    server._backend = fake_primary
    server._secondary_backend = fake_secondary

    app = _build_app_with_drain(drain_seconds=0.0)
    with TestClient(app):
        fake_primary.close.assert_not_called()
        fake_secondary.close.assert_not_called()

    fake_primary.close.assert_awaited_once()
    fake_secondary.close.assert_awaited_once()
    assert server._backend is None
    assert server._secondary_backend is None


async def test_shutdown_closes_secondary_even_when_primary_close_hangs() -> None:
    """A hanging primary close() must not starve the secondary's own close
    attempt - each is independently bounded by close_timeout_seconds."""
    import asyncio

    async def _hang() -> None:
        await asyncio.sleep(10)

    fake_primary = AsyncMock()
    fake_primary.close = AsyncMock(side_effect=_hang)
    fake_secondary = AsyncMock()
    server._backend = fake_primary
    server._secondary_backend = fake_secondary

    await server._drain_and_close_backend(0.0, close_timeout_seconds=0.05)

    fake_secondary.close.assert_awaited_once()
    assert server._backend is None
    assert server._secondary_backend is None


async def test_both_backends_close_concurrently_not_sequentially() -> None:
    """Regression: closing backends one after another would make total
    shutdown latency close_timeout_seconds * N instead of a single
    close_timeout_seconds - both must be in flight at the same time."""
    import asyncio

    async def _hang() -> None:
        await asyncio.sleep(10)

    fake_primary = AsyncMock()
    fake_primary.close = AsyncMock(side_effect=_hang)
    fake_secondary = AsyncMock()
    fake_secondary.close = AsyncMock(side_effect=_hang)
    server._backend = fake_primary
    server._secondary_backend = fake_secondary

    start = time.monotonic()
    await server._drain_and_close_backend(0.0, close_timeout_seconds=0.1)
    elapsed = time.monotonic() - start

    # Sequential closes of two hanging backends would take ~0.2s (2x the
    # per-backend timeout); concurrent closes take ~0.1s regardless of how
    # many backends are hanging.
    assert elapsed < 0.15


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

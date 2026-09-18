"""Proves BaseBackend.__init_subclass__'s generic query-cache wrapping
(see backends/base.py) through a real concrete backend, reusing
tests/test_jaeger.py's exact backend._client = AsyncMock(...) mocking
technique. No per-backend-file changes are needed for this wrapping to
apply - these tests are the proof.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

from opentelemetry_mcp.backends.jaeger import JaegerBackend


def _fake_response(payload: dict[str, Any]) -> AsyncMock:
    fake_response = AsyncMock()
    fake_response.raise_for_status = lambda: None
    fake_response.json = lambda: payload
    return fake_response


def _mocked_backend(payload: dict[str, Any]) -> tuple[JaegerBackend, AsyncMock]:
    """Returns (backend, mock_get) - mock_get is the exact AsyncMock
    assigned as backend._client.get, kept as its own typed reference so
    assertions on .call_count don't have to resolve through
    BaseBackend._client's strict `httpx.AsyncClient | None` annotation."""
    backend = JaegerBackend(url="http://localhost:16686")
    mock_get = AsyncMock(return_value=_fake_response(payload))
    backend._client = AsyncMock()
    backend._client.is_closed = False
    backend._client.get = mock_get
    return backend, mock_get


async def test_list_services_is_cached_across_repeated_calls() -> None:
    backend, mock_get = _mocked_backend({"data": ["svc-a"]})
    backend.configure_query_cache(60.0)

    first = await backend.list_services()
    second = await backend.list_services()

    assert first == ["svc-a"]
    assert second == ["svc-a"]
    assert mock_get.call_count == 1


async def test_caching_disabled_by_default_recomputes_every_call() -> None:
    backend, mock_get = _mocked_backend({"data": ["svc-a"]})
    # configure_query_cache() never called - matches production default.

    await backend.list_services()
    await backend.list_services()

    assert mock_get.call_count == 2


async def test_concurrent_identical_calls_are_coalesced() -> None:
    backend, mock_get = _mocked_backend({"data": ["svc-a"]})
    backend.configure_query_cache(60.0)

    results = await asyncio.gather(backend.list_services(), backend.list_services())

    assert list(results) == [["svc-a"], ["svc-a"]]
    assert mock_get.call_count == 1


async def test_different_arguments_are_not_coalesced() -> None:
    backend, mock_get = _mocked_backend({"data": ["op-a"]})
    backend.configure_query_cache(60.0)

    await backend.get_service_operations("service-a")
    await backend.get_service_operations("service-b")

    assert mock_get.call_count == 2


def test_health_check_is_excluded_from_query_caching() -> None:
    """health_check() is deliberately excluded from _CACHEABLE_METHODS -
    /ready and doctor both depend on it reflecting live state. Checked
    directly against the wrapping mechanism (functools.wraps sets
    __wrapped__ on a cached method), since JaegerBackend's own
    health_check() happens to delegate to the separately-cached
    list_services() internally, which would otherwise mask this exclusion
    at a call-count level."""
    assert not hasattr(JaegerBackend.health_check, "__wrapped__")
    assert hasattr(JaegerBackend.list_services, "__wrapped__")

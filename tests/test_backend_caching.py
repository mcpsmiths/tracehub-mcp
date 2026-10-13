"""Proves BaseBackend.__init_subclass__'s generic query-cache wrapping
(see backends/base.py) through a real concrete backend, reusing
tests/test_jaeger.py's exact backend._client = AsyncMock(...) mocking
technique. No per-backend-file changes are needed for this wrapping to
apply - these tests are the proof.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.backends.datadog import DatadogBackend
from opentelemetry_mcp.backends.honeycomb import HoneycombBackend
from opentelemetry_mcp.backends.jaeger import JaegerBackend
from opentelemetry_mcp.backends.newrelic import NewRelicBackend
from opentelemetry_mcp.backends.sentry import SentryBackend
from opentelemetry_mcp.backends.tempo import TempoBackend
from opentelemetry_mcp.backends.traceloop import TraceloopBackend
from opentelemetry_mcp.backends.xray import XRayBackend
from opentelemetry_mcp.models import Filter, FilterOperator, FilterType, TraceQuery


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
    __wrapped__ on a cached method); the live-probe behaviour itself is
    covered by the two tests below."""
    assert not hasattr(JaegerBackend.health_check, "__wrapped__")
    assert hasattr(JaegerBackend.list_services, "__wrapped__")
    assert not hasattr(JaegerBackend._list_services_uncached, "__wrapped__")


def _unauthorized_response() -> AsyncMock:
    request = httpx.Request("GET", "http://localhost:16686/api/services")
    response = httpx.Response(401, request=request)
    fake_response = AsyncMock()
    fake_response.raise_for_status = response.raise_for_status
    fake_response.json = lambda: {}
    return fake_response


async def test_jaeger_health_check_bypasses_query_cache() -> None:
    """Regression: JaegerBackend.health_check awaited the public, cache-
    wrapped list_services(), so with QUERY_CACHE_TTL_SECONDS set a warmed
    entry answered /ready and the Docker HEALTHCHECK "healthy" for up to a
    TTL after the backend failed or the credential was revoked."""
    backend, mock_get = _mocked_backend({"data": ["svc-a"]})
    backend.configure_query_cache(60.0)
    assert await backend.list_services() == ["svc-a"]  # warms the cache key

    mock_get.return_value = _unauthorized_response()
    health = await backend.health_check()

    assert health.status == "unhealthy"
    assert health.error is not None and "401" in health.error
    assert mock_get.call_count == 2


async def test_list_services_uncached_bypasses_cache() -> None:
    """The generic BaseBackend.list_services_uncached (what /ready and
    doctor probe with) invokes the pre-wrap original, so a cached entry
    never answers a readiness probe for any backend."""
    backend, mock_get = _mocked_backend({"data": ["svc-a"]})
    backend.configure_query_cache(60.0)
    await backend.list_services()
    await backend.list_services()
    assert mock_get.call_count == 1

    mock_get.return_value = _fake_response({"data": ["svc-a", "svc-live"]})
    live = await backend.list_services_uncached()

    assert live == ["svc-a", "svc-live"]
    assert mock_get.call_count == 2
    # The cached entry is untouched: the public method still serves it.
    assert await backend.list_services() == ["svc-a"]
    assert mock_get.call_count == 2


async def test_list_services_uncached_works_without_a_cache() -> None:
    backend, mock_get = _mocked_backend({"data": ["svc-a"]})

    assert await backend.list_services_uncached() == ["svc-a"]
    assert mock_get.call_count == 1


@pytest.mark.parametrize(
    "backend_cls",
    [
        JaegerBackend,
        TempoBackend,
        TraceloopBackend,
        DatadogBackend,
        SentryBackend,
        XRayBackend,
        NewRelicBackend,
        HoneycombBackend,
    ],
)
def test_every_backend_probes_through_the_uncached_base_method(
    backend_cls: type[BaseBackend],
) -> None:
    """No backend may shadow list_services_uncached with a cache-wrapped
    method: it is the one entry point _run_backend_checks trusts to be live."""
    assert backend_cls.list_services_uncached is BaseBackend.list_services_uncached
    assert not hasattr(backend_cls.list_services_uncached, "__wrapped__")
    assert hasattr(backend_cls.list_services, "__wrapped__")


async def test_cancelling_a_follower_does_not_fail_the_leader_end_to_end() -> None:
    """Through a real cache-wrapped backend method (not the cache class in
    isolation): a client that cancels its own coalesced list_services() call
    must not fail the leader's identical in-flight request."""
    backend, mock_get = _mocked_backend({"data": ["svc-a"]})
    backend.configure_query_cache(60.0)
    started = asyncio.Event()
    release = asyncio.Event()

    async def gated_get(*args: Any, **kwargs: Any) -> AsyncMock:
        started.set()
        await release.wait()
        return _fake_response({"data": ["svc-a"]})

    mock_get.side_effect = gated_get

    leader = asyncio.create_task(backend.list_services())
    await started.wait()
    follower = asyncio.create_task(backend.list_services())
    await asyncio.sleep(0)  # the follower joins the leader's in-flight future

    follower.cancel()
    with pytest.raises(asyncio.CancelledError):
        await follower

    release.set()
    assert await asyncio.wait_for(leader, timeout=1.0) == ["svc-a"]
    assert mock_get.call_count == 1


async def test_oversized_key_bypasses_cache() -> None:
    """A cache key is rendered from the validated arguments (model_dump_json),
    so its size tracks the request. For Jaeger a non-service.name filter is
    client-side, so a request-sized filter value costs the backend one small
    GET that succeeds empty - and used to mint a request-sized cache entry.
    Oversized keys now bypass the cache entirely: computed, never stored."""
    backend, mock_get = _mocked_backend({"data": []})
    backend.configure_query_cache(60.0)
    huge_filter = Filter(
        field="http.target",
        operator=FilterOperator.EQUALS,
        value="x" * (100 * 1024),
        value_type=FilterType.STRING,
    )
    oversized_query = TraceQuery(service_name="svc-a", filters=[huge_filter], limit=5)

    assert await backend.search_traces(oversized_query) == []
    assert await backend.search_traces(oversized_query) == []

    assert mock_get.call_count == 2
    assert backend._query_cache is not None
    assert len(backend._query_cache._entries) == 0

    normal_query = TraceQuery(service_name="svc-a", limit=5)
    await backend.search_traces(normal_query)
    await backend.search_traces(normal_query)

    assert mock_get.call_count == 3
    assert len(backend._query_cache._entries) == 1


def test_configure_query_cache_passes_the_entry_bound_through() -> None:
    backend, _ = _mocked_backend({"data": []})

    backend.configure_query_cache(60.0, max_entries=7)

    assert backend._query_cache is not None
    assert backend._query_cache._max_entries == 7

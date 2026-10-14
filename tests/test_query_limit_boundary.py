"""MAX_TRACES_PER_QUERY is enforced at the backend boundary, not only in the
tool handlers that happen to call server._clamp_limit.

Before this boundary existed the ceiling was applied to the caller's
requested result count only: get_llm_expensive_traces/get_llm_slow_traces
re-expanded it (limit * 10), handlers without a `limit` argument never
clamped at all, and every backend multiplied query.limit again before
issuing requests - one call fetched 10x-50x the operator's ceiling.
BaseBackend.__init_subclass__ now wraps search_traces/search_spans with a
query-limit clamp OUTSIDE the cache wrapper, so every present and future
caller is covered and the cache key is computed from the clamped query.
"""

# server.py imports its tools submodules without an explicit re-export; see
# tests/test_server.py for why attr-defined is disabled file-wide here too.
# mypy: disable-error-code="attr-defined"

from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastmcp import Client
from pydantic import HttpUrl

from opentelemetry_mcp import server
from opentelemetry_mcp.attributes import HealthCheckResponse
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.backends.datadog import DatadogBackend
from opentelemetry_mcp.backends.tempo import TempoBackend
from opentelemetry_mcp.config import BackendConfig, ServerConfig
from opentelemetry_mcp.models import FilterOperator, SpanData, SpanQuery, TraceData, TraceQuery
from opentelemetry_mcp.tools import expensive_traces

_BASE_TIME = datetime(2024, 1, 1, tzinfo=UTC)


def _trace(trace_id: str = "t1", service_name: str = "svc-a") -> TraceData:
    span = SpanData(
        trace_id=trace_id,
        span_id="s1",
        parent_span_id=None,
        operation_name="op",
        service_name=service_name,
        start_time=_BASE_TIME,
        duration_ms=10.0,
        status="OK",
    )
    return TraceData(
        trace_id=trace_id,
        spans=[span],
        start_time=_BASE_TIME,
        duration_ms=10.0,
        service_name=service_name,
        root_operation="op",
        status="OK",
    )


class RecordingBackend(BaseBackend):
    """Minimal concrete backend that records every query object it is handed
    (the object the wrappers deliver, not the one the caller built)."""

    def __init__(self, url: str = "http://recording.invalid", fail_get_trace: bool = False):
        super().__init__(url)
        self.recorded: list[TraceQuery | SpanQuery] = []
        self.fail_get_trace = fail_get_trace

    def _create_headers(self) -> dict[str, str]:
        return {}

    def get_supported_operators(self) -> set[FilterOperator]:
        return set()

    async def search_traces(self, query: TraceQuery) -> list[TraceData]:
        self.recorded.append(query)
        return []

    async def search_spans(self, query: SpanQuery) -> list[SpanData]:
        self.recorded.append(query)
        return []

    async def get_trace(self, trace_id: str) -> TraceData:
        if self.fail_get_trace:
            raise ValueError("not found")
        return _trace(trace_id)

    async def list_services(self) -> list[str]:
        return ["svc-a"]

    async def get_service_operations(self, service_name: str) -> list[str]:
        return ["op"]

    async def health_check(self) -> HealthCheckResponse:
        return HealthCheckResponse(status="healthy", backend="jaeger", url=self.url)


class TestBoundaryClamp:
    async def test_search_traces_limit_is_clamped_to_max_query_limit(self) -> None:
        backend = RecordingBackend()
        backend.max_query_limit = 50
        query = TraceQuery(limit=500)

        await backend.search_traces(query)

        assert backend.recorded[-1].limit == 50
        # Never mutate the caller's query: a new object is handed down.
        assert query.limit == 500
        assert backend.recorded[-1] is not query

    async def test_search_spans_limit_is_clamped_to_max_query_limit(self) -> None:
        backend = RecordingBackend()
        backend.max_query_limit = 50
        query = SpanQuery(limit=500)

        await backend.search_spans(query)

        assert backend.recorded[-1].limit == 50
        assert query.limit == 500
        assert backend.recorded[-1] is not query

    async def test_keyword_query_argument_is_clamped_too(self) -> None:
        backend = RecordingBackend()
        backend.max_query_limit = 7

        await backend.search_traces(query=TraceQuery(limit=100))

        assert backend.recorded[-1].limit == 7

    async def test_limit_at_or_below_the_ceiling_passes_the_same_object_through(self) -> None:
        backend = RecordingBackend()
        backend.max_query_limit = 50
        query = TraceQuery(limit=50)

        await backend.search_traces(query)

        assert backend.recorded[-1] is query

    async def test_no_ceiling_leaves_the_limit_unchanged(self) -> None:
        backend = RecordingBackend()
        assert backend.max_query_limit is None
        query = SpanQuery(limit=500)

        await backend.search_spans(query)

        assert backend.recorded[-1] is query
        assert backend.recorded[-1].limit == 500

    async def test_cache_key_is_built_from_the_clamped_query(self) -> None:
        """Clamp outside, cache inside: two requests that differ only in an
        over-ceiling limit are the same backend query and share one entry."""
        backend = RecordingBackend()
        backend.max_query_limit = 50
        backend.configure_query_cache(60.0)

        await backend.search_traces(TraceQuery(service_name="svc-a", limit=500))
        await backend.search_traces(TraceQuery(service_name="svc-a", limit=1000))

        assert len(backend.recorded) == 1
        assert backend.recorded[0].limit == 50

    def test_wrapper_chain_keeps_functools_wraps_metadata(self) -> None:
        # __wrapped__ chains clamp -> cache -> original, so introspection that
        # unwraps a cached method (tests/test_backend_caching.py) still works.
        assert hasattr(RecordingBackend.search_traces, "__wrapped__")
        assert hasattr(RecordingBackend.search_traces.__wrapped__, "__wrapped__")


class TestServerWiring:
    @pytest.fixture(autouse=True)
    def _reset_server_globals(self) -> Generator[None]:
        server._backend = None
        server._secondary_backend = None
        server._config = None
        yield
        server._backend = None
        server._secondary_backend = None
        server._config = None

    def _config(self) -> ServerConfig:
        return ServerConfig(
            backend=BackendConfig(type="jaeger", url=HttpUrl("http://localhost:16686")),
            secondary_backend=BackendConfig(type="jaeger", url=HttpUrl("http://localhost:16687")),
        )

    def test_create_backend_applies_the_ceiling(self) -> None:
        config = self._config()
        config.max_traces_per_query = 42

        backend = server._create_backend(config)

        assert backend.max_query_limit == 42

    async def test_secondary_backend_gets_the_ceiling_too(self) -> None:
        server._config = self._config()
        server._config.max_traces_per_query = 3
        secondary = RecordingBackend()

        with patch.object(server, "_build_backend_from_config", return_value=secondary):
            wired = await server._get_secondary_backend()

        assert wired is secondary
        assert secondary.max_query_limit == 3

    async def test_every_tool_stays_within_a_ceiling_of_one(self) -> None:
        """Drives every registered tool through the real FastMCP dispatcher
        with MAX_TRACES_PER_QUERY=1 - including the two ranking tools that
        re-expand limit * 10, the handler with no limit argument
        (get_llm_model_stats) and correlate_trace's per-service fallback
        search against the secondary backend."""
        server._config = self._config()
        server._config.max_traces_per_query = 1
        primary = RecordingBackend()
        secondary = RecordingBackend(url="http://secondary.invalid", fail_get_trace=True)
        with patch.object(server, "_build_backend_from_config", side_effect=[primary, secondary]):
            server._backend = server._create_backend(server._config)
            await server._get_secondary_backend()

        calls: list[tuple[str, dict[str, Any]]] = [
            ("search_traces", {}),
            ("search_spans_tool", {}),
            ("find_errors", {}),
            ("get_llm_usage", {}),
            ("list_llm_models", {}),
            ("get_llm_model_stats", {"model_name": "gpt-4"}),
            ("list_sessions", {}),
            ("get_session_stats", {"conversation_id": "c1"}),
            ("compare_time_windows", {}),
            (
                "investigate_cost_spike",
                {"recent_start": "2024-01-01T00:00:00Z", "recent_end": "2024-01-01T01:00:00Z"},
            ),
            (
                "investigate_error_spike",
                {"recent_start": "2024-01-01T00:00:00Z", "recent_end": "2024-01-01T01:00:00Z"},
            ),
            ("get_prompt_version_stats", {}),
            ("list_llm_tools_tool", {}),
            ("get_llm_expensive_traces", {"limit": 1000}),
            ("get_llm_slow_traces", {"limit": 1000}),
            ("correlate_trace", {"trace_id": "t1"}),
        ]
        errors: list[str] = []
        async with Client(server.mcp) as client:
            for name, arguments in calls:
                result = await client.call_tool(name, arguments, raise_on_error=False)
                if result.is_error:
                    errors.append(f"{name}: {result.content}")

        assert errors == []
        observed = primary.recorded + secondary.recorded
        assert len(primary.recorded) >= len(calls) - 1  # every search tool reached the backend
        assert secondary.recorded, "correlate_trace's fallback search never reached the secondary"
        assert all(q.limit <= 1 for q in observed), [q.limit for q in observed]


class TestBackendMultipliersUnderTheClamp:
    async def test_tempo_search_spans_caps_its_doubled_limit_at_1000(self) -> None:
        """tempo.py doubled query.limit into TraceQuery without jaeger.py's
        min(..., 1000) guard: a clamped limit of 501..1000 raised a pydantic
        ValidationError instead of a request."""
        backend = TempoBackend(url="http://localhost:3200")
        seen_limits: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_limits.append(request.url.params["limit"])
            return httpx.Response(200, json={"traces": []})

        backend._client = httpx.AsyncClient(
            base_url=backend.url, transport=httpx.MockTransport(handler)
        )

        assert await backend.search_spans(SpanQuery(limit=501)) == []
        assert seen_limits == ["1000"]

    async def test_datadog_over_fetch_is_bounded_by_the_ceiling(self) -> None:
        """get_llm_expensive_traces(limit=50) builds TraceQuery(limit=500);
        Datadog then multiplies by 5 - 2500 spans against a ceiling of 50.
        With the boundary clamp the span request is 50 * 5 = 250 at most."""
        backend = DatadogBackend(
            url="https://api.datadoghq.com", api_key="dd-key1", app_key="dd-app1"
        )
        backend.max_query_limit = 50
        raw = AsyncMock(return_value=[])

        with patch.object(backend, "_search_spans_raw", raw):
            await expensive_traces.get_expensive_traces(backend, limit=50)

        raw.assert_awaited_once()
        assert raw.await_args is not None
        requested = raw.await_args.args[3]
        assert requested <= 250, requested

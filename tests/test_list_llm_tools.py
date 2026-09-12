"""Tests for the list_llm_tools tool.

The tool identifies LLM "tool call" spans via a fixed
`traceloop.span.kind == tool` filter, fans out over a single
`backend.search_spans` call, and aggregates per-tool-name usage counts,
distinct services, and first/last-seen timestamps. The backend is mocked at
the BaseBackend interface level (not HTTP) since this module only ever
talks to `backend.search_spans`.
"""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

from opentelemetry.semconv_ai import TraceloopSpanKindValues

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.constants import Traceloop
from opentelemetry_mcp.models import FilterOperator, SpanData, SpanQuery
from opentelemetry_mcp.tools.list_llm_tools import list_llm_tools


def _fake_backend() -> AsyncMock:
    """An AsyncMock honoring BaseBackend's interface (search_spans is what
    this tool actually calls)."""
    return AsyncMock(spec=BaseBackend)


def _tool_span(**overrides: object) -> SpanData:
    """Build a span shaped like a `traceloop.span.kind == tool` call.

    Based on the shape of conftest.py's sample_span_data fixture, extended
    with the traceloop.span.kind attribute this module filters on.
    """
    defaults: dict[str, object] = dict(
        trace_id="abc123",
        span_id="span1",
        parent_span_id=None,
        operation_name="search_database",
        service_name="test-service",
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        duration_ms=50.0,
        status="OK",
        attributes=SpanAttributes.model_validate(
            {"traceloop.span.kind": TraceloopSpanKindValues.TOOL.value}
        ),
    )
    defaults.update(overrides)
    return SpanData(**defaults)  # type: ignore[arg-type]


class TestListLlmToolsHappyPath:
    """A successful backend call must transform raw spans into the
    documented {"count", "total_calls", "tools": [...]} shape, grouped and
    aggregated by operation_name."""

    async def test_single_tool_produces_correct_shape(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = [_tool_span()]

        raw = await list_llm_tools(backend, service_name="test-service")
        result = json.loads(raw)

        assert result["count"] == 1
        assert result["total_calls"] == 1
        tool = result["tools"][0]
        assert tool["tool_name"] == "search_database"
        assert tool["usage_count"] == 1
        assert tool["services"] == ["test-service"]
        assert tool["first_seen"] == "2024-01-01T00:00:00Z"
        assert tool["last_seen"] == "2024-01-01T00:00:00Z"

    async def test_spans_with_same_operation_name_are_grouped(self) -> None:
        """Real aggregation logic: two spans sharing operation_name must
        collapse into one tool entry with usage_count == 2, not two entries."""
        backend = _fake_backend()
        backend.search_spans.return_value = [
            _tool_span(span_id="s1"),
            _tool_span(span_id="s2"),
        ]

        raw = await list_llm_tools(backend)
        result = json.loads(raw)

        assert result["count"] == 1
        assert result["tools"][0]["usage_count"] == 2
        assert result["total_calls"] == 2

    async def test_services_list_is_deduplicated_and_sorted(self) -> None:
        """A tool called from multiple services must list each service once,
        in sorted order - not raw insertion order or duplicates."""
        backend = _fake_backend()
        backend.search_spans.return_value = [
            _tool_span(span_id="s1", service_name="zeta-service"),
            _tool_span(span_id="s2", service_name="alpha-service"),
            _tool_span(span_id="s3", service_name="alpha-service"),
        ]

        raw = await list_llm_tools(backend)
        result = json.loads(raw)

        assert result["tools"][0]["services"] == ["alpha-service", "zeta-service"]

    async def test_first_seen_and_last_seen_track_min_and_max_out_of_order(self) -> None:
        """Spans intentionally out of chronological order to exercise the
        min/max comparison logic rather than just 'first span wins'."""
        backend = _fake_backend()
        earliest = datetime(2024, 1, 1, tzinfo=UTC)
        middle = datetime(2024, 1, 5, tzinfo=UTC)
        latest = datetime(2024, 1, 10, tzinfo=UTC)
        backend.search_spans.return_value = [
            _tool_span(span_id="s2", start_time=middle),
            _tool_span(span_id="s1", start_time=earliest),
            _tool_span(span_id="s3", start_time=latest),
        ]

        raw = await list_llm_tools(backend)
        result = json.loads(raw)

        tool = result["tools"][0]
        assert tool["first_seen"] == earliest.isoformat().replace("+00:00", "Z")
        assert tool["last_seen"] == latest.isoformat().replace("+00:00", "Z")

    async def test_tools_sorted_by_usage_count_descending(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = [
            _tool_span(span_id="s1", operation_name="rare_tool"),
            _tool_span(span_id="s2", operation_name="popular_tool"),
            _tool_span(span_id="s3", operation_name="popular_tool"),
            _tool_span(span_id="s4", operation_name="popular_tool"),
        ]

        raw = await list_llm_tools(backend)
        result = json.loads(raw)

        assert result["count"] == 2
        assert result["tools"][0]["tool_name"] == "popular_tool"
        assert result["tools"][0]["usage_count"] == 3
        assert result["tools"][1]["tool_name"] == "rare_tool"
        assert result["tools"][1]["usage_count"] == 1


class TestListLlmToolsQueryConstruction:
    """The tool must build a SpanQuery with a fixed traceloop.span.kind ==
    tool filter, plus the caller's own parameters forwarded."""

    async def test_forwards_span_kind_tool_filter_and_params(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = []

        await list_llm_tools(
            backend,
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-02T00:00:00Z",
            service_name="svc-a",
            gen_ai_system="anthropic",
            limit=50,
        )

        query = backend.search_spans.call_args.args[0]
        assert isinstance(query, SpanQuery)
        assert query.service_name == "svc-a"
        assert query.gen_ai_system == "anthropic"
        assert query.limit == 50
        assert query.start_time == datetime(2024, 1, 1, tzinfo=UTC)
        assert query.end_time == datetime(2024, 1, 2, tzinfo=UTC)
        assert len(query.filters) == 1
        assert query.filters[0].field == Traceloop.SPAN_KIND
        assert query.filters[0].operator == FilterOperator.EQUALS
        assert query.filters[0].value == TraceloopSpanKindValues.TOOL.value

    async def test_default_limit_is_1000_when_unspecified(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = []

        await list_llm_tools(backend)

        query = backend.search_spans.call_args.args[0]
        assert query.limit == 1000


class TestListLlmToolsTimestampValidation:
    """Invalid ISO timestamps must surface as {"error": ...} JSON, per
    parse_iso_timestamp's (value, error) return shape - never raise."""

    async def test_invalid_start_time_returns_error_json(self) -> None:
        backend = _fake_backend()

        raw = await list_llm_tools(backend, start_time="not-a-timestamp")
        result = json.loads(raw)

        assert "error" in result
        assert "start_time" in result["error"]
        backend.search_spans.assert_not_awaited()

    async def test_invalid_end_time_returns_error_json(self) -> None:
        backend = _fake_backend()

        raw = await list_llm_tools(backend, end_time="also-not-a-timestamp")
        result = json.loads(raw)

        assert "error" in result
        assert "end_time" in result["error"]
        backend.search_spans.assert_not_awaited()

    async def test_start_time_checked_before_end_time(self) -> None:
        """Both start_time and end_time are invalid - start_time is parsed
        first, so its failure must be the one reported."""
        backend = _fake_backend()

        raw = await list_llm_tools(backend, start_time="bad-start", end_time="bad-end")
        result = json.loads(raw)

        assert "start_time" in result["error"]

    async def test_valid_iso_timestamps_reach_the_query(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = []

        await list_llm_tools(
            backend,
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-02T00:00:00Z",
        )

        query = backend.search_spans.call_args.args[0]
        assert query.start_time == datetime(2024, 1, 1, tzinfo=UTC)
        assert query.end_time == datetime(2024, 1, 2, tzinfo=UTC)


class TestListLlmToolsBackendExceptionHandling:
    """A backend that raises must be caught and reported as error JSON,
    never left to propagate out of the tool."""

    async def test_backend_exception_becomes_error_json(self) -> None:
        backend = _fake_backend()
        backend.search_spans.side_effect = RuntimeError("upstream unavailable")

        raw = await list_llm_tools(backend, service_name="svc")
        result = json.loads(raw)

        assert result == {"error": "Failed to list LLM tools: upstream unavailable"}

    async def test_backend_exception_does_not_propagate(self) -> None:
        backend = _fake_backend()
        backend.search_spans.side_effect = ConnectionError("boom")

        # Must not raise - the tool contract is always to return a JSON string.
        raw = await list_llm_tools(backend)
        assert json.loads(raw)["error"]


class TestListLlmToolsEdgeCases:
    """Edge cases in the module's own branches: empty results, and a
    tool used by only a single service."""

    async def test_empty_result_list_returns_documented_message(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = []

        raw = await list_llm_tools(backend, service_name="nonexistent")
        result = json.loads(raw)

        assert result == {
            "count": 0,
            "tools": [],
            "message": "No LLM tool spans found matching the criteria",
        }

    async def test_single_service_tool_has_single_element_list(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = [_tool_span()]

        raw = await list_llm_tools(backend)
        result = json.loads(raw)

        assert result["tools"][0]["services"] == ["test-service"]


class TestListLlmToolsLimitValidation:
    """SpanQuery(limit=...) construction is wrapped in its own try/except
    (matching the convention in tools/errors.py), so an out-of-range limit
    must surface as the documented {"error": "Invalid query parameters: ..."}
    JSON shape rather than raising a raw pydantic ValidationError.
    """

    async def test_limit_out_of_bounds_returns_error_json(self) -> None:
        backend = _fake_backend()

        raw = await list_llm_tools(backend, limit=0)
        result = json.loads(raw)

        assert "error" in result
        assert result["error"].startswith("Invalid query parameters:")
        backend.search_spans.assert_not_awaited()

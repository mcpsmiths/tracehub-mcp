"""Tests for the search_traces tool."""

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.models import SpanData, TraceData, TraceQuery
from opentelemetry_mcp.tools.search import search_traces


def _make_span(
    *,
    trace_id: str = "t1",
    span_id: str = "s1",
    service_name: str = "svc",
    operation_name: str = "op",
    duration_ms: float = 10.0,
    status: str = "OK",
    gen_ai: dict[str, Any] | None = None,
) -> SpanData:
    """Build a SpanData, optionally carrying gen_ai.* attributes."""
    return SpanData(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=None,
        operation_name=operation_name,
        service_name=service_name,
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        duration_ms=duration_ms,
        status=status,  # type: ignore[arg-type]
        attributes=SpanAttributes.model_validate(gen_ai or {}),
    )


def _make_trace(
    *,
    trace_id: str = "t1",
    spans: list[SpanData] | None = None,
    service_name: str = "svc",
    root_operation: str = "op",
    status: str = "OK",
    duration_ms: float = 10.0,
) -> TraceData:
    """Build a TraceData wrapping the given spans (defaults to one plain span)."""
    resolved_spans = spans if spans is not None else [_make_span(trace_id=trace_id)]
    return TraceData(
        trace_id=trace_id,
        spans=resolved_spans,
        start_time=resolved_spans[0].start_time,
        duration_ms=duration_ms,
        service_name=service_name,
        root_operation=root_operation,
        status=status,  # type: ignore[arg-type]
    )


def _fake_backend() -> AsyncMock:
    return AsyncMock(spec=BaseBackend)


class TestSearchTracesHappyPath:
    """Backend data -> correct JSON shape via TraceSummary transform."""

    async def test_returns_correct_top_level_shape_and_summary_fields(self) -> None:
        backend = _fake_backend()
        llm_span = _make_span(
            trace_id="t1",
            span_id="s1",
            gen_ai={
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4",
                "gen_ai.usage.total_tokens": 300,
            },
        )
        plain_span = _make_span(trace_id="t1", span_id="s2")
        trace = _make_trace(
            trace_id="t1",
            spans=[llm_span, plain_span],
            service_name="my-service",
            root_operation="root-op",
            status="OK",
        )
        backend.search_traces.return_value = [trace]

        result = json.loads(await search_traces(backend))

        assert set(result.keys()) == {"count", "traces"}
        assert result["count"] == 1
        summary = result["traces"][0]
        assert summary["trace_id"] == "t1"
        assert summary["service_name"] == "my-service"
        assert summary["operation_name"] == "root-op"
        assert summary["status"] == "OK"
        # Real transform logic: span_count/llm_span_count/total_tokens/has_errors
        # are derived from the trace's spans, not just echoed from input.
        assert summary["span_count"] == 2
        assert summary["llm_span_count"] == 1
        assert summary["total_tokens"] == 300
        assert summary["has_errors"] is False

    async def test_empty_backend_result_returns_empty_list(self) -> None:
        backend = _fake_backend()
        backend.search_traces.return_value = []

        result = json.loads(await search_traces(backend))

        assert result == {"count": 0, "traces": []}

    async def test_error_span_propagates_has_errors_to_summary(self) -> None:
        backend = _fake_backend()
        error_span = _make_span(trace_id="t2", span_id="s1", status="ERROR")
        trace = _make_trace(trace_id="t2", spans=[error_span], status="ERROR")
        backend.search_traces.return_value = [trace]

        result = json.loads(await search_traces(backend))

        assert result["traces"][0]["has_errors"] is True

    async def test_multiple_traces_are_all_included_in_order(self) -> None:
        backend = _fake_backend()
        trace_a = _make_trace(trace_id="a")
        trace_b = _make_trace(trace_id="b")
        backend.search_traces.return_value = [trace_a, trace_b]

        result = json.loads(await search_traces(backend))

        assert result["count"] == 2
        assert [t["trace_id"] for t in result["traces"]] == ["a", "b"]


class TestSearchTracesInputValidation:
    """Bad input must produce {"error": ...} JSON, never raise."""

    async def test_invalid_start_time_returns_error_without_raising(self) -> None:
        backend = _fake_backend()

        result = json.loads(await search_traces(backend, start_time="not-a-timestamp"))

        assert "error" in result
        assert "start_time" in result["error"]
        backend.search_traces.assert_not_called()

    async def test_invalid_end_time_returns_error_without_raising(self) -> None:
        backend = _fake_backend()

        result = json.loads(
            await search_traces(
                backend, start_time="2024-01-01T00:00:00Z", end_time="also-not-a-timestamp"
            )
        )

        assert "error" in result
        assert "end_time" in result["error"]
        backend.search_traces.assert_not_called()

    async def test_filter_missing_required_value_returns_validation_error(self) -> None:
        """Filter.validate_filter_values requires 'value' for EQUALS-type
        operators; the ValidationError must be caught and formatted."""
        backend = _fake_backend()
        bad_filter = {
            "field": "gen_ai.system",
            "operator": "equals",
            "value_type": "string",
            # 'value' intentionally omitted
        }

        result = json.loads(await search_traces(backend, filters=[bad_filter]))

        assert "error" in result
        assert "Invalid filter format" in result["error"]
        backend.search_traces.assert_not_called()

    async def test_filter_entry_that_is_not_a_mapping_returns_generic_parse_error(self) -> None:
        """A non-dict entry in `filters` can't hit the ValidationError branch
        (Filter(**entry) raises TypeError before pydantic validation runs) -
        it must fall into the generic except instead."""
        backend = _fake_backend()

        result = json.loads(await search_traces(backend, filters=["not-a-dict"]))  # type: ignore[list-item]

        assert "error" in result
        assert "Failed to parse filters" in result["error"]
        backend.search_traces.assert_not_called()

    async def test_limit_above_max_returns_invalid_query_parameters_error(self) -> None:
        backend = _fake_backend()

        result = json.loads(await search_traces(backend, limit=1001))

        assert "error" in result
        assert "Invalid query parameters" in result["error"]
        backend.search_traces.assert_not_called()

    async def test_negative_min_duration_returns_invalid_query_parameters_error(self) -> None:
        backend = _fake_backend()

        result = json.loads(await search_traces(backend, min_duration_ms=-1))

        assert "error" in result
        assert "Invalid query parameters" in result["error"]
        backend.search_traces.assert_not_called()


class TestSearchTracesBackendExceptionHandling:
    """A raising backend must be caught and turned into error JSON."""

    async def test_backend_exception_is_caught_and_formatted(self) -> None:
        backend = _fake_backend()
        backend.search_traces.side_effect = RuntimeError("connection reset")

        result = json.loads(await search_traces(backend))

        assert result == {"error": "Failed to search traces: connection reset"}

    async def test_backend_exception_after_valid_filters_still_caught(self) -> None:
        """Exercise the case where filter parsing succeeds but the actual
        backend call still fails - the two try/except blocks are separate."""
        backend = _fake_backend()
        backend.search_traces.side_effect = ValueError("Jaeger backend requires 'service_name'")

        result = json.loads(
            await search_traces(
                backend,
                filters=[
                    {
                        "field": "gen_ai.system",
                        "operator": "equals",
                        "value": "openai",
                        "value_type": "string",
                    }
                ],
            )
        )

        assert result == {
            "error": "Failed to search traces: Jaeger backend requires 'service_name'"
        }


class TestSearchTracesQueryConstruction:
    """The query object actually forwarded to the backend reflects real
    parsing/transform logic (parsed filters, parsed timestamps, limits)."""

    async def test_valid_explicit_filter_is_parsed_and_forwarded(self) -> None:
        backend = _fake_backend()
        backend.search_traces.return_value = []

        await search_traces(
            backend,
            filters=[
                {
                    "field": "gen_ai.usage.prompt_tokens",
                    "operator": "gt",
                    "value": 1000,
                    "value_type": "number",
                }
            ],
        )

        query = backend.search_traces.call_args.args[0]
        assert isinstance(query, TraceQuery)
        assert len(query.filters) == 1
        assert query.filters[0].field == "gen_ai.usage.prompt_tokens"
        assert query.filters[0].value == 1000

    async def test_timestamps_are_parsed_to_datetimes_before_reaching_backend(self) -> None:
        backend = _fake_backend()
        backend.search_traces.return_value = []

        await search_traces(
            backend, start_time="2024-01-01T00:00:00Z", end_time="2024-01-02T00:00:00Z"
        )

        query = backend.search_traces.call_args.args[0]
        assert query.start_time == datetime(2024, 1, 1, tzinfo=UTC)
        assert query.end_time == datetime(2024, 1, 2, tzinfo=UTC)

    async def test_limit_boundary_values_are_accepted(self) -> None:
        backend = _fake_backend()
        backend.search_traces.return_value = []

        for boundary in (1, 1000):
            await search_traces(backend, limit=boundary)
            query = backend.search_traces.call_args.args[0]
            assert query.limit == boundary

    async def test_no_filters_or_params_produces_empty_filter_list(self) -> None:
        backend = _fake_backend()
        backend.search_traces.return_value = []

        await search_traces(backend)

        query = backend.search_traces.call_args.args[0]
        assert query.filters == []

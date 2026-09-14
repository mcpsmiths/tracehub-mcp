"""Tests for the get_slow_traces tool."""

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.models import SpanData, TraceData, TraceQuery
from opentelemetry_mcp.tools.slow_traces import get_slow_traces


def _llm_span(
    *,
    trace_id: str = "t1",
    span_id: str = "s1",
    service_name: str = "svc",
    gen_ai_system: str = "openai",
    request_model: str | None = "gpt-4",
    response_model: str | None = None,
    total_tokens: int | None = 150,
) -> SpanData:
    """Build an LLM span with configurable gen_ai attributes."""
    attrs: dict[str, Any] = {"gen_ai.system": gen_ai_system}
    if request_model is not None:
        attrs["gen_ai.request.model"] = request_model
    if response_model is not None:
        attrs["gen_ai.response.model"] = response_model
    if total_tokens is not None:
        attrs["gen_ai.usage.total_tokens"] = total_tokens

    return SpanData(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=None,
        operation_name="chat",
        service_name=service_name,
        start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        duration_ms=100.0,
        status="OK",
        attributes=SpanAttributes.model_validate(attrs),
    )


def _non_llm_span(
    *, trace_id: str = "t1", span_id: str = "s2", service_name: str = "svc"
) -> SpanData:
    """Build a span with no gen_ai.* attributes at all."""
    return SpanData(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=None,
        operation_name="db_query",
        service_name=service_name,
        start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        duration_ms=10.0,
        status="OK",
        attributes=SpanAttributes.model_validate({}),
    )


def _trace(
    trace_id: str,
    spans: list[SpanData],
    *,
    service_name: str = "svc",
    root_operation: str = "chat",
    duration_ms: float = 1000.0,
    status: str = "OK",
) -> TraceData:
    """Build a TraceData instance from a list of spans."""
    return TraceData(
        trace_id=trace_id,
        spans=spans,
        start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        duration_ms=duration_ms,
        service_name=service_name,
        root_operation=root_operation,
        status=status,  # type: ignore[arg-type]
    )


def _mock_backend() -> AsyncMock:
    """A backend double scoped to BaseBackend's actual async methods."""
    return AsyncMock(spec=BaseBackend)


class TestHappyPath:
    """A realistic backend response must be transformed into the correct
    sorted JSON shape."""

    async def test_returns_slow_traces_sorted_by_duration_desc(self) -> None:
        backend = _mock_backend()
        fast_trace = _trace(
            "fast", [_llm_span(trace_id="fast")], duration_ms=100.0, root_operation="fast-op"
        )
        slow_trace = _trace(
            "slow", [_llm_span(trace_id="slow")], duration_ms=9000.0, root_operation="slow-op"
        )
        backend.search_traces.return_value = [fast_trace, slow_trace]

        result = json.loads(await get_slow_traces(backend))

        assert result["count"] == 2
        assert [t["trace_id"] for t in result["traces"]] == ["slow", "fast"]
        top = result["traces"][0]
        assert top["service_name"] == "svc"
        assert top["operation_name"] == "slow-op"
        assert top["duration_ms"] == 9000.0
        assert top["status"] == "OK"
        assert top["has_errors"] is False

    async def test_computes_total_tokens_across_multiple_llm_spans(self) -> None:
        backend = _mock_backend()
        span1 = _llm_span(trace_id="t1", span_id="s1", total_tokens=100)
        span2 = _llm_span(trace_id="t1", span_id="s2", total_tokens=50)
        trace = _trace("t1", [span1, span2])
        backend.search_traces.return_value = [trace]

        result = json.loads(await get_slow_traces(backend))

        assert result["traces"][0]["total_tokens"] == 150
        assert result["traces"][0]["llm_span_count"] == 2

    async def test_models_used_are_deduped_and_sorted(self) -> None:
        backend = _mock_backend()
        span_gpt = _llm_span(trace_id="t1", span_id="s1", request_model="gpt-4")
        span_claude = _llm_span(trace_id="t1", span_id="s2", request_model="claude-3")
        span_gpt_again = _llm_span(trace_id="t1", span_id="s3", request_model="gpt-4")
        trace = _trace("t1", [span_gpt, span_claude, span_gpt_again])
        backend.search_traces.return_value = [trace]

        result = json.loads(await get_slow_traces(backend))

        assert result["traces"][0]["models"] == ["claude-3", "gpt-4"]
        assert result["traces"][0]["llm_span_count"] == 3

    async def test_response_model_takes_precedence_over_request_model(self) -> None:
        backend = _mock_backend()
        span = _llm_span(request_model="gpt-4", response_model="gpt-4-0613")
        trace = _trace("t1", [span])
        backend.search_traces.return_value = [trace]

        result = json.loads(await get_slow_traces(backend))

        assert result["traces"][0]["models"] == ["gpt-4-0613"]

    async def test_duration_ms_is_rounded_to_two_decimals(self) -> None:
        backend = _mock_backend()
        trace = _trace("t1", [_llm_span()], duration_ms=1234.5678)
        backend.search_traces.return_value = [trace]

        result = json.loads(await get_slow_traces(backend))

        assert result["traces"][0]["duration_ms"] == 1234.57


class TestQueryConstruction:
    """The tool fetches a larger candidate set than `limit` so sorting can
    surface the true top N, and forwards every filter parameter."""

    async def test_fetch_limit_is_ten_times_requested_limit(self) -> None:
        backend = _mock_backend()
        backend.search_traces.return_value = []

        await get_slow_traces(backend, limit=5)

        query = backend.search_traces.call_args.args[0]
        assert isinstance(query, TraceQuery)
        assert query.limit == 50

    async def test_fetch_limit_is_capped_at_backend_maximum(self) -> None:
        backend = _mock_backend()
        backend.search_traces.return_value = []

        await get_slow_traces(backend, limit=200)

        query = backend.search_traces.call_args.args[0]
        assert query.limit == 1000

    async def test_min_duration_ms_is_forwarded_to_query(self) -> None:
        backend = _mock_backend()
        backend.search_traces.return_value = []

        await get_slow_traces(backend, min_duration_ms=500)

        query = backend.search_traces.call_args.args[0]
        assert query.min_duration_ms == 500

    async def test_model_and_service_filters_are_forwarded_to_query(self) -> None:
        backend = _mock_backend()
        backend.search_traces.return_value = []

        await get_slow_traces(
            backend,
            service_name="svc-a",
            gen_ai_request_model="gpt-4",
            gen_ai_response_model="gpt-4-0613",
        )

        query = backend.search_traces.call_args.args[0]
        assert query.service_name == "svc-a"
        assert query.gen_ai_request_model == "gpt-4"
        assert query.gen_ai_response_model == "gpt-4-0613"

    async def test_parses_valid_start_and_end_time_into_query(self) -> None:
        backend = _mock_backend()
        backend.search_traces.return_value = []

        await get_slow_traces(
            backend, start_time="2024-01-01T00:00:00Z", end_time="2024-01-02T00:00:00Z"
        )

        query = backend.search_traces.call_args.args[0]
        assert query.start_time == datetime(2024, 1, 1, tzinfo=UTC)
        assert query.end_time == datetime(2024, 1, 2, tzinfo=UTC)


class TestInputValidation:
    """Invalid inputs must raise, so the MCP server reports
    CallToolResult(isError=True) per SEP-2140, and must short-circuit
    before any backend call."""

    async def test_invalid_start_time_raises_without_calling_backend(self) -> None:
        backend = _mock_backend()

        with pytest.raises(ValueError, match="start_time"):
            await get_slow_traces(backend, start_time="not-a-date")
        backend.search_traces.assert_not_called()

    async def test_invalid_end_time_raises_without_calling_backend(self) -> None:
        backend = _mock_backend()

        with pytest.raises(ValueError, match="end_time"):
            await get_slow_traces(backend, end_time="not-a-date")
        backend.search_traces.assert_not_called()

    async def test_zero_limit_raises_validation_error(self) -> None:
        """limit=0 makes the fetch limit `min(0 * 10, 1000) == 0`, which fails
        TraceQuery's `ge=1` constraint - the resulting pydantic
        ValidationError must propagate."""
        backend = _mock_backend()

        with pytest.raises(ValidationError):
            await get_slow_traces(backend, limit=0)
        backend.search_traces.assert_not_called()


class TestBackendExceptionHandling:
    """A backend that raises must let the exception propagate, so the MCP
    server reports CallToolResult(isError=True) per SEP-2140."""

    async def test_search_traces_exception_propagates(self) -> None:
        backend = _mock_backend()
        backend.search_traces.side_effect = RuntimeError("backend down")

        with pytest.raises(RuntimeError, match="backend down"):
            await get_slow_traces(backend)


class TestEdgeCases:
    """Branches specific to this module's own filtering and sorting logic."""

    async def test_empty_result_list_returns_zero_count(self) -> None:
        backend = _mock_backend()
        backend.search_traces.return_value = []

        result = json.loads(await get_slow_traces(backend))

        assert result == {"count": 0, "traces": []}

    async def test_trace_without_llm_spans_is_excluded(self) -> None:
        backend = _mock_backend()
        trace_with_llm = _trace("t1", [_llm_span(trace_id="t1")], duration_ms=5000.0)
        trace_without_llm = _trace("t2", [_non_llm_span(trace_id="t2")], duration_ms=9999.0)
        backend.search_traces.return_value = [trace_with_llm, trace_without_llm]

        result = json.loads(await get_slow_traces(backend))

        assert result["count"] == 1
        assert result["traces"][0]["trace_id"] == "t1"

    async def test_llm_span_with_no_usage_tokens_still_included_with_zero_total(self) -> None:
        """Unlike get_expensive_traces (which drops traces whose total_tokens
        ends up 0), get_slow_traces has no such exclusion - a trace is kept
        as long as it has at least one LLM span, regardless of token usage."""
        backend = _mock_backend()
        span = _llm_span(total_tokens=None)
        trace = _trace("t1", [span])
        backend.search_traces.return_value = [trace]

        result = json.loads(await get_slow_traces(backend))

        assert result["count"] == 1
        assert result["traces"][0]["total_tokens"] == 0

    async def test_limit_truncates_to_requested_count_after_sorting(self) -> None:
        backend = _mock_backend()
        traces = [
            _trace(f"t{i}", [_llm_span(trace_id=f"t{i}")], duration_ms=float(i))
            for i in range(1, 4)
        ]
        backend.search_traces.return_value = traces

        result = json.loads(await get_slow_traces(backend, limit=2))

        assert result["count"] == 2
        # Sorted descending by duration_ms: t3 (3.0), t2 (2.0) beat t1 (1.0).
        assert [t["trace_id"] for t in result["traces"]] == ["t3", "t2"]

    async def test_has_errors_reflects_trace_level_flag(self) -> None:
        backend = _mock_backend()
        error_span = _llm_span(trace_id="t1")
        error_span = error_span.model_copy(update={"status": "ERROR"})
        trace = _trace("t1", [error_span], status="ERROR")
        backend.search_traces.return_value = [trace]

        result = json.loads(await get_slow_traces(backend))

        assert result["traces"][0]["has_errors"] is True
        assert result["traces"][0]["status"] == "ERROR"

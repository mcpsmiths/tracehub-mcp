"""Tests for the get_llm_model_stats tool (model_stats.py).

Backend is mocked with AsyncMock(spec=BaseBackend) - no HTTP involved. Tests
exercise the real transform/aggregation logic in ``get_model_stats`` and the
``calculate_percentiles`` helper.
"""

import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.models import SpanData, TraceData, TraceSummary
from opentelemetry_mcp.tools.model_stats import calculate_percentiles, get_model_stats


def _span(
    trace_id: str,
    span_id: str,
    *,
    model_attrs: dict[str, Any] | None = None,
    duration_ms: float = 100.0,
    status: str = "OK",
) -> SpanData:
    """Build an LLM span with the given gen_ai.* attributes."""
    return SpanData(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=None,
        operation_name="chat",
        service_name="svc",
        start_time=datetime.now(),
        duration_ms=duration_ms,
        status=status,  # type: ignore[arg-type]
        attributes=SpanAttributes.model_validate(model_attrs or {}),
    )


def _trace(trace_id: str, spans: list[SpanData], status: str = "OK") -> TraceData:
    return TraceData(
        trace_id=trace_id,
        spans=spans,
        start_time=spans[0].start_time,
        duration_ms=spans[0].duration_ms,
        service_name=spans[0].service_name,
        root_operation=spans[0].operation_name,
        status=status,  # type: ignore[arg-type]
    )


def _summary_for(trace: TraceData) -> TraceSummary:
    return TraceSummary.from_trace(trace)


class TestCalculatePercentiles:
    """Test the pure percentile-calculation helper."""

    def test_empty_list_returns_zeros(self) -> None:
        result = calculate_percentiles([])
        assert result == {"mean": 0.0, "median": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

    def test_single_value_all_stats_equal_it(self) -> None:
        result = calculate_percentiles([42.0])
        assert result["mean"] == 42.0
        assert result["p50"] == 42.0
        assert result["p95"] == 42.0
        assert result["p99"] == 42.0

    def test_known_distribution_matches_hand_computed_percentiles(self) -> None:
        # sorted: [10, 20, 30, 40, 50] -> length 5, index = (5-1)*p
        values = [50, 10, 40, 20, 30]
        result = calculate_percentiles(values)

        assert result["mean"] == 30.0
        # p50: index = 4*0.5 = 2.0 -> sorted_values[2] = 30
        assert result["median"] == 30.0
        assert result["p50"] == 30.0
        # p95: index = 4*0.95 = 3.8 -> lower=3(40) upper=4(50) weight=0.8
        assert result["p95"] == 40 * 0.2 + 50 * 0.8
        # p99: index = 4*0.99 = 3.96 -> lower=3(40) upper=4(50) weight=0.96
        assert result["p99"] == 40 * 0.04 + 50 * 0.96

    def test_percentile_upper_bound_is_clamped_to_last_index(self) -> None:
        """p99 on a two-element list must not index out of range."""
        result = calculate_percentiles([1.0, 2.0])
        assert result["p99"] <= 2.0


class TestGetModelStatsHappyPath:
    """Test the backend-data -> summary-JSON transform."""

    async def test_returns_expected_shape_and_aggregates_correctly(self) -> None:
        backend = AsyncMock(spec=BaseBackend)

        trace1 = _trace(
            "t1",
            [
                _span(
                    "t1",
                    "s1",
                    model_attrs={
                        "gen_ai.system": "openai",
                        "gen_ai.response.model": "gpt-4",
                        "gen_ai.usage.prompt_tokens": 100,
                        "gen_ai.usage.completion_tokens": 50,
                        "gen_ai.usage.total_tokens": 150,
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    duration_ms=200.0,
                    status="OK",
                )
            ],
        )
        trace2 = _trace(
            "t2",
            [
                _span(
                    "t2",
                    "s2",
                    model_attrs={
                        "gen_ai.system": "openai",
                        "gen_ai.response.model": "gpt-4",
                        "gen_ai.usage.prompt_tokens": 10,
                        "gen_ai.usage.completion_tokens": 5,
                        "gen_ai.usage.total_tokens": 15,
                        "gen_ai.response.finish_reasons": ["length"],
                    },
                    duration_ms=400.0,
                    status="ERROR",
                )
            ],
            status="ERROR",
        )

        backend.search_traces.return_value = [_summary_for(trace1), _summary_for(trace2)]
        backend.get_trace.side_effect = lambda trace_id: {"t1": trace1, "t2": trace2}[trace_id]

        raw = await get_model_stats(backend, model_name="gpt-4")
        result = json.loads(raw)

        assert result["model"] == "gpt-4"
        assert result["request_count"] == 2
        assert result["success_count"] == 1
        assert result["error_count"] == 1
        assert result["success_rate"] == 50.0
        assert result["error_rate"] == 50.0
        assert result["duration_ms"]["mean"] == 300.0
        assert result["tokens"]["prompt"]["mean"] == 55.0
        assert result["tokens"]["completion"]["mean"] == 27.5
        assert result["tokens"]["total"]["mean"] == 82.5
        assert result["finish_reasons"] == {"stop": 1, "length": 1}

    async def test_matches_on_request_model_when_response_model_absent(self) -> None:
        """span_model = response_model or request_model - request_model must
        still match when response_model was never set on the span."""
        backend = AsyncMock(spec=BaseBackend)

        trace = _trace(
            "t1",
            [
                _span(
                    "t1",
                    "s1",
                    model_attrs={
                        "gen_ai.system": "anthropic",
                        "gen_ai.request.model": "claude-3-opus",
                    },
                )
            ],
        )
        backend.search_traces.return_value = [_summary_for(trace)]
        backend.get_trace.return_value = trace

        raw = await get_model_stats(backend, model_name="claude-3-opus")
        result = json.loads(raw)

        assert result["request_count"] == 1
        assert result["model"] == "claude-3-opus"

    async def test_passes_query_params_through_to_backend(self) -> None:
        """start_time/end_time/service_name/limit must reach search_traces
        via the constructed TraceQuery."""
        backend = AsyncMock(spec=BaseBackend)
        backend.search_traces.return_value = []

        await get_model_stats(
            backend,
            model_name="gpt-4",
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-02T00:00:00Z",
            service_name="my-service",
            limit=50,
        )

        query = backend.search_traces.call_args.args[0]
        assert query.service_name == "my-service"
        assert query.limit == 50
        assert query.start_time is not None
        assert query.end_time is not None


class TestGetModelStatsValidation:
    """Test that invalid inputs raise, so the MCP server reports
    CallToolResult(isError=True) per SEP-2140."""

    async def test_invalid_start_time_raises(self) -> None:
        backend = AsyncMock(spec=BaseBackend)

        with pytest.raises(ValueError, match="start_time"):
            await get_model_stats(backend, model_name="gpt-4", start_time="not-a-date")
        backend.search_traces.assert_not_called()

    async def test_invalid_end_time_raises(self) -> None:
        backend = AsyncMock(spec=BaseBackend)

        with pytest.raises(ValueError, match="end_time"):
            await get_model_stats(backend, model_name="gpt-4", end_time="also-not-a-date")
        backend.search_traces.assert_not_called()

    async def test_invalid_start_time_checked_before_end_time(self) -> None:
        """Both are invalid - start_time's error must surface, matching the
        function's sequential validation order."""
        backend = AsyncMock(spec=BaseBackend)

        with pytest.raises(ValueError, match="start_time"):
            await get_model_stats(
                backend, model_name="gpt-4", start_time="bad-start", end_time="bad-end"
            )


class TestGetModelStatsBackendExceptionHandling:
    """Test that backend exceptions propagate, so the MCP server reports
    CallToolResult(isError=True) per SEP-2140, rather than being swallowed
    into a fake-success error JSON payload."""

    async def test_search_traces_exception_propagates(self) -> None:
        backend = AsyncMock(spec=BaseBackend)
        backend.search_traces.side_effect = RuntimeError("backend unreachable")

        with pytest.raises(RuntimeError, match="backend unreachable"):
            await get_model_stats(backend, model_name="gpt-4")

    async def test_get_trace_exception_propagates(self) -> None:
        backend = AsyncMock(spec=BaseBackend)
        trace = _trace("t1", [_span("t1", "s1", model_attrs={"gen_ai.system": "openai"})])
        backend.search_traces.return_value = [_summary_for(trace)]
        backend.get_trace.side_effect = RuntimeError("trace fetch failed")

        with pytest.raises(RuntimeError, match="trace fetch failed"):
            await get_model_stats(backend, model_name="gpt-4")


class TestGetModelStatsEdgeCases:
    """Test branches this module's own logic actually takes."""

    async def test_empty_search_results_returns_no_traces_found_error(self) -> None:
        backend = AsyncMock(spec=BaseBackend)
        backend.search_traces.return_value = []

        raw = await get_model_stats(backend, model_name="gpt-4")
        result = json.loads(raw)

        assert "error" in result
        assert "gpt-4" in result["error"]
        assert "No traces found" in result["error"]
        backend.get_trace.assert_not_called()

    async def test_no_span_matches_requested_model_returns_no_traces_found(self) -> None:
        """Traces exist, but none contain a span for the requested model -
        request_count stays 0."""
        backend = AsyncMock(spec=BaseBackend)
        trace = _trace(
            "t1",
            [
                _span(
                    "t1",
                    "s1",
                    model_attrs={
                        "gen_ai.system": "openai",
                        "gen_ai.response.model": "gpt-3.5-turbo",
                    },
                )
            ],
        )
        backend.search_traces.return_value = [_summary_for(trace)]
        backend.get_trace.return_value = trace

        raw = await get_model_stats(backend, model_name="gpt-4")
        result = json.loads(raw)

        assert "error" in result
        assert "No traces found" in result["error"]

    async def test_non_llm_span_is_skipped_without_error(self) -> None:
        """A span with no gen_ai.* attributes at all (is_llm_span False) must
        be skipped, not crash the aggregation loop."""
        backend = AsyncMock(spec=BaseBackend)
        plain_span = _span("t1", "s1", model_attrs={})
        llm_span = _span(
            "t1",
            "s2",
            model_attrs={"gen_ai.system": "openai", "gen_ai.response.model": "gpt-4"},
        )
        trace = _trace("t1", [plain_span, llm_span])
        backend.search_traces.return_value = [_summary_for(trace)]
        backend.get_trace.return_value = trace

        raw = await get_model_stats(backend, model_name="gpt-4")
        result = json.loads(raw)

        assert result["request_count"] == 1

    async def test_missing_optional_token_fields_yield_empty_percentiles(self) -> None:
        """A matching span with no usage.* attributes must not populate the
        token lists - calculate_percentiles falls back to its zeroed shape."""
        backend = AsyncMock(spec=BaseBackend)
        trace = _trace(
            "t1",
            [
                _span(
                    "t1",
                    "s1",
                    model_attrs={"gen_ai.system": "openai", "gen_ai.response.model": "gpt-4"},
                )
            ],
        )
        backend.search_traces.return_value = [_summary_for(trace)]
        backend.get_trace.return_value = trace

        raw = await get_model_stats(backend, model_name="gpt-4")
        result = json.loads(raw)

        assert result["request_count"] == 1
        assert result["tokens"]["prompt"] == {
            "mean": 0.0,
            "median": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    async def test_no_finish_reasons_collected_yields_null(self) -> None:
        """finish_reasons_count starts empty; when nothing was collected the
        result must be JSON null, not an empty object."""
        backend = AsyncMock(spec=BaseBackend)
        trace = _trace(
            "t1",
            [
                _span(
                    "t1",
                    "s1",
                    model_attrs={"gen_ai.system": "openai", "gen_ai.response.model": "gpt-4"},
                )
            ],
        )
        backend.search_traces.return_value = [_summary_for(trace)]
        backend.get_trace.return_value = trace

        raw = await get_model_stats(backend, model_name="gpt-4")
        result = json.loads(raw)

        assert result["finish_reasons"] is None

    async def test_multiple_finish_reasons_on_one_span_are_all_counted(self) -> None:
        backend = AsyncMock(spec=BaseBackend)
        trace = _trace(
            "t1",
            [
                _span(
                    "t1",
                    "s1",
                    model_attrs={
                        "gen_ai.system": "openai",
                        "gen_ai.response.model": "gpt-4",
                        "gen_ai.response.finish_reasons": ["stop", "length"],
                    },
                )
            ],
        )
        backend.search_traces.return_value = [_summary_for(trace)]
        backend.get_trace.return_value = trace

        raw = await get_model_stats(backend, model_name="gpt-4")
        result = json.loads(raw)

        assert result["finish_reasons"] == {"stop": 1, "length": 1}

    async def test_traces_with_multiple_llm_spans_only_counts_matching_ones(self) -> None:
        """A single trace can contain spans for several different models -
        only the requested model's spans should feed the aggregates."""
        backend = AsyncMock(spec=BaseBackend)
        gpt4_span = _span(
            "t1",
            "s1",
            model_attrs={"gen_ai.system": "openai", "gen_ai.response.model": "gpt-4"},
            duration_ms=100.0,
        )
        other_span = _span(
            "t1",
            "s2",
            model_attrs={"gen_ai.system": "openai", "gen_ai.response.model": "gpt-3.5-turbo"},
            duration_ms=999.0,
        )
        trace = _trace("t1", [gpt4_span, other_span])
        backend.search_traces.return_value = [_summary_for(trace)]
        backend.get_trace.return_value = trace

        raw = await get_model_stats(backend, model_name="gpt-4")
        result = json.loads(raw)

        assert result["request_count"] == 1
        assert result["duration_ms"]["mean"] == 100.0

    async def test_default_limit_is_used_when_not_specified(self) -> None:
        backend = AsyncMock(spec=BaseBackend)
        backend.search_traces.return_value = []

        await get_model_stats(backend, model_name="gpt-4")

        query = backend.search_traces.call_args.args[0]
        assert query.limit == 1000

"""Tests for the get_expensive_traces tool."""

import json
from datetime import UTC, datetime
from typing import Any, Literal
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.models import SpanData, TraceData
from opentelemetry_mcp.tools.expensive_traces import get_expensive_traces


def _llm_span(
    *,
    trace_id: str,
    span_id: str,
    service_name: str = "svc",
    status: Literal["OK", "ERROR", "UNSET"] = "OK",
    gen_ai_system: str | None = "openai",
    request_model: str | None = "gpt-4",
    response_model: str | None = None,
    prompt_tokens: int | None = 100,
    completion_tokens: int | None = 50,
    total_tokens: int | None = 150,
) -> SpanData:
    """Build an LLM span with configurable gen_ai attributes.

    Passing ``gen_ai_system=None`` produces a non-LLM span (no gen_ai.*
    attributes at all). Passing a token field as ``None`` omits that
    attribute entirely rather than setting it to zero.
    """
    attrs: dict[str, Any] = {}
    if gen_ai_system is not None:
        attrs["gen_ai.system"] = gen_ai_system
    if request_model is not None:
        attrs["gen_ai.request.model"] = request_model
    if response_model is not None:
        attrs["gen_ai.response.model"] = response_model
    if prompt_tokens is not None:
        attrs["gen_ai.usage.prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        attrs["gen_ai.usage.completion_tokens"] = completion_tokens
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
        status=status,
        attributes=SpanAttributes.model_validate(attrs),
    )


def _trace(
    trace_id: str,
    spans: list[SpanData],
    *,
    service_name: str = "svc",
    operation_name: str = "chat",
    duration_ms: float = 100.0,
    status: Literal["OK", "ERROR", "UNSET"] = "OK",
) -> TraceData:
    """Build a TraceData instance from a list of spans."""
    return TraceData(
        trace_id=trace_id,
        spans=spans,
        start_time=spans[0].start_time,
        duration_ms=duration_ms,
        service_name=service_name,
        root_operation=operation_name,
        status=status,
    )


def _mock_backend() -> AsyncMock:
    """A backend double scoped to BaseBackend's actual async methods."""
    return AsyncMock(spec=BaseBackend)


def _wire_backend(backend: AsyncMock, traces: list[TraceData]) -> None:
    """Wire search_traces to return the given full traces - its documented
    contract (see BaseBackend.search_traces) is traces with all spans
    already attached, so no separate get_trace mocking is needed."""
    backend.search_traces.return_value = traces


class TestHappyPath:
    """A realistic backend response is transformed into the correct
    top-N expensive-trace JSON shape."""

    async def test_returns_trace_with_token_breakdown_and_metadata(self) -> None:
        backend = _mock_backend()
        span = _llm_span(
            trace_id="t1",
            span_id="s1",
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
        )
        trace = _trace("t1", [span], service_name="svc-a", operation_name="chat-op")
        _wire_backend(backend, [trace])

        result = json.loads(await get_expensive_traces(backend))

        assert result["count"] == 1
        info = result["traces"][0]
        assert info["trace_id"] == "t1"
        assert info["service_name"] == "svc-a"
        assert info["operation_name"] == "chat-op"
        assert info["models"] == ["gpt-4"]
        assert info["tokens"] == {"prompt": 100, "completion": 200, "total": 300}
        assert info["status"] == "OK"
        assert info["has_errors"] is False

    async def test_multiple_llm_spans_in_one_trace_are_summed(self) -> None:
        backend = _mock_backend()
        span1 = _llm_span(
            trace_id="t1",
            span_id="s1",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        span2 = _llm_span(
            trace_id="t1",
            span_id="s2",
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
        )
        trace = _trace("t1", [span1, span2])
        _wire_backend(backend, [trace])

        result = json.loads(await get_expensive_traces(backend))

        tokens = result["traces"][0]["tokens"]
        assert tokens == {"prompt": 300, "completion": 150, "total": 450}

    async def test_models_used_are_deduplicated_and_sorted(self) -> None:
        backend = _mock_backend()
        span1 = _llm_span(
            trace_id="t1", span_id="s1", request_model="gpt-4", response_model="gpt-4-0613"
        )
        span2 = _llm_span(
            trace_id="t1", span_id="s2", request_model="gpt-4", response_model="claude-3"
        )
        trace = _trace("t1", [span1, span2])
        _wire_backend(backend, [trace])

        result = json.loads(await get_expensive_traces(backend))

        assert result["traces"][0]["models"] == ["claude-3", "gpt-4-0613"]

    async def test_response_model_takes_precedence_over_request_model(self) -> None:
        backend = _mock_backend()
        span = _llm_span(
            trace_id="t1", span_id="s1", request_model="gpt-4", response_model="gpt-4-0613"
        )
        trace = _trace("t1", [span])
        _wire_backend(backend, [trace])

        result = json.loads(await get_expensive_traces(backend))

        assert result["traces"][0]["models"] == ["gpt-4-0613"]

    async def test_results_sorted_descending_by_total_tokens_and_limited(self) -> None:
        backend = _mock_backend()
        traces = [
            _trace("small", [_llm_span(trace_id="small", span_id="s1", total_tokens=100)]),
            _trace("big", [_llm_span(trace_id="big", span_id="s1", total_tokens=900)]),
            _trace("medium", [_llm_span(trace_id="medium", span_id="s1", total_tokens=500)]),
        ]
        _wire_backend(backend, traces)

        result = json.loads(await get_expensive_traces(backend, limit=2))

        assert result["count"] == 2
        assert [t["trace_id"] for t in result["traces"]] == ["big", "medium"]

    async def test_duration_ms_is_rounded_to_two_decimals(self) -> None:
        backend = _mock_backend()
        span = _llm_span(trace_id="t1", span_id="s1")
        trace = _trace("t1", [span], duration_ms=123.456789)
        _wire_backend(backend, [trace])

        result = json.loads(await get_expensive_traces(backend))

        assert result["traces"][0]["duration_ms"] == 123.46

    async def test_has_errors_true_when_a_span_has_error_status(self) -> None:
        backend = _mock_backend()
        span = _llm_span(trace_id="t1", span_id="s1", status="ERROR")
        trace = _trace("t1", [span], status="ERROR")
        _wire_backend(backend, [trace])

        result = json.loads(await get_expensive_traces(backend))

        assert result["traces"][0]["has_errors"] is True
        assert result["traces"][0]["status"] == "ERROR"


class TestFilteringLogic:
    """Branches specific to this module's own token-threshold and
    LLM-usage filtering."""

    async def test_min_tokens_excludes_trace_below_threshold(self) -> None:
        backend = _mock_backend()
        trace = _trace("t1", [_llm_span(trace_id="t1", span_id="s1", total_tokens=50)])
        _wire_backend(backend, [trace])

        result = json.loads(await get_expensive_traces(backend, min_tokens=100))

        assert result == {"count": 0, "traces": []}

    async def test_min_tokens_boundary_is_inclusive(self) -> None:
        backend = _mock_backend()
        trace = _trace("t1", [_llm_span(trace_id="t1", span_id="s1", total_tokens=100)])
        _wire_backend(backend, [trace])

        result = json.loads(await get_expensive_traces(backend, min_tokens=100))

        assert result["count"] == 1

    async def test_trace_with_no_token_usage_is_skipped_even_if_llm_span_present(self) -> None:
        """A span can be an LLM span (gen_ai.system present) yet carry no
        usage attributes at all - total_tokens stays 0, and the trace must
        be dropped as having "no LLM usage", not surfaced with zero tokens."""
        backend = _mock_backend()
        span = _llm_span(
            trace_id="t1",
            span_id="s1",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        )
        trace = _trace("t1", [span])
        _wire_backend(backend, [trace])

        result = json.loads(await get_expensive_traces(backend))

        assert result == {"count": 0, "traces": []}

    async def test_trace_with_only_non_llm_spans_is_skipped(self) -> None:
        backend = _mock_backend()
        span = _llm_span(trace_id="t1", span_id="s1", gen_ai_system=None)
        trace = _trace("t1", [span])
        _wire_backend(backend, [trace])

        result = json.loads(await get_expensive_traces(backend))

        assert result == {"count": 0, "traces": []}


class TestQueryConstruction:
    """The tool must build a TraceQuery that oversamples (limit * 10, capped
    at 1000) and forwards the caller's filter parameters."""

    async def test_query_limit_is_limit_times_ten_under_cap(self) -> None:
        backend = _mock_backend()
        backend.search_traces.return_value = []

        await get_expensive_traces(backend, limit=5)

        query = backend.search_traces.call_args.args[0]
        assert query.limit == 50

    async def test_query_limit_is_capped_at_one_thousand(self) -> None:
        backend = _mock_backend()
        backend.search_traces.return_value = []

        await get_expensive_traces(backend, limit=200)

        query = backend.search_traces.call_args.args[0]
        assert query.limit == 1000

    async def test_query_forwards_filter_parameters_and_parsed_datetimes(self) -> None:
        backend = _mock_backend()
        backend.search_traces.return_value = []

        await get_expensive_traces(
            backend,
            service_name="svc-a",
            gen_ai_request_model="gpt-4",
            gen_ai_response_model="gpt-4-0613",
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-02T00:00:00Z",
        )

        query = backend.search_traces.call_args.args[0]
        assert query.service_name == "svc-a"
        assert query.gen_ai_request_model == "gpt-4"
        assert query.gen_ai_response_model == "gpt-4-0613"
        assert query.start_time == datetime(2024, 1, 1, tzinfo=UTC)
        assert query.end_time == datetime(2024, 1, 2, tzinfo=UTC)


class TestInputValidation:
    """Invalid inputs must raise, so the MCP server reports
    CallToolResult(isError=True) per SEP-2140."""

    async def test_invalid_start_time_raises_without_calling_backend(self) -> None:
        backend = _mock_backend()

        with pytest.raises(ValueError, match="start_time"):
            await get_expensive_traces(backend, start_time="not-a-date")
        backend.search_traces.assert_not_called()

    async def test_invalid_end_time_raises_without_calling_backend(self) -> None:
        backend = _mock_backend()

        with pytest.raises(ValueError, match="end_time"):
            await get_expensive_traces(backend, end_time="not-a-date")
        backend.search_traces.assert_not_called()

    async def test_limit_that_produces_an_invalid_query_raises_validation_error(self) -> None:
        """limit=0 makes the internal TraceQuery(limit=min(0*10, 1000)=0),
        which fails TraceQuery's `ge=1` constraint - the resulting pydantic
        ValidationError must propagate, confirmed here rather than assumed."""
        backend = _mock_backend()

        with pytest.raises(ValidationError):
            await get_expensive_traces(backend, limit=0)
        backend.search_traces.assert_not_called()


class TestBackendExceptionHandling:
    """A backend that raises - at either await point - must let the
    exception propagate, so the MCP server reports
    CallToolResult(isError=True) per SEP-2140."""

    async def test_search_traces_exception_propagates(self) -> None:
        backend = _mock_backend()
        backend.search_traces.side_effect = RuntimeError("backend down")

        with pytest.raises(RuntimeError, match="backend down"):
            await get_expensive_traces(backend)


class TestEdgeCases:
    """Empty results and the top-N limit boundary."""

    async def test_empty_search_result_returns_zero_count(self) -> None:
        backend = _mock_backend()
        backend.search_traces.return_value = []

        result = json.loads(await get_expensive_traces(backend))

        assert result == {"count": 0, "traces": []}

    async def test_limit_truncates_to_top_n_even_with_more_qualifying_traces(self) -> None:
        backend = _mock_backend()
        traces = [
            _trace("t1", [_llm_span(trace_id="t1", span_id="s1", total_tokens=100)]),
            _trace("t2", [_llm_span(trace_id="t2", span_id="s1", total_tokens=200)]),
            _trace("t3", [_llm_span(trace_id="t3", span_id="s1", total_tokens=300)]),
        ]
        _wire_backend(backend, traces)

        result = json.loads(await get_expensive_traces(backend, limit=1))

        assert result["count"] == 1
        assert result["traces"][0]["trace_id"] == "t3"

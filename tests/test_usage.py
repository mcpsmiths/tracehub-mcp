"""Tests for the get_llm_usage tool."""

import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.models import SpanData, TraceData, TraceQuery
from opentelemetry_mcp.tools.usage import get_llm_usage


def _llm_span(
    *,
    trace_id: str,
    span_id: str,
    service_name: str,
    gen_ai_system: str = "openai",
    request_model: str | None = "gpt-4",
    response_model: str | None = None,
    prompt_tokens: int | None = 100,
    completion_tokens: int | None = 50,
    total_tokens: int | None = 150,
) -> SpanData:
    """Build an LLM span with configurable gen_ai attributes."""
    attrs: dict[str, Any] = {"gen_ai.system": gen_ai_system}
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
        start_time=datetime(2024, 1, 1, 12, 0, 0),
        duration_ms=100.0,
        status="OK",
        attributes=SpanAttributes.model_validate(attrs),
    )


def _non_llm_span(*, trace_id: str, span_id: str, service_name: str) -> SpanData:
    """Build a span with no gen_ai.* attributes at all."""
    return SpanData(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=None,
        operation_name="db_query",
        service_name=service_name,
        start_time=datetime(2024, 1, 1, 12, 0, 0),
        duration_ms=10.0,
        status="OK",
        attributes=SpanAttributes.model_validate({}),
    )


def _trace(trace_id: str, service_name: str, spans: list[SpanData]) -> TraceData:
    return TraceData(
        trace_id=trace_id,
        spans=spans,
        start_time=spans[0].start_time,
        duration_ms=100.0,
        service_name=service_name,
        root_operation=spans[0].operation_name,
        status="OK",
    )


def _mock_backend() -> AsyncMock:
    return AsyncMock(spec=BaseBackend)


class TestGetLlmUsageValidation:
    """Invalid time parameters must produce an error JSON, not an exception,
    and must short-circuit before ever calling the backend."""

    async def test_invalid_start_time_returns_error_without_calling_backend(self) -> None:
        backend = _mock_backend()

        result = await get_llm_usage(backend, start_time="not-a-timestamp")
        parsed = json.loads(result)

        assert "error" in parsed
        assert "start_time" in parsed["error"]
        backend.search_traces.assert_not_called()

    async def test_invalid_end_time_returns_error_without_calling_backend(self) -> None:
        backend = _mock_backend()

        result = await get_llm_usage(
            backend, start_time="2024-01-01T00:00:00Z", end_time="also-not-a-timestamp"
        )
        parsed = json.loads(result)

        assert "error" in parsed
        assert "end_time" in parsed["error"]
        backend.search_traces.assert_not_called()


class TestGetLlmUsageHappyPath:
    """A realistic backend response must be transformed into the correct
    aggregated JSON shape."""

    async def test_aggregates_totals_across_traces(self) -> None:
        backend = _mock_backend()
        trace1 = _trace(
            "t1",
            "svc-a",
            [
                _llm_span(
                    trace_id="t1",
                    span_id="s1",
                    service_name="svc-a",
                    gen_ai_system="openai",
                    request_model="gpt-4",
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                )
            ],
        )
        trace2 = _trace(
            "t2",
            "svc-b",
            [
                _llm_span(
                    trace_id="t2",
                    span_id="s2",
                    service_name="svc-b",
                    gen_ai_system="anthropic",
                    request_model="claude-3",
                    prompt_tokens=200,
                    completion_tokens=100,
                    total_tokens=300,
                )
            ],
        )
        backend.search_traces.return_value = [trace1, trace2]

        result = await get_llm_usage(backend)
        parsed = json.loads(result)

        assert parsed["summary"] == {
            "total_requests": 2,
            "total_prompt_tokens": 300,
            "total_completion_tokens": 150,
            "total_tokens": 450,
        }
        assert parsed["by_model"]["gpt-4"] == {
            "requests": 1,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        assert parsed["by_model"]["claude-3"] == {
            "requests": 1,
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
        }
        assert parsed["by_service"]["svc-a"]["requests"] == 1
        assert parsed["by_service"]["svc-b"]["requests"] == 1

    async def test_response_model_takes_precedence_over_request_model_in_breakdown(self) -> None:
        backend = _mock_backend()
        span = _llm_span(
            trace_id="t1",
            span_id="s1",
            service_name="svc-a",
            request_model="gpt-4",
            response_model="gpt-4-0613",
        )
        backend.search_traces.return_value = [_trace("t1", "svc-a", [span])]

        result = await get_llm_usage(backend)
        parsed = json.loads(result)

        assert "gpt-4-0613" in parsed["by_model"]
        assert "gpt-4" not in parsed["by_model"]

    async def test_empty_trace_list_produces_zeroed_summary_and_empty_breakdowns(self) -> None:
        backend = _mock_backend()
        backend.search_traces.return_value = []

        result = await get_llm_usage(backend)
        parsed = json.loads(result)

        assert parsed["summary"] == {
            "total_requests": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
        }
        assert parsed["by_model"] == {}
        assert parsed["by_service"] == {}

    async def test_period_and_filters_reflect_provided_parameters(self) -> None:
        backend = _mock_backend()
        backend.search_traces.return_value = []

        result = await get_llm_usage(
            backend,
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-02T00:00:00Z",
            service_name="svc-a",
            gen_ai_system="openai",
            gen_ai_request_model="gpt-4",
            gen_ai_response_model="gpt-4-0613",
        )
        parsed = json.loads(result)

        assert parsed["period"] == {
            "start_time": "2024-01-01T00:00:00+00:00",
            "end_time": "2024-01-02T00:00:00+00:00",
        }
        assert parsed["filters"] == {
            "service_name": "svc-a",
            "gen_ai_system": "openai",
            "gen_ai_request_model": "gpt-4",
            "gen_ai_response_model": "gpt-4-0613",
        }

    async def test_period_is_null_when_no_times_provided(self) -> None:
        backend = _mock_backend()
        backend.search_traces.return_value = []

        result = await get_llm_usage(backend)
        parsed = json.loads(result)

        assert parsed["period"] == {"start_time": None, "end_time": None}

    async def test_builds_trace_query_with_parsed_datetimes_and_filters(self) -> None:
        """The tool must forward every filter parameter and the parsed
        datetimes (not the raw strings) into the TraceQuery it hands the
        backend."""
        backend = _mock_backend()
        backend.search_traces.return_value = []

        await get_llm_usage(
            backend,
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-02T00:00:00Z",
            service_name="svc-a",
            gen_ai_system="openai",
            gen_ai_request_model="gpt-4",
            gen_ai_response_model="gpt-4-0613",
            limit=42,
        )

        backend.search_traces.assert_awaited_once()
        query = backend.search_traces.await_args.args[0]
        assert isinstance(query, TraceQuery)
        assert query.service_name == "svc-a"
        assert query.gen_ai_system == "openai"
        assert query.gen_ai_request_model == "gpt-4"
        assert query.gen_ai_response_model == "gpt-4-0613"
        assert query.limit == 42
        assert query.start_time == datetime.fromisoformat("2024-01-01T00:00:00+00:00")
        assert query.end_time == datetime.fromisoformat("2024-01-02T00:00:00+00:00")


class TestGetLlmUsageBackendError:
    """A backend exception must be caught and turned into an error JSON,
    never left to propagate."""

    async def test_backend_exception_is_caught_and_formatted(self) -> None:
        backend = _mock_backend()
        backend.search_traces.side_effect = ConnectionError("backend unreachable")

        result = await get_llm_usage(backend)
        parsed = json.loads(result)

        assert parsed == {"error": "Failed to get usage metrics: backend unreachable"}


class TestGetLlmUsageEdgeCases:
    """Branches specific to this module's own aggregation logic."""

    async def test_non_llm_spans_are_excluded_from_aggregation(self) -> None:
        backend = _mock_backend()
        llm_span = _llm_span(trace_id="t1", span_id="s1", service_name="svc-a")
        other_span = _non_llm_span(trace_id="t1", span_id="s2", service_name="svc-a")
        backend.search_traces.return_value = [_trace("t1", "svc-a", [llm_span, other_span])]

        result = await get_llm_usage(backend)
        parsed = json.loads(result)

        assert parsed["summary"]["total_requests"] == 1

    async def test_missing_optional_token_fields_contribute_zero_not_error(self) -> None:
        backend = _mock_backend()
        span = _llm_span(
            trace_id="t1",
            span_id="s1",
            service_name="svc-a",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        )
        backend.search_traces.return_value = [_trace("t1", "svc-a", [span])]

        result = await get_llm_usage(backend)
        parsed = json.loads(result)

        assert parsed["summary"] == {
            "total_requests": 1,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
        }

    async def test_span_with_no_model_name_falls_back_to_unknown_bucket(self) -> None:
        backend = _mock_backend()
        span = _llm_span(
            trace_id="t1",
            span_id="s1",
            service_name="svc-a",
            request_model=None,
            response_model=None,
        )
        backend.search_traces.return_value = [_trace("t1", "svc-a", [span])]

        result = await get_llm_usage(backend)
        parsed = json.loads(result)

        assert parsed["by_model"]["unknown"]["requests"] == 1

    async def test_multiple_models_on_same_service_are_kept_separate(self) -> None:
        backend = _mock_backend()
        span_a = _llm_span(trace_id="t1", span_id="s1", service_name="svc-a", request_model="gpt-4")
        span_b = _llm_span(
            trace_id="t1", span_id="s2", service_name="svc-a", request_model="gpt-3.5"
        )
        backend.search_traces.return_value = [_trace("t1", "svc-a", [span_a, span_b])]

        result = await get_llm_usage(backend)
        parsed = json.loads(result)

        assert set(parsed["by_model"].keys()) == {"gpt-4", "gpt-3.5"}
        assert parsed["by_service"]["svc-a"]["requests"] == 2


class TestGetLlmUsageLimitBoundary:
    """TraceQuery constrains 'limit' to [1, 1000]; construction happens
    inside its own try/except in get_llm_usage so an out-of-range value
    is converted to the documented error-JSON contract."""

    async def test_limit_below_minimum_returns_error_json(self) -> None:
        """An out-of-range 'limit' (a normal, externally controlled MCP
        tool argument) must be converted to `{"error": ...}` JSON, not
        raise a raw pydantic.ValidationError, matching the error-JSON
        contract every other invalid-input path in this tool honors."""
        backend = _mock_backend()

        result = await get_llm_usage(backend, limit=0)
        parsed = json.loads(result)

        assert "error" in parsed
        assert parsed["error"].startswith("Invalid query parameters:")
        backend.search_traces.assert_not_called()

    async def test_limit_at_maximum_boundary_is_accepted(self) -> None:
        backend = _mock_backend()
        backend.search_traces.return_value = []

        result = await get_llm_usage(backend, limit=1000)
        parsed = json.loads(result)

        assert "error" not in parsed
        query = backend.search_traces.await_args.args[0]
        assert query.limit == 1000

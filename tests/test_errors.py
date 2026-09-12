"""Tests for the find_errors tool."""

import json
from datetime import UTC, datetime
from typing import Any, Literal
from unittest.mock import AsyncMock

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.models import SpanData, TraceData
from opentelemetry_mcp.tools.errors import find_errors


def _span(
    *,
    span_id: str = "span1",
    trace_id: str = "trace1",
    service_name: str = "svc",
    operation_name: str = "op",
    status: Literal["OK", "ERROR", "UNSET"] = "ERROR",
    attributes: dict[str, Any] | None = None,
) -> SpanData:
    """Build a SpanData instance with sensible defaults for error-path tests."""
    return SpanData(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=None,
        operation_name=operation_name,
        service_name=service_name,
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        duration_ms=100,
        status=status,
        attributes=SpanAttributes.model_validate(attributes or {}),
    )


def _trace(
    spans: list[SpanData],
    *,
    trace_id: str = "trace1",
    status: Literal["OK", "ERROR", "UNSET"] = "ERROR",
) -> TraceData:
    """Build a TraceData instance from a list of spans."""
    return TraceData(
        trace_id=trace_id,
        spans=spans,
        start_time=spans[0].start_time,
        duration_ms=100,
        service_name=spans[0].service_name,
        root_operation=spans[0].operation_name,
        status=status,
    )


def _backend() -> AsyncMock:
    """A backend double scoped to BaseBackend's actual async methods."""
    return AsyncMock(spec=BaseBackend)


class TestHappyPath:
    """Backend returns real error data -> correct JSON shape and transforms."""

    async def test_returns_error_traces_with_extracted_details(self) -> None:
        backend = _backend()
        span = _span(attributes={"error.message": "boom", "error.type": "ValueError"})
        trace = _trace([span])
        backend.search_traces.return_value = [trace]

        result = json.loads(await find_errors(backend, service_name="svc"))

        assert result["count"] == 1
        trace_info = result["error_traces"][0]
        assert trace_info["trace_id"] == "trace1"
        assert trace_info["service_name"] == "svc"
        assert len(trace_info["error_spans"]) == 1
        err = trace_info["error_spans"][0]
        assert err["span_id"] == "span1"
        assert err["operation_name"] == "op"
        assert err["service_name"] == "svc"
        assert err["status"] == "ERROR"
        assert err["error_message"] == "boom"
        assert err["error_type"] == "ValueError"
        assert "is_llm_error" not in err

    async def test_error_message_falls_back_to_exception_message(self) -> None:
        backend = _backend()
        span = _span(attributes={"exception.message": "kaboom"})
        trace = _trace([span])
        backend.search_traces.return_value = [trace]

        result = json.loads(await find_errors(backend))

        err = result["error_traces"][0]["error_spans"][0]
        assert err["error_message"] == "kaboom"

    async def test_error_message_defaults_to_unknown_when_absent(self) -> None:
        backend = _backend()
        span = _span(attributes={})
        trace = _trace([span])
        backend.search_traces.return_value = [trace]

        result = json.loads(await find_errors(backend))

        err = result["error_traces"][0]["error_spans"][0]
        assert err["error_message"] == "Unknown error"
        assert "error_type" not in err

    async def test_stack_trace_included_verbatim_when_short(self) -> None:
        backend = _backend()
        span = _span(attributes={"exception.stacktrace": "short trace"})
        trace = _trace([span])
        backend.search_traces.return_value = [trace]

        result = json.loads(await find_errors(backend))

        err = result["error_traces"][0]["error_spans"][0]
        assert err["stack_trace"] == "short trace"

    async def test_stack_trace_truncated_at_500_chars_when_long(self) -> None:
        backend = _backend()
        long_trace = "x" * 600
        span = _span(attributes={"exception.stacktrace": long_trace})
        trace = _trace([span])
        backend.search_traces.return_value = [trace]

        result = json.loads(await find_errors(backend))

        err = result["error_traces"][0]["error_spans"][0]
        assert err["stack_trace"] == "x" * 500 + "..."

    async def test_stack_trace_key_absent_when_not_present(self) -> None:
        backend = _backend()
        span = _span(attributes={"error.message": "boom"})
        trace = _trace([span])
        backend.search_traces.return_value = [trace]

        result = json.loads(await find_errors(backend))

        err = result["error_traces"][0]["error_spans"][0]
        assert "stack_trace" not in err

    async def test_llm_error_span_includes_provider_and_request_model(self) -> None:
        backend = _backend()
        span = _span(
            attributes={
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4",
                "error.message": "rate limited",
            }
        )
        trace = _trace([span])
        backend.search_traces.return_value = [trace]

        result = json.loads(await find_errors(backend))

        err = result["error_traces"][0]["error_spans"][0]
        assert err["is_llm_error"] is True
        assert err["llm_provider"] == "openai"
        assert err["llm_model"] == "gpt-4"

    async def test_llm_error_span_falls_back_to_response_model(self) -> None:
        backend = _backend()
        span = _span(
            attributes={"gen_ai.system": "anthropic", "gen_ai.response.model": "claude-3-opus"}
        )
        trace = _trace([span])
        backend.search_traces.return_value = [trace]

        result = json.loads(await find_errors(backend))

        err = result["error_traces"][0]["error_spans"][0]
        assert err["llm_model"] == "claude-3-opus"

    async def test_llm_error_span_with_no_model_info_has_llm_model_none(self) -> None:
        backend = _backend()
        span = _span(attributes={"gen_ai.system": "openai"})
        trace = _trace([span])
        backend.search_traces.return_value = [trace]

        result = json.loads(await find_errors(backend))

        err = result["error_traces"][0]["error_spans"][0]
        assert err["is_llm_error"] is True
        assert err["llm_model"] is None

    async def test_non_llm_span_has_no_llm_fields(self) -> None:
        backend = _backend()
        span = _span(attributes={"error.message": "boom"})
        trace = _trace([span])
        backend.search_traces.return_value = [trace]

        result = json.loads(await find_errors(backend))

        err = result["error_traces"][0]["error_spans"][0]
        assert "is_llm_error" not in err
        assert "llm_provider" not in err
        assert "llm_model" not in err

    async def test_empty_gen_ai_system_span_has_no_llm_fields(self) -> None:
        """gen_ai.system="" is treated as absent (matching SpanData.is_llm_span
        and LLMSpanAttributes.from_span), so a request model set alongside an
        empty system must not surface as an LLM error - this locks in that
        consistency rather than reporting a half-populated LLM error."""
        backend = _backend()
        span = _span(attributes={"gen_ai.system": "", "gen_ai.request.model": "gpt-4"})
        trace = _trace([span])
        backend.search_traces.return_value = [trace]

        result = json.loads(await find_errors(backend))

        err = result["error_traces"][0]["error_spans"][0]
        assert "is_llm_error" not in err
        assert "llm_provider" not in err
        assert "llm_model" not in err

    async def test_multiple_error_spans_in_one_trace_are_all_reported(self) -> None:
        backend = _backend()
        span1 = _span(span_id="s1", attributes={"error.message": "first"})
        span2 = _span(span_id="s2", attributes={"error.message": "second"})
        trace = _trace([span1, span2])
        backend.search_traces.return_value = [trace]

        result = json.loads(await find_errors(backend))

        error_spans = result["error_traces"][0]["error_spans"]
        assert {s["span_id"] for s in error_spans} == {"s1", "s2"}
        assert {s["error_message"] for s in error_spans} == {"first", "second"}


class TestEdgeCases:
    """Empty results and traces without error spans."""

    async def test_empty_result_list_returns_zero_count(self) -> None:
        backend = _backend()
        backend.search_traces.return_value = []

        result = json.loads(await find_errors(backend))

        assert result == {"count": 0, "error_traces": []}

    async def test_trace_with_no_error_spans_has_empty_error_spans_list(self) -> None:
        """A trace can come back from search_traces (e.g. matched has_error at
        the trace level) without any individual span whose own status is ERROR."""
        backend = _backend()
        ok_span = _span(status="OK", attributes={})
        trace = _trace([ok_span], status="OK")
        backend.search_traces.return_value = [trace]

        result = json.loads(await find_errors(backend))

        assert result["count"] == 1
        assert result["error_traces"][0]["error_spans"] == []


class TestQueryConstruction:
    """The tool must build a TraceQuery with has_error forced True and the
    caller's parameters passed through."""

    async def test_builds_query_with_has_error_true_and_params(self) -> None:
        backend = _backend()
        backend.search_traces.return_value = []

        await find_errors(backend, service_name="svc", limit=50)

        query = backend.search_traces.call_args.args[0]
        assert query.has_error is True
        assert query.service_name == "svc"
        assert query.limit == 50

    async def test_parses_valid_start_and_end_time_into_query(self) -> None:
        backend = _backend()
        backend.search_traces.return_value = []

        await find_errors(
            backend, start_time="2024-01-01T00:00:00Z", end_time="2024-01-02T00:00:00Z"
        )

        query = backend.search_traces.call_args.args[0]
        assert query.start_time == datetime(2024, 1, 1, tzinfo=UTC)
        assert query.end_time == datetime(2024, 1, 2, tzinfo=UTC)

    async def test_default_limit_is_100(self) -> None:
        backend = _backend()
        backend.search_traces.return_value = []

        await find_errors(backend)

        query = backend.search_traces.call_args.args[0]
        assert query.limit == 100


class TestInputValidation:
    """Invalid inputs must produce {"error": ...} JSON, never raise, and must
    short-circuit before any backend call."""

    async def test_invalid_start_time_returns_error_without_calling_backend(self) -> None:
        backend = _backend()

        result = json.loads(await find_errors(backend, start_time="not-a-date"))

        assert "error" in result
        assert "start_time" in result["error"]
        backend.search_traces.assert_not_called()

    async def test_invalid_end_time_returns_error_without_calling_backend(self) -> None:
        backend = _backend()

        result = json.loads(await find_errors(backend, end_time="not-a-date"))

        assert "error" in result
        assert "end_time" in result["error"]
        backend.search_traces.assert_not_called()

    async def test_limit_below_minimum_returns_validation_error(self) -> None:
        backend = _backend()

        result = json.loads(await find_errors(backend, limit=0))

        assert "Invalid query parameters" in result["error"]
        backend.search_traces.assert_not_called()

    async def test_limit_above_maximum_returns_validation_error(self) -> None:
        backend = _backend()

        result = json.loads(await find_errors(backend, limit=1001))

        assert "Invalid query parameters" in result["error"]
        backend.search_traces.assert_not_called()

    async def test_limit_at_maximum_boundary_is_accepted(self) -> None:
        backend = _backend()
        backend.search_traces.return_value = []

        result = json.loads(await find_errors(backend, limit=1000))

        assert "error" not in result
        backend.search_traces.assert_called_once()

    async def test_limit_at_minimum_boundary_is_accepted(self) -> None:
        backend = _backend()
        backend.search_traces.return_value = []

        result = json.loads(await find_errors(backend, limit=1))

        assert "error" not in result
        backend.search_traces.assert_called_once()


class TestBackendExceptionHandling:
    """A backend that raises must be caught and turned into error JSON."""

    async def test_search_traces_exception_is_caught_and_formatted(self) -> None:
        backend = _backend()
        backend.search_traces.side_effect = RuntimeError("backend down")

        result = json.loads(await find_errors(backend))

        assert result == {"error": "Failed to find error traces: backend down"}

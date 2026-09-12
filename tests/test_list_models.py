"""Tests for the list_models tool.

The tool fans out over search_traces -> get_trace, extracts LLM span
attributes, and aggregates per-model request counts / first-seen / last-seen
timestamps. The backend is mocked at the BaseBackend interface level (not
HTTP) since this module only ever talks to `backend.search_traces` and
`backend.get_trace`.
"""

import asyncio
import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.models import SpanData, TraceData, TraceQuery, TraceSummary
from opentelemetry_mcp.tools.list_models import list_models


def _make_span(
    *,
    trace_id: str = "t1",
    span_id: str = "s1",
    service_name: str = "svc",
    start_time: datetime,
    gen_ai_system: str = "openai",
    request_model: str | None = None,
    response_model: str | None = None,
    status: Any = "OK",
) -> SpanData:
    """Build a SpanData with gen_ai.* attributes for a given LLM call."""
    attrs: dict[str, Any] = {"gen_ai.system": gen_ai_system}
    if request_model:
        attrs["gen_ai.request.model"] = request_model
    if response_model:
        attrs["gen_ai.response.model"] = response_model

    return SpanData(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=None,
        operation_name="chat",
        service_name=service_name,
        start_time=start_time,
        duration_ms=100,
        status=status,
        attributes=SpanAttributes.model_validate(attrs),
    )


def _make_trace(
    trace_id: str,
    spans: list[SpanData],
    *,
    service_name: str = "svc",
    root_operation: str = "chat",
    status: Any = "OK",
) -> TraceData:
    return TraceData(
        trace_id=trace_id,
        spans=spans,
        start_time=spans[0].start_time,
        duration_ms=100,
        service_name=service_name,
        root_operation=root_operation,
        status=status,
    )


@pytest.fixture
def mock_backend() -> AsyncMock:
    """A fake backend exposing only the BaseBackend async interface."""
    return AsyncMock(spec=BaseBackend)


class TestListModelsHappyPath:
    """Real backend data flows through to the correct output shape."""

    async def test_single_model_from_sample_trace_data(
        self, mock_backend: AsyncMock, sample_trace_data: TraceData
    ) -> None:
        mock_backend.search_traces.return_value = [TraceSummary.from_trace(sample_trace_data)]
        mock_backend.get_trace.return_value = sample_trace_data

        result = await list_models(mock_backend)
        data = json.loads(result)

        assert data["count"] == 1
        model = data["models"][0]
        assert model["model"] == "gpt-4"
        assert model["provider"] == "openai"
        assert model["request_count"] == 1
        span = sample_trace_data.spans[0]
        assert model["first_seen"] == span.start_time.isoformat()
        assert model["last_seen"] == span.start_time.isoformat()

    async def test_prefers_response_model_over_request_model(self, mock_backend: AsyncMock) -> None:
        span = _make_span(
            start_time=datetime(2024, 1, 1, tzinfo=UTC),
            request_model="gpt-4",
            response_model="gpt-4-0613",
        )
        trace = _make_trace("t1", [span])
        mock_backend.search_traces.return_value = [TraceSummary.from_trace(trace)]
        mock_backend.get_trace.return_value = trace

        result = await list_models(mock_backend)
        data = json.loads(result)

        assert data["models"][0]["model"] == "gpt-4-0613"

    async def test_falls_back_to_request_model_when_no_response_model(
        self, mock_backend: AsyncMock
    ) -> None:
        span = _make_span(
            start_time=datetime(2024, 1, 1, tzinfo=UTC),
            request_model="gpt-4",
            response_model=None,
        )
        trace = _make_trace("t1", [span])
        mock_backend.search_traces.return_value = [TraceSummary.from_trace(trace)]
        mock_backend.get_trace.return_value = trace

        result = await list_models(mock_backend)
        data = json.loads(result)

        assert data["models"][0]["model"] == "gpt-4"

    async def test_unknown_model_when_no_model_name_present(self, mock_backend: AsyncMock) -> None:
        span = _make_span(
            start_time=datetime(2024, 1, 1, tzinfo=UTC),
            request_model=None,
            response_model=None,
        )
        trace = _make_trace("t1", [span])
        mock_backend.search_traces.return_value = [TraceSummary.from_trace(trace)]
        mock_backend.get_trace.return_value = trace

        result = await list_models(mock_backend)
        data = json.loads(result)

        assert data["models"][0]["model"] == "unknown"

    async def test_models_sorted_by_request_count_descending(self, mock_backend: AsyncMock) -> None:
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        t1 = datetime(2024, 1, 2, tzinfo=UTC)
        t2 = datetime(2024, 1, 3, tzinfo=UTC)

        trace_a = _make_trace("a", [_make_span(trace_id="a", start_time=t0, request_model="gpt-4")])
        trace_b = _make_trace("b", [_make_span(trace_id="b", start_time=t1, request_model="gpt-4")])
        trace_c = _make_trace(
            "c",
            [
                _make_span(
                    trace_id="c",
                    start_time=t2,
                    gen_ai_system="anthropic",
                    request_model="claude-3",
                )
            ],
        )

        mock_backend.search_traces.return_value = [
            TraceSummary.from_trace(t) for t in (trace_a, trace_b, trace_c)
        ]
        traces_by_id = {"a": trace_a, "b": trace_b, "c": trace_c}
        mock_backend.get_trace.side_effect = lambda trace_id: traces_by_id[trace_id]

        result = await list_models(mock_backend)
        data = json.loads(result)

        assert data["count"] == 2
        assert data["models"][0]["model"] == "gpt-4"
        assert data["models"][0]["request_count"] == 2
        assert data["models"][1]["model"] == "claude-3"
        assert data["models"][1]["request_count"] == 1

    async def test_first_seen_and_last_seen_span_multiple_spans_same_model(
        self, mock_backend: AsyncMock
    ) -> None:
        earliest = datetime(2024, 1, 1, tzinfo=UTC)
        middle = datetime(2024, 1, 5, tzinfo=UTC)
        latest = datetime(2024, 1, 10, tzinfo=UTC)

        # Spans intentionally out of chronological order to exercise the
        # min/max comparison logic rather than just "first span wins".
        spans = [
            _make_span(span_id="s2", start_time=middle, request_model="gpt-4"),
            _make_span(span_id="s1", start_time=earliest, request_model="gpt-4"),
            _make_span(span_id="s3", start_time=latest, request_model="gpt-4"),
        ]
        trace = _make_trace("t1", spans)
        mock_backend.search_traces.return_value = [TraceSummary.from_trace(trace)]
        mock_backend.get_trace.return_value = trace

        result = await list_models(mock_backend)
        data = json.loads(result)

        model = data["models"][0]
        assert model["request_count"] == 3
        assert model["first_seen"] == earliest.isoformat()
        assert model["last_seen"] == latest.isoformat()


class TestListModelsConcurrency:
    """Regression test for the N+1 sequential-fetch bug: `get_trace` calls
    must be issued concurrently (via asyncio.gather), not awaited one trace
    at a time. Under the old sequential-loop implementation this test's
    max_in_flight would be 1; the fixed implementation fetches all traces
    concurrently, so it must be equal to the number of traces."""

    async def test_get_trace_calls_are_concurrent_not_sequential(
        self, mock_backend: AsyncMock
    ) -> None:
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        traces = [
            _make_trace(
                f"t{i}",
                [_make_span(trace_id=f"t{i}", start_time=t0, request_model="gpt-4")],
            )
            for i in range(3)
        ]
        mock_backend.search_traces.return_value = [TraceSummary.from_trace(t) for t in traces]
        traces_by_id = {t.trace_id: t for t in traces}

        in_flight = 0
        max_in_flight = 0

        async def _get_trace(trace_id: str) -> TraceData:
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return traces_by_id[trace_id]

        mock_backend.get_trace.side_effect = _get_trace

        await list_models(mock_backend)

        assert max_in_flight == len(traces)


class TestListModelsQueryConstruction:
    """The tool's params must actually reach the backend's TraceQuery."""

    async def test_params_are_translated_into_trace_query(self, mock_backend: AsyncMock) -> None:
        mock_backend.search_traces.return_value = []

        await list_models(
            mock_backend,
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-02T00:00:00Z",
            service_name="svc-a",
            gen_ai_system="anthropic",
            limit=50,
        )

        query = mock_backend.search_traces.call_args.args[0]
        assert isinstance(query, TraceQuery)
        assert query.service_name == "svc-a"
        assert query.gen_ai_system == "anthropic"
        assert query.limit == 50
        assert query.start_time == datetime(2024, 1, 1, tzinfo=UTC)
        assert query.end_time == datetime(2024, 1, 2, tzinfo=UTC)


class TestListModelsValidationErrors:
    """Invalid inputs must produce {"error": ...} JSON, never raise."""

    async def test_invalid_start_time_returns_error_json(self, mock_backend: AsyncMock) -> None:
        result = await list_models(mock_backend, start_time="not-a-timestamp")
        data = json.loads(result)

        assert "error" in data
        assert "start_time" in data["error"]
        mock_backend.search_traces.assert_not_called()

    async def test_invalid_end_time_returns_error_json(self, mock_backend: AsyncMock) -> None:
        result = await list_models(
            mock_backend, start_time="2024-01-01T00:00:00Z", end_time="also-not-a-timestamp"
        )
        data = json.loads(result)

        assert "error" in data
        assert "end_time" in data["error"]
        mock_backend.search_traces.assert_not_called()

    async def test_limit_out_of_bounds_returns_error_json_not_raise(
        self, mock_backend: AsyncMock
    ) -> None:
        """TraceQuery enforces limit ge=1/le=1000; the pydantic ValidationError
        raised while building the query must be caught by the tool's own
        try/except and surfaced as an error payload, not propagate."""
        result = await list_models(mock_backend, limit=0)
        data = json.loads(result)

        assert "error" in data
        assert "Failed to list models" in data["error"]
        mock_backend.search_traces.assert_not_called()


class TestListModelsBackendExceptionHandling:
    """A raising backend must be turned into an error payload, not crash."""

    async def test_search_traces_exception_is_caught(self, mock_backend: AsyncMock) -> None:
        mock_backend.search_traces.side_effect = RuntimeError("backend unreachable")

        result = await list_models(mock_backend)
        data = json.loads(result)

        assert data == {"error": "Failed to list models: backend unreachable"}

    async def test_get_trace_exception_is_caught(self, mock_backend: AsyncMock) -> None:
        span = _make_span(start_time=datetime(2024, 1, 1, tzinfo=UTC), request_model="gpt-4")
        trace = _make_trace("t1", [span])
        mock_backend.search_traces.return_value = [TraceSummary.from_trace(trace)]
        mock_backend.get_trace.side_effect = RuntimeError("trace fetch failed")

        result = await list_models(mock_backend)
        data = json.loads(result)

        assert data == {"error": "Failed to list models: trace fetch failed"}


class TestListModelsEdgeCases:
    async def test_no_traces_returns_empty_models_list(self, mock_backend: AsyncMock) -> None:
        mock_backend.search_traces.return_value = []

        result = await list_models(mock_backend)
        data = json.loads(result)

        assert data == {"count": 0, "models": []}
        mock_backend.get_trace.assert_not_called()

    async def test_trace_with_no_llm_spans_produces_no_models(
        self, mock_backend: AsyncMock
    ) -> None:
        non_llm_span = SpanData(
            trace_id="t1",
            span_id="s1",
            parent_span_id=None,
            operation_name="http.request",
            service_name="svc",
            start_time=datetime(2024, 1, 1, tzinfo=UTC),
            duration_ms=50,
            status="OK",
            attributes=SpanAttributes.model_validate({}),
        )
        trace = _make_trace("t1", [non_llm_span])
        mock_backend.search_traces.return_value = [TraceSummary.from_trace(trace)]
        mock_backend.get_trace.return_value = trace

        result = await list_models(mock_backend)
        data = json.loads(result)

        assert data == {"count": 0, "models": []}

"""Tests for the get_trace tool."""

import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.models import SpanData, TraceData
from opentelemetry_mcp.tools.trace import get_trace


def _make_span(
    *,
    span_id: str = "span1",
    parent_span_id: str | None = None,
    operation_name: str = "op",
    service_name: str = "svc",
    duration_ms: float = 100.0,
    status: str = "OK",
    attrs: dict[str, Any] | None = None,
) -> SpanData:
    """Build a SpanData instance for a test, defaulting to no gen_ai attributes."""
    return SpanData(
        trace_id="t1",
        span_id=span_id,
        parent_span_id=parent_span_id,
        operation_name=operation_name,
        service_name=service_name,
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        duration_ms=duration_ms,
        status=status,  # type: ignore[arg-type]
        attributes=SpanAttributes.model_validate(attrs or {}),
    )


def _make_trace(spans: list[SpanData], **overrides: Any) -> TraceData:
    defaults: dict[str, Any] = {
        "trace_id": "t1",
        "start_time": datetime(2024, 1, 1, 0, 0, 0),
        "duration_ms": 100.0,
        "service_name": "svc",
        "root_operation": "root-op",
        "status": "OK",
    }
    defaults.update(overrides)
    return TraceData(spans=spans, **defaults)


def _fake_backend() -> AsyncMock:
    return AsyncMock(spec=BaseBackend)


async def test_get_trace_happy_path_returns_full_shape(sample_trace_data: TraceData) -> None:
    """A trace with one LLM span produces the documented top-level shape,
    a fully populated span entry, and an llm_summary block."""
    backend = _fake_backend()
    backend.get_trace.return_value = sample_trace_data

    raw = await get_trace(backend, "abc123")
    result = json.loads(raw)

    backend.get_trace.assert_awaited_once_with("abc123")

    assert result["trace_id"] == "abc123"
    assert result["service_name"] == "test-service"
    assert result["root_operation"] == "test_operation"
    assert result["duration_ms"] == 5000
    assert result["status"] == "OK"
    assert result["span_count"] == 1
    assert result["has_errors"] is False

    span_data = result["spans"][0]
    assert span_data["span_id"] == "span1"
    assert span_data["parent_span_id"] is None
    assert span_data["attributes"]["gen_ai.system"] == "openai"
    assert span_data["llm_attributes"]["system"] == "openai"
    assert span_data["llm_attributes"]["request_model"] == "gpt-4"
    assert span_data["llm_attributes"]["total_tokens"] == 300

    assert result["llm_summary"]["llm_span_count"] == 1
    assert result["llm_summary"]["total_tokens"] == 300
    assert result["llm_summary"]["models_used"] == ["gpt-4"]


async def test_get_trace_non_llm_span_omits_llm_attributes_and_summary() -> None:
    """A span with no gen_ai.* attributes must not get an llm_attributes
    entry, and a trace with no LLM spans at all must omit llm_summary."""
    backend = _fake_backend()
    span = _make_span(attrs=None)
    backend.get_trace.return_value = _make_trace([span])

    result = json.loads(await get_trace(backend, "t1"))

    assert "llm_attributes" not in result["spans"][0]
    assert "llm_summary" not in result


async def test_get_trace_empty_spans_list() -> None:
    """A trace fetched with zero spans still produces a valid, well-shaped
    response rather than raising on empty iteration."""
    backend = _fake_backend()
    backend.get_trace.return_value = _make_trace([])

    result = json.loads(await get_trace(backend, "t1"))

    assert result["spans"] == []
    assert result["span_count"] == 0
    assert result["has_errors"] is False
    assert "llm_summary" not in result


async def test_get_trace_backend_exception_returns_error_json() -> None:
    """When the backend raises (e.g. trace not found), get_trace must catch
    it and return an error JSON payload instead of propagating."""
    backend = _fake_backend()
    backend.get_trace.side_effect = ValueError("no spans found for trace_id")

    raw = await get_trace(backend, "missing-trace")
    result = json.loads(raw)

    assert result == {"error": "Failed to fetch trace: no spans found for trace_id"}


async def test_get_trace_error_status_span_propagates_has_errors() -> None:
    """A single ERROR-status span must flip the top-level has_errors flag."""
    backend = _fake_backend()
    ok_span = _make_span(span_id="s1", status="OK")
    error_span = _make_span(span_id="s2", status="ERROR")
    backend.get_trace.return_value = _make_trace([ok_span, error_span], status="ERROR")

    result = json.loads(await get_trace(backend, "t1"))

    assert result["has_errors"] is True
    statuses = {s["span_id"]: s["status"] for s in result["spans"]}
    assert statuses == {"s1": "OK", "s2": "ERROR"}


class TestModelsUsedDedup:
    """models_used is built as a set keyed on request_model-or-response_model;
    verify it actually dedups rather than just passing through unique input."""

    async def test_dedups_when_same_model_seen_via_request_and_response_fields(self) -> None:
        backend = _fake_backend()
        span_a = _make_span(
            span_id="a",
            attrs={"gen_ai.system": "openai", "gen_ai.request.model": "gpt-4"},
        )
        span_b = _make_span(
            span_id="b",
            attrs={"gen_ai.system": "openai", "gen_ai.response.model": "gpt-4"},
        )
        backend.get_trace.return_value = _make_trace([span_a, span_b])

        result = json.loads(await get_trace(backend, "t1"))

        assert result["llm_summary"]["llm_span_count"] == 2
        assert result["llm_summary"]["models_used"] == ["gpt-4"]

    async def test_keeps_distinct_models_separate(self) -> None:
        backend = _fake_backend()
        span_a = _make_span(
            span_id="a",
            attrs={"gen_ai.system": "openai", "gen_ai.request.model": "gpt-4"},
        )
        span_b = _make_span(
            span_id="b",
            attrs={"gen_ai.system": "anthropic", "gen_ai.request.model": "claude-3-opus"},
        )
        backend.get_trace.return_value = _make_trace([span_a, span_b])

        result = json.loads(await get_trace(backend, "t1"))

        assert sorted(result["llm_summary"]["models_used"]) == ["claude-3-opus", "gpt-4"]


async def test_get_trace_sums_tokens_across_multiple_llm_spans() -> None:
    """total_tokens in llm_summary must be the sum across every LLM span in
    the trace, not just the first one."""
    backend = _fake_backend()
    span_a = _make_span(
        span_id="a",
        attrs={
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4",
            "gen_ai.usage.total_tokens": 100,
        },
    )
    span_b = _make_span(
        span_id="b",
        attrs={
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4",
            "gen_ai.usage.total_tokens": 250,
        },
    )
    backend.get_trace.return_value = _make_trace([span_a, span_b])

    result = json.loads(await get_trace(backend, "t1"))

    assert result["llm_summary"]["total_tokens"] == 350
    assert result["llm_summary"]["llm_span_count"] == 2


async def test_get_trace_empty_gen_ai_system_is_counted_but_has_no_llm_attributes() -> None:
    """SpanData.is_llm_span only checks `gen_ai_system is not None`, so an
    empty string still counts as an LLM span for is_llm_span and therefore
    for trace.llm_spans / llm_span_count. But LLMSpanAttributes.from_span
    treats an empty string as falsy and returns None. The result is a span
    that inflates llm_span_count without ever getting an llm_attributes
    entry, and without contributing to total_tokens. This looks like a real
    inconsistency in the module's counting logic (flagged, not fixed here).
    """
    backend = _fake_backend()
    span = _make_span(
        span_id="a",
        attrs={"gen_ai.system": "", "gen_ai.usage.total_tokens": 500},
    )
    backend.get_trace.return_value = _make_trace([span])

    result = json.loads(await get_trace(backend, "t1"))

    assert "llm_attributes" not in result["spans"][0]
    assert result["llm_summary"]["llm_span_count"] == 1
    assert result["llm_summary"]["total_tokens"] == 0

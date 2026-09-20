"""Tests for the triage_trace tool implementation (tools/triage.py).

Mocks at the `backend: AsyncMock(spec=BaseBackend)` level per
test_investigate.py's convention, driving real TraceData/SpanData fixtures
through the real span_tree.py algorithms - only backend.get_trace itself is
faked, everything downstream is exercised for real.
"""

from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from unittest.mock import AsyncMock

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.models import SpanData, TraceData
from opentelemetry_mcp.tools.triage import triage_trace

_BASE_TIME = datetime(2024, 1, 1, tzinfo=UTC)


def _span(
    span_id: str,
    *,
    parent_span_id: str | None = None,
    start_offset_ms: float = 0.0,
    duration_ms: float = 100.0,
    status: Literal["OK", "ERROR", "UNSET"] = "OK",
    attributes: dict[str, Any] | None = None,
) -> SpanData:
    return SpanData(
        trace_id="t1",
        span_id=span_id,
        parent_span_id=parent_span_id,
        operation_name=f"op-{span_id}",
        service_name="svc",
        start_time=_BASE_TIME + timedelta(milliseconds=start_offset_ms),
        duration_ms=duration_ms,
        status=status,
        attributes=SpanAttributes.model_validate(attributes or {}),
    )


def _trace(spans: list[SpanData]) -> TraceData:
    return TraceData(
        trace_id="t1",
        spans=spans,
        start_time=_BASE_TIME,
        duration_ms=100.0,
        service_name="svc",
        root_operation="op-root",
        status="OK",
    )


def _backend() -> AsyncMock:
    return AsyncMock(spec=BaseBackend)


class TestNoSpans:
    async def test_no_spans_returns_low_confidence_with_no_root_cause(self) -> None:
        backend = _backend()
        backend.get_trace.return_value = _trace([])

        result = await triage_trace(backend, "t1")

        assert result.verdict.likely_root_cause is None
        assert result.verdict.confidence == "low"
        assert "no spans" in result.verdict.reasoning.lower()
        assert result.critical_path == []
        assert result.top_latency_contributors == []
        assert result.error_chain is None


class TestErrorDrivenVerdict:
    def _single_error_chain_trace(self) -> TraceData:
        root = _span("root", duration_ms=100.0, status="OK")
        mid = _span("mid", parent_span_id="root", duration_ms=50.0, status="OK")
        leaf = _span(
            "leaf",
            parent_span_id="mid",
            duration_ms=20.0,
            status="ERROR",
            attributes={"error.message": "boom", "error.type": "ValueError"},
        )
        return _trace([root, mid, leaf])

    async def test_single_deepest_error_chain_yields_high_confidence(self) -> None:
        backend = _backend()
        backend.get_trace.return_value = self._single_error_chain_trace()

        result = await triage_trace(backend, "t1")

        assert result.verdict.confidence == "high"
        assert result.verdict.likely_root_cause is not None
        assert result.verdict.likely_root_cause.span_id == "leaf"
        assert "deepest error span" in result.verdict.reasoning.lower()
        assert result.error_chain is not None
        assert [s.span_id for s in result.error_chain] == ["root", "mid", "leaf"]
        assert [s.span_id for s in result.critical_path] == ["root", "mid", "leaf"]

    async def test_multiple_equally_deep_error_chains_yield_medium_confidence(self) -> None:
        root = _span("root", status="OK")
        error_a = _span("error_a", parent_span_id="root", status="ERROR")
        error_b = _span("error_b", parent_span_id="root", status="ERROR")
        backend = _backend()
        backend.get_trace.return_value = _trace([root, error_a, error_b])

        result = await triage_trace(backend, "t1")

        assert result.verdict.confidence == "medium"
        assert "equally-deep" in result.verdict.reasoning.lower()

    async def test_detail_level_full_attaches_error_detail_to_root_cause(self) -> None:
        backend = _backend()
        backend.get_trace.return_value = self._single_error_chain_trace()

        result = await triage_trace(backend, "t1", detail_level="full")

        root_cause = result.verdict.likely_root_cause
        assert root_cause is not None
        assert root_cause.error_detail is not None
        assert root_cause.error_detail["error_message"] == "boom"
        assert root_cause.error_detail["error_type"] == "ValueError"

    async def test_detail_level_summary_leaves_error_detail_none(self) -> None:
        backend = _backend()
        backend.get_trace.return_value = self._single_error_chain_trace()

        result = await triage_trace(backend, "t1", detail_level="summary")

        root_cause = result.verdict.likely_root_cause
        assert root_cause is not None
        assert root_cause.error_detail is None

    async def test_detail_level_full_error_detail_matches_between_verdict_and_error_chain(
        self,
    ) -> None:
        backend = _backend()
        backend.get_trace.return_value = self._single_error_chain_trace()

        result = await triage_trace(backend, "t1", detail_level="full")

        assert result.error_chain is not None
        assert result.verdict.likely_root_cause == result.error_chain[-1]
        assert result.error_chain[-1].error_detail is not None
        assert result.error_chain[-1].error_detail["error_message"] == "boom"

    async def test_error_under_a_second_root_is_still_found(self) -> None:
        # Two disconnected root spans in one trace (e.g. an async
        # fire-and-forget span, or partial context propagation) - the error
        # lives entirely under the second root, which must not be ignored.
        root1 = _span("root1", status="OK", duration_ms=10.0)
        root2 = _span("root2", status="OK", duration_ms=10.0)
        error_leaf = _span(
            "error_leaf",
            parent_span_id="root2",
            status="ERROR",
            attributes={"error.message": "boom"},
        )
        backend = _backend()
        backend.get_trace.return_value = _trace([root1, root2, error_leaf])

        result = await triage_trace(backend, "t1")

        assert result.verdict.confidence == "high"
        assert result.verdict.likely_root_cause is not None
        assert result.verdict.likely_root_cause.span_id == "error_leaf"
        assert result.error_chain is not None
        assert [s.span_id for s in result.error_chain] == ["root2", "error_leaf"]
        # Critical path is computed for the root the verdict is actually about.
        assert [s.span_id for s in result.critical_path] == ["root2", "error_leaf"]


class TestPureLatencyFallback:
    async def test_no_error_falls_back_to_highest_self_time_span_with_low_confidence(
        self,
    ) -> None:
        root = _span("root", duration_ms=100.0, status="OK")
        leaf = _span(
            "leaf", parent_span_id="root", start_offset_ms=10.0, duration_ms=80.0, status="OK"
        )
        backend = _backend()
        backend.get_trace.return_value = _trace([root, leaf])

        result = await triage_trace(backend, "t1")

        assert result.verdict.confidence == "low"
        assert result.verdict.likely_root_cause is not None
        assert result.verdict.likely_root_cause.span_id == "leaf"
        assert "pure-latency" in result.verdict.reasoning.lower()
        assert result.error_chain is None

    async def test_multi_root_trace_picks_critical_path_for_the_owning_root(self) -> None:
        root1 = _span("root1", status="OK", duration_ms=5.0)
        root2 = _span("root2", status="OK", duration_ms=100.0)
        busy_leaf = _span("busy_leaf", parent_span_id="root2", status="OK", duration_ms=80.0)
        backend = _backend()
        backend.get_trace.return_value = _trace([root1, root2, busy_leaf])

        result = await triage_trace(backend, "t1")

        assert result.verdict.confidence == "low"
        assert result.verdict.likely_root_cause is not None
        assert result.verdict.likely_root_cause.span_id == "busy_leaf"
        assert [s.span_id for s in result.critical_path] == ["root2", "busy_leaf"]


class TestUnreachableErrorFallback:
    async def test_error_in_a_disconnected_cycle_is_reported_not_hidden_as_pure_latency(
        self,
    ) -> None:
        # root is a genuine, reachable root with no error. a/b form a
        # separate 2-cycle (a's parent is b, b's parent is a - both
        # present) that find_roots can never discover, with an ERROR span
        # trapped inside it. Before this fix, this trace's error would be
        # invisible to error_chains and silently fall through to the
        # pure-latency fallback, misreporting the trace as error-free.
        root = _span("root", status="OK", duration_ms=50.0)
        a = _span("a", parent_span_id="b", status="ERROR", duration_ms=10.0)
        b = _span("b", parent_span_id="a", status="OK", duration_ms=5.0)
        backend = _backend()
        backend.get_trace.return_value = _trace([root, a, b])

        result = await triage_trace(backend, "t1")

        assert result.verdict.confidence == "low"
        assert result.verdict.likely_root_cause is not None
        assert result.verdict.likely_root_cause.span_id == "a"
        assert "not reachable" in result.verdict.reasoning.lower()
        assert result.error_chain is not None
        assert [s.span_id for s in result.error_chain] == ["a"]

    async def test_real_root_error_chain_takes_priority_over_a_disconnected_cycle(self) -> None:
        # A real, reachable error chain from `root` must win even when an
        # unrelated disconnected cyclic island also contains an error.
        root = _span("root", status="OK", duration_ms=50.0)
        real_error = _span("real_error", parent_span_id="root", status="ERROR", duration_ms=20.0)
        a = _span("a", parent_span_id="b", status="ERROR", duration_ms=10.0)
        b = _span("b", parent_span_id="a", status="OK", duration_ms=5.0)
        backend = _backend()
        backend.get_trace.return_value = _trace([root, real_error, a, b])

        result = await triage_trace(backend, "t1")

        assert result.verdict.likely_root_cause is not None
        assert result.verdict.likely_root_cause.span_id == "real_error"
        assert result.verdict.confidence == "high"


class TestBackendInteraction:
    async def test_get_trace_awaited_with_the_requested_trace_id(self) -> None:
        backend = _backend()
        backend.get_trace.return_value = _trace([_span("root")])

        await triage_trace(backend, "specific-trace-id")

        backend.get_trace.assert_awaited_once_with("specific-trace-id")

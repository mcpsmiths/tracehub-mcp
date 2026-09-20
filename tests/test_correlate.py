"""Tests for the correlate_trace tool implementation (tools/correlate.py).

Mocks both backends at the `AsyncMock(spec=BaseBackend)` level per
test_investigate.py's convention - the real span_tree.py error-chain walk
(via _error_signal) runs for real against the TraceData fixtures built here,
only the backend I/O itself is faked.
"""

from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from unittest.mock import AsyncMock

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.models import SpanData, TraceData, TraceQuery
from opentelemetry_mcp.tools.correlate import correlate_trace

_BASE_TIME = datetime(2024, 1, 1, tzinfo=UTC)


def _span(
    span_id: str,
    *,
    parent_span_id: str | None = None,
    service_name: str = "svc-a",
    status: Literal["OK", "ERROR", "UNSET"] = "OK",
    duration_ms: float = 100.0,
    attributes: dict[str, Any] | None = None,
) -> SpanData:
    return SpanData(
        trace_id="t1",
        span_id=span_id,
        parent_span_id=parent_span_id,
        operation_name=f"op-{span_id}",
        service_name=service_name,
        start_time=_BASE_TIME,
        duration_ms=duration_ms,
        status=status,
        attributes=SpanAttributes.model_validate(attributes or {}),
    )


def _trace(spans: list[SpanData], *, trace_id: str = "t1") -> TraceData:
    return TraceData(
        trace_id=trace_id,
        spans=spans,
        start_time=_BASE_TIME,
        duration_ms=100.0,
        service_name=spans[0].service_name if spans else "svc-a",
        root_operation="root-op",
        status="OK",
    )


def _backend() -> AsyncMock:
    return AsyncMock(spec=BaseBackend)


class TestDirectMatch:
    async def test_direct_match_returns_a_single_high_confidence_match(self) -> None:
        primary = _backend()
        primary.get_trace.return_value = _trace([_span("p1", service_name="svc-a")])
        secondary = _backend()
        secondary.get_trace.return_value = _trace([_span("s1", service_name="svc-a")])

        result = await correlate_trace(primary, secondary, "t1")

        assert len(result.matches) == 1
        match = result.matches[0]
        assert match.secondary_trace_id == "t1"
        assert match.correlation_method == "trace_id_match"
        assert match.confidence == "high"
        assert match.service_overlap == ["svc-a"]
        assert match.root_cause_consistent is None  # neither trace has an error
        secondary.search_traces.assert_not_awaited()

    async def test_direct_match_short_circuits_before_the_heuristic_path(self) -> None:
        primary = _backend()
        primary.get_trace.return_value = _trace([_span("p1")])
        secondary = _backend()
        secondary.get_trace.return_value = _trace([_span("s1")])

        await correlate_trace(primary, secondary, "t1")

        secondary.search_traces.assert_not_awaited()

    async def test_primary_trace_with_no_spans_yields_no_candidates_but_no_crash(self) -> None:
        primary = _backend()
        primary.get_trace.return_value = _trace([])
        secondary = _backend()
        secondary.get_trace.side_effect = ValueError("not found")

        result = await correlate_trace(primary, secondary, "t1")

        assert result.matches == []
        secondary.search_traces.assert_not_awaited()

    async def test_limitations_are_always_present(self) -> None:
        primary = _backend()
        primary.get_trace.return_value = _trace([_span("p1")])
        secondary = _backend()
        secondary.get_trace.return_value = _trace([_span("s1")])

        result = await correlate_trace(primary, secondary, "t1")

        assert len(result.limitations) > 0


class TestRootCauseConsistency:
    async def test_neither_trace_has_error_yields_none(self) -> None:
        primary = _backend()
        primary.get_trace.return_value = _trace([_span("p1", status="OK")])
        secondary = _backend()
        secondary.get_trace.return_value = _trace([_span("s1", status="OK")])

        result = await correlate_trace(primary, secondary, "t1")

        assert result.matches[0].root_cause_consistent is None

    async def test_both_error_same_service_yields_true(self) -> None:
        primary = _backend()
        primary.get_trace.return_value = _trace([_span("p1", service_name="svc-a", status="ERROR")])
        secondary = _backend()
        secondary.get_trace.return_value = _trace(
            [_span("s1", service_name="svc-a", status="ERROR")]
        )

        result = await correlate_trace(primary, secondary, "t1")

        assert result.matches[0].root_cause_consistent is True

    async def test_both_error_different_service_yields_false(self) -> None:
        primary = _backend()
        primary.get_trace.return_value = _trace([_span("p1", service_name="svc-a", status="ERROR")])
        secondary = _backend()
        secondary.get_trace.return_value = _trace(
            [_span("s1", service_name="svc-b", status="ERROR")]
        )

        result = await correlate_trace(primary, secondary, "t1")

        assert result.matches[0].root_cause_consistent is False

    async def test_only_primary_has_error_yields_false(self) -> None:
        primary = _backend()
        primary.get_trace.return_value = _trace([_span("p1", service_name="svc-a", status="ERROR")])
        secondary = _backend()
        secondary.get_trace.return_value = _trace([_span("s1", service_name="svc-a", status="OK")])

        result = await correlate_trace(primary, secondary, "t1")

        assert result.matches[0].root_cause_consistent is False


class TestUnreachableErrorDetection:
    async def test_error_in_a_disconnected_cycle_is_still_detected(self) -> None:
        # a's parent is b and b's parent is a - both present, so no
        # root-first walk (including find_all_error_chains from a real
        # root) can discover this cluster at all. Before this fix,
        # _error_signal would report has_error=False for a trace that
        # genuinely contains an ERROR span.
        root = _span("root", service_name="svc-a", status="OK")
        a = _span("a", parent_span_id="b", service_name="svc-a", status="ERROR")
        b = _span("b", parent_span_id="a", service_name="svc-a", status="OK")
        primary = _backend()
        primary.get_trace.return_value = _trace([root, a, b])
        secondary = _backend()
        secondary.get_trace.return_value = _trace([_span("s1", service_name="svc-a", status="OK")])

        result = await correlate_trace(primary, secondary, "t1")

        assert result.matches[0].root_cause_consistent is False  # primary errors, secondary doesn't


class TestHeuristicFallback:
    async def test_direct_match_failure_falls_back_to_time_window_search(self) -> None:
        primary = _backend()
        primary.get_trace.return_value = _trace([_span("p1", service_name="svc-a")])
        secondary = _backend()
        secondary.get_trace.side_effect = ValueError("not found")
        secondary.search_traces.return_value = [
            _trace([_span("c1", service_name="svc-a")], trace_id="c1")
        ]

        result = await correlate_trace(primary, secondary, "t1")

        secondary.search_traces.assert_awaited_once()
        assert len(result.matches) == 1
        assert result.matches[0].correlation_method == "time_window_heuristic"
        assert result.matches[0].confidence == "low"
        assert result.matches[0].secondary_trace_id == "c1"

    async def test_candidate_matching_the_primary_trace_id_itself_is_excluded(self) -> None:
        """Regression: get_trace(trace_id) failing doesn't mean
        search_traces can never also turn up that same trace_id (different
        endpoint, different failure mode) - if it does, reporting it as a
        'heuristic, low-confidence' match would be misleading, since it's
        not independent evidence of anything, just the input echoed back."""
        primary = _backend()
        primary.get_trace.return_value = _trace([_span("p1", service_name="svc-a")])
        secondary = _backend()
        secondary.get_trace.side_effect = ValueError("not found")
        secondary.search_traces.return_value = [
            _trace([_span("self1", service_name="svc-a")], trace_id="t1"),  # same as primary
            _trace([_span("c1", service_name="svc-a")], trace_id="c1"),
        ]

        result = await correlate_trace(primary, secondary, "t1")

        assert [m.secondary_trace_id for m in result.matches] == ["c1"]

    async def test_search_uses_a_padded_window_around_the_primary_trace(self) -> None:
        primary = _backend()
        primary.get_trace.return_value = _trace([_span("p1", service_name="svc-a")])
        secondary = _backend()
        secondary.get_trace.side_effect = ValueError("not found")
        secondary.search_traces.return_value = []

        await correlate_trace(primary, secondary, "t1")

        query: TraceQuery = secondary.search_traces.await_args.args[0]
        assert query.service_name == "svc-a"
        assert query.start_time == _BASE_TIME - timedelta(seconds=5)
        assert query.end_time == _BASE_TIME + timedelta(milliseconds=100.0) + timedelta(seconds=5)

    async def test_candidate_with_no_service_overlap_is_filtered_out(self) -> None:
        primary = _backend()
        primary.get_trace.return_value = _trace([_span("p1", service_name="svc-a")])
        secondary = _backend()
        secondary.get_trace.side_effect = ValueError("not found")
        secondary.search_traces.return_value = [_trace([_span("c1", service_name="svc-z")])]

        result = await correlate_trace(primary, secondary, "t1")

        assert result.matches == []

    async def test_no_candidates_found_yields_empty_matches_but_limitations_stay(self) -> None:
        primary = _backend()
        primary.get_trace.return_value = _trace([_span("p1", service_name="svc-a")])
        secondary = _backend()
        secondary.get_trace.side_effect = ValueError("not found")
        secondary.search_traces.return_value = []

        result = await correlate_trace(primary, secondary, "t1")

        assert result.matches == []
        assert len(result.limitations) > 0

    async def test_search_traces_exception_is_swallowed_not_raised(self) -> None:
        primary = _backend()
        primary.get_trace.return_value = _trace([_span("p1", service_name="svc-a")])
        secondary = _backend()
        secondary.get_trace.side_effect = ValueError("not found")
        secondary.search_traces.side_effect = RuntimeError("backend unavailable")

        result = await correlate_trace(primary, secondary, "t1")

        assert result.matches == []

    async def test_candidates_are_ranked_by_service_overlap_and_capped_at_five(self) -> None:
        primary = _backend()
        primary.get_trace.return_value = _trace(
            [_span("p1", service_name="svc-a"), _span("p2", service_name="svc-b")]
        )
        secondary = _backend()
        secondary.get_trace.side_effect = ValueError("not found")

        high_overlap = [
            _trace(
                [_span(f"hi{i}a", service_name="svc-a"), _span(f"hi{i}b", service_name="svc-b")],
                trace_id=f"high-{i}",
            )
            for i in range(3)
        ]
        low_overlap = [
            _trace([_span(f"lo{i}", service_name="svc-a")], trace_id=f"low-{i}") for i in range(4)
        ]

        async def _search(query: TraceQuery) -> list[TraceData]:
            if query.service_name == "svc-a":
                return [*high_overlap, *low_overlap]
            return []

        secondary.search_traces.side_effect = _search

        result = await correlate_trace(primary, secondary, "t1")

        assert len(result.matches) == 5
        assert [m.secondary_trace_id for m in result.matches[:3]] == [
            "high-0",
            "high-1",
            "high-2",
        ]
        assert all(len(m.service_overlap) == 2 for m in result.matches[:3])
        assert all(len(m.service_overlap) == 1 for m in result.matches[3:])

    async def test_search_queries_once_per_distinct_primary_service(self) -> None:
        primary = _backend()
        primary.get_trace.return_value = _trace(
            [_span("p1", service_name="svc-a"), _span("p2", service_name="svc-b")]
        )
        secondary = _backend()
        secondary.get_trace.side_effect = ValueError("not found")
        secondary.search_traces.return_value = []

        await correlate_trace(primary, secondary, "t1")

        assert secondary.search_traces.await_count == 2
        queried_services = {
            call.args[0].service_name for call in secondary.search_traces.await_args_list
        }
        assert queried_services == {"svc-a", "svc-b"}

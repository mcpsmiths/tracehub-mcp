"""Tests for the correlate_trace tool implementation (tools/correlate.py).

Mocks both backends at the `AsyncMock(spec=BaseBackend)` level per
test_investigate.py's convention - the real span_tree.py error-chain walk
(via _error_signal) runs for real against the TraceData fixtures built here,
only the backend I/O itself is faked.
"""

import tracemalloc
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from unittest.mock import AsyncMock

import pytest

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.models import SpanData, TraceData, TraceQuery
from opentelemetry_mcp.tools import correlate as correlate_module
from opentelemetry_mcp.tools.correlate import (
    _LIMITATIONS,
    _MAX_SEARCH_WINDOW,
    _MAX_SERVICE_QUERIES,
    _SEARCH_WINDOW_PAD,
    _error_signal,
    correlate_trace,
)

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


def _trace(spans: list[SpanData], *, trace_id: str = "t1", duration_ms: float = 100.0) -> TraceData:
    return TraceData(
        trace_id=trace_id,
        spans=spans,
        start_time=_BASE_TIME,
        duration_ms=duration_ms,
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

    async def test_deepest_chain_under_a_later_root_is_the_blamed_service(self) -> None:
        # Parity guard for the deepest-only error-chain walk: a two-root
        # primary whose shallow error (depth 2) hangs off the FIRST root and
        # whose deeper error (depth 3) hangs off the SECOND root must still
        # blame the deeper chain's service - the global maximum across all
        # roots, not the first root's own maximum - exactly as the previous
        # flatten-then-max selection did.
        root1 = _span("root1", service_name="svc-shallow", status="OK")
        shallow_error = _span(
            "shallow_error", parent_span_id="root1", service_name="svc-shallow", status="ERROR"
        )
        root2 = _span("root2", service_name="svc-deep", status="OK")
        mid = _span("mid", parent_span_id="root2", service_name="svc-deep", status="OK")
        deep_error = _span(
            "deep_error", parent_span_id="mid", service_name="svc-deep", status="ERROR"
        )
        primary = _backend()
        primary.get_trace.return_value = _trace([root1, shallow_error, root2, mid, deep_error])

        blames_deep = _backend()
        blames_deep.get_trace.return_value = _trace(
            [_span("s1", service_name="svc-deep", status="ERROR")]
        )
        blames_shallow = _backend()
        blames_shallow.get_trace.return_value = _trace(
            [_span("s1", service_name="svc-shallow", status="ERROR")]
        )

        assert (await correlate_trace(primary, blames_deep, "t1")).matches[
            0
        ].root_cause_consistent is True
        assert (await correlate_trace(primary, blames_shallow, "t1")).matches[
            0
        ].root_cause_consistent is False


class TestUnreachableErrorDetection:
    async def test_error_in_a_disconnected_cycle_is_still_detected(self) -> None:
        # a's parent is b and b's parent is a - both present, so no
        # root-first walk (including find_deepest_error_chain from a real
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

    async def test_fallback_uses_the_requested_per_service_limit(self) -> None:
        """The server handler passes the operator-clamped candidate count;
        the helper must build its per-service TraceQuery from it. The public
        constant stays the default so the handler can clamp exactly it."""
        primary = _backend()
        primary.get_trace.return_value = _trace([_span("p1", service_name="svc-a")])
        secondary = _backend()
        secondary.get_trace.side_effect = ValueError("not found")
        secondary.search_traces.return_value = []

        await correlate_trace(primary, secondary, "t1", max_candidates_per_service=3)
        assert secondary.search_traces.call_args.args[0].limit == 3

        await correlate_trace(primary, secondary, "t1")
        assert secondary.search_traces.call_args.args[0].limit == 20
        assert correlate_module.MAX_CANDIDATES_PER_SERVICE_QUERY == 20

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

        result = await correlate_trace(primary, secondary, "t1")

        query: TraceQuery = secondary.search_traces.await_args.args[0]
        assert query.service_name == "svc-a"
        assert query.start_time == _BASE_TIME - timedelta(seconds=5)
        assert query.end_time == _BASE_TIME + timedelta(milliseconds=100.0) + timedelta(seconds=5)
        # An ordinary extent adds no per-call truncation note.
        assert result.limitations == _LIMITATIONS

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

        result = await correlate_trace(primary, secondary, "t1")

        assert secondary.search_traces.await_count == 2
        queried_services = {
            call.args[0].service_name for call in secondary.search_traces.await_args_list
        }
        assert queried_services == {"svc-a", "svc-b"}
        # Under the cap there is no truncation note.
        assert result.limitations == _LIMITATIONS


class TestSearchWindowCap:
    """The heuristic window is derived from telemetry-controlled
    TraceData.start_time/duration_ms, so a telemetry author who places two
    spans years apart in one trace would otherwise choose the time range of
    every secondary search issued under the operator's secondary credential."""

    async def test_heuristic_window_is_capped_for_wide_primary_extents(self) -> None:
        primary = _backend()
        primary.get_trace.return_value = _trace(
            [_span("p1", service_name="svc-a")],
            duration_ms=timedelta(days=365) / timedelta(milliseconds=1),
        )
        secondary = _backend()
        secondary.get_trace.side_effect = ValueError("not found")
        secondary.search_traces.return_value = []

        result = await correlate_trace(primary, secondary, "t1")

        query: TraceQuery = secondary.search_traces.await_args.args[0]
        assert query.start_time is not None and query.end_time is not None
        assert query.start_time == _BASE_TIME - _SEARCH_WINDOW_PAD
        assert query.end_time - query.start_time == _MAX_SEARCH_WINDOW + 2 * _SEARCH_WINDOW_PAD
        assert result.limitations[: len(_LIMITATIONS)] == _LIMITATIONS
        cap_notes = [note for note in result.limitations[len(_LIMITATIONS) :] if "capped" in note]
        assert len(cap_notes) == 1
        assert len(result.limitations) == len(_LIMITATIONS) + 1
        # The module-level fixed list itself is never mutated by a call.
        assert len(_LIMITATIONS) == 4
        assert all("capped" not in note for note in _LIMITATIONS)

    async def test_capped_window_is_identical_for_every_service_query(self) -> None:
        primary = _backend()
        primary.get_trace.return_value = _trace(
            [
                _span("p1", service_name="svc-a"),
                _span("p2", service_name="svc-b"),
                _span("p3", service_name="svc-c"),
            ],
            duration_ms=timedelta(days=24 * 365) / timedelta(milliseconds=1),
        )
        secondary = _backend()
        secondary.get_trace.side_effect = ValueError("not found")
        secondary.search_traces.return_value = []

        await correlate_trace(primary, secondary, "t1")

        assert secondary.search_traces.await_count == 3
        windows = {
            (call.args[0].start_time, call.args[0].end_time)
            for call in secondary.search_traces.await_args_list
        }
        assert windows == {
            (_BASE_TIME - _SEARCH_WINDOW_PAD, _BASE_TIME + _MAX_SEARCH_WINDOW + _SEARCH_WINDOW_PAD)
        }

    async def test_negative_duration_is_treated_as_zero(self) -> None:
        # SpanData/TraceData.duration_ms has no ge=0 constraint - clock skew
        # between services can yield a negative extent, which must not
        # invert the window or raise.
        primary = _backend()
        primary.get_trace.return_value = _trace(
            [_span("p1", service_name="svc-a")], duration_ms=-500.0
        )
        secondary = _backend()
        secondary.get_trace.side_effect = ValueError("not found")
        secondary.search_traces.return_value = []

        result = await correlate_trace(primary, secondary, "t1")

        query: TraceQuery = secondary.search_traces.await_args.args[0]
        assert query.start_time is not None and query.end_time is not None
        assert query.end_time - query.start_time == 2 * _SEARCH_WINDOW_PAD
        assert result.limitations == _LIMITATIONS

    @pytest.mark.parametrize("duration_ms", [1e17, float("inf"), float("nan")])
    async def test_degenerate_durations_are_capped_not_raised(self, duration_ms: float) -> None:
        # duration_ms is an unconstrained float. Converting 1e17 ms or inf
        # to a timedelta raises OverflowError (nan a ValueError) before any
        # cap could apply, so the clamp compares in the millisecond domain
        # first; nan compares False against everything and is treated as
        # over the cap rather than as a zero-width window.
        primary = _backend()
        primary.get_trace.return_value = _trace(
            [_span("p1", service_name="svc-a")], duration_ms=duration_ms
        )
        secondary = _backend()
        secondary.get_trace.side_effect = ValueError("not found")
        secondary.search_traces.return_value = []

        result = await correlate_trace(primary, secondary, "t1")

        query: TraceQuery = secondary.search_traces.await_args.args[0]
        assert query.start_time is not None and query.end_time is not None
        assert query.start_time == _BASE_TIME - _SEARCH_WINDOW_PAD
        assert query.end_time - query.start_time == _MAX_SEARCH_WINDOW + 2 * _SEARCH_WINDOW_PAD
        assert result.limitations[: len(_LIMITATIONS)] == _LIMITATIONS
        assert len(result.limitations) == len(_LIMITATIONS) + 1
        assert "capped" in result.limitations[-1]


class TestServiceFanoutCap:
    """service.name is a resource attribute every telemetry producer sets
    freely and most backends' get_trace returns every span of a trace, so
    without a cap one correlate_trace call became one secondary search per
    distinct name under the operator's secondary credential."""

    @staticmethod
    def _queried_services(secondary: AsyncMock) -> list[str]:
        return [call.args[0].service_name for call in secondary.search_traces.await_args_list]

    async def test_secondary_search_count_is_capped_for_wide_primary_traces(self) -> None:
        primary = _backend()
        primary.get_trace.return_value = _trace(
            [_span(f"p{i}", service_name=f"svc-{i:04d}") for i in range(1000)]
        )
        secondary = _backend()
        secondary.get_trace.side_effect = ValueError("not found")
        secondary.search_traces.return_value = []

        result = await correlate_trace(primary, secondary, "t1")

        assert secondary.search_traces.await_count == _MAX_SERVICE_QUERIES == 10
        assert result.limitations[: len(_LIMITATIONS)] == _LIMITATIONS
        cap_notes = [note for note in result.limitations if "most-represented" in note]
        assert len(cap_notes) == 1
        assert "1000 services" in cap_notes[0]
        assert len(_LIMITATIONS) == 4  # the module-level fixed list is never mutated

    async def test_most_represented_services_are_queried_first(self) -> None:
        # svc-00 has 15 spans, svc-01 has 14, ... svc-14 has 1: the ten
        # queried names are the ten highest span counts, most-represented
        # first, so the services that dominate the primary trace are the
        # ones correlation is actually attempted for.
        spans = [
            _span(f"p{i}-{j}", service_name=f"svc-{i:02d}")
            for i in range(15)
            for j in range(15 - i)
        ]
        primary = _backend()
        primary.get_trace.return_value = _trace(spans)
        secondary = _backend()
        secondary.get_trace.side_effect = ValueError("not found")
        secondary.search_traces.return_value = []

        await correlate_trace(primary, secondary, "t1")

        assert self._queried_services(secondary) == [f"svc-{i:02d}" for i in range(10)]

    async def test_equal_span_counts_are_tie_broken_by_name(self) -> None:
        # Every service has exactly one span, so the cap falls back to the
        # name order - deterministic across processes (str hashing is
        # randomized per process, so set order alone would not be).
        names = [f"svc-{c}" for c in "mlkjihgfedcba"]  # 13 services, deliberately unsorted
        primary = _backend()
        primary.get_trace.return_value = _trace(
            [_span(f"p{i}", service_name=name) for i, name in enumerate(names)]
        )
        secondary = _backend()
        secondary.get_trace.side_effect = ValueError("not found")
        secondary.search_traces.return_value = []

        await correlate_trace(primary, secondary, "t1")

        assert self._queried_services(secondary) == sorted(names)[:10]

    async def test_fanout_cap_holds_when_every_secondary_query_fails(self) -> None:
        # Per-query exceptions are swallowed so a throttling secondary does
        # not abort the loop - which is exactly why the loop itself has to
        # be bounded: a 429 storm must not become 25 retried searches.
        primary = _backend()
        primary.get_trace.return_value = _trace(
            [_span(f"p{i}", service_name=f"svc-{i:02d}") for i in range(25)]
        )
        secondary = _backend()
        secondary.get_trace.side_effect = ValueError("not found")
        secondary.search_traces.side_effect = RuntimeError("429 Too Many Requests")

        result = await correlate_trace(primary, secondary, "t1")

        assert secondary.search_traces.await_count == 10
        assert result.matches == []
        assert any("most-represented" in note for note in result.limitations)

    async def test_empty_service_name_is_never_queried(self) -> None:
        # An empty/whitespace service_name would turn the per-service
        # search into an unscoped search of the whole window.
        primary = _backend()
        primary.get_trace.return_value = _trace(
            [
                _span("p1", service_name="svc-a"),
                _span("p2", service_name=""),
                _span("p3", service_name="   "),
            ]
        )
        secondary = _backend()
        secondary.get_trace.side_effect = ValueError("not found")
        secondary.search_traces.return_value = []

        result = await correlate_trace(primary, secondary, "t1")

        assert self._queried_services(secondary) == ["svc-a"]
        assert result.limitations == _LIMITATIONS


class TestErrorSignalBounds:
    def test_8000_span_broom_error_signal_is_linear_in_memory(self) -> None:
        # Security regression: _error_signal runs once for the primary trace
        # and once per heuristic candidate, so a "broom" (4000 chained ERROR
        # spans whose last span has 4000 ERROR children, all tied at the
        # maximal depth) that cost 127 MiB and 1.1 s per call while every
        # tied leaf's path was rebuilt multiplied across every candidate
        # secondary trace. One chain is materialized now; the blamed service
        # is still the deepest (leaf) span's. Same tracemalloc delta
        # protocol as test_triage.py's bounds tests.
        handle = 4000
        leaves = 4000
        spans = [_span("s0", service_name="svc-handle", status="ERROR")]
        for i in range(1, handle):
            spans.append(
                _span(
                    f"s{i}", parent_span_id=f"s{i - 1}", service_name="svc-handle", status="ERROR"
                )
            )
        for j in range(leaves):
            spans.append(
                _span(
                    f"leaf{j}",
                    parent_span_id=f"s{handle - 1}",
                    service_name="svc-leaf",
                    status="ERROR",
                )
            )
        trace = _trace(spans)

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            before, _ = tracemalloc.get_traced_memory()
            has_error, blamed_service = _error_signal(trace)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert has_error is True
        assert blamed_service == "svc-leaf"
        assert peak - before < 50 * 1024 * 1024

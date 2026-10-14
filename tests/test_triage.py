"""Tests for the triage_trace tool implementation (tools/triage.py).

Mocks at the `backend: AsyncMock(spec=BaseBackend)` level per
test_investigate.py's convention, driving real TraceData/SpanData fixtures
through the real span_tree.py algorithms - only backend.get_trace itself is
faked, everything downstream is exercised for real.
"""

import tracemalloc
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from unittest.mock import AsyncMock

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.models import SpanData, TraceData
from opentelemetry_mcp.tools.triage import (
    _CHAIN_HEAD,
    _ELIDED_SPAN_ID,
    _FULL_CHAIN_TAIL,
    _SUMMARY_CHAIN_TAIL,
    _no_root_reasoning,
    triage_trace,
)

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


class TestAlgorithmicBounds:
    async def test_8000_span_all_error_chain_triage_is_linear(self) -> None:
        # Security regression: one telemetry-authored trace whose spans form
        # a deep parent chain with every span in ERROR used to cost
        # N(N+1)/2 retained path pointers (about 270 MB at this size) and
        # quadratic CPU, synchronously on the shared event loop, before the
        # verdict kept only the deepest chain. The deepest-only walk must
        # give the same verdict while allocating linearly. tracemalloc
        # starts only after the fixture is built, so the assertion is on
        # the peak delta of triage_trace itself, not the 8000 pydantic spans.
        depth = 8000
        spans = [_span("s0", status="ERROR")]
        for i in range(1, depth):
            spans.append(_span(f"s{i}", parent_span_id=f"s{i - 1}", status="ERROR"))
        backend = _backend()
        backend.get_trace.return_value = _trace(spans)

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            before, _ = tracemalloc.get_traced_memory()
            result = await triage_trace(backend, "t1")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert result.verdict.confidence == "high"
        assert result.verdict.likely_root_cause is not None
        assert result.verdict.likely_root_cause.span_id == f"s{depth - 1}"
        assert result.error_chain is not None
        # The chain is reported elided (see TestResultBounds), never whole.
        assert len(result.error_chain) == _CHAIN_HEAD + 1 + _SUMMARY_CHAIN_TAIL
        assert result.error_chain[0].span_id == "s0"
        assert result.error_chain[-1].span_id == f"s{depth - 1}"
        assert peak - before < 50 * 1024 * 1024

    async def test_8000_span_broom_of_equally_deep_errors_triages_in_linear_memory(self) -> None:
        # Security regression: a "broom" - 4000 chained ERROR spans whose
        # last span has 4000 ERROR children - ties every leaf at the maximal
        # depth. Rebuilding a root-to-leaf path per tied leaf cost 135 MiB
        # and 1.8 s on the event loop at this size, even though the verdict
        # only ever used the first chain and the number of ties; now exactly
        # one chain is materialized and the ties are counted. Same
        # tracemalloc delta protocol as the chain test above.
        handle = 4000
        leaves = 4000
        spans = [_span("s0", status="ERROR")]
        for i in range(1, handle):
            spans.append(_span(f"s{i}", parent_span_id=f"s{i - 1}", status="ERROR"))
        for j in range(leaves):
            spans.append(_span(f"leaf{j}", parent_span_id=f"s{handle - 1}", status="ERROR"))
        backend = _backend()
        backend.get_trace.return_value = _trace(spans)

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            before, _ = tracemalloc.get_traced_memory()
            result = await triage_trace(backend, "t1")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert result.verdict.confidence == "medium"
        assert f"{leaves} equally-deep error chains" in result.verdict.reasoning
        assert result.verdict.likely_root_cause is not None
        assert result.verdict.likely_root_cause.span_id == f"leaf{leaves - 1}"
        assert result.error_chain is not None
        assert len(result.error_chain) == _CHAIN_HEAD + 1 + _SUMMARY_CHAIN_TAIL
        assert result.error_chain[0].span_id == "s0"
        assert result.error_chain[-2].span_id == f"s{handle - 1}"
        assert peak - before < 50 * 1024 * 1024


def _chain_trace(depth: int) -> TraceData:
    spans = [_span("s0", status="ERROR")]
    for i in range(1, depth):
        spans.append(_span(f"s{i}", parent_span_id=f"s{i - 1}", status="ERROR"))
    return _trace(spans)


def _ring_trace(size: int, error_index: int | None) -> TraceData:
    """`size` spans in one parent cycle: r{i}'s parent is r{i-1} and r0's
    parent is r{size-1}, so no span is a root and every parent is present
    and non-self (the shape build_children_map's orphan redirect cannot
    catch). `error_index` marks one span ERROR, or none."""
    spans = []
    for i in range(size):
        spans.append(
            _span(
                f"r{i}",
                parent_span_id=f"r{(i - 1) % size}",
                duration_ms=10.0 if i == 1 else 1.0,
                status="ERROR" if i == error_index else "OK",
            )
        )
    return _trace(spans)


class TestResultBounds:
    """A tool result is handed to the MCP client whole: the runtime probe
    (probe-3 F3 / T6) made triage_trace return 6.9 MB for a 20,000-span
    chain because error_chain and critical_path carried every span of a
    chain whose depth the telemetry author controls, and the mcp SDK client
    fails on results of ~2.9 MB and above. Both lists are now elided to a
    fixed head and tail with one explicit elision entry naming the gap; the
    verdict (first and last spans, root cause, confidence) is unchanged."""

    async def test_summary_elides_a_deep_error_chain_to_head_tail_and_one_marker(self) -> None:
        depth = 8000
        backend = _backend()
        backend.get_trace.return_value = _chain_trace(depth)

        result = await triage_trace(backend, "t1")

        assert result.error_chain is not None
        assert len(result.error_chain) == _CHAIN_HEAD + 1 + _SUMMARY_CHAIN_TAIL == 26
        assert [s.span_id for s in result.error_chain[:_CHAIN_HEAD]] == [f"s{i}" for i in range(5)]
        marker = result.error_chain[_CHAIN_HEAD]
        assert marker.span_id == _ELIDED_SPAN_ID
        assert marker.operation_name == f"[{depth - 25} intermediate span(s) elided]"
        assert marker.status == "UNSET"
        assert marker.error_detail is None
        assert [s.span_id for s in result.error_chain[-_SUMMARY_CHAIN_TAIL:]] == [
            f"s{i}" for i in range(depth - 20, depth)
        ]
        assert result.verdict.likely_root_cause is not None
        assert result.verdict.likely_root_cause.span_id == result.error_chain[-1].span_id
        assert result.verdict.confidence == "high"
        assert f"{depth} spans" in result.verdict.reasoning
        assert "first 5 and last 20" in result.verdict.reasoning
        # The critical path of a chain IS the chain: elided the same way.
        assert len(result.critical_path) == 26
        assert result.critical_path[_CHAIN_HEAD].span_id == _ELIDED_SPAN_ID
        assert result.critical_path[-1].span_id == f"s{depth - 1}"
        assert len(result.model_dump_json()) < 32 * 1024

    async def test_full_detail_caps_the_chain_at_one_thousand_real_spans(self) -> None:
        depth = 8000
        backend = _backend()
        backend.get_trace.return_value = _chain_trace(depth)

        result = await triage_trace(backend, "t1", detail_level="full")

        assert result.error_chain is not None
        assert _CHAIN_HEAD + _FULL_CHAIN_TAIL == 1000
        assert len(result.error_chain) == 1001
        assert result.error_chain[_CHAIN_HEAD].span_id == _ELIDED_SPAN_ID
        assert result.error_chain[-1].span_id == f"s{depth - 1}"
        assert result.error_chain[-1].error_detail is not None  # still the diagnosed span
        assert len(result.critical_path) == 1001
        assert "first 5 and last 995" in result.verdict.reasoning
        assert len(result.model_dump_json()) < 512 * 1024

    async def test_chains_that_fit_are_reported_whole(self) -> None:
        depth = _CHAIN_HEAD + _SUMMARY_CHAIN_TAIL + 1  # eliding one span would gain nothing
        backend = _backend()
        backend.get_trace.return_value = _chain_trace(depth)

        result = await triage_trace(backend, "t1")

        assert result.error_chain is not None
        assert len(result.error_chain) == depth
        assert all(s.span_id != _ELIDED_SPAN_ID for s in result.error_chain)
        assert "elided" not in result.verdict.reasoning

    async def test_rootless_parent_ring_is_low_confidence_and_names_the_cycle(self) -> None:
        """probe-3 F3: a 5000-span ring with no root was reported with
        confidence "high" and "5000 span(s) deep from the root" because an
        arbitrary span stood in for the missing root. No span is a root, so
        no depth exists; the verdict must say so and name the cycle."""
        size = 5000
        backend = _backend()
        backend.get_trace.return_value = _ring_trace(size, error_index=1)

        result = await triage_trace(backend, "t1")

        assert result.verdict.confidence == "low"
        assert result.verdict.likely_root_cause is not None
        assert result.verdict.likely_root_cause.span_id == "r1"
        reasoning = result.verdict.reasoning
        assert "no root span" in reasoning.lower()
        assert f"parent cycle of {size} span(s)" in reasoning
        assert "deep from the root" not in reasoning
        assert result.error_chain is not None
        assert [s.span_id for s in result.error_chain] == ["r1"]
        # The critical path walked around the ring is elided like a chain.
        assert len(result.critical_path) == 26
        assert len(result.model_dump_json()) < 32 * 1024

    async def test_rootless_ring_full_detail_attaches_error_detail_to_the_trapped_span(
        self,
    ) -> None:
        backend = _backend()
        backend.get_trace.return_value = _ring_trace(8, error_index=3)

        result = await triage_trace(backend, "t1", detail_level="full")

        assert result.verdict.confidence == "low"
        assert result.error_chain is not None
        assert [s.span_id for s in result.error_chain] == ["r3"]
        assert result.error_chain[0].error_detail is not None
        assert result.verdict.likely_root_cause is result.error_chain[0]
        assert "parent cycle of 8 span(s)" in result.verdict.reasoning

    async def test_rootless_ring_without_errors_falls_back_to_latency_and_says_no_root(
        self,
    ) -> None:
        backend = _backend()
        backend.get_trace.return_value = _ring_trace(50, error_index=None)

        result = await triage_trace(backend, "t1")

        assert result.verdict.confidence == "low"
        assert result.verdict.likely_root_cause is not None
        assert result.verdict.likely_root_cause.span_id == "r1"  # the 10 ms span
        assert "no root span" in result.verdict.reasoning.lower()
        assert "parent cycle of 50 span(s)" in result.verdict.reasoning
        assert result.error_chain is None

    async def test_no_root_reasoning_reads_as_one_sentence_sequence(self) -> None:
        backend = _backend()
        backend.get_trace.return_value = _ring_trace(50, error_index=None)

        result = await triage_trace(backend, "t1")

        assert "  " not in result.verdict.reasoning
        assert "instrumentation. No error span found" in result.verdict.reasoning

    def test_no_root_reasoning_without_a_cycle_still_explains_the_missing_root(self) -> None:
        """Defensive branch: parent_cycle_span_ids finds no cycle (only
        reachable with duplicate span_ids resolving to a different copy);
        the text must still say why confidence is low."""
        orphan = _span("orphan", parent_span_id="gone")

        text = _no_root_reasoning(orphan, [orphan])

        assert text.startswith("Trace has no root span")
        assert "never reaches a root" in text
        assert not text.endswith(" ")

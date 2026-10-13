"""Tests for the span-tree/critical-path utilities in tools/span_tree.py."""

import tracemalloc
from datetime import datetime, timedelta
from typing import Literal

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.models import SpanData
from opentelemetry_mcp.tools import span_tree

_BASE_TIME = datetime(2024, 1, 1, 0, 0, 0)


def _span(
    span_id: str,
    *,
    parent_span_id: str | None = None,
    start_offset_ms: float = 0.0,
    duration_ms: float = 100.0,
    status: Literal["OK", "ERROR", "UNSET"] = "OK",
    service_name: str = "svc",
    operation_name: str = "op",
) -> SpanData:
    """Build a SpanData instance for span-tree tests, defaulting to no
    gen_ai attributes - only span_id/parent_span_id/timing/status matter
    for these pure tree-walking functions."""
    return SpanData(
        trace_id="t1",
        span_id=span_id,
        parent_span_id=parent_span_id,
        operation_name=operation_name,
        service_name=service_name,
        start_time=_BASE_TIME + timedelta(milliseconds=start_offset_ms),
        duration_ms=duration_ms,
        status=status,
        attributes=SpanAttributes.model_validate({}),
    )


class TestBuildChildrenMap:
    def test_groups_by_parent_span_id(self) -> None:
        root = _span("root")
        child = _span("child", parent_span_id="root")

        children_map = span_tree.build_children_map([root, child])

        assert children_map[None] == [root]
        assert children_map["root"] == [child]

    def test_orphan_span_with_missing_parent_is_bucketed_as_a_root(self) -> None:
        root = _span("root")
        orphan = _span("orphan", parent_span_id="missing-parent")

        children_map = span_tree.build_children_map([root, orphan])

        assert children_map[None] == [root, orphan]
        assert "missing-parent" not in children_map

    def test_span_that_is_its_own_parent_is_bucketed_as_a_root(self) -> None:
        root = _span("root")
        self_parented = _span("weird", parent_span_id="weird")

        children_map = span_tree.build_children_map([root, self_parented])

        assert children_map[None] == [root, self_parented]
        assert "weird" not in children_map


class TestFindRoots:
    def test_returns_the_none_bucket(self) -> None:
        root = _span("root")
        child = _span("child", parent_span_id="root")
        children_map = span_tree.build_children_map([root, child])

        assert span_tree.find_roots(children_map) == [root]

    def test_empty_when_no_root_or_orphan_spans_exist(self) -> None:
        assert span_tree.find_roots({}) == []


class TestSubtreeSpanIds:
    def test_includes_root_and_all_descendants(self) -> None:
        root = _span("root")
        child = _span("child", parent_span_id="root")
        grandchild = _span("grandchild", parent_span_id="child")
        children_map = span_tree.build_children_map([root, child, grandchild])

        assert span_tree.subtree_span_ids(root, children_map) == {"root", "child", "grandchild"}

    def test_terminates_on_a_cycle_among_two_spans(self) -> None:
        # a's parent is b and b's parent is a - a genuine cycle among two
        # otherwise-valid span IDs (build_children_map only redirects a
        # *direct* self-reference to None, not a longer cycle like this).
        a = _span("a", parent_span_id="b")
        b = _span("b", parent_span_id="a")
        children_map = span_tree.build_children_map([a, b])

        assert span_tree.subtree_span_ids(a, children_map) == {"a", "b"}


class TestFindOwningRoot:
    def test_returns_the_root_whose_subtree_contains_the_span(self) -> None:
        root1 = _span("root1")
        child1 = _span("child1", parent_span_id="root1")
        root2 = _span("root2")
        children_map = span_tree.build_children_map([root1, child1, root2])

        assert span_tree.find_owning_root(child1, [root1, root2], children_map) is root1
        assert span_tree.find_owning_root(root2, [root1, root2], children_map) is root2

    def test_falls_back_to_the_span_itself_when_no_root_reaches_it(self) -> None:
        # a and b form their own cycle with no orphan/root entry point at
        # all (both have a valid, non-self, non-missing parent) - an
        # "island" unreachable from the trace's actual root.
        root = _span("root")
        a = _span("a", parent_span_id="b")
        b = _span("b", parent_span_id="a")
        children_map = span_tree.build_children_map([root, a, b])

        assert span_tree.find_owning_root(a, [root], children_map) is a


class TestComputeCriticalPath:
    def test_picks_last_finishing_child_at_each_level(self) -> None:
        root = _span("root", duration_ms=100.0)
        # child_a ends at 0+40=40ms; child_b ends at 10+80=90ms - b finishes later.
        child_a = _span("child_a", parent_span_id="root", start_offset_ms=0.0, duration_ms=40.0)
        child_b = _span("child_b", parent_span_id="root", start_offset_ms=10.0, duration_ms=80.0)
        grandchild = _span(
            "grandchild", parent_span_id="child_b", start_offset_ms=20.0, duration_ms=50.0
        )
        spans = [root, child_a, child_b, grandchild]
        children_map = span_tree.build_children_map(spans)

        path = span_tree.compute_critical_path(root, children_map)

        assert [s.span_id for s in path] == ["root", "child_b", "grandchild"]

    def test_leaf_root_returns_single_element_path(self) -> None:
        root = _span("root")
        children_map = span_tree.build_children_map([root])

        assert span_tree.compute_critical_path(root, children_map) == [root]

    def test_cycle_among_valid_span_ids_terminates_instead_of_looping_forever(self) -> None:
        # a's parent is b and b's parent is a - build_children_map can't
        # redirect this to a root (neither is a direct self-reference or a
        # missing parent), so the walk itself must be the thing that stops.
        a = _span("a", parent_span_id="b")
        b = _span("b", parent_span_id="a")
        children_map = span_tree.build_children_map([a, b])

        path = span_tree.compute_critical_path(a, children_map)

        assert [s.span_id for s in path] == ["a", "b"]

    def test_span_that_is_its_own_parent_does_not_loop_forever(self) -> None:
        weird = _span("weird", parent_span_id="weird")
        children_map = span_tree.build_children_map([weird])

        assert span_tree.compute_critical_path(weird, children_map) == [weird]


class TestComputeSelfTimeMs:
    def test_leaf_span_self_time_equals_duration(self) -> None:
        span = _span("leaf", duration_ms=50.0)
        children_map = span_tree.build_children_map([span])

        assert span_tree.compute_self_time_ms(span, children_map) == 50.0

    def test_non_overlapping_children_subtract_their_combined_duration(self) -> None:
        root = _span("root", duration_ms=100.0)
        child_a = _span("a", parent_span_id="root", start_offset_ms=0.0, duration_ms=20.0)
        child_b = _span("b", parent_span_id="root", start_offset_ms=50.0, duration_ms=20.0)
        children_map = span_tree.build_children_map([root, child_a, child_b])

        assert span_tree.compute_self_time_ms(root, children_map) == 60.0

    def test_overlapping_children_are_unioned_not_double_subtracted(self) -> None:
        root = _span("root", duration_ms=100.0)
        child_a = _span("a", parent_span_id="root", start_offset_ms=0.0, duration_ms=50.0)
        child_b = _span("b", parent_span_id="root", start_offset_ms=20.0, duration_ms=50.0)
        children_map = span_tree.build_children_map([root, child_a, child_b])

        # Union of [0,50] and [20,70] is [0,70] = 70ms covered, not 100ms.
        assert span_tree.compute_self_time_ms(root, children_map) == 30.0

    def test_child_interval_is_clipped_to_parent_interval(self) -> None:
        root = _span("root", duration_ms=50.0)
        # Child appears to start before the parent and end after it (clock skew).
        child = _span("child", parent_span_id="root", start_offset_ms=-10.0, duration_ms=100.0)
        children_map = span_tree.build_children_map([root, child])

        assert span_tree.compute_self_time_ms(root, children_map) == 0.0

    def test_child_entirely_outside_parent_interval_never_inflates_self_time_above_duration(
        self,
    ) -> None:
        # A 0ms-duration parent with a child whose clock skew places it
        # entirely after the parent's own interval ends - clipping produces
        # an inverted (end < start) interval, which must contribute zero
        # covered time, not a negative width that would inflate self-time
        # past duration_ms.
        root = _span("root", duration_ms=0.0)
        child = _span("child", parent_span_id="root", start_offset_ms=5.0, duration_ms=10.0)
        children_map = span_tree.build_children_map([root, child])

        assert span_tree.compute_self_time_ms(root, children_map) == 0.0

    def test_negative_duration_leaf_span_clamps_to_zero_not_negative(self) -> None:
        # SpanData.duration_ms has no ge=0 constraint - a backend affected
        # by clock skew can legitimately supply a negative value. The
        # no-children early return must clamp it the same way the
        # merged-interval path already does, not return it raw.
        span = _span("leaf", duration_ms=-50.0)
        children_map = span_tree.build_children_map([span])

        assert span_tree.compute_self_time_ms(span, children_map) == 0.0

    def test_negative_duration_span_with_children_but_no_valid_intervals_clamps_to_zero(
        self,
    ) -> None:
        # Same clamp, but via the "no valid clipped intervals" early return
        # (all children clip to inverted/empty intervals) rather than the
        # no-children path.
        root = _span("root", duration_ms=-50.0)
        child = _span("child", parent_span_id="root", start_offset_ms=100.0, duration_ms=10.0)
        children_map = span_tree.build_children_map([root, child])

        assert span_tree.compute_self_time_ms(root, children_map) == 0.0


class TestRankByLatencyContribution:
    def test_sorted_descending_by_self_time(self) -> None:
        root = _span("root", duration_ms=100.0)
        low = _span("low", parent_span_id="root", start_offset_ms=0.0, duration_ms=5.0)
        high = _span("high", parent_span_id="root", start_offset_ms=10.0, duration_ms=80.0)
        spans = [root, low, high]
        children_map = span_tree.build_children_map(spans)

        ranked = span_tree.rank_by_latency_contribution(spans, children_map)

        assert [span.span_id for span, _ in ranked] == ["high", "root", "low"]

    def test_truncates_to_top_n(self) -> None:
        spans = [_span(f"s{i}", duration_ms=float(i)) for i in range(5)]
        children_map = span_tree.build_children_map(spans)

        ranked = span_tree.rank_by_latency_contribution(spans, children_map, top_n=2)

        assert len(ranked) == 2
        assert [span.span_id for span, _ in ranked] == ["s4", "s3"]


class TestFindDeepestErrorChain:
    def test_no_error_spans_returns_zero_counts_and_no_chain(self) -> None:
        root = _span("root")
        child = _span("child", parent_span_id="root")
        children_map = span_tree.build_children_map([root, child])

        assert span_tree.find_deepest_error_chain(root, children_map) == (0, 0, [])

    def test_root_itself_erroring_is_its_own_chain(self) -> None:
        root = _span("root", status="ERROR")
        children_map = span_tree.build_children_map([root])

        result = span_tree.find_deepest_error_chain(root, children_map)

        assert result.error_count == 1
        assert result.tie_count == 1
        assert [s.span_id for s in result.chain] == ["root"]

    def test_deep_error_span_produces_full_root_to_leaf_path_regardless_of_intermediate_status(
        self,
    ) -> None:
        root = _span("root", status="OK")
        mid = _span("mid", parent_span_id="root", status="OK")
        leaf = _span("leaf", parent_span_id="mid", status="ERROR")
        children_map = span_tree.build_children_map([root, mid, leaf])

        result = span_tree.find_deepest_error_chain(root, children_map)

        assert result.error_count == 1
        assert result.tie_count == 1
        assert [s.span_id for s in result.chain] == ["root", "mid", "leaf"]

    def test_two_branch_tree_reports_only_the_deepest_chain(self) -> None:
        # Both error spans are counted, but only the deepest root-to-error
        # path is materialized - the shallower ("root", "error_a") chain is
        # never built, which is what keeps the walk linear in the span count.
        root = _span("root", status="OK")
        error_a = _span("error_a", parent_span_id="root", status="ERROR")
        mid = _span("mid", parent_span_id="root", status="OK")
        error_b = _span("error_b", parent_span_id="mid", status="ERROR")
        children_map = span_tree.build_children_map([root, error_a, mid, error_b])

        result = span_tree.find_deepest_error_chain(root, children_map)

        assert result.error_count == 2
        assert result.tie_count == 1
        assert [s.span_id for s in result.chain] == ["root", "mid", "error_b"]

    def test_equal_depth_error_leaves_are_counted_and_the_first_popped_is_the_chain(self) -> None:
        # Two error leaves at the same depth: both are counted as a tie, and
        # the chain is the one the explicit-stack DFS pops first (children
        # are pushed in children_map order, so the LAST child is popped
        # first) - the same chain the original all-chains walk listed first,
        # which triage_trace's "medium confidence, picked the first one" tie
        # rule depends on. The other leaf's path is never built.
        root = _span("root", status="OK")
        error_a = _span("error_a", parent_span_id="root", status="ERROR")
        error_b = _span("error_b", parent_span_id="root", status="ERROR")
        children_map = span_tree.build_children_map([root, error_a, error_b])

        result = span_tree.find_deepest_error_chain(root, children_map)

        assert result.error_count == 2
        assert result.tie_count == 2
        assert [s.span_id for s in result.chain] == ["root", "error_b"]

    def test_long_linear_chain_does_not_raise_recursion_error(self) -> None:
        # A deep recursive agent/tool-calling loop is a realistic shape for
        # this server's workloads - a naive recursive DFS would blow
        # Python's default recursion limit on a chain this long.
        depth = 2000
        spans = [_span("s0", status="OK")]
        for i in range(1, depth):
            status: Literal["OK", "ERROR"] = "ERROR" if i == depth - 1 else "OK"
            spans.append(_span(f"s{i}", parent_span_id=f"s{i - 1}", status=status))
        children_map = span_tree.build_children_map(spans)

        result = span_tree.find_deepest_error_chain(spans[0], children_map)

        assert result.error_count == 1
        assert result.tie_count == 1
        assert len(result.chain) == depth
        assert result.chain[-1].span_id == f"s{depth - 1}"

    def test_cycle_among_valid_span_ids_terminates_instead_of_looping_forever(self) -> None:
        a = _span("a", parent_span_id="b", status="ERROR")
        b = _span("b", parent_span_id="a", status="OK")
        children_map = span_tree.build_children_map([a, b])

        result = span_tree.find_deepest_error_chain(a, children_map)

        assert result.error_count == 1
        assert result.tie_count == 1
        assert [s.span_id for s in result.chain] == ["a"]

    def test_duplicate_span_id_copies_are_both_walked_and_the_deeper_one_is_blamed(self) -> None:
        # Malformed trace: two different spans share span_id "x", one a
        # direct child of root (depth 2) and one under mid (depth 3).
        # build_children_map files both copies under their own parents and
        # the walk's cycle guard is keyed by object identity, so both are
        # counted and the deeper copy's chain is reported - as the original
        # per-path walk did. A span_id-keyed visited set would skip whichever
        # copy is popped second (here the deeper one) and blame svc-shallow.
        root = _span("root", status="OK")
        mid = _span("mid", parent_span_id="root", status="OK")
        x_shallow = _span("x", parent_span_id="root", status="ERROR", service_name="svc-shallow")
        x_deep = _span("x", parent_span_id="mid", status="ERROR", service_name="svc-deep")
        children_map = span_tree.build_children_map([root, mid, x_shallow, x_deep])

        result = span_tree.find_deepest_error_chain(root, children_map)

        assert result.error_count == 2
        assert result.tie_count == 1
        assert [s.span_id for s in result.chain] == ["root", "mid", "x"]
        assert result.chain[-1] is x_deep
        assert result.chain[-1].service_name == "svc-deep"

    def test_deep_all_error_chain_is_linear_in_time_and_memory(self) -> None:
        # Security regression: a telemetry author can plant one trace whose
        # N spans form a deep parent chain with every span in ERROR. The
        # original walk copied the whole root-to-span path on every edge and
        # retained one full path per ERROR span - N(N+1)/2 pointers, about
        # 1.6 GB for this fixture - synchronously on the shared event loop.
        # The deepest-only walk keeps a predecessor map instead, so the
        # memory it allocates during the walk stays linear in N. Measured as
        # the tracemalloc peak DELTA of the walk alone: tracing starts only
        # after the fixture is built, so the 20k pydantic spans themselves
        # are not what is being asserted on.
        depth = 20_000
        spans = [_span("s0", status="ERROR")]
        for i in range(1, depth):
            spans.append(_span(f"s{i}", parent_span_id=f"s{i - 1}", status="ERROR"))
        children_map = span_tree.build_children_map(spans)

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            before, _ = tracemalloc.get_traced_memory()
            result = span_tree.find_deepest_error_chain(spans[0], children_map)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert result.error_count == depth
        assert result.tie_count == 1
        assert len(result.chain) == depth
        assert result.chain[0].span_id == "s0"
        assert result.chain[-1].span_id == f"s{depth - 1}"
        assert peak - before < 50 * 1024 * 1024

    def test_broom_of_equally_deep_errors_materializes_one_chain_in_linear_memory(self) -> None:
        # Security regression: a "broom" - a handle of N/2 chained ERROR
        # spans whose last span has N/2 ERROR children - puts every leaf at
        # the same maximal depth. A deepest-only walk that still rebuilt a
        # root-to-leaf path for every tied leaf retained (N/2)^2 pointers
        # (813.9 MiB and 7 s at this size), the same order as the all-chains
        # walk it replaced, even though callers only use the first chain and
        # the number of ties. One chain and one counter must come back,
        # within a linear memory budget - same tracemalloc delta protocol as
        # the chain test above.
        handle = 10_000
        leaves = 10_000
        spans = [_span("s0", status="ERROR")]
        for i in range(1, handle):
            spans.append(_span(f"s{i}", parent_span_id=f"s{i - 1}", status="ERROR"))
        for j in range(leaves):
            spans.append(_span(f"leaf{j}", parent_span_id=f"s{handle - 1}", status="ERROR"))
        children_map = span_tree.build_children_map(spans)

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            before, _ = tracemalloc.get_traced_memory()
            result = span_tree.find_deepest_error_chain(spans[0], children_map)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert result.error_count == handle + leaves
        assert result.tie_count == leaves
        assert len(result.chain) == handle + 1
        assert result.chain[0].span_id == "s0"
        assert result.chain[-2].span_id == f"s{handle - 1}"
        assert result.chain[-1].span_id == f"leaf{leaves - 1}"  # last pushed, first popped
        assert peak - before < 50 * 1024 * 1024


class TestFindDeepestErrorChainAcrossRoots:
    def test_no_roots_or_no_errors_returns_zero_counts_and_no_chain(self) -> None:
        root = _span("root")
        children_map = span_tree.build_children_map([root])

        assert span_tree.find_deepest_error_chain_across_roots([], children_map) == (0, 0, [])
        assert span_tree.find_deepest_error_chain_across_roots([root], children_map) == (0, 0, [])

    def test_deeper_chain_under_a_later_root_wins(self) -> None:
        # The GLOBAL maximum across roots, not the first root's own maximum:
        # the shallow error (depth 2) hangs off root1, the deeper one
        # (depth 3) off root2, and root2's chain must be the one reported.
        root1 = _span("root1", status="OK")
        shallow_error = _span("shallow_error", parent_span_id="root1", status="ERROR")
        root2 = _span("root2", status="OK")
        mid = _span("mid", parent_span_id="root2", status="OK")
        deep_error = _span("deep_error", parent_span_id="mid", status="ERROR")
        children_map = span_tree.build_children_map([root1, shallow_error, root2, mid, deep_error])
        roots = span_tree.find_roots(children_map)

        result = span_tree.find_deepest_error_chain_across_roots(roots, children_map)

        assert [r.span_id for r in roots] == ["root1", "root2"]
        assert result.error_count == 2
        assert result.tie_count == 1
        assert [s.span_id for s in result.chain] == ["root2", "mid", "deep_error"]

    def test_equal_depths_across_roots_are_a_tie_and_the_first_root_is_reported(self) -> None:
        root1 = _span("root1", status="OK")
        error1 = _span("error1", parent_span_id="root1", status="ERROR")
        root2 = _span("root2", status="OK")
        error2 = _span("error2", parent_span_id="root2", status="ERROR")
        children_map = span_tree.build_children_map([root1, error1, root2, error2])
        roots = span_tree.find_roots(children_map)

        result = span_tree.find_deepest_error_chain_across_roots(roots, children_map)

        assert result.error_count == 2
        assert result.tie_count == 2
        assert [s.span_id for s in result.chain] == ["root1", "error1"]

    def test_tie_counts_sum_within_and_across_roots_at_the_maximal_depth_only(self) -> None:
        # root1 has two equally-deep errors (depth 2); root2 errors itself
        # (depth 1) and has one error at depth 2. Ties at the maximal depth
        # sum to 3 across both roots; root2's shallower own error is counted
        # but is not a tie, and root1's first-popped chain is the one kept.
        root1 = _span("root1", status="OK")
        a = _span("a", parent_span_id="root1", status="ERROR")
        b = _span("b", parent_span_id="root1", status="ERROR")
        root2 = _span("root2", status="ERROR")
        c = _span("c", parent_span_id="root2", status="ERROR")
        children_map = span_tree.build_children_map([root1, a, b, root2, c])
        roots = span_tree.find_roots(children_map)

        result = span_tree.find_deepest_error_chain_across_roots(roots, children_map)

        assert result.error_count == 4
        assert result.tie_count == 3
        assert [s.span_id for s in result.chain] == ["root1", "b"]

    def test_root_without_errors_is_skipped_but_counted_as_zero(self) -> None:
        root1 = _span("root1", status="OK")
        child1 = _span("child1", parent_span_id="root1", status="OK")
        root2 = _span("root2", status="OK")
        error2 = _span("error2", parent_span_id="root2", status="ERROR")
        children_map = span_tree.build_children_map([root1, child1, root2, error2])
        roots = span_tree.find_roots(children_map)

        result = span_tree.find_deepest_error_chain_across_roots(roots, children_map)

        assert result.error_count == 1
        assert result.tie_count == 1
        assert [s.span_id for s in result.chain] == ["root2", "error2"]


class TestFindUnreachableErrorSpans:
    def test_no_unreachable_spans_returns_empty_list(self) -> None:
        root = _span("root", status="OK")
        child = _span("child", parent_span_id="root", status="ERROR")
        children_map = span_tree.build_children_map([root, child])
        roots = span_tree.find_roots(children_map)

        assert span_tree.find_unreachable_error_spans([root, child], roots, children_map) == []

    def test_error_in_a_disconnected_cycle_is_found(self) -> None:
        # A genuine root reachable normally, plus a separate 2-cycle (a's
        # parent is b, b's parent is a - both present, neither self- nor
        # missing-referential, so build_children_map's orphan redirect
        # never fires for either) that no root-first walk can discover.
        root = _span("root", status="OK")
        a = _span("a", parent_span_id="b", status="ERROR")
        b = _span("b", parent_span_id="a", status="OK")
        spans = [root, a, b]
        children_map = span_tree.build_children_map(spans)
        roots = span_tree.find_roots(children_map)

        assert [r.span_id for r in roots] == ["root"]  # confirms a/b are genuinely unreachable
        unreachable = span_tree.find_unreachable_error_spans(spans, roots, children_map)
        assert [s.span_id for s in unreachable] == ["a"]

    def test_non_error_spans_in_a_disconnected_cycle_are_not_reported(self) -> None:
        root = _span("root", status="OK")
        a = _span("a", parent_span_id="b", status="OK")
        b = _span("b", parent_span_id="a", status="OK")
        spans = [root, a, b]
        children_map = span_tree.build_children_map(spans)
        roots = span_tree.find_roots(children_map)

        assert span_tree.find_unreachable_error_spans(spans, roots, children_map) == []


class TestParentCycleSpanIds:
    def test_ring_returns_every_span_of_the_cycle_starting_where_the_walk_entered(self) -> None:
        a = _span("a", parent_span_id="c")
        b = _span("b", parent_span_id="a")
        c = _span("c", parent_span_id="b")
        spans = [a, b, c]

        assert span_tree.parent_cycle_span_ids(b, spans) == ["b", "a", "c"]

    def test_tail_leading_into_a_cycle_reports_only_the_cycle(self) -> None:
        a = _span("a", parent_span_id="b")
        b = _span("b", parent_span_id="a")
        tail = _span("tail", parent_span_id="a")
        spans = [a, b, tail]

        assert span_tree.parent_cycle_span_ids(tail, spans) == ["a", "b"]

    def test_self_parent_is_a_cycle_of_one(self) -> None:
        loner = _span("loner", parent_span_id="loner")

        assert span_tree.parent_cycle_span_ids(loner, [loner]) == ["loner"]

    def test_walk_ending_at_a_root_or_missing_parent_is_not_a_cycle(self) -> None:
        root = _span("root")
        child = _span("child", parent_span_id="root")
        orphan = _span("orphan", parent_span_id="gone")

        assert span_tree.parent_cycle_span_ids(child, [root, child]) == []
        assert span_tree.parent_cycle_span_ids(orphan, [root, child, orphan]) == []

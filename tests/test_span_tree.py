"""Tests for the span-tree/critical-path utilities in tools/span_tree.py."""

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


class TestFindAllErrorChains:
    def test_no_error_spans_returns_empty_list(self) -> None:
        root = _span("root")
        child = _span("child", parent_span_id="root")
        children_map = span_tree.build_children_map([root, child])

        assert span_tree.find_all_error_chains(root, children_map) == []

    def test_root_itself_erroring_is_its_own_chain(self) -> None:
        root = _span("root", status="ERROR")
        children_map = span_tree.build_children_map([root])

        chains = span_tree.find_all_error_chains(root, children_map)

        assert len(chains) == 1
        assert [s.span_id for s in chains[0]] == ["root"]

    def test_deep_error_span_produces_full_root_to_leaf_path_regardless_of_intermediate_status(
        self,
    ) -> None:
        root = _span("root", status="OK")
        mid = _span("mid", parent_span_id="root", status="OK")
        leaf = _span("leaf", parent_span_id="mid", status="ERROR")
        children_map = span_tree.build_children_map([root, mid, leaf])

        chains = span_tree.find_all_error_chains(root, children_map)

        assert len(chains) == 1
        assert [s.span_id for s in chains[0]] == ["root", "mid", "leaf"]

    def test_multiple_error_spans_produce_one_chain_each(self) -> None:
        root = _span("root", status="OK")
        error_a = _span("error_a", parent_span_id="root", status="ERROR")
        mid = _span("mid", parent_span_id="root", status="OK")
        error_b = _span("error_b", parent_span_id="mid", status="ERROR")
        children_map = span_tree.build_children_map([root, error_a, mid, error_b])

        chains = span_tree.find_all_error_chains(root, children_map)

        chain_ids = {tuple(s.span_id for s in chain) for chain in chains}
        assert chain_ids == {("root", "error_a"), ("root", "mid", "error_b")}

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

        chains = span_tree.find_all_error_chains(spans[0], children_map)

        assert len(chains) == 1
        assert len(chains[0]) == depth
        assert chains[0][-1].span_id == f"s{depth - 1}"

    def test_cycle_among_valid_span_ids_terminates_instead_of_looping_forever(self) -> None:
        a = _span("a", parent_span_id="b", status="ERROR")
        b = _span("b", parent_span_id="a", status="OK")
        children_map = span_tree.build_children_map([a, b])

        chains = span_tree.find_all_error_chains(a, children_map)

        assert len(chains) == 1
        assert [s.span_id for s in chains[0]] == ["a"]


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

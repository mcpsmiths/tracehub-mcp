"""Backend-agnostic span-tree and critical-path utilities.

No backend or tool in this codebase walks parent/child span relationships
beyond finding a trace's root span - every backend independently repeats the
same one-liner (`root_spans = [s for s in spans if not s.parent_span_id]`)
and stops there. These pure functions build the shared tree/critical-path/
self-time/error-chain logic that tools/triage.py composes on top of, so it
doesn't have to reimplement span-tree walking itself.

Every tree-walking function here is iterative (no recursion) and
cycle-safe: real backend data can contain a span whose parent_span_id points
at itself, or a longer cycle among otherwise-valid span IDs (buggy
instrumentation, adversarial input), and a naive recursive walk either hangs
(an unbounded `while True` loop) or raises RecursionError (Python's call
stack has no special handling for a cycle, and a long-but-acyclic chain -
e.g. a deep recursive agent/tool-calling loop - hits the same limit).
"""

from opentelemetry_mcp.models import SpanData


def build_children_map(spans: list[SpanData]) -> dict[str | None, list[SpanData]]:
    """Group spans by parent_span_id.

    A span whose parent_span_id points at a span not present in `spans`
    (partial trace visibility from sampling, or a backend that couldn't
    hydrate the full tree), or at itself (malformed instrumentation), is an
    orphan - it is bucketed under the same `None` key as the true root(s)
    rather than dropped or raising, so every caller gets a usable,
    non-empty set of tree entry points whenever `spans` itself is
    non-empty. This only catches a direct self-reference; a longer cycle
    among otherwise-valid parent/child links (A's parent is B, B's parent
    is A) still requires the cycle guards in the walking functions below.
    """
    span_ids = {span.span_id for span in spans}
    children_map: dict[str | None, list[SpanData]] = {}
    for span in spans:
        parent_id = span.parent_span_id
        if parent_id is not None and (parent_id not in span_ids or parent_id == span.span_id):
            parent_id = None
        children_map.setdefault(parent_id, []).append(span)
    return children_map


def find_roots(children_map: dict[str | None, list[SpanData]]) -> list[SpanData]:
    """Root spans plus any orphaned spans - both live under children_map[None]."""
    return children_map.get(None, [])


def subtree_span_ids(root: SpanData, children_map: dict[str | None, list[SpanData]]) -> set[str]:
    """Every span_id reachable from `root` (root included), via an
    iterative, cycle-safe traversal - used to determine which root a given
    span belongs to in a multi-root trace (see find_owning_root)."""
    visited = {root.span_id}
    stack = [root]
    while stack:
        current = stack.pop()
        for child in children_map.get(current.span_id, []):
            if child.span_id not in visited:
                visited.add(child.span_id)
                stack.append(child)
    return visited


def find_owning_root(
    span: SpanData, roots: list[SpanData], children_map: dict[str | None, list[SpanData]]
) -> SpanData:
    """Which of `roots` has `span` in its subtree - for picking a critical
    path in a multi-root trace once a specific span (e.g. the top
    latency contributor) has already been chosen. Falls back to `span`
    itself if it belongs to none of `roots` (shouldn't happen given
    `roots` and `span` both come from the same children_map, but a single
    coherent path is still preferable to raising)."""
    for root in roots:
        if span.span_id in subtree_span_ids(root, children_map):
            return root
    return span


def compute_critical_path(
    root: SpanData, children_map: dict[str | None, list[SpanData]]
) -> list[SpanData]:
    """ "Last Finishing Child" critical path: root, then repeatedly descend
    into the child whose interval end (start_time + duration_ms) is latest
    among its siblings - the child still running when its parent's own work
    finished, and therefore the one actually responsible for the parent's
    total latency. Returns root-to-leaf, inclusive of root.

    A visited set excludes any child already on the current path, so a
    cycle (including a span whose parent is itself, in the rare case
    build_children_map's own redirect didn't already catch it - e.g. a
    longer cycle) terminates the descent instead of looping forever.
    """
    path = [root]
    visited = {root.span_id}
    current = root
    while True:
        children = [c for c in children_map.get(current.span_id, []) if c.span_id not in visited]
        if not children:
            break
        current = max(children, key=lambda s: s.start_time.timestamp() * 1000 + s.duration_ms)
        visited.add(current.span_id)
        path.append(current)
    return path


def compute_self_time_ms(span: SpanData, children_map: dict[str | None, list[SpanData]]) -> float:
    """span's own duration_ms minus the union of time covered by its direct
    children's intervals - "where the time actually went" for this span
    specifically, as opposed to duration_ms's raw, inclusive-of-children
    value. Child intervals are clipped to the parent's own interval before
    unioning, since real-world clock skew can otherwise make a child appear
    to start slightly before or end slightly after its parent. A clipped
    interval that comes out inverted (the child's clock skew places it
    entirely outside the parent's own interval) contributes zero covered
    time rather than a negative width, which would otherwise inflate
    self-time above duration_ms - not a sane result for any consumer of
    this field."""
    children = children_map.get(span.span_id, [])
    if not children:
        return span.duration_ms

    span_start_ms = span.start_time.timestamp() * 1000
    span_end_ms = span_start_ms + span.duration_ms
    intervals = sorted(
        interval
        for interval in (
            (
                max(span_start_ms, child.start_time.timestamp() * 1000),
                min(span_end_ms, child.start_time.timestamp() * 1000 + child.duration_ms),
            )
            for child in children
        )
        if interval[1] > interval[0]
    )
    if not intervals:
        return span.duration_ms

    covered_ms = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start > current_end:
            covered_ms += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    covered_ms += current_end - current_start

    return max(0.0, span.duration_ms - covered_ms)


def rank_by_latency_contribution(
    spans: list[SpanData],
    children_map: dict[str | None, list[SpanData]],
    top_n: int = 10,
) -> list[tuple[SpanData, float]]:
    """Every span paired with its self-time, sorted descending and
    truncated to top_n - the single-trace-scale, deterministic analog of
    TraceDiag's Shapley-value attribution and TraceRCA's SBFL-style
    scoring (both fleet-of-incidents-scale, overkill for one trace)."""
    ranked = sorted(
        ((span, compute_self_time_ms(span, children_map)) for span in spans),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return ranked[:top_n]


def find_all_error_chains(
    root: SpanData, children_map: dict[str | None, list[SpanData]]
) -> list[list[SpanData]]:
    """Every root-to-error-span path in the subtree reachable from `root`,
    one per span with status == "ERROR" (regardless of any intermediate
    span's own status) - root included if it errors itself. Empty list if
    no span in the subtree has status == "ERROR". The deepest of these (by
    path length) is the most likely actual root cause, since the leaf-most
    failure in a call chain is typically closest to where the error
    actually originated; callers pick the deepest path(s) themselves
    rather than this function guessing.

    Iterative (explicit stack, not recursion) so a long-but-acyclic chain
    (e.g. a deep recursive agent/tool-calling loop) doesn't raise
    RecursionError, and each stack entry carries its own path's span_id set
    so a cycle terminates that branch instead of looping forever.
    """
    chains: list[list[SpanData]] = []
    stack: list[tuple[SpanData, list[SpanData], frozenset[str]]] = [
        (root, [root], frozenset({root.span_id}))
    ]
    while stack:
        span, path, path_ids = stack.pop()
        if span.status == "ERROR":
            chains.append(path)
        for child in children_map.get(span.span_id, []):
            if child.span_id in path_ids:
                continue
            stack.append((child, [*path, child], path_ids | {child.span_id}))
    return chains

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


def find_unreachable_error_spans(
    spans: list[SpanData],
    roots: list[SpanData],
    children_map: dict[str | None, list[SpanData]],
) -> list[SpanData]:
    """ERROR-status spans not reachable from any of `roots`.

    build_children_map's orphan redirect only catches a parent_span_id that
    is missing or self-referential; a longer cycle where every span in it
    has a present, non-self parent (A's parent is B, B's parent is A, both
    in `spans`) has no entry point any root-first walk can discover, so
    find_deepest_error_chains from the real root(s) alone silently misses an
    error trapped there. Returns spans in no particular order - there is no
    reachable root to build a path from, so callers that want a "chain" for
    one of these can only treat the span itself as a standalone root (see
    find_owning_root's same fallback)."""
    reachable: set[str] = set()
    for root in roots:
        reachable |= subtree_span_ids(root, children_map)
    return [span for span in spans if span.span_id not in reachable and span.status == "ERROR"]


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
        # Clamped for the same reason as the merged-interval return below: a
        # backend affected by clock skew can supply a negative duration_ms
        # (no ge=0 constraint on the model), and a negative self_time_ms is
        # a semantically-impossible result that can surface directly as
        # triage_trace's headline likely_root_cause value.
        return max(0.0, span.duration_ms)

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
        return max(0.0, span.duration_ms)

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


def find_deepest_error_chains(
    root: SpanData, children_map: dict[str | None, list[SpanData]]
) -> tuple[int, list[list[SpanData]]]:
    """(number of ERROR spans reachable from `root`, root-to-span paths of
    maximal depth) for the subtree reachable from `root` - root included if
    it errors itself; `(0, [])` if no span in the subtree has status ==
    "ERROR". The deepest error span (by path length) is the most likely
    actual root cause, since the leaf-most failure in a call chain is
    typically closest to where the error originated; every span at that
    maximal depth gets its own path, in the order the walk pops them, so
    callers can tell an unambiguous verdict from a tie.

    Linear in the number of reachable spans, by construction: the walk
    keeps one predecessor-map entry and one depth per visited span and
    materializes only the deepest path(s) afterwards by following
    predecessors back to the root. The previous all-chains walk copied the
    whole path on every edge and retained one path per ERROR span, which a
    telemetry author could turn into N(N+1)/2 retained pointers (about
    1.6 GB for a 20 000-span all-ERROR chain) blocking the shared event
    loop - the deepest-only result is all either caller ever reported.

    Iterative (explicit stack, not recursion) so a long-but-acyclic chain
    (e.g. a deep recursive agent/tool-calling loop) doesn't raise
    RecursionError. Membership in the predecessor map is the cycle guard:
    build_children_map files every span under exactly one parent and
    redirects self-references, so each span reachable from `root` has a
    single incoming edge and a global visited set is exactly as selective
    as the per-path set it replaces - a cycle terminates that branch
    instead of looping forever.
    """
    pred: dict[str, SpanData | None] = {root.span_id: None}
    depth: dict[str, int] = {root.span_id: 1}
    stack: list[SpanData] = [root]
    error_count = 0
    max_depth = 0
    deepest: list[SpanData] = []
    while stack:
        span = stack.pop()
        span_depth = depth[span.span_id]
        if span.status == "ERROR":
            error_count += 1
            if span_depth > max_depth:
                max_depth = span_depth
                deepest = [span]
            elif span_depth == max_depth:
                deepest.append(span)
        for child in children_map.get(span.span_id, []):
            if child.span_id in pred:
                continue
            pred[child.span_id] = span
            depth[child.span_id] = span_depth + 1
            stack.append(child)

    chains: list[list[SpanData]] = []
    for span in deepest:
        chain: list[SpanData] = []
        current: SpanData | None = span
        while current is not None:
            chain.append(current)
            current = pred[current.span_id]
        chain.reverse()
        chains.append(chain)
    return error_count, chains


def find_deepest_error_chains_across_roots(
    roots: list[SpanData], children_map: dict[str | None, list[SpanData]]
) -> tuple[int, list[list[SpanData]]]:
    """find_deepest_error_chains over every root of a multi-root trace
    (async fire-and-forget spans, partial context propagation): the total
    reachable ERROR count, and the chain(s) at the GLOBAL maximal depth
    across all roots - ties keep every root's chains in `roots` order, so a
    deeper error chain hanging off a later root wins over a shallower one
    under the first, exactly as the previous flatten-then-max selection
    did. Shared by triage_trace and correlate_trace so both blame the same
    span for the same trace."""
    total_error_count = 0
    max_depth = 0
    deepest: list[list[SpanData]] = []
    for root in roots:
        error_count, chains = find_deepest_error_chains(root, children_map)
        total_error_count += error_count
        if not chains:
            continue
        root_depth = len(chains[0])
        if root_depth > max_depth:
            max_depth = root_depth
            deepest = list(chains)
        elif root_depth == max_depth:
            deepest.extend(chains)
    return total_error_count, deepest

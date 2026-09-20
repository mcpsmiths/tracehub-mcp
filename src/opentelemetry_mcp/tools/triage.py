"""Agent-native triage: synthesize a likely-root-cause diagnosis from a
single trace, instead of returning raw trace data for an agent to re-derive
one from every time.

Deterministic by design (no LLM call in the loop), matching every other
tool's shape in this codebase - see CLAUDE.md's "Two valid return patterns"
note. Composes span_tree.py's critical-path/self-time/error-chain utilities,
which is where the actual tree-walking algorithms live.
"""

from typing import Literal

from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.models import SpanData, TriagedSpanSummary, TriageResult, TriageVerdict
from opentelemetry_mcp.tools import span_tree
from opentelemetry_mcp.tools.errors import extract_error_details


def _summarize(span: SpanData, self_time_ms: float) -> TriagedSpanSummary:
    return TriagedSpanSummary(
        span_id=span.span_id,
        operation_name=span.operation_name,
        service_name=span.service_name,
        duration_ms=span.duration_ms,
        self_time_ms=self_time_ms,
        status=span.status,
    )


async def triage_trace(
    backend: BaseBackend,
    trace_id: str,
    detail_level: Literal["summary", "full"] = "summary",
) -> TriageResult:
    """Fetch a trace and return a synthesized root-cause diagnosis.

    Args:
        backend: Backend instance to query
        trace_id: Trace identifier
        detail_level: "summary" (default) returns a compact diagnosis only.
            "full" additionally attaches the diagnosed root cause's raw
            error detail (message/type/stacktrace, via the same
            extract_error_details used by find_errors) to both
            verdict.likely_root_cause.error_detail and the matching entry
            in error_chain (the same underlying span, in both places),
            when the verdict is error-driven - a pure-latency verdict (no
            error chain in the trace) has nothing to attach either way.

    Returns:
        TriageResult with a verdict, critical path (computed against
        whichever root the verdict is actually about), the top 10 spans by
        latency contribution across the whole trace, and (when present)
        the error chain that led to the verdict.
    """
    trace = await backend.get_trace(trace_id)
    spans = trace.spans

    if not spans:
        return TriageResult(
            trace_id=trace_id,
            verdict=TriageVerdict(
                likely_root_cause=None,
                confidence="low",
                reasoning="Trace has no spans to analyze.",
            ),
            critical_path=[],
            top_latency_contributors=[],
            error_chain=None,
        )

    children_map = span_tree.build_children_map(spans)
    roots = span_tree.find_roots(children_map) or [spans[0]]

    # A trace can legitimately have more than one root (async fire-and-
    # forget spans, or partial context propagation across services) - an
    # error chain anywhere under ANY root is relevant, not just the first
    # one, so every root's subtree is searched rather than picking one
    # arbitrarily up front.
    error_chains = [
        chain for root in roots for chain in span_tree.find_all_error_chains(root, children_map)
    ]
    top_contributors = span_tree.rank_by_latency_contribution(spans, children_map)

    error_chain_summaries: list[TriagedSpanSummary] | None = None
    # None of the real root(s) can reach an error trapped in a disconnected
    # cyclic parent chain (malformed instrumentation) - checked only when
    # find_all_error_chains came back empty, since a genuine error chain
    # from a real root always takes priority over this fallback.
    unreachable_errors = (
        [] if error_chains else span_tree.find_unreachable_error_spans(spans, roots, children_map)
    )

    if error_chains:
        max_depth = max(len(chain) for chain in error_chains)
        deepest_chains = [chain for chain in error_chains if len(chain) == max_depth]
        chain = deepest_chains[0]
        confidence: Literal["high", "medium"] = "high" if len(deepest_chains) == 1 else "medium"

        error_chain_summaries = [
            _summarize(span, span_tree.compute_self_time_ms(span, children_map)) for span in chain
        ]
        if detail_level == "full":
            error_chain_summaries[-1] = error_chain_summaries[-1].model_copy(
                update={"error_detail": extract_error_details(chain[-1])}
            )
        likely_root_cause = error_chain_summaries[-1]

        reasoning = (
            f"Deepest error span in the trace's error chain ({len(chain)} span(s) "
            "deep from the root)."
            if confidence == "high"
            else (
                f"{len(deepest_chains)} equally-deep error chains found in this trace; "
                "picked the first one, but blame is ambiguous between them."
            )
        )
        verdict = TriageVerdict(
            likely_root_cause=likely_root_cause, confidence=confidence, reasoning=reasoning
        )
        # The chain's own first element is its root - computing the
        # critical path from there (rather than an arbitrarily-picked
        # `roots[0]`) keeps it about the same subtree as the verdict.
        critical_path_root = chain[0]
    elif unreachable_errors:
        # An error exists but no real root can reach it - almost certainly a
        # disconnected/cyclic parent chain (malformed instrumentation), not
        # a genuinely error-free trace. Report it directly rather than
        # falling through to the pure-latency fallback below, which would
        # otherwise silently hide a real error.
        trapped = max(unreachable_errors, key=lambda s: s.duration_ms)
        error_chain_summaries = [
            _summarize(trapped, span_tree.compute_self_time_ms(trapped, children_map))
        ]
        if detail_level == "full":
            error_chain_summaries[0] = error_chain_summaries[0].model_copy(
                update={"error_detail": extract_error_details(trapped)}
            )
        likely_root_cause = error_chain_summaries[0]
        reasoning = (
            "Error span found, but it is not reachable from any detected trace "
            "root - likely a disconnected/cyclic parent chain (malformed "
            "instrumentation) rather than a genuinely error-free trace."
            + (
                f" {len(unreachable_errors)} such span(s) found; picked the longest-running one."
                if len(unreachable_errors) > 1
                else ""
            )
        )
        verdict = TriageVerdict(
            likely_root_cause=likely_root_cause, confidence="low", reasoning=reasoning
        )
        # No reachable root exists for this span - it stands in as its own
        # critical-path root, same fallback find_owning_root already uses.
        critical_path_root = trapped
    else:
        top_span, top_self_time = top_contributors[0]
        verdict = TriageVerdict(
            likely_root_cause=_summarize(top_span, top_self_time),
            confidence="low",
            reasoning=(
                "No error span found in this trace; falling back to the span with "
                "the highest self-time (latency contribution) as a pure-latency "
                "diagnosis, not an error-driven one."
            ),
        )
        critical_path_root = span_tree.find_owning_root(top_span, roots, children_map)

    critical_path = span_tree.compute_critical_path(critical_path_root, children_map)

    return TriageResult(
        trace_id=trace_id,
        verdict=verdict,
        critical_path=[
            _summarize(span, span_tree.compute_self_time_ms(span, children_map))
            for span in critical_path
        ],
        top_latency_contributors=[
            _summarize(span, self_time) for span, self_time in top_contributors
        ],
        error_chain=error_chain_summaries,
    )

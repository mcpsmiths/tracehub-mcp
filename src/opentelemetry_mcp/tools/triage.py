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

# How much of a root-to-span chain (error_chain and critical_path) one triage
# result carries. A chain is as deep as the telemetry author made the trace:
# the runtime probe's 20,000-span chain produced a 6.9 MB tool result, and
# the mcp SDK client fails on results of ~2.9 MB and above. Each list
# therefore keeps its first _CHAIN_HEAD and last N spans around ONE explicit
# elision entry (span_id _ELIDED_SPAN_ID, operation_name naming how many
# spans were dropped); the chain's root and the diagnosed span are always
# among those kept, and the verdict text states the real depth. "summary"
# keeps 5 + 20 spans, "full" 5 + 995 = 1000. A chain that would lose a
# single span to the marker is reported whole instead.
_CHAIN_HEAD = 5
_SUMMARY_CHAIN_TAIL = 20
_FULL_CHAIN_TAIL = 995
_ELIDED_SPAN_ID = "..."


def _summarize(span: SpanData, self_time_ms: float) -> TriagedSpanSummary:
    return TriagedSpanSummary(
        span_id=span.span_id,
        operation_name=span.operation_name,
        service_name=span.service_name,
        duration_ms=span.duration_ms,
        self_time_ms=self_time_ms,
        status=span.status,
    )


def _elision_marker(elided_count: int) -> TriagedSpanSummary:
    """The one entry that stands in for the elided middle of a chain. Not a
    span: the sentinel span_id and the bracketed operation_name make it
    unmistakable to a consumer iterating the list."""
    return TriagedSpanSummary(
        span_id=_ELIDED_SPAN_ID,
        operation_name=f"[{elided_count} intermediate span(s) elided]",
        service_name=_ELIDED_SPAN_ID,
        duration_ms=0.0,
        self_time_ms=0.0,
        status="UNSET",
    )


def _summarize_chain(
    chain: list[SpanData], children_map: dict[str | None, list[SpanData]], tail: int
) -> tuple[list[TriagedSpanSummary], int]:
    """Summaries for a root-to-span chain, elided to its first _CHAIN_HEAD
    and last `tail` spans around one elision marker when it is longer than
    that plus one. Returns the list and the number of spans elided (0 when
    the chain is reported whole). Only the kept spans are summarised, so
    the work is bounded by the output, not by the chain."""
    elided = len(chain) - _CHAIN_HEAD - tail
    if elided <= 1:
        return [
            _summarize(span, span_tree.compute_self_time_ms(span, children_map)) for span in chain
        ], 0
    kept = [*chain[:_CHAIN_HEAD], *chain[-tail:]]
    summaries = [
        _summarize(span, span_tree.compute_self_time_ms(span, children_map)) for span in kept
    ]
    summaries.insert(_CHAIN_HEAD, _elision_marker(elided))
    return summaries, elided


def _elision_note(field_name: str, total: int, elided: int, tail: int) -> str:
    return (
        f" {field_name} lists the first {_CHAIN_HEAD} and last {tail} of its {total} spans; "
        f"one elision entry (span_id {_ELIDED_SPAN_ID!r}) marks the {elided} left out."
    )


def _no_root_reasoning(span: SpanData, spans: list[SpanData]) -> str:
    """Why a trace with no root span gets confidence "low": there is no depth
    from a root to measure, only a parent cycle, which is named here."""
    cycle = span_tree.parent_cycle_span_ids(span, spans)
    where = (
        f"This span sits in a parent cycle of {len(cycle)} span(s), entered at span {cycle[0]!r}"
        if cycle
        else "This span's parent chain never reaches a root"
    )
    return (
        "Trace has no root span: every span's parent_span_id points at another span in the "
        f"trace, so no depth from a root exists. {where} - malformed instrumentation."
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
            Both levels elide the middle of a long error_chain /
            critical_path (see _CHAIN_HEAD): "summary" keeps 25 spans of
            each, "full" 1000, with one elision entry marking the gap and
            the real depth stated in the verdict's reasoning.

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
    # No stand-in root: a trace with no root at all (every parent present
    # and non-self, i.e. one parent cycle) used to be walked from spans[0]
    # as if that were the root, which gave a 5000-span ring confidence
    # "high" and "5000 span(s) deep from the root". With `roots` empty the
    # error chain is empty, every ERROR span is unreachable, and the
    # branches below report the cycle for what it is.
    roots = span_tree.find_roots(children_map)
    tail = _SUMMARY_CHAIN_TAIL if detail_level == "summary" else _FULL_CHAIN_TAIL

    # A trace can legitimately have more than one root (async fire-and-
    # forget spans, or partial context propagation across services) - an
    # error chain anywhere under ANY root is relevant, not just the first
    # one, so every root's subtree is searched rather than picking one
    # arbitrarily up front. Exactly one chain is ever materialized - the
    # first at the global maximal depth - and equally-deep errors are
    # counted rather than rebuilt: the walk is linear in the span count, so
    # a telemetry author cannot make one trace_id cost quadratic time and
    # memory on the event loop.
    deepest = span_tree.find_deepest_error_chain_across_roots(roots, children_map)
    top_contributors = span_tree.rank_by_latency_contribution(spans, children_map)

    error_chain_summaries: list[TriagedSpanSummary] | None = None
    chain_elided = 0
    # None of the real root(s) can reach an error trapped in a disconnected
    # cyclic parent chain (malformed instrumentation) - checked only when
    # no root reaches any ERROR span, since a genuine error chain from a
    # real root always takes priority over this fallback.
    unreachable_errors = (
        []
        if deepest.error_count
        else span_tree.find_unreachable_error_spans(spans, roots, children_map)
    )

    if deepest.chain:
        chain = deepest.chain
        confidence: Literal["high", "medium"] = "high" if deepest.tie_count == 1 else "medium"

        error_chain_summaries, chain_elided = _summarize_chain(chain, children_map, tail)
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
                f"{deepest.tie_count} equally-deep error chains found in this trace; "
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
            _no_root_reasoning(trapped, spans)
            if not roots
            else (
                "Error span found, but it is not reachable from any detected trace "
                "root - likely a disconnected/cyclic parent chain (malformed "
                "instrumentation) rather than a genuinely error-free trace."
            )
        ) + (
            f" {len(unreachable_errors)} such span(s) found; picked the longest-running one."
            if len(unreachable_errors) > 1
            else ""
        )
        verdict = TriageVerdict(
            likely_root_cause=likely_root_cause, confidence="low", reasoning=reasoning
        )
        # No reachable root exists for this span - it stands in as its own
        # critical-path root, same fallback find_owning_root already uses.
        critical_path_root = trapped
    else:
        top_span, top_self_time = top_contributors[0]
        reasoning = (
            "No error span found in this trace; falling back to the span with "
            "the highest self-time (latency contribution) as a pure-latency "
            "diagnosis, not an error-driven one."
        )
        if not roots:
            reasoning = _no_root_reasoning(top_span, spans) + " " + reasoning
        verdict = TriageVerdict(
            likely_root_cause=_summarize(top_span, top_self_time),
            confidence="low",
            reasoning=reasoning,
        )
        critical_path_root = span_tree.find_owning_root(top_span, roots, children_map)

    critical_path = span_tree.compute_critical_path(critical_path_root, children_map)
    critical_path_summaries, path_elided = _summarize_chain(critical_path, children_map, tail)

    notes = ""
    if chain_elided:
        notes += _elision_note("error_chain", len(deepest.chain), chain_elided, tail)
    if path_elided:
        notes += _elision_note("critical_path", len(critical_path), path_elided, tail)
    if notes:
        verdict = verdict.model_copy(update={"reasoning": verdict.reasoning + notes})

    return TriageResult(
        trace_id=trace_id,
        verdict=verdict,
        critical_path=critical_path_summaries,
        top_latency_contributors=[
            _summarize(span, self_time) for span, self_time in top_contributors
        ],
        error_chain=error_chain_summaries,
    )

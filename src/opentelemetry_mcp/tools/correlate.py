"""Cross-backend correlation: given a trace_id known to a primary backend,
try to find the corresponding data in a second, independently-configured
backend - e.g. a Datadog trace and its downstream Sentry error, joined.

Deliberately NOT a schema-level join. Per the project's research: W3C Trace
Context was built for live in-flight propagation across instrumentation
boundaries, not for after-the-fact joining of records already stored in two
independent, non-communicating backend platforms - trace-id continuity is
not even guaranteed across a vendor boundary under normal, non-adversarial
propagation (a receiving vendor may legally mint a new trace-id on a parse
failure). The best available prior art (Grafana's Correlations feature)
does not rely on a shared ID standard either - it extracts identifying
context (trace_id, service names, time bounds) and re-queries the target
system in its own terms. This module follows that pattern: try a direct
trace_id match first, and fall back to a time-window + service-overlap
heuristic search when that fails, with explicit, always-reported
limitations rather than presenting a heuristic match as a guaranteed join.
"""

import logging
from datetime import timedelta

from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.models import CorrelationMatch, CorrelationResult, TraceData, TraceQuery
from opentelemetry_mcp.tools import span_tree

logger = logging.getLogger(__name__)

_SEARCH_WINDOW_PAD = timedelta(seconds=5)
_MAX_CANDIDATES_PER_SERVICE_QUERY = 20
_MAX_REPORTED_MATCHES = 5

_LIMITATIONS = [
    "Clock skew between backends can shift apparent timing alignment.",
    "Sampling-rate mismatches mean a span visible in one backend may not "
    "exist at all in the other.",
    "Partial trace visibility: each backend may only see the services "
    "actually instrumented to point at it.",
    "A time-window/service-overlap match is a heuristic, not a guaranteed "
    "join - trace-id continuity across a vendor boundary is not required "
    "by the W3C Trace Context spec even under normal propagation.",
]


def _error_signal(trace: TraceData) -> tuple[bool, str | None]:
    """(has_error, blamed_service_name) for one trace - reuses
    span_tree.py's already cycle/recursion-hardened error-chain walk
    (the same one triage_trace uses) rather than a naive "any span has
    status == ERROR" check, so multi-root trace shapes are handled
    consistently between the two tools."""
    spans = trace.spans
    if not spans:
        return False, None

    children_map = span_tree.build_children_map(spans)
    roots = span_tree.find_roots(children_map) or [spans[0]]
    error_chains = [
        chain for root in roots for chain in span_tree.find_all_error_chains(root, children_map)
    ]
    if error_chains:
        deepest = max(error_chains, key=len)
        return True, deepest[-1].service_name

    # No root-reachable error chain, but an error can still be trapped in a
    # disconnected/cyclic parent chain no root-first walk can discover (see
    # span_tree.find_unreachable_error_spans) - the same blind spot
    # triage_trace's own error-chain search has, and just as real here.
    unreachable_errors = span_tree.find_unreachable_error_spans(spans, roots, children_map)
    if not unreachable_errors:
        return False, None
    trapped = max(unreachable_errors, key=lambda s: s.duration_ms)
    return True, trapped.service_name


def _root_cause_consistent(
    primary_has_error: bool,
    primary_blamed_service: str | None,
    secondary_has_error: bool,
    secondary_blamed_service: str | None,
) -> bool | None:
    """None when neither side has an error to compare (a pure-latency
    correlation has nothing to confirm or refute); otherwise True only
    when both sides agree on both whether there is an error and which
    service it traces back to."""
    if not primary_has_error and not secondary_has_error:
        return None
    return (
        primary_has_error == secondary_has_error
        and primary_blamed_service == secondary_blamed_service
    )


async def correlate_trace(
    primary_backend: BaseBackend, secondary_backend: BaseBackend, trace_id: str
) -> CorrelationResult:
    """Fetch a trace from the primary backend and try to find the
    corresponding trace in the secondary backend.

    Args:
        primary_backend: The backend `trace_id` is already known to.
        secondary_backend: The independently-configured backend to search.
        trace_id: Trace identifier, as known to `primary_backend`.

    Returns:
        CorrelationResult with every candidate match found (direct
        trace_id match, if any, else up to 5 time-window/service-overlap
        heuristic candidates) and a fixed list of limitations that always
        apply to this kind of best-effort cross-backend correlation.
    """
    primary_trace = await primary_backend.get_trace(trace_id)
    primary_services = {span.service_name for span in primary_trace.spans}
    primary_has_error, primary_blamed_service = _error_signal(primary_trace)

    matches: list[CorrelationMatch] = []

    try:
        secondary_trace: TraceData | None = await secondary_backend.get_trace(trace_id)
    except Exception as e:
        logger.debug(f"correlate_trace: secondary get_trace({trace_id!r}) failed: {e}")
        secondary_trace = None

    if secondary_trace is not None:
        secondary_has_error, secondary_blamed_service = _error_signal(secondary_trace)
        matches.append(
            CorrelationMatch(
                secondary_trace_id=trace_id,
                correlation_method="trace_id_match",
                confidence="high",
                service_overlap=sorted(
                    primary_services & {span.service_name for span in secondary_trace.spans}
                ),
                root_cause_consistent=_root_cause_consistent(
                    primary_has_error,
                    primary_blamed_service,
                    secondary_has_error,
                    secondary_blamed_service,
                ),
            )
        )
        return CorrelationResult(
            primary_trace_id=trace_id, matches=matches, limitations=_LIMITATIONS
        )

    # Heuristic fallback: query per primary service name (not one filter-
    # less call) so this works against backends like Jaeger that require
    # `service_name` on search_traces, not just the ones where it's optional.
    start_time = primary_trace.start_time - _SEARCH_WINDOW_PAD
    end_time = (
        primary_trace.start_time
        + timedelta(milliseconds=primary_trace.duration_ms)
        + _SEARCH_WINDOW_PAD
    )
    candidates_by_id: dict[str, TraceData] = {}
    # Sorted rather than iterated directly off the set: str hashing (and
    # therefore set iteration order) is randomized per-process by default
    # (PYTHONHASHSEED), which would otherwise make candidates_by_id's
    # insertion order - and thus which candidate wins a tie in the ranked
    # sort below - non-deterministic across runs for the exact same input.
    for service_name in sorted(primary_services):
        query = TraceQuery(
            service_name=service_name,
            start_time=start_time,
            end_time=end_time,
            limit=_MAX_CANDIDATES_PER_SERVICE_QUERY,
        )
        try:
            for candidate in await secondary_backend.search_traces(query):
                # get_trace(trace_id) failing (why we're in this fallback
                # at all) does not guarantee search_traces can never also
                # turn up that same trace_id (different endpoint, different
                # failure mode) - reporting it back as a "heuristic,
                # low-confidence" match would be actively misleading, since
                # it's not independent evidence of anything, just the input
                # echoed back.
                if candidate.trace_id != trace_id:
                    candidates_by_id[candidate.trace_id] = candidate
        except Exception as e:
            logger.warning(
                f"correlate_trace: secondary search_traces failed for "
                f"service_name={service_name!r}: {e}"
            )

    ranked = sorted(
        candidates_by_id.values(),
        key=lambda t: len(primary_services & {span.service_name for span in t.spans}),
        reverse=True,
    )
    for candidate in ranked:
        overlap = sorted(primary_services & {span.service_name for span in candidate.spans})
        if not overlap:
            continue  # no shared service at all - not worth reporting as a candidate
        candidate_has_error, candidate_blamed_service = _error_signal(candidate)
        matches.append(
            CorrelationMatch(
                secondary_trace_id=candidate.trace_id,
                correlation_method="time_window_heuristic",
                confidence="low",
                service_overlap=overlap,
                root_cause_consistent=_root_cause_consistent(
                    primary_has_error,
                    primary_blamed_service,
                    candidate_has_error,
                    candidate_blamed_service,
                ),
            )
        )
        if len(matches) >= _MAX_REPORTED_MATCHES:
            break

    return CorrelationResult(primary_trace_id=trace_id, matches=matches, limitations=_LIMITATIONS)

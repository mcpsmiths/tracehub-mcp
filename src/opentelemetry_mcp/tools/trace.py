"""Get trace tool implementation."""

from typing import Any, Literal

from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.constants import GenAI
from opentelemetry_mcp.models import LLMSpanAttributes, SpanDetail, TraceDetail

# Same fields LLMSpanAttributes.prompt_preview/completion_preview already
# only ever preview (100 chars) - get_trace's own "attributes"/"events"
# dump bypasses that and exposes the full, unbounded values independently.
_LARGE_ATTRIBUTE_KEYS = (
    GenAI.INPUT_MESSAGES,
    GenAI.OUTPUT_MESSAGES,
    GenAI.SYSTEM_INSTRUCTIONS,
    GenAI.RETRIEVAL_DOCUMENTS,
)
_EVENT_ATTRIBUTE_TRUNCATE_AT = 500


def _elide_large_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Replace known-large gen_ai.* attribute values with an omission
    marker, reusing the same truncation intent as LLMSpanAttributes'
    previews but applied to the full attributes dict get_trace exposes."""
    elided = dict(attributes)
    for key in _LARGE_ATTRIBUTE_KEYS:
        value = elided.get(key)
        if value is not None:
            elided[key] = f"<omitted {len(value)} item(s) - pass detail_level='full' to include>"
    return elided


def _truncate_event_attributes(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Truncate long string event-attribute values, same slicing
    convention (value[:N] + "...") as LLMSpanAttributes' previews. Applies
    generically to every event attribute, not just named prompt/completion
    events, since arbitrary instrumentations can attach large event data."""
    truncated: list[dict[str, Any]] = []
    for event in events:
        attrs = event.get("attributes")
        if isinstance(attrs, dict):
            new_attrs = {
                key: (
                    value[:_EVENT_ATTRIBUTE_TRUNCATE_AT] + "..."
                    if isinstance(value, str) and len(value) > _EVENT_ATTRIBUTE_TRUNCATE_AT
                    else value
                )
                for key, value in attrs.items()
            }
            event = {**event, "attributes": new_attrs}
        truncated.append(event)
    return truncated


async def get_trace(
    backend: BaseBackend, trace_id: str, detail_level: Literal["summary", "full"] = "full"
) -> TraceDetail:
    """Get complete trace details by trace ID.

    Args:
        backend: Backend instance to query
        trace_id: Trace identifier
        detail_level: "full" (default) reproduces this tool's original,
            unbounded behavior - every existing caller's output is
            unchanged unless it opts into "summary". "summary" elides
            known-large gen_ai.* attribute fields (input/output messages,
            system instructions, retrieval documents) and truncates long
            event-attribute values.

    Returns:
        TraceDetail with complete trace data including all spans
    """
    trace = await backend.get_trace(trace_id)

    spans: list[SpanDetail] = []
    for span in trace.spans:
        attributes = span.attributes.to_dict()
        events = [event.model_dump(mode="json") for event in span.events]
        if detail_level == "summary":
            attributes = _elide_large_attributes(attributes)
            events = _truncate_event_attributes(events)

        llm_attributes: dict[str, Any] | None = None
        if span.is_llm_span:
            llm_attrs = LLMSpanAttributes.from_span(span)
            if llm_attrs:
                llm_attributes = llm_attrs.model_dump(mode="json", exclude_none=True)

        spans.append(
            SpanDetail(
                span_id=span.span_id,
                parent_span_id=span.parent_span_id,
                operation_name=span.operation_name,
                service_name=span.service_name,
                start_time=span.start_time,
                duration_ms=span.duration_ms,
                status=span.status,
                attributes=attributes,
                events=events,
                llm_attributes=llm_attributes,
            )
        )

    llm_summary: dict[str, Any] | None = None
    if trace.llm_spans:
        llm_summary = {
            "llm_span_count": len(trace.llm_spans),
            "total_tokens": trace.total_llm_tokens,
            "models_used": list(
                {
                    span.attributes.gen_ai_request_model or span.attributes.gen_ai_response_model
                    for span in trace.llm_spans
                    if span.attributes.gen_ai_request_model or span.attributes.gen_ai_response_model
                }
            ),
        }

    return TraceDetail(
        trace_id=trace.trace_id,
        service_name=trace.service_name,
        root_operation=trace.root_operation,
        start_time=trace.start_time,
        duration_ms=trace.duration_ms,
        status=trace.status,
        span_count=len(trace.spans),
        has_errors=trace.has_errors,
        spans=spans,
        llm_summary=llm_summary,
        detail_level=detail_level,
    )

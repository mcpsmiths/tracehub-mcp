"""List sessions / session stats tool implementation.

Groups spans by the `gen_ai.conversation.id` attribute - a real,
cross-industry-adopted OTel semantic convention for session/conversation
grouping (confirmed adoption in Weave, LangWatch, Sentry, Google ADK, Azure
SDK, OpenLit, Dynatrace) - to answer "what conversations happened" and
"how did this specific conversation perform" without the caller having to
manually correlate spans by hand.
"""

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.constants import GenAI
from opentelemetry_mcp.models import (
    Filter,
    FilterOperator,
    FilterType,
    LLMSpanAttributes,
    SpanQuery,
)
from opentelemetry_mcp.tools.model_stats import calculate_percentiles
from opentelemetry_mcp.utils import parse_iso_timestamp


class SessionInfo(BaseModel):
    """Aggregated statistics for one gen_ai.conversation.id."""

    conversation_id: str
    span_count: int
    services: list[str]
    first_seen: datetime
    last_seen: datetime
    total_tokens: int


class ListSessionsResult(BaseModel):
    """Structured response shape for the list_sessions tool - a real
    return type (rather than a bare str) lets FastMCP auto-derive a
    genuinely useful MCP outputSchema/structuredContent."""

    count: int
    sessions: list[SessionInfo]
    message: str | None = None


def _span_tokens(span: Any) -> int:
    """Total tokens for a span, reusing LLMSpanAttributes' enhanced
    calculation (explicit total -> sum of gen_ai.usage.* -> prompt+completion).
    Non-LLM spans (e.g. tool calls sharing a conversation_id) contribute 0."""
    llm_attrs = LLMSpanAttributes.from_span(span)
    return (llm_attrs.total_tokens or 0) if llm_attrs else 0


async def list_sessions(
    backend: BaseBackend,
    start_time: str | None = None,
    end_time: str | None = None,
    service_name: str | None = None,
    gen_ai_system: str | None = None,
    limit: int = 1000,
) -> ListSessionsResult:
    """List conversations/sessions grouped by gen_ai.conversation.id.

    Args:
        backend: Backend instance to query
        start_time: Start time in ISO 8601 format
        end_time: End time in ISO 8601 format
        service_name: Filter by service name
        gen_ai_system: Filter by LLM provider (openai, anthropic, etc.)
        limit: Maximum spans to analyze (default: 1000)

    Returns:
        List of sessions with their statistics
    """
    start_dt, error = parse_iso_timestamp(start_time, "start_time")
    if error:
        raise ValueError(error)

    end_dt, error = parse_iso_timestamp(end_time, "end_time")
    if error:
        raise ValueError(error)

    query = SpanQuery(
        service_name=service_name,
        start_time=start_dt,
        end_time=end_dt,
        gen_ai_system=gen_ai_system,
        filters=[
            Filter(
                field=GenAI.CONVERSATION_ID,
                operator=FilterOperator.EXISTS,
                value_type=FilterType.STRING,
            )
        ],
        limit=limit,
    )

    spans = await backend.search_spans(query)

    if not spans:
        return ListSessionsResult(
            count=0,
            sessions=[],
            message="No spans with gen_ai.conversation.id found matching the criteria",
        )

    sessions_map: dict[str, dict[str, Any]] = {}

    for span in spans:
        conversation_id = span.attributes.gen_ai_conversation_id
        if not conversation_id:
            continue

        if conversation_id not in sessions_map:
            sessions_map[conversation_id] = {
                "conversation_id": conversation_id,
                "span_count": 0,
                "services": set(),
                "first_seen": span.start_time,
                "last_seen": span.start_time,
                "total_tokens": 0,
            }

        session_data = sessions_map[conversation_id]
        session_data["span_count"] += 1
        session_data["services"].add(span.service_name)
        session_data["total_tokens"] += _span_tokens(span)

        if span.start_time < session_data["first_seen"]:
            session_data["first_seen"] = span.start_time
        if span.start_time > session_data["last_seen"]:
            session_data["last_seen"] = span.start_time

    sessions_list: list[SessionInfo] = []
    for session_data in sessions_map.values():
        session_data["services"] = sorted(session_data["services"])
        sessions_list.append(SessionInfo(**session_data))

    sessions_list.sort(key=lambda s: s.span_count, reverse=True)

    return ListSessionsResult(count=len(sessions_list), sessions=sessions_list)


async def get_session_stats(
    backend: BaseBackend,
    conversation_id: str,
    start_time: str | None = None,
    end_time: str | None = None,
    service_name: str | None = None,
    limit: int = 1000,
) -> str:
    """Get detailed statistics for a single conversation/session.

    Args:
        backend: Backend instance to query
        conversation_id: The gen_ai.conversation.id to analyze
        start_time: Start time in ISO 8601 format
        end_time: End time in ISO 8601 format
        service_name: Filter by service name
        limit: Maximum spans to analyze (default: 1000)

    Returns:
        JSON string with comprehensive session statistics
    """
    start_dt, error = parse_iso_timestamp(start_time, "start_time")
    if error:
        raise ValueError(error)

    end_dt, error = parse_iso_timestamp(end_time, "end_time")
    if error:
        raise ValueError(error)

    query = SpanQuery(
        service_name=service_name,
        start_time=start_dt,
        end_time=end_dt,
        filters=[
            Filter(
                field=GenAI.CONVERSATION_ID,
                operator=FilterOperator.EQUALS,
                value=conversation_id,
                value_type=FilterType.STRING,
            )
        ],
        limit=limit,
    )

    spans = await backend.search_spans(query)

    if not spans:
        return json.dumps(
            {
                "error": f"No spans found for conversation '{conversation_id}' in the specified time range"
            }
        )

    services: set[str] = set()
    first_seen = spans[0].start_time
    last_seen = spans[0].start_time
    durations: list[float] = []
    prompt_tokens_list: list[int] = []
    completion_tokens_list: list[int] = []
    total_tokens_list: list[int] = []
    finish_reasons_count: dict[str, int] = {}
    error_count = 0
    success_count = 0
    request_count = 0

    for span in spans:
        services.add(span.service_name)
        if span.start_time < first_seen:
            first_seen = span.start_time
        if span.start_time > last_seen:
            last_seen = span.start_time

        llm_attrs = LLMSpanAttributes.from_span(span)
        if not llm_attrs:
            continue

        request_count += 1
        durations.append(span.duration_ms)

        if llm_attrs.prompt_tokens:
            prompt_tokens_list.append(llm_attrs.prompt_tokens)
        if llm_attrs.completion_tokens:
            completion_tokens_list.append(llm_attrs.completion_tokens)
        if llm_attrs.total_tokens:
            total_tokens_list.append(llm_attrs.total_tokens)

        if llm_attrs.finish_reasons:
            for reason in llm_attrs.finish_reasons:
                finish_reasons_count[reason] = finish_reasons_count.get(reason, 0) + 1

        if span.has_error:
            error_count += 1
        else:
            success_count += 1

    result = {
        "conversation_id": conversation_id,
        "span_count": len(spans),
        "services": sorted(services),
        "first_seen": first_seen.isoformat(),
        "last_seen": last_seen.isoformat(),
        "llm_request_count": request_count,
        "success_count": success_count,
        "error_count": error_count,
        "success_rate": round(success_count / request_count * 100, 2) if request_count else None,
        "error_rate": round(error_count / request_count * 100, 2) if request_count else None,
        "duration_ms": calculate_percentiles(durations),
        "tokens": {
            "prompt": calculate_percentiles(prompt_tokens_list),
            "completion": calculate_percentiles(completion_tokens_list),
            "total": calculate_percentiles(total_tokens_list),
        },
        "finish_reasons": finish_reasons_count if finish_reasons_count else None,
    }

    return json.dumps(result, indent=2, default=str)

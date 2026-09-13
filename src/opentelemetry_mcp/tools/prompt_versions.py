"""Prompt version stats tool implementation.

Groups spans by gen_ai.prompt.name + gen_ai.prompt.version - mirroring
Langfuse's shipped per-prompt Metrics tab (median latency/cost/tokens
grouped by version). Real-world adoption of these two attributes is still
thin (early-stage OTel GenAI semconv, Development status), so this tool
will often return an empty list until more instrumentations populate them -
that is expected, not a bug.
"""

import json
from typing import Any

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


async def get_prompt_version_stats(
    backend: BaseBackend,
    start_time: str | None = None,
    end_time: str | None = None,
    service_name: str | None = None,
    gen_ai_system: str | None = None,
    limit: int = 1000,
) -> str:
    """Get aggregated performance stats grouped by prompt name and version.

    Args:
        backend: Backend instance to query
        start_time: Start time in ISO 8601 format
        end_time: End time in ISO 8601 format
        service_name: Filter by service name
        gen_ai_system: Filter by LLM provider (openai, anthropic, etc.)
        limit: Maximum spans to analyze (default: 1000)

    Returns:
        JSON string with one entry per (prompt_name, prompt_version) pair,
        each carrying request count, time bounds, and duration/token
        percentiles (mirroring get_llm_model_stats' shape)
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
                field=GenAI.PROMPT_NAME,
                operator=FilterOperator.EXISTS,
                value_type=FilterType.STRING,
            )
        ],
        limit=limit,
    )

    spans = await backend.search_spans(query)

    if not spans:
        return json.dumps(
            {
                "count": 0,
                "prompt_versions": [],
                "message": "No spans with gen_ai.prompt.name found matching the criteria",
            }
        )

    groups: dict[tuple[str, str | None], dict[str, Any]] = {}

    for span in spans:
        prompt_name = span.attributes.gen_ai_prompt_name
        if not prompt_name:
            continue
        prompt_version = span.attributes.gen_ai_prompt_version
        key = (prompt_name, prompt_version)

        if key not in groups:
            groups[key] = {
                "prompt_name": prompt_name,
                "prompt_version": prompt_version,
                "request_count": 0,
                "first_seen": span.start_time,
                "last_seen": span.start_time,
                "durations": [],
                "total_tokens_list": [],
            }

        group = groups[key]
        group["request_count"] += 1
        group["durations"].append(span.duration_ms)
        if span.start_time < group["first_seen"]:
            group["first_seen"] = span.start_time
        if span.start_time > group["last_seen"]:
            group["last_seen"] = span.start_time

        llm_attrs = LLMSpanAttributes.from_span(span)
        if llm_attrs and llm_attrs.total_tokens:
            group["total_tokens_list"].append(llm_attrs.total_tokens)

    prompt_versions = []
    for group in groups.values():
        prompt_versions.append(
            {
                "prompt_name": group["prompt_name"],
                "prompt_version": group["prompt_version"],
                "request_count": group["request_count"],
                "first_seen": group["first_seen"].isoformat(),
                "last_seen": group["last_seen"].isoformat(),
                "duration_ms": calculate_percentiles(group["durations"]),
                "total_tokens": calculate_percentiles(group["total_tokens_list"]),
            }
        )

    prompt_versions.sort(key=lambda x: x["request_count"], reverse=True)

    result = {
        "count": len(prompt_versions),
        "prompt_versions": prompt_versions,
    }

    return json.dumps(result, indent=2, default=str)

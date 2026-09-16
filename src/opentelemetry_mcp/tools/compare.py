"""Compare LLM usage between two time windows tool implementation.

Pure composition over get_llm_usage: call the same aggregation for each
window, then diff the two summaries. This is the concrete, evidence-backed
realization of a recurring comparison request (Langfuse has a currently-
open PR requesting exactly this shape as an MCP tool, plus three shipped
features for the same underlying capability - see the research report for
details) - not new aggregation logic, just composition of an existing tool.
"""

import json
from typing import Any

from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.tools.usage import get_llm_usage

_SUMMARY_FIELDS = (
    "total_requests",
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_tokens",
    "total_cost_usd",
)


def compute_delta(value_a: float, value_b: float) -> dict[str, Any]:
    """(value_b - value_a) plus percent change relative to value_a.

    Exported so other composition-over-get_llm_usage tools (e.g.
    investigate_cost_spike) can reuse the same delta math instead of
    re-deriving this formula.
    """
    change = value_b - value_a
    percent_change = round((change / value_a) * 100, 2) if value_a else None
    return {"change": change, "percent_change": percent_change}


def _diff_summary(summary_a: dict[str, Any], summary_b: dict[str, Any]) -> dict[str, Any]:
    """Per-field (range_b - range_a) plus percent change relative to range_a."""
    delta: dict[str, Any] = {}
    for field in _SUMMARY_FIELDS:
        value_a = summary_a.get(field, 0) or 0
        value_b = summary_b.get(field, 0) or 0
        delta[field] = compute_delta(value_a, value_b)
    return delta


async def compare_time_windows(
    backend: BaseBackend,
    range_a_start: str | None,
    range_a_end: str | None,
    range_b_start: str | None,
    range_b_end: str | None,
    service_name: str | None = None,
    gen_ai_system: str | None = None,
    gen_ai_request_model: str | None = None,
    gen_ai_response_model: str | None = None,
    limit: int = 1000,
) -> str:
    """Compare aggregated LLM usage metrics between two time windows.

    Calls the same get_llm_usage aggregation for range A and range B, then
    diffs the two summaries (request/token counts). Useful for "this week
    vs last week" or "before/after a deploy" style comparisons.

    Args:
        backend: Backend instance to query
        range_a_start: Range A start time in ISO 8601 format
        range_a_end: Range A end time in ISO 8601 format
        range_b_start: Range B start time in ISO 8601 format
        range_b_end: Range B end time in ISO 8601 format
        service_name: Filter by service name (applied to both ranges)
        gen_ai_system: Filter by LLM provider (applied to both ranges)
        gen_ai_request_model: Filter by requested model name (applied to both ranges)
        gen_ai_response_model: Filter by actual model used (applied to both ranges)
        limit: Maximum number of traces to analyze per range (default: 1000)

    Returns:
        JSON string with range_a, range_b (each the full get_llm_usage shape)
        and a delta summarizing the change (range_b minus range_a) per field
    """
    range_a_raw = await get_llm_usage(
        backend,
        start_time=range_a_start,
        end_time=range_a_end,
        service_name=service_name,
        gen_ai_system=gen_ai_system,
        gen_ai_request_model=gen_ai_request_model,
        gen_ai_response_model=gen_ai_response_model,
        limit=limit,
    )
    range_b_raw = await get_llm_usage(
        backend,
        start_time=range_b_start,
        end_time=range_b_end,
        service_name=service_name,
        gen_ai_system=gen_ai_system,
        gen_ai_request_model=gen_ai_request_model,
        gen_ai_response_model=gen_ai_response_model,
        limit=limit,
    )

    range_a = json.loads(range_a_raw)
    range_b = json.loads(range_b_raw)

    result = {
        "range_a": range_a,
        "range_b": range_b,
        "delta": _diff_summary(range_a["summary"], range_b["summary"]),
    }

    return json.dumps(result, indent=2, default=str)

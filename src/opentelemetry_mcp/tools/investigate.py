"""investigate_cost_spike / investigate_error_spike tool implementations.

Both compare a "recent" window against a "baseline" window and RANK
per-model/per-service (and, for errors, per-error-type) contributors to
what changed - the agent-invoked, on-request analysis pattern SigNoz's own
"investigate telemetry cost" skill uses (explicitly pull-based, not a push-
alert daemon, which doesn't map onto a stateless MCP tool the way this
does).
"""

import json
from typing import Any

from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.models import TraceQuery
from opentelemetry_mcp.tools.compare import compute_delta
from opentelemetry_mcp.tools.errors import extract_error_details
from opentelemetry_mcp.tools.usage import get_llm_usage
from opentelemetry_mcp.utils import parse_iso_timestamp

_MAX_TOP_N = 50


def _resolve_baseline_window(
    recent_start: str | None,
    recent_end: str | None,
    baseline_start: str | None,
    baseline_end: str | None,
) -> tuple[str, str, bool]:
    """Resolve the baseline window - explicit if both bounds are given,
    else auto-computed as "same duration, immediately preceding" the
    recent window. Returns (baseline_start, baseline_end, auto_computed)."""
    if bool(baseline_start) != bool(baseline_end):
        raise ValueError("baseline_start and baseline_end must both be provided, or both omitted")
    if baseline_start and baseline_end:
        return baseline_start, baseline_end, False

    if not recent_start or not recent_end:
        raise ValueError("an auto-computed baseline requires both recent_start and recent_end")

    recent_start_dt, error = parse_iso_timestamp(recent_start, "recent_start")
    if error or recent_start_dt is None:
        raise ValueError(error or "invalid recent_start")
    recent_end_dt, error = parse_iso_timestamp(recent_end, "recent_end")
    if error or recent_end_dt is None:
        raise ValueError(error or "invalid recent_end")

    duration = recent_end_dt - recent_start_dt
    auto_baseline_end = recent_start_dt
    auto_baseline_start = recent_start_dt - duration
    return auto_baseline_start.isoformat(), auto_baseline_end.isoformat(), True


def _rank_by_delta(
    baseline_values: dict[str, float], recent_values: dict[str, float], top_n: int, value_key: str
) -> list[dict[str, Any]]:
    """Union the keys present in either window, delta each, and return the
    top_n entries with the largest absolute change - a key missing from one
    side is treated as 0 (a brand-new model/service with zero baseline is
    exactly the interesting signal, not something to skip)."""
    keys = set(baseline_values) | set(recent_values)
    ranked = []
    for key in keys:
        baseline_value = baseline_values.get(key, 0) or 0
        recent_value = recent_values.get(key, 0) or 0
        ranked.append(
            {
                "name": key,
                f"baseline_{value_key}": baseline_value,
                f"recent_{value_key}": recent_value,
                **compute_delta(baseline_value, recent_value),
            }
        )
    ranked.sort(key=lambda item: abs(item["change"]), reverse=True)
    return ranked[:top_n]


async def investigate_cost_spike(
    backend: BaseBackend,
    recent_start: str,
    recent_end: str,
    baseline_start: str | None = None,
    baseline_end: str | None = None,
    service_name: str | None = None,
    gen_ai_system: str | None = None,
    gen_ai_request_model: str | None = None,
    gen_ai_response_model: str | None = None,
    limit: int = 1000,
    top_n: int = 5,
) -> str:
    """Compare LLM cost between a recent window and a baseline, ranking
    which models/services contributed most to the change.

    Args:
        backend: Backend instance to query
        recent_start: Recent window start time (ISO 8601 format)
        recent_end: Recent window end time (ISO 8601 format)
        baseline_start: Baseline window start (ISO 8601). If omitted along
            with baseline_end, auto-computed as the same duration
            immediately preceding recent_start.
        baseline_end: Baseline window end (ISO 8601)
        service_name: Filter by service name (applied to both windows)
        gen_ai_system: Filter by LLM provider (applied to both windows)
        gen_ai_request_model: Filter by requested model (applied to both windows)
        gen_ai_response_model: Filter by actual model used (applied to both windows)
        limit: Maximum number of traces to analyze per window (default: 1000)
        top_n: Maximum ranked contributors to return per breakdown (default: 5, max: 50)

    Returns:
        JSON string with recent/baseline get_llm_usage results, a
        summary_delta, and top_model_contributors/top_service_contributors
        ranked by absolute cost_usd change
    """
    resolved_baseline_start, resolved_baseline_end, auto_computed = _resolve_baseline_window(
        recent_start, recent_end, baseline_start, baseline_end
    )
    top_n = max(1, min(top_n, _MAX_TOP_N))

    recent = json.loads(
        await get_llm_usage(
            backend,
            start_time=recent_start,
            end_time=recent_end,
            service_name=service_name,
            gen_ai_system=gen_ai_system,
            gen_ai_request_model=gen_ai_request_model,
            gen_ai_response_model=gen_ai_response_model,
            limit=limit,
        )
    )
    baseline = json.loads(
        await get_llm_usage(
            backend,
            start_time=resolved_baseline_start,
            end_time=resolved_baseline_end,
            service_name=service_name,
            gen_ai_system=gen_ai_system,
            gen_ai_request_model=gen_ai_request_model,
            gen_ai_response_model=gen_ai_response_model,
            limit=limit,
        )
    )

    summary_delta = {
        field: compute_delta(
            baseline["summary"].get(field, 0) or 0, recent["summary"].get(field, 0) or 0
        )
        for field in ("total_requests", "total_tokens", "total_cost_usd")
    }

    baseline_model_cost = {k: v["cost_usd"] for k, v in baseline["by_model"].items()}
    recent_model_cost = {k: v["cost_usd"] for k, v in recent["by_model"].items()}
    baseline_service_cost = {k: v["cost_usd"] for k, v in baseline["by_service"].items()}
    recent_service_cost = {k: v["cost_usd"] for k, v in recent["by_service"].items()}

    result = {
        "recent": recent,
        "baseline": {**baseline, "auto_computed": auto_computed},
        "summary_delta": summary_delta,
        "top_model_contributors": _rank_by_delta(
            baseline_model_cost, recent_model_cost, top_n, "cost_usd"
        ),
        "top_service_contributors": _rank_by_delta(
            baseline_service_cost, recent_service_cost, top_n, "cost_usd"
        ),
        "cost_usd_is_partial": bool(
            recent["summary"]["cost_usd_is_partial"] or baseline["summary"]["cost_usd_is_partial"]
        ),
    }

    return json.dumps(result, indent=2, default=str)


async def investigate_error_spike(
    backend: BaseBackend,
    recent_start: str,
    recent_end: str,
    baseline_start: str | None = None,
    baseline_end: str | None = None,
    service_name: str | None = None,
    limit: int = 1000,
    top_n: int = 5,
    min_error_count_increase: int = 3,
    rate_multiplier_threshold: float = 2.0,
) -> str:
    """Compare error rate between a recent window and a baseline, ranking
    which services/models/error types contributed most to the change.

    is_spike requires BOTH conditions to hold, to avoid tiny-sample noise
    (e.g. 1 error becoming 2 reading as a "spike"):
    - error count increased by at least min_error_count_increase
    - error rate increased by at least rate_multiplier_threshold x (or the
      baseline rate was 0 and the recent rate is nonzero at all)

    Args:
        backend: Backend instance to query
        recent_start: Recent window start time (ISO 8601 format)
        recent_end: Recent window end time (ISO 8601 format)
        baseline_start: Baseline window start (ISO 8601). If omitted along
            with baseline_end, auto-computed as the same duration
            immediately preceding recent_start.
        baseline_end: Baseline window end (ISO 8601)
        service_name: Filter by service name (applied to both windows)
        limit: Maximum number of traces to analyze per window (default: 1000)
        top_n: Maximum ranked contributors to return per breakdown (default: 5, max: 50)
        min_error_count_increase: Minimum absolute error-count increase to
            count as a spike (default: 3)
        rate_multiplier_threshold: Minimum error-rate multiplier (recent /
            baseline) to count as a spike (default: 2.0)

    Returns:
        JSON string with recent/baseline error stats, is_spike, and ranked
        top_service_contributors/top_model_contributors/top_error_type_contributors
    """
    resolved_baseline_start, resolved_baseline_end, auto_computed = _resolve_baseline_window(
        recent_start, recent_end, baseline_start, baseline_end
    )
    top_n = max(1, min(top_n, _MAX_TOP_N))

    recent_stats = await _window_error_stats(backend, recent_start, recent_end, service_name, limit)
    baseline_stats = await _window_error_stats(
        backend, resolved_baseline_start, resolved_baseline_end, service_name, limit
    )

    count_increase = recent_stats["error_count"] - baseline_stats["error_count"]
    baseline_rate = baseline_stats["error_rate"] or 0.0
    recent_rate = recent_stats["error_rate"] or 0.0
    if baseline_rate == 0.0:
        rate_condition = recent_rate > 0.0
    else:
        rate_condition = recent_rate >= rate_multiplier_threshold * baseline_rate
    is_spike = count_increase >= min_error_count_increase and rate_condition

    top_service_contributors = _rank_by_delta(
        baseline_stats["by_service_count"], recent_stats["by_service_count"], top_n, "count"
    )
    top_model_contributors = _rank_by_delta(
        baseline_stats["by_model_count"], recent_stats["by_model_count"], top_n, "count"
    )
    top_error_type_contributors = _rank_by_delta(
        baseline_stats["by_error_type_count"], recent_stats["by_error_type_count"], top_n, "count"
    )
    for entry in top_error_type_contributors:
        entry["sample_messages"] = recent_stats["sample_messages_by_error_type"].get(
            entry["name"], []
        )

    result = {
        "recent": {
            "total_traces": recent_stats["total_traces"],
            "error_traces": recent_stats["error_count"],
            "error_rate": recent_stats["error_rate"],
        },
        "baseline": {
            "total_traces": baseline_stats["total_traces"],
            "error_traces": baseline_stats["error_count"],
            "error_rate": baseline_stats["error_rate"],
            "auto_computed": auto_computed,
        },
        "delta": {
            "error_traces": compute_delta(
                baseline_stats["error_count"], recent_stats["error_count"]
            )
        },
        "is_spike": is_spike,
        "spike_criteria": {
            "min_error_count_increase": min_error_count_increase,
            "rate_multiplier_threshold": rate_multiplier_threshold,
        },
        "top_service_contributors": top_service_contributors,
        "top_model_contributors": top_model_contributors,
        "top_error_type_contributors": top_error_type_contributors,
    }

    return json.dumps(result, indent=2, default=str)


async def _window_error_stats(
    backend: BaseBackend,
    start: str,
    end: str,
    service_name: str | None,
    limit: int,
) -> dict[str, Any]:
    """Fetch one window's traces (no has_error filter, so the total-trace
    denominator and the error breakdown both come from the same fetch) and
    build every aggregate investigate_error_spike needs from it."""
    start_dt, error = parse_iso_timestamp(start, "start_time")
    if error:
        raise ValueError(error)
    end_dt, error = parse_iso_timestamp(end, "end_time")
    if error:
        raise ValueError(error)

    query = TraceQuery(service_name=service_name, start_time=start_dt, end_time=end_dt, limit=limit)
    traces = await backend.search_traces(query)

    total_traces = len(traces)
    error_traces = [trace for trace in traces if trace.has_errors]
    error_count = len(error_traces)
    error_rate: float | None = error_count / total_traces if total_traces else None

    by_service_count: dict[str, int] = {}
    by_model_count: dict[str, int] = {}
    by_error_type_count: dict[str, int] = {}
    sample_messages_by_error_type: dict[str, list[str]] = {}

    for trace in error_traces:
        for span in trace.spans:
            if not span.has_error:
                continue
            details = extract_error_details(span)
            by_service_count[details["service_name"]] = (
                by_service_count.get(details["service_name"], 0) + 1
            )
            if details.get("is_llm_error"):
                model = details.get("llm_model") or "unknown"
                by_model_count[model] = by_model_count.get(model, 0) + 1
            error_type = details.get("error_type", "unknown")
            by_error_type_count[error_type] = by_error_type_count.get(error_type, 0) + 1
            samples = sample_messages_by_error_type.setdefault(error_type, [])
            if len(samples) < 3:
                samples.append(details["error_message"])

    return {
        "total_traces": total_traces,
        "error_count": error_count,
        "error_rate": error_rate,
        "by_service_count": by_service_count,
        "by_model_count": by_model_count,
        "by_error_type_count": by_error_type_count,
        "sample_messages_by_error_type": sample_messages_by_error_type,
    }

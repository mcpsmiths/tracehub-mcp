"""README "What responses report and what they do not": nothing in a response reports provenance.

The section says a response never names the backend, states the limit the server applied, flags
truncation or carries a redaction policy, and that lowering a limit to the ceiling is not logged.
A denylist of field names cannot hold that claim (a renamed field walks past it), so this pins the
whole vocabulary instead: every field name each of the 19 tools returns against a loopback backend
that holds LLM, tool and error spans must be in the list below. A new field fails here until
someone has decided it does not contradict the README section, and then adds it.
"""

import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastmcp import Client

import opentelemetry_mcp.server as server
from opentelemetry_mcp.attributes import SpanAttributes, SpanEvent
from opentelemetry_mcp.models import SpanData, SpanQuery, TraceData, TraceQuery
from tests.readme_response_support import _body, _call, _ceiling, _Fake, _install
from tests.test_readme_response_windows_and_redaction import _EVERY_TOOL

_BASE = datetime(2024, 1, 1, tzinfo=UTC)
_LLM = {
    "gen_ai.system": "openai",
    "gen_ai.request.model": "gpt-4",
    "gen_ai.response.model": "gpt-4",
    "gen_ai.usage.prompt_tokens": 10,
    "gen_ai.usage.completion_tokens": 5,
    "gen_ai.response.finish_reasons": ["stop"],
    "gen_ai.conversation.id": "c1",
    "gen_ai.prompt.name": "greet",
    "gen_ai.prompt.version": "2",
}


def _span(span_id: str, parent: str | None, name: str, offset_ms: int, **fields: Any) -> SpanData:
    return SpanData(
        trace_id="t0",
        span_id=span_id,
        parent_span_id=parent,
        operation_name=name,
        service_name="svc-a",
        start_time=_BASE + timedelta(milliseconds=offset_ms),
        duration_ms=100.0,
        status=fields.pop("status", "OK"),
        attributes=SpanAttributes.model_validate(fields.pop("attributes")),
        events=fields.pop("events", []),
    )


_SPANS = [
    _span(
        "s1",
        None,
        "chat",
        0,
        attributes=_LLM,
        events=[SpanEvent(name="gen_ai.content.prompt", timestamp=1, attributes={"body": "hi"})],
    ),
    _span("s2", "s1", "web_search", 10, attributes={"traceloop.span.kind": "tool"}),
    _span(
        "s3",
        "s1",
        "chat",
        20,
        status="ERROR",
        attributes={
            **_LLM,
            "error.type": "RateLimit",
            "exception.message": "boom",
            "exception.stacktrace": "trace",
        },
    ),
]
_TRACE = TraceData(
    trace_id="t0",
    spans=_SPANS,
    start_time=_BASE,
    duration_ms=200.0,
    service_name="svc-a",
    root_operation="chat",
    status="ERROR",
)


class _DataBackend(_Fake):
    """A loopback backend that answers every search with one trace holding LLM, tool and error spans."""

    async def search_traces(self, query: TraceQuery) -> list[TraceData]:
        self.seen_limits.append(query.limit)
        return [_TRACE]

    async def search_spans(self, query: SpanQuery) -> list[SpanData]:
        return list(_SPANS)

    async def get_trace(self, trace_id: str) -> TraceData:
        return _TRACE


@pytest.fixture(autouse=True)
def _reset_server_globals() -> Iterator[None]:
    server._backend = server._secondary_backend = server._config = None
    yield
    server._backend = server._secondary_backend = server._config = None


# -- the ceiling is not logged, whatever the limit ----------------------------------------


async def test_the_logs_are_the_same_below_at_and_above_the_ceiling(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ceiling = _ceiling()
    fake = _Fake(traces=ceiling + 200)
    await _install(fake)
    caplog.set_level(logging.DEBUG)

    def logged_by_one_call() -> list[tuple[str, int, str]]:
        records = [
            (r.name, r.levelno, r.getMessage())
            for r in caplog.records
            if r.name.startswith("opentelemetry_mcp")
        ]
        caplog.clear()
        return records

    caplog.clear()
    await _call("search_traces", {"limit": ceiling - 1})
    below = logged_by_one_call()
    await _call("search_traces", {"limit": ceiling})
    at = logged_by_one_call()
    await _call("search_traces", {"limit": 1000})
    above = logged_by_one_call()

    assert fake.seen_limits == [ceiling - 1, ceiling, ceiling]
    assert below == at == above


# -- every field name a response can carry ------------------------------------------------

_DATA_KEYED = frozenset(
    {"by_model", "by_service", "finish_reasons", "attributes", "extra_attributes"}
)


def _field_names(value: Any) -> set[str]:
    """Every field name in a response, except the keys of a map keyed by data (a model, a service,
    a finish reason or a stored attribute name): for those only what sits under each key counts."""
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(key)
            if key in _DATA_KEYED and isinstance(item, dict):
                for entry in item.values():
                    found |= _field_names(entry)
            else:
                found |= _field_names(item)
    elif isinstance(value, list):
        for item in value:
            found |= _field_names(item)
    return found


_USAGE = frozenset(
    {
        "by_model",
        "by_service",
        "completion_tokens",
        "cost_usd",
        "cost_usd_is_partial",
        "end_time",
        "filters",
        "gen_ai_request_model",
        "gen_ai_response_model",
        "gen_ai_system",
        "period",
        "prompt_tokens",
        "requests",
        "service_name",
        "start_time",
        "summary",
        "total_completion_tokens",
        "total_cost_usd",
        "total_prompt_tokens",
        "total_requests",
        "total_tokens",
    }
)
_PERCENTILES = frozenset({"mean", "median", "p50", "p95", "p99"})
_TRACE_ROW = frozenset(
    {
        "duration_ms",
        "has_errors",
        "operation_name",
        "service_name",
        "start_time",
        "status",
        "trace_id",
    }
)
_RANKED = frozenset({"change", "name", "percent_change"})

_EXPECTED: dict[str, frozenset[str]] = {
    "search_traces": _TRACE_ROW
    | {"count", "llm_span_count", "span_count", "total_tokens", "traces"},
    "search_spans_tool": frozenset(
        {
            "count",
            "duration_ms",
            "extra_attributes",
            "gen_ai_system",
            "is_llm_span",
            "operation_name",
            "parent_span_id",
            "service_name",
            "span_id",
            "spans",
            "start_time",
            "status",
            "total_tokens",
            "trace_id",
        }
    ),
    "get_trace": frozenset(
        {
            "attributes",
            "completion_tokens",
            "detail_level",
            "duration_ms",
            "events",
            "finish_reasons",
            "has_errors",
            "is_streaming",
            "llm_attributes",
            "llm_span_count",
            "llm_summary",
            "models_used",
            "name",
            "operation_name",
            "parent_span_id",
            "prompt_tokens",
            "request_model",
            "response_model",
            "root_operation",
            "service_name",
            "span_count",
            "span_id",
            "spans",
            "start_time",
            "status",
            "system",
            "timestamp",
            "total_tokens",
            "trace_id",
        }
    ),
    "triage_trace": frozenset(
        {
            "confidence",
            "critical_path",
            "duration_ms",
            "error_chain",
            "error_detail",
            "likely_root_cause",
            "operation_name",
            "reasoning",
            "self_time_ms",
            "service_name",
            "span_id",
            "status",
            "top_latency_contributors",
            "trace_id",
            "verdict",
        }
    ),
    "correlate_trace": frozenset(
        {
            "confidence",
            "correlation_method",
            "limitations",
            "matches",
            "primary_trace_id",
            "root_cause_consistent",
            "secondary_trace_id",
            "service_overlap",
        }
    ),
    "get_llm_usage": _USAGE,
    "list_services": frozenset({"count", "services"}),
    "find_errors": _TRACE_ROW
    | {
        "count",
        "error_message",
        "error_spans",
        "error_traces",
        "error_type",
        "is_llm_error",
        "llm_model",
        "llm_provider",
        "llm_span_count",
        "span_count",
        "span_id",
        "stack_trace",
        "total_tokens",
    },
    "list_llm_models": frozenset(
        {"count", "first_seen", "last_seen", "model", "models", "provider", "request_count"}
    ),
    "get_llm_model_stats": _PERCENTILES
    | {
        "completion",
        "duration_ms",
        "error_count",
        "error_rate",
        "finish_reasons",
        "model",
        "prompt",
        "request_count",
        "success_count",
        "success_rate",
        "tokens",
        "total",
    },
    "list_sessions": frozenset(
        {
            "conversation_id",
            "count",
            "first_seen",
            "last_seen",
            "message",
            "services",
            "sessions",
            "span_count",
            "total_tokens",
        }
    ),
    "get_session_stats": _PERCENTILES
    | {
        "completion",
        "conversation_id",
        "duration_ms",
        "error_count",
        "error_rate",
        "finish_reasons",
        "first_seen",
        "last_seen",
        "llm_request_count",
        "prompt",
        "services",
        "span_count",
        "success_count",
        "success_rate",
        "tokens",
        "total",
    },
    "compare_time_windows": _USAGE | {"change", "delta", "percent_change", "range_a", "range_b"},
    "investigate_cost_spike": _USAGE
    | _RANKED
    | {
        "auto_computed",
        "baseline",
        "baseline_cost_usd",
        "recent",
        "recent_cost_usd",
        "summary_delta",
        "top_model_contributors",
        "top_service_contributors",
    },
    "investigate_error_spike": _RANKED
    | {
        "auto_computed",
        "baseline",
        "baseline_count",
        "delta",
        "error_rate",
        "error_traces",
        "is_spike",
        "min_error_count_increase",
        "rate_multiplier_threshold",
        "recent",
        "recent_count",
        "sample_messages",
        "spike_criteria",
        "top_error_type_contributors",
        "top_model_contributors",
        "top_service_contributors",
        "total_traces",
    },
    "get_prompt_version_stats": _PERCENTILES
    | {
        "count",
        "duration_ms",
        "first_seen",
        "last_seen",
        "prompt_name",
        "prompt_version",
        "prompt_versions",
        "request_count",
        "total_tokens",
    },
    "get_llm_expensive_traces": frozenset(
        {
            "completion",
            "count",
            "duration_ms",
            "has_errors",
            "models",
            "operation_name",
            "prompt",
            "service_name",
            "start_time",
            "status",
            "tokens",
            "total",
            "trace_id",
            "traces",
        }
    ),
    "get_llm_slow_traces": _TRACE_ROW
    | {"count", "llm_span_count", "models", "total_tokens", "traces"},
    "list_llm_tools_tool": frozenset(
        {
            "count",
            "first_seen",
            "last_seen",
            "services",
            "tool_name",
            "tools",
            "total_calls",
            "usage_count",
        }
    ),
}


async def test_every_tool_returns_exactly_the_fields_the_snapshot_lists() -> None:
    await _install(_DataBackend(), _DataBackend(url="http://secondary.invalid"))
    async with Client(server.mcp) as client:
        registered = {tool.name for tool in await client.list_tools()}
        assert registered == set(_EXPECTED) == {name for name, _ in _EVERY_TOOL}
        for name, arguments in _EVERY_TOOL:
            body = _body(await client.call_tool(name, arguments, raise_on_error=False))
            names = _field_names(body)
            assert names == _EXPECTED[name], (
                f"{name}: new {sorted(names - _EXPECTED[name])}, "
                f"missing {sorted(_EXPECTED[name] - names)}. A field that names the backend, the "
                "applied limit, truncation or a redaction policy contradicts the README section "
                "'What responses report and what they do not'; update both if one is intended."
            )

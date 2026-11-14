"""README "What responses report and what they do not": windows, absent fields, redaction.

The default time window of each backend, the tools that echo a window, the absence of any
provenance-style field in every tool's response and output schema, the fact that a successful
response is not redacted, and the signals responses do carry - each read back out of README.md and
compared with the code or with a real tool call against a loopback fake.
"""

import inspect
import json
import re
import time
from collections.abc import Callable, Iterator
from datetime import timedelta
from typing import Any

import httpx
import pytest
from fastmcp import Client

import opentelemetry_mcp.server as server
from opentelemetry_mcp import observability
from opentelemetry_mcp.backends.datadog import DatadogBackend
from opentelemetry_mcp.backends.honeycomb import HoneycombBackend
from opentelemetry_mcp.backends.jaeger import JaegerBackend
from opentelemetry_mcp.backends.newrelic import NewRelicBackend
from opentelemetry_mcp.backends.sentry import SentryBackend
from opentelemetry_mcp.backends.traceloop import TraceloopBackend
from opentelemetry_mcp.backends.xray import XRayBackend
from opentelemetry_mcp.models import TraceQuery
from opentelemetry_mcp.tools.trace import _elide_large_attributes, _truncate_event_attributes
from tests.readme_response_support import (
    _FAKE_CREDENTIAL,
    _FAKE_HEADER_LINE,
    _FAKE_PAIR,
    _LOGGED,
    _SAYS_SO,
    _VALUE,
    _body,
    _call,
    _Fake,
    _install,
    _ints,
    _row,
    _section,
    _TempoFake,
    _tid,
)


@pytest.fixture(autouse=True)
def _reset_server_globals() -> Iterator[None]:
    server._backend = server._secondary_backend = server._config = None
    yield
    server._backend = server._secondary_backend = server._config = None


@pytest.fixture
def utc_local_timezone(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Tempo and Traceloop build their default window from the naive local clock."""
    with monkeypatch.context() as patch_env:
        patch_env.setenv("TZ", "UTC")
        time.tzset()
        yield
    time.tzset()


# -- time windows --------------------------------------------------------------------------

_LOOKBACK_BACKENDS: list[tuple[str, Callable[..., Any]]] = [
    ("Datadog", DatadogBackend._time_range),
    ("Sentry", SentryBackend._time_range),
    ("AWS X-Ray", XRayBackend._time_range),
    ("New Relic", NewRelicBackend._time_range),
    ("Honeycomb", HoneycombBackend._time_range),
]


@pytest.mark.parametrize(
    ("name", "time_range"), _LOOKBACK_BACKENDS, ids=[name for name, _ in _LOOKBACK_BACKENDS]
)
def test_default_lookback_days_match_the_readme(name: str, time_range: Callable[..., Any]) -> None:
    match = re.search(rf"\b{re.escape(name)} (\d+) d\b", _row("Default time window")[_VALUE])
    assert match, f"no '{name} <n> d' in the default time window row"
    lookback = inspect.signature(time_range).parameters["lookback"].default
    assert lookback == timedelta(days=int(match.group(1)))


async def test_tempo_default_window_matches_the_readme(utc_local_timezone: None) -> None:
    cell = _row("Default time window")[_VALUE]
    match = re.search(r"Tempo (\d+) h back to (\d+) h ahead", cell)
    assert match
    back, ahead = int(match.group(1)), int(match.group(2))
    fake = _TempoFake()

    await fake.backend.search_traces(TraceQuery(limit=10))

    params = fake.search_params[0]
    now = time.time()
    assert abs(int(params["start"]) - (now - back * 3600)) < 5
    assert abs(int(params["end"]) - (now + ahead * 3600)) < 5


async def test_traceloop_and_jaeger_default_windows_match_the_readme(
    utc_local_timezone: None,
) -> None:
    cell = _row("Default time window")[_VALUE]
    match = re.search(r"Traceloop (\d+) h", cell)
    assert match
    assert "Jaeger none" in cell
    sent: list[dict[str, Any]] = []

    def traceloop_handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"root_spans": {"data": []}})

    traceloop = TraceloopBackend(url="https://api.traceloop.com/v2", api_key=_FAKE_CREDENTIAL)
    traceloop._client = httpx.AsyncClient(
        base_url=traceloop.url, transport=httpx.MockTransport(traceloop_handler)
    )
    await traceloop.search_traces(TraceQuery(limit=10))
    window = sent[0]["to_timestamp_sec"] - sent[0]["from_timestamp_sec"]
    assert abs(window - int(match.group(1)) * 3600) < 5

    jaeger_params: list[dict[str, str]] = []

    def jaeger_handler(request: httpx.Request) -> httpx.Response:
        jaeger_params.append(dict(request.url.params))
        return httpx.Response(200, json={"data": []})

    jaeger = JaegerBackend(url="http://localhost:16686")
    jaeger._client = httpx.AsyncClient(
        base_url=jaeger.url, transport=httpx.MockTransport(jaeger_handler)
    )
    await jaeger.search_traces(TraceQuery(service_name="svc", limit=10))
    assert "start" not in jaeger_params[0]
    assert "end" not in jaeger_params[0]


async def test_only_the_documented_tools_echo_a_window() -> None:
    cell = _row("Default time window")[_SAYS_SO]
    assert cell.startswith("No")
    await _install(_Fake())
    recent = {"recent_start": "2024-01-01T01:00:00Z", "recent_end": "2024-01-01T02:00:00Z"}
    baseline = {"baseline_start": "2023-12-31T00:00:00Z", "baseline_end": "2023-12-31T01:00:00Z"}

    omitted = _body(await _call("get_llm_usage", {}))
    explicit = _body(
        await _call(
            "get_llm_usage",
            {"start_time": "2024-01-01T00:00:00Z", "end_time": "2024-01-01T01:00:00Z"},
        )
    )
    cost = _body(await _call("investigate_cost_spike", recent))
    errors = _body(await _call("investigate_error_spike", recent))
    given = _body(await _call("investigate_cost_spike", {**recent, **baseline}))

    assert omitted["period"] == {"start_time": None, "end_time": None}
    assert explicit["period"]["start_time"] == "2024-01-01T00:00:00+00:00"
    assert cost["baseline"]["period"]["start_time"] == "2024-01-01T00:00:00+00:00"
    assert cost["baseline"]["auto_computed"] is True
    assert given["baseline"]["auto_computed"] is False
    assert set(errors["baseline"]) == {
        "total_traces",
        "error_traces",
        "error_rate",
        "auto_computed",
    }
    for tool in ("get_llm_usage", "compare_time_windows", "investigate_cost_spike"):
        assert f"`{tool}`" in cell
    assert "`baseline.period`" in cell
    assert "only `baseline.auto_computed`" in cell


def test_window_and_redaction_rows_keep_every_number_in_a_pinned_place() -> None:
    window = _row("Default time window")
    # Tempo (back, ahead), Traceloop (hours) and five lookbacks, each pinned by a test above.
    assert len(_ints(window[_VALUE])) == 2 + 1 + len(_LOOKBACK_BACKENDS)
    assert _ints(" ".join([window[0], window[_SAYS_SO], window[_LOGGED]])) == []
    assert _ints(" ".join(_row("Redaction"))) == []


# -- no field promised by a provenance block exists ----------------------------------------

_ABSENT_FIELD_NAMES = {
    "backend",
    "limit",
    "requested_limit",
    "applied_limit",
    "truncated",
    "is_truncated",
    "partial",
    "is_partial",
    "provenance",
    "redaction",
    "redaction_policy",
    "resolved_start",
    "resolved_end",
}
_WINDOW = {"recent_start": "2024-01-01T00:00:00Z", "recent_end": "2024-01-01T01:00:00Z"}
_EVERY_TOOL: list[tuple[str, dict[str, Any]]] = [
    ("search_traces", {}),
    ("search_spans_tool", {}),
    ("get_trace", {"trace_id": "t0"}),
    ("triage_trace", {"trace_id": "t0"}),
    ("correlate_trace", {"trace_id": "t0"}),
    ("get_llm_usage", {}),
    ("list_services", {}),
    ("find_errors", {}),
    ("list_llm_models", {}),
    ("get_llm_model_stats", {"model_name": "gpt-4"}),
    ("list_sessions", {}),
    ("get_session_stats", {"conversation_id": "c1"}),
    ("compare_time_windows", {}),
    ("investigate_cost_spike", _WINDOW),
    ("investigate_error_spike", _WINDOW),
    ("get_prompt_version_stats", {}),
    ("get_llm_expensive_traces", {}),
    ("get_llm_slow_traces", {}),
    ("list_llm_tools_tool", {}),
]


def _key_names(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {name for item in value.values() for name in _key_names(item)}
    if isinstance(value, list):
        return {name for item in value for name in _key_names(item)}
    return set()


def _schema_property_names(schema: Any) -> set[str]:
    if isinstance(schema, dict):
        properties = schema.get("properties")
        own = set(properties) if isinstance(properties, dict) else set()
        return own | {name for item in schema.values() for name in _schema_property_names(item)}
    if isinstance(schema, list):
        return {name for item in schema for name in _schema_property_names(item)}
    return set()


async def test_no_tool_response_or_output_schema_carries_a_provenance_field() -> None:
    opening = _section().split("|", 1)[0]
    for promised in ("backend", "`limit`", "truncation", "redaction policy"):
        assert promised in opening, promised
    await _install(_Fake(traces=2), _Fake(url="http://secondary.invalid", fail_get_trace=True))
    windows: set[str] = set()
    async with Client(server.mcp) as client:
        tools = await client.list_tools()
        assert len(tools) == len(_EVERY_TOOL)
        schemas = _schema_property_names([tool.output_schema for tool in tools])
        assert schemas.isdisjoint(_ABSENT_FIELD_NAMES)
        for name, arguments in _EVERY_TOOL:
            body = _body(await client.call_tool(name, arguments, raise_on_error=False))
            names = _key_names(body)
            assert names.isdisjoint(_ABSENT_FIELD_NAMES), (name, names & _ABSENT_FIELD_NAMES)
            if "period" in names:
                windows.add(name)
    assert windows == {"get_llm_usage", "compare_time_windows", "investigate_cost_spike"}


# -- redaction -----------------------------------------------------------------------------


async def test_a_successful_response_is_not_redacted_but_the_span_redactor_would_mask_it() -> None:
    row = _row("Redaction")
    assert "not redacted" in row[0]
    assert "`MCP_INCLUDE_ARGS_IN_SPANS`" in row[_LOGGED]
    extra = [
        {"key": "http.request.header", "value": {"stringValue": _FAKE_HEADER_LINE}},
        {"key": "exception.message", "value": {"stringValue": f"call failed: {_FAKE_PAIR}"}},
    ]
    fake = _TempoFake(candidates=1, error=True, extra=extra)
    fake.install()

    trace = json.dumps(_body(await _call("get_trace", {"trace_id": _tid(0)})))
    errors = _body(await _call("find_errors", {}))

    assert _FAKE_HEADER_LINE in trace
    assert _FAKE_PAIR in trace
    assert errors["count"] == 1
    assert _FAKE_PAIR in json.dumps(errors)
    assert _FAKE_HEADER_LINE not in observability._redact_secrets(_FAKE_HEADER_LINE)
    assert _FAKE_PAIR not in observability._redact_secrets(_FAKE_PAIR)


# -- what responses do report --------------------------------------------------------------


async def test_the_signals_the_readme_says_exist_do_exist() -> None:
    reported = _section().split("What responses do report", 1)[1]
    for phrase in (
        "`count`",
        "`cost_usd_is_partial`",
        "`limitations`",
        "`detail_level`",
        "<omitted N item(s)",
        "`baseline.auto_computed`",
    ):
        assert phrase in reported, phrase
    await _install(_Fake(traces=1))

    usage = _body(await _call("get_llm_usage", {}))
    summary = _body(await _call("get_trace", {"trace_id": "t0", "detail_level": "summary"}))

    assert usage["summary"]["cost_usd_is_partial"] is False
    assert summary["detail_level"] == "summary"
    elided = _elide_large_attributes({"gen_ai.input.messages": ["a", "b"]})
    assert str(elided["gen_ai.input.messages"]).startswith("<omitted 2 item(s)")
    cut = _truncate_event_attributes([{"attributes": {"body": "x" * 600}}])
    assert cut[0]["attributes"]["body"].endswith("...")

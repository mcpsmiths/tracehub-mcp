"""README "What responses report and what they do not": shortened fields and elision markers.

Responses are not scrubbed, but a few tools do shorten what the backend stored: find_errors (and the
error_detail triage_trace attaches at detail_level "full") cut a stack trace, get_trace at
detail_level "summary" replaces large gen_ai fields and cuts long event values, and get_trace at
either detail level gives each LLM span a prompt_preview and completion_preview cut to a shorter
length. triage_trace marks the middle of a long chain it drops. Each claim is read back out of
README.md and compared with the constant it describes or with a real tool call against a loopback
Tempo fake.
"""

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

import opentelemetry_mcp.server as server
from opentelemetry_mcp.attributes import SpanEvent
from opentelemetry_mcp.constants import GenAI
from opentelemetry_mcp.models import TraceData
from opentelemetry_mcp.tools import triage
from opentelemetry_mcp.tools.trace import _EVENT_ATTRIBUTE_TRUNCATE_AT, _truncate_event_attributes
from tests.readme_response_support import (
    _LOGGED,
    _README,
    _SAYS_SO,
    _VALUE,
    _body,
    _call,
    _Fake,
    _install,
    _ints,
    _row,
    _section,
    _tempo_trace,
    _TempoFake,
    _tid,
    _trace,
)

_STACK_KEY = "exception.stacktrace"


@pytest.fixture(autouse=True)
def _reset_server_globals() -> Iterator[None]:
    server._backend = server._secondary_backend = server._config = None
    yield
    server._backend = server._secondary_backend = server._config = None


def _text(key: str, value: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": value}}


def _cut_at() -> int:
    return _ints(_row("Shortened fields")[_VALUE])[0]


async def _first_error_span(stack_length: int) -> dict[str, Any]:
    fake = _TempoFake(candidates=1, error=True, extra=[_text(_STACK_KEY, "S" * stack_length)])
    fake.install()
    errors = _body(await _call("find_errors", {}))
    span: dict[str, Any] = errors["error_traces"][0]["error_spans"][0]
    return span


# -- the Shortened fields row ----------------------------------------------------------------


def _preview_cut() -> int:
    return _ints(_row("Shortened fields")[_VALUE])[1]


def test_the_shortened_fields_row_states_the_stack_trace_cut_then_the_preview_cut() -> None:
    row = _row("Shortened fields")
    numbers = _ints(row[_VALUE])
    stack_and_events, previews = row[_VALUE].split(";")
    assert len(numbers) == 2
    assert numbers[0] == _EVENT_ATTRIBUTE_TRUNCATE_AT
    assert _ints(" ".join(row)) == numbers
    assert "stack traces and event values" in stack_and_events
    assert "previews" in previews
    assert row[_LOGGED].startswith("No")


@pytest.mark.parametrize("extra", [0, 1])
async def test_find_errors_cuts_a_stack_trace_only_past_the_documented_length(extra: int) -> None:
    cut_at = _cut_at()

    span = await _first_error_span(cut_at + extra)

    stack = span["stack_trace"]
    assert stack == "S" * cut_at + ("..." if extra else "")


async def test_find_errors_and_triage_full_cut_the_stack_trace_but_get_trace_does_not() -> None:
    cut_at = _cut_at()
    whole = cut_at * 6
    row = _row("Shortened fields")
    for name in ("`find_errors`", "`triage_trace`", "`get_trace`"):
        assert name in row[0], name
    fake = _TempoFake(candidates=1, error=True, extra=[_text(_STACK_KEY, "S" * whole)])
    fake.install()

    trace = _body(await _call("get_trace", {"trace_id": _tid(0)}))
    errors = _body(await _call("find_errors", {}))
    full = _body(await _call("triage_trace", {"trace_id": _tid(0), "detail_level": "full"}))
    summary = _body(await _call("triage_trace", {"trace_id": _tid(0)}))

    assert trace["spans"][0]["attributes"][_STACK_KEY] == "S" * whole
    found = errors["error_traces"][0]["error_spans"][0]["stack_trace"]
    detail = full["verdict"]["likely_root_cause"]["error_detail"]["stack_trace"]
    assert found == detail == "S" * cut_at + "..."
    assert summary["verdict"]["likely_root_cause"]["error_detail"] is None


async def test_summary_get_trace_replaces_the_named_fields_and_full_returns_them() -> None:
    keys = (
        GenAI.INPUT_MESSAGES,
        GenAI.OUTPUT_MESSAGES,
        GenAI.SYSTEM_INSTRUCTIONS,
        GenAI.RETRIEVAL_DOCUMENTS,
    )
    row = _row("Shortened fields")
    for key in keys:
        assert f"`{key}`" in row[0], key
    fake = _TempoFake(candidates=1, extra=[_text(key, f"value of {key}") for key in keys])
    fake.install()

    full = _body(await _call("get_trace", {"trace_id": _tid(0)}))
    summary = _body(await _call("get_trace", {"trace_id": _tid(0), "detail_level": "summary"}))

    for key in keys:
        kept = json.dumps(full["spans"][0]["attributes"][key])
        assert f"value of {key}" in kept and "<omitted" not in kept, key
        omitted = summary["spans"][0]["attributes"][key]
        assert omitted.startswith("<omitted ") and "item(s)" in omitted, omitted
    assert "<omitted N item(s) ...>" in row[_SAYS_SO]
    assert "`detail_level`" in row[_SAYS_SO]


class _EventBackend(_Fake):
    """A loopback backend whose one span carries an event whose body is `length` characters."""

    def __init__(self, length: int) -> None:
        super().__init__()
        self.length = length

    async def get_trace(self, trace_id: str) -> TraceData:
        event = SpanEvent(name="e", timestamp=1, attributes={"body": "E" * self.length})
        span = _trace(trace_id).spans[0].model_copy(update={"events": [event]})
        return _trace(trace_id).model_copy(update={"spans": [span]})


@pytest.mark.parametrize("extra", [0, 1])
async def test_get_trace_cuts_an_event_value_in_summary_mode_only_past_the_documented_length(
    extra: int,
) -> None:
    cut_at = _cut_at()
    await _install(_EventBackend(cut_at + extra))

    full = _body(await _call("get_trace", {"trace_id": "t0"}))
    summary = _body(await _call("get_trace", {"trace_id": "t0", "detail_level": "summary"}))

    whole = "E" * (cut_at + extra)
    assert full["spans"][0]["events"][0]["attributes"]["body"] == whole
    cut = summary["spans"][0]["events"][0]["attributes"]["body"]
    assert cut == "E" * cut_at + ("..." if extra else "")


@pytest.mark.parametrize("extra", [0, 1])
def test_summary_mode_cuts_an_event_value_only_past_the_documented_length(extra: int) -> None:
    cut_at = _cut_at()
    events = [{"name": "e", "attributes": {"body": "E" * (cut_at + extra), "n": 7}}]

    cut = _truncate_event_attributes(events)

    assert cut[0]["attributes"] == {"body": "E" * cut_at + ("..." if extra else ""), "n": 7}


def _event(name: str, key: str, value: str) -> dict[str, Any]:
    return {"name": name, "timeUnixNano": "1000000001", "attributes": [_text(key, value)]}


class _PreviewFake(_TempoFake):
    """The real TempoBackend serving one LLM span whose prompt and completion events are `length` long."""

    def __init__(self, length: int) -> None:
        self.length = length
        super().__init__(candidates=1, llm=True)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/search":
            return super()._handle(request)
        trace_id = request.url.path.rsplit("/", 1)[-1]
        payload = _tempo_trace(trace_id, llm=True)
        span = payload["batches"][0]["scopeSpans"][0]["spans"][0]
        span["events"] = [
            _event(GenAI.EVENT_CONTENT_PROMPT, GenAI.EVENT_PROMPT_CONTENT, "P" * self.length),
            _event(
                GenAI.EVENT_CONTENT_COMPLETION, GenAI.EVENT_COMPLETION_CONTENT, "C" * self.length
            ),
        ]
        return httpx.Response(200, json=payload)


def test_the_shortened_fields_row_names_both_previews_and_the_events_they_cut() -> None:
    cell = _row("Shortened fields")[0]
    for name in (
        "`llm_attributes.prompt_preview`",
        "`llm_attributes.completion_preview`",
        f"`{GenAI.EVENT_CONTENT_PROMPT}`",
        f"`{GenAI.EVENT_CONTENT_COMPLETION}`",
    ):
        assert name in cell, name
    assert "At both detail levels `get_trace` also gives each LLM span" in cell


@pytest.mark.parametrize("detail_level", ["full", "summary"])
@pytest.mark.parametrize("extra", [0, 1])
async def test_get_trace_cuts_each_preview_only_past_the_documented_length_at_both_levels(
    extra: int, detail_level: str
) -> None:
    cut_at = _preview_cut()
    length = cut_at + extra
    assert length < _EVENT_ATTRIBUTE_TRUNCATE_AT, "the event itself must not be cut by the 500 rule"
    _PreviewFake(length).install()

    body = _body(await _call("get_trace", {"trace_id": _tid(0), "detail_level": detail_level}))

    span = body["spans"][0]
    previews = span["llm_attributes"]
    suffix = "..." if extra else ""
    assert previews["prompt_preview"] == "P" * cut_at + suffix
    assert previews["completion_preview"] == "C" * cut_at + suffix
    events = {event["name"]: event["attributes"] for event in span["events"]}
    assert events[GenAI.EVENT_CONTENT_PROMPT][GenAI.EVENT_PROMPT_CONTENT] == "P" * length
    assert events[GenAI.EVENT_CONTENT_COMPLETION][GenAI.EVENT_COMPLETION_CONTENT] == "C" * length


def test_the_redaction_row_defers_to_the_shortened_fields_row() -> None:
    redaction = _row("Redaction")[0]
    assert "not redacted" in redaction
    assert "as the backend stored them, except where a tool shortens a field" in redaction
    assert "(see **Shortened fields**)" in redaction


def test_the_triage_paragraph_no_longer_calls_the_error_detail_raw() -> None:
    readme = _README.read_text(encoding="utf-8")
    assert "raw error detail" not in readme


# -- triage_trace marks its own elision ---------------------------------------------------------


def _chain_payload(trace_id: str, depth: int) -> dict[str, Any]:
    spans = [
        {
            "traceId": trace_id,
            "spanId": f"s{index:04d}",
            "parentSpanId": f"s{index - 1:04d}" if index else "",
            "name": f"op{index}",
            "startTimeUnixNano": str(1_000_000_000 + index),
            "endTimeUnixNano": str(2_000_000_000 - index),
            "status": {"code": 2 if index == depth - 1 else 1},
            "attributes": [],
        }
        for index in range(depth)
    ]
    resource = {"attributes": [{"key": "service.name", "value": {"stringValue": "checkout"}}]}
    return {"batches": [{"resource": resource, "scopeSpans": [{"spans": spans}]}]}


class _ChainFake(_TempoFake):
    """The real TempoBackend serving one trace that is a single parent chain `depth` spans deep."""

    def __init__(self, depth: int) -> None:
        self.depth = depth
        super().__init__(candidates=1)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/search":
            return super()._handle(request)
        trace_id = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, json=_chain_payload(trace_id, self.depth))


async def test_triage_marks_the_middle_of_a_long_chain_and_names_it_in_the_reasoning() -> None:
    reported = _section().split("What responses do report", 1)[1]
    for phrase in ("`triage_trace`", "`[K intermediate span(s) elided]`", "`verdict.reasoning`"):
        assert phrase in reported, phrase
    kept = triage._CHAIN_HEAD + triage._SUMMARY_CHAIN_TAIL
    depth = kept + 15
    _ChainFake(depth).install()

    result = _body(await _call("triage_trace", {"trace_id": _tid(0)}))

    for field in ("critical_path", "error_chain"):
        entries = result[field]
        markers = [entry for entry in entries if entry["span_id"] == triage._ELIDED_SPAN_ID]
        assert len(entries) == kept + 1, field
        assert [m["operation_name"] for m in markers] == ["[15 intermediate span(s) elided]"]
        assert (
            f"{field} lists the first {triage._CHAIN_HEAD} and last"
            in result["verdict"]["reasoning"]
        )
    assert (
        f"one elision entry (span_id '{triage._ELIDED_SPAN_ID}')" in result["verdict"]["reasoning"]
    )


async def test_a_chain_that_fits_carries_no_marker() -> None:
    _ChainFake(triage._CHAIN_HEAD + triage._SUMMARY_CHAIN_TAIL).install()

    result = _body(await _call("triage_trace", {"trace_id": _tid(0)}))

    for field in ("critical_path", "error_chain"):
        assert all(entry["span_id"] != triage._ELIDED_SPAN_ID for entry in result[field])
    assert "elision entry" not in result["verdict"]["reasoning"]

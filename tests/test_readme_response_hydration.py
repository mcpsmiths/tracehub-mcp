"""README "What responses report and what they do not": backends that do not hydrate, and Tempo.

Two claims of the Hydration cap and Failed hydration rows that prose alone cannot keep true:

* "Jaeger and Traceloop query once and do not hydrate". A search_traces call against the real
  JaegerBackend and TraceloopBackend, each terminating in an httpx.MockTransport, sends exactly one
  request, never fetches a trace by ID and is not held to the hydration cap.
* Tempo raises when, and only when, it tried at least one per-trace fetch and every one of them
  raised. A trace that comes back with no spans (or that cannot be parsed) is left out with no error,
  so a search can return count 0 although some fetches raised.

Every README number and sentence is read back out of README.md; no real backend is ever contacted.
"""

import json
import logging
import re
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

import opentelemetry_mcp.server as server
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.backends.jaeger import JaegerBackend
from opentelemetry_mcp.backends.traceloop import TraceloopBackend
from tests.readme_response_support import (
    _FAKE_CREDENTIAL,
    _LOGGED,
    _SAYS_SO,
    _body,
    _call,
    _ceiling,
    _config,
    _hydration_cap,
    _row,
    _tempo_trace,
    _TempoFake,
    _tid,
    _warnings,
)

_LIMIT_WARNING = "Limiting trace fetch to"
_NO_HYDRATION = re.compile(r"(?P<names>[A-Z]\w+ and [A-Z]\w+) query once and do not hydrate")


@pytest.fixture(autouse=True)
def _reset_server_globals() -> Iterator[None]:
    server._backend = server._secondary_backend = server._config = None
    yield
    server._backend = server._secondary_backend = server._config = None


# -- loopback fakes ------------------------------------------------------------------------


class _Loopback:
    """A real backend whose HTTP client ends in an httpx.MockTransport that records every request."""

    def __init__(
        self, backend: BaseBackend, respond: Callable[[httpx.Request], httpx.Response]
    ) -> None:
        self.backend = backend
        self.requests: list[httpx.Request] = []
        self._respond = respond
        backend._client = httpx.AsyncClient(
            base_url=backend.url, transport=httpx.MockTransport(self._handle)
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._respond(request)

    def install(self) -> None:
        config = _config()
        server._config = config
        self.backend.max_query_limit = config.max_traces_per_query
        server._backend = self.backend

    @property
    def sent(self) -> list[tuple[str, str]]:
        return [(request.method, request.url.path) for request in self.requests]


def _jaeger_trace(index: int) -> dict[str, Any]:
    trace_id = _tid(index)
    span = {
        "traceID": trace_id,
        "spanID": f"s{index:04x}",
        "operationName": "op",
        "startTime": 1_700_000_000_000_000,
        "duration": 5_000,
        "processID": "p1",
        "tags": [],
        "references": [],
        "logs": [],
    }
    return {"traceID": trace_id, "spans": [span], "processes": {"p1": {"serviceName": "svc"}}}


def _jaeger(candidates: int) -> _Loopback:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/traces":
            return httpx.Response(200, json={"data": [_jaeger_trace(i) for i in range(candidates)]})
        return httpx.Response(404, json={"error": "unexpected request"})

    return _Loopback(JaegerBackend(url="http://localhost:16686"), respond)


def _traceloop_root_span(index: int) -> dict[str, Any]:
    return {
        "trace_id": _tid(index),
        "span_id": f"s{index:04x}",
        "span_name": "op",
        "service_name": "svc",
        "timestamp": 1_700_000_000_000,
        "duration": 5.0,
        "status_code": "OK",
        "span_attributes": {},
    }


def _traceloop(candidates: int) -> _Loopback:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/projects/default/traces/root-spans":
            rows = [_traceloop_root_span(i) for i in range(candidates)]
            return httpx.Response(200, json={"root_spans": {"data": rows}})
        return httpx.Response(404, json={"error": "unexpected request"})

    backend = TraceloopBackend(url="https://api.traceloop.com", api_key=_FAKE_CREDENTIAL)
    return _Loopback(backend, respond)


# -- "Jaeger and Traceloop query once and do not hydrate" -----------------------------------

_SEARCH_REQUEST = {
    "jaeger": ("GET", "/api/traces"),
    "traceloop": ("POST", "/v2/projects/default/traces/root-spans"),
}
_FAKES: dict[str, Callable[[int], _Loopback]] = {"jaeger": _jaeger, "traceloop": _traceloop}


def _named_non_hydrating_backends() -> list[str]:
    """The backend names the README sentence says query once, lower-cased."""
    cell = _row("Hydration cap")[0]
    match = _NO_HYDRATION.search(cell)
    assert match, "the Hydration cap row no longer says '<A> and <B> query once and do not hydrate'"
    return [name.lower() for name in match["names"].split(" and ")]


def test_the_hydration_cap_row_names_jaeger_and_traceloop_as_the_backends_that_query_once() -> None:
    cell = _row("Hydration cap")[0]

    assert _named_non_hydrating_backends() == ["jaeger", "traceloop"]
    assert "Jaeger and Traceloop query once and do not hydrate" in cell
    hydrating = cell.split(";", 1)[0]
    for display in ("Tempo", "Datadog", "Sentry", "AWS X-Ray", "New Relic", "Honeycomb"):
        assert display in hydrating, display
    for display in ("Jaeger", "Traceloop"):
        assert display not in hydrating, display


def test_the_readme_sentence_and_the_fakes_cover_the_same_backends() -> None:
    assert sorted(_named_non_hydrating_backends()) == sorted(_FAKES) == sorted(_SEARCH_REQUEST)


@pytest.mark.parametrize("backend_name", sorted(_FAKES))
async def test_a_non_hydrating_backend_sends_one_search_and_fetches_no_trace_by_id(
    backend_name: str, caplog: pytest.LogCaptureFixture
) -> None:
    candidates = _hydration_cap() + 10
    loopback = _FAKES[backend_name](candidates)
    loopback.install()
    caplog.set_level(logging.WARNING)

    body = _body(await _call("search_traces", {"service_name": "svc", "limit": candidates + 5}))

    assert loopback.sent == [_SEARCH_REQUEST[backend_name]]
    assert set(body) == {"count", "traces"}
    assert body["count"] == len(body["traces"]) == candidates
    assert [w for w in _warnings(caplog) if _LIMIT_WARNING in w] == []


# -- Tempo raises when, and only when, every attempted fetch raised -------------------------

_HTTP_ERROR, _NOT_JSON, _NO_BATCHES, _NO_SPANS, _NOT_A_TRACE, _OK = (
    "http_error",
    "not_json",
    "no_batches",
    "no_spans",
    "not_a_trace",
    "ok",
)


class _ShapedTempo(_TempoFake):
    """The real TempoBackend; shapes[i] decides how the per-trace fetch of candidate i answers.

    `unusable` extra search rows carry an empty traceID, which the backend skips without a fetch.
    """

    def __init__(self, shapes: list[str], *, unusable: int = 0) -> None:
        self.shapes, self.unusable = shapes, unusable
        super().__init__(candidates=len(shapes))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/search":
            rows = [{"traceID": _tid(i)} for i in range(len(self.shapes))]
            rows += [{"traceID": ""}] * self.unusable
            return httpx.Response(200, json={"traces": rows})
        trace_id = request.url.path.rsplit("/", 1)[-1]
        self.fetched.append(trace_id)
        shape = self.shapes[int(trace_id, 16)]
        if shape == _HTTP_ERROR:
            return httpx.Response(500, json={"error": "boom"})
        if shape == _NOT_JSON:
            return httpx.Response(200, content=b"this is not json")
        if shape == _NO_BATCHES:
            return httpx.Response(200, json={"batches": []})
        if shape == _NO_SPANS:
            return httpx.Response(200, json={"batches": [{"resource": {}, "scopeSpans": []}]})
        if shape == _NOT_A_TRACE:
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=_tempo_trace(trace_id))


# (shapes, unusable search rows, expected count; None = the search raises)
_CASES = [
    pytest.param([_HTTP_ERROR], 0, None, id="one-http-error-raises"),
    pytest.param([_HTTP_ERROR, _HTTP_ERROR], 0, None, id="all-http-errors-raise"),
    pytest.param([_NOT_JSON, _HTTP_ERROR], 0, None, id="not-json-and-http-error-raise"),
    pytest.param([_HTTP_ERROR], 2, None, id="unusable-rows-are-not-fetches"),
    pytest.param([_NO_BATCHES], 0, 0, id="no-batches-is-count-zero"),
    pytest.param([_NO_SPANS], 0, 0, id="no-spans-is-count-zero"),
    pytest.param([_NOT_A_TRACE], 0, 0, id="unparseable-is-count-zero"),
    pytest.param([_NO_BATCHES, _NO_SPANS, _NOT_A_TRACE], 0, 0, id="all-empty-is-count-zero"),
    pytest.param([_HTTP_ERROR, _NO_BATCHES], 0, 0, id="raise-plus-empty-is-count-zero"),
    pytest.param([_NO_SPANS, _HTTP_ERROR, _NOT_A_TRACE], 0, 0, id="mixed-is-count-zero"),
    pytest.param([_HTTP_ERROR, _OK], 0, 1, id="raise-plus-ok-keeps-the-ok"),
    pytest.param([_NO_BATCHES, _OK], 0, 1, id="empty-plus-ok-keeps-the-ok"),
    pytest.param([], 1, 0, id="only-an-unusable-row-sends-no-fetch"),
    pytest.param([], 0, 0, id="no-rows-at-all"),
]


@pytest.mark.parametrize(("shapes", "unusable", "expected"), _CASES)
async def test_tempo_raises_only_when_every_attempted_fetch_raised(
    shapes: list[str], unusable: int, expected: int | None
) -> None:
    fake = _ShapedTempo(shapes, unusable=unusable)
    fake.install()

    result = await _call("search_traces", {})

    assert len(fake.fetched) == len(shapes)
    if expected is None:
        assert result.is_error
        assert "per-trace fetch" in str(result.content)
        return
    assert not result.is_error
    body = _body(result)
    assert body["count"] == len(body["traces"]) == expected


def test_the_failed_hydration_row_states_the_exact_condition_under_which_tempo_raises() -> None:
    row = _row("Failed hydration")
    says = row[_SAYS_SO]

    assert (
        "Tempo raises an error when, and only when, it tried at least one fetch and every one "
        "of them raised" in says
    )
    assert "for example an HTTP error status, a transport error or a body that is not JSON" in says
    assert (
        "a trace that comes back with no spans or that cannot be parsed is left out without "
        "an error" in says
    )
    assert "`count: 0`" in says and "Tempo" in says
    assert "even though some fetches raised" in says


@pytest.mark.parametrize(
    ("shape", "level", "message"),
    [
        (_NO_BATCHES, logging.WARNING, "No batches in trace"),
        (_NO_SPANS, logging.WARNING, "No valid spans found"),
        (_NOT_A_TRACE, logging.ERROR, "Error parsing Tempo trace"),
    ],
)
async def test_an_empty_or_unparseable_tempo_trace_is_logged_without_the_failed_fetch_line(
    shape: str, level: int, message: str, caplog: pytest.LogCaptureFixture
) -> None:
    logged = _row("Failed hydration")[_LOGGED]
    assert message in logged
    fake = _ShapedTempo([shape])
    fake.install()
    caplog.set_level(logging.WARNING)

    result = _body(await _call("search_traces", {}))

    assert result == {"count": 0, "traces": []}
    records = [r for r in caplog.records if r.name.startswith("opentelemetry_mcp.backends.tempo")]
    assert [r.getMessage() for r in records if r.levelno == level][0].startswith(message)
    assert [r for r in records if r.getMessage().startswith("Failed to fetch trace")] == []
    assert _tid(0) not in " ".join(r.getMessage() for r in records)


# -- the cap holds on the six that hydrate; Jaeger and Traceloop reach the clamped limit ----


def _limited_jaeger(candidates: int) -> _Loopback:
    """A Jaeger fake that, like the real API, answers at most the `limit` it was sent."""

    def respond(request: httpx.Request) -> httpx.Response:
        limit = int(request.url.params["limit"])
        rows = [_jaeger_trace(i) for i in range(min(candidates, limit))]
        return httpx.Response(200, json={"data": rows})

    return _Loopback(JaegerBackend(url="http://localhost:16686"), respond)


def _limited_traceloop(candidates: int) -> _Loopback:
    """A Traceloop fake that answers at most the `limit` in the POST body it was sent."""

    def respond(request: httpx.Request) -> httpx.Response:
        limit = int(json.loads(request.content)["limit"])
        rows = [_traceloop_root_span(i) for i in range(min(candidates, limit))]
        return httpx.Response(200, json={"root_spans": {"data": rows}})

    backend = TraceloopBackend(url="https://api.traceloop.com", api_key=_FAKE_CREDENTIAL)
    return _Loopback(backend, respond)


_LIMITED: dict[str, Callable[[int], _Loopback]] = {
    "jaeger": _limited_jaeger,
    "traceloop": _limited_traceloop,
}


def _sent_limit(request: httpx.Request) -> int:
    if request.method == "GET":
        return int(request.url.params["limit"])
    return int(json.loads(request.content)["limit"])


def test_the_hydration_cap_row_scopes_the_cap_to_those_six() -> None:
    says = _row("Hydration cap")[_SAYS_SO]

    assert says.startswith("No. On those six, `count` never exceeds the cap whatever `limit` was")
    assert (
        "Jaeger and Traceloop apply no hydration cap, so their `count` can reach the `limit` "
        "as lowered to the ceiling" in says
    )
    assert "`get_llm_usage`" in says


@pytest.mark.parametrize("backend_name", sorted(_LIMITED))
@pytest.mark.parametrize("over_the_ceiling", [False, True])
async def test_a_non_hydrating_backend_returns_up_to_the_clamped_limit_not_the_cap(
    backend_name: str, over_the_ceiling: bool
) -> None:
    cap, ceiling = _hydration_cap(), _ceiling()
    asked = ceiling + 500 if over_the_ceiling else cap + 70
    expected = ceiling if over_the_ceiling else cap + 70
    loopback = _LIMITED[backend_name](ceiling + 200)
    loopback.install()

    body = _body(await _call("search_traces", {"service_name": "svc", "limit": asked}))

    assert [_sent_limit(request) for request in loopback.requests] == [expected]
    assert body["count"] == len(body["traces"]) == expected
    assert expected > cap

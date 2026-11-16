"""README "What responses report and what they do not": the bounds and failures a response hides.

Search caps (page and row), Honeycomb's row cap, the hydration cap on every backend that has one,
and failed hydration on Tempo, Honeycomb and AWS X-Ray - each number and claim is read back out of
README.md and compared with the constant it describes or with what a real backend object does
against a loopback fake. No real backend is ever contacted.
"""

import logging
from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import opentelemetry_mcp.server as server
from opentelemetry_mcp.backends import datadog, honeycomb, newrelic, sentry, xray
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.backends.datadog import DatadogBackend
from opentelemetry_mcp.backends.honeycomb import HoneycombBackend
from opentelemetry_mcp.backends.newrelic import NewRelicBackend
from opentelemetry_mcp.backends.sentry import SentryBackend
from opentelemetry_mcp.backends.xray import XRayBackend
from opentelemetry_mcp.models import TraceData, TraceQuery
from tests.readme_response_support import (
    _FAKE_CREDENTIAL,
    _LOGGED,
    _SAYS_SO,
    _VALUE,
    _body,
    _call,
    _hydration_cap,
    _install,
    _ints,
    _row,
    _TempoFake,
    _tid,
    _trace,
    _warnings,
)

_END = datetime(2024, 1, 2, tzinfo=UTC)
_START = _END - timedelta(days=1)
_LIMIT_WARNING = "Limiting trace fetch to {cap} out of {found} results to avoid excessive API calls"


@pytest.fixture(autouse=True)
def _reset_server_globals() -> Iterator[None]:
    server._backend = server._secondary_backend = server._config = None
    yield
    server._backend = server._secondary_backend = server._config = None


def _datadog() -> DatadogBackend:
    return DatadogBackend(
        url="https://api.datadoghq.com", api_key=_FAKE_CREDENTIAL, app_key=_FAKE_CREDENTIAL
    )


def _sentry() -> SentryBackend:
    return SentryBackend(url="https://sentry.io", api_key=_FAKE_CREDENTIAL, org_slug="org")


def _newrelic() -> NewRelicBackend:
    return NewRelicBackend(
        url="https://api.newrelic.com/graphql", api_key=_FAKE_CREDENTIAL, account_id="123456"
    )


def _honeycomb() -> HoneycombBackend:
    return HoneycombBackend(url="https://api.honeycomb.io", api_key=_FAKE_CREDENTIAL, dataset="ds")


def _xray() -> XRayBackend:
    return XRayBackend(url="https://xray.us-east-1.amazonaws.com", aws_region="us-east-1")


# -- the Search caps row -------------------------------------------------------------------


def _search_caps() -> tuple[int, int]:
    pages, rows = _ints(_row("Search caps")[_VALUE])
    return pages, rows


def test_the_search_caps_row_states_the_code_constants_and_nothing_else() -> None:
    pages, rows = _search_caps()
    assert pages == datadog._MAX_SEARCH_PAGES == sentry._MAX_SEARCH_PAGES == xray._MAX_SEARCH_PAGES
    assert rows == newrelic._MAX_NRQL_LIMIT
    assert set(_ints(" ".join(_row("Search caps")))) == {pages, rows}


def _paged_http_client(
    backend: BaseBackend, responder: Callable[[int], httpx.Response]
) -> list[int]:
    """Serve `responder(n)` for the n-th request and return the (growing) list of requests."""
    served: list[int] = []

    def handle(request: httpx.Request) -> httpx.Response:
        served.append(len(served))
        return responder(len(served))

    backend._client = httpx.AsyncClient(base_url=backend.url, transport=httpx.MockTransport(handle))
    return served


async def _datadog_pages() -> int:
    backend = _datadog()

    def page(number: int) -> httpx.Response:
        item = {"attributes": {"trace_id": f"t{number}"}}
        return httpx.Response(200, json={"data": [item], "meta": {"page": {"after": "next"}}})

    served = _paged_http_client(backend, page)
    await backend._search_spans_raw("*", _START, _END, limit=1_000_000)
    return len(served)


async def _sentry_pages() -> int:
    backend = _sentry()
    link = '<https://sentry.io/next>; rel="next"; results="true"; cursor="c"'

    def page(number: int) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"trace": f"t{number}"}]}, headers={"Link": link})

    served = _paged_http_client(backend, page)
    await backend._search_events_raw("", ["trace"], _START, _END, limit=1_000_000)
    return len(served)


async def _xray_pages() -> int:
    backend = _xray()
    call = AsyncMock(return_value={"TraceSummaries": [{"Id": "1-0-a"}], "NextToken": "next"})
    with patch.object(backend, "_call", call):
        await backend._get_trace_summaries_raw(None, _START, _END, limit=1_000_000)
    return call.await_count


_PAGE_LOOPS: list[tuple[str, Callable[[], Awaitable[int]]]] = [
    ("Datadog", _datadog_pages),
    ("Sentry", _sentry_pages),
    ("AWS X-Ray", _xray_pages),
]


@pytest.mark.parametrize(
    ("display", "run"), _PAGE_LOOPS, ids=[display for display, _ in _PAGE_LOOPS]
)
async def test_a_search_stops_after_the_documented_pages_and_only_logs_it(
    display: str, run: Callable[[], Awaitable[int]], caplog: pytest.LogCaptureFixture
) -> None:
    pages, _ = _search_caps()
    row = _row("Search caps")
    assert display in row[0]
    assert row[_SAYS_SO].startswith("No")
    assert f"WARNING `Stopped after {pages} pages with more results available" in row[_LOGGED]
    caplog.set_level(logging.WARNING)

    requests = await run()

    assert requests == pages
    stopped = [w for w in _warnings(caplog) if w.startswith("Stopped after")]
    assert [w.split("(", 1)[0] for w in stopped] == [
        f"Stopped after {pages} pages with more results available "
    ]


async def test_new_relic_clamps_the_nrql_limit_without_a_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, rows = _search_caps()
    row = _row("Search caps")
    assert "New Relic" in row[0]
    assert "nothing for New Relic" in row[_LOGGED]
    backend = _newrelic()
    run = AsyncMock(return_value=[])
    caplog.set_level(logging.DEBUG)
    with patch.object(backend, "_run_account_nrql", run):
        await backend.search_traces(TraceQuery(limit=rows // 2))
        await backend.search_traces(TraceQuery(limit=10))

    statements = [call.args[0] for call in run.await_args_list]
    assert statements[0].endswith(f" LIMIT {rows}")
    assert statements[1].endswith(" LIMIT 50")
    assert [r for r in caplog.records if r.name.startswith("opentelemetry_mcp")] == []


# -- the Honeycomb query rows row ----------------------------------------------------------


def _honeycomb_rows() -> int:
    return _ints(_row("Honeycomb query rows")[_VALUE])[0]


def test_the_honeycomb_row_states_the_code_constant_and_nothing_else() -> None:
    row = _row("Honeycomb query rows")
    assert _honeycomb_rows() == honeycomb._MAX_QUERY_ROWS
    assert set(_ints(" ".join(row))) == {honeycomb._MAX_QUERY_ROWS}
    assert row[_SAYS_SO].startswith("No")


async def test_honeycomb_lowers_every_span_search_to_the_documented_rows_without_a_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cap = _honeycomb_rows()
    backend = _honeycomb()
    run = AsyncMock(return_value=[])
    caplog.set_level(logging.DEBUG)
    with patch.object(backend, "_run_query", run):
        await backend._run_search([], _START, _END, limit=cap * 5)
        await backend._run_search([], _START, _END, limit=7)

    assert [call.kwargs["limit"] for call in run.await_args_list] == [cap, 7]
    assert [r for r in caplog.records if r.name.startswith("opentelemetry_mcp")] == []
    assert (
        "lowering any other query to the cap is not logged" in _row("Honeycomb query rows")[_LOGGED]
    )


@pytest.mark.parametrize(("name", "extra", "warns"), [("below", -1, False), ("at", 0, True)])
async def test_honeycomb_warns_only_when_a_batch_query_returns_exactly_the_documented_rows(
    name: str, extra: int, warns: bool, caplog: pytest.LogCaptureFixture
) -> None:
    cap = _honeycomb_rows()
    prefix = f"Batch hydration for 2 trace(s) returned the maximum {cap} rows"
    assert prefix.replace("for 2", "for N") in _row("Honeycomb query rows")[_LOGGED]
    backend = _honeycomb()
    rows = [{"data": {"trace.trace_id": "t0"}}] * (cap + extra)
    caplog.set_level(logging.WARNING)
    with patch.object(backend, "_run_search", AsyncMock(return_value=rows)):
        grouped = await backend._batch_fetch_trace_rows(["t0", "t1"])

    batch = [w for w in _warnings(caplog) if w.startswith("Batch hydration")]
    assert [w.startswith(prefix) for w in batch] == ([True] if warns else []), name
    assert set(grouped) == {"t0"}


# -- the hydration cap holds on every backend that hydrates --------------------------------


async def _hydrated_ids(kind: str, found: int) -> list[str]:
    """Run search_traces through the server on a real backend of `kind` that finds `found` trace
    ids, and return the ids it asked to hydrate, in order."""
    ids = [f"t{index}" for index in range(found)]
    if kind == "honeycomb":
        return await _hydrated_by_honeycomb(ids)
    if kind == "xray":
        return await _hydrated_by_xray([f"1-0-{index:024x}" for index in range(found)])
    backend: BaseBackend
    if kind == "datadog":
        backend, discovery = _datadog(), "_search_spans_raw"
        rows: list[dict[str, Any]] = [{"attributes": {"trace_id": item}} for item in ids]
    elif kind == "sentry":
        backend, discovery = _sentry(), "_search_events_raw"
        rows = [{"trace": item} for item in ids]
    else:
        backend, discovery = _newrelic(), "_run_nrql_search"
        rows = [{"trace.id": item} for item in ids]

    async def hydrate(trace_id: str) -> TraceData:
        return _trace(trace_id)

    fetch = AsyncMock(side_effect=hydrate)
    with (
        patch.object(backend, discovery, AsyncMock(return_value=rows)),
        patch.object(backend, "get_trace", fetch),
    ):
        await _install(backend)
        assert not (await _call("search_traces", {})).is_error
    return [call.args[0] for call in fetch.await_args_list]


async def _hydrated_by_honeycomb(ids: list[str]) -> list[str]:
    backend = _honeycomb()
    discovered = [{"data": {"trace.trace_id": item}} for item in ids]
    searches: list[list[Any]] = []

    async def run_search(filters: list[Any], start: Any, end: Any, limit: int) -> list[Any]:
        searches.append(filters)
        return discovered if len(searches) == 1 else []

    with patch.object(backend, "_run_search", run_search):
        await _install(backend)
        assert not (await _call("search_traces", {})).is_error
    return list(searches[1][0].values) if len(searches) > 1 else []


async def _hydrated_by_xray(ids: list[str]) -> list[str]:
    backend = _xray()
    batch = AsyncMock(return_value=({}, set()))
    with (
        patch.object(
            backend, "_get_trace_summaries_raw", AsyncMock(return_value=[{"Id": i} for i in ids])
        ),
        patch.object(backend, "_batch_get_traces_raw", batch),
    ):
        await _install(backend)
        assert not (await _call("search_traces", {})).is_error
    return list(batch.await_args.args[0]) if batch.await_args else []


_HYDRATING = ["datadog", "sentry", "newrelic", "honeycomb", "xray"]


@pytest.mark.parametrize("kind", _HYDRATING)
async def test_every_hydrating_backend_fetches_only_the_documented_number_of_traces(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    cap = _hydration_cap()
    caplog.set_level(logging.DEBUG)

    hydrated = await _hydrated_ids(kind, cap + 10)

    assert len(hydrated) == cap
    limiting = [w for w in _warnings(caplog) if w.startswith("Limiting trace fetch")]
    assert limiting == [_LIMIT_WARNING.format(cap=cap, found=cap + 10)]


@pytest.mark.parametrize("kind", _HYDRATING)
async def test_at_exactly_the_cap_no_backend_limits_or_logs(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    cap = _hydration_cap()
    caplog.set_level(logging.DEBUG)

    hydrated = await _hydrated_ids(kind, cap)

    assert len(hydrated) == cap
    assert [w for w in _warnings(caplog) if w.startswith("Limiting trace fetch")] == []


# -- failed hydration beyond Datadog, Sentry and New Relic ---------------------------------


async def test_honeycomb_returns_count_zero_when_no_candidate_hydrates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    row = _row("Failed hydration")
    assert "Honeycomb" in row[_SAYS_SO]
    assert "`count: 0`" in row[_SAYS_SO]
    assert "Failed to fetch trace <id>: no spans found" in row[_LOGGED]
    cap = _hydration_cap()
    backend = _honeycomb()
    candidates = [{"data": {"trace.trace_id": f"t{index}"}} for index in range(cap + 30)]
    found = AsyncMock(side_effect=[candidates, []])
    caplog.set_level(logging.WARNING)
    with patch.object(backend, "_run_search", found):
        await _install(backend)
        result = await _call("search_traces", {})

    assert not result.is_error
    assert _body(result) == {"count": 0, "traces": []}
    failed = [w for w in _warnings(caplog) if w.startswith("Failed to fetch trace")]
    assert failed == [f"Failed to fetch trace t{index}: no spans found" for index in range(cap)]


async def test_xray_returns_count_zero_when_every_id_is_unprocessed_and_logs_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    row = _row("Failed hydration")
    assert "AWS X-Ray" in row[_SAYS_SO]
    assert "`count: 0`" in row[_SAYS_SO]
    assert "left unprocessed" in row[_LOGGED]
    cap = _hydration_cap()
    ids = [f"1-0-{index:024x}" for index in range(cap)]
    backend = _xray()
    caplog.set_level(logging.WARNING)
    with (
        patch.object(
            backend, "_get_trace_summaries_raw", AsyncMock(return_value=[{"Id": i} for i in ids])
        ),
        patch.object(backend, "_batch_get_traces_raw", AsyncMock(return_value=({}, set(ids)))),
    ):
        await _install(backend)
        result = await _call("search_traces", {})

    assert not result.is_error
    assert _body(result) == {"count": 0, "traces": []}
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert warnings[0].startswith(f"X-Ray reported {cap} unprocessed trace IDs: ")


async def test_xray_logs_nothing_for_a_trace_without_segments_or_for_search_spans(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logged = _row("Failed hydration")[_LOGGED]
    assert "no segments" in logged
    assert "`search_spans`" in logged
    ids = [f"1-0-{index:024x}" for index in range(3)]
    backend = _xray()
    caplog.set_level(logging.DEBUG)
    with patch.object(
        backend, "_get_trace_summaries_raw", AsyncMock(return_value=[{"Id": i} for i in ids])
    ):
        with patch.object(
            backend, "_batch_get_traces_raw", AsyncMock(return_value=({i: [] for i in ids}, set()))
        ):
            await _install(backend)
            empty = _body(await _call("search_traces", {}))
        with patch.object(backend, "_batch_get_traces_raw", AsyncMock(return_value=({}, set(ids)))):
            spans = _body(await _call("search_spans_tool", {}))

    assert empty == {"count": 0, "traces": []}
    assert spans["count"] == 0
    assert [r for r in caplog.records if r.name == "opentelemetry_mcp.backends.xray"] == []


async def test_tempo_logs_every_failed_fetch(caplog: pytest.LogCaptureFixture) -> None:
    row = _row("Failed hydration")
    assert "Tempo" in row[_LOGGED]
    fake = _TempoFake(candidates=3, failing={1})
    fake.install()
    caplog.set_level(logging.WARNING)

    result = _body(await _call("search_traces", {}))

    assert result["count"] == 2
    failed = [w for w in _warnings(caplog) if w.startswith("Failed to fetch trace")]
    assert [w.split(":", 1)[0] for w in failed] == [f"Failed to fetch trace {_tid(1)}"]

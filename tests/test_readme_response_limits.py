"""README "What responses report and what they do not": limits, hydration and correlation.

Every number, backend name and behaviour the README states about the ceiling, the hydration cap,
failed hydration and the correlate_trace fallback is read back out of README.md and compared with
the constant it describes or with what a real tool call returns against a loopback fake, so the
prose and the code cannot drift apart in either direction.
"""

import importlib
import logging
import pkgutil
from collections.abc import Iterator
from datetime import timedelta
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

import opentelemetry_mcp.server as server
from opentelemetry_mcp import backends
from opentelemetry_mcp.backends import datadog, honeycomb, newrelic, sentry, xray
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.backends.datadog import DatadogBackend
from opentelemetry_mcp.backends.newrelic import NewRelicBackend
from opentelemetry_mcp.backends.sentry import SentryBackend
from opentelemetry_mcp.backends.xray import XRayBackend
from opentelemetry_mcp.config import ServerConfig
from opentelemetry_mcp.models import SearchSpansResult, SearchTracesResult, SpanQuery
from opentelemetry_mcp.tools import correlate
from tests.readme_response_support import (
    _FAKE_CREDENTIAL,
    _LOGGED,
    _SAYS_SO,
    _VALUE,
    _body,
    _call,
    _ceiling,
    _Fake,
    _hydration_cap,
    _install,
    _ints,
    _row,
    _section,
    _TempoFake,
    _warnings,
)


@pytest.fixture(autouse=True)
def _reset_server_globals() -> Iterator[None]:
    server._backend = server._secondary_backend = server._config = None
    yield
    server._backend = server._secondary_backend = server._config = None


# -- the ceiling ---------------------------------------------------------------------------


def test_ceiling_default_and_range_match_the_config_field() -> None:
    field = ServerConfig.model_fields["max_traces_per_query"]
    lower = next(m.ge for m in field.metadata if hasattr(m, "ge"))
    upper = next(m.le for m in field.metadata if hasattr(m, "le"))

    assert _ints(_row("Limit above the ceiling")[_VALUE]) == [field.default, lower, upper]


async def test_a_limit_above_the_ceiling_is_lowered_and_the_response_says_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ceiling = _ceiling()
    fake = _Fake(traces=ceiling + 200)
    await _install(fake)
    assert _row("Limit above the ceiling")[_SAYS_SO].startswith("No")
    assert _row("Limit above the ceiling")[_LOGGED].startswith("No")
    caplog.set_level(logging.DEBUG)
    caplog.clear()

    over = _body(await _call("search_traces", {"limit": 1000}))
    at = _body(await _call("search_traces", {"limit": ceiling}))

    assert fake.seen_limits == [ceiling, ceiling]
    assert set(over) == set(at) == {"count", "traces"}
    assert over["count"] == at["count"] == ceiling
    mentions = [
        r.getMessage()
        for r in caplog.records
        if r.name.startswith("opentelemetry_mcp") and "limit" in r.getMessage().lower()
    ]
    assert mentions == []


def test_typed_results_carry_only_what_the_readme_says_they_carry() -> None:
    assert "exactly `count`" in _section()
    assert set(SearchTracesResult.model_fields) == {"count", "traces"}
    assert set(SearchSpansResult.model_fields) == {"count", "spans"}


def test_no_limits_row_states_a_number_the_code_does_not_pin() -> None:
    field = ServerConfig.model_fields["max_traces_per_query"]
    lower = next(m.ge for m in field.metadata if hasattr(m, "ge"))
    upper = next(m.le for m in field.metadata if hasattr(m, "le"))
    hours = int(correlate._MAX_SEARCH_WINDOW / timedelta(hours=1))
    pinned = {
        "Limit above the ceiling": {field.default, lower, upper},
        "Hydration cap": {xray._MAX_TRACES_TO_HYDRATE},
        "Failed hydration": {0},
        "Correlation fallback": {
            correlate._MAX_REPORTED_MATCHES,
            correlate.MAX_CANDIDATES_PER_SERVICE_QUERY,
            correlate._MAX_SERVICE_QUERIES,
            hours,
        },
    }
    for label, allowed in pinned.items():
        stray = set(_ints(" ".join(_row(label)))) - allowed
        assert not stray, (label, stray)


# -- the hydration cap ---------------------------------------------------------------------

_HYDRATING = [datadog, newrelic, sentry, honeycomb, xray]


@pytest.mark.parametrize("module", _HYDRATING, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_each_hydrating_backend_constant_matches_the_readme(module: ModuleType) -> None:
    assert [module._MAX_TRACES_TO_HYDRATE] == _ints(_row("Hydration cap")[_VALUE])


def test_the_readme_names_exactly_the_backends_that_hydrate() -> None:
    names = [m.name for m in pkgutil.iter_modules(backends.__path__)]
    capped = {
        name
        for name in names
        if hasattr(
            importlib.import_module(f"opentelemetry_mcp.backends.{name}"), "_MAX_TRACES_TO_HYDRATE"
        )
    }
    assert capped == {"datadog", "newrelic", "sentry", "honeycomb", "xray"}
    behaviour = _row("Hydration cap")[0]
    for display in ("Tempo", "Datadog", "Sentry", "AWS X-Ray", "New Relic", "Honeycomb"):
        assert display in behaviour
    assert "Jaeger and Traceloop" in behaviour


async def test_tempo_fetches_only_the_documented_number_of_traces(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cap = _hydration_cap()
    fake = _TempoFake(candidates=cap + 10)
    fake.install()
    caplog.set_level(logging.WARNING)

    body = _body(await _call("search_traces", {"limit": cap + 50}))

    assert len(fake.fetched) == cap
    assert set(body) == {"count", "traces"}
    assert body["count"] == len(body["traces"]) == cap
    message = (
        f"Limiting trace fetch to {cap} out of {cap + 10} results to avoid excessive API calls"
    )
    assert _warnings(caplog) == [message]
    assert message.replace(f"out of {cap + 10}", "out of N") in _row("Hydration cap")[_LOGGED]
    assert _row("Hydration cap")[_SAYS_SO].startswith("No")


async def test_at_exactly_the_cap_nothing_is_limited_and_nothing_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cap = _hydration_cap()
    fake = _TempoFake(candidates=cap)
    fake.install()
    caplog.set_level(logging.WARNING)

    body = _body(await _call("search_traces", {"limit": cap + 50}))

    assert body["count"] == cap
    assert _warnings(caplog) == []


async def test_usage_totals_cover_only_the_hydrated_traces() -> None:
    cap = _hydration_cap()
    fake = _TempoFake(candidates=cap + 10, llm=True)
    fake.install()

    usage = _body(await _call("get_llm_usage", {"limit": 1000}))

    assert usage["summary"]["total_requests"] == cap
    assert "`get_llm_usage`" in _row("Hydration cap")[_SAYS_SO]


async def test_xray_applies_the_same_cap_to_spans_and_service_discovery_without_a_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cap = _hydration_cap()
    logged = _row("Hydration cap")[_LOGGED]
    for name in ("`search_spans`", "`get_service_operations`", "`list_services`"):
        assert name in logged
    backend = XRayBackend(url="https://xray.us-east-1.amazonaws.com", aws_region="us-east-1")
    summaries = [{"Id": f"1-0-{index:024x}"} for index in range(cap + 10)]
    batch = AsyncMock(return_value=({}, []))
    caplog.set_level(logging.WARNING)
    with (
        patch.object(backend, "_get_trace_summaries_raw", AsyncMock(return_value=summaries)),
        patch.object(backend, "_batch_get_traces_raw", batch),
        patch.object(backend, "_call", AsyncMock(return_value={"Services": []})),
    ):
        await backend.search_spans(SpanQuery(limit=10))
        await backend.get_service_operations("svc")
        await backend.list_services()

    assert [len(call.args[0]) for call in batch.await_args_list] == [cap, cap, cap]
    assert _warnings(caplog) == []


# -- failed hydration ----------------------------------------------------------------------


def _hydrating_backend(kind: str) -> tuple[BaseBackend, str, list[dict[str, Any]]]:
    """A real backend, the method that discovers candidate trace ids, and two rows for it."""
    if kind == "datadog":
        backend: BaseBackend = DatadogBackend(
            url="https://api.datadoghq.com", api_key=_FAKE_CREDENTIAL, app_key=_FAKE_CREDENTIAL
        )
        rows: list[dict[str, Any]] = [{"attributes": {"trace_id": f"t{i}"}} for i in range(2)]
        return backend, "_search_spans_raw", rows
    if kind == "sentry":
        backend = SentryBackend(url="https://sentry.io", api_key=_FAKE_CREDENTIAL, org_slug="org")
        return backend, "_search_events_raw", [{"trace": f"t{i}"} for i in range(2)]
    backend = NewRelicBackend(
        url="https://api.newrelic.com/graphql", api_key=_FAKE_CREDENTIAL, account_id="123456"
    )
    return backend, "_run_nrql_search", [{"trace.id": f"t{i}"} for i in range(2)]


@pytest.mark.parametrize(
    ("kind", "display"), [("datadog", "Datadog"), ("sentry", "Sentry"), ("newrelic", "New Relic")]
)
async def test_when_every_hydration_fails_these_backends_return_count_zero(
    kind: str, display: str, caplog: pytest.LogCaptureFixture
) -> None:
    says_so = _row("Failed hydration")[_SAYS_SO]
    assert display in says_so
    assert "`count: 0`" in says_so
    backend, discovery, rows = _hydrating_backend(kind)
    caplog.set_level(logging.WARNING)
    with (
        patch.object(backend, discovery, AsyncMock(return_value=rows)),
        patch.object(backend, "get_trace", AsyncMock(side_effect=RuntimeError("outage"))),
    ):
        await _install(backend)
        result = await _call("search_traces", {})

    assert not result.is_error
    assert _body(result) == {"count": 0, "traces": []}
    assert [w for w in _warnings(caplog) if w.startswith("Failed to fetch trace")] == [
        "Failed to fetch trace t0: outage",
        "Failed to fetch trace t1: outage",
    ]


async def test_tempo_skips_one_failed_trace_silently_and_raises_only_if_every_fetch_fails() -> None:
    row = _row("Failed hydration")
    assert "Tempo raises an error" in row[_SAYS_SO]
    assert "Datadog, Sentry and New Relic" in row[_SAYS_SO]
    assert "`count: 0`" in row[_SAYS_SO]
    assert "Failed to fetch trace" in row[_LOGGED]
    partial = _TempoFake(candidates=3, failing={1})
    partial.install()

    one_failed = await _call("search_traces", {})

    assert not one_failed.is_error
    assert _body(one_failed)["count"] == 2

    every = _TempoFake(candidates=3, all_fail=True)
    every.install()
    assert (await _call("search_traces", {})).is_error


# -- correlate_trace -----------------------------------------------------------------------


def test_correlation_constants_match_the_readme() -> None:
    row = _row("Correlation fallback")
    assert _ints(row[_VALUE]) == [
        correlate._MAX_REPORTED_MATCHES,
        correlate.MAX_CANDIDATES_PER_SERVICE_QUERY,
    ]
    hours = int(correlate._MAX_SEARCH_WINDOW / timedelta(hours=1))
    assert f"more than {correlate._MAX_SERVICE_QUERIES} services" in row[_SAYS_SO]
    assert f"more than {hours} h" in row[_SAYS_SO]


async def test_more_candidates_than_the_match_cap_are_cut_without_a_note() -> None:
    cap = _ints(_row("Correlation fallback")[_VALUE])[0]
    await _install(_Fake(), _Fake(traces=cap + 3, fail_get_trace=True))

    result = _body(await _call("correlate_trace", {"trace_id": "t1"}))

    assert len(result["matches"]) == cap
    assert result["limitations"] == correlate._LIMITATIONS


async def test_a_failing_secondary_search_leaves_matches_empty_and_only_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    row = _row("Correlation fallback")
    await _install(_Fake(), _Fake(fail_get_trace=True, fail_search=True))
    caplog.set_level(logging.DEBUG)

    result = _body(await _call("correlate_trace", {"trace_id": "t1"}))

    assert result["matches"] == []
    assert result["limitations"] == correlate._LIMITATIONS
    search_prefix = "correlate_trace: secondary search_traces failed for service_name="
    assert [w for w in _warnings(caplog) if w.startswith(search_prefix)]
    assert search_prefix in row[_LOGGED]
    debug = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any(r.getMessage().startswith("correlate_trace: secondary get_trace(") for r in debug)
    assert "DEBUG" in row[_LOGGED]

"""README "What responses report and what they do not": a failed batch call is not an empty answer.

AWS X-Ray and Honeycomb hydrate candidate traces through batch calls (BatchGetTraces; one batched
query). The Failed hydration row says that when such a call ANSWERS with nothing the search returns
count 0, but when the call itself FAILS the whole search fails with an error, even if earlier calls
had already returned traces. Each claim is run through the real tool against the real backend object
with only its network leaf replaced (tests/readme_response_batch_support.py).
"""

import logging
from collections.abc import Callable, Iterator

import pytest
from botocore.exceptions import ReadTimeoutError

import opentelemetry_mcp.server as server
from opentelemetry_mcp.backends.honeycomb import HoneycombRateLimitedError
from tests.readme_response_batch_support import (
    XRAY_BATCH_SIZE,
    HoneycombLoopback,
    XRayLoopback,
    honeycomb_ids,
    http_500,
    throttling_error,
    xray_ids,
)
from tests.readme_response_support import _LOGGED, _SAYS_SO, _body, _call, _install, _row, _warnings


@pytest.fixture(autouse=True)
def _reset_server_globals() -> Iterator[None]:
    server._backend = server._secondary_backend = server._config = None
    yield
    server._backend = server._secondary_backend = server._config = None


def _read_timeout() -> ReadTimeoutError:
    return ReadTimeoutError(endpoint_url="https://xray.us-east-1.amazonaws.com/")


def _rate_limited() -> HoneycombRateLimitedError:
    return HoneycombRateLimitedError("no Create Query Result slot frees in time")


_XRAY_ERRORS: list[Callable[[], Exception]] = [throttling_error, _read_timeout]
_HONEYCOMB_ERRORS: list[Callable[[], Exception]] = [http_500, _rate_limited]


# -- the README sentence -------------------------------------------------------------------


def test_the_failed_hydration_row_says_a_failed_batch_call_fails_the_whole_search() -> None:
    row = _row("Failed hydration")
    says = row[_SAYS_SO]

    assert "A batch call that itself fails is different from an empty answer" in says
    assert (
        "on AWS X-Ray (any one BatchGetTraces call, even after earlier calls returned traces; "
        "a botocore error such as `ThrottlingException`)" in says
    )
    assert "and on Honeycomb (the one batch query)" in says
    assert "it fails the whole search with an error, not `count: 0`" in says
    assert "`Error executing search_traces: <error>`" in row[_LOGGED]
    assert "logs no hydration warning" in row[_LOGGED]


# -- AWS X-Ray -----------------------------------------------------------------------------


@pytest.mark.parametrize("make_error", _XRAY_ERRORS, ids=["throttling", "read-timeout"])
async def test_a_failing_batchgettraces_call_fails_the_whole_xray_search(
    make_error: Callable[[], Exception], caplog: pytest.LogCaptureFixture
) -> None:
    error = make_error()
    loopback = XRayLoopback(xray_ids(1), fail_on=1, error=error)
    caplog.set_level(logging.WARNING)

    with loopback.patched():
        await _install(loopback.backend)
        result = await _call("search_traces", {})

    assert result.is_error
    assert str(error) in str(result.content)
    assert loopback.batches == [xray_ids(1)]
    logged = [r for r in caplog.records if r.name == "opentelemetry_mcp.server"]
    assert [r.getMessage() for r in logged if r.levelno == logging.ERROR] == [
        f"Error executing search_traces: {error}"
    ]
    assert [w for w in _warnings(caplog) if w.startswith("Failed to fetch trace")] == []


async def test_a_later_failing_xray_call_discards_the_traces_earlier_calls_returned() -> None:
    ids = xray_ids(XRAY_BATCH_SIZE + 2)
    loopback = XRayLoopback(ids, fail_on=2, error=throttling_error())

    with loopback.patched():
        await _install(loopback.backend)
        result = await _call("search_traces", {})

    assert [len(batch) for batch in loopback.batches] == [XRAY_BATCH_SIZE, 2]
    assert result.is_error


async def test_the_same_xray_search_without_a_failing_call_returns_every_trace() -> None:
    ids = xray_ids(XRAY_BATCH_SIZE + 2)
    loopback = XRayLoopback(ids)

    with loopback.patched():
        await _install(loopback.backend)
        body = _body(await _call("search_traces", {}))

    assert [len(batch) for batch in loopback.batches] == [XRAY_BATCH_SIZE, 2]
    assert body["count"] == len(body["traces"]) == len(ids)


async def test_an_xray_answer_that_leaves_every_id_unprocessed_is_count_zero_not_an_error() -> None:
    loopback = XRayLoopback(xray_ids(3), unprocessed=True)

    with loopback.patched():
        await _install(loopback.backend)
        result = await _call("search_traces", {})

    assert not result.is_error
    assert _body(result) == {"count": 0, "traces": []}


# -- Honeycomb -----------------------------------------------------------------------------


@pytest.mark.parametrize("make_error", _HONEYCOMB_ERRORS, ids=["http-500", "rate-limited"])
async def test_a_failing_honeycomb_batch_query_fails_the_whole_search(
    make_error: Callable[[], Exception], caplog: pytest.LogCaptureFixture
) -> None:
    error = make_error()
    loopback = HoneycombLoopback(honeycomb_ids(3), batch_error=error)
    caplog.set_level(logging.WARNING)

    with loopback.patched():
        await _install(loopback.backend)
        result = await _call("search_traces", {})

    assert result.is_error
    assert str(error) in str(result.content)
    assert loopback.queries == 2
    logged = [r for r in caplog.records if r.name == "opentelemetry_mcp.server"]
    assert [r.getMessage() for r in logged if r.levelno == logging.ERROR] == [
        f"Error executing search_traces: {error}"
    ]
    assert [w for w in _warnings(caplog) if w.startswith("Failed to fetch trace")] == []


async def test_the_same_honeycomb_search_without_a_failing_query_returns_every_trace() -> None:
    ids = honeycomb_ids(3)
    loopback = HoneycombLoopback(ids)

    with loopback.patched():
        await _install(loopback.backend)
        body = _body(await _call("search_traces", {}))

    assert loopback.queries == 2
    assert body["count"] == len(body["traces"]) == len(ids)


async def test_a_honeycomb_batch_answer_with_no_spans_is_count_zero_not_an_error() -> None:
    loopback = HoneycombLoopback(honeycomb_ids(2), empty=True)

    with loopback.patched():
        await _install(loopback.backend)
        result = await _call("search_traces", {})

    assert not result.is_error
    assert _body(result) == {"count": 0, "traces": []}

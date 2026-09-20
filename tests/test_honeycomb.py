"""Tests for the Honeycomb backend.

Fixtures below are hand-built from a completed deep-research pass over
docs.honeycomb.io and honeycomb.io/pricing - not against a live account
(none was available, and the research itself found the programmatic Query
Data API is likely Enterprise-plan exclusive). See the module docstring in
``opentelemetry_mcp/backends/honeycomb.py`` for the specific things the
research couldn't pin down (the exact Enterprise-gating status code, the
raw-row-extraction technique, and the Query Result response shape) and how
this implementation handles each.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from opentelemetry_mcp.backends.honeycomb import (
    _MAX_QUERY_ROWS,
    HoneycombBackend,
    HoneycombEnterpriseRequiredError,
    _RateLimiter,
)
from opentelemetry_mcp.models import Filter, FilterOperator, FilterType, SpanQuery, TraceQuery

FAKE_CONFIG_KEY = "fake-honeycomb-configkey-1"
FAKE_DATASET = "my-dataset"


def _backend() -> HoneycombBackend:
    return HoneycombBackend(
        url="https://api.honeycomb.io", api_key=FAKE_CONFIG_KEY, dataset=FAKE_DATASET
    )


def test_honeycomb_backend_requires_https() -> None:
    with pytest.raises(ValueError, match="https://"):
        HoneycombBackend(
            url="http://api.honeycomb.io", api_key=FAKE_CONFIG_KEY, dataset=FAKE_DATASET
        )


def test_honeycomb_backend_requires_api_key() -> None:
    with pytest.raises(ValueError, match="requires a Configuration Key"):
        HoneycombBackend(url="https://api.honeycomb.io", api_key=None, dataset=FAKE_DATASET)


def test_honeycomb_backend_requires_dataset() -> None:
    with pytest.raises(ValueError, match="requires a dataset"):
        HoneycombBackend(url="https://api.honeycomb.io", api_key=FAKE_CONFIG_KEY, dataset=None)


def test_honeycomb_backend_initialization() -> None:
    backend = HoneycombBackend(
        url="https://api.honeycomb.io",
        api_key=FAKE_CONFIG_KEY,
        dataset=FAKE_DATASET,
        timeout=15.0,
    )
    assert backend.url == "https://api.honeycomb.io"
    assert backend.api_key == FAKE_CONFIG_KEY
    assert backend.dataset == FAKE_DATASET
    assert backend.timeout == 15.0


def test_honeycomb_client_disables_redirects() -> None:
    """X-Honeycomb-Team is a non-standard header (like Datadog's DD-API-KEY)
    that httpx does not strip on cross-origin redirects, so follow_redirects
    must be disabled to avoid leaking it to an unexpected host."""
    backend = _backend()
    assert backend.client.follow_redirects is False


def test_honeycomb_client_headers() -> None:
    backend = _backend()
    client = backend.client
    assert client.headers["X-Honeycomb-Team"] == FAKE_CONFIG_KEY


class TestRateLimiter:
    async def test_allows_calls_under_the_cap_without_waiting(self) -> None:
        limiter = _RateLimiter(max_per_minute=10)
        for _ in range(10):
            await limiter.acquire()
        # No assertion on timing needed - this must simply not raise/hang.

    async def test_waits_when_at_capacity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """acquire() loops and rechecks after waking (see its own docstring
        on why a bare check-then-sleep-then-append would let concurrent
        callers blow through the cap) - the fake clock must advance across
        a faked sleep the way a real one would across a real one, or the
        recheck loop spins forever against a clock that never moves."""
        limiter = _RateLimiter(max_per_minute=2)
        fake_now = [1000.0]
        sleep_calls: list[float] = []

        def _fake_monotonic() -> float:
            return fake_now[0]

        async def _fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)
            fake_now[0] += seconds

        monkeypatch.setattr("opentelemetry_mcp.backends.honeycomb.time.monotonic", _fake_monotonic)
        monkeypatch.setattr("opentelemetry_mcp.backends.honeycomb.asyncio.sleep", _fake_sleep)

        await limiter.acquire()
        await limiter.acquire()
        await limiter.acquire()  # third call within the same minute should wait

        assert len(sleep_calls) == 1
        assert sleep_calls[0] == pytest.approx(60.0)

    async def test_concurrent_callers_do_not_exceed_the_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: a bare check-then-sleep-then-append (no lock, no
        recheck after waking) lets every caller blocked on the same "wait
        for one slot" computation wake and append at once, blowing through
        the cap in a burst."""
        import asyncio

        limiter = _RateLimiter(max_per_minute=2)
        fake_now = [1000.0]

        def _fake_monotonic() -> float:
            return fake_now[0]

        async def _fake_sleep(seconds: float) -> None:
            fake_now[0] += seconds

        monkeypatch.setattr("opentelemetry_mcp.backends.honeycomb.time.monotonic", _fake_monotonic)
        monkeypatch.setattr("opentelemetry_mcp.backends.honeycomb.asyncio.sleep", _fake_sleep)

        # Fill the cap, then launch 3 more callers concurrently - all 3 must
        # still be serialized (never more than max_per_minute in any given
        # 60s window), not all wake and append together.
        await limiter.acquire()
        await limiter.acquire()
        await asyncio.gather(*(limiter.acquire() for _ in range(3)))

        # 5 total calls into a 2/min limiter, with the fake clock only ever
        # advanced by acquire()'s own waits - if serialized correctly, the
        # deque never holds more than max_per_minute entries whose
        # timestamps fall within any 60s window of each other.
        times = sorted(limiter._call_times)
        for i in range(len(times) - limiter._max_per_minute):
            window = times[i : i + limiter._max_per_minute + 1]
            assert window[-1] - window[0] >= 60.0, (
                f"{limiter._max_per_minute + 1} calls landed within a 60s window: {window}"
            )


class TestFilterToHoneycombFilter:
    def test_equals(self) -> None:
        backend = _backend()
        f = Filter(
            field="service.name",
            operator=FilterOperator.EQUALS,
            value="svc",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_honeycomb_filter(f) == {
            "column": "service.name",
            "op": "=",
            "value": "svc",
        }

    def test_not_equals(self) -> None:
        backend = _backend()
        f = Filter(
            field="service.name",
            operator=FilterOperator.NOT_EQUALS,
            value="svc",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_honeycomb_filter(f) == {
            "column": "service.name",
            "op": "!=",
            "value": "svc",
        }

    def test_gt_lt_gte_lte(self) -> None:
        backend = _backend()
        for operator, op in (
            (FilterOperator.GT, ">"),
            (FilterOperator.LT, "<"),
            (FilterOperator.GTE, ">="),
            (FilterOperator.LTE, "<="),
        ):
            f = Filter(field="duration", operator=operator, value=100, value_type=FilterType.NUMBER)
            assert backend._filter_to_honeycomb_filter(f) == {
                "column": "duration_ms",
                "op": op,
                "value": 100,
            }

    def test_contains(self) -> None:
        backend = _backend()
        f = Filter(
            field="name",
            operator=FilterOperator.CONTAINS,
            value="chat",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_honeycomb_filter(f) == {
            "column": "name",
            "op": "contains",
            "value": "chat",
        }

    def test_exists(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system", operator=FilterOperator.EXISTS, value_type=FilterType.STRING
        )
        assert backend._filter_to_honeycomb_filter(f) == {"column": "gen_ai.system", "op": "exists"}

    def test_not_exists(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system", operator=FilterOperator.NOT_EXISTS, value_type=FilterType.STRING
        )
        assert backend._filter_to_honeycomb_filter(f) == {
            "column": "gen_ai.system",
            "op": "does-not-exist",
        }

    def test_in(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system",
            operator=FilterOperator.IN,
            values=["openai", "anthropic"],
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_honeycomb_filter(f) == {
            "column": "gen_ai.system",
            "op": "in",
            "value": ["openai", "anthropic"],
        }

    def test_field_name_mapping(self) -> None:
        backend = _backend()
        assert backend._honeycomb_field("service.name") == "service.name"
        assert backend._honeycomb_field("operation_name") == "name"
        assert backend._honeycomb_field("duration") == "duration_ms"
        assert backend._honeycomb_field("status") == "otel.status_code"
        assert backend._honeycomb_field("gen_ai.system") == "gen_ai.system"

    def test_unsafe_field_name_is_rejected(self) -> None:
        backend = _backend()
        f = Filter(
            field="x'); DROP TABLE spans;--",
            operator=FilterOperator.EQUALS,
            value="v",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_honeycomb_filter(f) is None


class TestInferStatus:
    def test_ok(self) -> None:
        assert _backend()._infer_status("OK") == "OK"
        assert _backend()._infer_status("ok") == "OK"

    def test_error(self) -> None:
        assert _backend()._infer_status("ERROR") == "ERROR"

    def test_unrecognized_defaults_to_unset(self) -> None:
        assert _backend()._infer_status(None) == "UNSET"
        assert _backend()._infer_status(123) == "UNSET"


class TestParseTimestamp:
    def test_epoch_seconds_int(self) -> None:
        backend = _backend()
        result = backend._parse_honeycomb_timestamp(1704067200)
        assert result == datetime(2024, 1, 1, tzinfo=UTC)

    def test_none_returns_none(self) -> None:
        assert _backend()._parse_honeycomb_timestamp(None) is None

    def test_unparseable_returns_none(self) -> None:
        assert _backend()._parse_honeycomb_timestamp("not-a-timestamp") is None

    def test_bool_returns_none(self) -> None:
        assert _backend()._parse_honeycomb_timestamp(True) is None


class TestParseHoneycombRow:
    def _row(self, **data_overrides: Any) -> dict[str, Any]:
        data = {
            "trace.trace_id": "t1",
            "trace.span_id": "s1",
            "trace.parent_id": None,
            "name": "chat_completion",
            "service.name": "my-llm-app",
            "timestamp": 1704067200,
            "duration_ms": 150.5,
            "otel.status_code": "OK",
            "gen_ai.system": "openai",
        }
        data.update(data_overrides)
        return {"data": data}

    def test_happy_path(self) -> None:
        backend = _backend()
        span = backend._parse_honeycomb_row(self._row())
        assert span is not None
        assert span.span_id == "s1"
        assert span.trace_id == "t1"
        assert span.operation_name == "chat_completion"
        assert span.service_name == "my-llm-app"
        assert span.duration_ms == 150.5
        assert span.status == "OK"
        assert span.attributes.gen_ai_system == "openai"

    def test_non_dict_data_is_rejected(self) -> None:
        backend = _backend()
        assert backend._parse_honeycomb_row({"data": "not-a-dict"}) is None

    def test_missing_span_id_rejected(self) -> None:
        backend = _backend()
        assert backend._parse_honeycomb_row(self._row(**{"trace.span_id": None})) is None

    def test_missing_trace_id_rejected(self) -> None:
        backend = _backend()
        assert backend._parse_honeycomb_row(self._row(**{"trace.trace_id": None})) is None

    def test_missing_operation_name_rejected(self) -> None:
        backend = _backend()
        assert backend._parse_honeycomb_row(self._row(name=None)) is None

    def test_missing_timestamp_rejected(self) -> None:
        backend = _backend()
        assert backend._parse_honeycomb_row(self._row(timestamp=None)) is None

    def test_negative_duration_rejected(self) -> None:
        backend = _backend()
        assert backend._parse_honeycomb_row(self._row(duration_ms=-5.0)) is None

    def test_parent_id_is_carried_through(self) -> None:
        backend = _backend()
        span = backend._parse_honeycomb_row(self._row(**{"trace.parent_id": "parent1"}))
        assert span is not None
        assert span.parent_span_id == "parent1"


class TestGroupIntoTrace:
    def test_any_error_span_makes_trace_status_error(self) -> None:
        backend = _backend()
        root = backend._parse_honeycomb_row(
            {
                "data": {
                    "trace.trace_id": "t1",
                    "trace.span_id": "root",
                    "trace.parent_id": None,
                    "name": "root-op",
                    "service.name": "svc",
                    "timestamp": 1704067200,
                    "duration_ms": 100.0,
                    "otel.status_code": "OK",
                }
            }
        )
        child = backend._parse_honeycomb_row(
            {
                "data": {
                    "trace.trace_id": "t1",
                    "trace.span_id": "child",
                    "trace.parent_id": "root",
                    "name": "child-op",
                    "service.name": "svc",
                    "timestamp": 1704067200,
                    "duration_ms": 50.0,
                    "otel.status_code": "ERROR",
                }
            }
        )
        assert root is not None and child is not None
        trace = backend._group_into_trace("t1", [root, child])
        assert trace.status == "ERROR"

    def test_all_spans_ok_makes_trace_status_ok(self) -> None:
        backend = _backend()
        root = backend._parse_honeycomb_row(_query_row("t1", "root", **{"otel.status_code": "OK"}))
        child = backend._parse_honeycomb_row(
            _query_row("t1", "child", **{"trace.parent_id": "root", "otel.status_code": "OK"})
        )
        assert root is not None and child is not None

        trace = backend._group_into_trace("t1", [root, child])

        assert trace.status == "OK"

    def test_no_error_but_not_all_ok_makes_trace_status_unset(self) -> None:
        """A span with no recognized otel.status_code (UNSET, not ERROR)
        must not make the whole trace silently read as OK."""
        backend = _backend()
        root = backend._parse_honeycomb_row(_query_row("t1", "root", **{"otel.status_code": "OK"}))
        child = backend._parse_honeycomb_row(
            _query_row("t1", "child", **{"trace.parent_id": "root", "otel.status_code": None})
        )
        assert root is not None and child is not None

        trace = backend._group_into_trace("t1", [root, child])

        assert trace.status == "UNSET"


def _query_row(trace_id: str, span_id: str, **overrides: Any) -> dict[str, Any]:
    data = {
        "trace.trace_id": trace_id,
        "trace.span_id": span_id,
        "trace.parent_id": None,
        "name": "op",
        "service.name": "svc",
        "timestamp": 1704067200,
        "duration_ms": 100.0,
        "otel.status_code": "OK",
    }
    data.update(overrides)
    return {"data": data}


class TestCreateQueryResultEnterpriseGating:
    async def test_402_is_wrapped_as_enterprise_required(self) -> None:
        backend = _backend()

        async def _fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
            return httpx.Response(402, request=httpx.Request("POST", "https://api.honeycomb.io/x"))

        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.post = _fake_post

        with pytest.raises(HoneycombEnterpriseRequiredError, match="Enterprise-tier"):
            await backend._create_query_result("query-1")

    async def test_403_is_wrapped_as_enterprise_required(self) -> None:
        backend = _backend()

        async def _fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
            return httpx.Response(403, request=httpx.Request("POST", "https://api.honeycomb.io/x"))

        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.post = _fake_post

        with pytest.raises(HoneycombEnterpriseRequiredError, match="Enterprise-tier"):
            await backend._create_query_result("query-1")

    async def test_other_error_status_propagates_normally(self) -> None:
        backend = _backend()

        async def _fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
            return httpx.Response(500, request=httpx.Request("POST", "https://api.honeycomb.io/x"))

        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.post = _fake_post

        with pytest.raises(httpx.HTTPStatusError):
            await backend._create_query_result("query-1")

    async def test_malformed_response_shape_raises_with_payload_in_message(self) -> None:
        backend = _backend()

        async def _fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
            return httpx.Response(
                200,
                json={"unexpected": "shape"},
                request=httpx.Request("POST", "https://api.honeycomb.io/x"),
            )

        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.post = _fake_post

        with pytest.raises(ValueError, match="unexpected"):
            await backend._create_query_result("query-1")

    async def test_query_spec_creation_also_checks_enterprise_gating(self) -> None:
        """The docstring's own disclosure is that the exact gating boundary
        (query creation vs. query-result creation) was never confirmed live
        - this backend defensively checks both."""
        backend = _backend()

        async def _fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
            return httpx.Response(403, request=httpx.Request("POST", "https://api.honeycomb.io/x"))

        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.post = _fake_post

        with pytest.raises(HoneycombEnterpriseRequiredError, match="Enterprise-tier"):
            await backend._create_query_spec({"breakdowns": []})


class TestPollQueryResult:
    async def test_returns_results_once_complete(self) -> None:
        backend = _backend()
        rows = [_query_row("t1", "s1")]

        async def _fake_get(*args: Any, **kwargs: Any) -> httpx.Response:
            return httpx.Response(
                200,
                json={"complete": True, "data": {"results": rows}},
                request=httpx.Request("GET", "https://api.honeycomb.io/x"),
            )

        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = _fake_get

        result = await backend._poll_query_result("result-1")

        assert result == rows

    async def test_malformed_complete_response_raises_with_payload_in_message(self) -> None:
        backend = _backend()

        async def _fake_get(*args: Any, **kwargs: Any) -> httpx.Response:
            return httpx.Response(
                200,
                json={"complete": True, "data": "not-an-object"},
                request=httpx.Request("GET", "https://api.honeycomb.io/x"),
            )

        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = _fake_get

        with pytest.raises(ValueError, match="not-an-object"):
            await backend._poll_query_result("result-1")

    async def test_non_object_response_raises(self) -> None:
        backend = _backend()

        async def _fake_get(*args: Any, **kwargs: Any) -> httpx.Response:
            return httpx.Response(
                200,
                json=["not", "an", "object"],
                request=httpx.Request("GET", "https://api.honeycomb.io/x"),
            )

        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = _fake_get

        with pytest.raises(ValueError, match="expected an object"):
            await backend._poll_query_result("result-1")

    async def test_polls_again_when_not_yet_complete(self) -> None:
        backend = _backend()
        call_count = 0

        async def _fake_get(*args: Any, **kwargs: Any) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(
                    200,
                    json={"complete": False},
                    request=httpx.Request("GET", "https://api.honeycomb.io/x"),
                )
            return httpx.Response(
                200,
                json={"complete": True, "data": {"results": []}},
                request=httpx.Request("GET", "https://api.honeycomb.io/x"),
            )

        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = _fake_get

        result = await backend._poll_query_result("result-1")

        assert result == []
        assert call_count == 2

    async def test_times_out_if_never_complete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _backend()

        async def _fake_get(*args: Any, **kwargs: Any) -> httpx.Response:
            return httpx.Response(
                200,
                json={"complete": False},
                request=httpx.Request("GET", "https://api.honeycomb.io/x"),
            )

        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = _fake_get

        times = iter([0.0, 100.0])
        monkeypatch.setattr(
            "opentelemetry_mcp.backends.honeycomb.time.monotonic", lambda: next(times, 100.0)
        )

        async def _fake_sleep(seconds: float) -> None:
            return None

        monkeypatch.setattr("opentelemetry_mcp.backends.honeycomb.asyncio.sleep", _fake_sleep)

        with pytest.raises(ValueError, match="did not complete"):
            await backend._poll_query_result("result-1")


class TestGetTrace:
    async def test_happy_path(self) -> None:
        backend = _backend()
        backend._run_search = AsyncMock(  # type: ignore[method-assign]
            return_value=[_query_row("t1", "s1")]
        )

        trace = await backend.get_trace("t1")

        assert trace.trace_id == "t1"
        assert len(trace.spans) == 1

    async def test_no_spans_found_raises(self) -> None:
        backend = _backend()
        backend._run_search = AsyncMock(return_value=[])  # type: ignore[method-assign]

        with pytest.raises(ValueError, match="No spans found"):
            await backend.get_trace("missing")

    async def test_mismatched_trace_id_is_filtered_out(self) -> None:
        backend = _backend()
        backend._run_search = AsyncMock(  # type: ignore[method-assign]
            return_value=[_query_row("different-trace", "s1")]
        )

        with pytest.raises(ValueError, match="No spans found"):
            await backend.get_trace("t1")


class TestHealthCheck:
    async def test_healthy(self, fake_json_client: Callable[..., Any]) -> None:
        backend = _backend()
        backend._client = fake_json_client({"team": {"slug": "acme"}})

        result = await backend.health_check()

        assert result.status == "healthy"
        assert result.backend == "honeycomb"

    async def test_unhealthy_wraps_exception(self) -> None:
        backend = _backend()

        async def _fake_get(*args: Any, **kwargs: Any) -> httpx.Response:
            return httpx.Response(401, request=httpx.Request("GET", "https://api.honeycomb.io/x"))

        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = _fake_get

        result = await backend.health_check()

        assert result.status == "unhealthy"
        assert result.error is not None


class TestSearchSpans:
    async def test_happy_path(self) -> None:
        backend = _backend()
        backend._run_search = AsyncMock(  # type: ignore[method-assign]
            return_value=[_query_row("t1", "s1")]
        )

        result = await backend.search_spans(SpanQuery(limit=10))

        assert len(result) == 1
        assert result[0].span_id == "s1"

    async def test_unsupported_operator_is_applied_client_side(self) -> None:
        """NOT_IN isn't in get_supported_operators() - it must still filter
        correctly, just applied locally after the (unfiltered-by-it) search
        rather than pushed down to Honeycomb's own filter syntax."""
        backend = _backend()
        backend._run_search = AsyncMock(  # type: ignore[method-assign]
            return_value=[_query_row("t1", "s1", name="op-a"), _query_row("t1", "s2", name="op-b")]
        )

        result = await backend.search_spans(
            SpanQuery(
                limit=10,
                filters=[
                    Filter(
                        field="operation_name",
                        operator=FilterOperator.NOT_IN,
                        values=["op-b"],
                        value_type=FilterType.STRING,
                    )
                ],
            )
        )

        assert [s.span_id for s in result] == ["s1"]


class TestSearchTraces:
    """search_traces's real search-then-batch-hydrate flow, mirroring the
    New Relic backend's own tests for the same enrichment pattern (see
    _enrich_trace_with_search_rows, module docstring point 2) - but
    hydration itself is one batched `trace.trace_id IN (...)` query here,
    not one call per trace (see _batch_fetch_trace_rows's own docstring
    for why: composing per-trace hydration with _query_result_rate_limiter
    could otherwise stall a single search_traces() call for minutes)."""

    async def test_happy_path_hydrates_via_batch_fetch(self) -> None:
        backend = _backend()
        backend._run_search = AsyncMock(return_value=[_query_row("t1", "s1")])  # type: ignore[method-assign]

        result = await backend.search_traces(TraceQuery(limit=10))

        assert len(result) == 1
        assert result[0].trace_id == "t1"

    async def test_no_matching_spans_returns_empty_list(self) -> None:
        backend = _backend()
        backend._run_search = AsyncMock(return_value=[])  # type: ignore[method-assign]

        result = await backend.search_traces(TraceQuery(limit=10))

        assert result == []
        # No trace_ids discovered - the batch-hydration query must not
        # even be attempted.
        backend._run_search.assert_awaited_once()

    async def test_multiple_traces_are_hydrated_in_a_single_batched_call(self) -> None:
        """Regression for the fix described in this class's own docstring:
        discovering N distinct trace_ids must cost exactly one more
        _run_search call for hydration, not N more."""
        backend = _backend()
        discovery_rows = [
            _query_row("t1", "s1"),
            _query_row("t1", "s2"),
            _query_row("t2", "s3"),
        ]
        hydration_rows = [
            _query_row("t1", "s1"),
            _query_row("t1", "s2"),
            _query_row("t2", "s3"),
        ]
        backend._run_search = AsyncMock(  # type: ignore[method-assign]
            side_effect=[discovery_rows, hydration_rows]
        )

        result = await backend.search_traces(TraceQuery(limit=10))

        assert backend._run_search.await_count == 2
        assert {t.trace_id for t in result} == {"t1", "t2"}

    async def test_trace_missing_from_the_batch_hydration_result_is_skipped_not_raised(
        self,
    ) -> None:
        """A trace_id discovered in the initial search but absent from the
        batch-hydration result (e.g. it aged out of the window between the
        two calls) must be dropped, not crash the whole search_traces()."""
        backend = _backend()
        backend._run_search = AsyncMock(  # type: ignore[method-assign]
            side_effect=[
                [_query_row("t1", "s1"), _query_row("t2", "s2")],
                [_query_row("t1", "s1")],  # t2 missing from the hydration result
            ]
        )

        result = await backend.search_traces(TraceQuery(limit=10))

        assert [t.trace_id for t in result] == ["t1"]

    async def test_rows_with_no_trace_id_are_skipped(self) -> None:
        backend = _backend()
        row_no_trace = _query_row("t1", "s1")
        row_no_trace["data"]["trace.trace_id"] = None
        backend._run_search = AsyncMock(return_value=[row_no_trace])  # type: ignore[method-assign]

        result = await backend.search_traces(TraceQuery(limit=10))

        assert result == []

    async def test_enriches_hydrated_trace_with_search_row_gen_ai_attributes(self) -> None:
        backend = _backend()
        backend._run_search = AsyncMock(  # type: ignore[method-assign]
            return_value=[_query_row("t1", "s1", **{"gen_ai.system": "openai"})]
        )

        result = await backend.search_traces(TraceQuery(limit=10))

        assert len(result) == 1
        assert result[0].spans[0].attributes.gen_ai_system == "openai"


class TestRunQueryFullFlow:
    """Exercises the real create-query -> create-query-result -> poll wire-
    up end to end (via list_services/get_service_operations, which don't
    mock _run_search/_run_query the way the other test classes above do) -
    every other test in this file mocks at least one layer of this chain,
    so nothing else actually proves the three real HTTP calls are sequenced
    and parsed correctly together."""

    async def test_list_services_full_flow(self, fake_json_client: Callable[..., Any]) -> None:
        backend = _backend()
        backend._client = fake_json_client(
            {"id": "query-1"},  # POST /1/queries/{dataset}
            {"id": "result-1"},  # POST /1/query_results/{dataset}
            {  # GET /1/query_results/{dataset}/{resultId}
                "complete": True,
                "data": {
                    "results": [
                        {"data": {"service.name": "svc-a"}},
                        {"data": {"service.name": "svc-b"}},
                    ]
                },
            },
        )

        result = await backend.list_services()

        assert result == ["svc-a", "svc-b"]

    async def test_get_service_operations_full_flow(
        self, fake_json_client: Callable[..., Any]
    ) -> None:
        backend = _backend()
        backend._client = fake_json_client(
            {"id": "query-1"},
            {"id": "result-1"},
            {"complete": True, "data": {"results": [{"data": {"name": "op-a"}}]}},
        )

        result = await backend.get_service_operations("svc-a")

        assert result == ["op-a"]

    async def test_search_spans_full_flow(self, fake_json_client: Callable[..., Any]) -> None:
        backend = _backend()
        backend._client = fake_json_client(
            {"id": "query-1"},
            {"id": "result-1"},
            {"complete": True, "data": {"results": [_query_row("t1", "s1")]}},
        )

        result = await backend.search_spans(SpanQuery(limit=10))

        assert len(result) == 1
        assert result[0].span_id == "s1"


class TestRunSearchDynamicBreakdowns:
    """Regression for the fix described in _run_search's own docstring:
    filtering on an attribute outside the fixed _SEARCH_BREAKDOWNS/
    _GEN_AI_BREAKDOWNS allowlist (e.g. gen_ai.request.temperature) must
    still get that column back in the result rows, since Honeycomb's Query
    API only ever returns data for columns actually requested as
    breakdowns - regardless of what the server-side filter matched on."""

    async def test_filter_on_non_breakdown_column_is_added_to_breakdowns(self) -> None:
        backend = _backend()
        backend._run_query = AsyncMock(return_value=[])  # type: ignore[method-assign]

        await backend._run_search(
            [
                Filter(
                    field="gen_ai.request.temperature",
                    operator=FilterOperator.GT,
                    value=0.5,
                    value_type=FilterType.NUMBER,
                )
            ],
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
            limit=10,
        )

        backend._run_query.assert_awaited_once()
        breakdowns = backend._run_query.await_args.kwargs["breakdowns"]
        assert "gen_ai.request.temperature" in breakdowns

    async def test_filter_on_already_present_breakdown_column_is_not_duplicated(self) -> None:
        backend = _backend()
        backend._run_query = AsyncMock(return_value=[])  # type: ignore[method-assign]

        await backend._run_search(
            [
                Filter(
                    field="service.name",
                    operator=FilterOperator.EQUALS,
                    value="svc",
                    value_type=FilterType.STRING,
                )
            ],
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
            limit=10,
        )

        breakdowns = backend._run_query.await_args.kwargs["breakdowns"]
        assert breakdowns.count("service.name") == 1

    async def test_dynamically_added_breakdown_column_surfaces_in_parsed_span(
        self, fake_json_client: Callable[..., Any]
    ) -> None:
        """End-to-end: the extra breakdown column must actually come back
        on the row and land in SpanData.attributes, not just be present in
        the outgoing query spec."""
        backend = _backend()
        row = _query_row("t1", "s1", **{"gen_ai.request.temperature": 0.7})
        backend._client = fake_json_client(
            {"id": "query-1"},
            {"id": "result-1"},
            {"complete": True, "data": {"results": [row]}},
        )

        result = await backend.search_spans(
            SpanQuery(
                limit=10,
                filters=[
                    Filter(
                        field="gen_ai.request.temperature",
                        operator=FilterOperator.GT,
                        value=0.5,
                        value_type=FilterType.NUMBER,
                    )
                ],
            )
        )

        assert len(result) == 1
        assert result[0].attributes.gen_ai_request_temperature == 0.7


class TestBatchFetchTraceRowsCapWarning:
    """Regression: a trace that loses only SOME (not all) of its spans to
    the _MAX_QUERY_ROWS hydration cap previously had no warning signal -
    only a trace that lost every span was caught elsewhere (as "no spans
    found"). A partially-truncated trace can silently look healthy (missing
    error span) or faster than it was (missing longest span)."""

    async def test_hitting_the_row_cap_logs_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = _backend()
        backend._run_search = AsyncMock(  # type: ignore[method-assign]
            return_value=[_query_row("t1", f"s{i}") for i in range(_MAX_QUERY_ROWS)]
        )

        with caplog.at_level("WARNING"):
            await backend._batch_fetch_trace_rows(["t1"])

        assert any(
            "maximum" in r.message and str(_MAX_QUERY_ROWS) in r.message for r in caplog.records
        )

    async def test_below_the_cap_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        backend = _backend()
        backend._run_search = AsyncMock(return_value=[_query_row("t1", "s1")])  # type: ignore[method-assign]

        with caplog.at_level("WARNING"):
            await backend._batch_fetch_trace_rows(["t1"])

        assert caplog.records == []

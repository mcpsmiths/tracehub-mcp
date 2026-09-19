"""Tests for the New Relic backend.

Fixtures below are hand-built from a completed deep-research pass over
docs.newrelic.com (NerdGraph structure, NRQL syntax, the Span event's
standard attribute set) - not against a live account (none was available).
See the module docstring in ``opentelemetry_mcp/backends/newrelic.py`` for
the specific things the research couldn't pin down (the exact shape of the
``attributes`` field inside a ``distributedTracing.trace`` span, and the
account's real data-retention window) and how this implementation handles
each.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from opentelemetry_mcp.backends.newrelic import _MAX_TRACES_TO_HYDRATE, NewRelicBackend
from opentelemetry_mcp.models import Filter, FilterOperator, FilterType, SpanQuery, TraceQuery

FAKE_USER_KEY = "fake-newrelic-userkey-1"
FAKE_ACCOUNT_ID = "1234567"


def _backend() -> NewRelicBackend:
    return NewRelicBackend(
        url="https://api.newrelic.com/graphql",
        api_key=FAKE_USER_KEY,
        account_id=FAKE_ACCOUNT_ID,
    )


def test_newrelic_backend_requires_api_key() -> None:
    with pytest.raises(ValueError, match="requires a User API key"):
        NewRelicBackend(
            url="https://api.newrelic.com/graphql", api_key=None, account_id=FAKE_ACCOUNT_ID
        )


def test_newrelic_backend_requires_account_id() -> None:
    with pytest.raises(ValueError, match="requires an account ID"):
        NewRelicBackend(
            url="https://api.newrelic.com/graphql", api_key=FAKE_USER_KEY, account_id=None
        )


def test_newrelic_backend_rejects_non_numeric_account_id() -> None:
    with pytest.raises(ValueError, match="numeric account ID"):
        NewRelicBackend(
            url="https://api.newrelic.com/graphql", api_key=FAKE_USER_KEY, account_id="not-a-number"
        )


def test_newrelic_backend_initialization() -> None:
    backend = NewRelicBackend(
        url="https://api.newrelic.com/graphql",
        api_key=FAKE_USER_KEY,
        account_id=FAKE_ACCOUNT_ID,
        timeout=15.0,
    )
    assert backend.url == "https://api.newrelic.com/graphql"
    assert backend.api_key == FAKE_USER_KEY
    assert backend.account_id == 1234567
    assert backend.timeout == 15.0


def test_newrelic_client_disables_redirects() -> None:
    """API-Key is a non-standard header (like Datadog's DD-API-KEY) that
    httpx does not strip on cross-origin redirects, so follow_redirects
    must be disabled to avoid leaking it to an unexpected host."""
    backend = _backend()
    assert backend.client.follow_redirects is False


def test_newrelic_client_headers() -> None:
    backend = _backend()
    client = backend.client
    assert client.headers["API-Key"] == FAKE_USER_KEY


class TestFilterToNrqlCondition:
    """Filter -> NRQL WHERE condition conversion."""

    def test_equals_string(self) -> None:
        backend = _backend()
        f = Filter(
            field="service.name",
            operator=FilterOperator.EQUALS,
            value="svc",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_nrql_condition(f) == "service.name = 'svc'"

    def test_not_equals(self) -> None:
        backend = _backend()
        f = Filter(
            field="service.name",
            operator=FilterOperator.NOT_EQUALS,
            value="svc",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_nrql_condition(f) == "service.name != 'svc'"

    def test_numeric_equals_is_unquoted(self) -> None:
        backend = _backend()
        f = Filter(
            field="duration", operator=FilterOperator.EQUALS, value=42, value_type=FilterType.NUMBER
        )
        assert backend._filter_to_nrql_condition(f) == "duration.ms = 42"

    def test_gt_gte_lt_lte(self) -> None:
        backend = _backend()
        for operator, symbol in (
            (FilterOperator.GT, ">"),
            (FilterOperator.GTE, ">="),
            (FilterOperator.LT, "<"),
            (FilterOperator.LTE, "<="),
        ):
            f = Filter(field="duration", operator=operator, value=100, value_type=FilterType.NUMBER)
            assert backend._filter_to_nrql_condition(f) == f"duration.ms {symbol} 100"

    def test_gt_with_non_numeric_value_is_rejected(self) -> None:
        backend = _backend()
        f = Filter(
            field="duration",
            operator=FilterOperator.GT,
            value="not-a-number",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_nrql_condition(f) is None

    def test_exists(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system", operator=FilterOperator.EXISTS, value_type=FilterType.STRING
        )
        assert backend._filter_to_nrql_condition(f) == "gen_ai.system IS NOT NULL"

    def test_not_exists(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system", operator=FilterOperator.NOT_EXISTS, value_type=FilterType.STRING
        )
        assert backend._filter_to_nrql_condition(f) == "gen_ai.system IS NULL"

    def test_in(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system",
            operator=FilterOperator.IN,
            values=["openai", "anthropic"],
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_nrql_condition(f) == "gen_ai.system IN ('openai', 'anthropic')"

    def test_field_name_mapping(self) -> None:
        backend = _backend()
        assert backend._newrelic_field("service.name") == "service.name"
        assert backend._newrelic_field("operation_name") == "name"
        assert backend._newrelic_field("duration") == "duration.ms"
        assert backend._newrelic_field("status") == "otel.status_code"
        assert backend._newrelic_field("gen_ai.system") == "gen_ai.system"


class TestFieldNameInjection:
    def test_unsafe_field_name_is_rejected(self) -> None:
        backend = _backend()
        f = Filter(
            field="x) OR (1=1",
            operator=FilterOperator.EQUALS,
            value="v",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_nrql_condition(f) is None

    def test_build_where_skips_rejected_filters(self) -> None:
        backend = _backend()
        safe = Filter(
            field="service.name",
            operator=FilterOperator.EQUALS,
            value="svc",
            value_type=FilterType.STRING,
        )
        unsafe = Filter(
            field="x) OR (1=1",
            operator=FilterOperator.EQUALS,
            value="v",
            value_type=FilterType.STRING,
        )
        assert backend._build_nrql_where([safe, unsafe]) == "service.name = 'svc'"


class TestEscaping:
    def test_embedded_single_quote_is_doubled(self) -> None:
        backend = _backend()
        assert backend._escape_nrql_string("O'Brien") == "'O''Brien'"

    def test_embedded_backslash_is_escaped(self) -> None:
        backend = _backend()
        assert backend._escape_nrql_string("a\\b") == "'a\\\\b'"

    def test_format_value_bool(self) -> None:
        backend = _backend()
        assert backend._format_nrql_value(True) == "true"
        assert backend._format_nrql_value(False) == "false"

    def test_format_value_numeric(self) -> None:
        backend = _backend()
        assert backend._format_nrql_value(42) == "42"
        assert backend._format_nrql_value(3.5) == "3.5"


class TestInferStatus:
    def test_ok(self) -> None:
        assert _backend()._infer_status("OK") == "OK"
        assert _backend()._infer_status("ok") == "OK"

    def test_error(self) -> None:
        assert _backend()._infer_status("ERROR") == "ERROR"
        assert _backend()._infer_status("error") == "ERROR"

    def test_unrecognized_defaults_to_unset(self) -> None:
        assert _backend()._infer_status("UNSET") == "UNSET"
        assert _backend()._infer_status(None) == "UNSET"
        assert _backend()._infer_status(123) == "UNSET"


class TestParseTimestamp:
    def test_epoch_millis_int(self) -> None:
        backend = _backend()
        result = backend._parse_newrelic_timestamp(1704067200000)
        assert result == datetime(2024, 1, 1, tzinfo=UTC)

    def test_epoch_millis_string(self) -> None:
        backend = _backend()
        result = backend._parse_newrelic_timestamp("1704067200000")
        assert result == datetime(2024, 1, 1, tzinfo=UTC)

    def test_none_returns_none(self) -> None:
        assert _backend()._parse_newrelic_timestamp(None) is None

    def test_unparseable_returns_none(self) -> None:
        assert _backend()._parse_newrelic_timestamp("not-a-timestamp") is None

    def test_bool_returns_none(self) -> None:
        """bool is an int subclass in Python - True/False must not be
        silently accepted as a valid timestamp."""
        assert _backend()._parse_newrelic_timestamp(True) is None


class TestParseNewRelicRow:
    def _row(self, **overrides: Any) -> dict[str, Any]:
        row = {
            "id": "span1",
            "trace.id": "trace1",
            "parent.id": None,
            "name": "chat_completion",
            "service.name": "my-llm-app",
            "timestamp": 1704067200000,
            "duration.ms": 150.5,
            "otel.status_code": "OK",
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4",
        }
        row.update(overrides)
        return row

    def test_happy_path(self) -> None:
        backend = _backend()
        span = backend._parse_newrelic_row(self._row())
        assert span is not None
        assert span.span_id == "span1"
        assert span.trace_id == "trace1"
        assert span.parent_span_id is None
        assert span.operation_name == "chat_completion"
        assert span.service_name == "my-llm-app"
        assert span.duration_ms == 150.5
        assert span.status == "OK"
        assert span.attributes.gen_ai_system == "openai"
        assert span.attributes.gen_ai_request_model == "gpt-4"

    def test_missing_span_id_rejected(self) -> None:
        backend = _backend()
        assert backend._parse_newrelic_row(self._row(id=None)) is None

    def test_missing_trace_id_rejected(self) -> None:
        backend = _backend()
        assert backend._parse_newrelic_row(self._row(**{"trace.id": None})) is None

    def test_missing_operation_name_rejected(self) -> None:
        backend = _backend()
        assert backend._parse_newrelic_row(self._row(name=None)) is None

    def test_missing_service_name_rejected(self) -> None:
        backend = _backend()
        assert backend._parse_newrelic_row(self._row(**{"service.name": None})) is None

    def test_missing_timestamp_rejected(self) -> None:
        backend = _backend()
        assert backend._parse_newrelic_row(self._row(timestamp=None)) is None

    def test_missing_duration_rejected(self) -> None:
        backend = _backend()
        assert backend._parse_newrelic_row(self._row(**{"duration.ms": None})) is None

    def test_negative_duration_rejected(self) -> None:
        backend = _backend()
        assert backend._parse_newrelic_row(self._row(**{"duration.ms": -5.0})) is None

    def test_parent_id_is_carried_through(self) -> None:
        backend = _backend()
        span = backend._parse_newrelic_row(self._row(**{"parent.id": "parent1"}))
        assert span is not None
        assert span.parent_span_id == "parent1"


class TestParseNewRelicGraphQLSpan:
    def _span(self, **overrides: Any) -> dict[str, Any]:
        span = {
            "id": "span1",
            "traceId": "trace1",
            "parentId": None,
            "name": "chat_completion",
            "durationMs": 150.5,
            "timestamp": 1704067200000,
            "attributes": {
                "service.name": "my-llm-app",
                "otel.status_code": "OK",
                "gen_ai.system": "openai",
            },
        }
        span.update(overrides)
        return span

    def test_happy_path(self) -> None:
        backend = _backend()
        span = backend._parse_newrelic_graphql_span(self._span(), "trace1")
        assert span is not None
        assert span.span_id == "span1"
        assert span.trace_id == "trace1"
        assert span.operation_name == "chat_completion"
        assert span.service_name == "my-llm-app"
        assert span.status == "OK"
        assert span.attributes.gen_ai_system == "openai"

    def test_non_dict_attributes_defaults_to_empty(self) -> None:
        backend = _backend()
        span = backend._parse_newrelic_graphql_span(self._span(attributes="not-a-dict"), "trace1")
        # service.name lives inside attributes - with attributes coerced to
        # {}, there is no service name, so the span is rejected rather than
        # fabricating one.
        assert span is None

    def test_missing_service_name_rejected(self) -> None:
        backend = _backend()
        span = backend._parse_newrelic_graphql_span(self._span(attributes={}), "trace1")
        assert span is None

    def test_missing_trace_id_falls_back_to_requested_trace_id(self) -> None:
        backend = _backend()
        span = backend._parse_newrelic_graphql_span(self._span(traceId=None), "requested-trace")
        assert span is not None
        assert span.trace_id == "requested-trace"

    def test_missing_duration_rejected(self) -> None:
        backend = _backend()
        span = backend._parse_newrelic_graphql_span(self._span(durationMs=None), "trace1")
        assert span is None


class TestGroupIntoTrace:
    def test_root_detected_by_no_parent(self) -> None:
        backend = _backend()
        root = backend._parse_newrelic_row(
            {
                "id": "root",
                "trace.id": "t1",
                "parent.id": None,
                "name": "root-op",
                "service.name": "svc",
                "timestamp": 1704067200000,
                "duration.ms": 100.0,
                "otel.status_code": "OK",
            }
        )
        assert root is not None
        trace = backend._group_into_trace("t1", [root])
        assert trace.root_operation == "root-op"
        assert trace.service_name == "svc"
        assert trace.status == "OK"

    def test_any_error_span_makes_trace_status_error(self) -> None:
        backend = _backend()
        root = backend._parse_newrelic_row(
            {
                "id": "root",
                "trace.id": "t1",
                "parent.id": None,
                "name": "root-op",
                "service.name": "svc",
                "timestamp": 1704067200000,
                "duration.ms": 100.0,
                "otel.status_code": "OK",
            }
        )
        child = backend._parse_newrelic_row(
            {
                "id": "child",
                "trace.id": "t1",
                "parent.id": "root",
                "name": "child-op",
                "service.name": "svc",
                "timestamp": 1704067200000,
                "duration.ms": 50.0,
                "otel.status_code": "ERROR",
            }
        )
        assert root is not None and child is not None
        trace = backend._group_into_trace("t1", [root, child])
        assert trace.status == "ERROR"


class TestGetTrace:
    async def test_happy_path(self, fake_json_client: Callable[..., Any]) -> None:
        backend = _backend()
        backend._client = fake_json_client(
            {
                "data": {
                    "actor": {
                        "distributedTracing": {
                            "trace": {
                                "spans": [
                                    {
                                        "id": "s1",
                                        "traceId": "t1",
                                        "parentId": None,
                                        "name": "root-op",
                                        "durationMs": 100.0,
                                        "timestamp": 1704067200000,
                                        "attributes": {"service.name": "svc"},
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        )

        trace = await backend.get_trace("t1")

        assert trace.trace_id == "t1"
        assert len(trace.spans) == 1
        assert trace.spans[0].span_id == "s1"

    async def test_no_trace_found_raises(self, fake_json_client: Callable[..., Any]) -> None:
        backend = _backend()
        backend._client = fake_json_client(
            {"data": {"actor": {"distributedTracing": {"trace": None}}}}
        )

        with pytest.raises(ValueError, match="No trace found"):
            await backend.get_trace("missing")

    async def test_unexpected_spans_shape_raises(
        self, fake_json_client: Callable[..., Any]
    ) -> None:
        backend = _backend()
        backend._client = fake_json_client(
            {"data": {"actor": {"distributedTracing": {"trace": {"spans": "not-a-list"}}}}}
        )

        with pytest.raises(ValueError, match="Unexpected New Relic trace response shape"):
            await backend.get_trace("t1")

    async def test_mismatched_trace_id_is_filtered_out(
        self, fake_json_client: Callable[..., Any]
    ) -> None:
        backend = _backend()
        backend._client = fake_json_client(
            {
                "data": {
                    "actor": {
                        "distributedTracing": {
                            "trace": {
                                "spans": [
                                    {
                                        "id": "s1",
                                        "traceId": "different-trace",
                                        "name": "op",
                                        "durationMs": 100.0,
                                        "timestamp": 1704067200000,
                                        "attributes": {"service.name": "svc"},
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        )

        with pytest.raises(ValueError, match="No spans found"):
            await backend.get_trace("t1")


class TestCallNerdgraph:
    async def test_graphql_errors_raise(self, fake_json_client: Callable[..., Any]) -> None:
        backend = _backend()
        backend._client = fake_json_client({"errors": [{"message": "boom"}]})

        with pytest.raises(ValueError, match="NerdGraph query returned errors"):
            await backend._call_nerdgraph("query {}", {})

    async def test_missing_data_raises(self, fake_json_client: Callable[..., Any]) -> None:
        backend = _backend()
        backend._client = fake_json_client({"no_data_key": True})

        with pytest.raises(ValueError, match="expected a 'data' object"):
            await backend._call_nerdgraph("query {}", {})

    async def test_non_object_response_raises(self, fake_json_client: Callable[..., Any]) -> None:
        backend = _backend()
        backend._client = fake_json_client(["not", "an", "object"])

        with pytest.raises(ValueError, match="expected an object"):
            await backend._call_nerdgraph("query {}", {})


class TestListServices:
    async def test_happy_path(self, fake_json_client: Callable[..., Any]) -> None:
        backend = _backend()
        backend._client = fake_json_client(
            {
                "data": {
                    "actor": {"account": {"nrql": {"results": [{"services": ["svc-a", "svc-b"]}]}}}
                }
            }
        )

        result = await backend.list_services()

        assert result == ["svc-a", "svc-b"]

    async def test_empty_results_returns_empty_list(
        self, fake_json_client: Callable[..., Any]
    ) -> None:
        backend = _backend()
        backend._client = fake_json_client(
            {"data": {"actor": {"account": {"nrql": {"results": []}}}}}
        )

        assert await backend.list_services() == []

    async def test_unexpected_shape_returns_empty_list(
        self, fake_json_client: Callable[..., Any]
    ) -> None:
        backend = _backend()
        backend._client = fake_json_client(
            {"data": {"actor": {"account": {"nrql": {"results": "not-a-list"}}}}}
        )

        assert await backend.list_services() == []


class TestGetServiceOperations:
    async def test_happy_path(self, fake_json_client: Callable[..., Any]) -> None:
        backend = _backend()
        backend._client = fake_json_client(
            {
                "data": {
                    "actor": {"account": {"nrql": {"results": [{"operations": ["op-a", "op-b"]}]}}}
                }
            }
        )

        result = await backend.get_service_operations("svc-a")

        assert result == ["op-a", "op-b"]


class TestHealthCheck:
    async def test_healthy(self, fake_json_client: Callable[..., Any]) -> None:
        backend = _backend()
        backend._client = fake_json_client({"data": {"actor": {"user": {"name": "me"}}}})

        result = await backend.health_check()

        assert result.status == "healthy"
        assert result.backend == "newrelic"

    async def test_unhealthy_wraps_exception(self, fake_json_client: Callable[..., Any]) -> None:
        backend = _backend()
        backend._client = fake_json_client({"errors": [{"message": "unauthorized"}]})

        result = await backend.health_check()

        assert result.status == "unhealthy"
        assert result.error is not None


class TestSearchSpans:
    async def test_happy_path(self, fake_json_client: Callable[..., Any]) -> None:
        backend = _backend()
        backend._client = fake_json_client(
            {
                "data": {
                    "actor": {
                        "account": {
                            "nrql": {
                                "results": [
                                    {
                                        "id": "s1",
                                        "trace.id": "t1",
                                        "parent.id": None,
                                        "name": "op",
                                        "service.name": "svc",
                                        "timestamp": 1704067200000,
                                        "duration.ms": 100.0,
                                        "otel.status_code": "OK",
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        )

        result = await backend.search_spans(SpanQuery(limit=10))

        assert len(result) == 1
        assert result[0].span_id == "s1"


def _search_row(span_id: str, trace_id: str, **overrides: Any) -> dict[str, Any]:
    row = {
        "id": span_id,
        "trace.id": trace_id,
        "parent.id": None,
        "name": "op",
        "service.name": "svc",
        "timestamp": 1704067200000,
        "duration.ms": 100.0,
        "otel.status_code": "OK",
    }
    row.update(overrides)
    return row


def _hydrated_trace(backend: NewRelicBackend, trace_id: str, span_id: str) -> Any:
    span = backend._parse_newrelic_graphql_span(
        {
            "id": span_id,
            "traceId": trace_id,
            "parentId": None,
            "name": "op",
            "durationMs": 100.0,
            "timestamp": 1704067200000,
            "attributes": {"service.name": "svc"},
        },
        trace_id,
    )
    assert span is not None
    return backend._group_into_trace(trace_id, [span])


class TestSearchTraces:
    """search_traces's real search-then-hydrate flow: discover distinct
    trace_ids from an NRQL search, hydrate each via get_trace, overlay the
    richer search-row attributes/status back onto the hydrated result (see
    _enrich_trace_with_search_rows - added specifically because get_trace's
    GraphQL hydration path relies on an unverified assumption about the
    shape of distributedTracing.trace's attributes field)."""

    async def test_happy_path_hydrates_via_get_trace(self) -> None:
        backend = _backend()
        backend._run_nrql_search = AsyncMock(return_value=[_search_row("s1", "t1")])  # type: ignore[method-assign]
        backend.get_trace = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda trace_id: _hydrated_trace(backend, trace_id, "s1")
        )

        result = await backend.search_traces(TraceQuery(limit=10))

        assert len(result) == 1
        assert result[0].trace_id == "t1"

    async def test_no_matching_spans_returns_empty_list(self) -> None:
        backend = _backend()
        backend._run_nrql_search = AsyncMock(return_value=[])  # type: ignore[method-assign]
        backend.get_trace = AsyncMock()  # type: ignore[method-assign]

        result = await backend.search_traces(TraceQuery(limit=10))

        assert result == []
        backend.get_trace.assert_not_awaited()

    async def test_discovers_and_hydrates_multiple_distinct_traces(self) -> None:
        backend = _backend()
        backend._run_nrql_search = AsyncMock(  # type: ignore[method-assign]
            return_value=[_search_row("s1", "t1"), _search_row("s2", "t2")]
        )
        backend.get_trace = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda trace_id: _hydrated_trace(backend, trace_id, "s")
        )

        result = await backend.search_traces(TraceQuery(limit=10))

        assert {t.trace_id for t in result} == {"t1", "t2"}
        assert backend.get_trace.await_count == 2

    async def test_groups_multiple_rows_under_the_same_trace_id_into_one_hydration_call(
        self,
    ) -> None:
        """Two search rows sharing a trace_id must hydrate that trace
        exactly once, not once per row."""
        backend = _backend()
        backend._run_nrql_search = AsyncMock(  # type: ignore[method-assign]
            return_value=[_search_row("s1", "t1"), _search_row("s2", "t1")]
        )
        backend.get_trace = AsyncMock(  # type: ignore[method-assign]
            return_value=_hydrated_trace(backend, "t1", "s1")
        )

        result = await backend.search_traces(TraceQuery(limit=10))

        assert len(result) == 1
        backend.get_trace.assert_awaited_once_with("t1")

    async def test_rows_with_no_trace_id_are_skipped(self) -> None:
        backend = _backend()
        row_no_trace = _search_row("s1", "t1")
        del row_no_trace["trace.id"]
        backend._run_nrql_search = AsyncMock(return_value=[row_no_trace])  # type: ignore[method-assign]
        backend.get_trace = AsyncMock()  # type: ignore[method-assign]

        result = await backend.search_traces(TraceQuery(limit=10))

        assert result == []
        backend.get_trace.assert_not_awaited()

    async def test_truncates_hydration_to_max_traces_to_hydrate(self) -> None:
        """More distinct trace_ids than _MAX_TRACES_TO_HYDRATE are
        discovered - only the cap's worth are actually hydrated, rather than
        silently hydrating everything or crashing."""
        backend = _backend()
        rows = [_search_row(f"s{i}", f"t{i}") for i in range(_MAX_TRACES_TO_HYDRATE + 5)]
        backend._run_nrql_search = AsyncMock(return_value=rows)  # type: ignore[method-assign]
        backend.get_trace = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda trace_id: _hydrated_trace(backend, trace_id, "s")
        )

        result = await backend.search_traces(TraceQuery(limit=1000))

        assert backend.get_trace.await_count == _MAX_TRACES_TO_HYDRATE
        assert len(result) == _MAX_TRACES_TO_HYDRATE

    async def test_a_failed_hydration_is_logged_and_skipped_not_propagated(self) -> None:
        """One trace's get_trace() call raising must not abort the whole
        search - the other, successfully-hydrated trace is still returned."""
        backend = _backend()
        backend._run_nrql_search = AsyncMock(  # type: ignore[method-assign]
            return_value=[_search_row("s1", "t1"), _search_row("s2", "t2")]
        )

        async def _get_trace(trace_id: str) -> Any:
            if trace_id == "t1":
                raise ValueError("No spans found for trace t1")
            return _hydrated_trace(backend, trace_id, "s2")

        backend.get_trace = AsyncMock(side_effect=_get_trace)  # type: ignore[method-assign]

        result = await backend.search_traces(TraceQuery(limit=10))

        assert len(result) == 1
        assert result[0].trace_id == "t2"

    async def test_enriches_hydrated_trace_with_search_row_gen_ai_attributes(self) -> None:
        """Regression for the case get_trace()'s GraphQL hydration path
        relies on an unverified attributes shape - the original NRQL search
        row's gen_ai.* attributes must survive onto the final result even if
        get_trace()'s own hydration didn't carry them."""
        backend = _backend()
        backend._run_nrql_search = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                _search_row(
                    "s1", "t1", **{"gen_ai.system": "openai", "gen_ai.request.model": "gpt-4"}
                )
            ]
        )
        # Simulates get_trace() hydrating successfully but with NO gen_ai.*
        # attributes at all - exactly the failure mode this backend's own
        # module docstring flags as unverified/possible.
        backend.get_trace = AsyncMock(  # type: ignore[method-assign]
            return_value=_hydrated_trace(backend, "t1", "s1")
        )

        result = await backend.search_traces(TraceQuery(limit=10))

        assert len(result) == 1
        enriched_span = result[0].spans[0]
        assert enriched_span.attributes.gen_ai_system == "openai"
        assert enriched_span.attributes.gen_ai_request_model == "gpt-4"

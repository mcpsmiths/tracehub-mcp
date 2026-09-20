"""Regression test for the Jaeger backend silently dropping spans.

Fixture is a real raw Jaeger /api/traces/{id} response shape (confirmed
against a live jaegertracing/all-in-one container during an end-to-end dry
run) for a trace containing one span with a real
gen_ai.response.finish_reasons tag - which Jaeger's tag model always
serializes as a JSON-encoded string, never a native list.
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

from opentelemetry_mcp.backends.jaeger import JaegerBackend
from opentelemetry_mcp.models import (
    Filter,
    FilterOperator,
    FilterType,
    SpanQuery,
    TraceQuery,
)


def _raw_trace_with_finish_reasons_tag() -> dict[str, Any]:
    return {
        "traceID": "abc123",
        "spans": [
            {
                "traceID": "abc123",
                "spanID": "root1",
                "operationName": "handle_checkout_request",
                "references": [],
                "startTime": 1000000,
                "duration": 50000,
                "tags": [{"key": "otel.status_code", "type": "string", "value": "OK"}],
                "logs": [],
                "processID": "p1",
            },
            {
                "traceID": "abc123",
                "spanID": "llm1",
                "operationName": "llm_summarize_cart",
                "references": [{"refType": "CHILD_OF", "traceID": "abc123", "spanID": "root1"}],
                "startTime": 1010000,
                "duration": 20000,
                "tags": [
                    {"key": "gen_ai.system", "type": "string", "value": "openai"},
                    {"key": "gen_ai.request.model", "type": "string", "value": "gpt-4"},
                    # This is exactly how Jaeger represents a list-valued
                    # OTel attribute - a JSON-encoded string, not a list.
                    {
                        "key": "gen_ai.response.finish_reasons",
                        "type": "string",
                        "value": '["stop"]',
                    },
                    {"key": "otel.status_code", "type": "string", "value": "OK"},
                ],
                "logs": [],
                "processID": "p1",
            },
        ],
        "processes": {"p1": {"serviceName": "e2e-checkout-service"}},
    }


def test_span_with_finish_reasons_tag_is_not_dropped() -> None:
    backend = JaegerBackend(url="http://localhost:16686")

    trace = backend._parse_jaeger_trace(_raw_trace_with_finish_reasons_tag())

    assert trace is not None
    assert len(trace.spans) == 2
    span_ops = {s.operation_name for s in trace.spans}
    assert "llm_summarize_cart" in span_ops


def test_finish_reasons_tag_resolves_to_a_real_list_on_the_span() -> None:
    backend = JaegerBackend(url="http://localhost:16686")

    trace = backend._parse_jaeger_trace(_raw_trace_with_finish_reasons_tag())

    assert trace is not None
    llm_span = next(s for s in trace.spans if s.operation_name == "llm_summarize_cart")
    assert llm_span.attributes.gen_ai_response_finish_reasons == ["stop"]


class TestListServicesNullData:
    """A fresh/empty Jaeger instance can return {"data": null} rather than
    {"data": []} - the dict.get default only applies when the key is
    missing, not when the key is present with an explicit null value."""

    async def test_null_data_returns_empty_list_not_a_crash(self) -> None:
        backend = JaegerBackend(url="http://localhost:16686")
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"data": None}
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        assert await backend.list_services() == []

    async def test_populated_data_still_resolves_unchanged(self) -> None:
        backend = JaegerBackend(url="http://localhost:16686")
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"data": ["svc-a", "svc-b"]}
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        assert await backend.list_services() == ["svc-a", "svc-b"]


class TestGetServiceOperationsNullData:
    """Same null-data shape as list_services, for the sibling endpoint."""

    async def test_null_data_returns_empty_list_not_a_crash(self) -> None:
        backend = JaegerBackend(url="http://localhost:16686")
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"data": None}
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        assert await backend.get_service_operations("my-service") == []

    async def test_populated_data_still_resolves_unchanged(self) -> None:
        backend = JaegerBackend(url="http://localhost:16686")
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"data": ["chat_completion", "embedding"]}
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        assert await backend.get_service_operations("my-service") == [
            "chat_completion",
            "embedding",
        ]


def _minimal_raw_trace(trace_id: str = "trace-1") -> dict[str, Any]:
    return {
        "traceID": trace_id,
        "spans": [
            {
                "traceID": trace_id,
                "spanID": "span-1",
                "operationName": "op",
                "references": [],
                "startTime": 1_000_000,
                "duration": 5_000,
                "tags": [],
                "logs": [],
                "processID": "p1",
            }
        ],
        "processes": {"p1": {"serviceName": "svc"}},
    }


class TestSearchSpansLimitDoubling:
    """search_spans() builds an internal TraceQuery with limit=query.limit*2
    to fetch enough traces to flatten into spans. SpanQuery.limit's own
    documented valid range is 1-1000, but TraceQuery.limit shares that same
    le=1000 constraint - doubling unclamped raises an unhandled pydantic
    ValidationError for any query.limit above 500."""

    async def test_limit_above_500_does_not_raise_validation_error(self) -> None:
        backend = JaegerBackend(url="http://localhost:16686")
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"data": []}
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        query = SpanQuery(service_name="my-service", limit=600)

        assert await backend.search_spans(query) == []

    async def test_limit_at_1000_still_works(self) -> None:
        backend = JaegerBackend(url="http://localhost:16686")
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"data": []}
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        query = SpanQuery(service_name="my-service", limit=1000)

        assert await backend.search_spans(query) == []


class TestUrlPathEscaping:
    """trace_id/service_name can originate from an external MCP tool call
    and are interpolated into the request URL *path*. Left unescaped, a
    crafted value (e.g. containing '../') could redirect the outbound
    request to a different path - see sentry.py's get_trace, which already
    quotes trace_id (safe='') for the identical reason."""

    async def test_get_trace_encodes_path_traversal_trace_id(self) -> None:
        backend = JaegerBackend(url="http://localhost:16686")
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"data": [_minimal_raw_trace()]}
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        await backend.get_trace("../secret")

        called_path = backend._client.get.call_args[0][0]
        assert called_path == "/api/traces/..%2Fsecret"

    async def test_get_service_operations_encodes_path_traversal_service_name(self) -> None:
        backend = JaegerBackend(url="http://localhost:16686")
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"data": ["op1"]}
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        await backend.get_service_operations("../secret")

        called_path = backend._client.get.call_args[0][0]
        assert called_path == "/api/services/..%2Fsecret/operations"


class TestNativeFilterApplication:
    """native_filters is computed (a service.name EQUALS filter) but
    to_backend_params() only ever reads query.service_name directly, never
    consulting native_filters - so an explicit generic filter on
    service.name, distinct from the service_name convenience parameter, was
    classified as native (skipped by client-side FilterEngine) yet never
    actually pushed into the outgoing request params either, silently
    dropping it."""

    async def test_explicit_service_name_filter_overrides_convenience_param(self) -> None:
        backend = JaegerBackend(url="http://localhost:16686")
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"data": []}
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        query = TraceQuery(
            service_name="service-a",
            filters=[
                Filter(
                    field="service.name",
                    operator=FilterOperator.EQUALS,
                    value="service-b",
                    value_type=FilterType.STRING,
                )
            ],
        )

        await backend.search_traces(query)

        _, kwargs = backend._client.get.call_args
        assert kwargs["params"]["service"] == "service-b"


class TestNullSpanTimingFields:
    """A malformed span with startTime/duration as an explicit JSON null
    (not a missing key) used to raise a TypeError inside
    datetime.fromtimestamp(None / 1_000_000), caught by the broad
    `except Exception` in _parse_jaeger_span and silently dropping the whole
    span - and, if it was the trace's only span, the whole trace."""

    def test_null_start_time_and_duration_does_not_drop_the_span(self) -> None:
        backend = JaegerBackend(url="http://localhost:16686")
        raw_trace = {
            "traceID": "trace-null",
            "spans": [
                {
                    "traceID": "trace-null",
                    "spanID": "span-null",
                    "operationName": "op",
                    "references": [],
                    "startTime": None,
                    "duration": None,
                    "tags": [],
                    "logs": [],
                    "processID": "p1",
                }
            ],
            "processes": {"p1": {"serviceName": "svc"}},
        }

        trace = backend._parse_jaeger_trace(raw_trace)

        assert trace is not None
        assert len(trace.spans) == 1
        assert trace.spans[0].duration_ms == 0
        assert trace.spans[0].start_time == datetime.fromtimestamp(0, tz=UTC)


class TestSpanStartTimeTimezone:
    """datetime.fromtimestamp() with no explicit tz interprets a Jaeger
    epoch timestamp in the server process's local timezone instead of UTC,
    silently skewing every returned timestamp unless the host happens to run
    with TZ=UTC."""

    def test_start_time_is_parsed_as_utc(self) -> None:
        backend = JaegerBackend(url="http://localhost:16686")
        start_time_us = 1_700_000_000_000_000
        raw_trace = {
            "traceID": "trace-utc",
            "spans": [
                {
                    "traceID": "trace-utc",
                    "spanID": "span-utc",
                    "operationName": "op",
                    "references": [],
                    "startTime": start_time_us,
                    "duration": 1_000_000,
                    "tags": [],
                    "logs": [],
                    "processID": "p1",
                }
            ],
            "processes": {"p1": {"serviceName": "svc"}},
        }

        trace = backend._parse_jaeger_trace(raw_trace)

        assert trace is not None
        span = trace.spans[0]
        assert span.start_time.tzinfo == UTC
        assert span.start_time == datetime.fromtimestamp(start_time_us / 1_000_000, tz=UTC)

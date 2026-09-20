"""Tests for Traceloop backend."""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from opentelemetry_mcp.backends.traceloop import TraceloopBackend
from opentelemetry_mcp.constants import GenAI
from opentelemetry_mcp.models import Filter, FilterOperator, FilterType, SpanQuery, TraceQuery


def test_traceloop_backend_requires_api_key() -> None:
    """Test that Traceloop backend requires an API key."""
    with pytest.raises(ValueError, match="requires an API key"):
        TraceloopBackend(url="https://api.traceloop.com/v2", api_key=None)


def test_traceloop_backend_initialization() -> None:
    """Test Traceloop backend initializes correctly."""
    backend = TraceloopBackend(
        url="https://api.traceloop.com/v2",
        api_key="test_key",
        timeout=30.0,
    )

    assert backend.url == "https://api.traceloop.com/v2"
    assert backend.api_key == "test_key"
    assert backend.timeout == 30.0
    assert backend.project_id == "default"


def test_traceloop_client_headers() -> None:
    """Test that Traceloop client has correct headers."""
    backend = TraceloopBackend(
        url="https://api.traceloop.com/v2",
        api_key="test_key",
    )

    client = backend.client
    assert client.headers["Authorization"] == "Bearer test_key"
    assert client.headers["Content-Type"] == "application/json"


def test_build_filters_for_search() -> None:
    """Test filter building for search_traces."""
    query = TraceQuery(
        service_name="my-service",
        operation_name="my-operation",
        gen_ai_system="openai",
        gen_ai_request_model="gpt-4",
        min_duration_ms=1000,
        max_duration_ms=5000,
        has_error=True,
        tags={"custom.tag": "value"},
        limit=50,
    )

    # This would be built by the search_traces method
    # We're just testing the logic

    filters = []

    if query.service_name:
        filters.append({"field": "service_name", "operator": "equals", "value": query.service_name})

    if query.operation_name:
        filters.append({"field": "span_name", "operator": "equals", "value": query.operation_name})

    if query.gen_ai_system:
        filters.append(
            {
                "field": "span_attributes.gen_ai.system",
                "operator": "equals",
                "value": query.gen_ai_system,
            }
        )

    if query.gen_ai_request_model:
        filters.append(
            {
                "field": "span_attributes.gen_ai.request.model",
                "operator": "equals",
                "value": query.gen_ai_request_model,
            }
        )

    if query.min_duration_ms:
        filters.append(
            {
                "field": "duration",
                "operator": "greater_than",
                "value": str(query.min_duration_ms),
            }
        )

    if query.max_duration_ms:
        filters.append(
            {
                "field": "duration",
                "operator": "less_than",
                "value": str(query.max_duration_ms),
            }
        )

    if query.has_error:
        filters.append({"field": "status_code", "operator": "equals", "value": "ERROR"})

    for key, value in query.tags.items():
        filters.append(
            {
                "field": f"span_attributes.{key}",
                "operator": "equals",
                "value": value,
            }
        )

    # Verify all filters were added
    assert len(filters) == 8
    assert filters[0] == {"field": "service_name", "operator": "equals", "value": "my-service"}
    assert filters[1] == {"field": "span_name", "operator": "equals", "value": "my-operation"}
    assert filters[2] == {
        "field": "span_attributes.gen_ai.system",
        "operator": "equals",
        "value": "openai",
    }
    assert filters[3] == {
        "field": "span_attributes.gen_ai.request.model",
        "operator": "equals",
        "value": "gpt-4",
    }
    assert filters[4] == {"field": "duration", "operator": "greater_than", "value": "1000"}
    assert filters[5] == {"field": "duration", "operator": "less_than", "value": "5000"}
    assert filters[6] == {"field": "status_code", "operator": "equals", "value": "ERROR"}
    assert filters[7] == {
        "field": "span_attributes.custom.tag",
        "operator": "equals",
        "value": "value",
    }


def test_convert_root_span_to_trace() -> None:
    """Test converting Traceloop root span to TraceData."""
    backend = TraceloopBackend(url="https://api.traceloop.com/v2", api_key="test_key")

    root_span = {
        "trace_id": "abc123",
        "span_id": "span1",
        "parent_span_id": "",
        "span_name": "workflow",
        "service_name": "my-service",
        "timestamp": 1704120000000,  # milliseconds
        "duration": 3000,  # milliseconds
        "status_code": "OK",
        "span_attributes": {
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4",
            "gen_ai.usage.total_tokens": 1500,
        },
    }

    trace = backend._convert_root_span_to_trace(root_span)

    assert trace is not None
    assert trace.trace_id == "abc123"
    assert len(trace.spans) == 1
    assert trace.spans[0].span_id == "span1"
    assert trace.spans[0].operation_name == "workflow"
    assert trace.service_name == "my-service"
    assert trace.status == "OK"
    assert trace.duration_ms == 3000


def test_convert_spans_to_trace() -> None:
    """Test converting Traceloop spans array to TraceData."""
    backend = TraceloopBackend(url="https://api.traceloop.com/v2", api_key="test_key")

    spans_data = [
        {
            "trace_id": "abc123",
            "span_id": "root",
            "parent_span_id": "",
            "span_name": "workflow",
            "timestamp": 1704120000000,
            "duration": 3000,
            "span_attributes": {"traceloop.workflow.name": "chat"},
        },
        {
            "trace_id": "abc123",
            "span_id": "child",
            "parent_span_id": "root",
            "span_name": "llm.completion",
            "timestamp": 1704120001000,
            "duration": 2000,
            "span_attributes": {
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4",
                "gen_ai.usage.total_tokens": 1500,
            },
        },
    ]

    trace = backend._convert_spans_to_trace("abc123", spans_data)

    assert trace is not None
    assert trace.trace_id == "abc123"
    assert len(trace.spans) == 2
    assert trace.spans[0].span_id == "root"
    assert trace.spans[1].span_id == "child"
    assert trace.spans[1].attributes.gen_ai_system == "openai"


def test_timestamp_conversion() -> None:
    """Test timestamp conversion from milliseconds to datetime."""
    # Traceloop returns timestamps in milliseconds
    timestamp_ms = 1704120000000

    # Convert to datetime
    dt = datetime.fromtimestamp(timestamp_ms / 1000)

    # Verify conversion
    assert dt.year == 2024
    assert dt.month == 1
    assert dt.day == 1


def test_duration_conversion() -> None:
    """Test duration stays in milliseconds."""
    # Traceloop returns duration in milliseconds
    duration_ms = 3500

    # Our internal model also uses milliseconds
    assert float(duration_ms) == 3500.0


_FAKE_API_KEY = "test" + "_key"


class TestFilterToTraceloopGenAiProviderRename:
    """Both gen_ai.system and its OTel semconv v1.37.0 rename,
    gen_ai.provider.name, must map to Traceloop's one underlying llm.vendor
    field - a caller filtering by either name should reach the same data."""

    def _backend(self) -> TraceloopBackend:
        return TraceloopBackend(url="https://api.traceloop.com/v2", api_key=_FAKE_API_KEY)

    def test_gen_ai_system_maps_to_llm_vendor(self) -> None:
        backend = self._backend()
        f = Filter(
            field=GenAI.SYSTEM,
            operator=FilterOperator.EQUALS,
            value="openai",
            value_type=FilterType.STRING,
        )
        converted = backend._filter_to_traceloop(f)
        assert converted is not None
        assert converted["field"] == "llm.vendor"

    def test_gen_ai_provider_name_also_maps_to_llm_vendor(self) -> None:
        backend = self._backend()
        f = Filter(
            field=GenAI.PROVIDER_NAME,
            operator=FilterOperator.EQUALS,
            value="openai",
            value_type=FilterType.STRING,
        )
        converted = backend._filter_to_traceloop(f)
        assert converted is not None
        assert converted["field"] == "llm.vendor"


class TestFilterToTraceloopBooleanSerialization:
    """Filter.value is typed str | int | float | bool | None, and pydantic's
    smart-mode union validation keeps a caller-supplied string like "false"
    as the str "false" rather than coercing it to the bool False. Plain
    Python truthiness on that non-empty string is True, which would
    silently invert a boolean filter's meaning."""

    def _backend(self) -> TraceloopBackend:
        return TraceloopBackend(url="https://api.traceloop.com/v2", api_key=_FAKE_API_KEY)

    def test_string_false_value_serializes_to_false(self) -> None:
        backend = self._backend()
        f = Filter(
            field="gen_ai.request.is_streaming",
            operator=FilterOperator.EQUALS,
            value="false",
            value_type=FilterType.BOOLEAN,
        )

        # Sanity check on the premise: pydantic kept the string, not a bool.
        assert f.value == "false"
        assert isinstance(f.value, str)

        converted = backend._filter_to_traceloop(f)
        assert converted is not None
        assert converted["value"] == "false"

    def test_string_true_value_serializes_to_true(self) -> None:
        backend = self._backend()
        f = Filter(
            field="gen_ai.request.is_streaming",
            operator=FilterOperator.EQUALS,
            value="true",
            value_type=FilterType.BOOLEAN,
        )
        converted = backend._filter_to_traceloop(f)
        assert converted is not None
        assert converted["value"] == "true"

    def test_real_bool_false_still_serializes_to_false(self) -> None:
        backend = self._backend()
        f = Filter(
            field="gen_ai.request.is_streaming",
            operator=FilterOperator.EQUALS,
            value=False,
            value_type=FilterType.BOOLEAN,
        )
        converted = backend._filter_to_traceloop(f)
        assert converted is not None
        assert converted["value"] == "false"

    def test_real_bool_true_still_serializes_to_true(self) -> None:
        backend = self._backend()
        f = Filter(
            field="gen_ai.request.is_streaming",
            operator=FilterOperator.EQUALS,
            value=True,
            value_type=FilterType.BOOLEAN,
        )
        converted = backend._filter_to_traceloop(f)
        assert converted is not None
        assert converted["value"] == "true"


class TestSearchSpansServiceNameFromTopLevelField:
    """Real Traceloop API responses put service_name as a top-level field on
    each span object, not nested inside span_attributes["service.name"] -
    that key is never populated. search_spans (and get_trace, which shares
    the same parsing shape) must read the top-level field."""

    async def test_search_spans_reads_top_level_service_name(self) -> None:
        backend = TraceloopBackend(url="https://api.traceloop.com/v2", api_key=_FAKE_API_KEY)

        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {
            "spans": {
                "data": [
                    {
                        "trace_id": "abc123",
                        "span_id": "span1",
                        "parent_span_id": "",
                        "span_name": "openai.chat",
                        "service_name": "travel-agent-demo2",
                        "timestamp": 1704120000000,
                        "duration": 3000,
                        "status_code": "STATUS_CODE_OK",
                        # span_attributes deliberately omits "service.name" -
                        # this key is never populated by the real API.
                        "span_attributes": {"gen_ai.system": "openai"},
                    }
                ]
            }
        }
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.post = AsyncMock(return_value=fake_response)

        spans = await backend.search_spans(SpanQuery(limit=10))

        assert len(spans) == 1
        assert spans[0].service_name == "travel-agent-demo2"


class TestConvertSpansToTraceServiceNameFromTopLevelField:
    """Same top-level service_name fix as search_spans, for the get_trace
    parsing path (_convert_spans_to_trace)."""

    def test_reads_top_level_service_name(self) -> None:
        backend = TraceloopBackend(url="https://api.traceloop.com/v2", api_key=_FAKE_API_KEY)

        spans_data = [
            {
                "trace_id": "abc123",
                "span_id": "root",
                "parent_span_id": "",
                "span_name": "workflow",
                "service_name": "travel-agent-demo2",
                "timestamp": 1704120000000,
                "duration": 3000,
                "span_attributes": {},
            }
        ]

        trace = backend._convert_spans_to_trace("abc123", spans_data)

        assert trace is not None
        assert trace.spans[0].service_name == "travel-agent-demo2"


class TestConvertSpansToTraceSkipsMalformedSpan:
    """_convert_spans_to_trace (used by get_trace()) must skip a malformed
    span (e.g. missing "duration") and log+continue, matching search_spans's
    own per-span try/except Exception pattern in this same file - not crash
    the entire get_trace() call with a bare KeyError."""

    def test_malformed_span_is_skipped_not_raised(self) -> None:
        backend = TraceloopBackend(url="https://api.traceloop.com/v2", api_key=_FAKE_API_KEY)

        spans_data = [
            {
                "trace_id": "abc123",
                "span_id": "bad-span",
                "parent_span_id": "",
                "span_name": "broken",
                "service_name": "svc",
                "timestamp": 1704120000000,
                # "duration" is deliberately missing
                "span_attributes": {},
            },
            {
                "trace_id": "abc123",
                "span_id": "good-span",
                "parent_span_id": "",
                "span_name": "workflow",
                "service_name": "svc",
                "timestamp": 1704120001000,
                "duration": 2000,
                "span_attributes": {},
            },
        ]

        # Must not raise despite the first span being malformed.
        trace = backend._convert_spans_to_trace("abc123", spans_data)

        assert trace is not None
        assert len(trace.spans) == 1
        assert trace.spans[0].span_id == "good-span"


class TestGetServiceOperationsScopedToService:
    """get_service_operations(service_name) must only return operations
    actually emitted by that service - not every workflow name across the
    whole account."""

    async def test_only_returns_operations_for_requested_service(self) -> None:
        backend = TraceloopBackend(url="https://api.traceloop.com/v2", api_key=_FAKE_API_KEY)

        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {
            "root_spans": {
                "data": [
                    {
                        "trace_id": "t1",
                        "span_id": "s1",
                        "span_name": "service-a-workflow",
                        "service_name": "service-a",
                        "timestamp": 1704120000000,
                        "duration": 1000,
                        "span_attributes": {},
                    },
                    {
                        "trace_id": "t2",
                        "span_id": "s2",
                        "span_name": "service-b-workflow",
                        "service_name": "service-b",
                        "timestamp": 1704120001000,
                        "duration": 2000,
                        "span_attributes": {},
                    },
                ]
            }
        }
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.post = AsyncMock(return_value=fake_response)

        operations = await backend.get_service_operations("service-a")

        assert operations == ["service-a-workflow"]


class TestStartTimeIsUtcAware:
    """datetime.fromtimestamp() without tz=UTC interprets an epoch-ms
    timestamp in the server process's local timezone, silently skewing
    every returned timestamp unless the host runs with TZ=UTC - matching
    the identical fix already applied in jaeger.py/tempo.py/sentry.py."""

    async def test_search_spans_start_time_is_utc_aware(self) -> None:
        backend = TraceloopBackend(url="https://api.traceloop.com/v2", api_key=_FAKE_API_KEY)

        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {
            "spans": {
                "data": [
                    {
                        "trace_id": "abc123",
                        "span_id": "span1",
                        "parent_span_id": "",
                        "span_name": "openai.chat",
                        "service_name": "travel-agent-demo2",
                        "timestamp": 1704120000000,
                        "duration": 3000,
                        "status_code": "STATUS_CODE_OK",
                        "span_attributes": {},
                    }
                ]
            }
        }
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.post = AsyncMock(return_value=fake_response)

        spans = await backend.search_spans(SpanQuery(limit=10))

        offset = spans[0].start_time.utcoffset()
        assert offset is not None
        assert offset.total_seconds() == 0

    async def test_search_traces_root_span_start_time_is_utc_aware(self) -> None:
        # _convert_root_span_to_trace (used by both search_traces and
        # get_service_operations) has its own separate fromtimestamp()
        # call site - get_service_operations only returns operation name
        # strings, so it can't expose start_time to assert on; search_traces
        # returns full TraceData and exercises the identical parsing path.
        backend = TraceloopBackend(url="https://api.traceloop.com/v2", api_key=_FAKE_API_KEY)

        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {
            "root_spans": {
                "data": [
                    {
                        "trace_id": "t1",
                        "span_id": "s1",
                        "span_name": "service-a-workflow",
                        "service_name": "service-a",
                        "timestamp": 1704120000000,
                        "duration": 2000,
                        "span_attributes": {},
                    }
                ]
            }
        }
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.post = AsyncMock(return_value=fake_response)

        traces = await backend.search_traces(TraceQuery(limit=10))

        assert len(traces) == 1
        assert traces[0].start_time.tzinfo is not None
        offset = traces[0].start_time.utcoffset()
        assert offset is not None
        assert offset.total_seconds() == 0


class TestGetTraceEscapesTraceId:
    """trace_id is an unvalidated MCP tool argument used as a URL path
    segment - a crafted value (e.g. containing "../") must not be able to
    redirect this request to a different path. Matches the identical fix
    in jaeger.py/sentry.py/tempo.py for the same get_trace(trace_id) shape."""

    async def test_get_trace_url_escapes_path_traversal_trace_id(self) -> None:
        backend = TraceloopBackend(url="https://api.traceloop.com/v2", api_key=_FAKE_API_KEY)

        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"spans": []}
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        malicious_trace_id = "../admin/config"
        with pytest.raises(ValueError, match="not found"):
            await backend.get_trace(malicious_trace_id)

        requested_endpoint = backend._client.get.call_args[0][0]
        assert "../" not in requested_endpoint
        assert "%2F.." in requested_endpoint or "..%2F" in requested_endpoint

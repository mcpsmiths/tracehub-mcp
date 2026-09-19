"""Tests for Datadog backend.

Fixtures below are hand-built from Datadog's published OpenAPI spec for the
Spans API (v2) - specifically the `Span`/`SpansAttributes` schema in
https://github.com/DataDog/datadog-api-client-python/blob/master/.generator/schemas/v2/openapi.yaml
- not against a live account (none was available). See the module docstring
in `opentelemetry_mcp/backends/datadog.py` for the two things the public
schema does not pin down (where gen_ai.* attributes land, and how error
status is represented) and how this implementation handles both possibilities
it could find documented.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from opentelemetry_mcp.backends.base import MetadataEndpointBlockedError, _RetryingTransport
from opentelemetry_mcp.backends.datadog import (
    _MAX_SEARCH_PAGES,
    _MAX_TRACES_TO_HYDRATE,
    DatadogBackend,
)
from opentelemetry_mcp.models import (
    Filter,
    FilterOperator,
    FilterType,
    SpanQuery,
    TraceData,
    TraceQuery,
)

FAKE_API_KEY = "dd-api1"
FAKE_APP_KEY = "dd-app1"


def test_datadog_backend_requires_api_key() -> None:
    """Test that Datadog backend requires an API key."""
    with pytest.raises(ValueError, match="requires an API key"):
        DatadogBackend(url="https://api.datadoghq.com", api_key=None, app_key=FAKE_APP_KEY)


def test_datadog_backend_requires_app_key() -> None:
    """Test that Datadog backend requires an Application key even with an API key."""
    with pytest.raises(ValueError, match="Application key"):
        DatadogBackend(url="https://api.datadoghq.com", api_key=FAKE_API_KEY, app_key=None)


def test_datadog_backend_rejects_non_https_url() -> None:
    """Test that Datadog backend refuses to send credentials over plain http."""
    with pytest.raises(ValueError, match="https://"):
        DatadogBackend(url="http://api.datadoghq.com", api_key=FAKE_API_KEY, app_key=FAKE_APP_KEY)


def test_datadog_client_disables_redirects() -> None:
    """Test that the client never follows redirects (custom credential headers
    are not stripped by httpx on cross-origin redirects)."""
    backend = DatadogBackend(
        url="https://api.datadoghq.com", api_key=FAKE_API_KEY, app_key=FAKE_APP_KEY
    )
    assert backend.client.follow_redirects is False


def test_datadog_client_still_uses_the_retrying_transport() -> None:
    """Regression test: overriding client() to disable redirects previously
    dropped the shared _RetryingTransport entirely (no connect/timeout retry,
    no slow-request logging), unlike every other backend."""
    backend = DatadogBackend(
        url="https://api.datadoghq.com", api_key=FAKE_API_KEY, app_key=FAKE_APP_KEY
    )
    assert isinstance(backend.client._transport, _RetryingTransport)


async def test_datadog_request_to_metadata_ip_is_blocked_even_with_redirects_disabled() -> None:
    """Datadog's client disables follow_redirects entirely (see
    test_datadog_client_disables_redirects above), so a redirect-to-metadata
    scenario can never even reach httpx's redirect-following logic for this
    backend - there is nothing to follow. The genuinely meaningful
    defense-in-depth case is that _RetryingTransport's metadata-host check
    (see TestMetadataEndpointBlocking in test_base_backend.py) runs on every
    request, including the *initial* one, independent of a client's
    follow_redirects setting - so a request aimed directly at a metadata
    address is still blocked even through Datadog's redirect-disabled
    client."""
    handler_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal handler_calls
        handler_calls += 1
        return httpx.Response(200)

    backend = DatadogBackend(
        url="https://api.datadoghq.com", api_key=FAKE_API_KEY, app_key=FAKE_APP_KEY
    )
    backend._client = httpx.AsyncClient(
        follow_redirects=False,
        transport=_RetryingTransport(wrapped=httpx.MockTransport(handler)),
    )

    with pytest.raises(MetadataEndpointBlockedError):
        await backend.client.get("http://169.254.169.254/latest/meta-data/")

    assert handler_calls == 0


def test_datadog_backend_initialization() -> None:
    """Test Datadog backend initializes correctly with both keys."""
    backend = DatadogBackend(
        url="https://api.datadoghq.com",
        api_key=FAKE_API_KEY,
        app_key=FAKE_APP_KEY,
        timeout=15.0,
    )

    assert backend.url == "https://api.datadoghq.com"
    assert backend.api_key == FAKE_API_KEY
    assert backend.app_key == FAKE_APP_KEY
    assert backend.timeout == 15.0


def test_datadog_client_headers() -> None:
    """Test that Datadog client sends DD-API-KEY and DD-APPLICATION-KEY headers."""
    backend = DatadogBackend(
        url="https://api.datadoghq.com",
        api_key=FAKE_API_KEY,
        app_key=FAKE_APP_KEY,
    )

    client = backend.client
    assert client.headers["DD-API-KEY"] == FAKE_API_KEY
    assert client.headers["DD-APPLICATION-KEY"] == FAKE_APP_KEY
    assert client.headers["Content-Type"] == "application/json"


def _backend() -> DatadogBackend:
    return DatadogBackend(
        url="https://api.datadoghq.com", api_key=FAKE_API_KEY, app_key=FAKE_APP_KEY
    )


class TestBuildDatadogQuery:
    """Test Filter -> Datadog span search query string conversion."""

    def test_equals_facet_field(self) -> None:
        backend = _backend()
        f = Filter(
            field="service.name",
            operator=FilterOperator.EQUALS,
            value="my-service",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_dd_query(f) == 'service:"my-service"'

    def test_equals_custom_attribute_is_at_prefixed(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system",
            operator=FilterOperator.EQUALS,
            value="openai",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_dd_query(f) == '@gen_ai.system:"openai"'

    def test_not_equals(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system",
            operator=FilterOperator.NOT_EQUALS,
            value="openai",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_dd_query(f) == '-@gen_ai.system:"openai"'

    def test_status_error_equals(self) -> None:
        backend = _backend()
        f = Filter(
            field="status",
            operator=FilterOperator.EQUALS,
            value="ERROR",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_dd_query(f) == "status:error"

    def test_status_ok_not_equals(self) -> None:
        """Regression test: the EQUALS branch has always special-cased
        status=="OK" (-> "status:ok"), but the NOT_EQUALS branch was missing
        the equivalent case, so status != "OK" fell through to the generic
        field:value branch and produced the wrong query (-status:"OK"
        instead of -status:ok - wrong case, and quoted instead of bare)."""
        backend = _backend()
        f = Filter(
            field="status",
            operator=FilterOperator.NOT_EQUALS,
            value="OK",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_dd_query(f) == "-status:ok"

    def test_duration_gte_converts_ms_to_ns(self) -> None:
        backend = _backend()
        f = Filter(
            field="duration", operator=FilterOperator.GTE, value=1000, value_type=FilterType.NUMBER
        )
        # 1000ms -> 1_000_000_000ns
        assert backend._filter_to_dd_query(f) == "@duration:[1000000000 TO *]"

    def test_duration_lt_converts_ms_to_ns(self) -> None:
        backend = _backend()
        f = Filter(
            field="duration", operator=FilterOperator.LT, value=5000, value_type=FilterType.NUMBER
        )
        assert backend._filter_to_dd_query(f) == "@duration:{* TO 5000000000}"

    def test_equals_escapes_embedded_quote(self) -> None:
        """A crafted filter value can't inject additional query clauses."""
        backend = _backend()
        f = Filter(
            field="gen_ai.system",
            operator=FilterOperator.EQUALS,
            value='a" OR *:*',
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_dd_query(f) == '@gen_ai.system:"a\\" OR *:*"'

    def test_range_operator_rejects_non_numeric_value(self) -> None:
        """Filter.value_type isn't enforced against the actual Python type,
        so a range operator with a string value must be rejected rather than
        interpolated unchecked into a numeric range expression."""
        backend = _backend()
        f = Filter(
            field="duration",
            operator=FilterOperator.GT,
            value="1000 TO *} OR @duration:{0",
            value_type=FilterType.NUMBER,
        )
        assert backend._filter_to_dd_query(f) is None

    def test_range_operator_rejects_bool_value(self) -> None:
        """bool is an int subclass in Python but not a sensible range operand."""
        backend = _backend()
        f = Filter(
            field="duration", operator=FilterOperator.GTE, value=True, value_type=FilterType.NUMBER
        )
        assert backend._filter_to_dd_query(f) is None

    def test_exists(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system", operator=FilterOperator.EXISTS, value_type=FilterType.STRING
        )
        assert backend._filter_to_dd_query(f) == "@gen_ai.system:*"

    def test_not_exists(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system", operator=FilterOperator.NOT_EXISTS, value_type=FilterType.STRING
        )
        assert backend._filter_to_dd_query(f) == "-@gen_ai.system:*"

    def test_in_builds_or(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system",
            operator=FilterOperator.IN,
            values=["openai", "anthropic"],
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_dd_query(f) == (
            '(@gen_ai.system:"openai" OR @gen_ai.system:"anthropic")'
        )

    def test_build_dd_query_empty_defaults_to_wildcard(self) -> None:
        backend = _backend()
        assert backend._build_dd_query([]) == "*"

    def test_build_dd_query_joins_with_and(self) -> None:
        backend = _backend()
        filters = [
            Filter(
                field="service.name",
                operator=FilterOperator.EQUALS,
                value="svc",
                value_type=FilterType.STRING,
            ),
            Filter(
                field="gen_ai.system",
                operator=FilterOperator.EQUALS,
                value="openai",
                value_type=FilterType.STRING,
            ),
        ]
        assert backend._build_dd_query(filters) == ('service:"svc" AND @gen_ai.system:"openai"')


class TestFieldNameInjection:
    """Filter.field is an unvalidated str reachable from any MCP tool call.

    Unlike the filter *value*, which is always escaped/quoted via
    _escape_dd_query_value, the field name (via _dd_field's `@`-prefixed
    fallback for non-facet fields) used to be spliced directly into the
    query string - so a malicious field name could inject arbitrary
    structure (e.g. breaking out of an AND with an OR/paren group) into the
    Datadog search syntax. Every operator branch must reject an unsafe
    field name instead.
    """

    def test_equals_rejects_injection_field(self) -> None:
        backend = _backend()
        f = Filter(
            field="x) OR (a:b",
            operator=FilterOperator.EQUALS,
            value="v",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_dd_query(f) is None

    def test_exists_rejects_injection_field(self) -> None:
        backend = _backend()
        f = Filter(
            field="x) OR (@span.op:*",
            operator=FilterOperator.EXISTS,
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_dd_query(f) is None

    def test_in_rejects_injection_field(self) -> None:
        backend = _backend()
        f = Filter(
            field="x) OR (a:b",
            operator=FilterOperator.IN,
            values=["v1", "v2"],
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_dd_query(f) is None

    def test_build_dd_query_drops_injection_filter_from_and_join(self) -> None:
        """A rejected field must not leak its raw text into the joined
        query at all - not even as a dropped-but-still-malicious fragment."""
        backend = _backend()
        filters = [
            Filter(
                field="gen_ai.system",
                operator=FilterOperator.EQUALS,
                value="openai",
                value_type=FilterType.STRING,
            ),
            Filter(
                field="x) OR (a:b",
                operator=FilterOperator.EXISTS,
                value_type=FilterType.STRING,
            ),
        ]
        assert backend._build_dd_query(filters) == '@gen_ai.system:"openai"'

    def test_valid_dotted_field_is_still_accepted(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.usage.total_tokens",
            operator=FilterOperator.EQUALS,
            value=100,
            value_type=FilterType.NUMBER,
        )
        assert backend._filter_to_dd_query(f) == '@gen_ai.usage.total_tokens:"100"'


class TestParseDatadogSpan:
    """Test parsing raw Datadog Span resources into SpanData."""

    def test_parse_root_span(self) -> None:
        backend = _backend()
        span_obj = {
            "id": "AAAAAWgN8Xwgr1vKDQAAAABBV2dOOFh3ZzZobm1mWXJFYTR0OA",
            "type": "spans",
            "attributes": {
                "trace_id": "1234567890987654321",
                "span_id": "1234567890987654321",
                "parent_id": "0",
                "service": "my-llm-service",
                "resource_name": "chat_completion",
                "start_timestamp": "2023-01-02T09:42:36.320Z",
                "end_timestamp": "2023-01-02T09:42:36.420Z",
                "tags": ["env:prod", "team:A"],
                "custom": {
                    "gen_ai.system": "openai",
                    "gen_ai.request.model": "gpt-4",
                    "gen_ai.usage.total_tokens": 150,
                },
            },
        }

        span = backend._parse_dd_span(span_obj)

        assert span is not None
        assert span.trace_id == "1234567890987654321"
        assert span.span_id == "1234567890987654321"
        assert span.parent_span_id is None  # "0" sentinel -> no parent
        assert span.service_name == "my-llm-service"
        assert span.operation_name == "chat_completion"
        assert span.duration_ms == pytest.approx(100.0)
        assert span.status == "UNSET"
        assert span.attributes.gen_ai_system == "openai"
        assert span.attributes.gen_ai_request_model == "gpt-4"

    def test_parse_child_span_with_real_parent(self) -> None:
        backend = _backend()
        span_obj = {
            "id": "child",
            "attributes": {
                "trace_id": "trace1",
                "span_id": "span2",
                "parent_id": "span1",
                "service": "my-service",
                "resource_name": "op",
                "start_timestamp": "2023-01-02T09:42:36.320Z",
                "end_timestamp": "2023-01-02T09:42:36.420Z",
            },
        }

        span = backend._parse_dd_span(span_obj)

        assert span is not None
        assert span.parent_span_id == "span1"

    def test_parse_span_missing_required_ids_returns_none(self) -> None:
        backend = _backend()
        assert backend._parse_dd_span({"attributes": {"service": "svc"}}) is None

    def test_error_inferred_from_custom_attribute(self) -> None:
        backend = _backend()
        span_obj = {
            "attributes": {
                "trace_id": "t1",
                "span_id": "s1",
                "service": "svc",
                "resource_name": "op",
                "start_timestamp": "2023-01-02T09:42:36.320Z",
                "end_timestamp": "2023-01-02T09:42:36.420Z",
                "custom": {"error": True, "error.message": "boom"},
            },
        }

        span = backend._parse_dd_span(span_obj)

        assert span is not None
        assert span.status == "ERROR"

    def test_error_inferred_from_tag(self) -> None:
        backend = _backend()
        span_obj = {
            "attributes": {
                "trace_id": "t1",
                "span_id": "s1",
                "service": "svc",
                "resource_name": "op",
                "start_timestamp": "2023-01-02T09:42:36.320Z",
                "end_timestamp": "2023-01-02T09:42:36.420Z",
                "tags": ["error:true"],
            },
        }

        span = backend._parse_dd_span(span_obj)

        assert span is not None
        assert span.status == "ERROR"

    def test_nested_gen_ai_attributes_are_flattened(self) -> None:
        """Handles the case where Datadog ingest nests OTel attrs as objects
        rather than flat dotted keys - see module docstring."""
        backend = _backend()
        span_obj = {
            "attributes": {
                "trace_id": "t1",
                "span_id": "s1",
                "service": "svc",
                "resource_name": "op",
                "start_timestamp": "2023-01-02T09:42:36.320Z",
                "end_timestamp": "2023-01-02T09:42:36.420Z",
                "custom": {"gen_ai": {"system": "anthropic", "request": {"model": "claude"}}},
            },
        }

        span = backend._parse_dd_span(span_obj)

        assert span is not None
        assert span.attributes.gen_ai_system == "anthropic"
        assert span.attributes.gen_ai_request_model == "claude"

    def test_top_level_status_field_takes_precedence_over_heuristics(self) -> None:
        """Regression test for a real bug found via a live Datadog account:
        an OTLP-ingested span carries a first-class top-level status field
        ("ok"/"error"), previously not checked at all - see module
        docstring. Confirmed authoritative, so it must win even when no
        custom-attribute/tag heuristic would otherwise fire."""
        backend = _backend()
        span_obj = {
            "attributes": {
                "trace_id": "t1",
                "span_id": "s1",
                "service": "svc",
                "resource_name": "op",
                "start_timestamp": "2023-01-02T09:42:36.320Z",
                "end_timestamp": "2023-01-02T09:42:36.420Z",
                "status": "error",
            },
        }

        span = backend._parse_dd_span(span_obj)

        assert span is not None
        assert span.status == "ERROR"

    def test_top_level_status_ok_is_respected(self) -> None:
        backend = _backend()
        span_obj = {
            "attributes": {
                "trace_id": "t1",
                "span_id": "s1",
                "service": "svc",
                "resource_name": "op",
                "start_timestamp": "2023-01-02T09:42:36.320Z",
                "end_timestamp": "2023-01-02T09:42:36.420Z",
                "status": "ok",
            },
        }

        span = backend._parse_dd_span(span_obj)

        assert span is not None
        assert span.status == "OK"

    def test_custom_attributes_are_nested_under_the_custom_key_not_attributes(self) -> None:
        """Regression test for a real bug found via a live Datadog account:
        the custom/OTel attributes object is keyed "custom" on the real API
        response, not "attributes" nested inside itself - the old code
        looked for the wrong key and silently got an empty dict every time,
        so no gen_ai.* field was ever populated against a real account."""
        backend = _backend()
        span_obj = {
            "attributes": {
                "trace_id": "t1",
                "span_id": "s1",
                "service": "svc",
                "resource_name": "op",
                "start_timestamp": "2023-01-02T09:42:36.320Z",
                "end_timestamp": "2023-01-02T09:42:36.420Z",
                "custom": {"gen_ai.system": "openai"},
            },
        }

        span = backend._parse_dd_span(span_obj)

        assert span is not None
        assert span.attributes.gen_ai_system == "openai"


class TestGroupIntoTrace:
    """Test grouping a flat span list into TraceData."""

    def test_group_selects_root_and_aggregates_status(self) -> None:
        backend = _backend()
        now = datetime(2023, 1, 2, 9, 42, 36, tzinfo=UTC)

        root = backend._parse_dd_span(
            {
                "attributes": {
                    "trace_id": "t1",
                    "span_id": "root",
                    "parent_id": "0",
                    "service": "svc",
                    "resource_name": "root-op",
                    "start_timestamp": now.isoformat().replace("+00:00", "Z"),
                    "end_timestamp": now.isoformat().replace("+00:00", "Z"),
                }
            }
        )
        child = backend._parse_dd_span(
            {
                "attributes": {
                    "trace_id": "t1",
                    "span_id": "child",
                    "parent_id": "root",
                    "service": "svc",
                    "resource_name": "child-op",
                    "start_timestamp": now.isoformat().replace("+00:00", "Z"),
                    "end_timestamp": now.isoformat().replace("+00:00", "Z"),
                    "custom": {"error": True},
                }
            }
        )
        assert root is not None and child is not None

        trace = backend._group_into_trace("t1", [root, child])

        assert trace.trace_id == "t1"
        assert trace.service_name == "svc"
        assert trace.root_operation == "root-op"
        assert trace.status == "ERROR"  # child span error propagates to trace status
        assert len(trace.spans) == 2

    def test_group_preserves_unset_when_no_span_confirms_ok_or_error(self) -> None:
        """No span has an explicit error, but none is explicitly OK either -
        the trace status must not silently claim OK."""
        backend = _backend()
        now = datetime(2023, 1, 2, 9, 42, 36, tzinfo=UTC)

        span = backend._parse_dd_span(
            {
                "attributes": {
                    "trace_id": "t2",
                    "span_id": "s1",
                    "service": "svc",
                    "resource_name": "op",
                    "start_timestamp": now.isoformat().replace("+00:00", "Z"),
                    "end_timestamp": now.isoformat().replace("+00:00", "Z"),
                }
            }
        )
        assert span is not None
        assert span.status == "UNSET"

        trace = backend._group_into_trace("t2", [span])

        assert trace.status == "UNSET"


class TestQueryEscaping:
    """Test that untrusted values can't inject additional query clauses."""

    def test_escape_quotes_and_wraps_value(self) -> None:
        backend = _backend()
        assert backend._escape_dd_query_value("abc123") == '"abc123"'

    def test_escape_handles_embedded_quotes(self) -> None:
        backend = _backend()
        assert backend._escape_dd_query_value('a" OR *:*') == '"a\\" OR *:*"'


class TestParseDatadogSpanRejectsBadTimestamps:
    """Test that spans with missing/invalid timing data are rejected rather
    than parsed with fabricated data."""

    def test_missing_end_timestamp_returns_none(self) -> None:
        backend = _backend()
        span_obj = {
            "attributes": {
                "trace_id": "t1",
                "span_id": "s1",
                "service": "svc",
                "resource_name": "op",
                "start_timestamp": "2023-01-02T09:42:36.320Z",
                # end_timestamp missing
            },
        }
        assert backend._parse_dd_span(span_obj) is None

    def test_missing_start_timestamp_returns_none(self) -> None:
        backend = _backend()
        span_obj = {
            "attributes": {
                "trace_id": "t1",
                "span_id": "s1",
                "service": "svc",
                "resource_name": "op",
                "end_timestamp": "2023-01-02T09:42:36.420Z",
                # start_timestamp missing
            },
        }
        assert backend._parse_dd_span(span_obj) is None

    def test_missing_service_returns_none(self) -> None:
        backend = _backend()
        span_obj = {
            "attributes": {
                "trace_id": "t1",
                "span_id": "s1",
                "resource_name": "op",
                # service missing
                "start_timestamp": "2023-01-02T09:42:36.320Z",
                "end_timestamp": "2023-01-02T09:42:36.420Z",
            },
        }
        assert backend._parse_dd_span(span_obj) is None

    def test_missing_resource_name_and_service_returns_none(self) -> None:
        backend = _backend()
        span_obj = {
            "attributes": {
                "trace_id": "t1",
                "span_id": "s1",
                # resource_name and service both missing
                "start_timestamp": "2023-01-02T09:42:36.320Z",
                "end_timestamp": "2023-01-02T09:42:36.420Z",
            },
        }
        assert backend._parse_dd_span(span_obj) is None


class TestGetTraceExactMatch:
    """Test get_trace only keeps spans exactly matching the requested trace_id."""

    async def test_filters_out_non_matching_spans(self) -> None:
        backend = _backend()
        now = "2023-01-02T09:42:36.320Z"
        later = "2023-01-02T09:42:36.420Z"

        backend._search_spans_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                {
                    "attributes": {
                        "trace_id": "requested",
                        "span_id": "s1",
                        "service": "svc",
                        "resource_name": "op",
                        "start_timestamp": now,
                        "end_timestamp": later,
                    }
                },
                {
                    # Wrong trace_id - should be filtered out even though it
                    # came back from the search.
                    "attributes": {
                        "trace_id": "other",
                        "span_id": "s2",
                        "service": "svc",
                        "resource_name": "op",
                        "start_timestamp": now,
                        "end_timestamp": later,
                    }
                },
            ]
        )

        trace = await backend.get_trace("requested")

        assert trace.trace_id == "requested"
        assert len(trace.spans) == 1
        assert trace.spans[0].span_id == "s1"

    async def test_escapes_trace_id_in_query(self) -> None:
        backend = _backend()
        backend._search_spans_raw = AsyncMock(return_value=[])  # type: ignore[method-assign]

        with pytest.raises(ValueError, match="No spans found"):
            await backend.get_trace('evil" OR *:*')

        call_args = backend._search_spans_raw.call_args
        dd_query = call_args.args[0]
        assert dd_query == 'trace_id:"evil\\" OR *:*"'


class TestSearchSpansRawPagination:
    """Test that _search_spans_raw follows Datadog's cursor pagination."""

    async def test_follows_cursor_across_pages(self, fake_json_client: Callable[..., Any]) -> None:
        backend = _backend()

        page_1 = {
            "data": [{"attributes": {"span_id": "s1"}}],
            "meta": {"page": {"after": "cursor-1"}},
        }
        page_2 = {
            "data": [{"attributes": {"span_id": "s2"}}],
            "meta": {},  # no cursor -> stop
        }
        backend._client = fake_json_client(page_1, page_2)

        now = datetime(2023, 1, 2, tzinfo=UTC)
        result = await backend._search_spans_raw("*", now, now, limit=2)

        assert [s["attributes"]["span_id"] for s in result] == ["s1", "s2"]

    async def test_stops_at_max_pages_without_infinite_loop(
        self, fake_json_client: Callable[..., Any]
    ) -> None:
        backend = _backend()

        # Always returns a cursor - would loop forever without a cap.
        page = {
            "data": [{"attributes": {"span_id": "s"}}],
            "meta": {"page": {"after": "always-more"}},
        }
        backend._client = fake_json_client(*([page] * _MAX_SEARCH_PAGES))

        now = datetime(2023, 1, 2, tzinfo=UTC)
        result = await backend._search_spans_raw("*", now, now, limit=100_000)

        assert len(result) == 10  # _MAX_SEARCH_PAGES pages x 1 span each

    async def test_malformed_data_field_does_not_crash(
        self, fake_json_client: Callable[..., Any]
    ) -> None:
        """A 200 response whose 'data' field isn't a list (or contains a
        non-dict entry) must not corrupt the collected results."""
        backend = _backend()
        backend._client = fake_json_client({"data": {"unexpected": {}}, "meta": {}})

        now = datetime(2023, 1, 2, tzinfo=UTC)
        result = await backend._search_spans_raw("*", now, now, limit=10)

        assert result == []

    async def test_non_dict_entries_in_data_are_skipped(
        self, fake_json_client: Callable[..., Any]
    ) -> None:
        backend = _backend()
        backend._client = fake_json_client(
            {"data": [{"attributes": {"span_id": "ok"}}, "not-a-span", 123], "meta": {}}
        )

        now = datetime(2023, 1, 2, tzinfo=UTC)
        result = await backend._search_spans_raw("*", now, now, limit=10)

        assert result == [{"attributes": {"span_id": "ok"}}]

    async def test_entry_with_non_dict_attributes_is_skipped(
        self, fake_json_client: Callable[..., Any]
    ) -> None:
        """An item that is itself a dict, but whose 'attributes' value isn't
        one, must also be excluded - every consumer does
        `item.get("attributes", {}).get(...)` directly."""
        backend = _backend()
        backend._client = fake_json_client(
            {
                "data": [{"attributes": {"span_id": "ok"}}, {"attributes": "bad"}],
                "meta": {},
            }
        )

        now = datetime(2023, 1, 2, tzinfo=UTC)
        result = await backend._search_spans_raw("*", now, now, limit=10)

        assert result == [{"attributes": {"span_id": "ok"}}]


class TestGetServiceOperationsEscaping:
    """Test that get_service_operations escapes the service name in its query."""

    async def test_escapes_service_name(self) -> None:
        backend = _backend()
        backend._search_spans_raw = AsyncMock(return_value=[])  # type: ignore[method-assign]

        await backend.get_service_operations('svc" OR *:*')

        call_args = backend._search_spans_raw.call_args
        dd_query = call_args.args[0]
        assert dd_query == 'service:"svc\\" OR *:*"'


class TestGetTraceUsesFullPaginationCapacity:
    """get_trace's contract is "the complete trace" - it should target the
    full pagination capacity, not a single page's worth."""

    async def test_requests_full_pagination_capacity(self) -> None:
        backend = _backend()
        backend._search_spans_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                {
                    "attributes": {
                        "trace_id": "t1",
                        "span_id": "s1",
                        "service": "svc",
                        "resource_name": "op",
                        "start_timestamp": "2023-01-02T09:42:36.320Z",
                        "end_timestamp": "2023-01-02T09:42:36.420Z",
                    }
                }
            ]
        )

        await backend.get_trace("t1")

        call_args = backend._search_spans_raw.call_args
        assert call_args.kwargs.get("limit") == _MAX_SEARCH_PAGES * 1000


class TestSearchSpansRawMalformedEnvelope:
    """Test that a malformed-but-200 response body doesn't crash
    _search_spans_raw at any navigation step (top-level, meta, meta.page)."""

    async def test_non_dict_top_level_body_does_not_crash(
        self, fake_json_client: Callable[..., Any]
    ) -> None:
        backend = _backend()
        backend._client = fake_json_client(["not", "an", "object"])

        now = datetime(2023, 1, 2, tzinfo=UTC)
        result = await backend._search_spans_raw("*", now, now, limit=10)

        assert result == []

    async def test_non_dict_meta_does_not_crash(self, fake_json_client: Callable[..., Any]) -> None:
        backend = _backend()
        backend._client = fake_json_client(
            {"data": [{"attributes": {"span_id": "s1"}}], "meta": "not-an-object"}
        )

        now = datetime(2023, 1, 2, tzinfo=UTC)
        result = await backend._search_spans_raw("*", now, now, limit=10)

        # The one valid span is still collected; malformed meta just means
        # "no cursor" rather than a crash.
        assert result == [{"attributes": {"span_id": "s1"}}]

    async def test_non_dict_meta_page_does_not_crash(
        self, fake_json_client: Callable[..., Any]
    ) -> None:
        backend = _backend()
        backend._client = fake_json_client(
            {"data": [{"attributes": {"span_id": "s1"}}], "meta": {"page": "nope"}}
        )

        now = datetime(2023, 1, 2, tzinfo=UTC)
        result = await backend._search_spans_raw("*", now, now, limit=10)

        assert result == [{"attributes": {"span_id": "s1"}}]


def _fake_trace(trace_id: str, status: str = "OK") -> TraceData:
    now = datetime(2023, 1, 2, tzinfo=UTC)
    return TraceData(
        trace_id=trace_id,
        spans=[],
        start_time=now,
        duration_ms=10.0,
        service_name="svc",
        root_operation="op",
        status=status,  # type: ignore[arg-type]
    )


class TestSearchTraces:
    """Test the search_traces() orchestration: discover trace_ids from a
    span search, hydrate each via get_trace, and re-verify filters against
    the fully-hydrated trace - the top-level method itself was previously
    untested even though its two building blocks (_search_spans_raw,
    get_trace) each have their own dedicated coverage above."""

    async def test_discovers_and_hydrates_each_distinct_trace_id(self) -> None:
        backend = _backend()
        backend._search_spans_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                {"attributes": {"trace_id": "t1"}},
                {"attributes": {"trace_id": "t2"}},
            ]
        )
        backend.get_trace = AsyncMock(side_effect=lambda tid: _fake_trace(tid))  # type: ignore[method-assign]

        result = await backend.search_traces(TraceQuery(limit=100))

        assert {t.trace_id for t in result} == {"t1", "t2"}
        assert backend.get_trace.await_count == 2

    async def test_dedupes_trace_ids_preserving_discovery_order(self) -> None:
        backend = _backend()
        backend._search_spans_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                {"attributes": {"trace_id": "t1"}},
                {"attributes": {"trace_id": "t2"}},
                {"attributes": {"trace_id": "t1"}},  # duplicate, same trace
            ]
        )
        backend.get_trace = AsyncMock(side_effect=lambda tid: _fake_trace(tid))  # type: ignore[method-assign]

        await backend.search_traces(TraceQuery(limit=100))

        hydrated_ids = [call.args[0] for call in backend.get_trace.await_args_list]
        assert hydrated_ids == ["t1", "t2"]

    async def test_caps_hydration_at_max_traces_to_hydrate_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = _backend()
        discovered = _MAX_TRACES_TO_HYDRATE + 5
        backend._search_spans_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[{"attributes": {"trace_id": f"t{i}"}} for i in range(discovered)]
        )
        backend.get_trace = AsyncMock(side_effect=lambda tid: _fake_trace(tid))  # type: ignore[method-assign]

        with caplog.at_level("WARNING"):
            result = await backend.search_traces(TraceQuery(limit=1000))

        assert backend.get_trace.await_count == _MAX_TRACES_TO_HYDRATE
        assert len(result) == _MAX_TRACES_TO_HYDRATE
        assert any("Limiting trace fetch" in r.message for r in caplog.records)

    async def test_a_failed_hydration_is_skipped_not_propagated(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = _backend()
        backend._search_spans_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                {"attributes": {"trace_id": "good"}},
                {"attributes": {"trace_id": "bad"}},
            ]
        )

        async def _get_trace(trace_id: str) -> TraceData:
            if trace_id == "bad":
                raise ValueError("no spans found for trace bad")
            return _fake_trace(trace_id)

        backend.get_trace = AsyncMock(side_effect=_get_trace)  # type: ignore[method-assign]

        with caplog.at_level("WARNING"):
            result = await backend.search_traces(TraceQuery(limit=100))

        assert [t.trace_id for t in result] == ["good"]
        assert any("Failed to fetch trace bad" in r.message for r in caplog.records)

    async def test_client_side_filter_excludes_a_hydrated_trace_that_does_not_match(
        self,
    ) -> None:
        """Datadog's native operators don't include CONTAINS - a filter
        using it must still be re-verified against the fully-hydrated
        trace (FilterEngine.apply_filters), not just passed through
        because the initial span search couldn't apply it either."""
        backend = _backend()
        backend._search_spans_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                {"attributes": {"trace_id": "matches"}},
                {"attributes": {"trace_id": "does-not-match"}},
            ]
        )

        async def _get_trace(trace_id: str) -> TraceData:
            return _fake_trace(trace_id, status="ERROR" if trace_id == "matches" else "OK")

        backend.get_trace = AsyncMock(side_effect=_get_trace)  # type: ignore[method-assign]

        result = await backend.search_traces(
            TraceQuery(
                limit=100,
                filters=[
                    Filter(
                        field="status",
                        operator=FilterOperator.CONTAINS,
                        value="ERR",
                        value_type=FilterType.STRING,
                    )
                ],
            )
        )

        assert [t.trace_id for t in result] == ["matches"]

    async def test_truncates_to_query_limit_after_hydration(self) -> None:
        backend = _backend()
        backend._search_spans_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[{"attributes": {"trace_id": f"t{i}"}} for i in range(5)]
        )
        backend.get_trace = AsyncMock(side_effect=lambda tid: _fake_trace(tid))  # type: ignore[method-assign]

        result = await backend.search_traces(TraceQuery(limit=2))

        assert len(result) == 2


class TestSearchSpans:
    """Test the search_spans() orchestration: parse raw spans returned by
    _search_spans_raw, re-apply client-side filters, and truncate to the
    query limit - previously untested at the top-level-method scope even
    though _parse_dd_span and _search_spans_raw each have their own
    dedicated coverage above."""

    def _raw_span(self, span_id: str, service: str = "svc") -> dict[str, Any]:
        now = "2023-01-02T09:42:36.320Z"
        later = "2023-01-02T09:42:36.420Z"
        return {
            "attributes": {
                "trace_id": "t1",
                "span_id": span_id,
                "service": service,
                "resource_name": "op",
                "start_timestamp": now,
                "end_timestamp": later,
            }
        }

    async def test_parses_every_valid_span_returned_by_search(self) -> None:
        backend = _backend()
        backend._search_spans_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[self._raw_span("s1"), self._raw_span("s2")]
        )

        result = await backend.search_spans(SpanQuery(limit=100))

        assert {s.span_id for s in result} == {"s1", "s2"}

    async def test_skips_spans_that_fail_to_parse(self) -> None:
        backend = _backend()
        backend._search_spans_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                self._raw_span("s1"),
                {"attributes": {"span_id": "s2"}},  # missing trace_id -> unparseable
            ]
        )

        result = await backend.search_spans(SpanQuery(limit=100))

        assert [s.span_id for s in result] == ["s1"]

    async def test_client_side_filter_excludes_non_matching_spans(self) -> None:
        """CONTAINS is not one of Datadog's natively supported operators
        (get_supported_operators) - a filter using it must be re-applied
        client-side against the parsed spans."""
        backend = _backend()
        backend._search_spans_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[self._raw_span("s1", service="foo"), self._raw_span("s2", service="bar")]
        )

        result = await backend.search_spans(
            SpanQuery(
                limit=100,
                filters=[
                    Filter(
                        field="service.name",
                        operator=FilterOperator.CONTAINS,
                        value="fo",
                        value_type=FilterType.STRING,
                    )
                ],
            )
        )

        assert [s.span_id for s in result] == ["s1"]

    async def test_truncates_to_query_limit(self) -> None:
        backend = _backend()
        backend._search_spans_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[self._raw_span(f"s{i}") for i in range(5)]
        )

        result = await backend.search_spans(SpanQuery(limit=2))

        assert len(result) == 2


class TestListServices:
    """Test list_services()'s sampling-based service extraction - the
    existing escaping test for get_service_operations always returned an
    empty span list, so this backend's actual dedup/sort logic over real
    span data was never exercised."""

    async def test_extracts_unique_sorted_services_from_sampled_spans(self) -> None:
        backend = _backend()
        backend._search_spans_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                {"attributes": {"service": "b-service"}},
                {"attributes": {"service": "a-service"}},
                {"attributes": {"service": "b-service"}},  # duplicate
            ]
        )

        result = await backend.list_services()

        assert result == ["a-service", "b-service"]

    async def test_spans_missing_service_attribute_are_ignored(self) -> None:
        backend = _backend()
        backend._search_spans_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[{"attributes": {}}, {"attributes": {"service": "real"}}]
        )

        result = await backend.list_services()

        assert result == ["real"]


class TestGetServiceOperationsExtraction:
    """Complements TestGetServiceOperationsEscaping (which only verifies
    the query string) by exercising the actual operation-extraction logic
    over real span data."""

    async def test_extracts_unique_sorted_operations(self) -> None:
        backend = _backend()
        backend._search_spans_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                {"attributes": {"resource_name": "GET /b"}},
                {"attributes": {"resource_name": "GET /a"}},
                {"attributes": {"resource_name": "GET /b"}},
            ]
        )

        result = await backend.get_service_operations("svc")

        assert result == ["GET /a", "GET /b"]


class TestHealthCheck:
    async def test_healthy_when_search_succeeds(self) -> None:
        backend = _backend()
        backend._search_spans_raw = AsyncMock(return_value=[])  # type: ignore[method-assign]

        result = await backend.health_check()

        assert result.status == "healthy"
        assert result.backend == "datadog"
        assert result.error is None

    async def test_unhealthy_wraps_the_exception(self) -> None:
        backend = _backend()
        backend._search_spans_raw = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]

        result = await backend.health_check()

        assert result.status == "unhealthy"
        assert result.backend == "datadog"
        assert result.error == "boom"

"""Tests for Sentry backend.

Fixtures below are hand-built from docs.sentry.io (Auth, Pagination, the
Explore/table-format Events API, and Span Properties) plus the
``getsentry/sentry`` endpoint source read at research time - not against a
live account (none was available). See the module docstring in
``opentelemetry_mcp/backends/sentry.py`` for the specific things the public
docs don't pin down (raw span column names, the exact SerializedTraceItem
shape, and the query-value escape sequence) and how this implementation
handles each.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from opentelemetry_mcp.backends.base import MetadataEndpointBlockedError, _RetryingTransport
from opentelemetry_mcp.backends.sentry import (
    _MAX_SEARCH_PAGES,
    _MAX_TRACES_TO_HYDRATE,
    _SPAN_SEARCH_FIELDS,
    SentryBackend,
)
from opentelemetry_mcp.models import (
    Filter,
    FilterOperator,
    FilterType,
    SpanQuery,
    TraceData,
    TraceQuery,
)

FAKE_AUTH = "sentry-key1"
FAKE_ORG = "acme"


def test_sentry_backend_requires_api_key() -> None:
    """Test that Sentry backend requires an auth token."""
    with pytest.raises(ValueError, match="requires an auth token"):
        SentryBackend(url="https://sentry.io", api_key=None, org_slug=FAKE_ORG)


def test_sentry_backend_requires_org_slug() -> None:
    """Test that Sentry backend requires an organization slug even with a token."""
    with pytest.raises(ValueError, match="organization slug"):
        SentryBackend(url="https://sentry.io", api_key=FAKE_AUTH, org_slug=None)


def test_sentry_backend_rejects_non_https_url() -> None:
    """Test that Sentry backend refuses to send the auth token over plain http."""
    with pytest.raises(ValueError, match="https://"):
        SentryBackend(url="http://sentry.io", api_key=FAKE_AUTH, org_slug=FAKE_ORG)


def test_sentry_client_does_not_disable_redirects() -> None:
    """Standard Authorization headers are stripped by httpx on cross-origin
    redirects, so - unlike Datadog's non-standard headers - this backend
    must not override the base class's follow_redirects=True."""
    backend = SentryBackend(url="https://sentry.io", api_key=FAKE_AUTH, org_slug=FAKE_ORG)
    assert backend.client.follow_redirects is True


async def test_sentry_redirect_to_metadata_ip_is_blocked() -> None:
    """Unlike Datadog (which disables follow_redirects outright - see
    test_datadog_client_disables_redirects in test_datadog.py), Sentry keeps
    follow_redirects=True by design (see
    test_sentry_client_does_not_disable_redirects above), so a backend
    redirecting toward a metadata address here would actually be followed by
    httpx if not for _RetryingTransport's own mid-redirect metadata-host
    check (see TestMetadataEndpointBlocking in test_base_backend.py)."""
    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if str(request.url) == "https://sentry.io/search":
            return httpx.Response(
                302, headers={"Location": "http://169.254.169.254/latest/meta-data/"}
            )
        return httpx.Response(200)

    backend = SentryBackend(url="https://sentry.io", api_key=FAKE_AUTH, org_slug=FAKE_ORG)
    backend._client = httpx.AsyncClient(
        base_url="https://sentry.io",
        follow_redirects=True,
        transport=_RetryingTransport(wrapped=httpx.MockTransport(handler)),
    )

    with pytest.raises(MetadataEndpointBlockedError):
        await backend.client.get("/search")

    assert call_count == 1


def test_sentry_backend_initialization() -> None:
    """Test Sentry backend initializes correctly with token, org, and project."""
    backend = SentryBackend(
        url="https://sentry.io",
        api_key=FAKE_AUTH,
        org_slug=FAKE_ORG,
        project_slug="my-llm-app",
        timeout=15.0,
    )

    assert backend.url == "https://sentry.io"
    assert backend.api_key == FAKE_AUTH
    assert backend.org_slug == FAKE_ORG
    assert backend.project_slug == "my-llm-app"
    assert backend.timeout == 15.0


def test_sentry_backend_project_slug_is_optional() -> None:
    """Test that project_slug can be omitted."""
    backend = SentryBackend(url="https://sentry.io", api_key=FAKE_AUTH, org_slug=FAKE_ORG)
    assert backend.project_slug is None


def test_sentry_client_headers() -> None:
    """Test that Sentry client sends a standard Bearer Authorization header."""
    backend = SentryBackend(url="https://sentry.io", api_key=FAKE_AUTH, org_slug=FAKE_ORG)

    client = backend.client
    assert client.headers["Authorization"] == f"Bearer {FAKE_AUTH}"


def _backend() -> SentryBackend:
    return SentryBackend(url="https://sentry.io", api_key=FAKE_AUTH, org_slug=FAKE_ORG)


class TestBuildSentryQuery:
    """Test Filter -> Sentry search syntax query string conversion."""

    def test_equals_facet_field(self) -> None:
        backend = _backend()
        f = Filter(
            field="service.name",
            operator=FilterOperator.EQUALS,
            value="my-service",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_sentry_query(f) == 'project:"my-service"'

    def test_equals_custom_attribute_is_bare(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system",
            operator=FilterOperator.EQUALS,
            value="openai",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_sentry_query(f) == 'gen_ai.system:"openai"'

    def test_not_equals(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system",
            operator=FilterOperator.NOT_EQUALS,
            value="openai",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_sentry_query(f) == '!gen_ai.system:"openai"'

    def test_status_error_equals(self) -> None:
        backend = _backend()
        f = Filter(
            field="status",
            operator=FilterOperator.EQUALS,
            value="ERROR",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_sentry_query(f) == "!span.status:ok"

    def test_status_ok_equals(self) -> None:
        backend = _backend()
        f = Filter(
            field="status",
            operator=FilterOperator.EQUALS,
            value="OK",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_sentry_query(f) == "span.status:ok"

    def test_status_error_not_equals(self) -> None:
        backend = _backend()
        f = Filter(
            field="status",
            operator=FilterOperator.NOT_EQUALS,
            value="ERROR",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_sentry_query(f) == "span.status:ok"

    def test_duration_gte_uses_ms_suffix_unquoted(self) -> None:
        backend = _backend()
        f = Filter(
            field="duration", operator=FilterOperator.GTE, value=1000, value_type=FilterType.NUMBER
        )
        assert backend._filter_to_sentry_query(f) == "span.duration:>=1000ms"

    def test_duration_lt_uses_ms_suffix_unquoted(self) -> None:
        backend = _backend()
        f = Filter(
            field="duration", operator=FilterOperator.LT, value=5000, value_type=FilterType.NUMBER
        )
        assert backend._filter_to_sentry_query(f) == "span.duration:<5000ms"

    def test_equals_numeric_value_is_unquoted(self) -> None:
        """Numeric literals must not go through the string-quoting escaper -
        `field:">150"` would be a broken query."""
        backend = _backend()
        f = Filter(
            field="gen_ai.usage.total_tokens",
            operator=FilterOperator.EQUALS,
            value=150,
            value_type=FilterType.NUMBER,
        )
        assert backend._filter_to_sentry_query(f) == "gen_ai.usage.total_tokens:150"

    def test_equals_escapes_embedded_quote(self) -> None:
        """A crafted filter value can't inject additional query clauses."""
        backend = _backend()
        f = Filter(
            field="gen_ai.system",
            operator=FilterOperator.EQUALS,
            value='a" OR system:*',
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_sentry_query(f) == 'gen_ai.system:"a\\" OR system:*"'

    def test_range_operator_rejects_non_numeric_value(self) -> None:
        """Filter.value_type isn't enforced against the actual Python type,
        so a range operator with a string value must be rejected rather than
        interpolated unchecked into a numeric range expression."""
        backend = _backend()
        f = Filter(
            field="duration",
            operator=FilterOperator.GT,
            value="1000ms OR span.duration:>0",
            value_type=FilterType.NUMBER,
        )
        assert backend._filter_to_sentry_query(f) is None

    def test_range_operator_rejects_bool_value(self) -> None:
        """bool is an int subclass in Python but not a sensible range operand."""
        backend = _backend()
        f = Filter(
            field="duration", operator=FilterOperator.GTE, value=True, value_type=FilterType.NUMBER
        )
        assert backend._filter_to_sentry_query(f) is None

    def test_exists(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system", operator=FilterOperator.EXISTS, value_type=FilterType.STRING
        )
        assert backend._filter_to_sentry_query(f) == "has:gen_ai.system"

    def test_not_exists(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system", operator=FilterOperator.NOT_EXISTS, value_type=FilterType.STRING
        )
        assert backend._filter_to_sentry_query(f) == "!has:gen_ai.system"

    def test_in_builds_or(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system",
            operator=FilterOperator.IN,
            values=["openai", "anthropic"],
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_sentry_query(f) == (
            '(gen_ai.system:"openai" OR gen_ai.system:"anthropic")'
        )

    def test_build_sentry_query_empty_defaults_to_match_all(self) -> None:
        backend = _backend()
        assert backend._build_sentry_query([]) == ""

    def test_build_sentry_query_joins_with_and(self) -> None:
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
        assert backend._build_sentry_query(filters) == ('project:"svc" AND gen_ai.system:"openai"')


class TestFieldNameInjection:
    """Filter.field is an unvalidated str reachable from any MCP tool call.

    Unlike the filter *value*, which is always escaped/quoted via
    _escape_sentry_query_value, the field name used to be spliced directly
    into the query string - so a malicious field name could inject
    arbitrary structure (e.g. breaking out of an AND with an OR/paren
    group) into the Sentry search DSL. Every operator branch must reject an
    unsafe field name instead.
    """

    def test_equals_rejects_injection_field(self) -> None:
        backend = _backend()
        f = Filter(
            field="x) OR (a:b",
            operator=FilterOperator.EQUALS,
            value="v",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_sentry_query(f) is None

    def test_exists_rejects_injection_field(self) -> None:
        backend = _backend()
        f = Filter(
            field="x) OR (span.op:*",
            operator=FilterOperator.EXISTS,
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_sentry_query(f) is None

    def test_not_exists_rejects_injection_field(self) -> None:
        backend = _backend()
        f = Filter(
            field="x) OR (span.op:*",
            operator=FilterOperator.NOT_EXISTS,
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_sentry_query(f) is None

    def test_range_operator_rejects_injection_field(self) -> None:
        backend = _backend()
        f = Filter(
            field="x) OR (a:b",
            operator=FilterOperator.GT,
            value=5,
            value_type=FilterType.NUMBER,
        )
        assert backend._filter_to_sentry_query(f) is None

    def test_in_rejects_injection_field(self) -> None:
        backend = _backend()
        f = Filter(
            field="x) OR (a:b",
            operator=FilterOperator.IN,
            values=["v1", "v2"],
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_sentry_query(f) is None

    def test_build_sentry_query_drops_injection_filter_from_and_join(self) -> None:
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
        assert backend._build_sentry_query(filters) == 'gen_ai.system:"openai"'

    def test_valid_dotted_field_is_still_accepted(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.usage.total_tokens",
            operator=FilterOperator.EQUALS,
            value=100,
            value_type=FilterType.NUMBER,
        )
        assert backend._filter_to_sentry_query(f) == "gen_ai.usage.total_tokens:100"


def test_span_search_fields_requests_conversation_and_prompt_version_columns() -> None:
    """The other half of the list_sessions/get_prompt_version_stats bug: a
    span row can only carry gen_ai.conversation.id/prompt.name/prompt.version
    if the search request actually asked the Events API for those columns."""
    assert "gen_ai.conversation.id" in _SPAN_SEARCH_FIELDS
    assert "gen_ai.prompt.name" in _SPAN_SEARCH_FIELDS
    assert "gen_ai.prompt.version" in _SPAN_SEARCH_FIELDS


def test_span_search_fields_requests_the_gen_ai_provider_name_rename_too() -> None:
    """Same class of bug, different attribute: an account whose spans only
    carry gen_ai.provider.name (the OTel semconv v1.37.0 rename of
    gen_ai.system) needs that column actually requested from the Events API
    before attributes.py's model-level rename-fallback ever sees it."""
    assert "gen_ai.provider.name" in _SPAN_SEARCH_FIELDS


class TestQueryEscaping:
    """Test that untrusted values can't inject additional query clauses."""

    def test_escape_quotes_and_wraps_value(self) -> None:
        backend = _backend()
        assert backend._escape_sentry_query_value("abc123") == '"abc123"'

    def test_escape_handles_embedded_quotes(self) -> None:
        backend = _backend()
        assert backend._escape_sentry_query_value('a" OR *:*') == '"a\\" OR *:*"'

    def test_escape_handles_embedded_backslash(self) -> None:
        backend = _backend()
        assert backend._escape_sentry_query_value("a\\b") == '"a\\\\b"'


class TestParseSentryRow:
    """Test parsing raw table-format Events API rows into SpanData."""

    def test_parse_root_span(self) -> None:
        backend = _backend()
        row = {
            "id": "span1",
            "trace": "trace1",
            "parent_span": None,
            "span.op": "chat_completion",
            "project": "my-llm-service",
            "timestamp": "2023-01-02T09:42:36.320Z",
            "span.duration": 100.0,
            "span.status": "ok",
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4",
            "gen_ai.usage.total_tokens": 150,
        }

        span = backend._parse_sentry_row(row)

        assert span is not None
        assert span.trace_id == "trace1"
        assert span.span_id == "span1"
        assert span.parent_span_id is None
        assert span.service_name == "my-llm-service"
        assert span.operation_name == "chat_completion"
        assert span.duration_ms == pytest.approx(100.0)
        assert span.status == "OK"
        assert span.attributes.gen_ai_system == "openai"
        assert span.attributes.gen_ai_request_model == "gpt-4"

    def test_parse_root_span_carries_conversation_and_prompt_version_fields(self) -> None:
        """Regression test: _GEN_AI_FIELDS previously omitted
        gen_ai.conversation.id/prompt.name/prompt.version, so even a span
        row that actually contained them would have them silently dropped
        before this method ever saw them (never requested from the search
        API in the first place) - list_sessions/get_session_stats/
        get_prompt_version_stats always returned empty against Sentry
        regardless of what data existed. This test proves the parse side:
        once present in the row, they must reach SpanAttributes."""
        backend = _backend()
        row = {
            "id": "span1",
            "trace": "trace1",
            "parent_span": None,
            "span.op": "chat_completion",
            "project": "my-llm-service",
            "timestamp": "2023-01-02T09:42:36.320Z",
            "span.duration": 100.0,
            "span.status": "ok",
            "gen_ai.conversation.id": "conv-123",
            "gen_ai.prompt.name": "summarize",
            "gen_ai.prompt.version": "2",
        }

        span = backend._parse_sentry_row(row)

        assert span is not None
        assert span.attributes.gen_ai_conversation_id == "conv-123"
        assert span.attributes.gen_ai_prompt_name == "summarize"
        assert span.attributes.gen_ai_prompt_version == "2"

    def test_parse_child_span_with_real_parent(self) -> None:
        backend = _backend()
        row = {
            "id": "span2",
            "trace": "trace1",
            "parent_span": "span1",
            "span.op": "op",
            "project": "svc",
            "timestamp": "2023-01-02T09:42:36.320Z",
            "span.duration": 50.0,
        }

        span = backend._parse_sentry_row(row)

        assert span is not None
        assert span.parent_span_id == "span1"

    def test_parse_row_missing_id_returns_none(self) -> None:
        backend = _backend()
        assert backend._parse_sentry_row({"trace": "t1", "project": "svc"}) is None

    def test_parse_row_missing_trace_returns_none(self) -> None:
        backend = _backend()
        assert backend._parse_sentry_row({"id": "s1", "project": "svc"}) is None

    def test_parse_row_missing_timestamp_returns_none(self) -> None:
        backend = _backend()
        row = {"id": "s1", "trace": "t1", "project": "svc", "span.duration": 10.0}
        assert backend._parse_sentry_row(row) is None

    def test_parse_row_missing_duration_returns_none(self) -> None:
        backend = _backend()
        row = {
            "id": "s1",
            "trace": "t1",
            "project": "svc",
            "timestamp": "2023-01-02T09:42:36.320Z",
        }
        assert backend._parse_sentry_row(row) is None

    def test_status_error_from_span_status(self) -> None:
        backend = _backend()
        row = {
            "id": "s1",
            "trace": "t1",
            "project": "svc",
            "span.op": "op",
            "timestamp": "2023-01-02T09:42:36.320Z",
            "span.duration": 10.0,
            "span.status": "internal_error",
        }
        span = backend._parse_sentry_row(row)
        assert span is not None
        assert span.status == "ERROR"

    def test_status_unset_when_missing(self) -> None:
        backend = _backend()
        row = {
            "id": "s1",
            "trace": "t1",
            "project": "svc",
            "span.op": "op",
            "timestamp": "2023-01-02T09:42:36.320Z",
            "span.duration": 10.0,
        }
        span = backend._parse_sentry_row(row)
        assert span is not None
        assert span.status == "UNSET"

    def test_parse_row_missing_operation_returns_none(self) -> None:
        """Neither span.op nor transaction is present - reject rather than
        fabricating operation_name="unknown", which would silently merge
        structurally-unrelated spans into one fake bucket downstream."""
        backend = _backend()
        row = {
            "id": "s1",
            "trace": "t1",
            "project": "svc",
            "timestamp": "2023-01-02T09:42:36.320Z",
            "span.duration": 10.0,
        }
        assert backend._parse_sentry_row(row) is None

    def test_parse_row_missing_project_returns_none(self) -> None:
        """No project present - reject rather than fabricating
        service_name="unknown"."""
        backend = _backend()
        row = {
            "id": "s1",
            "trace": "t1",
            "span.op": "op",
            "timestamp": "2023-01-02T09:42:36.320Z",
            "span.duration": 10.0,
        }
        assert backend._parse_sentry_row(row) is None

    def test_parse_row_negative_duration_returns_none(self) -> None:
        backend = _backend()
        row = {
            "id": "s1",
            "trace": "t1",
            "project": "svc",
            "span.op": "op",
            "timestamp": "2023-01-02T09:42:36.320Z",
            "span.duration": -50.0,
        }
        assert backend._parse_sentry_row(row) is None

    def test_parse_row_absurdly_large_duration_returns_none(self) -> None:
        backend = _backend()
        row = {
            "id": "s1",
            "trace": "t1",
            "project": "svc",
            "span.op": "op",
            "timestamp": "2023-01-02T09:42:36.320Z",
            "span.duration": 1e19,
        }
        assert backend._parse_sentry_row(row) is None


class TestParseSentryTraceItem:
    """Test parsing SerializedTraceItem entries from /trace/{id}/."""

    def test_parse_item_with_start_and_end_timestamp(self) -> None:
        backend = _backend()
        item = {
            "span_id": "s1",
            "trace_id": "t1",
            "op": "chat_completion",
            "project_slug": "svc",
            "start_timestamp": "2023-01-02T09:42:36.320Z",
            "end_timestamp": "2023-01-02T09:42:36.420Z",
        }
        span = backend._parse_sentry_trace_item(item, "t1")
        assert span is not None
        assert span.trace_id == "t1"
        assert span.span_id == "s1"
        assert span.duration_ms == pytest.approx(100.0)

    def test_parse_item_missing_trace_id_falls_back_to_requested_trace_id(self) -> None:
        """Regression test for a real bug found via a live Sentry account:
        GET /trace/{id}/ response items never carry their own trace_id/trace
        field - the endpoint is already scoped to one trace by the URL path,
        so Sentry doesn't repeat it per item. The old code treated the
        absence of that field as a rejection signal and discarded every
        single item unconditionally, so get_trace() (and everything built
        on it - search_traces, list_models, find_errors, ...) always failed
        with "no spans found" against any real account."""
        backend = _backend()
        item = {
            "span_id": "s1",
            "op": "op",
            "project": "svc",
            "start_timestamp": "2023-01-02T09:42:36.320Z",
            "duration": 25.0,
        }
        span = backend._parse_sentry_trace_item(item, "requested-trace")
        assert span is not None
        assert span.trace_id == "requested-trace"

    def test_parse_item_missing_span_id_returns_none(self) -> None:
        backend = _backend()
        item = {"trace_id": "t1", "start_timestamp": "2023-01-02T09:42:36.320Z"}
        assert backend._parse_sentry_trace_item(item, "t1") is None

    def test_parse_item_missing_start_timestamp_returns_none(self) -> None:
        backend = _backend()
        item = {"span_id": "s1", "trace_id": "t1", "duration": 10.0}
        assert backend._parse_sentry_trace_item(item, "t1") is None

    def test_parse_item_missing_duration_and_end_returns_none(self) -> None:
        backend = _backend()
        item = {
            "span_id": "s1",
            "trace_id": "t1",
            "start_timestamp": "2023-01-02T09:42:36.320Z",
        }
        assert backend._parse_sentry_trace_item(item, "t1") is None

    def test_error_inferred_from_nonempty_errors_list(self) -> None:
        backend = _backend()
        item = {
            "span_id": "s1",
            "trace_id": "t1",
            "op": "op",
            "project": "svc",
            "start_timestamp": "2023-01-02T09:42:36.320Z",
            "duration": 10.0,
            "errors": [{"event_id": "e1"}],
        }
        span = backend._parse_sentry_trace_item(item, "t1")
        assert span is not None
        assert span.status == "ERROR"

    def test_error_not_inferred_from_empty_errors_list(self) -> None:
        backend = _backend()
        item = {
            "span_id": "s1",
            "trace_id": "t1",
            "op": "op",
            "project": "svc",
            "start_timestamp": "2023-01-02T09:42:36.320Z",
            "duration": 10.0,
            "errors": [],
            "status": "ok",
        }
        span = backend._parse_sentry_trace_item(item, "t1")
        assert span is not None
        assert span.status == "OK"

    def test_error_inferred_from_nonzero_error_count(self) -> None:
        """`errors` may come back as an integer count rather than a list -
        an untrusted/non-list truthy value must still be read as a real
        error signal, not silently treated as "no errors"."""
        backend = _backend()
        item = {
            "span_id": "s1",
            "trace_id": "t1",
            "op": "op",
            "project": "svc",
            "start_timestamp": "2023-01-02T09:42:36.320Z",
            "duration": 10.0,
            "errors": 2,
            "status": "ok",
        }
        span = backend._parse_sentry_trace_item(item, "t1")
        assert span is not None
        assert span.status == "ERROR"

    def test_error_not_inferred_from_zero_error_count(self) -> None:
        backend = _backend()
        item = {
            "span_id": "s1",
            "trace_id": "t1",
            "op": "op",
            "project": "svc",
            "start_timestamp": "2023-01-02T09:42:36.320Z",
            "duration": 10.0,
            "errors": 0,
            "status": "ok",
        }
        span = backend._parse_sentry_trace_item(item, "t1")
        assert span is not None
        assert span.status == "OK"

    def test_parse_item_missing_operation_and_project_returns_none(self) -> None:
        backend = _backend()
        item = {
            "span_id": "s1",
            "trace_id": "t1",
            "start_timestamp": "2023-01-02T09:42:36.320Z",
            "duration": 10.0,
        }
        assert backend._parse_sentry_trace_item(item, "t1") is None

    def test_parse_item_negative_duration_returns_none(self) -> None:
        backend = _backend()
        item = {
            "span_id": "s1",
            "trace_id": "t1",
            "op": "op",
            "project": "svc",
            "start_timestamp": "2023-01-02T09:42:36.320Z",
            "duration": -10.0,
        }
        assert backend._parse_sentry_trace_item(item, "t1") is None

    def test_parse_item_end_before_start_returns_none(self) -> None:
        """A clock-skew or malformed end_timestamp that precedes
        start_timestamp must not silently produce a negative duration."""
        backend = _backend()
        item = {
            "span_id": "s1",
            "trace_id": "t1",
            "op": "op",
            "project": "svc",
            "start_timestamp": "2023-01-02T09:42:36.420Z",
            "end_timestamp": "2023-01-02T09:42:36.320Z",
        }
        assert backend._parse_sentry_trace_item(item, "t1") is None

    def test_parse_item_absurdly_large_duration_returns_none(self) -> None:
        """A corrupted/garbage duration value must be rejected instead of
        flowing into the datetime arithmetic in _group_into_trace, where it
        could overflow."""
        backend = _backend()
        item = {
            "span_id": "s1",
            "trace_id": "t1",
            "op": "op",
            "project": "svc",
            "start_timestamp": "2023-01-02T09:42:36.320Z",
            "duration": 1e19,
        }
        assert backend._parse_sentry_trace_item(item, "t1") is None


class TestFlattenTraceItems:
    """Test flattening a nested SerializedTraceItem tree."""

    def test_flattens_nested_children(self) -> None:
        backend = _backend()
        items = [
            {
                "span_id": "root",
                "children": [
                    {"span_id": "child1", "children": [{"span_id": "grandchild"}]},
                    {"span_id": "child2"},
                ],
            }
        ]
        flat = backend._flatten_trace_items(items)
        span_ids = {item["span_id"] for item in flat}
        assert span_ids == {"root", "child1", "child2", "grandchild"}

    def test_skips_non_dict_entries(self) -> None:
        backend = _backend()
        items = [{"span_id": "root"}, "not-a-dict", 123]
        flat = backend._flatten_trace_items(items)
        assert flat == [{"span_id": "root"}]


class TestGetTraceUsesNativeEndpoint:
    """get_trace must call Sentry's native /trace/{id}/ lookup directly
    rather than reconstructing a trace via search+group, unlike Datadog."""

    async def test_calls_native_trace_endpoint(self) -> None:
        backend = _backend()
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: [
            {
                "span_id": "s1",
                "trace_id": "requested",
                "op": "root-op",
                "project": "svc",
                "start_timestamp": "2023-01-02T09:42:36.320Z",
                "duration": 10.0,
            }
        ]
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        trace = await backend.get_trace("requested")

        assert trace.trace_id == "requested"
        call_args = backend._client.get.call_args
        assert call_args.args[0] == "/api/0/organizations/acme/trace/requested/"
        # search_traces's span-search endpoint is never touched by get_trace.
        assert backend._client.get.call_count == 1

    async def test_escapes_trace_id_in_path(self) -> None:
        backend = _backend()
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: []
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        with pytest.raises(ValueError, match="No spans found"):
            await backend.get_trace("evil/../ trace")

        call_args = backend._client.get.call_args
        assert "evil" in call_args.args[0]
        assert " " not in call_args.args[0]


class TestGetTraceExactMatch:
    """Test get_trace only keeps spans exactly matching the requested trace_id."""

    async def test_filters_out_non_matching_spans(self) -> None:
        backend = _backend()
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: [
            {
                "span_id": "s1",
                "trace_id": "requested",
                "op": "op",
                "project": "svc",
                "start_timestamp": "2023-01-02T09:42:36.320Z",
                "duration": 10.0,
            },
            {
                # Wrong trace_id - should be filtered out even though it
                # came back from the endpoint.
                "span_id": "s2",
                "trace_id": "other",
                "op": "op",
                "project": "svc",
                "start_timestamp": "2023-01-02T09:42:36.320Z",
                "duration": 10.0,
            },
        ]
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        trace = await backend.get_trace("requested")

        assert trace.trace_id == "requested"
        assert len(trace.spans) == 1
        assert trace.spans[0].span_id == "s1"

    async def test_raises_when_no_spans_found(self) -> None:
        backend = _backend()
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: []
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        with pytest.raises(ValueError, match="No spans found"):
            await backend.get_trace("missing")

    async def test_raises_on_non_list_response(self) -> None:
        backend = _backend()
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"not": "a list"}
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        with pytest.raises(ValueError, match="Unexpected Sentry trace response shape"):
            await backend.get_trace("t1")


class TestGroupIntoTrace:
    """Test grouping a flat span list into TraceData."""

    def test_group_selects_root_and_aggregates_status(self) -> None:
        backend = _backend()
        now = datetime(2023, 1, 2, 9, 42, 36, tzinfo=UTC)

        root = backend._parse_sentry_trace_item(
            {
                "span_id": "root",
                "trace_id": "t1",
                "op": "root-op",
                "project": "svc",
                "start_timestamp": now.isoformat().replace("+00:00", "Z"),
                "duration": 0.0,
            },
            "t1",
        )
        child = backend._parse_sentry_trace_item(
            {
                "span_id": "child",
                "trace_id": "t1",
                "parent_span_id": "root",
                "op": "child-op",
                "project": "svc",
                "start_timestamp": now.isoformat().replace("+00:00", "Z"),
                "duration": 0.0,
                "errors": [{"event_id": "e1"}],
            },
            "t1",
        )
        assert root is not None and child is not None

        trace = backend._group_into_trace("t1", [root, child])

        assert trace.trace_id == "t1"
        assert trace.service_name == "svc"
        assert trace.root_operation == "root-op"
        assert trace.status == "ERROR"
        assert len(trace.spans) == 2

    def test_group_preserves_unset_when_no_span_confirms_ok_or_error(self) -> None:
        """No span has an explicit error, but none is explicitly OK either -
        the trace status must not silently claim OK (unlike Tempo's
        default-to-OK anti-pattern)."""
        backend = _backend()
        now = datetime(2023, 1, 2, 9, 42, 36, tzinfo=UTC)

        span = backend._parse_sentry_trace_item(
            {
                "span_id": "s1",
                "trace_id": "t2",
                "op": "op",
                "project": "svc",
                "start_timestamp": now.isoformat().replace("+00:00", "Z"),
                "duration": 0.0,
            },
            "t2",
        )
        assert span is not None
        assert span.status == "UNSET"

        trace = backend._group_into_trace("t2", [span])

        assert trace.status == "UNSET"


class TestSearchEventsRawPagination:
    """Test that _search_events_raw follows Sentry's Link-header cursor pagination."""

    async def test_follows_cursor_across_pages(self, fake_json_client: Callable[..., Any]) -> None:
        backend = _backend()

        page_1 = (
            {"data": [{"id": "s1", "trace": "t1"}]},
            {"next": {"results": "true", "cursor": "cursor-1"}},
        )
        page_2 = (
            {"data": [{"id": "s2", "trace": "t1"}]},
            {"next": {"results": "false", "cursor": "cursor-2"}},
        )
        backend._client = fake_json_client(page_1, page_2)

        now = datetime(2023, 1, 2, tzinfo=UTC)
        result = await backend._search_events_raw("", ["id", "trace"], now, now, limit=2)

        assert [row["id"] for row in result] == ["s1", "s2"]

    async def test_stops_at_max_pages_without_infinite_loop(
        self, fake_json_client: Callable[..., Any]
    ) -> None:
        backend = _backend()

        # Always returns a "results": "true" next link - would loop forever
        # without a cap.
        page = (
            {"data": [{"id": "s", "trace": "t"}]},
            {"next": {"results": "true", "cursor": "always-more"}},
        )
        backend._client = fake_json_client(*([page] * _MAX_SEARCH_PAGES))

        now = datetime(2023, 1, 2, tzinfo=UTC)
        result = await backend._search_events_raw("", ["id"], now, now, limit=100_000)

        assert len(result) == _MAX_SEARCH_PAGES

    async def test_no_next_link_stops_pagination(
        self, fake_json_client: Callable[..., Any]
    ) -> None:
        backend = _backend()
        backend._client = fake_json_client(({"data": [{"id": "s1"}]}, {}))

        now = datetime(2023, 1, 2, tzinfo=UTC)
        result = await backend._search_events_raw("", ["id"], now, now, limit=10)

        assert result == [{"id": "s1"}]

    async def test_malformed_data_field_does_not_crash(
        self, fake_json_client: Callable[..., Any]
    ) -> None:
        backend = _backend()
        backend._client = fake_json_client({"data": {"unexpected": {}}})

        now = datetime(2023, 1, 2, tzinfo=UTC)
        result = await backend._search_events_raw("", ["id"], now, now, limit=10)

        assert result == []

    async def test_non_dict_top_level_body_does_not_crash(
        self, fake_json_client: Callable[..., Any]
    ) -> None:
        backend = _backend()
        backend._client = fake_json_client(["not", "an", "object"])

        now = datetime(2023, 1, 2, tzinfo=UTC)
        result = await backend._search_events_raw("", ["id"], now, now, limit=10)

        assert result == []

    async def test_non_dict_entries_in_data_are_skipped(
        self, fake_json_client: Callable[..., Any]
    ) -> None:
        backend = _backend()
        backend._client = fake_json_client({"data": [{"id": "ok"}, "not-a-row", 123]})

        now = datetime(2023, 1, 2, tzinfo=UTC)
        result = await backend._search_events_raw("", ["id"], now, now, limit=10)

        assert result == [{"id": "ok"}]


class TestNextCursorFromLinks:
    """Test the Link-header parsing helper directly."""

    def test_returns_cursor_when_more_results(self) -> None:
        backend = _backend()
        links: dict[str | None, dict[str, str]] = {"next": {"results": "true", "cursor": "abc"}}
        assert backend._next_cursor_from_links(links) == "abc"

    def test_returns_none_when_no_more_results(self) -> None:
        backend = _backend()
        links: dict[str | None, dict[str, str]] = {"next": {"results": "false", "cursor": "abc"}}
        assert backend._next_cursor_from_links(links) is None

    def test_returns_none_when_no_next_link(self) -> None:
        backend = _backend()
        assert backend._next_cursor_from_links({}) is None

    def test_returns_none_when_next_link_not_a_dict(self) -> None:
        backend = _backend()
        assert backend._next_cursor_from_links({"next": "not-a-dict"}) is None  # type: ignore[dict-item]


class TestListServices:
    """Test list_services maps Sentry projects onto the "services" concept."""

    async def test_returns_sorted_project_slugs(self) -> None:
        backend = _backend()
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: [{"slug": "beta"}, {"slug": "alpha"}]
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        services = await backend.list_services()

        assert services == ["alpha", "beta"]

    async def test_non_list_response_returns_empty(self) -> None:
        backend = _backend()
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"not": "a list"}
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        assert await backend.list_services() == []

    async def test_malformed_entries_are_skipped(self) -> None:
        backend = _backend()
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: [{"slug": "ok"}, "not-a-dict", {"no_slug": True}]
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        assert await backend.list_services() == ["ok"]


class TestGetServiceOperations:
    """Test get_service_operations tries attribute-values first, then samples."""

    async def test_uses_attribute_values_endpoint_when_available(self) -> None:
        backend = _backend()
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: ["chat_completion", "embedding"]
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        ops = await backend.get_service_operations("my-service")

        assert ops == ["chat_completion", "embedding"]
        backend._client.get.assert_called_once()

    async def test_falls_back_to_sampling_on_attribute_endpoint_error(self) -> None:
        backend = _backend()
        backend._get_operations_via_attribute_values = AsyncMock(  # type: ignore[method-assign]
            return_value=None
        )
        backend._search_events_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[{"span.op": "chat_completion"}, {"span.op": "embedding"}]
        )

        ops = await backend.get_service_operations("my-service")

        assert ops == ["chat_completion", "embedding"]

    async def test_falls_back_to_sampling_on_unexpected_shape(self) -> None:
        backend = _backend()
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"unexpected": "shape"}
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)
        backend._search_events_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[{"span.op": "sampled_op"}]
        )

        ops = await backend.get_service_operations("my-service")

        assert ops == ["sampled_op"]

    async def test_escapes_service_name_in_sampling_fallback_query(self) -> None:
        backend = _backend()
        backend._search_events_raw = AsyncMock(return_value=[])  # type: ignore[method-assign]

        await backend._get_operations_via_sampling('svc" OR *:*')

        call_args = backend._search_events_raw.call_args
        sentry_query = call_args.args[0]
        assert sentry_query == 'project:"svc\\" OR *:*"'


class TestHealthCheck:
    """Test health_check reports healthy/unhealthy correctly."""

    async def test_healthy_on_success(self) -> None:
        backend = _backend()
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        health = await backend.health_check()

        assert health.status == "healthy"
        assert health.backend == "sentry"

    async def test_unhealthy_on_exception(self) -> None:
        backend = _backend()
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(side_effect=RuntimeError("boom"))

        health = await backend.health_check()

        assert health.status == "unhealthy"
        assert health.backend == "sentry"
        assert health.error == "boom"


def _search_row(
    span_id: str,
    trace_id: str,
    *,
    op: str = "op",
    project: str = "svc",
    timestamp: str = "2023-01-02T09:42:36.320Z",
    duration: float = 10.0,
    status: str = "ok",
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": span_id,
        "trace": trace_id,
        "span.op": op,
        "project": project,
        "timestamp": timestamp,
        "span.duration": duration,
        "span.status": status,
    }
    row.update(extra)
    return row


def _trace_item(
    span_id: str,
    trace_id: str,
    *,
    op: str = "op",
    project: str = "svc",
    start_timestamp: str = "2023-01-02T09:42:36.320Z",
    duration: float = 10.0,
) -> dict[str, Any]:
    return {
        "span_id": span_id,
        "trace_id": trace_id,
        "op": op,
        "project": project,
        "start_timestamp": start_timestamp,
        "duration": duration,
    }


def _hydrated_trace(backend: SentryBackend, trace_id: str, span_id: str) -> TraceData:
    """Build a real TraceData the way get_trace() would, for use as a
    controlled AsyncMock return/side_effect value in the search_traces
    tests below - avoids re-mocking the HTTP layer just to get a
    plausible hydration result."""
    span = backend._parse_sentry_trace_item(_trace_item(span_id, trace_id), trace_id)
    assert span is not None
    return backend._group_into_trace(trace_id, [span])


class TestSearchSpans:
    """search_spans: unlike search_traces, no hydration step - each row from
    the span search is parsed and returned directly (optionally filtered
    client-side for operators Sentry's search syntax doesn't support)."""

    async def test_parses_rows_into_spans(self) -> None:
        backend = _backend()
        backend._search_events_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                _search_row("s1", "t1", op="db.query"),
                _search_row("s2", "t1", op="http.client"),
            ]
        )

        spans = await backend.search_spans(SpanQuery(limit=10))

        assert {s.span_id for s in spans} == {"s1", "s2"}
        assert {s.operation_name for s in spans} == {"db.query", "http.client"}

    async def test_unparseable_rows_are_skipped_not_raised(self) -> None:
        backend = _backend()
        backend._search_events_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                _search_row("s1", "t1"),
                {"id": "s2", "trace": "t1"},  # missing op/project/timestamp/duration
            ]
        )

        spans = await backend.search_spans(SpanQuery(limit=10))

        assert len(spans) == 1
        assert spans[0].span_id == "s1"

    async def test_client_side_filters_are_applied_after_native_search(self) -> None:
        """CONTAINS has no native Sentry mapping (get_supported_operators
        doesn't include it) - it must still be applied client-side rather
        than silently ignored."""
        backend = _backend()
        backend._search_events_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                _search_row("s1", "t1", op="db.query.select"),
                _search_row("s2", "t1", op="http.client"),
            ]
        )

        spans = await backend.search_spans(
            SpanQuery(
                limit=10,
                filters=[
                    Filter(
                        field="operation_name",
                        operator=FilterOperator.CONTAINS,
                        value="db.query",
                        value_type=FilterType.STRING,
                    )
                ],
            )
        )

        assert len(spans) == 1
        assert spans[0].operation_name == "db.query.select"

    async def test_result_is_truncated_to_query_limit(self) -> None:
        backend = _backend()
        backend._search_events_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[_search_row(f"s{i}", "t1") for i in range(5)]
        )

        spans = await backend.search_spans(SpanQuery(limit=2))

        assert len(spans) == 2


class TestSearchTraces:
    """search_traces's real search-then-hydrate flow: discover distinct
    trace_ids from a span search, hydrate each via get_trace, overlay the
    richer search-row attributes/status back onto the hydrated result."""

    async def test_discovers_and_hydrates_multiple_distinct_traces(self) -> None:
        backend = _backend()
        backend._search_events_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                _search_row("s1", "t1"),
                _search_row("s2", "t2"),
            ]
        )
        backend.get_trace = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda trace_id: _hydrated_trace(backend, trace_id, "root-" + trace_id)
        )

        traces = await backend.search_traces(TraceQuery(limit=10))

        assert {t.trace_id for t in traces} == {"t1", "t2"}
        assert backend.get_trace.await_count == 2

    async def test_groups_multiple_rows_under_the_same_trace_id_into_one_hydration_call(
        self,
    ) -> None:
        """Two search rows sharing a trace_id must hydrate that trace exactly
        once, not once per row."""
        backend = _backend()
        backend._search_events_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                _search_row("s1", "t1"),
                _search_row("s2", "t1"),
            ]
        )
        backend.get_trace = AsyncMock(  # type: ignore[method-assign]
            return_value=_hydrated_trace(backend, "t1", "s1")
        )

        traces = await backend.search_traces(TraceQuery(limit=10))

        assert len(traces) == 1
        backend.get_trace.assert_awaited_once_with("t1")

    async def test_rows_with_no_trace_id_are_skipped(self) -> None:
        backend = _backend()
        row_no_trace = _search_row("s1", "t1")
        del row_no_trace["trace"]
        backend._search_events_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[row_no_trace]
        )
        backend.get_trace = AsyncMock()  # type: ignore[method-assign]

        traces = await backend.search_traces(TraceQuery(limit=10))

        assert traces == []
        backend.get_trace.assert_not_awaited()

    async def test_truncates_hydration_to_max_traces_to_hydrate(self) -> None:
        """More distinct trace_ids than _MAX_TRACES_TO_HYDRATE are discovered
        - only the cap's worth are actually hydrated, and a warning is logged
        rather than silently hydrating everything or crashing."""
        backend = _backend()
        rows = [_search_row(f"s{i}", f"t{i}") for i in range(_MAX_TRACES_TO_HYDRATE + 5)]
        backend._search_events_raw = AsyncMock(return_value=rows)  # type: ignore[method-assign]
        backend.get_trace = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda trace_id: _hydrated_trace(backend, trace_id, "s")
        )

        traces = await backend.search_traces(TraceQuery(limit=1000))

        assert backend.get_trace.await_count == _MAX_TRACES_TO_HYDRATE
        assert len(traces) == _MAX_TRACES_TO_HYDRATE

    async def test_a_failed_hydration_is_logged_and_skipped_not_propagated(self) -> None:
        """One trace's get_trace() call raising must not abort the whole
        search - the other, successfully-hydrated traces are still returned."""
        backend = _backend()
        backend._search_events_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                _search_row("s1", "t1"),
                _search_row("s2", "t2"),
            ]
        )

        async def _get_trace(trace_id: str) -> TraceData:
            if trace_id == "t1":
                raise ValueError("No spans found for trace t1")
            return _hydrated_trace(backend, trace_id, "s2")

        backend.get_trace = AsyncMock(side_effect=_get_trace)  # type: ignore[method-assign]

        traces = await backend.search_traces(TraceQuery(limit=10))

        assert len(traces) == 1
        assert traces[0].trace_id == "t2"

    async def test_result_is_truncated_to_query_limit(self) -> None:
        backend = _backend()
        backend._search_events_raw = AsyncMock(  # type: ignore[method-assign]
            return_value=[_search_row(f"s{i}", f"t{i}") for i in range(5)]
        )
        backend.get_trace = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda trace_id: _hydrated_trace(backend, trace_id, "s")
        )

        traces = await backend.search_traces(TraceQuery(limit=2))

        assert len(traces) == 2


class TestEnrichTraceWithSearchRows:
    """_enrich_trace_with_search_rows overlays richer per-span
    attributes/status from the original span search onto a get_trace()
    -hydrated trace, matched by span_id only."""

    def test_overlays_attributes_and_status_for_matching_span_id(self) -> None:
        backend = _backend()
        hydrated_span = backend._parse_sentry_trace_item(_trace_item("s1", "t1"), "t1")
        assert hydrated_span is not None
        # get_trace()'s own endpoint has no gen_ai attributes and no
        # reliable status - simulate that sparse baseline explicitly.
        hydrated_span = hydrated_span.model_copy(update={"status": "UNSET"})
        trace = backend._group_into_trace("t1", [hydrated_span])

        search_row = _search_row("s1", "t1", status="ok")
        search_row["gen_ai.system"] = "openai"

        enriched = backend._enrich_trace_with_search_rows(trace, [search_row])

        assert enriched.spans[0].status == "OK"
        assert enriched.spans[0].attributes.get("gen_ai.system") == "openai"

    def test_spans_with_no_matching_search_row_pass_through_unchanged(self) -> None:
        backend = _backend()
        hydrated_span = backend._parse_sentry_trace_item(_trace_item("s1", "t1"), "t1")
        assert hydrated_span is not None
        trace = backend._group_into_trace("t1", [hydrated_span])

        # Search row is for a completely different span_id.
        enriched = backend._enrich_trace_with_search_rows(
            trace, [_search_row("some-other-span", "t1")]
        )

        assert enriched.spans[0].status == hydrated_span.status
        assert enriched.spans[0] == hydrated_span

    def test_returns_original_trace_unchanged_when_no_rows_parse(self) -> None:
        """Every row fails to parse (e.g. missing required fields) -
        _enrich_trace_with_search_rows must return the original trace as-is,
        not an empty/broken one."""
        backend = _backend()
        hydrated_span = backend._parse_sentry_trace_item(_trace_item("s1", "t1"), "t1")
        assert hydrated_span is not None
        trace = backend._group_into_trace("t1", [hydrated_span])

        unparseable_row = {"id": "s1", "trace": "t1"}  # missing op/project/timestamp/duration

        enriched = backend._enrich_trace_with_search_rows(trace, [unparseable_row])

        assert enriched is trace

    def test_multiple_spans_only_matching_ones_are_overlaid(self) -> None:
        backend = _backend()
        span1 = backend._parse_sentry_trace_item(_trace_item("s1", "t1"), "t1")
        span2 = backend._parse_sentry_trace_item(_trace_item("s2", "t1"), "t1")
        assert span1 is not None and span2 is not None
        span1 = span1.model_copy(update={"status": "UNSET"})
        span2 = span2.model_copy(update={"status": "UNSET"})
        trace = backend._group_into_trace("t1", [span1, span2])

        enriched = backend._enrich_trace_with_search_rows(
            trace, [_search_row("s1", "t1", status="ok")]
        )

        by_id = {s.span_id: s for s in enriched.spans}
        assert by_id["s1"].status == "OK"
        assert by_id["s2"].status == "UNSET"


class TestGetOperationsViaAttributeValues:
    """_get_operations_via_attribute_values: the primary (feature-gated)
    path for get_service_operations, before falling back to sampling."""

    async def test_accepts_bare_string_list(self) -> None:
        backend = _backend()
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: ["db.query", "http.client"]
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        result = await backend._get_operations_via_attribute_values("svc")

        assert result == ["db.query", "http.client"]

    async def test_accepts_list_of_dicts_with_value_key(self) -> None:
        backend = _backend()
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: [{"value": "db.query"}, {"value": "http.client"}]
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        result = await backend._get_operations_via_attribute_values("svc")

        assert result == ["db.query", "http.client"]

    async def test_returns_none_when_response_is_not_a_list(self) -> None:
        backend = _backend()
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"not": "a list"}
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        result = await backend._get_operations_via_attribute_values("svc")

        assert result is None

    async def test_returns_none_when_no_values_extracted(self) -> None:
        """A 200 response that's a list, but contains nothing this backend
        can interpret as a value (e.g. bare numbers) - must degrade to the
        sampling fallback rather than returning an empty list as if that
        were a confirmed answer."""
        backend = _backend()
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: [1, 2, 3]
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        result = await backend._get_operations_via_attribute_values("svc")

        assert result is None

    async def test_returns_none_on_request_exception_and_falls_back(self) -> None:
        backend = _backend()
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "not found",
                request=httpx.Request("GET", "https://sentry.io"),
                response=httpx.Response(404),
            )
        )

        result = await backend._get_operations_via_attribute_values("svc")

        assert result is None

    async def test_get_service_operations_falls_back_to_sampling_on_none(self) -> None:
        backend = _backend()
        backend._get_operations_via_attribute_values = AsyncMock(  # type: ignore[method-assign]
            return_value=None
        )
        backend._get_operations_via_sampling = AsyncMock(  # type: ignore[method-assign]
            return_value=["sampled.op"]
        )

        result = await backend.get_service_operations("svc")

        assert result == ["sampled.op"]
        backend._get_operations_via_sampling.assert_awaited_once_with("svc")

    async def test_get_service_operations_skips_fallback_when_primary_succeeds(self) -> None:
        backend = _backend()
        backend._get_operations_via_attribute_values = AsyncMock(  # type: ignore[method-assign]
            return_value=["primary.op"]
        )
        backend._get_operations_via_sampling = AsyncMock()  # type: ignore[method-assign]

        result = await backend.get_service_operations("svc")

        assert result == ["primary.op"]
        backend._get_operations_via_sampling.assert_not_awaited()


class TestListServicesShapeValidation:
    async def test_returns_empty_list_when_response_is_not_a_list(self) -> None:
        backend = _backend()
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"not": "a list"}
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        result = await backend.list_services()

        assert result == []

    async def test_skips_entries_without_a_valid_slug(self) -> None:
        backend = _backend()
        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: [
            {"slug": "real-project"},
            {"slug": ""},
            {"slug": 123},
            "not-a-dict",
            {},
        ]
        backend._client = AsyncMock()
        backend._client.is_closed = False
        backend._client.get = AsyncMock(return_value=fake_response)

        result = await backend.list_services()

        assert result == ["real-project"]

"""Regression test for the Tempo backend silently dropping spans.

Fixture is a real OTLP JSON trace shape (Tempo's /api/traces/{id} returns
raw OTLP) for a trace containing one span with a real
gen_ai.response.finish_reasons attribute, encoded exactly as OTLP encodes
array-valued attributes on the wire (an arrayValue of stringValue
elements) - not a Python list handed directly to the parser.
"""

import base64
import re
from collections.abc import Callable
from datetime import timedelta
from typing import Any
from urllib.parse import quote

import httpx
import pytest

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.backends.tempo import TempoBackend
from opentelemetry_mcp.models import Filter, FilterOperator, FilterType, SpanQuery, TraceQuery

FAKE_API_KEY = "dd-api1"
FAKE_TEMPO_INSTANCE_ID = "123456"


def _raw_otlp_trace_with_finish_reasons() -> dict[str, Any]:
    return {
        "batches": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "e2e-checkout-service"}}
                    ]
                },
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "abc123",
                                "spanId": "root1",
                                "name": "handle_checkout_request",
                                "startTimeUnixNano": "1000000000",
                                "endTimeUnixNano": "1050000000",
                                "attributes": [],
                            },
                            {
                                "traceId": "abc123",
                                "spanId": "llm1",
                                "parentSpanId": "root1",
                                "name": "llm_summarize_cart",
                                "startTimeUnixNano": "1010000000",
                                "endTimeUnixNano": "1030000000",
                                "attributes": [
                                    {
                                        "key": "gen_ai.system",
                                        "value": {"stringValue": "openai"},
                                    },
                                    {
                                        "key": "gen_ai.request.model",
                                        "value": {"stringValue": "gpt-4"},
                                    },
                                    # Real OTLP wire encoding of a list-valued
                                    # attribute - not a Python list.
                                    {
                                        "key": "gen_ai.response.finish_reasons",
                                        "value": {
                                            "arrayValue": {"values": [{"stringValue": "stop"}]}
                                        },
                                    },
                                ],
                            },
                        ]
                    }
                ],
            }
        ]
    }


def test_span_with_array_valued_attribute_is_not_dropped() -> None:
    backend = TempoBackend(url="http://localhost:3200")

    trace = backend._parse_tempo_trace(_raw_otlp_trace_with_finish_reasons(), trace_id_hex="abc123")

    assert trace is not None
    span_ops = {s.operation_name for s in trace.spans}
    assert "llm_summarize_cart" in span_ops


def test_array_valued_attribute_resolves_to_a_real_list_on_the_span() -> None:
    backend = TempoBackend(url="http://localhost:3200")

    trace = backend._parse_tempo_trace(_raw_otlp_trace_with_finish_reasons(), trace_id_hex="abc123")

    assert trace is not None
    llm_span = next(s for s in trace.spans if s.operation_name == "llm_summarize_cart")
    assert llm_span.attributes.gen_ai_response_finish_reasons == ["stop"]


def test_otlp_array_item_to_str_checks_key_presence_not_truthiness() -> None:
    """A real 0 or False value inside an array must not be mistaken for an
    absent field - the original `or`-chain would have skipped it."""
    assert TempoBackend._otlp_array_item_to_str({"intValue": 0}) == "0"
    assert TempoBackend._otlp_array_item_to_str({"boolValue": False}) == "False"
    assert TempoBackend._otlp_array_item_to_str({"doubleValue": 0.0}) == "0.0"
    assert TempoBackend._otlp_array_item_to_str({}) == ""


def _raw_otlp_trace_with_kvlist_array_attribute() -> dict[str, Any]:
    """Real OTLP wire encoding of gen_ai.input.messages: an arrayValue whose
    elements are kvlistValue objects (role/content pairs), not scalars -
    the shape gen_ai.input.messages/output.messages/retrieval.documents
    all actually use."""
    return {
        "batches": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "e2e-checkout-service"}}
                    ]
                },
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "abc123",
                                "spanId": "llm1",
                                "name": "llm_summarize_cart",
                                "startTimeUnixNano": "1010000000",
                                "endTimeUnixNano": "1030000000",
                                "attributes": [
                                    {
                                        "key": "gen_ai.system",
                                        "value": {"stringValue": "openai"},
                                    },
                                    {
                                        "key": "gen_ai.input.messages",
                                        "value": {
                                            "arrayValue": {
                                                "values": [
                                                    {
                                                        "kvlistValue": {
                                                            "values": [
                                                                {
                                                                    "key": "role",
                                                                    "value": {
                                                                        "stringValue": "user"
                                                                    },
                                                                },
                                                                {
                                                                    "key": "content",
                                                                    "value": {"stringValue": "Hi"},
                                                                },
                                                            ]
                                                        }
                                                    },
                                                    {
                                                        "kvlistValue": {
                                                            "values": [
                                                                {
                                                                    "key": "role",
                                                                    "value": {
                                                                        "stringValue": "assistant"
                                                                    },
                                                                },
                                                                {
                                                                    "key": "content",
                                                                    "value": {
                                                                        "stringValue": "Hello!"
                                                                    },
                                                                },
                                                            ]
                                                        }
                                                    },
                                                ]
                                            }
                                        },
                                    },
                                ],
                            },
                        ]
                    }
                ],
            }
        ]
    }


def test_span_with_kvlist_array_attribute_produces_real_dicts_not_empty_strings() -> None:
    """Before this fix, each kvlistValue array element silently became ""
    - gen_ai.input.messages/output.messages/retrieval.documents are all
    exactly this shape (arrays of objects, not arrays of scalars). Proves
    the fix all the way through to SpanAttributes' typed
    gen_ai_input_messages field, not just the raw parser output."""
    backend = TempoBackend(url="http://localhost:3200")

    trace = backend._parse_tempo_trace(
        _raw_otlp_trace_with_kvlist_array_attribute(), trace_id_hex="abc123"
    )

    assert trace is not None
    llm_span = next(s for s in trace.spans if s.operation_name == "llm_summarize_cart")
    assert llm_span.attributes.gen_ai_input_messages == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]


def test_otlp_kvlist_item_to_dict_checks_key_presence_not_truthiness() -> None:
    """A real 0 or False leaf value must not be mistaken for an absent
    field, mirroring the scalar-array equivalent test above."""
    item = {
        "kvlistValue": {
            "values": [
                {"key": "count", "value": {"intValue": 0}},
                {"key": "verified", "value": {"boolValue": False}},
            ]
        }
    }

    assert TempoBackend._otlp_kvlist_item_to_dict(item) == {"count": 0, "verified": False}
    assert TempoBackend._otlp_kvlist_item_to_dict({}) == {}


def test_scalar_array_attributes_still_resolve_unchanged() -> None:
    """Regression guard: the kvlist dispatch branch must not change
    behavior for the pre-existing pure-scalar-array case."""
    backend = TempoBackend(url="http://localhost:3200")

    trace = backend._parse_tempo_trace(_raw_otlp_trace_with_finish_reasons(), trace_id_hex="abc123")

    assert trace is not None
    llm_span = next(s for s in trace.spans if s.operation_name == "llm_summarize_cart")
    assert llm_span.attributes.gen_ai_response_finish_reasons == ["stop"]


def test_event_attribute_containing_kvlist_array_is_json_encoded_not_raising() -> None:
    """SpanEvent.attributes is scalar-only - a list-of-dicts value on an
    event (not just a span) must be JSON-encoded rather than crashing the
    old ", ".join(v) path, which assumed every list was list[str]."""
    backend = TempoBackend(url="http://localhost:3200")
    raw_trace = {
        "batches": [
            {
                "resource": {"attributes": []},
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "abc123",
                                "spanId": "llm1",
                                "name": "llm_call",
                                "startTimeUnixNano": "1010000000",
                                "endTimeUnixNano": "1030000000",
                                "attributes": [],
                                "events": [
                                    {
                                        "name": "gen_ai.evaluation.result",
                                        "timeUnixNano": "1020000000",
                                        "attributes": [
                                            {
                                                "key": "gen_ai.evaluation.details",
                                                "value": {
                                                    "arrayValue": {
                                                        "values": [
                                                            {
                                                                "kvlistValue": {
                                                                    "values": [
                                                                        {
                                                                            "key": "name",
                                                                            "value": {
                                                                                "stringValue": "relevance"
                                                                            },
                                                                        }
                                                                    ]
                                                                }
                                                            }
                                                        ]
                                                    }
                                                },
                                            }
                                        ],
                                    }
                                ],
                            },
                        ]
                    }
                ],
            }
        ]
    }

    trace = backend._parse_tempo_trace(raw_trace, trace_id_hex="abc123")

    assert trace is not None
    llm_span = next(s for s in trace.spans if s.operation_name == "llm_call")
    assert len(llm_span.events) == 1
    event = llm_span.events[0]
    assert event.name == "gen_ai.evaluation.result"
    assert event.attributes["gen_ai.evaluation.details"] == '[{"name": "relevance"}]'


def test_gen_ai_retrieval_documents_kvlist_array_survives_end_to_end() -> None:
    """Same kvlist-array shape as gen_ai.input.messages, proving the Phase
    1 parser fix generalizes to gen_ai.retrieval.documents too, all the
    way through to its own typed SpanAttributes field."""
    backend = TempoBackend(url="http://localhost:3200")
    raw_attrs = [
        {
            "key": "gen_ai.retrieval.documents",
            "value": {
                "arrayValue": {
                    "values": [
                        {
                            "kvlistValue": {
                                "values": [
                                    {"key": "id", "value": {"stringValue": "doc-1"}},
                                    {"key": "score", "value": {"doubleValue": 0.9}},
                                ]
                            }
                        }
                    ]
                }
            },
        }
    ]

    parsed = backend._parse_otlp_attributes(raw_attrs)
    attrs = SpanAttributes.model_validate(parsed)

    assert attrs.gen_ai_retrieval_documents == [{"id": "doc-1", "score": 0.9}]


def test_bearer_auth_used_when_no_instance_id_set() -> None:
    """Self-hosted Tempo's existing Bearer-token auth must be unchanged
    when tempo_instance_id is not configured."""
    backend = TempoBackend(url="http://localhost:3200", api_key=FAKE_API_KEY)

    headers = backend._create_headers()

    assert headers == {"Authorization": f"Bearer {FAKE_API_KEY}"}


def test_no_auth_header_when_no_api_key_set() -> None:
    """A local Tempo install with no auth configured must get no
    Authorization header at all."""
    backend = TempoBackend(url="http://localhost:3200")

    assert backend._create_headers() == {}


def test_basic_auth_used_when_instance_id_and_api_key_both_set() -> None:
    """Grafana Cloud-hosted Tempo requires Basic Auth with the stack/
    instance ID as username and a Cloud Access Policy token as password -
    this must take precedence over the Bearer-token path."""
    backend = TempoBackend(
        url="https://tempo-prod-01.grafana.net",
        api_key=FAKE_API_KEY,
        tempo_instance_id=FAKE_TEMPO_INSTANCE_ID,
    )

    headers = backend._create_headers()

    expected = base64.b64encode(f"{FAKE_TEMPO_INSTANCE_ID}:{FAKE_API_KEY}".encode()).decode()
    assert headers == {"Authorization": f"Basic {expected}"}


def test_instance_id_without_api_key_falls_back_to_no_auth() -> None:
    """tempo_instance_id alone (no api_key) has no credential to encode,
    so it must not produce a broken Basic Auth header."""
    backend = TempoBackend(
        url="https://tempo-prod-01.grafana.net", tempo_instance_id=FAKE_TEMPO_INSTANCE_ID
    )

    assert backend._create_headers() == {}


# -- Regression tests: TraceQL injection, root-service misattribution, ------
# -- silently-dropped filters, path-injection, and silent-outage findings --


class _FakeResponse:
    """Minimal httpx.Response stand-in for the hand-rolled fake clients
    below - conftest.py's fake_json_client fixture pops canned payloads in
    call order but never exposes the requested URL/params to the test,
    which several of the regression tests below need to assert on."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> Any:
        return self._payload


def _minimal_otlp_trace(trace_id: str, status_code: int) -> dict[str, Any]:
    """Minimal valid /api/traces/{id} OTLP payload with a single span whose
    numeric OTLP status code is 1 (OK) or 2 (ERROR)."""
    return {
        "batches": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "checkout-service"}}
                    ]
                },
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": trace_id,
                                "spanId": "span1",
                                "name": "handle_request",
                                "startTimeUnixNano": "1000000000",
                                "endTimeUnixNano": "1050000000",
                                "status": {"code": status_code},
                                "attributes": [],
                            }
                        ]
                    }
                ],
            }
        ]
    }


def test_filter_to_traceql_escapes_quotes_in_string_value() -> None:
    """A crafted filter value containing a double quote must not be able to
    break out of the TraceQL string literal and inject additional
    structure - the original code spliced Filter.value into the query
    unescaped (f'{traceql_field} = "{value}"')."""
    backend = TempoBackend(url="http://localhost:3200")
    malicious_value = 'x" || resource.service.name =~ ".*'
    filter_obj = Filter(
        field="gen_ai.request.model",
        operator=FilterOperator.EQUALS,
        value=malicious_value,
        value_type=FilterType.STRING,
    )

    condition = backend._filter_to_traceql(filter_obj)

    assert condition == ('span.gen_ai.request.model = "x\\" || resource.service.name =~ \\".*"')


def test_filter_to_traceql_rejects_unsafe_field_name() -> None:
    """Filter.field is an unvalidated MCP tool argument spliced directly
    into the query as `span.{field}` - a crafted field name must be
    rejected rather than interpolated unchecked, mirroring datadog.py's/
    sentry.py's field-name allowlisting."""
    backend = TempoBackend(url="http://localhost:3200")
    filter_obj = Filter(
        field='x" || resource.service.name =~ ".*',
        operator=FilterOperator.EQUALS,
        value="anything",
        value_type=FilterType.STRING,
    )

    assert backend._filter_to_traceql(filter_obj) is None


def test_filter_to_traceql_escapes_regex_metacharacters_for_contains() -> None:
    """CONTAINS is documented as a literal substring match, not a regex
    match - the original code built raw regex via f'.*{value}.*' with no
    re.escape, so a literal '.' in a value like "gpt-4.5" would match "any
    character", causing an unrelated string like "gpt-405" to false-positive."""
    backend = TempoBackend(url="http://localhost:3200")
    filter_obj = Filter(
        field="gen_ai.request.model",
        operator=FilterOperator.CONTAINS,
        value="gpt-4.5",
        value_type=FilterType.STRING,
    )

    condition = backend._filter_to_traceql(filter_obj)

    assert condition is not None
    quoted = condition.split("=~", 1)[1].strip()
    assert quoted.startswith('"') and quoted.endswith('"')
    # Reconstruct the regex TraceQL would actually evaluate by undoing the
    # string-literal escaping (backslash-then-quote) applied on top of the
    # regex escaping.
    raw_pattern = quoted[1:-1].replace('\\"', '"').replace("\\\\", "\\")

    assert re.search(raw_pattern, "call to gpt-4.5 done") is not None
    assert re.search(raw_pattern, "call to gpt-405 done") is None


async def test_get_service_operations_escapes_service_name_in_query() -> None:
    """service_name is an unvalidated MCP tool argument spliced into the
    TraceQL query value - a crafted value must not be able to break out of
    the quoted string and inject additional TraceQL structure."""
    backend = TempoBackend(url="http://localhost:3200")
    captured_params: dict[str, Any] = {}

    class _CapturingClient:
        is_closed = False

        async def get(self, url: str, params: dict[str, Any] | None = None) -> Any:
            captured_params.update(params or {})
            return _FakeResponse({"traces": []})

    backend._client = _CapturingClient()  # type: ignore[assignment]

    malicious_service_name = 'x" || resource.service.name =~ ".*'
    await backend.get_service_operations(malicious_service_name)

    assert captured_params["q"] == (
        '{ resource.service.name = "x\\" || resource.service.name =~ \\".*" }'
    )


async def test_get_service_operations_only_counts_matching_root_service(
    fake_json_client: Callable[..., Any],
) -> None:
    """get_service_operations's TraceQL query matches if service_name
    appears ANYWHERE in the trace (any span), not just at the root - the
    original code unconditionally trusted rootServiceName/rootTraceName,
    misattributing another service's root operation to the queried
    service."""
    backend = TempoBackend(url="http://localhost:3200")
    backend._client = fake_json_client(
        {
            "traces": [
                {"rootServiceName": "checkout-service", "rootTraceName": "checkout_flow"},
                {"rootServiceName": "billing-service", "rootTraceName": "billing_flow"},
            ]
        }
    )

    operations = await backend.get_service_operations("checkout-service")

    assert operations == ["checkout_flow"]


def test_filter_to_traceql_maps_not_equals_error_to_negation() -> None:
    """has_error=False (models.py's _convert_params_to_filters) produces a
    NOT_EQUALS "ERROR" status filter - the original code's status branch
    only handled EQUALS, silently dropping every other operator (including
    this one) by falling through to `return None` with no warning and no
    client-side fallback."""
    backend = TempoBackend(url="http://localhost:3200")
    filter_obj = Filter(
        field="status",
        operator=FilterOperator.NOT_EQUALS,
        value="ERROR",
        value_type=FilterType.STRING,
    )

    assert backend._filter_to_traceql(filter_obj) == "status != error"


async def test_search_traces_applies_unconvertible_status_filter_client_side() -> None:
    """A status filter whose value _filter_to_traceql can't express (e.g.
    "TIMEOUT", neither "ERROR" nor "OK") must still be enforced
    client-side - the original classification logic only checked whether
    an operator was globally "native" (get_supported_operators()), never
    whether _filter_to_traceql actually produced a usable condition for
    this specific field/value, so the filter was silently dropped and
    search_traces returned every trace unfiltered, as if the filter never
    existed."""
    backend = TempoBackend(url="http://localhost:3200")
    ok_trace_id = "aaaa000000000000000000000000000"
    error_trace_id = "bbbb000000000000000000000000000"

    class _SequencedClient:
        is_closed = False

        def __init__(self) -> None:
            self._responses = [
                _FakeResponse({"traces": [{"traceID": ok_trace_id}, {"traceID": error_trace_id}]}),
                _FakeResponse(_minimal_otlp_trace(ok_trace_id, status_code=1)),
                _FakeResponse(_minimal_otlp_trace(error_trace_id, status_code=2)),
            ]

        async def get(self, url: str, *args: Any, **kwargs: Any) -> Any:
            return self._responses.pop(0)

    backend._client = _SequencedClient()  # type: ignore[assignment]

    query = TraceQuery(
        limit=10,
        filters=[
            Filter(
                field="status",
                operator=FilterOperator.EQUALS,
                value="TIMEOUT",
                value_type=FilterType.STRING,
            )
        ],
    )

    traces = await backend.search_traces(query)

    assert traces == []


async def test_get_trace_url_escapes_trace_id() -> None:
    """trace_id is an unvalidated MCP tool argument used as a URL path
    segment - it must be percent-encoded before interpolation (mirrors
    sentry.py's identical quote(trace_id, safe='') at its own
    /trace/{id}/ path-segment call site), so a crafted value can't inject
    additional path structure."""
    backend = TempoBackend(url="http://localhost:3200")
    captured_urls: list[str] = []
    malicious_trace_id = "abc/../../etc?x=1"

    class _CapturingClient:
        is_closed = False

        async def get(self, url: str, *args: Any, **kwargs: Any) -> Any:
            captured_urls.append(url)
            return _FakeResponse(_minimal_otlp_trace("deadbeef", status_code=1))

    backend._client = _CapturingClient()  # type: ignore[assignment]

    await backend.get_trace(malicious_trace_id)

    assert captured_urls == [f"/api/traces/{quote(malicious_trace_id, safe='')}"]
    suffix = captured_urls[0].removeprefix("/api/traces/")
    assert "/" not in suffix
    assert "?" not in suffix


async def test_search_traces_escapes_trace_id_from_search_result_in_fetch_url() -> None:
    """The trace_id used to build the per-trace hydration URL comes from
    Tempo's own /api/search response, but the original code spliced it
    into the request path unescaped - the same unescaped-interpolation
    pattern as get_trace's own trace_id argument."""
    backend = TempoBackend(url="http://localhost:3200")
    captured_urls: list[str] = []
    tricky_trace_id = "abc/def?x=1"

    class _CapturingClient:
        is_closed = False

        def __init__(self) -> None:
            self._call_count = 0

        async def get(self, url: str, params: dict[str, Any] | None = None) -> Any:
            self._call_count += 1
            captured_urls.append(url)
            if self._call_count == 1:
                return _FakeResponse({"traces": [{"traceID": tricky_trace_id}]})
            return _FakeResponse(_minimal_otlp_trace(tricky_trace_id, status_code=1))

    backend._client = _CapturingClient()  # type: ignore[assignment]

    await backend.search_traces(TraceQuery(limit=10))

    assert captured_urls[1] == f"/api/traces/{quote(tricky_trace_id, safe='')}"


async def test_search_traces_raises_when_all_trace_fetches_fail() -> None:
    """A total outage of Tempo's /api/traces/{id} endpoint (while
    /api/search still works) must surface as an error, not be silently
    swallowed into an empty result indistinguishable from "no traces
    matched"."""
    backend = TempoBackend(url="http://localhost:3200")

    class _AllFetchesFailClient:
        is_closed = False

        def __init__(self) -> None:
            self._call_count = 0

        async def get(self, url: str, *args: Any, **kwargs: Any) -> Any:
            self._call_count += 1
            if self._call_count == 1:
                return _FakeResponse({"traces": [{"traceID": "aaaa"}, {"traceID": "bbbb"}]})
            raise httpx.ConnectError("connection refused")

    backend._client = _AllFetchesFailClient()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="per-trace fetch"):
        await backend.search_traces(TraceQuery(limit=10))


async def test_search_spans_raises_when_all_trace_fetches_fail() -> None:
    """Mirrors the search_traces regression above for search_spans, whose
    per-trace hydration loop had the identical silent-failure bug."""
    backend = TempoBackend(url="http://localhost:3200")

    class _AllFetchesFailClient:
        is_closed = False

        def __init__(self) -> None:
            self._call_count = 0

        async def get(self, url: str, *args: Any, **kwargs: Any) -> Any:
            self._call_count += 1
            if self._call_count == 1:
                return _FakeResponse({"traces": [{"traceID": "aaaa"}, {"traceID": "bbbb"}]})
            raise httpx.ConnectError("connection refused")

    backend._client = _AllFetchesFailClient()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="per-trace fetch"):
        await backend.search_spans(SpanQuery(limit=10))


def test_parse_otlp_span_start_time_is_utc_aware() -> None:
    """OTLP's startTimeUnixNano is a UTC epoch timestamp - fromtimestamp()
    without tz=UTC silently shifts it into the server's local timezone and
    returns a naive datetime, unlike every sibling backend doing this same
    conversion (sentry.py, honeycomb.py, newrelic.py, xray.py all pass
    tz=UTC)."""
    backend = TempoBackend(url="http://localhost:3200")
    span_data = {
        "traceId": "abc123",
        "spanId": "span1",
        "name": "handle_request",
        "startTimeUnixNano": "1700000000000000000",
        "endTimeUnixNano": "1700000001000000000",
        "attributes": [],
    }

    span = backend._parse_otlp_span(span_data, "checkout-service")

    assert span is not None
    assert span.start_time.tzinfo is not None
    assert span.start_time.utcoffset() == timedelta(0)


# -- Regression tests: non-string Filter operands spliced raw into TraceQL ----
# Audit: src/opentelemetry_mcp/backends/tempo.py:TempoBackend._filter_to_traceql:unescaped-non-string-value-splice

# A str that merely *claims* value_type number/boolean. Before the fix the
# non-STRING EQUALS/NOT_EQUALS/IN branches and every GT/LT/GTE/LTE branch
# f-string-spliced it verbatim, so it added a top-level OR to the query.
_TRACEQL_PAYLOAD = '1 || resource.service.name =~ ".*"'


def _raw_filter(
    field: str,
    operator: FilterOperator,
    value_type: FilterType,
    *,
    value: object = None,
    values: list[object] | None = None,
) -> Filter:
    """Build a Filter with the model-level operand normalisation bypassed,
    so these tests exercise the TraceQL renderer's own type gate (the last
    decision point before the splice) rather than the pydantic layer."""
    return Filter.model_construct(
        field=field, operator=operator, value=value, values=values, value_type=value_type
    )


class TestFilterToTraceqlNonStringOperands:
    def _backend(self) -> TempoBackend:
        return TempoBackend(url="http://localhost:3200")

    @pytest.mark.parametrize("operator", [FilterOperator.EQUALS, FilterOperator.NOT_EQUALS])
    @pytest.mark.parametrize("value_type", [FilterType.NUMBER, FilterType.BOOLEAN])
    def test_equality_with_str_operand_claiming_non_string_type_is_not_spliced(
        self, operator: FilterOperator, value_type: FilterType
    ) -> None:
        f = _raw_filter("gen_ai.usage.total_tokens", operator, value_type, value=_TRACEQL_PAYLOAD)

        condition = self._backend()._filter_to_traceql(f)

        # None -> _build_traceql_from_filters reclassifies the filter as
        # client-side, where FilterEngine fails closed on the unparsable
        # operand; the payload must never reach the query unquoted.
        assert condition is None

    @pytest.mark.parametrize(
        "operator",
        [FilterOperator.GT, FilterOperator.LT, FilterOperator.GTE, FilterOperator.LTE],
    )
    @pytest.mark.parametrize("field", ["gen_ai.usage.total_tokens", "duration", "duration_ms"])
    def test_range_operator_with_str_operand_returns_none(
        self, operator: FilterOperator, field: str
    ) -> None:
        f = _raw_filter(field, operator, FilterType.NUMBER, value=_TRACEQL_PAYLOAD)
        assert self._backend()._filter_to_traceql(f) is None

    @pytest.mark.parametrize(
        "operator",
        [FilterOperator.GT, FilterOperator.LT, FilterOperator.GTE, FilterOperator.LTE],
    )
    def test_range_operator_with_string_value_type_returns_none(
        self, operator: FilterOperator
    ) -> None:
        f = Filter(field="duration", operator=operator, value="5", value_type=FilterType.STRING)
        assert self._backend()._filter_to_traceql(f) is None

    def test_range_operator_with_genuine_int_on_duration_renders_ms_literal(self) -> None:
        f = Filter(
            field="duration", operator=FilterOperator.GT, value=1, value_type=FilterType.NUMBER
        )
        assert self._backend()._filter_to_traceql(f) == "duration > 1ms"

    def test_range_operator_with_genuine_number_on_attribute_renders_literal(self) -> None:
        f = Filter(
            field="gen_ai.usage.total_tokens",
            operator=FilterOperator.GTE,
            value=5000,
            value_type=FilterType.NUMBER,
        )
        assert self._backend()._filter_to_traceql(f) == "span.gen_ai.usage.total_tokens >= 5000"

    def test_in_number_with_a_str_payload_item_returns_none(self) -> None:
        f = _raw_filter(
            "gen_ai.usage.total_tokens",
            FilterOperator.IN,
            FilterType.NUMBER,
            values=[1, _TRACEQL_PAYLOAD],
        )
        assert self._backend()._filter_to_traceql(f) is None

    def test_in_number_with_genuine_numbers_renders_or_of_literals(self) -> None:
        f = Filter(
            field="x", operator=FilterOperator.IN, values=[1, 2], value_type=FilterType.NUMBER
        )
        assert self._backend()._filter_to_traceql(f) == "(span.x = 1 || span.x = 2)"

    @pytest.mark.parametrize(
        ("value", "value_type", "expected"),
        [
            (1, FilterType.NUMBER, "span.x = 1"),
            (1.5, FilterType.NUMBER, "span.x = 1.5"),
            (True, FilterType.BOOLEAN, "span.x = true"),
            (False, FilterType.BOOLEAN, "span.x = false"),
        ],
    )
    def test_equals_with_genuine_scalar_renders_literal(
        self, value: object, value_type: FilterType, expected: str
    ) -> None:
        f = _raw_filter("x", FilterOperator.EQUALS, value_type, value=value)
        assert self._backend()._filter_to_traceql(f) == expected

    @pytest.mark.parametrize("value", [float("inf"), float("nan"), True, "1"])
    def test_equals_number_with_non_renderable_operand_returns_none(self, value: object) -> None:
        f = _raw_filter("x", FilterOperator.EQUALS, FilterType.NUMBER, value=value)
        assert self._backend()._filter_to_traceql(f) is None

    @pytest.mark.parametrize("value", [1, "true", "x"])
    def test_equals_boolean_with_non_bool_operand_returns_none(self, value: object) -> None:
        f = _raw_filter("x", FilterOperator.EQUALS, FilterType.BOOLEAN, value=value)
        assert self._backend()._filter_to_traceql(f) is None


async def test_tools_call_with_payload_filter_never_reaches_tempo_unquoted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end through the real FastMCP tools/call dispatcher: a filter
    whose str value claims value_type number must not produce a GET
    /api/search whose q parameter carries the payload's injected structure
    (beside test_filter_to_traceql_escapes_quotes_in_string_value, which
    covers the STRING branch). With the model-level normaliser the call is
    rejected as a validation error before any request; the renderer gate
    covers the model_construct bypass separately above."""
    from unittest.mock import AsyncMock

    from fastmcp import Client
    from mcp.shared.exceptions import MCPError

    import opentelemetry_mcp.server as server_module

    async def call_is_rejected(client: Client[Any], name: str, arguments: dict[str, Any]) -> bool:
        # The MCP SDK maps a pydantic ValidationError raised inside a tool to
        # a JSON-RPC INVALID_PARAMS error (surfaced by the client as MCPError
        # even with raise_on_error=False); any other failure is a
        # CallToolResult(isError=True). Both mean the call never ran.
        try:
            result = await client.call_tool(name, arguments, raise_on_error=False)
        except MCPError:
            return True
        return bool(result.is_error)

    backend = TempoBackend(url="http://localhost:3200")
    captured_queries: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/search":
            captured_queries.append(request.url.params.get("q", ""))
            return httpx.Response(200, json={"traces": []})
        return httpx.Response(200, json=_minimal_otlp_trace("deadbeef", status_code=1))

    backend._client = httpx.AsyncClient(
        base_url=backend.url, transport=httpx.MockTransport(handler)
    )
    monkeypatch.setattr(server_module, "_get_backend", AsyncMock(return_value=backend))

    equality_filter = {
        "field": "gen_ai.usage.total_tokens",
        "operator": "equals",
        "value": _TRACEQL_PAYLOAD,
        "value_type": "number",
    }
    range_filter = {
        "field": "duration",
        "operator": "gte",
        "value": _TRACEQL_PAYLOAD,
        "value_type": "number",
    }

    async with Client(server_module.mcp) as client:
        traces_rejected = await call_is_rejected(
            client, "search_traces", {"filters": [equality_filter]}
        )
        spans_rejected = await call_is_rejected(
            client, "search_spans_tool", {"filters": [range_filter]}
        )

    assert traces_rejected
    assert spans_rejected
    assert all("resource.service.name =~" not in q for q in captured_queries)


# -- Regression tests: dot-segment trace ids collapse the request path --------
# Audit: src/opentelemetry_mcp/backends/sentry.py:SentryBackend.get_trace:dot-segment-trace-id


def _wire_tempo_backend(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[TempoBackend, list[str]]:
    """TempoBackend terminating in a MockTransport that records every
    emitted request's raw path (the string handed to a mocked client.get
    cannot show httpx collapsing "." / ".." during the base_url merge)."""
    backend = TempoBackend(url="http://localhost:3200")
    raw_paths: list[str] = []

    async def recording_handler(request: httpx.Request) -> httpx.Response:
        raw_paths.append(request.url.raw_path.decode())
        return handler(request)

    backend._client = httpx.AsyncClient(
        base_url=backend.url, transport=httpx.MockTransport(recording_handler)
    )
    return backend, raw_paths


class TestTempoPathSegmentSafety:
    @pytest.mark.parametrize("trace_id", ["..", ".", ""])
    async def test_get_trace_with_dot_segment_id_raises_and_emits_no_request(
        self, trace_id: str
    ) -> None:
        backend, raw_paths = _wire_tempo_backend(
            lambda request: httpx.Response(200, json=_minimal_otlp_trace("abc", status_code=1))
        )

        with pytest.raises(ValueError, match="Invalid trace_id"):
            await backend.get_trace(trace_id)

        assert raw_paths == []

    @pytest.mark.parametrize("search", ["search_traces", "search_spans"])
    async def test_hydration_skips_dot_segment_ids_without_leaving_api_traces(
        self, search: str
    ) -> None:
        """Both hydration loops read trace ids from Tempo's own /api/search
        rows; a ".." or "." row must never produce a GET outside
        /api/traces/, and the remaining row still hydrates."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/search":
                return httpx.Response(
                    200, json={"traces": [{"traceID": ".."}, {"traceID": "."}, {"traceID": "abc"}]}
                )
            return httpx.Response(200, json=_minimal_otlp_trace("abc", status_code=1))

        backend, raw_paths = _wire_tempo_backend(handler)

        if search == "search_traces":
            results: list[Any] = await backend.search_traces(TraceQuery(limit=10))
        else:
            results = await backend.search_spans(SpanQuery(limit=10))

        assert len(results) >= 1
        fetches = [p for p in raw_paths if not p.startswith("/api/search")]
        assert fetches == ["/api/traces/abc"]

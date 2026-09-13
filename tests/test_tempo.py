"""Regression test for the Tempo backend silently dropping spans.

Fixture is a real OTLP JSON trace shape (Tempo's /api/traces/{id} returns
raw OTLP) for a trace containing one span with a real
gen_ai.response.finish_reasons attribute, encoded exactly as OTLP encodes
array-valued attributes on the wire (an arrayValue of stringValue
elements) - not a Python list handed directly to the parser.
"""

import base64
from typing import Any

from opentelemetry_mcp.backends.tempo import TempoBackend

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

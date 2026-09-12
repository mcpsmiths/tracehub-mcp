"""Regression test for the Tempo backend silently dropping spans.

Fixture is a real OTLP JSON trace shape (Tempo's /api/traces/{id} returns
raw OTLP) for a trace containing one span with a real
gen_ai.response.finish_reasons attribute, encoded exactly as OTLP encodes
array-valued attributes on the wire (an arrayValue of stringValue
elements) - not a Python list handed directly to the parser.
"""

from typing import Any

from opentelemetry_mcp.backends.tempo import TempoBackend


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

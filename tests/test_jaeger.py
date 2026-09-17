"""Regression test for the Jaeger backend silently dropping spans.

Fixture is a real raw Jaeger /api/traces/{id} response shape (confirmed
against a live jaegertracing/all-in-one container during an end-to-end dry
run) for a trace containing one span with a real
gen_ai.response.finish_reasons tag - which Jaeger's tag model always
serializes as a JSON-encoded string, never a native list.
"""

from typing import Any
from unittest.mock import AsyncMock

from opentelemetry_mcp.backends.jaeger import JaegerBackend


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

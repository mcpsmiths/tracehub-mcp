"""Tests for the client-side FilterEngine's logging behaviour.

Audit: tracehub-mcp/logging/filter-engine-logs-raw-span-attribute

PRIVACY.md promises span data is returned to the MCP client and never
stored or forwarded beyond that single request/response. The engine's
type-conversion diagnostic used to interpolate the raw span-side attribute
value (e.g. a prompt) into a WARNING record on the server's stderr log for
every evaluated span. It must carry only the filter field name, operator,
value_type and the Python type name of the offending value - and be emitted
once per apply_filters call, not once per span.
"""

import logging
from datetime import datetime
from typing import Any

import pytest

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.backends.filter_engine import FilterEngine
from opentelemetry_mcp.models import Filter, FilterOperator, FilterType, SpanData

ENGINE_LOGGER = "opentelemetry_mcp.backends.filter_engine"

# Long enough that a truncated interpolation would still be recognisable.
PROMPT_MARKER = "SECRET-PROMPT-" + "x" * 46


def _span(index: int, attrs: dict[str, Any]) -> SpanData:
    return SpanData(
        trace_id="t1",
        span_id=f"s{index}",
        operation_name="chat",
        service_name="svc",
        start_time=datetime.now(),
        duration_ms=10,
        attributes=SpanAttributes.model_validate(attrs),
    )


def _number_filter(field: str, operator: FilterOperator = FilterOperator.GT) -> Filter:
    return Filter(field=field, operator=operator, value=100, value_type=FilterType.NUMBER)


def _records_containing(caplog: pytest.LogCaptureFixture, marker: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if marker in r.getMessage() or marker in str(r.args)]


def _engine_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == ENGINE_LOGGER and r.levelno == logging.WARNING]


class TestTypeConversionWarningIsPayloadFree:
    @pytest.fixture(autouse=True)
    def _capture_everything(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.DEBUG, logger=ENGINE_LOGGER)

    def test_number_filter_on_string_prompt_logs_field_and_types_not_the_value(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        spans = [_span(0, {"gen_ai.prompt": PROMPT_MARKER})]

        assert FilterEngine.apply_filters(spans, [_number_filter("gen_ai.prompt")]) == []

        assert _records_containing(caplog, PROMPT_MARKER) == []
        warnings = _engine_warnings(caplog)
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "gen_ai.prompt" in message
        assert "number" in message
        assert "str" in message

    def test_non_numeric_expected_value_never_logs_the_attribute(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # model_construct: the Filter model itself rejects "abc" for NUMBER,
        # so bypass it to reach the engine's compile-time failure path.
        bad_expected = Filter.model_construct(
            field="gen_ai.prompt",
            operator=FilterOperator.EQUALS,
            value="abc",
            values=None,
            value_type=FilterType.NUMBER,
        )
        spans = [_span(0, {"gen_ai.prompt": PROMPT_MARKER})]

        assert FilterEngine.apply_filters(spans, [bad_expected]) == []

        assert _records_containing(caplog, PROMPT_MARKER) == []
        assert _records_containing(caplog, "abc") == []

    def test_list_typed_attribute_under_number_filter_logs_actual_type_list(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        messages = [{"role": "user", "content": PROMPT_MARKER}]
        spans = [_span(0, {"gen_ai.input.messages": messages})]

        assert FilterEngine.apply_filters(spans, [_number_filter("gen_ai.input.messages")]) == []

        assert _records_containing(caplog, PROMPT_MARKER) == []
        warnings = _engine_warnings(caplog)
        assert len(warnings) == 1
        assert "list" in warnings[0].getMessage()

    def test_string_contains_filter_on_prompt_emits_no_record_with_the_value(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        spans = [_span(0, {"gen_ai.prompt": PROMPT_MARKER})]
        contains = Filter(
            field="gen_ai.prompt",
            operator=FilterOperator.CONTAINS,
            value="SECRET",
            value_type=FilterType.STRING,
        )

        assert FilterEngine.apply_filters(spans, [contains]) == spans

        assert _records_containing(caplog, PROMPT_MARKER) == []
        assert _engine_warnings(caplog) == []

    def test_many_spans_emit_one_warning_and_leak_no_marker(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        markers = [f"{PROMPT_MARKER}-{i}" for i in range(3)]
        spans = [_span(i, {"gen_ai.prompt": marker}) for i, marker in enumerate(markers)]

        assert FilterEngine.apply_filters(spans, [_number_filter("gen_ai.prompt")]) == []

        for marker in markers:
            assert _records_containing(caplog, marker) == []
        # Deduplicated per apply_filters call on (field, value_type, actual
        # type): a 1000-span window must not produce 1000 WARNING lines.
        assert len(_engine_warnings(caplog)) == 1

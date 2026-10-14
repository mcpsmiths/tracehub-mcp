"""Tests for data models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.models import (
    Filter,
    FilterOperator,
    FilterType,
    LLMSpanAttributes,
    SpanData,
    SpanQuery,
    TraceData,
    TraceQuery,
    UsageMetrics,
)


def test_span_data_is_llm_span() -> None:
    """Test LLM span detection."""
    # LLM span
    llm_span = SpanData(
        trace_id="test",
        span_id="span1",
        operation_name="chat.completions",
        service_name="test",
        start_time=datetime.now(),
        duration_ms=100,
        attributes=SpanAttributes.model_validate({"gen_ai.system": "openai"}),
    )
    assert llm_span.is_llm_span is True

    # Non-LLM span
    regular_span = SpanData(
        trace_id="test",
        span_id="span2",
        operation_name="http.request",
        service_name="test",
        start_time=datetime.now(),
        duration_ms=50,
        attributes=SpanAttributes.model_validate({"http.method": "GET"}),
    )
    assert regular_span.is_llm_span is False


def test_llm_span_attributes_from_span() -> None:
    """Test extracting LLM attributes from span."""
    span = SpanData(
        trace_id="test",
        span_id="span1",
        operation_name="chat.completions",
        service_name="test",
        start_time=datetime.now(),
        duration_ms=100,
        attributes=SpanAttributes.model_validate(
            {
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4",
                "gen_ai.usage.prompt_tokens": 150,
                "gen_ai.usage.completion_tokens": 300,
                "gen_ai.usage.total_tokens": 450,
            }
        ),
    )

    llm_attrs = LLMSpanAttributes.from_span(span)
    assert llm_attrs is not None
    assert llm_attrs.system == "openai"
    assert llm_attrs.request_model == "gpt-4"
    assert llm_attrs.prompt_tokens == 150
    assert llm_attrs.completion_tokens == 300
    assert llm_attrs.total_tokens == 450


def test_llm_span_attributes_anthropic_tokens() -> None:
    """Test handling Anthropic token naming (input_tokens vs prompt_tokens)."""
    span = SpanData(
        trace_id="test",
        span_id="span1",
        operation_name="anthropic.messages",
        service_name="test",
        start_time=datetime.now(),
        duration_ms=200,
        attributes=SpanAttributes.model_validate(
            {
                "gen_ai.system": "anthropic",
                "gen_ai.request.model": "claude-3-opus",
                "gen_ai.usage.input_tokens": 100,  # Anthropic uses input_tokens
                "gen_ai.usage.output_tokens": 200,  # Anthropic uses output_tokens
            }
        ),
    )

    llm_attrs = LLMSpanAttributes.from_span(span)
    assert llm_attrs is not None
    assert llm_attrs.system == "anthropic"
    assert llm_attrs.prompt_tokens == 100
    assert llm_attrs.completion_tokens == 200


def test_llm_span_attributes_parses_anthropic_cache_tokens() -> None:
    """Cache-creation/cache-read tokens exist as constants but were never
    wired into parsing - needed for cost attribution's cache-rate lookup."""
    span = SpanData(
        trace_id="test",
        span_id="span1",
        operation_name="anthropic.messages",
        service_name="test",
        start_time=datetime.now(),
        duration_ms=200,
        attributes=SpanAttributes.model_validate(
            {
                "gen_ai.system": "anthropic",
                "gen_ai.request.model": "claude-3-7-sonnet-20250219",
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 200,
                "gen_ai.usage.cache_creation_input_tokens": 50,
                "gen_ai.usage.cache_read_input_tokens": 25,
            }
        ),
    )

    llm_attrs = LLMSpanAttributes.from_span(span)
    assert llm_attrs is not None
    assert llm_attrs.cache_creation_input_tokens == 50
    assert llm_attrs.cache_read_input_tokens == 25


def test_llm_span_attributes_cache_tokens_absent_by_default() -> None:
    span = SpanData(
        trace_id="test",
        span_id="span1",
        operation_name="chat.completions",
        service_name="test",
        start_time=datetime.now(),
        duration_ms=100,
        attributes=SpanAttributes.model_validate({"gen_ai.system": "openai"}),
    )

    llm_attrs = LLMSpanAttributes.from_span(span)
    assert llm_attrs is not None
    assert llm_attrs.cache_creation_input_tokens is None
    assert llm_attrs.cache_read_input_tokens is None


def _llm_span_data(*, model: str, prompt_tokens: int, completion_tokens: int) -> SpanData:
    return SpanData(
        trace_id="test",
        span_id="span1",
        operation_name="chat.completions",
        service_name="svc-a",
        start_time=datetime.now(),
        duration_ms=100,
        attributes=SpanAttributes.model_validate(
            {
                "gen_ai.system": "openai",
                "gen_ai.request.model": model,
                "gen_ai.usage.prompt_tokens": prompt_tokens,
                "gen_ai.usage.completion_tokens": completion_tokens,
            }
        ),
    )


def test_usage_metrics_add_span_accumulates_cost() -> None:
    span = _llm_span_data(model="gpt-4", prompt_tokens=1000, completion_tokens=500)
    llm_attrs = LLMSpanAttributes.from_span(span)
    assert llm_attrs is not None
    metrics = UsageMetrics()

    metrics.add_span(span, llm_attrs)

    assert metrics.cost_usd > 0
    assert metrics.cost_usd_is_partial is False
    assert metrics.by_model["gpt-4"].cost_usd == metrics.cost_usd
    assert metrics.by_service["svc-a"].cost_usd == metrics.cost_usd


def test_usage_metrics_add_span_flips_partial_flag_for_unpriced_model() -> None:
    span = _llm_span_data(
        model="totally-unknown-model-xyz", prompt_tokens=100, completion_tokens=50
    )
    llm_attrs = LLMSpanAttributes.from_span(span)
    assert llm_attrs is not None
    metrics = UsageMetrics()

    metrics.add_span(span, llm_attrs)

    assert metrics.cost_usd == 0.0
    assert metrics.cost_usd_is_partial is True
    assert metrics.by_model["totally-unknown-model-xyz"].cost_usd_is_partial is True


def test_trace_data_llm_spans(sample_trace_data: TraceData) -> None:
    """Test filtering LLM spans from trace."""
    trace = sample_trace_data
    llm_spans = trace.llm_spans

    assert len(llm_spans) == 1
    assert llm_spans[0].is_llm_span is True


def test_trace_data_total_tokens(sample_trace_data: TraceData) -> None:
    """Test total token calculation."""
    trace = sample_trace_data
    total_tokens = trace.total_llm_tokens

    assert total_tokens == 300


def test_trace_query_to_backend_params() -> None:
    """Test converting TraceQuery to backend parameters."""
    query = TraceQuery(
        service_name="my-service",
        operation_name="my-operation",
        min_duration_ms=100,
        limit=50,
        gen_ai_system="openai",
        tags={"custom.tag": "value"},
    )

    params = query.to_backend_params()

    assert params["service"] == "my-service"
    assert params["operation"] == "my-operation"
    assert params["minDuration"] == "100ms"
    assert params["limit"] == 50
    tags = params["tags"]
    assert isinstance(tags, str)
    assert "openai" in tags
    assert "custom.tag" in tags


# -- Filter operand normalisation ---------------------------------------------
# Audit: src/opentelemetry_mcp/backends/tempo.py:TempoBackend._filter_to_traceql:unescaped-non-string-value-splice


class TestFilterOperandNormalization:
    """Filter binds value/values to the declared value_type at validation
    time, so every backend builder downstream can rely on a NUMBER operand
    being a genuine int/float and a BOOLEAN operand being a genuine bool."""

    def test_numeric_string_is_coerced_to_int_for_number(self) -> None:
        f = Filter(
            field="x", operator=FilterOperator.EQUALS, value="5000", value_type=FilterType.NUMBER
        )
        assert f.value == 5000
        assert isinstance(f.value, int)

    def test_decimal_string_is_coerced_to_float_for_number(self) -> None:
        f = Filter(field="x", operator=FilterOperator.GT, value="1.5", value_type=FilterType.NUMBER)
        assert f.value == 1.5
        assert isinstance(f.value, float)

    def test_genuine_numbers_are_left_unchanged(self) -> None:
        assert (
            Filter(
                field="x", operator=FilterOperator.GT, value=5, value_type=FilterType.NUMBER
            ).value
            == 5
        )
        assert (
            Filter(
                field="x", operator=FilterOperator.GT, value=5.5, value_type=FilterType.NUMBER
            ).value
            == 5.5
        )

    @pytest.mark.parametrize("value", ["abc", '1 || resource.service.name =~ ".*"', "nan", "inf"])
    def test_non_numeric_or_non_finite_string_is_rejected_for_number(self, value: str) -> None:
        with pytest.raises(ValidationError):
            Filter(
                field="x", operator=FilterOperator.EQUALS, value=value, value_type=FilterType.NUMBER
            )

    def test_bool_is_rejected_for_number(self) -> None:
        with pytest.raises(ValidationError):
            Filter(
                field="x", operator=FilterOperator.EQUALS, value=True, value_type=FilterType.NUMBER
            )

    @pytest.mark.parametrize(
        "value",
        ["9" * 400, 10**400, -(10**400), 2**1024],
        ids=["digit-string-x400", "10**400", "-(10**400)", "2**1024"],
    )
    def test_number_beyond_float_range_is_rejected(self, value: str | int) -> None:
        """int("9" * 400) - and a JSON integer literal of that size - is a
        genuine int that float() cannot hold (OverflowError, not ValueError).
        Every consumer converts through float() (TraceQL rendering,
        FilterEngine), so the model rejects it up front instead of letting a
        backend crash on it later."""
        with pytest.raises(ValidationError, match="float range"):
            Filter(
                field="x", operator=FilterOperator.EQUALS, value=value, value_type=FilterType.NUMBER
            )

    def test_largest_float_representable_int_is_still_accepted(self) -> None:
        # Boundary control for the test above: 2**1023 still converts to a float.
        f = Filter(
            field="x", operator=FilterOperator.EQUALS, value=2**1023, value_type=FilterType.NUMBER
        )
        assert f.value == 2**1023
        assert isinstance(f.value, int)

    def test_number_beyond_float_range_is_rejected_inside_values(self) -> None:
        with pytest.raises(ValidationError, match="float range"):
            Filter(
                field="x",
                operator=FilterOperator.IN,
                values=[1, 10**400],
                value_type=FilterType.NUMBER,
            )

    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
    def test_non_finite_float_is_rejected_for_number(self, value: float) -> None:
        """The string forms "inf"/"nan" were already rejected; a genuine
        non-finite float declared as a number is the same unusable operand."""
        with pytest.raises(ValidationError, match="finite"):
            Filter(
                field="x", operator=FilterOperator.EQUALS, value=value, value_type=FilterType.NUMBER
            )

    @pytest.mark.parametrize(("value", "expected"), [("true", True), ("FALSE", False)])
    def test_boolean_strings_are_coerced_case_insensitively(
        self, value: str, expected: bool
    ) -> None:
        f = Filter(
            field="x", operator=FilterOperator.EQUALS, value=value, value_type=FilterType.BOOLEAN
        )
        assert f.value is expected

    @pytest.mark.parametrize("value", ["yes", 1, 0, "1"])
    def test_non_boolean_operand_is_rejected_for_boolean(self, value: str | int) -> None:
        with pytest.raises(ValidationError):
            Filter(
                field="x",
                operator=FilterOperator.EQUALS,
                value=value,
                value_type=FilterType.BOOLEAN,
            )

    @pytest.mark.parametrize(("value", "expected"), [(1, "1"), (500, "500"), (1.5, "1.5")])
    def test_scalars_are_coerced_to_str_for_string(self, value: int | float, expected: str) -> None:
        f = Filter(
            field="x", operator=FilterOperator.EQUALS, value=value, value_type=FilterType.STRING
        )
        assert f.value == expected

    def test_values_items_are_normalised_too(self) -> None:
        numbers = Filter(
            field="x", operator=FilterOperator.IN, values=["1", "2.5"], value_type=FilterType.NUMBER
        )
        assert list(numbers.values or []) == [1, 2.5]
        strings = Filter(
            field="x", operator=FilterOperator.IN, values=[1, 2], value_type=FilterType.STRING
        )
        assert list(strings.values or []) == ["1", "2"]
        with pytest.raises(ValidationError):
            Filter(
                field="x", operator=FilterOperator.IN, values=[1, "x"], value_type=FilterType.NUMBER
            )

    def test_unknown_filter_key_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
            Filter(
                field="x",
                operator=FilterOperator.EQUALS,
                value="a",
                value_type=FilterType.STRING,
                logical_operator="AND",  # type: ignore[call-arg]
            )

    def test_empty_field_name_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Filter(
                field="", operator=FilterOperator.EQUALS, value="a", value_type=FilterType.STRING
            )

    def test_exists_operator_still_accepts_no_operand(self) -> None:
        f = Filter(field="x", operator=FilterOperator.EXISTS, value_type=FilterType.NUMBER)
        assert f.value is None and f.values is None


# -- Filter / query cardinality caps -----------------------------------------
# Audit: src/opentelemetry_mcp/models.py:Filter.values:unbounded-filter-cardinality-backend-query-clause-fanout
#
# The limits below are the documented contract (README "Generic Filter
# System"), pinned here as literals on purpose so a silent change to the
# constants cannot drift away from the docs unnoticed.


def _string_filter(index: int) -> Filter:
    return Filter(
        field=f"attr.{index}",
        operator=FilterOperator.EQUALS,
        value="x",
        value_type=FilterType.STRING,
    )


class TestFilterCardinalityCaps:
    def test_in_values_accepts_100_and_rejects_101(self) -> None:
        ok = Filter(
            field="x",
            operator=FilterOperator.IN,
            values=[f"v{i}" for i in range(100)],
            value_type=FilterType.STRING,
        )
        assert len(ok.values or []) == 100

        with pytest.raises(ValidationError, match="too_long|at most 100"):
            Filter(
                field="x",
                operator=FilterOperator.IN,
                values=[f"v{i}" for i in range(101)],
                value_type=FilterType.STRING,
            )

    def test_field_name_accepts_256_chars_and_rejects_257(self) -> None:
        assert (
            len(
                Filter(
                    field="x" * 256,
                    operator=FilterOperator.EQUALS,
                    value="a",
                    value_type=FilterType.STRING,
                ).field
            )
            == 256
        )
        with pytest.raises(ValidationError, match="too_long|at most 256"):
            Filter(
                field="x" * 257,
                operator=FilterOperator.EQUALS,
                value="a",
                value_type=FilterType.STRING,
            )

    @pytest.mark.parametrize("query_cls", [TraceQuery, SpanQuery])
    def test_filters_accept_50_and_reject_51(self, query_cls: type[TraceQuery | SpanQuery]) -> None:
        assert len(query_cls(filters=[_string_filter(i) for i in range(50)]).filters) == 50
        with pytest.raises(ValidationError, match="too_long|at most 50"):
            query_cls(filters=[_string_filter(i) for i in range(51)])

    @pytest.mark.parametrize("query_cls", [TraceQuery, SpanQuery])
    def test_tags_accept_50_and_reject_51(self, query_cls: type[TraceQuery | SpanQuery]) -> None:
        query = query_cls(tags={f"k{i}": "v" for i in range(50)})
        assert len(query.tags) == 50
        converted = query.get_all_filters()
        assert len(converted) == 50
        assert all(f.operator == FilterOperator.EQUALS for f in converted)

        with pytest.raises(ValidationError, match="too_long|at most 50"):
            query_cls(tags={f"k{i}": "v" for i in range(51)})

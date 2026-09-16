"""Tests for data models."""

from datetime import datetime

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.models import (
    LLMSpanAttributes,
    SpanData,
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

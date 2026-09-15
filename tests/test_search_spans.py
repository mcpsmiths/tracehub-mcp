"""Tests for the search_spans tool."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.models import SpanData
from opentelemetry_mcp.tools.search_spans import search_spans


def _fake_backend() -> AsyncMock:
    """An AsyncMock honoring BaseBackend's interface (search_spans is what
    this tool actually calls)."""
    return AsyncMock(spec=BaseBackend)


def _llm_span(**overrides: object) -> SpanData:
    """Build a realistic LLM span, based on the shape of
    conftest.py's sample_span_data/sample_trace_data fixtures."""
    defaults: dict[str, object] = dict(
        trace_id="abc123",
        span_id="span1",
        parent_span_id=None,
        operation_name="chat_completion",
        service_name="test-service",
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        duration_ms=250.0,
        status="OK",
        attributes=SpanAttributes.model_validate(
            {
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4",
                "gen_ai.usage.prompt_tokens": 100,
                "gen_ai.usage.completion_tokens": 200,
                "gen_ai.usage.total_tokens": 300,
            }
        ),
    )
    defaults.update(overrides)
    return SpanData(**defaults)  # type: ignore[arg-type]


def _plain_span(**overrides: object) -> SpanData:
    """A non-LLM span (no gen_ai.* attributes)."""
    defaults: dict[str, object] = dict(
        trace_id="trace1",
        span_id="span1",
        parent_span_id="root0",
        operation_name="db.query",
        service_name="test-service",
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        duration_ms=12.5,
        status="OK",
    )
    defaults.update(overrides)
    return SpanData(**defaults)  # type: ignore[arg-type]


class TestSearchSpansHappyPath:
    """A successful backend call must transform SpanData into the
    documented {"count", "spans": [...]} shape via SpanSummary."""

    async def test_returns_count_and_span_summaries(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = [_llm_span(), _plain_span()]

        result = await search_spans(backend, service_name="test-service")

        assert result.count == 2
        assert len(result.spans) == 2

    async def test_llm_span_summary_has_gen_ai_fields_populated(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = [_llm_span()]

        result = await search_spans(backend, gen_ai_system="openai")
        span = result.spans[0]

        # These fields come from SpanSummary.from_span's LLMSpanAttributes
        # extraction, not a passthrough of the mock - real transform logic.
        assert span.is_llm_span is True
        assert span.gen_ai_system == "openai"
        assert span.total_tokens == 300
        assert span.trace_id == "abc123"
        assert span.span_id == "span1"

    async def test_non_llm_span_summary_has_null_gen_ai_fields(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = [_plain_span()]

        result = await search_spans(backend)
        span = result.spans[0]

        assert span.is_llm_span is False
        assert span.gen_ai_system is None
        assert span.total_tokens is None
        assert span.parent_span_id == "root0"

    async def test_passes_query_object_to_backend(self) -> None:
        """The tool must actually build and forward a SpanQuery reflecting
        the caller's parameters, not just call the backend with anything."""
        backend = _fake_backend()
        backend.search_spans.return_value = []

        await search_spans(
            backend,
            service_name="checkout",
            operation_name="charge",
            min_duration_ms=10,
            max_duration_ms=500,
            has_error=True,
            limit=50,
        )

        query = backend.search_spans.call_args.args[0]
        assert query.service_name == "checkout"
        assert query.operation_name == "charge"
        assert query.min_duration_ms == 10
        assert query.max_duration_ms == 500
        assert query.has_error is True
        assert query.limit == 50

    async def test_tags_and_filters_reach_the_query(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = []

        await search_spans(
            backend,
            tags={"env": "prod"},
            filters=[
                {
                    "field": "traceloop.span.kind",
                    "operator": "equals",
                    "value": "tool",
                    "value_type": "string",
                }
            ],
        )

        query = backend.search_spans.call_args.args[0]
        assert query.tags == {"env": "prod"}
        assert len(query.filters) == 1
        assert query.filters[0].field == "traceloop.span.kind"
        assert query.filters[0].value == "tool"


class TestSearchSpansTimestampValidation:
    """Invalid ISO timestamps must raise (per parse_iso_timestamp's
    (value, error) return shape being converted to a ValueError), so the
    MCP server reports CallToolResult(isError=True) per SEP-2140."""

    async def test_invalid_start_time_raises(self) -> None:
        backend = _fake_backend()

        with pytest.raises(ValueError, match="start_time"):
            await search_spans(backend, start_time="not-a-timestamp")
        backend.search_spans.assert_not_awaited()

    async def test_invalid_end_time_raises(self) -> None:
        backend = _fake_backend()

        with pytest.raises(ValueError, match="end_time"):
            await search_spans(backend, end_time="also-not-a-timestamp")
        backend.search_spans.assert_not_awaited()

    async def test_start_time_checked_before_end_time(self) -> None:
        """Both start_time and end_time are invalid - the function must
        report the start_time failure first (it is parsed first) rather
        than silently picking one at random."""
        backend = _fake_backend()

        with pytest.raises(ValueError, match="start_time"):
            await search_spans(backend, start_time="bad-start", end_time="bad-end")

    async def test_valid_iso_timestamps_reach_the_query(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = []

        await search_spans(
            backend,
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-02T00:00:00Z",
        )

        query = backend.search_spans.call_args.args[0]
        assert query.start_time == datetime(2024, 1, 1, tzinfo=UTC)
        assert query.end_time == datetime(2024, 1, 2, tzinfo=UTC)


class TestSearchSpansFilterValidation:
    """Malformed filter dicts must raise, not be swallowed into error JSON."""

    async def test_missing_required_field_raises_validation_error(self) -> None:
        backend = _fake_backend()

        with pytest.raises(ValidationError):
            await search_spans(
                backend,
                filters=[{"operator": "equals", "value": "x", "value_type": "string"}],
            )
        backend.search_spans.assert_not_awaited()

    async def test_unknown_operator_raises_validation_error(self) -> None:
        backend = _fake_backend()

        with pytest.raises(ValidationError):
            await search_spans(
                backend,
                filters=[
                    {
                        "field": "gen_ai.system",
                        "operator": "not_a_real_operator",
                        "value": "openai",
                        "value_type": "string",
                    }
                ],
            )
        backend.search_spans.assert_not_awaited()

    async def test_in_operator_missing_values_raises_validation_error(self) -> None:
        """Filter's own model_validator requires 'values' (not 'value') for
        the 'in' operator - exercises Filter's cross-field validation, not
        just field presence."""
        backend = _fake_backend()

        with pytest.raises(ValidationError):
            await search_spans(
                backend,
                filters=[
                    {
                        "field": "gen_ai.system",
                        "operator": "in",
                        "value": "openai",
                        "value_type": "string",
                    }
                ],
            )
        backend.search_spans.assert_not_awaited()


class TestSearchSpansQueryValidation:
    """Invalid query-level parameters (caught by SpanQuery's own field
    constraints) must also raise, not become error JSON."""

    async def test_limit_over_max_raises_validation_error(self) -> None:
        backend = _fake_backend()

        with pytest.raises(ValidationError):
            await search_spans(backend, limit=5000)
        backend.search_spans.assert_not_awaited()

    async def test_limit_below_min_raises_validation_error(self) -> None:
        backend = _fake_backend()

        with pytest.raises(ValidationError):
            await search_spans(backend, limit=0)
        backend.search_spans.assert_not_awaited()

    async def test_negative_min_duration_raises_validation_error(self) -> None:
        backend = _fake_backend()

        with pytest.raises(ValidationError):
            await search_spans(backend, min_duration_ms=-1)
        backend.search_spans.assert_not_awaited()


class TestSearchSpansBackendExceptionHandling:
    """A backend that raises must let the exception propagate, so the MCP
    server reports CallToolResult(isError=True) per SEP-2140."""

    async def test_backend_exception_propagates(self) -> None:
        backend = _fake_backend()
        backend.search_spans.side_effect = RuntimeError("upstream unavailable")

        with pytest.raises(RuntimeError, match="upstream unavailable"):
            await search_spans(backend, service_name="svc")

    async def test_backend_exception_of_any_type_propagates(self) -> None:
        backend = _fake_backend()
        backend.search_spans.side_effect = ConnectionError("boom")

        with pytest.raises(ConnectionError, match="boom"):
            await search_spans(backend)


class TestSearchSpansEdgeCases:
    """Edge cases in the module's own branches: empty results, and spans
    with missing optional fields."""

    async def test_empty_result_list_returns_zero_count(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = []

        result = await search_spans(backend, service_name="nonexistent")

        assert result.count == 0
        assert result.spans == []

    async def test_span_with_no_parent_has_null_parent_span_id(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = [_llm_span(parent_span_id=None)]

        result = await search_spans(backend)

        assert result.spans[0].parent_span_id is None

    async def test_llm_span_missing_usage_attributes_has_null_total_tokens(self) -> None:
        """gen_ai.system present but no usage.* attributes at all - the
        LLMSpanAttributes extraction should leave total_tokens as None
        rather than fabricating 0."""
        backend = _fake_backend()
        span = _llm_span(attributes=SpanAttributes.model_validate({"gen_ai.system": "anthropic"}))
        backend.search_spans.return_value = [span]

        result = await search_spans(backend)
        summary = result.spans[0]

        assert summary.is_llm_span is True
        assert summary.gen_ai_system == "anthropic"
        assert summary.total_tokens is None

    async def test_error_status_span_is_preserved_in_summary(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = [_plain_span(status="ERROR")]

        result = await search_spans(backend, has_error=True)

        assert result.spans[0].status == "ERROR"

    async def test_default_limit_is_100_when_unspecified(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = []

        await search_spans(backend)

        query = backend.search_spans.call_args.args[0]
        assert query.limit == 100

    async def test_score_and_evaluation_attributes_are_surfaced(self) -> None:
        """SpanSummary (unlike get_trace's raw attributes.to_dict()) is a
        curated shape with no attributes field at all - extra_attributes
        is what makes score.*/evaluation.*-shaped attributes reach
        search_spans_tool's output."""
        backend = _fake_backend()
        span = _plain_span(
            attributes=SpanAttributes.model_validate(
                {"score.relevance": 0.75, "evaluation.passed": False}
            )
        )
        backend.search_spans.return_value = [span]

        result = await search_spans(backend)
        summary = result.spans[0]

        assert summary.extra_attributes == {"score.relevance": 0.75, "evaluation.passed": False}

    async def test_extra_attributes_is_null_when_none_present(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = [_plain_span()]

        result = await search_spans(backend)

        assert result.spans[0].extra_attributes is None

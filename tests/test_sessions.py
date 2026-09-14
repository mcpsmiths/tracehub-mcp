"""Tests for the list_sessions / get_session_stats tools.

Both tools group spans by the gen_ai.conversation.id attribute. The backend
is mocked at the BaseBackend interface level (not HTTP) since this module
only ever talks to backend.search_spans.
"""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.constants import GenAI
from opentelemetry_mcp.models import FilterOperator, SpanData, SpanQuery
from opentelemetry_mcp.tools.sessions import get_session_stats, list_sessions


def _fake_backend() -> AsyncMock:
    return AsyncMock(spec=BaseBackend)


def _llm_span(**overrides: object) -> SpanData:
    defaults: dict[str, object] = dict(
        trace_id="t1",
        span_id="s1",
        parent_span_id=None,
        operation_name="chat_completion",
        service_name="svc-a",
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        duration_ms=100.0,
        status="OK",
        attributes=SpanAttributes.model_validate(
            {
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4",
                "gen_ai.usage.prompt_tokens": 10,
                "gen_ai.usage.completion_tokens": 20,
                "gen_ai.usage.total_tokens": 30,
                "gen_ai.conversation.id": "conv-1",
            }
        ),
    )
    defaults.update(overrides)
    return SpanData(**defaults)  # type: ignore[arg-type]


def _non_llm_span_with_conversation(**overrides: object) -> SpanData:
    """A tool-call span that shares a conversation_id but carries no
    gen_ai.system - contributes to span_count, not to token/LLM stats."""
    defaults: dict[str, object] = dict(
        trace_id="t1",
        span_id="s2",
        parent_span_id="s1",
        operation_name="search_database",
        service_name="svc-a",
        start_time=datetime(2024, 1, 1, 0, 0, 1, tzinfo=UTC),
        duration_ms=15.0,
        status="OK",
        attributes=SpanAttributes.model_validate({"gen_ai.conversation.id": "conv-1"}),
    )
    defaults.update(overrides)
    return SpanData(**defaults)  # type: ignore[arg-type]


class TestListSessionsHappyPath:
    async def test_spans_sharing_conversation_id_are_grouped(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = [
            _llm_span(span_id="s1"),
            _non_llm_span_with_conversation(span_id="s2"),
        ]

        result = json.loads(await list_sessions(backend))

        assert result["count"] == 1
        session = result["sessions"][0]
        assert session["conversation_id"] == "conv-1"
        assert session["span_count"] == 2
        assert session["total_tokens"] == 30
        assert session["services"] == ["svc-a"]

    async def test_span_without_conversation_id_is_excluded(self) -> None:
        """Defensive check: even though the tool asks the backend to filter
        on EXISTS(gen_ai.conversation.id), a mocked backend does not enforce
        that - the tool's own guard must skip spans lacking the attribute."""
        backend = _fake_backend()
        no_conv_span = _llm_span(span_id="s3")
        no_conv_span.attributes = SpanAttributes.model_validate({"gen_ai.system": "openai"})
        backend.search_spans.return_value = [_llm_span(span_id="s1"), no_conv_span]

        result = json.loads(await list_sessions(backend))

        assert result["count"] == 1
        assert result["sessions"][0]["span_count"] == 1

    async def test_multiple_sessions_sorted_by_span_count_descending(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = [
            _llm_span(
                span_id="a1",
                attributes=SpanAttributes.model_validate(
                    {"gen_ai.system": "openai", "gen_ai.conversation.id": "small-conv"}
                ),
            ),
            _llm_span(span_id="b1"),
            _non_llm_span_with_conversation(span_id="b2"),
            _non_llm_span_with_conversation(span_id="b3"),
        ]

        result = json.loads(await list_sessions(backend))

        assert result["count"] == 2
        assert result["sessions"][0]["conversation_id"] == "conv-1"
        assert result["sessions"][0]["span_count"] == 3
        assert result["sessions"][1]["conversation_id"] == "small-conv"

    async def test_services_are_deduplicated_and_sorted(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = [
            _llm_span(span_id="s1", service_name="zeta"),
            _llm_span(span_id="s2", service_name="alpha"),
            _llm_span(span_id="s3", service_name="alpha"),
        ]

        result = json.loads(await list_sessions(backend))

        assert result["sessions"][0]["services"] == ["alpha", "zeta"]

    async def test_first_seen_and_last_seen_track_min_and_max(self) -> None:
        backend = _fake_backend()
        earliest = datetime(2024, 1, 1, tzinfo=UTC)
        latest = datetime(2024, 1, 10, tzinfo=UTC)
        backend.search_spans.return_value = [
            _llm_span(span_id="s1", start_time=latest),
            _llm_span(span_id="s2", start_time=earliest),
        ]

        result = json.loads(await list_sessions(backend))

        session = result["sessions"][0]
        assert session["first_seen"] == earliest.isoformat().replace("+00:00", "Z")
        assert session["last_seen"] == latest.isoformat().replace("+00:00", "Z")

    async def test_empty_backend_result_returns_documented_message(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = []

        result = json.loads(await list_sessions(backend))

        assert result == {
            "count": 0,
            "sessions": [],
            "message": "No spans with gen_ai.conversation.id found matching the criteria",
        }


class TestListSessionsQueryConstruction:
    async def test_forwards_conversation_id_exists_filter_and_params(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = []

        await list_sessions(
            backend,
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-02T00:00:00Z",
            service_name="svc-a",
            gen_ai_system="openai",
            limit=50,
        )

        query = backend.search_spans.call_args.args[0]
        assert isinstance(query, SpanQuery)
        assert query.service_name == "svc-a"
        assert query.gen_ai_system == "openai"
        assert query.limit == 50
        assert len(query.filters) == 1
        assert query.filters[0].field == GenAI.CONVERSATION_ID
        assert query.filters[0].operator == FilterOperator.EXISTS


class TestListSessionsValidation:
    async def test_invalid_start_time_raises(self) -> None:
        backend = _fake_backend()

        with pytest.raises(ValueError, match="start_time"):
            await list_sessions(backend, start_time="not-a-timestamp")
        backend.search_spans.assert_not_awaited()

    async def test_invalid_end_time_raises(self) -> None:
        backend = _fake_backend()

        with pytest.raises(ValueError, match="end_time"):
            await list_sessions(backend, end_time="not-a-timestamp")
        backend.search_spans.assert_not_awaited()


class TestListSessionsBackendExceptionHandling:
    async def test_backend_exception_propagates(self) -> None:
        backend = _fake_backend()
        backend.search_spans.side_effect = RuntimeError("upstream unavailable")

        with pytest.raises(RuntimeError, match="upstream unavailable"):
            await list_sessions(backend)


class TestGetSessionStatsHappyPath:
    async def test_aggregates_llm_spans_in_the_conversation(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = [
            _llm_span(span_id="s1"),
            _non_llm_span_with_conversation(span_id="s2"),
        ]

        result = json.loads(await get_session_stats(backend, conversation_id="conv-1"))

        assert result["conversation_id"] == "conv-1"
        assert result["span_count"] == 2
        assert result["llm_request_count"] == 1
        assert result["success_count"] == 1
        assert result["error_count"] == 0
        assert result["success_rate"] == 100.0
        assert result["tokens"]["total"]["mean"] == 30.0

    async def test_error_span_counted_in_error_rate(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = [
            _llm_span(span_id="s1", status="ERROR"),
            _llm_span(span_id="s2", status="OK"),
        ]

        result = json.loads(await get_session_stats(backend, conversation_id="conv-1"))

        assert result["llm_request_count"] == 2
        assert result["error_count"] == 1
        assert result["success_count"] == 1
        assert result["error_rate"] == 50.0

    async def test_services_reflect_all_spans_not_just_llm_spans(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = [
            _llm_span(span_id="s1", service_name="chat-svc"),
            _non_llm_span_with_conversation(span_id="s2", service_name="tool-svc"),
        ]

        result = json.loads(await get_session_stats(backend, conversation_id="conv-1"))

        assert result["services"] == ["chat-svc", "tool-svc"]

    async def test_no_spans_found_returns_error_message_not_raise(self) -> None:
        """An empty result is a legitimate answer (no such conversation),
        not a failure - it must not raise."""
        backend = _fake_backend()
        backend.search_spans.return_value = []

        result = json.loads(await get_session_stats(backend, conversation_id="missing-conv"))

        assert "error" in result
        assert "missing-conv" in result["error"]

    async def test_zero_llm_spans_gives_null_rates_not_a_crash(self) -> None:
        """Spans exist (span_count > 0) but none are LLM spans - dividing
        by a zero llm_request_count must produce None, not raise
        ZeroDivisionError."""
        backend = _fake_backend()
        backend.search_spans.return_value = [
            _non_llm_span_with_conversation(span_id="s1"),
            _non_llm_span_with_conversation(span_id="s2"),
        ]

        result = json.loads(await get_session_stats(backend, conversation_id="conv-1"))

        assert result["span_count"] == 2
        assert result["llm_request_count"] == 0
        assert result["success_rate"] is None
        assert result["error_rate"] is None

    async def test_first_seen_and_last_seen_track_min_and_max_out_of_order(self) -> None:
        """Spans intentionally out of chronological order (middle, then
        earliest, then latest) to exercise the min-update branch, not just
        the initial-value case."""
        backend = _fake_backend()
        earliest = datetime(2024, 1, 1, tzinfo=UTC)
        middle = datetime(2024, 1, 5, tzinfo=UTC)
        latest = datetime(2024, 1, 10, tzinfo=UTC)
        backend.search_spans.return_value = [
            _llm_span(span_id="s1", start_time=middle),
            _llm_span(span_id="s2", start_time=earliest),
            _llm_span(span_id="s3", start_time=latest),
        ]

        result = json.loads(await get_session_stats(backend, conversation_id="conv-1"))

        assert result["first_seen"] == earliest.isoformat()
        assert result["last_seen"] == latest.isoformat()

    async def test_finish_reasons_are_counted_across_spans(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = [
            _llm_span(
                span_id="s1",
                attributes=SpanAttributes.model_validate(
                    {
                        "gen_ai.system": "openai",
                        "gen_ai.conversation.id": "conv-1",
                        "gen_ai.response.finish_reasons": ["stop"],
                    }
                ),
            ),
            _llm_span(
                span_id="s2",
                attributes=SpanAttributes.model_validate(
                    {
                        "gen_ai.system": "openai",
                        "gen_ai.conversation.id": "conv-1",
                        "gen_ai.response.finish_reasons": ["stop"],
                    }
                ),
            ),
            _llm_span(
                span_id="s3",
                attributes=SpanAttributes.model_validate(
                    {
                        "gen_ai.system": "openai",
                        "gen_ai.conversation.id": "conv-1",
                        "gen_ai.response.finish_reasons": ["length"],
                    }
                ),
            ),
        ]

        result = json.loads(await get_session_stats(backend, conversation_id="conv-1"))

        assert result["finish_reasons"] == {"stop": 2, "length": 1}


class TestGetSessionStatsQueryConstruction:
    async def test_forwards_conversation_id_equals_filter(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = []

        await get_session_stats(backend, conversation_id="conv-42", service_name="svc-a")

        query = backend.search_spans.call_args.args[0]
        assert isinstance(query, SpanQuery)
        assert query.service_name == "svc-a"
        assert len(query.filters) == 1
        assert query.filters[0].field == GenAI.CONVERSATION_ID
        assert query.filters[0].operator == FilterOperator.EQUALS
        assert query.filters[0].value == "conv-42"


class TestGetSessionStatsValidation:
    async def test_invalid_start_time_raises(self) -> None:
        backend = _fake_backend()

        with pytest.raises(ValueError, match="start_time"):
            await get_session_stats(backend, conversation_id="conv-1", start_time="bad")
        backend.search_spans.assert_not_awaited()

    async def test_invalid_end_time_raises(self) -> None:
        backend = _fake_backend()

        with pytest.raises(ValueError, match="end_time"):
            await get_session_stats(backend, conversation_id="conv-1", end_time="bad")
        backend.search_spans.assert_not_awaited()


class TestGetSessionStatsBackendExceptionHandling:
    async def test_backend_exception_propagates(self) -> None:
        backend = _fake_backend()
        backend.search_spans.side_effect = RuntimeError("backend down")

        with pytest.raises(RuntimeError, match="backend down"):
            await get_session_stats(backend, conversation_id="conv-1")

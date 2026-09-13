"""Tests for the get_prompt_version_stats tool.

Groups spans by gen_ai.prompt.name + gen_ai.prompt.version. The backend is
mocked at the BaseBackend interface level (not HTTP) since this module
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
from opentelemetry_mcp.tools.prompt_versions import get_prompt_version_stats


def _fake_backend() -> AsyncMock:
    return AsyncMock(spec=BaseBackend)


def _prompt_span(**overrides: object) -> SpanData:
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
                "gen_ai.usage.total_tokens": 42,
                "gen_ai.prompt.name": "summarize",
                "gen_ai.prompt.version": "v1",
            }
        ),
    )
    defaults.update(overrides)
    return SpanData(**defaults)  # type: ignore[arg-type]


class TestGetPromptVersionStatsHappyPath:
    async def test_spans_sharing_name_and_version_are_grouped(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = [
            _prompt_span(span_id="s1"),
            _prompt_span(span_id="s2"),
        ]

        result = json.loads(await get_prompt_version_stats(backend))

        assert result["count"] == 1
        entry = result["prompt_versions"][0]
        assert entry["prompt_name"] == "summarize"
        assert entry["prompt_version"] == "v1"
        assert entry["request_count"] == 2
        assert entry["total_tokens"]["mean"] == 42.0

    async def test_different_versions_of_same_prompt_are_kept_separate(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = [
            _prompt_span(span_id="s1"),
            _prompt_span(
                span_id="s2",
                attributes=SpanAttributes.model_validate(
                    {
                        "gen_ai.system": "openai",
                        "gen_ai.prompt.name": "summarize",
                        "gen_ai.prompt.version": "v2",
                    }
                ),
            ),
        ]

        result = json.loads(await get_prompt_version_stats(backend))

        assert result["count"] == 2
        versions = {entry["prompt_version"] for entry in result["prompt_versions"]}
        assert versions == {"v1", "v2"}

    async def test_span_missing_prompt_name_is_excluded(self) -> None:
        """Defensive check: even though the tool asks the backend to filter
        on EXISTS(gen_ai.prompt.name), a mocked backend does not enforce
        that - the tool's own guard must skip spans lacking the attribute."""
        backend = _fake_backend()
        no_name_span = _prompt_span(span_id="s2")
        no_name_span.attributes = SpanAttributes.model_validate({"gen_ai.system": "openai"})
        backend.search_spans.return_value = [_prompt_span(span_id="s1"), no_name_span]

        result = json.loads(await get_prompt_version_stats(backend))

        assert result["count"] == 1
        assert result["prompt_versions"][0]["request_count"] == 1

    async def test_prompt_version_can_be_none_and_still_groups_separately(self) -> None:
        """A prompt.name with no explicit version must still form its own
        group (keyed by (name, None)), not crash or get merged incorrectly."""
        backend = _fake_backend()
        no_version_span = _prompt_span(
            span_id="s1",
            attributes=SpanAttributes.model_validate(
                {"gen_ai.system": "openai", "gen_ai.prompt.name": "summarize"}
            ),
        )
        backend.search_spans.return_value = [no_version_span]

        result = json.loads(await get_prompt_version_stats(backend))

        assert result["count"] == 1
        assert result["prompt_versions"][0]["prompt_version"] is None

    async def test_sorted_by_request_count_descending(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = [
            _prompt_span(
                span_id="rare",
                attributes=SpanAttributes.model_validate(
                    {
                        "gen_ai.system": "openai",
                        "gen_ai.prompt.name": "rare_prompt",
                        "gen_ai.prompt.version": "v1",
                    }
                ),
            ),
            _prompt_span(span_id="s1"),
            _prompt_span(span_id="s2"),
        ]

        result = json.loads(await get_prompt_version_stats(backend))

        assert result["prompt_versions"][0]["prompt_name"] == "summarize"
        assert result["prompt_versions"][0]["request_count"] == 2
        assert result["prompt_versions"][1]["prompt_name"] == "rare_prompt"

    async def test_first_seen_and_last_seen_track_min_and_max(self) -> None:
        """Ordered middle -> earliest -> latest so both the first_seen and
        last_seen update branches (not just the initial-value case) run."""
        backend = _fake_backend()
        earliest = datetime(2024, 1, 1, tzinfo=UTC)
        middle = datetime(2024, 1, 5, tzinfo=UTC)
        latest = datetime(2024, 1, 10, tzinfo=UTC)
        backend.search_spans.return_value = [
            _prompt_span(span_id="s1", start_time=middle),
            _prompt_span(span_id="s2", start_time=earliest),
            _prompt_span(span_id="s3", start_time=latest),
        ]

        result = json.loads(await get_prompt_version_stats(backend))

        entry = result["prompt_versions"][0]
        # prompt_versions.py uses raw datetime.isoformat(), matching
        # list_models.py's own established (unlike sessions.py's Pydantic
        # model_dump) convention - "+00:00", not "Z".
        assert entry["first_seen"] == earliest.isoformat()
        assert entry["last_seen"] == latest.isoformat()

    async def test_empty_backend_result_returns_documented_message(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = []

        result = json.loads(await get_prompt_version_stats(backend))

        assert result == {
            "count": 0,
            "prompt_versions": [],
            "message": "No spans with gen_ai.prompt.name found matching the criteria",
        }


class TestGetPromptVersionStatsQueryConstruction:
    async def test_forwards_prompt_name_exists_filter_and_params(self) -> None:
        backend = _fake_backend()
        backend.search_spans.return_value = []

        await get_prompt_version_stats(
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
        assert query.filters[0].field == GenAI.PROMPT_NAME
        assert query.filters[0].operator == FilterOperator.EXISTS


class TestGetPromptVersionStatsValidation:
    async def test_invalid_start_time_raises(self) -> None:
        backend = _fake_backend()

        with pytest.raises(ValueError, match="start_time"):
            await get_prompt_version_stats(backend, start_time="not-a-timestamp")
        backend.search_spans.assert_not_awaited()

    async def test_invalid_end_time_raises(self) -> None:
        backend = _fake_backend()

        with pytest.raises(ValueError, match="end_time"):
            await get_prompt_version_stats(backend, end_time="not-a-timestamp")
        backend.search_spans.assert_not_awaited()


class TestGetPromptVersionStatsBackendExceptionHandling:
    async def test_backend_exception_propagates(self) -> None:
        backend = _fake_backend()
        backend.search_spans.side_effect = RuntimeError("upstream unavailable")

        with pytest.raises(RuntimeError, match="upstream unavailable"):
            await get_prompt_version_stats(backend)

"""Tests for the compare_time_windows tool.

compare_time_windows is pure composition over get_llm_usage: it calls that
function twice and diffs the two summaries. Mock at
opentelemetry_mcp.tools.compare.get_llm_usage (the name compare.py actually
calls) rather than the backend, so these tests exercise the composition/
diff logic in isolation from get_llm_usage's own (separately tested)
aggregation behavior.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.tools import compare


def _usage_json(
    *,
    total_requests: int,
    total_prompt_tokens: int,
    total_completion_tokens: int,
    total_cost_usd: float = 1.0,
    cost_usd_is_partial: bool = False,
) -> str:
    return json.dumps(
        {
            "period": {"start_time": None, "end_time": None},
            "filters": {},
            "summary": {
                "total_requests": total_requests,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "total_cost_usd": total_cost_usd,
                "cost_usd_is_partial": cost_usd_is_partial,
            },
            "by_model": {},
            "by_service": {},
        }
    )


def _backend() -> AsyncMock:
    return AsyncMock(spec=BaseBackend)


class TestCompareTimeWindowsHappyPath:
    async def test_calls_get_llm_usage_twice_with_each_ranges_times(self) -> None:
        backend = _backend()
        with patch.object(
            compare,
            "get_llm_usage",
            AsyncMock(
                side_effect=[
                    _usage_json(
                        total_requests=10, total_prompt_tokens=100, total_completion_tokens=50
                    ),
                    _usage_json(
                        total_requests=15, total_prompt_tokens=150, total_completion_tokens=75
                    ),
                ]
            ),
        ) as mocked:
            raw = await compare.compare_time_windows(
                backend,
                range_a_start="2024-01-01T00:00:00Z",
                range_a_end="2024-01-08T00:00:00Z",
                range_b_start="2024-01-08T00:00:00Z",
                range_b_end="2024-01-15T00:00:00Z",
            )

        result = json.loads(raw)
        assert mocked.await_count == 2
        first_call_kwargs = mocked.await_args_list[0].kwargs
        second_call_kwargs = mocked.await_args_list[1].kwargs
        assert first_call_kwargs["start_time"] == "2024-01-01T00:00:00Z"
        assert first_call_kwargs["end_time"] == "2024-01-08T00:00:00Z"
        assert second_call_kwargs["start_time"] == "2024-01-08T00:00:00Z"
        assert second_call_kwargs["end_time"] == "2024-01-15T00:00:00Z"
        assert result["range_a"]["summary"]["total_requests"] == 10
        assert result["range_b"]["summary"]["total_requests"] == 15

    async def test_delta_reflects_range_b_minus_range_a(self) -> None:
        backend = _backend()
        with patch.object(
            compare,
            "get_llm_usage",
            AsyncMock(
                side_effect=[
                    _usage_json(
                        total_requests=10, total_prompt_tokens=100, total_completion_tokens=50
                    ),
                    _usage_json(
                        total_requests=15, total_prompt_tokens=150, total_completion_tokens=75
                    ),
                ]
            ),
        ):
            raw = await compare.compare_time_windows(
                backend,
                range_a_start="2024-01-01T00:00:00Z",
                range_a_end="2024-01-08T00:00:00Z",
                range_b_start="2024-01-08T00:00:00Z",
                range_b_end="2024-01-15T00:00:00Z",
            )

        delta = json.loads(raw)["delta"]
        assert delta["total_requests"] == {"change": 5, "percent_change": 50.0}
        assert delta["total_prompt_tokens"] == {"change": 50, "percent_change": 50.0}
        assert delta["total_completion_tokens"] == {"change": 25, "percent_change": 50.0}
        assert delta["total_tokens"] == {"change": 75, "percent_change": 50.0}

    async def test_delta_includes_total_cost_usd(self) -> None:
        backend = _backend()
        with patch.object(
            compare,
            "get_llm_usage",
            AsyncMock(
                side_effect=[
                    _usage_json(
                        total_requests=10,
                        total_prompt_tokens=100,
                        total_completion_tokens=50,
                        total_cost_usd=2.0,
                    ),
                    _usage_json(
                        total_requests=15,
                        total_prompt_tokens=150,
                        total_completion_tokens=75,
                        total_cost_usd=5.0,
                    ),
                ]
            ),
        ):
            raw = await compare.compare_time_windows(
                backend,
                range_a_start="2024-01-01T00:00:00Z",
                range_a_end="2024-01-08T00:00:00Z",
                range_b_start="2024-01-08T00:00:00Z",
                range_b_end="2024-01-15T00:00:00Z",
            )

        delta = json.loads(raw)["delta"]
        assert delta["total_cost_usd"] == {"change": 3.0, "percent_change": 150.0}

    async def test_negative_change_when_range_b_is_lower(self) -> None:
        backend = _backend()
        with patch.object(
            compare,
            "get_llm_usage",
            AsyncMock(
                side_effect=[
                    _usage_json(
                        total_requests=20, total_prompt_tokens=200, total_completion_tokens=100
                    ),
                    _usage_json(
                        total_requests=10, total_prompt_tokens=100, total_completion_tokens=50
                    ),
                ]
            ),
        ):
            raw = await compare.compare_time_windows(
                backend,
                range_a_start="2024-01-01T00:00:00Z",
                range_a_end="2024-01-08T00:00:00Z",
                range_b_start="2024-01-08T00:00:00Z",
                range_b_end="2024-01-15T00:00:00Z",
            )

        delta = json.loads(raw)["delta"]
        assert delta["total_requests"]["change"] == -10
        assert delta["total_requests"]["percent_change"] == -50.0

    async def test_percent_change_is_none_when_range_a_is_zero(self) -> None:
        """Division by zero must not raise - percent_change is None when
        there is nothing in range A to compare against."""
        backend = _backend()
        with patch.object(
            compare,
            "get_llm_usage",
            AsyncMock(
                side_effect=[
                    _usage_json(total_requests=0, total_prompt_tokens=0, total_completion_tokens=0),
                    _usage_json(
                        total_requests=5, total_prompt_tokens=50, total_completion_tokens=25
                    ),
                ]
            ),
        ):
            raw = await compare.compare_time_windows(
                backend,
                range_a_start="2024-01-01T00:00:00Z",
                range_a_end="2024-01-08T00:00:00Z",
                range_b_start="2024-01-08T00:00:00Z",
                range_b_end="2024-01-15T00:00:00Z",
            )

        delta = json.loads(raw)["delta"]
        assert delta["total_requests"] == {"change": 5, "percent_change": None}

    async def test_no_change_between_identical_ranges(self) -> None:
        backend = _backend()
        with patch.object(
            compare,
            "get_llm_usage",
            AsyncMock(
                side_effect=[
                    _usage_json(
                        total_requests=7, total_prompt_tokens=70, total_completion_tokens=30
                    ),
                    _usage_json(
                        total_requests=7, total_prompt_tokens=70, total_completion_tokens=30
                    ),
                ]
            ),
        ):
            raw = await compare.compare_time_windows(
                backend,
                range_a_start="2024-01-01T00:00:00Z",
                range_a_end="2024-01-08T00:00:00Z",
                range_b_start="2024-01-01T00:00:00Z",
                range_b_end="2024-01-08T00:00:00Z",
            )

        delta = json.loads(raw)["delta"]
        for field_delta in delta.values():
            assert field_delta == {"change": 0, "percent_change": 0.0}


class TestCompareTimeWindowsFilterForwarding:
    async def test_filters_are_forwarded_to_both_calls(self) -> None:
        backend = _backend()
        with patch.object(
            compare,
            "get_llm_usage",
            AsyncMock(
                side_effect=[
                    _usage_json(total_requests=1, total_prompt_tokens=1, total_completion_tokens=1),
                    _usage_json(total_requests=1, total_prompt_tokens=1, total_completion_tokens=1),
                ]
            ),
        ) as mocked:
            await compare.compare_time_windows(
                backend,
                range_a_start="2024-01-01T00:00:00Z",
                range_a_end="2024-01-08T00:00:00Z",
                range_b_start="2024-01-08T00:00:00Z",
                range_b_end="2024-01-15T00:00:00Z",
                service_name="svc-a",
                gen_ai_system="openai",
                gen_ai_request_model="gpt-4",
                gen_ai_response_model="gpt-4-turbo",
                limit=250,
            )

        for call in mocked.await_args_list:
            assert call.kwargs["service_name"] == "svc-a"
            assert call.kwargs["gen_ai_system"] == "openai"
            assert call.kwargs["gen_ai_request_model"] == "gpt-4"
            assert call.kwargs["gen_ai_response_model"] == "gpt-4-turbo"
            assert call.kwargs["limit"] == 250


class TestCompareTimeWindowsErrorPropagation:
    async def test_exception_from_first_call_propagates(self) -> None:
        backend = _backend()
        with (
            patch.object(compare, "get_llm_usage", AsyncMock(side_effect=ValueError("bad range"))),
            pytest.raises(ValueError, match="bad range"),
        ):
            await compare.compare_time_windows(
                backend,
                range_a_start="not-a-timestamp",
                range_a_end="2024-01-08T00:00:00Z",
                range_b_start="2024-01-08T00:00:00Z",
                range_b_end="2024-01-15T00:00:00Z",
            )

"""Tests for investigate_cost_spike / investigate_error_spike.

investigate_cost_spike composes over get_llm_usage exactly like
compare_time_windows does, so its tests mock
opentelemetry_mcp.tools.investigate.get_llm_usage at the module level
(isolating the ranking/delta logic from get_llm_usage's own, separately
tested aggregation). investigate_error_spike talks to the backend directly
(one search_traces call per window), so its tests mock backend.search_traces
instead.
"""

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.models import SpanData, TraceData
from opentelemetry_mcp.tools import investigate


def _usage_json(
    *,
    total_requests: int = 0,
    total_tokens: int = 0,
    total_cost_usd: float = 0.0,
    cost_usd_is_partial: bool = False,
    by_model: dict[str, float] | None = None,
    by_service: dict[str, float] | None = None,
) -> str:
    """Build a minimal get_llm_usage-shaped JSON string, keyed by model/
    service name -> cost_usd (the only field investigate_cost_spike reads
    from each breakdown entry)."""
    return json.dumps(
        {
            "period": {"start_time": None, "end_time": None},
            "filters": {},
            "summary": {
                "total_requests": total_requests,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": total_tokens,
                "total_cost_usd": total_cost_usd,
                "cost_usd_is_partial": cost_usd_is_partial,
            },
            "by_model": {k: {"cost_usd": v} for k, v in (by_model or {}).items()},
            "by_service": {k: {"cost_usd": v} for k, v in (by_service or {}).items()},
        }
    )


def _backend() -> AsyncMock:
    return AsyncMock(spec=BaseBackend)


class TestInvestigateCostSpikeAutoBaseline:
    async def test_auto_computed_baseline_is_same_duration_immediately_preceding(self) -> None:
        backend = _backend()
        with patch.object(
            investigate,
            "get_llm_usage",
            AsyncMock(side_effect=[_usage_json(), _usage_json()]),
        ) as mocked:
            raw = await investigate.investigate_cost_spike(
                backend,
                recent_start="2024-01-08T00:00:00+00:00",
                recent_end="2024-01-15T00:00:00+00:00",
            )

        result = json.loads(raw)
        assert result["baseline"]["auto_computed"] is True
        baseline_call_kwargs = mocked.await_args_list[1].kwargs
        assert baseline_call_kwargs["start_time"] == "2024-01-01T00:00:00+00:00"
        assert baseline_call_kwargs["end_time"] == "2024-01-08T00:00:00+00:00"

    async def test_missing_recent_bounds_with_no_explicit_baseline_raises(self) -> None:
        backend = _backend()

        with pytest.raises(ValueError, match="auto-computed baseline"):
            await investigate.investigate_cost_spike(backend, recent_start="", recent_end="")


class TestInvestigateCostSpikeExplicitBaseline:
    async def test_explicit_baseline_is_used_verbatim_and_not_marked_auto_computed(self) -> None:
        backend = _backend()
        with patch.object(
            investigate,
            "get_llm_usage",
            AsyncMock(side_effect=[_usage_json(), _usage_json()]),
        ) as mocked:
            raw = await investigate.investigate_cost_spike(
                backend,
                recent_start="2024-01-08T00:00:00Z",
                recent_end="2024-01-15T00:00:00Z",
                baseline_start="2023-01-08T00:00:00Z",
                baseline_end="2023-01-15T00:00:00Z",
            )

        result = json.loads(raw)
        assert result["baseline"]["auto_computed"] is False
        baseline_call_kwargs = mocked.await_args_list[1].kwargs
        assert baseline_call_kwargs["start_time"] == "2023-01-08T00:00:00Z"
        assert baseline_call_kwargs["end_time"] == "2023-01-15T00:00:00Z"

    async def test_partial_baseline_raises(self) -> None:
        backend = _backend()

        with pytest.raises(ValueError, match="both be provided, or both omitted"):
            await investigate.investigate_cost_spike(
                backend,
                recent_start="2024-01-08T00:00:00Z",
                recent_end="2024-01-15T00:00:00Z",
                baseline_start="2023-01-08T00:00:00Z",
            )


class TestInvestigateCostSpikeRanking:
    async def test_ranks_models_by_absolute_cost_change_descending(self) -> None:
        backend = _backend()
        with patch.object(
            investigate,
            "get_llm_usage",
            AsyncMock(
                side_effect=[
                    _usage_json(by_model={"gpt-4": 50.0, "claude-3": 1.0, "gpt-3.5": 5.0}),
                    _usage_json(by_model={"gpt-4": 10.0, "claude-3": 1.5, "gpt-3.5": 5.0}),
                ]
            ),
        ):
            raw = await investigate.investigate_cost_spike(
                backend,
                recent_start="2024-01-08T00:00:00Z",
                recent_end="2024-01-15T00:00:00Z",
                baseline_start="2024-01-01T00:00:00Z",
                baseline_end="2024-01-08T00:00:00Z",
                top_n=2,
            )

        result = json.loads(raw)
        names = [c["name"] for c in result["top_model_contributors"]]
        assert names == ["gpt-4", "claude-3"]
        assert len(result["top_model_contributors"]) == 2

    async def test_missing_key_on_one_side_is_treated_as_zero(self) -> None:
        """A brand-new model with zero baseline cost is exactly the
        interesting spike signal, not something to skip."""
        backend = _backend()
        with patch.object(
            investigate,
            "get_llm_usage",
            AsyncMock(
                side_effect=[
                    _usage_json(by_model={"gpt-5-preview": 100.0}),
                    _usage_json(by_model={}),
                ]
            ),
        ):
            raw = await investigate.investigate_cost_spike(
                backend,
                recent_start="2024-01-08T00:00:00Z",
                recent_end="2024-01-15T00:00:00Z",
                baseline_start="2024-01-01T00:00:00Z",
                baseline_end="2024-01-08T00:00:00Z",
            )

        result = json.loads(raw)
        entry = result["top_model_contributors"][0]
        assert entry["name"] == "gpt-5-preview"
        assert entry["baseline_cost_usd"] == 0
        assert entry["recent_cost_usd"] == 100.0
        assert entry["change"] == 100.0

    async def test_top_n_is_capped_at_50(self) -> None:
        backend = _backend()
        many_models = {f"model-{i}": float(i) for i in range(80)}
        with patch.object(
            investigate,
            "get_llm_usage",
            AsyncMock(side_effect=[_usage_json(by_model=many_models), _usage_json(by_model={})]),
        ):
            raw = await investigate.investigate_cost_spike(
                backend,
                recent_start="2024-01-08T00:00:00Z",
                recent_end="2024-01-15T00:00:00Z",
                baseline_start="2024-01-01T00:00:00Z",
                baseline_end="2024-01-08T00:00:00Z",
                top_n=1000,
            )

        result = json.loads(raw)
        assert len(result["top_model_contributors"]) == 50

    async def test_cost_usd_is_partial_bubbles_up_from_either_window(self) -> None:
        backend = _backend()
        with patch.object(
            investigate,
            "get_llm_usage",
            AsyncMock(
                side_effect=[
                    _usage_json(cost_usd_is_partial=False),
                    _usage_json(cost_usd_is_partial=True),
                ]
            ),
        ):
            raw = await investigate.investigate_cost_spike(
                backend,
                recent_start="2024-01-08T00:00:00Z",
                recent_end="2024-01-15T00:00:00Z",
                baseline_start="2024-01-01T00:00:00Z",
                baseline_end="2024-01-08T00:00:00Z",
            )

        assert json.loads(raw)["cost_usd_is_partial"] is True


def _error_span(
    *,
    span_id: str,
    service_name: str = "svc-a",
    error_type: str = "RuntimeError",
    error_message: str = "boom",
    gen_ai_system: str | None = None,
    request_model: str | None = None,
) -> SpanData:
    attrs: dict[str, Any] = {"error.type": error_type, "error.message": error_message}
    if gen_ai_system:
        attrs["gen_ai.system"] = gen_ai_system
    if request_model:
        attrs["gen_ai.request.model"] = request_model
    return SpanData(
        trace_id=f"t-{span_id}",
        span_id=span_id,
        parent_span_id=None,
        operation_name="op",
        service_name=service_name,
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        duration_ms=10.0,
        status="ERROR",
        attributes=SpanAttributes.model_validate(attrs),
    )


def _ok_span(*, span_id: str, service_name: str = "svc-a") -> SpanData:
    return SpanData(
        trace_id=f"t-{span_id}",
        span_id=span_id,
        parent_span_id=None,
        operation_name="op",
        service_name=service_name,
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        duration_ms=10.0,
        status="OK",
        attributes=SpanAttributes.model_validate({}),
    )


def _trace(trace_id: str, spans: list[SpanData], *, status: str = "OK") -> TraceData:
    return TraceData(
        trace_id=trace_id,
        spans=spans,
        start_time=spans[0].start_time,
        duration_ms=10.0,
        service_name=spans[0].service_name,
        root_operation=spans[0].operation_name,
        status=status,  # type: ignore[arg-type]
    )


class TestInvestigateErrorSpikeRateCalculation:
    async def test_error_rate_computed_per_window(self) -> None:
        backend = _backend()
        recent_traces = [
            _trace("t1", [_error_span(span_id="s1")], status="ERROR"),
            _trace("t2", [_ok_span(span_id="s2")]),
        ]
        baseline_traces = [_trace("t3", [_ok_span(span_id="s3")])]
        backend.search_traces = AsyncMock(side_effect=[recent_traces, baseline_traces])

        raw = await investigate.investigate_error_spike(
            backend,
            recent_start="2024-01-08T00:00:00Z",
            recent_end="2024-01-15T00:00:00Z",
            baseline_start="2024-01-01T00:00:00Z",
            baseline_end="2024-01-08T00:00:00Z",
        )

        result = json.loads(raw)
        assert result["recent"]["total_traces"] == 2
        assert result["recent"]["error_traces"] == 1
        assert result["recent"]["error_rate"] == 0.5
        assert result["baseline"]["error_rate"] == 0.0

    async def test_zero_trace_baseline_gives_null_rate_not_a_crash(self) -> None:
        backend = _backend()
        recent_traces = [_trace("t1", [_error_span(span_id="s1")], status="ERROR")]
        backend.search_traces = AsyncMock(side_effect=[recent_traces, []])

        raw = await investigate.investigate_error_spike(
            backend,
            recent_start="2024-01-08T00:00:00Z",
            recent_end="2024-01-15T00:00:00Z",
            baseline_start="2024-01-01T00:00:00Z",
            baseline_end="2024-01-08T00:00:00Z",
        )

        result = json.loads(raw)
        assert result["baseline"]["error_rate"] is None
        assert result["baseline"]["total_traces"] == 0


class TestInvestigateErrorSpikeIsSpikeThreshold:
    async def test_below_count_floor_is_not_a_spike_even_with_high_rate_multiplier(self) -> None:
        """1 -> 2 errors is a 2x rate multiplier but must not read as a
        spike - the absolute count floor (default 3) guards tiny samples."""
        backend = _backend()
        recent_traces = [
            _trace("t1", [_error_span(span_id="s1")], status="ERROR"),
            _trace("t2", [_error_span(span_id="s2")], status="ERROR"),
        ]
        baseline_traces = [
            _trace("t3", [_error_span(span_id="s3")], status="ERROR"),
            _trace("t4", [_ok_span(span_id="s4")]),
        ]
        backend.search_traces = AsyncMock(side_effect=[recent_traces, baseline_traces])

        raw = await investigate.investigate_error_spike(
            backend,
            recent_start="2024-01-08T00:00:00Z",
            recent_end="2024-01-15T00:00:00Z",
            baseline_start="2024-01-01T00:00:00Z",
            baseline_end="2024-01-08T00:00:00Z",
        )

        assert json.loads(raw)["is_spike"] is False

    async def test_zero_baseline_rate_with_any_nonzero_recent_rate_counts_as_spike(self) -> None:
        backend = _backend()
        recent_traces = [
            _trace(f"t{i}", [_error_span(span_id=f"s{i}")], status="ERROR") for i in range(4)
        ]
        baseline_traces = [_trace("t-b", [_ok_span(span_id="s-b")])]
        backend.search_traces = AsyncMock(side_effect=[recent_traces, baseline_traces])

        raw = await investigate.investigate_error_spike(
            backend,
            recent_start="2024-01-08T00:00:00Z",
            recent_end="2024-01-15T00:00:00Z",
            baseline_start="2024-01-01T00:00:00Z",
            baseline_end="2024-01-08T00:00:00Z",
        )

        assert json.loads(raw)["is_spike"] is True

    async def test_both_conditions_required_rate_ok_but_count_too_low(self) -> None:
        backend = _backend()
        # Baseline: 1 error in 10 traces (rate 0.1). Recent: 3 errors in 10
        # traces (rate 0.3, a 3x multiplier - passes rate_multiplier_threshold)
        # but count only increased by 2, below the default floor of 3.
        recent_traces = [
            _trace(f"t{i}", [_error_span(span_id=f"s{i}")], status="ERROR") for i in range(3)
        ] + [_trace(f"t{i}", [_ok_span(span_id=f"s{i}")]) for i in range(3, 10)]
        baseline_traces = [_trace("tb0", [_error_span(span_id="sb0")], status="ERROR")] + [
            _trace(f"tb{i}", [_ok_span(span_id=f"sb{i}")]) for i in range(1, 10)
        ]
        backend.search_traces = AsyncMock(side_effect=[recent_traces, baseline_traces])

        raw = await investigate.investigate_error_spike(
            backend,
            recent_start="2024-01-08T00:00:00Z",
            recent_end="2024-01-15T00:00:00Z",
            baseline_start="2024-01-01T00:00:00Z",
            baseline_end="2024-01-08T00:00:00Z",
        )

        assert json.loads(raw)["is_spike"] is False


class TestInvestigateErrorSpikeRanking:
    async def test_ranks_service_model_and_error_type_contributors(self) -> None:
        backend = _backend()
        recent_traces = [
            _trace(
                "t1",
                [
                    _error_span(
                        span_id="s1",
                        service_name="checkout",
                        error_type="RateLimitError",
                        gen_ai_system="openai",
                        request_model="gpt-4",
                    )
                ],
                status="ERROR",
            ),
            _trace(
                "t2",
                [
                    _error_span(
                        span_id="s2",
                        service_name="checkout",
                        error_type="RateLimitError",
                        gen_ai_system="openai",
                        request_model="gpt-4",
                    )
                ],
                status="ERROR",
            ),
            _trace(
                "t3",
                [_error_span(span_id="s3", service_name="checkout", error_type="RateLimitError")],
                status="ERROR",
            ),
        ]
        backend.search_traces = AsyncMock(side_effect=[recent_traces, []])

        raw = await investigate.investigate_error_spike(
            backend,
            recent_start="2024-01-08T00:00:00Z",
            recent_end="2024-01-15T00:00:00Z",
            baseline_start="2024-01-01T00:00:00Z",
            baseline_end="2024-01-08T00:00:00Z",
            min_error_count_increase=1,
        )

        result = json.loads(raw)
        assert result["top_service_contributors"][0]["name"] == "checkout"
        assert result["top_service_contributors"][0]["recent_count"] == 3
        assert result["top_model_contributors"][0]["name"] == "gpt-4"
        assert result["top_model_contributors"][0]["recent_count"] == 2
        assert result["top_error_type_contributors"][0]["name"] == "RateLimitError"
        assert result["top_error_type_contributors"][0]["recent_count"] == 3
        assert len(result["top_error_type_contributors"][0]["sample_messages"]) <= 3

    async def test_sample_messages_capped_at_three(self) -> None:
        backend = _backend()
        recent_traces = [
            _trace(
                f"t{i}",
                [_error_span(span_id=f"s{i}", error_type="Boom", error_message=f"msg-{i}")],
                status="ERROR",
            )
            for i in range(5)
        ]
        backend.search_traces = AsyncMock(side_effect=[recent_traces, []])

        raw = await investigate.investigate_error_spike(
            backend,
            recent_start="2024-01-08T00:00:00Z",
            recent_end="2024-01-15T00:00:00Z",
            baseline_start="2024-01-01T00:00:00Z",
            baseline_end="2024-01-08T00:00:00Z",
            min_error_count_increase=1,
        )

        result = json.loads(raw)
        entry = next(e for e in result["top_error_type_contributors"] if e["name"] == "Boom")
        assert len(entry["sample_messages"]) == 3


class TestInvestigateErrorSpikeValidation:
    async def test_lopsided_baseline_params_raise(self) -> None:
        backend = _backend()

        with pytest.raises(ValueError, match="both be provided, or both omitted"):
            await investigate.investigate_error_spike(
                backend,
                recent_start="2024-01-08T00:00:00Z",
                recent_end="2024-01-15T00:00:00Z",
                baseline_end="2024-01-08T00:00:00Z",
            )

"""Tests for ``XRayBackend.search_spans``.

X-Ray has no spans API, so ``search_spans`` runs GetTraceSummaries (trace
discovery, with only the natively-pushable filters in ``FilterExpression``),
hydrates at most ``_MAX_TRACES_TO_HYDRATE`` traces through BatchGetTraces,
flattens every trace's segment tree into spans, re-applies EVERY filter
client-side, and slices to ``query.limit``.

Like tests/test_xray.py this uses ``botocore.stub.Stubber``: every request
the backend builds is validated against the real ``xray`` service model, and
``expected_params`` pins the exact request shape (Stubber requires the
expected and actual parameter name sets to match, so an unexpected
``FilterExpression`` or ``NextToken`` fails a test).
"""

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber

from opentelemetry_mcp.backends.xray import (
    _BATCH_GET_TRACES_MAX_IDS,
    _MAX_TRACES_TO_HYDRATE,
    XRayBackend,
)
from opentelemetry_mcp.models import Filter, FilterOperator, FilterType, SpanQuery

_START_EPOCH = 1_700_000_000.0


def _backend() -> XRayBackend:
    return XRayBackend(url="https://xray.us-east-1.amazonaws.com", aws_region="us-east-1")


def _segment(seg_id: str, name: str, *, seconds: float = 1.0, **extra: Any) -> dict[str, Any]:
    """One BatchGetTraces ``Segments`` item: ``Document`` is the segment
    document as a JSON-encoded STRING (X-Ray's double encoding)."""
    document: dict[str, Any] = {
        "id": seg_id,
        "name": name,
        "start_time": _START_EPOCH,
        "end_time": _START_EPOCH + seconds,
    }
    document.update(extra)
    return {"Id": seg_id, "Document": json.dumps(document)}


def _trace(trace_id: str, *segments: dict[str, Any]) -> dict[str, Any]:
    return {"Id": trace_id, "Segments": list(segments)}


def _summaries(*trace_ids: str) -> dict[str, Any]:
    return {"TraceSummaries": [{"Id": trace_id} for trace_id in trace_ids]}


def _batch(*traces: dict[str, Any]) -> dict[str, Any]:
    return {"Traces": list(traces), "UnprocessedTraceIds": []}


def _eq_filter(field: str, value: str) -> Filter:
    return Filter(
        field=field, operator=FilterOperator.EQUALS, value=value, value_type=FilterType.STRING
    )


class _Between:
    """``expected_params`` matcher for a datetime computed from the wall
    clock inside the backend (the default window)."""

    def __init__(self, low: datetime, high: datetime) -> None:
        self._low = low
        self._high = high

    def __eq__(self, other: object) -> bool:
        return isinstance(other, datetime) and self._low <= other <= self._high


WINDOW_START = datetime(2025, 3, 1, 12, 0, 0, tzinfo=UTC)
WINDOW_END = datetime(2025, 3, 1, 13, 30, 0, tzinfo=UTC)


class TestRequestShape:
    async def test_explicit_window_is_sent_verbatim_and_no_filter_means_no_expression(
        self,
    ) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_response(
            "get_trace_summaries",
            _summaries("1-aaa", "1-bbb"),
            expected_params={"StartTime": WINDOW_START, "EndTime": WINDOW_END},
        )
        stubber.add_response(
            "batch_get_traces",
            _batch(),
            expected_params={"TraceIds": ["1-aaa", "1-bbb"]},
        )

        with stubber:
            spans = await backend.search_spans(
                SpanQuery(start_time=WINDOW_START, end_time=WINDOW_END)
            )

        assert spans == []
        stubber.assert_no_pending_responses()

    async def test_missing_window_defaults_to_the_last_30_days_ending_now(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        before = datetime.now(UTC)
        slack = timedelta(seconds=60)
        lookback = timedelta(days=30)
        stubber.add_response(
            "get_trace_summaries",
            _summaries(),
            expected_params={
                "StartTime": _Between(before - lookback, before - lookback + slack),
                "EndTime": _Between(before, before + slack),
            },
        )

        with stubber:
            assert await backend.search_spans(SpanQuery()) == []

        stubber.assert_no_pending_responses()

    async def test_end_time_alone_anchors_a_30_day_lookback_to_that_end(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_response(
            "get_trace_summaries",
            _summaries(),
            expected_params={"StartTime": WINDOW_END - timedelta(days=30), "EndTime": WINDOW_END},
        )

        with stubber:
            assert await backend.search_spans(SpanQuery(end_time=WINDOW_END)) == []

        stubber.assert_no_pending_responses()

    async def test_start_time_alone_ends_the_window_now(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        before = datetime.now(UTC)
        stubber.add_response(
            "get_trace_summaries",
            _summaries(),
            expected_params={
                "StartTime": WINDOW_START,
                "EndTime": _Between(before, before + timedelta(seconds=60)),
            },
        )

        with stubber:
            assert await backend.search_spans(SpanQuery(start_time=WINDOW_START)) == []

        stubber.assert_no_pending_responses()

    async def test_summaries_are_over_fetched_to_twice_the_span_limit(self) -> None:
        """limit=1 asks GetTraceSummaries for at least 2 traces: one page of
        1 summary is not enough, so the second page is requested; at 2 the
        paging stops even though a NextToken is still offered (a third call
        has no stubbed response and would fail the test)."""
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_response(
            "get_trace_summaries",
            {**_summaries("1-aaa"), "NextToken": "page-2"},
            expected_params={"StartTime": WINDOW_START, "EndTime": WINDOW_END},
        )
        stubber.add_response(
            "get_trace_summaries",
            {**_summaries("1-bbb"), "NextToken": "page-3"},
            expected_params={
                "StartTime": WINDOW_START,
                "EndTime": WINDOW_END,
                "NextToken": "page-2",
            },
        )
        stubber.add_response(
            "batch_get_traces", _batch(), expected_params={"TraceIds": ["1-aaa", "1-bbb"]}
        )

        with stubber:
            await backend.search_spans(
                SpanQuery(limit=1, start_time=WINDOW_START, end_time=WINDOW_END)
            )

        stubber.assert_no_pending_responses()

    async def test_duplicate_trace_ids_are_hydrated_once_in_discovery_order(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_response("get_trace_summaries", _summaries("1-bbb", "1-aaa", "1-bbb"))
        stubber.add_response(
            "batch_get_traces", _batch(), expected_params={"TraceIds": ["1-bbb", "1-aaa"]}
        )

        with stubber:
            await backend.search_spans(SpanQuery())

        stubber.assert_no_pending_responses()

    async def test_hydration_is_capped_and_batched_in_groups_of_five(self) -> None:
        assert _BATCH_GET_TRACES_MAX_IDS == 5
        assert _MAX_TRACES_TO_HYDRATE == 50
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        ids = [f"1-{i:03d}" for i in range(_MAX_TRACES_TO_HYDRATE + 10)]
        stubber.add_response("get_trace_summaries", _summaries(*ids))
        for start in range(0, _MAX_TRACES_TO_HYDRATE, _BATCH_GET_TRACES_MAX_IDS):
            stubber.add_response(
                "batch_get_traces",
                _batch(),
                expected_params={"TraceIds": ids[start : start + _BATCH_GET_TRACES_MAX_IDS]},
            )

        with stubber:
            await backend.search_spans(SpanQuery(limit=100))

        # Exactly 10 BatchGetTraces calls (ids 0-49); an 11th call for the
        # 10 ids over the cap has no stubbed response and would have raised.
        stubber.assert_no_pending_responses()


class TestFilterTranslation:
    async def test_native_filters_are_pushed_down_and_the_rest_stay_client_side(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_response(
            "get_trace_summaries",
            _summaries("1-aaa"),
            expected_params={
                "StartTime": WINDOW_START,
                "EndTime": WINDOW_END,
                "FilterExpression": (
                    'service("svc-a") AND duration >= 0.25 AND (error = true OR fault = true)'
                ),
            },
        )
        openai: dict[str, Any] = {"metadata": {"gen_ai": {"system": "openai"}}}
        anthropic: dict[str, Any] = {"metadata": {"gen_ai": {"system": "anthropic"}}}
        stubber.add_response(
            "batch_get_traces",
            _batch(
                _trace(
                    "1-aaa",
                    # The one span matching every filter.
                    _segment("keep", "svc-a", error=True, **openai),
                    # Each of these fails exactly one filter, and X-Ray's
                    # trace-level match would still have returned them.
                    _segment("wrong-provider", "svc-a", error=True, **anthropic),
                    _segment("no-provider", "svc-a", error=True),
                    _segment("other-service", "svc-b", error=True, **openai),
                    _segment("not-an-error", "svc-a", error=False, **openai),
                    _segment("too-fast", "svc-a", seconds=0.1, error=True, **openai),
                )
            ),
        )

        with stubber:
            spans = await backend.search_spans(
                SpanQuery(
                    start_time=WINDOW_START,
                    end_time=WINDOW_END,
                    service_name="svc-a",
                    min_duration_ms=250,
                    has_error=True,
                    gen_ai_system="openai",
                )
            )

        assert [s.span_id for s in spans] == ["keep"]
        stubber.assert_no_pending_responses()

    async def test_attribute_filter_is_never_pushed_into_the_filter_expression(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        # No FilterExpression key at all in the expected params.
        stubber.add_response(
            "get_trace_summaries",
            _summaries("1-aaa"),
            expected_params={"StartTime": WINDOW_START, "EndTime": WINDOW_END},
        )
        stubber.add_response(
            "batch_get_traces",
            _batch(
                _trace(
                    "1-aaa",
                    _segment("gpt", "svc", metadata={"gen_ai": {"request": {"model": "gpt-4"}}}),
                    _segment("claude", "svc", metadata={"gen_ai": {"request": {"model": "c-3"}}}),
                )
            ),
        )

        with stubber:
            spans = await backend.search_spans(
                SpanQuery(
                    start_time=WINDOW_START,
                    end_time=WINDOW_END,
                    filters=[_eq_filter("gen_ai.request.model", "gpt-4")],
                )
            )

        assert [s.span_id for s in spans] == ["gpt"]

    async def test_operation_name_is_applied_client_side_only(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_response(
            "get_trace_summaries",
            _summaries("1-aaa"),
            expected_params={"StartTime": WINDOW_START, "EndTime": WINDOW_END},
        )
        stubber.add_response(
            "batch_get_traces",
            _batch(_trace("1-aaa", _segment("a", "checkout"), _segment("b", "payment"))),
        )

        with stubber:
            spans = await backend.search_spans(
                SpanQuery(start_time=WINDOW_START, end_time=WINDOW_END, operation_name="payment")
            )

        assert [s.span_id for s in spans] == ["b"]

    async def test_service_in_filter_becomes_an_or_group(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_response(
            "get_trace_summaries",
            _summaries(),
            expected_params={
                "StartTime": WINDOW_START,
                "EndTime": WINDOW_END,
                "FilterExpression": '(service("svc-a") OR service("svc-b"))',
            },
        )

        with stubber:
            await backend.search_spans(
                SpanQuery(
                    start_time=WINDOW_START,
                    end_time=WINDOW_END,
                    filters=[
                        Filter(
                            field="service.name",
                            operator=FilterOperator.IN,
                            values=["svc-a", "svc-b"],
                            value_type=FilterType.STRING,
                        )
                    ],
                )
            )

        stubber.assert_no_pending_responses()


class TestResultMapping:
    async def test_segments_and_subsegments_become_spans(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_response("get_trace_summaries", _summaries("1-aaa"))
        stubber.add_response(
            "batch_get_traces",
            _batch(
                _trace(
                    "1-aaa",
                    _segment(
                        "root",
                        "checkout",
                        seconds=2.0,
                        error=False,
                        fault=False,
                        throttle=False,
                        subsegments=[
                            {
                                "id": "child",
                                "name": "llm-call",
                                "parent_id": "root",
                                "start_time": _START_EPOCH + 0.5,
                                "end_time": _START_EPOCH + 0.75,
                                "metadata": {"gen_ai": {"system": "openai"}},
                            }
                        ],
                    ),
                )
            ),
        )

        with stubber:
            spans = await backend.search_spans(SpanQuery())

        by_id = {s.span_id: s for s in spans}
        assert set(by_id) == {"root", "child"}

        root = by_id["root"]
        assert root.trace_id == "1-aaa"
        assert root.parent_span_id is None
        assert root.operation_name == "checkout"
        assert root.service_name == "checkout"
        assert root.start_time == datetime.fromtimestamp(_START_EPOCH, tz=UTC)
        assert root.duration_ms == pytest.approx(2000.0)
        assert root.status == "OK"

        child = by_id["child"]
        assert child.trace_id == "1-aaa"
        assert child.parent_span_id == "root"
        assert child.operation_name == "llm-call"
        # A subsegment never supplies its own service: it inherits the
        # top-level segment's name.
        assert child.service_name == "checkout"
        assert child.start_time == datetime.fromtimestamp(_START_EPOCH + 0.5, tz=UTC)
        assert child.duration_ms == pytest.approx(250.0)
        assert child.attributes.get("gen_ai.system") == "openai"

    async def test_spans_from_every_trace_follow_trace_discovery_order(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_response("get_trace_summaries", _summaries("1-bbb", "1-aaa"))
        stubber.add_response(
            "batch_get_traces",
            # BatchGetTraces answers in its own order; discovery order wins.
            _batch(
                _trace("1-aaa", _segment("a", "svc")),
                _trace("1-bbb", _segment("b", "svc")),
            ),
        )

        with stubber:
            spans = await backend.search_spans(SpanQuery())

        assert [(s.trace_id, s.span_id) for s in spans] == [("1-bbb", "b"), ("1-aaa", "a")]

    async def test_limit_truncates_the_flattened_spans(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        ids = ["1-aaa", "1-bbb", "1-ccc"]
        stubber.add_response("get_trace_summaries", _summaries(*ids))
        stubber.add_response(
            "batch_get_traces",
            _batch(*[_trace(t, _segment(f"s-{t}", "svc")) for t in ids]),
        )

        with stubber:
            spans = await backend.search_spans(SpanQuery(limit=2))

        assert [s.span_id for s in spans] == ["s-1-aaa", "s-1-bbb"]

    async def test_limit_applies_after_client_side_filtering(self) -> None:
        """Filtered-out spans must not consume the limit."""
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        ids = ["1-aaa", "1-bbb", "1-ccc"]
        stubber.add_response("get_trace_summaries", _summaries(*ids))
        stubber.add_response(
            "batch_get_traces",
            _batch(
                _trace("1-aaa", _segment("skip", "other")),
                _trace("1-bbb", _segment("hit-1", "wanted")),
                _trace("1-ccc", _segment("hit-2", "wanted")),
            ),
        )

        with stubber:
            spans = await backend.search_spans(SpanQuery(limit=2, operation_name="wanted"))

        assert [s.span_id for s in spans] == ["hit-1", "hit-2"]


class TestMissingAndMalformedData:
    async def test_no_summaries_returns_empty_without_hydrating(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_response("get_trace_summaries", _summaries())

        with stubber:
            assert await backend.search_spans(SpanQuery()) == []

        stubber.assert_no_pending_responses()

    async def test_traces_without_segments_are_skipped_not_fatal(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_response("get_trace_summaries", _summaries("1-aaa", "1-bbb", "1-ccc", "1-ddd"))
        stubber.add_response(
            "batch_get_traces",
            {
                "Traces": [
                    _trace("1-aaa", _segment("kept", "svc")),
                    # Known trace, no segments at all.
                    _trace("1-bbb"),
                    # Known trace whose only document is not valid JSON.
                    {"Id": "1-ccc", "Segments": [{"Id": "bad", "Document": "not json"}]},
                ],
                # 1-ddd is absent from Traces and reported as unprocessed.
                "UnprocessedTraceIds": ["1-ddd"],
            },
        )

        with stubber:
            spans = await backend.search_spans(SpanQuery())

        assert [(s.trace_id, s.span_id) for s in spans] == [("1-aaa", "kept")]

    async def test_segment_missing_required_fields_is_dropped_others_kept(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_response("get_trace_summaries", _summaries("1-aaa"))
        no_end = {
            "Id": "open",
            "Document": json.dumps({"id": "open", "name": "svc", "start_time": 1.0}),
        }
        stubber.add_response(
            "batch_get_traces", _batch(_trace("1-aaa", no_end, _segment("done", "svc")))
        )

        with stubber:
            spans = await backend.search_spans(SpanQuery())

        assert [s.span_id for s in spans] == ["done"]


class TestErrorPropagation:
    """``XRayBackend._call`` documents that botocore's ``ClientError`` is
    left to propagate bare; search_spans adds no wrapping or swallowing."""

    async def test_get_trace_summaries_client_error_propagates_and_stops_the_search(
        self,
    ) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_client_error(
            "get_trace_summaries",
            service_error_code="AccessDeniedException",
            service_message="not allowed",
            http_status_code=403,
        )

        with stubber, pytest.raises(ClientError) as exc_info:
            await backend.search_spans(SpanQuery())

        assert exc_info.value.response["Error"]["Code"] == "AccessDeniedException"
        # Nothing further was attempted: the stubber would have raised on a
        # BatchGetTraces call with no queued response.
        stubber.assert_no_pending_responses()

    async def test_batch_get_traces_client_error_propagates(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_response("get_trace_summaries", _summaries("1-aaa"))
        stubber.add_client_error(
            "batch_get_traces",
            service_error_code="AccessDeniedException",
            service_message="not allowed",
            http_status_code=403,
        )

        with stubber, pytest.raises(ClientError) as exc_info:
            await backend.search_spans(SpanQuery())

        assert exc_info.value.response["Error"]["Code"] == "AccessDeniedException"
        stubber.assert_no_pending_responses()

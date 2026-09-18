"""Tests for the AWS X-Ray backend.

Uses botocore.stub.Stubber (ships inside boto3 itself - zero new test
dependency) rather than moto, since Stubber validates every request this
backend builds against the real, current `xray` botocore service model -
catching a wrong param name/shape at test time - and there is no cassette
engineering needed for SigV4 traffic the way there would be for VCR.
"""

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from botocore.stub import Stubber

from opentelemetry_mcp.backends.xray import (
    _BATCH_GET_TRACES_MAX_IDS,
    _MAX_BATCH_PAGES,
    _MAX_SEARCH_PAGES,
    XRayBackend,
)
from opentelemetry_mcp.models import Filter, FilterOperator, FilterType


def _backend(**overrides: Any) -> XRayBackend:
    defaults: dict[str, Any] = {
        "url": "https://xray.us-east-1.amazonaws.com",
        "aws_region": "us-east-1",
    }
    defaults.update(overrides)
    return XRayBackend(**defaults)


def _segment_document(**overrides: Any) -> str:
    """Build a plausible, minimal X-Ray segment document JSON string."""
    doc: dict[str, Any] = {
        "id": "segment-1",
        "name": "my-service",
        "trace_id": "1-abc",
        "start_time": 1_700_000_000.0,
        "end_time": 1_700_000_001.5,
    }
    doc.update(overrides)
    return json.dumps(doc)


def test_requires_aws_region() -> None:
    with pytest.raises(ValueError, match="AWS region"):
        XRayBackend(url="https://xray.us-east-1.amazonaws.com", aws_region=None)


class TestSegmentDocumentParsing:
    """HANDOFF item 10: validate response shape before indexing - the
    Document field is a JSON-encoded STRING nested inside an already-JSON
    response, so this is the most consequential item for this backend."""

    def test_parses_valid_document(self) -> None:
        backend = _backend()
        result = backend._parse_segment_document(_segment_document())
        assert result is not None
        assert result["id"] == "segment-1"

    def test_rejects_non_json_string(self) -> None:
        backend = _backend()
        assert backend._parse_segment_document("not json{{{") is None

    def test_rejects_document_whose_top_level_is_a_list(self) -> None:
        backend = _backend()
        assert backend._parse_segment_document(json.dumps([1, 2, 3])) is None

    def test_rejects_document_whose_top_level_is_a_scalar(self) -> None:
        backend = _backend()
        assert backend._parse_segment_document(json.dumps("just a string")) is None


class TestFlattenSegmentTree:
    def test_subsegment_inherits_parent_segment_service_name(self) -> None:
        backend = _backend()
        root = {
            "id": "root-1",
            "name": "checkout-service",
            "start_time": 1.0,
            "end_time": 2.0,
            "subsegments": [
                {"id": "sub-1", "name": "DynamoDB.PutItem", "start_time": 1.1, "end_time": 1.5}
            ],
        }
        flattened = backend._flatten_segment_tree([root])
        by_id = {item["id"]: service_name for item, service_name in flattened}

        assert by_id["root-1"] == "checkout-service"
        assert by_id["sub-1"] == "checkout-service"

    def test_nested_subsegments_all_inherit_the_same_root_service_name(self) -> None:
        backend = _backend()
        root = {
            "id": "root-1",
            "name": "checkout-service",
            "start_time": 1.0,
            "end_time": 2.0,
            "subsegments": [
                {
                    "id": "sub-1",
                    "name": "outer-call",
                    "start_time": 1.1,
                    "end_time": 1.9,
                    "subsegments": [
                        {"id": "sub-2", "name": "inner-call", "start_time": 1.2, "end_time": 1.3}
                    ],
                }
            ],
        }
        flattened = backend._flatten_segment_tree([root])
        by_id = {item["id"]: service_name for item, service_name in flattened}

        assert by_id["sub-2"] == "checkout-service"

    def test_skips_root_document_missing_name(self) -> None:
        backend = _backend()
        root = {"id": "root-1", "start_time": 1.0, "end_time": 2.0}
        assert backend._flatten_segment_tree([root]) == []


class TestParseSegmentItem:
    def test_parses_valid_segment(self) -> None:
        backend = _backend()
        item = {
            "id": "seg-1",
            "name": "my-op",
            "start_time": 1_700_000_000.0,
            "end_time": 1_700_000_001.5,
        }
        span = backend._parse_segment_item(item, "my-service", "1-abc")

        assert span is not None
        assert span.span_id == "seg-1"
        assert span.operation_name == "my-op"
        assert span.service_name == "my-service"
        assert span.trace_id == "1-abc"
        assert span.duration_ms == pytest.approx(1500.0)

    def test_rejects_missing_id(self) -> None:
        backend = _backend()
        item = {"name": "my-op", "start_time": 1.0, "end_time": 2.0}
        assert backend._parse_segment_item(item, "svc", "1-abc") is None

    def test_rejects_missing_name(self) -> None:
        backend = _backend()
        item = {"id": "seg-1", "start_time": 1.0, "end_time": 2.0}
        assert backend._parse_segment_item(item, "svc", "1-abc") is None

    def test_rejects_missing_start_time(self) -> None:
        backend = _backend()
        item = {"id": "seg-1", "name": "my-op", "end_time": 2.0}
        assert backend._parse_segment_item(item, "svc", "1-abc") is None

    def test_rejects_in_progress_segment_with_no_end_time(self) -> None:
        backend = _backend()
        item = {"id": "seg-1", "name": "my-op", "start_time": 1.0, "in_progress": True}
        assert backend._parse_segment_item(item, "svc", "1-abc") is None

    def test_rejects_missing_end_time_when_not_in_progress(self) -> None:
        backend = _backend()
        item = {"id": "seg-1", "name": "my-op", "start_time": 1.0}
        assert backend._parse_segment_item(item, "svc", "1-abc") is None

    def test_rejects_negative_duration(self) -> None:
        backend = _backend()
        item = {"id": "seg-1", "name": "my-op", "start_time": 2.0, "end_time": 1.0}
        assert backend._parse_segment_item(item, "svc", "1-abc") is None

    def test_status_unset_when_error_fault_throttle_absent(self) -> None:
        backend = _backend()
        item = {"id": "seg-1", "name": "my-op", "start_time": 1.0, "end_time": 2.0}
        span = backend._parse_segment_item(item, "svc", "1-abc")
        assert span is not None
        assert span.status == "UNSET"

    def test_status_ok_when_error_fault_throttle_all_explicitly_false(self) -> None:
        backend = _backend()
        item = {
            "id": "seg-1",
            "name": "my-op",
            "start_time": 1.0,
            "end_time": 2.0,
            "error": False,
            "fault": False,
            "throttle": False,
        }
        span = backend._parse_segment_item(item, "svc", "1-abc")
        assert span is not None
        assert span.status == "OK"

    def test_status_error_when_error_true(self) -> None:
        backend = _backend()
        item = {"id": "seg-1", "name": "my-op", "start_time": 1.0, "end_time": 2.0, "error": True}
        span = backend._parse_segment_item(item, "svc", "1-abc")
        assert span is not None
        assert span.status == "ERROR"

    def test_status_error_when_fault_true(self) -> None:
        backend = _backend()
        item = {"id": "seg-1", "name": "my-op", "start_time": 1.0, "end_time": 2.0, "fault": True}
        span = backend._parse_segment_item(item, "svc", "1-abc")
        assert span is not None
        assert span.status == "ERROR"

    def test_status_unset_when_only_some_flags_present(self) -> None:
        backend = _backend()
        item = {
            "id": "seg-1",
            "name": "my-op",
            "start_time": 1.0,
            "end_time": 2.0,
            "error": False,
        }
        span = backend._parse_segment_item(item, "svc", "1-abc")
        assert span is not None
        assert span.status == "UNSET"

    def test_metadata_and_annotations_flattened_into_attributes(self) -> None:
        backend = _backend()
        item = {
            "id": "seg-1",
            "name": "my-op",
            "start_time": 1.0,
            "end_time": 2.0,
            "annotations": {"gen_ai": {"system": "openai"}},
            "metadata": {"custom": {"nested": {"value": 42}}},
        }
        span = backend._parse_segment_item(item, "svc", "1-abc")
        assert span is not None
        assert span.attributes.get("gen_ai.system") == "openai"
        assert span.attributes.get("custom.nested.value") == 42

    def test_parent_id_maps_to_parent_span_id(self) -> None:
        backend = _backend()
        item = {
            "id": "seg-1",
            "name": "my-op",
            "start_time": 1.0,
            "end_time": 2.0,
            "parent_id": "seg-0",
        }
        span = backend._parse_segment_item(item, "svc", "1-abc")
        assert span is not None
        assert span.parent_span_id == "seg-0"


class TestFilterExpression:
    def test_service_name_equals_builds_service_predicate(self) -> None:
        backend = _backend()
        f = Filter(
            field="service.name",
            operator=FilterOperator.EQUALS,
            value="foo",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_xray_expression(f) == 'service("foo")'

    def test_service_name_escapes_embedded_quote(self) -> None:
        backend = _backend()
        f = Filter(
            field="service.name",
            operator=FilterOperator.EQUALS,
            value='foo"bar',
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_xray_expression(f) == 'service("foo\\"bar")'

    def test_service_name_in_builds_or_group(self) -> None:
        backend = _backend()
        f = Filter(
            field="service.name",
            operator=FilterOperator.IN,
            values=["foo", "bar"],
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_xray_expression(f) == '(service("foo") OR service("bar"))'

    def test_duration_gt_scales_ms_to_seconds(self) -> None:
        backend = _backend()
        f = Filter(
            field="duration", operator=FilterOperator.GT, value=5000, value_type=FilterType.NUMBER
        )
        assert backend._filter_to_xray_expression(f) == "duration > 5.0"

    def test_duration_rejects_bool_operand(self) -> None:
        backend = _backend()
        f = Filter(
            field="duration", operator=FilterOperator.GT, value=True, value_type=FilterType.NUMBER
        )
        assert backend._filter_to_xray_expression(f) is None

    def test_duration_rejects_non_numeric_operand(self) -> None:
        backend = _backend()
        f = Filter(
            field="duration", operator=FilterOperator.GT, value="fast", value_type=FilterType.STRING
        )
        assert backend._filter_to_xray_expression(f) is None

    @pytest.mark.parametrize(
        ("operator", "expected"),
        [
            (FilterOperator.EQUALS, "duration = 5.0"),
            (FilterOperator.NOT_EQUALS, "duration != 5.0"),
            (FilterOperator.GTE, "duration >= 5.0"),
            (FilterOperator.LT, "duration < 5.0"),
            (FilterOperator.LTE, "duration <= 5.0"),
        ],
    )
    def test_duration_operators(self, operator: FilterOperator, expected: str) -> None:
        backend = _backend()
        f = Filter(field="duration", operator=operator, value=5000, value_type=FilterType.NUMBER)
        assert backend._filter_to_xray_expression(f) == expected

    def test_service_name_not_equals_builds_negated_predicate(self) -> None:
        backend = _backend()
        f = Filter(
            field="service.name",
            operator=FilterOperator.NOT_EQUALS,
            value="foo",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_xray_expression(f) == 'NOT service("foo")'

    def test_status_not_equals_error_maps_to_all_false(self) -> None:
        backend = _backend()
        f = Filter(
            field="status",
            operator=FilterOperator.NOT_EQUALS,
            value="ERROR",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_xray_expression(f) == "(error = false AND fault = false)"

    def test_status_not_equals_ok_maps_to_any_true(self) -> None:
        backend = _backend()
        f = Filter(
            field="status",
            operator=FilterOperator.NOT_EQUALS,
            value="OK",
            value_type=FilterType.STRING,
        )
        assert (
            backend._filter_to_xray_expression(f)
            == "(error = true OR fault = true OR throttle = true)"
        )

    def test_status_equals_unrecognized_value_returns_none(self) -> None:
        backend = _backend()
        f = Filter(
            field="status",
            operator=FilterOperator.EQUALS,
            value="WEIRD",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_xray_expression(f) is None

    def test_unsupported_field_returns_none(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system",
            operator=FilterOperator.EQUALS,
            value="openai",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_xray_expression(f) is None

    def test_service_name_gt_operator_unsupported_returns_none(self) -> None:
        backend = _backend()
        f = Filter(
            field="service.name",
            operator=FilterOperator.GT,
            value="foo",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_xray_expression(f) is None

    def test_status_equals_error_maps_to_error_or_fault(self) -> None:
        backend = _backend()
        f = Filter(
            field="status",
            operator=FilterOperator.EQUALS,
            value="ERROR",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_xray_expression(f) == "(error = true OR fault = true)"

    def test_status_equals_ok_maps_to_all_false(self) -> None:
        backend = _backend()
        f = Filter(
            field="status", operator=FilterOperator.EQUALS, value="OK", value_type=FilterType.STRING
        )
        assert (
            backend._filter_to_xray_expression(f)
            == "(error = false AND fault = false AND throttle = false)"
        )

    def test_gen_ai_field_never_pushed_down_stays_client_side(self) -> None:
        """Regression test for the annotations-vs-metadata finding: OTel
        attributes land in unindexed segment metadata by default, so a
        gen_ai.* filter must never be treated as natively pushable."""
        backend = _backend()
        f = Filter(
            field="gen_ai.system",
            operator=FilterOperator.EQUALS,
            value="openai",
            value_type=FilterType.STRING,
        )
        native, client_side = backend._split_filters([f])
        assert native == []
        assert client_side == [f]

    def test_build_filter_expression_joins_with_and(self) -> None:
        backend = _backend()
        filters = [
            Filter(
                field="service.name",
                operator=FilterOperator.EQUALS,
                value="foo",
                value_type=FilterType.STRING,
            ),
            Filter(
                field="duration",
                operator=FilterOperator.GT,
                value=1000,
                value_type=FilterType.NUMBER,
            ),
        ]
        assert backend._build_filter_expression(filters) == 'service("foo") AND duration > 1.0'

    def test_build_filter_expression_returns_none_for_no_native_filters(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system",
            operator=FilterOperator.EQUALS,
            value="openai",
            value_type=FilterType.STRING,
        )
        assert backend._build_filter_expression([f]) is None


class TestSearchTraces:
    async def test_search_then_hydrate_happy_path(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)

        stubber.add_response(
            "get_trace_summaries",
            {
                "TraceSummaries": [{"Id": "1-aaa"}, {"Id": "1-bbb"}],
                "TracesProcessedCount": 2,
            },
        )
        stubber.add_response(
            "batch_get_traces",
            {
                "Traces": [
                    {
                        "Id": "1-aaa",
                        "Segments": [
                            {"Id": "seg-a", "Document": _segment_document(id="seg-a", name="svc-a")}
                        ],
                    },
                    {
                        "Id": "1-bbb",
                        "Segments": [
                            {"Id": "seg-b", "Document": _segment_document(id="seg-b", name="svc-b")}
                        ],
                    },
                ],
                "UnprocessedTraceIds": [],
            },
        )

        with stubber:
            from opentelemetry_mcp.models import TraceQuery

            traces = await backend.search_traces(TraceQuery())

        trace_ids = {t.trace_id for t in traces}
        assert trace_ids == {"1-aaa", "1-bbb"}

    async def test_batches_trace_ids_in_groups_of_five(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)

        summaries = [{"Id": f"1-{i}"} for i in range(6)]
        stubber.add_response("get_trace_summaries", {"TraceSummaries": summaries})

        assert _BATCH_GET_TRACES_MAX_IDS == 5
        stubber.add_response(
            "batch_get_traces",
            {
                "Traces": [{"Id": f"1-{i}", "Segments": []} for i in range(5)],
                "UnprocessedTraceIds": [],
            },
            expected_params={"TraceIds": [f"1-{i}" for i in range(5)]},
        )
        stubber.add_response(
            "batch_get_traces",
            {"Traces": [{"Id": "1-5", "Segments": []}], "UnprocessedTraceIds": []},
            expected_params={"TraceIds": ["1-5"]},
        )

        with stubber:
            from opentelemetry_mcp.models import TraceQuery

            await backend.search_traces(TraceQuery())

        stubber.assert_no_pending_responses()

    async def test_reports_unprocessed_trace_ids(self, caplog: pytest.LogCaptureFixture) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)

        stubber.add_response("get_trace_summaries", {"TraceSummaries": [{"Id": "1-aaa"}]})
        stubber.add_response(
            "batch_get_traces",
            {"Traces": [], "UnprocessedTraceIds": ["1-aaa"]},
        )

        with stubber:
            from opentelemetry_mcp.models import TraceQuery

            traces = await backend.search_traces(TraceQuery())

        assert traces == []

    async def test_get_trace_summaries_pagination_capped(self) -> None:
        """A pathological account with more than _MAX_SEARCH_PAGES pages of
        results must not loop forever - mirrors DatadogBackend's own
        _MAX_SEARCH_PAGES test."""
        backend = _backend()
        stubber = Stubber(backend._xray_client)

        for page in range(_MAX_SEARCH_PAGES):
            stubber.add_response(
                "get_trace_summaries",
                {"TraceSummaries": [{"Id": f"1-page{page}"}], "NextToken": f"token-{page}"},
            )

        now = datetime.now(UTC)
        with stubber:
            summaries = await backend._get_trace_summaries_raw(None, now, now, 1000)

        assert len(summaries) == _MAX_SEARCH_PAGES
        stubber.assert_no_pending_responses()

    async def test_batch_get_traces_pagination_capped(self) -> None:
        """A single batch's own NextToken pagination must also be capped -
        distinct from the trace-discovery pagination above, since
        BatchGetTraces can itself paginate for one batch of <=5 trace IDs."""
        backend = _backend()
        stubber = Stubber(backend._xray_client)

        for page in range(_MAX_BATCH_PAGES):
            stubber.add_response(
                "batch_get_traces",
                {
                    "Traces": [
                        {"Id": "1-abc", "Segments": [{"Id": f"seg-{page}", "Document": "{}"}]}
                    ],
                    "UnprocessedTraceIds": [],
                    "NextToken": f"token-{page}",
                },
            )

        with stubber:
            segments_by_id, _unprocessed = await backend._batch_get_traces_raw(["1-abc"])

        assert len(segments_by_id["1-abc"]) == _MAX_BATCH_PAGES
        stubber.assert_no_pending_responses()


class TestDefensiveResponseShapeValidation:
    """HANDOFF item 10: validate response shape before indexing - a 200
    response with an unexpected shape must be treated as empty, not crash
    or extend bad entries into the collected results."""

    async def test_get_trace_summaries_raw_handles_non_dict_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _backend()

        async def fake_call(operation: str, **kwargs: Any) -> Any:
            return "not a dict"

        monkeypatch.setattr(backend, "_call", fake_call)
        now = datetime.now(UTC)
        assert await backend._get_trace_summaries_raw(None, now, now, 100) == []

    async def test_get_trace_summaries_raw_handles_non_list_summaries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _backend()

        async def fake_call(operation: str, **kwargs: Any) -> Any:
            return {"TraceSummaries": "not a list"}

        monkeypatch.setattr(backend, "_call", fake_call)
        now = datetime.now(UTC)
        assert await backend._get_trace_summaries_raw(None, now, now, 100) == []

    async def test_batch_get_traces_raw_handles_non_dict_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _backend()

        async def fake_call(operation: str, **kwargs: Any) -> Any:
            return "not a dict"

        monkeypatch.setattr(backend, "_call", fake_call)
        segments_by_id, unprocessed = await backend._batch_get_traces_raw(["1-abc"])
        assert segments_by_id == {}
        assert unprocessed == set()


class TestGetTrace:
    async def test_get_trace_happy_path(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_response(
            "batch_get_traces",
            {
                "Traces": [
                    {
                        "Id": "1-abc",
                        "Segments": [{"Id": "seg-1", "Document": _segment_document()}],
                    }
                ],
                "UnprocessedTraceIds": [],
            },
            expected_params={"TraceIds": ["1-abc"]},
        )

        with stubber:
            trace = await backend.get_trace("1-abc")

        assert trace.trace_id == "1-abc"
        assert len(trace.spans) == 1

    async def test_get_trace_raises_when_not_found(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_response("batch_get_traces", {"Traces": [], "UnprocessedTraceIds": []})

        with stubber, pytest.raises(ValueError, match="No trace found"):
            await backend.get_trace("1-missing")

    async def test_get_trace_raises_when_id_is_unprocessed(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_response("batch_get_traces", {"Traces": [], "UnprocessedTraceIds": ["1-abc"]})

        with stubber, pytest.raises(ValueError, match="unprocessed"):
            await backend.get_trace("1-abc")


class TestListServicesAndOperations:
    async def test_list_services_prefers_service_graph(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_response(
            "get_service_graph",
            {
                "Services": [
                    {"ReferenceId": 0, "Name": "svc-a"},
                    {"ReferenceId": 1, "Name": "svc-b"},
                ]
            },
        )

        with stubber:
            services = await backend.list_services()

        assert services == ["svc-a", "svc-b"]

    async def test_list_services_falls_back_to_sampling_on_access_denied(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_client_error("get_service_graph", service_error_code="AccessDeniedException")
        stubber.add_response(
            "get_trace_summaries",
            {"TraceSummaries": [{"Id": "1-abc"}]},
        )
        stubber.add_response(
            "batch_get_traces",
            {
                "Traces": [
                    {
                        "Id": "1-abc",
                        "Segments": [
                            {
                                "Id": "seg-1",
                                "Document": _segment_document(name="sampled-service"),
                            }
                        ],
                    }
                ],
                "UnprocessedTraceIds": [],
            },
        )

        with stubber:
            services = await backend.list_services()

        assert services == ["sampled-service"]

    async def test_get_service_operations_via_sampling(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_response("get_trace_summaries", {"TraceSummaries": [{"Id": "1-abc"}]})
        stubber.add_response(
            "batch_get_traces",
            {
                "Traces": [
                    {
                        "Id": "1-abc",
                        "Segments": [
                            {
                                "Id": "seg-1",
                                "Document": _segment_document(name="target-service"),
                            }
                        ],
                    }
                ],
                "UnprocessedTraceIds": [],
            },
        )

        with stubber:
            operations = await backend.get_service_operations("target-service")

        assert operations == ["target-service"]


class TestHealthCheck:
    async def test_healthy_when_list_services_succeeds(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_response("get_service_graph", {"Services": [{"ReferenceId": 0, "Name": "svc"}]})

        with stubber:
            result = await backend.health_check()

        assert result.status == "healthy"
        assert result.backend == "xray"

    async def test_unhealthy_wraps_client_error(self) -> None:
        backend = _backend()
        stubber = Stubber(backend._xray_client)
        stubber.add_client_error("get_service_graph", service_error_code="AccessDeniedException")
        stubber.add_client_error("get_trace_summaries", service_error_code="AccessDeniedException")

        with stubber:
            result = await backend.health_check()

        assert result.status == "unhealthy"
        assert result.error is not None


class TestClose:
    async def test_close_calls_boto3_client_close_synchronously(self) -> None:
        backend = _backend()
        backend._xray_client.close = lambda: setattr(backend, "_closed", True)  # type: ignore[method-assign]

        await backend.close()

        assert getattr(backend, "_closed", False) is True


class TestGroupIntoTrace:
    def test_status_error_when_any_span_has_error(self) -> None:
        backend = _backend()
        raw_spans = [
            backend._parse_segment_item(
                {"id": "s1", "name": "op1", "start_time": 1.0, "end_time": 2.0, "error": True},
                "svc",
                "1-abc",
            ),
            backend._parse_segment_item(
                {
                    "id": "s2",
                    "name": "op2",
                    "start_time": 1.0,
                    "end_time": 2.0,
                    "error": False,
                    "fault": False,
                    "throttle": False,
                },
                "svc",
                "1-abc",
            ),
        ]
        spans = [s for s in raw_spans if s is not None]
        trace = backend._group_into_trace("1-abc", spans)
        assert trace.status == "ERROR"

    def test_status_ok_when_all_spans_ok(self) -> None:
        backend = _backend()
        item = {
            "id": "s1",
            "name": "op1",
            "start_time": 1.0,
            "end_time": 2.0,
            "error": False,
            "fault": False,
            "throttle": False,
        }
        span = backend._parse_segment_item(item, "svc", "1-abc")
        assert span is not None
        trace = backend._group_into_trace("1-abc", [span])
        assert trace.status == "OK"

    def test_status_unset_when_mixed_unset_and_ok(self) -> None:
        backend = _backend()
        ok_span = backend._parse_segment_item(
            {
                "id": "s1",
                "name": "op1",
                "start_time": 1.0,
                "end_time": 2.0,
                "error": False,
                "fault": False,
                "throttle": False,
            },
            "svc",
            "1-abc",
        )
        unset_span = backend._parse_segment_item(
            {"id": "s2", "name": "op2", "start_time": 1.0, "end_time": 2.0}, "svc", "1-abc"
        )
        assert ok_span is not None and unset_span is not None
        trace = backend._group_into_trace("1-abc", [ok_span, unset_span])
        assert trace.status == "UNSET"

    def test_root_span_is_the_one_without_a_parent(self) -> None:
        backend = _backend()
        root = backend._parse_segment_item(
            {"id": "root", "name": "root-op", "start_time": 1.0, "end_time": 3.0},
            "root-svc",
            "1-abc",
        )
        child = backend._parse_segment_item(
            {
                "id": "child",
                "name": "child-op",
                "start_time": 1.1,
                "end_time": 1.5,
                "parent_id": "root",
            },
            "root-svc",
            "1-abc",
        )
        assert root is not None and child is not None
        trace = backend._group_into_trace("1-abc", [child, root])
        assert trace.root_operation == "root-op"

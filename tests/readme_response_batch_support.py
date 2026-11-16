"""Loopback fakes for the batch-hydrating backends, shared by the README response-limits tests.

AWS X-Ray hydrates through BatchGetTraces calls and Honeycomb through one batched query. Both fakes
run the REAL backend object and replace only its network leaf (X-Ray's boto3 `_call`, Honeycomb's
`_run_query`), so discovery, the hydration cap, batching and error propagation are the real code.
Nothing here contacts a backend.
"""

import json
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
from botocore.exceptions import ClientError

from opentelemetry_mcp.backends.honeycomb import HoneycombBackend
from opentelemetry_mcp.backends.xray import _BATCH_GET_TRACES_MAX_IDS, XRayBackend
from tests.readme_response_support import _FAKE_CREDENTIAL

XRAY_BATCH_SIZE = _BATCH_GET_TRACES_MAX_IDS


def throttling_error() -> ClientError:
    """The boto3 error AWS returns when BatchGetTraces is throttled."""
    return ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}, "BatchGetTraces"
    )


def http_500() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.honeycomb.io/1/query_results/ds")
    return httpx.HTTPStatusError(
        "Server error '500 Internal Server Error'", request=request, response=httpx.Response(500)
    )


def xray_ids(count: int) -> list[str]:
    return [f"1-0-{index:024x}" for index in range(count)]


class XRayLoopback:
    """The real XRayBackend whose boto3 `_call` leaf is replaced.

    get_trace_summaries lists `ids`. BatchGetTraces call number `fail_on` (1-based) raises `error`;
    every other call returns its traces, or reports every requested ID unprocessed when
    `unprocessed` is set. `batches` records the IDs each BatchGetTraces call was given.
    """

    def __init__(
        self,
        ids: list[str],
        *,
        fail_on: int | None = None,
        error: Exception | None = None,
        unprocessed: bool = False,
    ) -> None:
        self.backend = XRayBackend(
            url="https://xray.us-east-1.amazonaws.com", aws_region="us-east-1"
        )
        self.ids, self.fail_on, self.error, self.unprocessed = ids, fail_on, error, unprocessed
        self.batches: list[list[str]] = []

    async def answer(self, operation: str, **kwargs: Any) -> dict[str, Any]:
        if operation == "get_trace_summaries":
            return {"TraceSummaries": [{"Id": trace_id} for trace_id in self.ids]}
        assert operation == "batch_get_traces", operation
        requested: list[str] = list(kwargs["TraceIds"])
        self.batches.append(requested)
        if self.fail_on == len(self.batches):
            assert self.error is not None
            raise self.error
        if self.unprocessed:
            return {"Traces": [], "UnprocessedTraceIds": requested}
        return {"Traces": [_xray_trace(trace_id) for trace_id in requested]}

    def patched(self) -> AbstractContextManager[Any]:
        return patch.object(self.backend, "_call", AsyncMock(side_effect=self.answer))


def _xray_trace(trace_id: str) -> dict[str, Any]:
    document = {
        "id": "segment-1",
        "name": "svc",
        "trace_id": trace_id,
        "start_time": 1_700_000_000.0,
        "end_time": 1_700_000_001.5,
    }
    return {"Id": trace_id, "Segments": [{"Id": "segment-1", "Document": json.dumps(document)}]}


def honeycomb_ids(count: int) -> list[str]:
    return [f"hc-{index}" for index in range(count)]


class HoneycombLoopback:
    """The real HoneycombBackend whose `_run_query` leaf is replaced.

    The first query (discovery) returns one row per id. The second (the one batched hydration
    query) raises `batch_error` if given, else returns one span per id, or nothing when `empty`.
    """

    def __init__(
        self,
        ids: list[str],
        *,
        batch_error: Exception | None = None,
        empty: bool = False,
    ) -> None:
        self.backend = HoneycombBackend(
            url="https://api.honeycomb.io", api_key=_FAKE_CREDENTIAL, dataset="ds"
        )
        self.ids, self.batch_error, self.empty = ids, batch_error, empty
        self.queries = 0

    async def answer(self, **_: Any) -> list[dict[str, Any]]:
        self.queries += 1
        if self.queries == 1:
            return [{"data": {"trace.trace_id": trace_id}} for trace_id in self.ids]
        if self.batch_error is not None:
            raise self.batch_error
        if self.empty:
            return []
        when = datetime(2024, 1, 1, tzinfo=UTC).isoformat()
        return [
            {
                "data": {
                    "trace.trace_id": trace_id,
                    "trace.span_id": f"{trace_id}-span",
                    "name": "op",
                    "service.name": "svc",
                    "timestamp": when,
                    "duration_ms": 5.0,
                }
            }
            for trace_id in self.ids
        ]

    def patched(self) -> AbstractContextManager[Any]:
        return patch.object(self.backend, "_run_query", AsyncMock(side_effect=self.answer))

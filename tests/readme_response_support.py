"""Shared helpers for the README response-limits tests.

Parses the README section "What responses report and what they do not" and provides the in-process
loopback fakes the tests drive: a recording BaseBackend subclass, and the real TempoBackend
terminating in an httpx.MockTransport. No real backend or warehouse is ever contacted.
"""

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastmcp import Client
from pydantic import HttpUrl

import opentelemetry_mcp.server as server
from opentelemetry_mcp.attributes import HealthCheckResponse
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.backends.tempo import TempoBackend
from opentelemetry_mcp.config import BackendConfig, ServerConfig
from opentelemetry_mcp.models import FilterOperator, SpanData, SpanQuery, TraceData, TraceQuery

_README = Path(__file__).resolve().parent.parent / "README.md"
_HEADING = "### What responses report and what they do not"
_SECTION_END = re.compile(r"^(?:#{1,3} |---$)", re.MULTILINE)
_VALUE, _SAYS_SO, _LOGGED = 1, 2, 3
_BASE_TIME = datetime(2024, 1, 1, tzinfo=UTC)

# Fake credentials, assembled at runtime so no credential-shaped literal sits in the source.
_FAKE_CREDENTIAL = "".join(["fake-", "credential1"])
_FAKE_HEADER_LINE = " ".join(["Authorization:", "Bear" + "er", "fake0123456789abcdefghij"])
_FAKE_PAIR = "=".join(["api" + "_key", "fakefakefake12345"])


# -- README parsing ------------------------------------------------------------------------


def _section() -> str:
    text = _README.read_text(encoding="utf-8")
    assert _HEADING in text, f"README.md has no section headed {_HEADING!r}"
    body = text.split(_HEADING, 1)[1]
    end = _SECTION_END.search(body)
    return body[: end.start()] if end else body


def _row(label: str) -> list[str]:
    """The four cells of the table row whose first cell starts with **<label>.**"""
    prefix = f"| **{label}.**"
    for line in _section().splitlines():
        if line.startswith(prefix):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            assert len(cells) == 4, f"row {label!r} must have 4 cells, has {len(cells)}"
            return cells
    raise AssertionError(f"README section has no table row {label!r}")


def _ints(text: str) -> list[int]:
    return [int(number) for number in re.findall(r"\b\d+\b", text)]


def _hydration_cap() -> int:
    return _ints(_row("Hydration cap")[_VALUE])[0]


def _ceiling() -> int:
    return _ints(_row("Limit above the ceiling")[_VALUE])[0]


# -- loopback fakes ------------------------------------------------------------------------


def _trace(trace_id: str, service: str = "svc-a") -> TraceData:
    span = SpanData(
        trace_id=trace_id,
        span_id="s1",
        parent_span_id=None,
        operation_name="op",
        service_name=service,
        start_time=_BASE_TIME,
        duration_ms=10.0,
        status="OK",
    )
    return TraceData(
        trace_id=trace_id,
        spans=[span],
        start_time=_BASE_TIME,
        duration_ms=10.0,
        service_name=service,
        root_operation="op",
        status="OK",
    )


class _Fake(BaseBackend):
    """In-memory backend: answers min(limit, traces) traces and records the limits it is handed."""

    def __init__(
        self,
        url: str = "http://fake.invalid",
        *,
        traces: int = 0,
        fail_get_trace: bool = False,
        fail_search: bool = False,
    ) -> None:
        super().__init__(url)
        self.traces = traces
        self.fail_get_trace = fail_get_trace
        self.fail_search = fail_search
        self.seen_limits: list[int] = []

    def _create_headers(self) -> dict[str, str]:
        return {}

    def get_supported_operators(self) -> set[FilterOperator]:
        return set()

    async def search_traces(self, query: TraceQuery) -> list[TraceData]:
        self.seen_limits.append(query.limit)
        if self.fail_search:
            raise RuntimeError("secondary search failed")
        return [_trace(f"t{i}") for i in range(min(query.limit, self.traces))]

    async def search_spans(self, query: SpanQuery) -> list[SpanData]:
        return [_trace("t0").spans[0]] if self.traces else []

    async def get_trace(self, trace_id: str) -> TraceData:
        if self.fail_get_trace:
            raise ValueError("not found")
        return _trace(trace_id)

    async def list_services(self) -> list[str]:
        return ["svc-a"]

    async def get_service_operations(self, service_name: str) -> list[str]:
        return ["op"]

    async def health_check(self) -> HealthCheckResponse:
        return HealthCheckResponse(status="healthy", backend="jaeger", url=self.url)


def _config(*, secondary: bool = False) -> ServerConfig:
    return ServerConfig(
        backend=BackendConfig(type="jaeger", url=HttpUrl("http://localhost:16686")),
        secondary_backend=(
            BackendConfig(type="jaeger", url=HttpUrl("http://localhost:16687"))
            if secondary
            else None
        ),
    )


async def _install(primary: BaseBackend, secondary: BaseBackend | None = None) -> None:
    """Wire fakes in as the server's primary (and secondary) backend, ceiling included."""
    config = _config(secondary=secondary is not None)
    server._config = config
    built = [primary] if secondary is None else [primary, secondary]
    with patch.object(server, "_build_backend_from_config", side_effect=built):
        server._backend = server._create_backend(config)
        if secondary is not None:
            await server._get_secondary_backend()


def _tid(index: int) -> str:
    return f"{index:032x}"


def _tempo_trace(
    trace_id: str,
    *,
    llm: bool = False,
    error: bool = False,
    extra: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    attributes: list[dict[str, Any]] = list(extra or [])
    if llm:
        attributes += [
            {"key": "gen_ai.system", "value": {"stringValue": "openai"}},
            {"key": "gen_ai.request.model", "value": {"stringValue": "gpt-4"}},
            {"key": "gen_ai.usage.prompt_tokens", "value": {"intValue": "10"}},
            {"key": "gen_ai.usage.completion_tokens", "value": {"intValue": "5"}},
        ]
    span = {
        "traceId": trace_id,
        "spanId": "span1",
        "name": "handle_request",
        "startTimeUnixNano": "1000000000",
        "endTimeUnixNano": "1050000000",
        "status": {"code": 2 if error else 1},
        "attributes": attributes,
    }
    resource = {"attributes": [{"key": "service.name", "value": {"stringValue": "checkout"}}]}
    return {"batches": [{"resource": resource, "scopeSpans": [{"spans": [span]}]}]}


class _TempoFake:
    """The real TempoBackend, terminating in a MockTransport that serves `candidates` search rows."""

    def __init__(
        self,
        candidates: int = 0,
        *,
        llm: bool = False,
        error: bool = False,
        extra: list[dict[str, Any]] | None = None,
        failing: set[int] | None = None,
        all_fail: bool = False,
    ) -> None:
        self.candidates, self.llm, self.error, self.extra = candidates, llm, error, extra
        self.failing, self.all_fail = failing or set(), all_fail
        self.fetched: list[str] = []
        self.search_params: list[dict[str, str]] = []
        self.backend = TempoBackend(url="http://localhost:3200")
        self.backend._client = httpx.AsyncClient(
            base_url=self.backend.url, transport=httpx.MockTransport(self._handle)
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/search":
            self.search_params.append(dict(request.url.params))
            rows = [{"traceID": _tid(i)} for i in range(self.candidates)]
            return httpx.Response(200, json={"traces": rows})
        trace_id = request.url.path.rsplit("/", 1)[-1]
        self.fetched.append(trace_id)
        if self.all_fail or int(trace_id, 16) in self.failing:
            return httpx.Response(500, json={"error": "boom"})
        payload = _tempo_trace(trace_id, llm=self.llm, error=self.error, extra=self.extra)
        return httpx.Response(200, json=payload)

    def install(self) -> None:
        config = _config()
        server._config = config
        self.backend.max_query_limit = config.max_traces_per_query
        server._backend = self.backend


async def _call(name: str, arguments: dict[str, Any] | None = None) -> Any:
    async with Client(server.mcp) as client:
        return await client.call_tool(name, arguments or {}, raise_on_error=False)


def _body(result: Any) -> dict[str, Any]:
    """The tool's response as a dict, whether it is a typed model or a JSON string."""
    assert not result.is_error, result.content
    structured = result.structured_content
    assert isinstance(structured, dict)
    wrapped = structured.get("result")
    if set(structured) == {"result"} and isinstance(wrapped, str):
        loaded = json.loads(wrapped)
        assert isinstance(loaded, dict)
        return loaded
    return structured


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]

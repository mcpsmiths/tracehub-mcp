"""Pin the time window each backend puts on the wire (Jaeger, Tempo, Traceloop).

``start_time``/``end_time`` decide which traces come back, and each backend
speaks a different unit: Jaeger takes microseconds since the epoch, Tempo
(``/api/search``) and Traceloop (``from_timestamp_sec``/``to_timestamp_sec``)
take whole seconds. A unit mix-up returns the wrong window with every other
test green, so these tests capture the real outgoing request and assert the
exact value for an explicit window, for a half-open window and for no window
(Jaeger sends nothing and lets the server default; Tempo and Traceloop fill
in their documented defaults, see the comments in the backends).

Each case runs through both ``search_traces`` and ``search_spans`` since the
two build the window independently.
"""

import json
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, cast

import pytest

from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.backends.jaeger import JaegerBackend
from opentelemetry_mcp.backends.tempo import TempoBackend
from opentelemetry_mcp.backends.traceloop import TraceloopBackend
from opentelemetry_mcp.models import SpanQuery, TraceQuery

_FAKE_API_KEY = "test" + "_key"

WINDOW_START = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
WINDOW_END = datetime(2025, 1, 1, 13, 30, 0, tzinfo=UTC)
WINDOW_START_EPOCH_SEC = 1_735_732_800
WINDOW_END_EPOCH_SEC = 1_735_738_200

# The same two instants expressed in a non-UTC zone: the wire value is an
# absolute epoch, so the offset must not leak into it.
KOLKATA_TZ = timezone(timedelta(hours=5, minutes=30))
WINDOW_START_IST = datetime(2025, 1, 1, 17, 30, 0, tzinfo=KOLKATA_TZ)
WINDOW_END_IST = datetime(2025, 1, 1, 19, 0, 0, tzinfo=KOLKATA_TZ)

ONE_DAY_SEC = 24 * 60 * 60
ONE_HOUR_SEC = 60 * 60
KINDS = ["traces", "spans"]


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> Any:
        return self._payload


class _CapturingClient:
    """Records the search request and answers it with an empty page."""

    is_closed = False

    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.requests: list[tuple[str, dict[str, Any]]] = []

    async def get(self, url: str, params: dict[str, Any] | None = None) -> _FakeResponse:
        self.requests.append((url, dict(params or {})))
        return _FakeResponse(self._payload)

    async def post(self, url: str, json: dict[str, Any] | None = None) -> _FakeResponse:
        self.requests.append((url, dict(json or {})))
        return _FakeResponse(self._payload)

    @property
    def sent(self) -> dict[str, Any]:
        assert len(self.requests) == 1, self.requests
        return self.requests[0][1]


async def _search(backend: BaseBackend, kind: str, **query: Any) -> None:
    if kind == "traces":
        await backend.search_traces(TraceQuery(service_name="svc", limit=10, **query))
    else:
        await backend.search_spans(SpanQuery(service_name="svc", limit=10, **query))


def _jaeger() -> tuple[JaegerBackend, _CapturingClient]:
    backend = JaegerBackend(url="http://localhost:16686")
    client = _CapturingClient({"data": []})
    backend._client = cast(Any, client)
    return backend, client


def _tempo() -> tuple[TempoBackend, _CapturingClient]:
    backend = TempoBackend(url="http://localhost:3200")
    client = _CapturingClient({"traces": []})
    backend._client = cast(Any, client)
    return backend, client


def _traceloop(kind: str) -> tuple[TraceloopBackend, _CapturingClient]:
    backend = TraceloopBackend(url="https://api.traceloop.com/v2", api_key=_FAKE_API_KEY)
    payload: dict[str, Any] = (
        {"root_spans": {"data": []}} if kind == "traces" else {"spans": {"data": []}}
    )
    client = _CapturingClient(payload)
    backend._client = cast(Any, client)
    return backend, client


@pytest.fixture
def utc_local_timezone(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Tempo and Traceloop derive their default window from the naive local
    ``datetime.now()``. Pin the process zone to UTC so "24 hours ago" is
    24 hours on any host, including across a DST change."""
    with monkeypatch.context() as patch:
        patch.setenv("TZ", "UTC")
        time.tzset()
        yield
    time.tzset()


class TestJaegerTimeWindow:
    """Jaeger: ``start``/``end`` in microseconds since the epoch."""

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize(
        ("start", "end"),
        [(WINDOW_START, WINDOW_END), (WINDOW_START_IST, WINDOW_END_IST)],
        ids=["utc", "offset-aware-same-instants"],
    )
    async def test_explicit_window_is_sent_in_epoch_microseconds(
        self, kind: str, start: datetime, end: datetime
    ) -> None:
        backend, client = _jaeger()

        await _search(backend, kind, start_time=start, end_time=end)

        assert client.requests[0][0] == "/api/traces"
        assert client.sent["start"] == WINDOW_START_EPOCH_SEC * 1_000_000
        assert client.sent["end"] == WINDOW_END_EPOCH_SEC * 1_000_000

    @pytest.mark.parametrize("kind", KINDS)
    async def test_sub_second_precision_is_kept(self, kind: str) -> None:
        backend, client = _jaeger()

        await _search(
            backend,
            kind,
            start_time=datetime(2025, 1, 1, 12, 0, 0, 500_000, tzinfo=UTC),
            end_time=datetime(2025, 1, 1, 12, 0, 1, 250_000, tzinfo=UTC),
        )

        assert client.sent["start"] == 1_735_732_800_500_000
        assert client.sent["end"] == 1_735_732_801_250_000

    @pytest.mark.parametrize("kind", KINDS)
    async def test_missing_window_sends_neither_bound(self, kind: str) -> None:
        backend, client = _jaeger()

        await _search(backend, kind)

        assert "start" not in client.sent
        assert "end" not in client.sent

    @pytest.mark.parametrize("kind", KINDS)
    async def test_start_alone_sends_no_end(self, kind: str) -> None:
        backend, client = _jaeger()

        await _search(backend, kind, start_time=WINDOW_START)

        assert client.sent["start"] == WINDOW_START_EPOCH_SEC * 1_000_000
        assert "end" not in client.sent

    @pytest.mark.parametrize("kind", KINDS)
    async def test_end_alone_sends_no_start(self, kind: str) -> None:
        backend, client = _jaeger()

        await _search(backend, kind, end_time=WINDOW_END)

        assert client.sent["end"] == WINDOW_END_EPOCH_SEC * 1_000_000
        assert "start" not in client.sent

    @pytest.mark.parametrize("kind", KINDS)
    async def test_duration_bounds_are_sent_as_millisecond_strings(self, kind: str) -> None:
        backend, client = _jaeger()

        await _search(backend, kind, min_duration_ms=100, max_duration_ms=750)

        assert client.sent["minDuration"] == "100ms"
        assert client.sent["maxDuration"] == "750ms"

    @pytest.mark.parametrize("kind", KINDS)
    async def test_absent_duration_bounds_are_not_sent(self, kind: str) -> None:
        backend, client = _jaeger()

        await _search(backend, kind)

        assert "minDuration" not in client.sent
        assert "maxDuration" not in client.sent

    @pytest.mark.parametrize("kind", KINDS)
    async def test_response_model_rides_in_the_json_tags_with_the_other_tags(
        self, kind: str
    ) -> None:
        backend, client = _jaeger()

        await _search(
            backend,
            kind,
            gen_ai_system="openai",
            gen_ai_request_model="gpt-4",
            gen_ai_response_model="gpt-4-0613",
            tags={"custom.tag": "value"},
        )

        assert json.loads(client.sent["tags"]) == {
            "custom.tag": "value",
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4",
            "gen_ai.response.model": "gpt-4-0613",
        }

    @pytest.mark.parametrize("kind", KINDS)
    async def test_no_tags_param_without_tags_or_gen_ai_filters(self, kind: str) -> None:
        backend, client = _jaeger()

        await _search(backend, kind)

        assert "tags" not in client.sent


class TestTempoTimeWindow:
    """Tempo ``/api/search``: ``start``/``end`` in whole epoch seconds."""

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize(
        ("start", "end"),
        [(WINDOW_START, WINDOW_END), (WINDOW_START_IST, WINDOW_END_IST)],
        ids=["utc", "offset-aware-same-instants"],
    )
    async def test_explicit_window_is_sent_in_epoch_seconds(
        self, kind: str, start: datetime, end: datetime
    ) -> None:
        backend, client = _tempo()

        await _search(backend, kind, start_time=start, end_time=end)

        assert client.requests[0][0] == "/api/search"
        assert client.sent["start"] == WINDOW_START_EPOCH_SEC
        assert client.sent["end"] == WINDOW_END_EPOCH_SEC

    @pytest.mark.usefixtures("utc_local_timezone")
    @pytest.mark.parametrize("kind", KINDS)
    async def test_missing_window_defaults_to_last_24h_through_one_hour_ahead(
        self, kind: str
    ) -> None:
        backend, client = _tempo()

        before = int(time.time())
        await _search(backend, kind)
        after = int(time.time())

        assert before - ONE_DAY_SEC <= client.sent["start"] <= after - ONE_DAY_SEC
        assert before + ONE_HOUR_SEC <= client.sent["end"] <= after + ONE_HOUR_SEC

    @pytest.mark.usefixtures("utc_local_timezone")
    @pytest.mark.parametrize("kind", KINDS)
    async def test_start_alone_keeps_the_caller_start_and_defaults_the_end(self, kind: str) -> None:
        backend, client = _tempo()

        before = int(time.time())
        await _search(backend, kind, start_time=WINDOW_START)
        after = int(time.time())

        assert client.sent["start"] == WINDOW_START_EPOCH_SEC
        assert before + ONE_HOUR_SEC <= client.sent["end"] <= after + ONE_HOUR_SEC

    @pytest.mark.usefixtures("utc_local_timezone")
    @pytest.mark.parametrize("kind", KINDS)
    async def test_end_alone_keeps_the_caller_end_and_defaults_the_start(self, kind: str) -> None:
        backend, client = _tempo()

        before = int(time.time())
        await _search(backend, kind, end_time=WINDOW_END)
        after = int(time.time())

        assert client.sent["end"] == WINDOW_END_EPOCH_SEC
        assert before - ONE_DAY_SEC <= client.sent["start"] <= after - ONE_DAY_SEC


class TestTraceloopTimeWindow:
    """Traceloop: ``from_timestamp_sec``/``to_timestamp_sec`` in the POST body."""

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize(
        ("start", "end"),
        [(WINDOW_START, WINDOW_END), (WINDOW_START_IST, WINDOW_END_IST)],
        ids=["utc", "offset-aware-same-instants"],
    )
    async def test_explicit_window_is_sent_in_epoch_seconds(
        self, kind: str, start: datetime, end: datetime
    ) -> None:
        backend, client = _traceloop(kind)

        await _search(backend, kind, start_time=start, end_time=end)

        endpoint = "/traces/root-spans" if kind == "traces" else "/spans"
        assert client.requests[0][0].endswith(endpoint)
        assert client.sent["from_timestamp_sec"] == WINDOW_START_EPOCH_SEC
        assert client.sent["to_timestamp_sec"] == WINDOW_END_EPOCH_SEC

    @pytest.mark.usefixtures("utc_local_timezone")
    @pytest.mark.parametrize("kind", KINDS)
    async def test_missing_window_defaults_to_the_last_24h_ending_now(self, kind: str) -> None:
        backend, client = _traceloop(kind)

        before = int(time.time())
        await _search(backend, kind)
        after = int(time.time())

        assert before - ONE_DAY_SEC <= client.sent["from_timestamp_sec"] <= after - ONE_DAY_SEC
        assert before <= client.sent["to_timestamp_sec"] <= after

    @pytest.mark.usefixtures("utc_local_timezone")
    @pytest.mark.parametrize("kind", KINDS)
    async def test_start_alone_keeps_the_caller_start_and_defaults_the_end(self, kind: str) -> None:
        backend, client = _traceloop(kind)

        before = int(time.time())
        await _search(backend, kind, start_time=WINDOW_START)
        after = int(time.time())

        assert client.sent["from_timestamp_sec"] == WINDOW_START_EPOCH_SEC
        assert before <= client.sent["to_timestamp_sec"] <= after

    @pytest.mark.usefixtures("utc_local_timezone")
    @pytest.mark.parametrize("kind", KINDS)
    async def test_end_alone_keeps_the_caller_end_and_defaults_the_start(self, kind: str) -> None:
        backend, client = _traceloop(kind)

        before = int(time.time())
        await _search(backend, kind, end_time=WINDOW_END)
        after = int(time.time())

        assert client.sent["to_timestamp_sec"] == WINDOW_END_EPOCH_SEC
        assert before - ONE_DAY_SEC <= client.sent["from_timestamp_sec"] <= after - ONE_DAY_SEC

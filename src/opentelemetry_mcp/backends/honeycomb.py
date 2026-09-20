"""Honeycomb backend implementation using the Query Data API.

Honeycomb's REST API separates *building* a query specification
(``POST /1/queries/{dataset}``, ungated on every plan) from *running* it:
``POST /1/query_results/{dataset}`` kicks off an async run, and
``GET /1/query_results/{dataset}/{resultId}`` polls for completion. Per
this project's research, the run/poll pair (the only way to actually get
data back) is documented as Enterprise-plan exclusive - see point 1 below.

Schema note: this implementation is grounded in a completed deep-research
pass over docs.honeycomb.io and honeycomb.io/pricing, not a live Honeycomb
account (none was available) - mirroring the same disclosed-assumptions
precedent this project already established for the Sentry and New Relic
backends (see their own module docstrings). This backend carries the most
unverified surface area of the three, flagged here explicitly:

1. **Enterprise-tier gating.** The Query Data API (create-query-result +
   get-query-result) is documented as Enterprise-plan exclusive as of the
   research date, but the *exact* HTTP status code/response body a
   non-Enterprise account gets back was never confirmed against a live
   call. This implementation treats a 402 or 403 from
   create_query_result as a tier-gating signal and raises a clear
   domain-specific error; any other status is left to raise_for_status()
   normally. If the real status code differs, a non-Enterprise account
   would see a generic HTTP error instead of the clearer message - not a
   correctness bug, just a UX gap pending live confirmation.
2. **Raw per-span row extraction.** Honeycomb's Query API is aggregation-
   oriented (calculations grouped by breakdowns), not a raw
   "SELECT * WHERE ..." event fetch the way NRQL or Sentry's Discover API
   are. This implementation extracts individual spans by breaking down on
   a column set that includes ``trace.span_id`` (unique per span) paired
   with a trivial ``COUNT`` calculation, so each resulting group
   corresponds to exactly one span - a documented, commonly-used technique
   for this kind of API shape, but the *exact* set of columns available as
   breakdowns for a given account's ingested data (especially whether
   ``duration_ms`` and a per-event timestamp are breakdown-able at all,
   as opposed to only usable inside a calculation) is unverified. Search
   queries also request ``{"column": "timestamp", "order": "descending"}``
   in the query spec's ``orders`` array so a result set larger than the
   requested limit truncates by recency rather than arbitrarily (the same
   rationale as Datadog's own "-timestamp" search ordering) - the exact
   accepted shape of ``orders`` for a plain breakdown column (as opposed
   to a calculation) is itself unverified.
3. **Result row shape.** ``GET .../query_results/{id}`` is assumed to
   return ``{"data": {"results": [{"data": {<column>: <value>, ...}}]}}``
   per Honeycomb's documented Query Result shape - each row's actual
   values nested one level under a ``"data"`` key.
4. **NRQL-equivalent column names.** ``trace.trace_id``/``trace.span_id``/
   ``trace.parent_id`` are Honeycomb's own documented standard column
   names for distributed tracing (these ARE confirmed, unlike points 1-3
   above) - ``service.name``/``name``/``duration_ms``/``otel.status_code``
   follow the same OTel-ingestion convention already assumed for the New
   Relic backend, not independently reconfirmed here.
5. **Health-check endpoint.** ``health_check()`` uses ``GET /1/auth``
   (Honeycomb's own documented key-metadata endpoint) specifically to
   avoid depending on the Query Data API for a basic health signal - this
   endpoint's existence and behavior is well-documented, but, like every
   other endpoint here, was not exercised against a live account.

Someone with a live Honeycomb Enterprise account and real gen_ai-
instrumented traces should verify all of the above against actual payloads
before relying on this backend in production.
"""

import asyncio
import collections
import logging
import re
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import httpx

from opentelemetry_mcp.attributes import HealthCheckResponse, SpanAttributes
from opentelemetry_mcp.backends.base import BaseBackend, _RetryingTransport
from opentelemetry_mcp.backends.filter_engine import FilterEngine
from opentelemetry_mcp.constants import Fields, Status
from opentelemetry_mcp.models import (
    Filter,
    FilterOperator,
    FilterType,
    SpanData,
    SpanQuery,
    TraceData,
    TraceQuery,
)

logger = logging.getLogger(__name__)

# Documented Create Query Result rate limit - proactively spaced out here
# rather than relied on BaseBackend's generic _RetryingTransport (3 attempts,
# exponential backoff capped at 10s) to recover from after the fact, since
# that generic retry budget is well short of the 60-second window this limit
# actually needs. Research also documented a stricter 1/min tier for queries
# using Honeycomb's "Relational Fields" - not implemented here since this
# backend's own query construction never uses them (see module docstring
# point 2's breakdown-based technique instead), so only the 10/min limiter
# is wired up; a future extension adding relational-field support would need
# its own stricter limiter for that path.
_QUERY_RESULT_RATE_LIMIT_PER_MINUTE = 10

# Honeycomb's own documented hard server-side query execution timeout.
# Polling gives up shortly after this, rather than waiting indefinitely for
# a query that will never complete.
_QUERY_EXECUTION_TIMEOUT_SECONDS = 10.0
_POLL_INTERVAL_SECONDS = 0.5

_MAX_TRACES_TO_HYDRATE = 50

# Sanity upper bound (10 years, in milliseconds) on a parsed span duration -
# same rationale/value as the Sentry/X-Ray/New Relic backends' own bound.
_MAX_REASONABLE_DURATION_MS = 1000 * 60 * 60 * 24 * 365 * 10

# A mapped Honeycomb column name must look like a plain dotted identifier.
# Filter.field (models.py) is an unvalidated `str` reachable from any MCP
# tool call. Honeycomb's filters are native JSON objects (not a string query
# language), so there is no structural-injection risk the way there is for
# Sentry/New Relic's own query builders - this allowlist is defense in depth
# against a malformed/control-character-laden column name reaching the API,
# not an injection guard.
_VALID_HONEYCOMB_FIELD_RE = re.compile(r"^[A-Za-z0-9_.]+$")

# Breakdown columns requested on every search, so results carry enough
# structure to reconstruct SpanData - see module docstring point 2 for why
# trace.span_id (unique per span) is what makes one COUNT-per-group
# correspond to exactly one span, and point 4 for the column-name caveats.
_STRUCTURAL_BREAKDOWNS = (
    "trace.trace_id",
    "trace.span_id",
    "trace.parent_id",
    "name",
    "service.name",
    "duration_ms",
    "timestamp",
    Status.CODE,
)

# A small, well-known set of gen_ai.* semantic-convention columns requested
# alongside the structural ones, so search results carry LLM attributes
# without requiring a caller to know to ask for them - same convention (and
# same unverified-shape caveat) as the Sentry/New Relic backends' own
# equivalent lists.
_GEN_AI_BREAKDOWNS = (
    "gen_ai.system",
    "gen_ai.provider.name",
    "gen_ai.request.model",
    "gen_ai.response.model",
    "gen_ai.usage.prompt_tokens",
    "gen_ai.usage.completion_tokens",
    "gen_ai.usage.total_tokens",
    "gen_ai.conversation.id",
    "gen_ai.prompt.name",
    "gen_ai.prompt.version",
)

_SEARCH_BREAKDOWNS = list(_STRUCTURAL_BREAKDOWNS) + list(_GEN_AI_BREAKDOWNS)

_STRUCTURAL_ROW_FIELDS = frozenset(_STRUCTURAL_BREAKDOWNS)


class HoneycombEnterpriseRequiredError(ValueError):
    """Raised when Honeycomb's Query Data API rejects a request with what
    this backend interprets as tier-gating (see module docstring point 1)."""


class _RateLimiter:
    """Proactively spaces out calls to stay under a per-minute cap, rather
    than reactively retrying after a 429 - see the module-level constant's
    own comment for why BaseBackend's generic retry budget isn't enough
    for Honeycomb's specific rate limit.

    Serialized through a single lock rather than a bare check-then-act on
    the deque: without it, multiple callers blocked on the same "wait for
    one slot" computation would all wake and append at once once their
    sleep elapsed, letting a burst blow straight through the cap this
    class exists to enforce. Holding the lock across the sleep means only
    one caller is ever waiting/appending at a time, so the cap is a true
    ceiling on throughput, not just a per-caller estimate.
    """

    def __init__(self, max_per_minute: int) -> None:
        self._max_per_minute = max_per_minute
        self._call_times: collections.deque[float] = collections.deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until another call is safely under the per-minute cap."""
        async with self._lock:
            while True:
                now = time.monotonic()
                window_start = now - 60.0
                # <=, not <: an entry exactly 60s old must be evicted the
                # moment its own wait_seconds computation below reaches
                # exactly 0 - otherwise neither the eviction condition nor
                # the sleep-if-positive check below ever fires again, and
                # this loop spins forever making no progress.
                while self._call_times and self._call_times[0] <= window_start:
                    self._call_times.popleft()

                if len(self._call_times) < self._max_per_minute:
                    self._call_times.append(now)
                    return

                wait_seconds = 60.0 - (now - self._call_times[0])
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)


class HoneycombBackend(BaseBackend):
    """Honeycomb Query Data API backend."""

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        dataset: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize Honeycomb backend.

        Args:
            url: Honeycomb API base URL (e.g. https://api.honeycomb.io, or
                https://api.eu1.honeycomb.io for the EU instance).
            api_key: A Configuration Key with "Manage Queries and Columns"
                and "Run Queries" permissions (required) - not an Ingest
                or Management key. Sent via the non-standard
                `X-Honeycomb-Team` header (not `Authorization`).
            dataset: Dataset slug to query (required) - the Query Data API
                is dataset-scoped.
            timeout: Request timeout in seconds.
        """
        super().__init__(url, api_key, timeout)

        if not self.url.startswith("https://"):
            raise ValueError(
                "Honeycomb backend requires an https:// URL - the X-Honeycomb-Team "
                "header must not be sent over plain http"
            )
        if not self.api_key:
            raise ValueError("Honeycomb backend requires a Configuration Key (BACKEND_API_KEY)")
        if not dataset:
            raise ValueError(
                "Honeycomb backend requires a dataset (BACKEND_HONEYCOMB_DATASET) "
                "since the Query Data API is dataset-scoped"
            )

        self.dataset = dataset
        self._query_result_rate_limiter = _RateLimiter(_QUERY_RESULT_RATE_LIMIT_PER_MINUTE)

    def _create_headers(self) -> dict[str, str]:
        """Create headers for Honeycomb API requests.

        Returns:
            Dictionary with the X-Honeycomb-Team header.
        """
        return {"X-Honeycomb-Team": self.api_key or "", "Content-Type": "application/json"}

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling.

        Overrides BaseBackend to disable automatic redirect-following, the
        same rationale as the Datadog/New Relic backends' own overrides:
        `X-Honeycomb-Team` is a non-standard header that httpx does not
        strip on cross-origin redirects.

        Returns:
            Reusable AsyncClient instance with automatic connection pooling
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.url,
                headers=self._create_headers(),
                timeout=self.timeout,
                follow_redirects=False,
                transport=_RetryingTransport(
                    slow_request_threshold_ms=self.slow_request_threshold_ms
                ),
            )
        return self._client

    def get_supported_operators(self) -> set[FilterOperator]:
        """Get natively supported operators via Honeycomb's filter ops.

        Returns:
            Set of supported FilterOperator values
        """
        return {
            FilterOperator.EQUALS,
            FilterOperator.NOT_EQUALS,
            FilterOperator.GT,
            FilterOperator.LT,
            FilterOperator.GTE,
            FilterOperator.LTE,
            FilterOperator.EXISTS,
            FilterOperator.NOT_EXISTS,
            FilterOperator.IN,
            FilterOperator.CONTAINS,
        }

    async def search_traces(self, query: TraceQuery) -> list[TraceData]:
        """Search traces by discovering candidate trace IDs then hydrating
        all of them in a single batched query.

        Honeycomb's Query API is span-centric (there is no trace-level
        concept in the API itself), so this mirrors the Sentry/New Relic
        search-then-hydrate pattern: search for matching spans, collect
        their distinct trace.trace_id values, then hydrate every trace's
        full span set with one `trace.trace_id IN (...)` query (not one
        get_trace()-equivalent call per trace - the same batching X-Ray's
        own hydration step already uses, via _batch_get_traces_raw), and
        overlay the original search rows' attributes back onto the
        hydrated result (see _enrich_trace_with_search_rows).

        Batching this way matters specifically because of
        _query_result_rate_limiter: each Create Query Result call is
        proactively throttled to 10/min, so hydrating up to
        _MAX_TRACES_TO_HYDRATE traces one at a time (one call each) could
        stall a single search_traces() call for several minutes; one
        batched call costs exactly one more rate-limited request no
        matter how many traces are being hydrated.

        Args:
            query: Trace query parameters

        Returns:
            List of matching traces with all spans
        """
        all_filters = query.get_all_filters()
        supported_operators = self.get_supported_operators()
        native_filters = [f for f in all_filters if f.operator in supported_operators]

        start, end = self._time_range(query.start_time, query.end_time)
        rows = await self._run_search(native_filters, start, end, query.limit * 5)

        rows_by_trace: dict[str, list[dict[str, Any]]] = {}
        trace_ids: list[str] = []
        for row in rows:
            data = row.get("data")
            raw_trace_id = data.get("trace.trace_id") if isinstance(data, dict) else None
            if not raw_trace_id:
                continue
            trace_id = str(raw_trace_id)
            if trace_id not in rows_by_trace:
                rows_by_trace[trace_id] = []
                trace_ids.append(trace_id)
            rows_by_trace[trace_id].append(row)

        max_to_fetch = min(len(trace_ids), _MAX_TRACES_TO_HYDRATE)
        if len(trace_ids) > max_to_fetch:
            logger.warning(
                f"Limiting trace fetch to {max_to_fetch} out of {len(trace_ids)} "
                f"results to avoid excessive API calls"
            )
        trace_ids_to_fetch = trace_ids[:max_to_fetch]

        traces: list[TraceData] = []
        if trace_ids_to_fetch:
            hydration_rows_by_trace = await self._batch_fetch_trace_rows(trace_ids_to_fetch)
            for trace_id in trace_ids_to_fetch:
                spans = [
                    span
                    for row in hydration_rows_by_trace.get(trace_id, [])
                    if (span := self._parse_honeycomb_row(row)) is not None
                ]
                if not spans:
                    logger.warning(f"Failed to fetch trace {trace_id}: no spans found")
                    continue
                hydrated = self._group_into_trace(trace_id, spans)
                traces.append(
                    self._enrich_trace_with_search_rows(hydrated, rows_by_trace[trace_id])
                )

        if all_filters:
            traces = FilterEngine.apply_filters(traces, all_filters)

        return traces[: query.limit]

    async def _batch_fetch_trace_rows(
        self, trace_ids: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch every span for a batch of trace IDs in one query, grouped
        by trace_id - the batched equivalent of calling get_trace() once
        per ID, used by search_traces to avoid one rate-limited Create
        Query Result call per trace.

        Args:
            trace_ids: Trace IDs to fetch (already capped at
                _MAX_TRACES_TO_HYDRATE by the caller)

        Returns:
            Dict mapping trace_id -> its raw result rows
        """
        start, end = self._time_range(None, None)
        filters = [
            Filter(
                field="trace.trace_id",
                operator=FilterOperator.IN,
                values=trace_ids,
                value_type=FilterType.STRING,
            )
        ]
        rows = await self._run_search(filters, start, end, limit=1000)

        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            data = row.get("data")
            raw_trace_id = data.get("trace.trace_id") if isinstance(data, dict) else None
            if not raw_trace_id:
                continue
            grouped.setdefault(str(raw_trace_id), []).append(row)
        return grouped

    def _enrich_trace_with_search_rows(
        self, trace: TraceData, rows: list[dict[str, Any]]
    ) -> TraceData:
        """Overlay richer per-span attributes/status from the original
        search onto a get_trace()-hydrated trace - see search_traces's
        docstring for why."""
        parsed_by_span_id: dict[str, SpanData] = {}
        for row in rows:
            parsed = self._parse_honeycomb_row(row)
            if parsed is not None:
                parsed_by_span_id[parsed.span_id] = parsed

        if not parsed_by_span_id:
            return trace

        enriched_spans = [
            span.model_copy(
                update={
                    "attributes": parsed_by_span_id[span.span_id].attributes,
                    "status": parsed_by_span_id[span.span_id].status,
                }
            )
            if span.span_id in parsed_by_span_id
            else span
            for span in trace.spans
        ]
        return self._group_into_trace(trace.trace_id, enriched_spans)

    async def search_spans(self, query: SpanQuery) -> list[SpanData]:
        """Search for individual spans matching the query.

        Args:
            query: Span query parameters

        Returns:
            List of matching spans
        """
        all_filters = query.get_all_filters()
        supported_operators = self.get_supported_operators()
        native_filters = [f for f in all_filters if f.operator in supported_operators]
        client_filters = [f for f in all_filters if f.operator not in supported_operators]

        if client_filters:
            logger.info(
                f"Will apply {len(client_filters)} span filters client-side: "
                f"{[(f.field, f.operator.value) for f in client_filters]}"
            )

        start, end = self._time_range(query.start_time, query.end_time)
        rows = await self._run_search(native_filters, start, end, query.limit * 2)

        spans: list[SpanData] = []
        for row in rows:
            span = self._parse_honeycomb_row(row)
            if span:
                spans.append(span)

        if client_filters:
            spans = FilterEngine.apply_filters(spans, client_filters)

        return spans[: query.limit]

    async def get_trace(self, trace_id: str) -> TraceData:
        """Get a specific trace by ID via a trace.trace_id-filtered search.

        No dedicated single-trace lookup endpoint is documented - this
        searches for every span carrying the requested trace.trace_id,
        the same search-then-hydrate approach the X-Ray/Datadog backends
        use, with every result re-verified against the exact ID requested.

        Args:
            trace_id: Trace identifier

        Returns:
            Complete trace data with all spans

        Raises:
            ValueError: If no spans are found for the trace ID
        """
        start, end = self._time_range(None, None)
        filters = [
            Filter(
                field="trace.trace_id",
                operator=FilterOperator.EQUALS,
                value=trace_id,
                value_type=FilterType.STRING,
            )
        ]
        rows = await self._run_search(filters, start, end, limit=1000)

        spans: list[SpanData] = []
        for row in rows:
            span = self._parse_honeycomb_row(row)
            if span and span.trace_id == trace_id:
                spans.append(span)

        if not spans:
            raise ValueError(f"No spans found for trace {trace_id}")

        return self._group_into_trace(trace_id, spans)

    async def list_services(self) -> list[str]:
        """List distinct service names seen in recent span data.

        Returns:
            List of service names
        """
        start, end = self._time_range(None, None)
        rows = await self._run_query(
            breakdowns=["service.name"],
            filters=[],
            start=start,
            end=end,
            limit=200,
        )
        services = {
            str(row["data"]["service.name"])
            for row in rows
            if isinstance(row.get("data"), dict) and row["data"].get("service.name")
        }
        return sorted(services)

    async def get_service_operations(self, service_name: str) -> list[str]:
        """Get distinct span names for one service.

        Args:
            service_name: Service name to scope the query to

        Returns:
            List of operation (span) names
        """
        start, end = self._time_range(None, None)
        rows = await self._run_query(
            breakdowns=["name"],
            filters=[{"column": "service.name", "op": "=", "value": service_name}],
            start=start,
            end=end,
            limit=200,
        )
        operations = {
            str(row["data"]["name"])
            for row in rows
            if isinstance(row.get("data"), dict) and row["data"].get("name")
        }
        return sorted(operations)

    async def health_check(self) -> HealthCheckResponse:
        """Check backend health via Honeycomb's key-metadata endpoint.

        Deliberately does not exercise the (possibly Enterprise-gated)
        Query Data API - GET /1/auth is a lightweight, standard way to
        confirm the Configuration Key authenticates and the API is
        reachable, without depending on the tier-gated path this backend
        is least confident about.

        Returns:
            Health status information
        """
        logger.debug("Checking backend health")

        try:
            response = await self.client.get("/1/auth")
            response.raise_for_status()
            return HealthCheckResponse(status="healthy", backend="honeycomb", url=self.url)
        except Exception as e:
            return HealthCheckResponse(
                status="unhealthy", backend="honeycomb", url=self.url, error=str(e)
            )

    # -- internal helpers ---------------------------------------------------

    async def _run_search(
        self, filters: list[Filter], start: datetime, end: datetime, limit: int
    ) -> list[dict[str, Any]]:
        """Run a raw-span-row search: breakdown on _SEARCH_BREAKDOWNS paired
        with a trivial COUNT, so each result group corresponds to one span
        (trace.span_id is unique per span) - see module docstring point 2.

        Args:
            filters: Filter objects already restricted to supported operators
            start: Start of the search window
            end: End of the search window
            limit: Target number of rows

        Returns:
            List of raw result rows
        """
        honeycomb_filters = [f for f in (self._filter_to_honeycomb_filter(f) for f in filters) if f]
        return await self._run_query(
            breakdowns=list(_SEARCH_BREAKDOWNS),
            filters=honeycomb_filters,
            start=start,
            end=end,
            limit=min(max(limit, 1), 1000),
            # Recency-biased truncation when more spans match than `limit`
            # - "timestamp" is always one of _SEARCH_BREAKDOWNS, same
            # rationale as Datadog's own "-timestamp" search ordering.
            order_by="timestamp",
        )

    async def _run_query(
        self,
        *,
        breakdowns: list[str],
        filters: list[dict[str, Any]],
        start: datetime,
        end: datetime,
        limit: int,
        order_by: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run the full create-query -> create-query-result -> poll flow.

        Args:
            breakdowns: Columns to group by (one result row per unique
                combination of values)
            filters: Already-built Honeycomb filter objects
            start: Start of the query window
            end: End of the query window
            limit: Result row limit
            order_by: Breakdown column to sort descending by, so a result
                set larger than `limit` truncates deterministically rather
                than in an arbitrary order - only meaningful when it's
                also one of `breakdowns`.

        Returns:
            List of raw result rows (each with a "data" sub-dict)
        """
        query_spec: dict[str, Any] = {
            "breakdowns": breakdowns,
            "calculations": [{"op": "COUNT"}],
            "filters": filters,
            "filter_combination": "AND",
            "orders": [{"column": order_by, "order": "descending"}] if order_by else [],
            "start_time": int(start.timestamp()),
            "end_time": int(end.timestamp()),
            "limit": limit,
        }

        query_id = await self._create_query_spec(query_spec)
        result_id = await self._create_query_result(query_id)
        return await self._poll_query_result(result_id)

    async def _post_checking_enterprise_gating(
        self, path: str, json_body: dict[str, Any]
    ) -> httpx.Response:
        """POST to `path`, translating a 402/403 into
        HoneycombEnterpriseRequiredError (module docstring point 1) rather
        than a generic HTTPStatusError. Applied to both query-spec creation
        and query-result creation - the docstring's own disclosure is that
        the exact gating boundary (creation vs. run) was never confirmed
        against a live account, so both are treated as possible gating
        points rather than assuming only one is.

        Args:
            path: Request path
            json_body: JSON request body

        Returns:
            The successful response

        Raises:
            HoneycombEnterpriseRequiredError: On a 402/403 response
        """
        try:
            response = await self.client.post(path, json=json_body)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (402, 403):
                raise HoneycombEnterpriseRequiredError(
                    "Honeycomb backend requires an Enterprise-tier account - the "
                    "Query Data API used by search_traces/search_spans/get_trace "
                    "is not available on this workspace's plan "
                    f"(HTTP {e.response.status_code})"
                ) from e
            raise

    async def _create_query_spec(self, query_spec: dict[str, Any]) -> str:
        """POST /1/queries/{dataset} - builds a reusable query specification.

        Documented as ungated on every plan, but re-checked for Enterprise
        gating anyway (see _post_checking_enterprise_gating's own docstring
        for why - module docstring point 1).

        Args:
            query_spec: Honeycomb query specification body

        Returns:
            The created query's ID

        Raises:
            HoneycombEnterpriseRequiredError: On a 402/403 response
            ValueError: If the response shape is unexpected
        """
        response = await self._post_checking_enterprise_gating(
            f"/1/queries/{self.dataset}", query_spec
        )
        payload = response.json()

        query_id = payload.get("id") if isinstance(payload, dict) else None
        if not isinstance(query_id, str):
            raise ValueError(
                f"Unexpected Honeycomb create-query response shape "
                f"(expected an 'id' string, got {type(payload).__name__}): {payload!r}"
            )
        return query_id

    async def _create_query_result(self, query_id: str) -> str:
        """POST /1/query_results/{dataset} - kicks off an async query run.

        Proactively rate-limited (see _RateLimiter). A 402/403 is
        interpreted as Enterprise-tier gating (module docstring point 1).

        Args:
            query_id: The query specification's ID to run

        Returns:
            The created query result's ID

        Raises:
            HoneycombEnterpriseRequiredError: On a 402/403 response
            ValueError: If the response shape is unexpected
        """
        await self._query_result_rate_limiter.acquire()

        response = await self._post_checking_enterprise_gating(
            f"/1/query_results/{self.dataset}", {"query_id": query_id}
        )
        payload = response.json()
        result_id = payload.get("id") if isinstance(payload, dict) else None
        if not isinstance(result_id, str):
            raise ValueError(
                f"Unexpected Honeycomb create-query-result response shape "
                f"(expected an 'id' string, got {type(payload).__name__}): {payload!r}"
            )
        return result_id

    async def _poll_query_result(self, result_id: str) -> list[dict[str, Any]]:
        """GET /1/query_results/{dataset}/{resultId} until complete or timeout.

        Args:
            result_id: The query result's ID to poll

        Returns:
            List of raw result rows

        Raises:
            ValueError: If the query never completes within the server's
                own documented execution timeout, or the response shape is
                unexpected
        """
        deadline = time.monotonic() + _QUERY_EXECUTION_TIMEOUT_SECONDS + _POLL_INTERVAL_SECONDS
        while True:
            response = await self.client.get(f"/1/query_results/{self.dataset}/{result_id}")
            response.raise_for_status()
            payload = response.json()

            if not isinstance(payload, dict):
                raise ValueError(
                    f"Unexpected Honeycomb query-result response shape "
                    f"(expected an object, got {type(payload).__name__}): {payload!r}"
                )

            if payload.get("complete"):
                data = payload.get("data")
                results = data.get("results") if isinstance(data, dict) else None
                if not isinstance(results, list):
                    raise ValueError(
                        "Unexpected Honeycomb query-result response shape "
                        f"(expected data.results to be a list): {payload!r}"
                    )
                return [r for r in results if isinstance(r, dict)]

            if time.monotonic() >= deadline:
                raise ValueError(
                    f"Honeycomb query {result_id} did not complete within "
                    f"{_QUERY_EXECUTION_TIMEOUT_SECONDS}s"
                )
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    def _time_range(
        self,
        start_time: datetime | None,
        end_time: datetime | None,
        lookback: timedelta = timedelta(days=7),
    ) -> tuple[datetime, datetime]:
        """Resolve a query's time range, defaulting to a lookback window.

        No documented Honeycomb-wide retention window was found (the same
        gap flagged for New Relic) - 7 days is used as the same kind of
        conservative placeholder.

        Args:
            start_time: Explicit start time, if any
            end_time: Explicit end time, if any
            lookback: Default window size when start_time is not given

        Returns:
            (start, end) tuple, both timezone-aware
        """
        end = end_time or datetime.now(UTC)
        start = start_time or (end - lookback)
        return start, end

    def _honeycomb_field(self, field: str) -> str:
        """Map an internal field name to its Honeycomb column name.

        Args:
            field: Internal field name (e.g. "service.name", "gen_ai.system")

        Returns:
            Honeycomb column name
        """
        if field == Fields.SERVICE_NAME:
            return "service.name"
        if field == Fields.OPERATION_NAME:
            return "name"
        if field == Fields.DURATION:
            return "duration_ms"
        if field == Fields.STATUS:
            return Status.CODE
        return field

    def _filter_to_honeycomb_filter(self, filter_obj: Filter) -> dict[str, Any] | None:
        """Convert a single Filter to a Honeycomb filter object.

        Honeycomb filters are native JSON objects, not a string query
        language - values pass through as-is (JSON encoding handles
        escaping), so there is no separate value-escaping step the way
        Sentry/New Relic's own query builders need.

        Args:
            filter_obj: Filter to convert

        Returns:
            Honeycomb filter dict, or None if unsupported/unsafe
        """
        field = self._honeycomb_field(filter_obj.field)
        if not _VALID_HONEYCOMB_FIELD_RE.match(field):
            logger.warning(f"Rejecting filter with unsafe/invalid Honeycomb column name: {field!r}")
            return None

        operator = filter_obj.operator
        value = filter_obj.value
        values = filter_obj.values

        op_map = {
            FilterOperator.EQUALS: "=",
            FilterOperator.NOT_EQUALS: "!=",
            FilterOperator.GT: ">",
            FilterOperator.LT: "<",
            FilterOperator.GTE: ">=",
            FilterOperator.LTE: "<=",
            FilterOperator.CONTAINS: "contains",
        }
        if operator in op_map:
            return {"column": field, "op": op_map[operator], "value": value}

        if operator == FilterOperator.EXISTS:
            return {"column": field, "op": "exists"}
        if operator == FilterOperator.NOT_EXISTS:
            return {"column": field, "op": "does-not-exist"}
        if operator == FilterOperator.IN:
            if not values:
                return None
            return {"column": field, "op": "in", "value": list(values)}

        logger.warning(f"Unsupported operator for Honeycomb query: {operator}")
        return None

    def _infer_status(self, raw_status: Any) -> Literal["OK", "ERROR", "UNSET"]:
        """Map a raw otel.status_code value onto this codebase's OK/ERROR/UNSET model.

        Args:
            raw_status: The raw otel.status_code attribute value

        Returns:
            "OK", "ERROR", or "UNSET" - defaults to UNSET for anything
            unrecognized rather than guessing OK
        """
        if isinstance(raw_status, str):
            status = raw_status.upper()
            if status == "OK":
                return "OK"
            if status == "ERROR":
                return "ERROR"
        return "UNSET"

    def _parse_honeycomb_timestamp(self, value: Any) -> datetime | None:
        """Parse a Honeycomb row timestamp value (epoch seconds).

        Args:
            value: Raw timestamp value from a Honeycomb response row

        Returns:
            Parsed timezone-aware datetime, or None if missing/unparseable
        """
        if value is None or isinstance(value, bool):
            return None
        try:
            if isinstance(value, int | float):
                return datetime.fromtimestamp(float(value), tz=UTC)
            if isinstance(value, str) and value:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, OverflowError, OSError):
            pass
        logger.warning(f"Could not parse Honeycomb timestamp: {value!r}")
        return None

    def _extract_extra_attributes(
        self, source: dict[str, Any], structural_fields: frozenset[str]
    ) -> dict[str, Any]:
        """Extract non-structural, JSON-scalar fields for SpanAttributes.

        Args:
            source: Raw row-data dict
            structural_fields: Field names already consumed into named
                SpanData attributes, to exclude from the passthrough

        Returns:
            Dict of remaining scalar attributes, suitable for
            SpanAttributes(**...)
        """
        return {
            key: value
            for key, value in source.items()
            if key not in structural_fields
            and value is not None
            and isinstance(value, str | int | float | bool)
        }

    def _parse_honeycomb_row(self, row: dict[str, Any]) -> SpanData | None:
        """Parse a raw Query Result row into SpanData.

        Args:
            row: Raw row dict from the query result's "results" array
                (with its own "data" sub-dict, per module docstring point 3)

        Returns:
            Parsed SpanData, or None if required fields are missing/invalid
        """
        try:
            data = row.get("data")
            if not isinstance(data, dict):
                return None

            span_id = data.get("trace.span_id")
            trace_id = data.get("trace.trace_id")
            if not span_id or not trace_id:
                return None

            parent_span_id = data.get("trace.parent_id")
            operation = data.get("name")
            service = data.get("service.name")
            if not operation or not service:
                logger.warning(f"Rejecting span {span_id}: missing operation name or service name")
                return None

            start_time = self._parse_honeycomb_timestamp(data.get("timestamp"))
            if start_time is None:
                logger.warning(f"Rejecting span {span_id}: missing or invalid timestamp")
                return None

            duration_raw = data.get("duration_ms")
            if not isinstance(duration_raw, int | float) or isinstance(duration_raw, bool):
                logger.warning(f"Rejecting span {span_id}: missing or invalid duration_ms")
                return None
            duration_ms = float(duration_raw)
            if duration_ms < 0 or duration_ms > _MAX_REASONABLE_DURATION_MS:
                logger.warning(
                    f"Rejecting span {span_id}: out-of-range duration_ms {duration_ms!r}"
                )
                return None

            status = self._infer_status(data.get(Status.CODE))
            extra = self._extract_extra_attributes(data, _STRUCTURAL_ROW_FIELDS)

            return SpanData(
                trace_id=str(trace_id),
                span_id=str(span_id),
                parent_span_id=str(parent_span_id) if parent_span_id else None,
                operation_name=str(operation),
                service_name=str(service),
                start_time=start_time,
                duration_ms=duration_ms,
                status=status,
                attributes=SpanAttributes(**extra),
                events=[],
            )
        except Exception as e:
            logger.error(f"Error parsing Honeycomb row: {e}")
            return None

    def _group_into_trace(self, trace_id: str, spans: list[SpanData]) -> TraceData:
        """Group a flat list of spans belonging to one trace into TraceData.

        Args:
            trace_id: The trace ID all spans share
            spans: All spans for this trace

        Returns:
            Assembled TraceData
        """
        root_spans = [s for s in spans if not s.parent_span_id]
        root_span = root_spans[0] if root_spans else spans[0]

        start_times = [s.start_time for s in spans]
        end_times = [
            datetime.fromtimestamp(
                s.start_time.timestamp() + (s.duration_ms / 1000), tz=s.start_time.tzinfo
            )
            for s in spans
        ]
        trace_start = min(start_times)
        trace_end = max(end_times)
        trace_duration_ms = (trace_end - trace_start).total_seconds() * 1000

        trace_status: Literal["OK", "ERROR", "UNSET"]
        if any(s.has_error for s in spans):
            trace_status = "ERROR"
        elif all(s.status == "OK" for s in spans):
            trace_status = "OK"
        else:
            trace_status = "UNSET"

        return TraceData(
            trace_id=trace_id,
            spans=spans,
            start_time=trace_start,
            duration_ms=trace_duration_ms,
            service_name=root_span.service_name,
            root_operation=root_span.operation_name,
            status=trace_status,
        )

"""New Relic backend implementation using the NerdGraph GraphQL API.

NerdGraph is New Relic's single GraphQL endpoint for both querying (NRQL)
and metadata/config operations - there is no separate REST surface this
codebase needs. Two distinct query shapes are used, per NerdGraph's own
schema: an NRQL-wrapped, account-scoped search (``actor { account(id) {
nrql(query) { results } } }``) for ``search_traces``/``search_spans``/
``list_services``/``get_service_operations``, and a dedicated,
account-agnostic hydrate path (``actor { distributedTracing { trace(traceId)
{ spans { ... } } } } }``) for ``get_trace`` - mirroring the search-then-
hydrate split Sentry's backend uses, for the same reason (the native lookup
endpoint is the authoritative source for topology).

Schema note: this implementation is grounded in a completed deep-research
pass over docs.newrelic.com (NerdGraph API structure, NRQL syntax, the
Span event's standard attribute set, and the User-vs-Ingest-vs-License API
key taxonomy) rather than a live New Relic account with real
gen_ai-instrumented OTLP traces (none was available). Two things the
research explicitly could not confirm are flagged here so a reviewer with
a live account can verify before relying on this in production:

1. **Exact shape of the ``attributes`` field inside
   ``distributedTracing.trace(traceId).spans``.** This implementation
   assumes it is a flat JSON object keyed by the raw OTel attribute name
   (e.g. ``"gen_ai.system"``), the same convention every other backend's
   ``SpanAttributes(**extra)`` construction already relies on - consistent
   with the field being requested as a GraphQL leaf (no sub-selection),
   which implies a JSON-scalar type in NerdGraph's schema rather than a
   typed object requiring its own field list. An "opaque untyped bag"
   framing was considered during research but could not be confirmed
   either way.
2. **Span/trace data retention window.** No source found a documented
   retention period for ``Span``-event/``distributedTracing`` data. A
   7-day default lookback is used here (``_DEFAULT_LOOKBACK``) as a
   conservative placeholder - an account with a shorter actual retention
   would simply see an empty tail on the extra window (harmless); an
   account with a materially longer retention gets a narrower default
   than it could otherwise use. Not currently exposed as a config knob.

Additionally unverified, in the same spirit as the Sentry backend's own
disclosed assumptions:

3. **NRQL string-literal escaping.** NRQL's documented string syntax uses
   single quotes; this implementation escapes an embedded single quote by
   doubling it (the common SQL convention) and backslashes literal
   backslashes, mirroring this codebase's existing escape-then-quote
   pattern for other backends' native query languages - unverified against
   a live account.
4. **Column names for a flat ``SELECT * FROM Span`` row.** ``trace.id``,
   ``id``, ``parent.id``, ``name``, ``duration.ms``, ``timestamp``,
   ``service.name``, and ``otel.status_code`` are New Relic's own
   documented standard attribute names for OTel-ingested Span events, but
   the exact set actually present on any given account's data depends on
   how spans were ingested (this codebase's other backends make the same
   kind of best-effort column-name assumption for their own native query
   languages).

Someone with a live New Relic account and real gen_ai-instrumented traces
should verify all of the above against actual payloads before relying on
this in production.
"""

import asyncio
import logging
import re
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
    SpanData,
    SpanQuery,
    TraceData,
    TraceQuery,
)

logger = logging.getLogger(__name__)

# Proactive cap on concurrent NerdGraph requests, held below the documented
# 25-concurrent-requests-per-user limit - retry-on-429 alone does not
# prevent breaching a *concurrency* ceiling the way it does a throughput
# one, since a burst of concurrent calls can all be in flight before any
# of them gets a chance to see a 429 and back off.
_MAX_CONCURRENT_REQUESTS = 20

# NRQL's own documented per-query row cap - queries beyond this are
# rejected by New Relic itself, not just a self-imposed limit. Unlike
# Sentry's Link-header cursor pagination, no multi-page pagination is
# implemented here for v1: a query needing more than this many matching
# spans is bounded/truncated rather than paginated across multiple NRQL
# calls.
_MAX_NRQL_LIMIT = 2000

_MAX_TRACES_TO_HYDRATE = 50

# See module docstring point 2 - unverified, conservative placeholder.
_DEFAULT_LOOKBACK = timedelta(days=7)

# Sanity upper bound (10 years, in milliseconds) on a parsed span duration -
# same rationale and value as the Sentry/X-Ray backends' own bound: reject
# an untrustworthy value (clock skew, corrupted data) rather than letting it
# silently corrupt downstream duration-based filtering/aggregation.
_MAX_REASONABLE_DURATION_MS = 1000 * 60 * 60 * 24 * 365 * 10

# A mapped New Relic/NRQL field name must look like a plain dotted
# identifier - Filter.field (models.py) is an unvalidated `str` reachable
# from any MCP tool call, and the field name is spliced directly into the
# WHERE clause in every operator branch of _filter_to_nrql_condition (unlike
# the filter *value*, which always goes through _format_nrql_value's
# escaping quoter) - reject anything that doesn't match this allowlist
# instead of risking injected query structure.
_VALID_NRQL_FIELD_RE = re.compile(r"^[A-Za-z0-9_.]+$")

# Keys on a flat `SELECT * FROM Span` row that this backend treats as
# structural (already extracted into named SpanData fields) rather than
# passed through as custom SpanAttributes - see module docstring point 4.
_NRQL_STRUCTURAL_FIELDS = frozenset(
    {
        "trace.id",
        "id",
        "parent.id",
        "name",
        "duration.ms",
        "timestamp",
        "service.name",
        "entity.name",
        "entity.guid",
        "otel.status_code",
        "appName",
    }
)

# Keys inside a distributedTracing.trace().spans[].attributes object that
# this backend already consumes into named SpanData fields (service_name,
# status) - excluded from the passthrough the same way _NRQL_STRUCTURAL_FIELDS
# excludes their NRQL-row equivalents, so they aren't duplicated into
# SpanAttributes' extras.
_GRAPHQL_ATTRIBUTES_STRUCTURAL_FIELDS = frozenset(
    {"service.name", "entity.name", "otel.status_code"}
)

_GET_TRACE_QUERY = """
query($traceId: ID!) {
  actor {
    distributedTracing {
      trace(traceId: $traceId) {
        spans {
          attributes
          clientType
          durationMs
          entityGuid
          id
          name
          parentId
          processBoundary
          timestamp
          traceId
        }
      }
    }
  }
}
"""

_NRQL_QUERY = """
query($accountId: Int!, $nrql: Nrql!) {
  actor {
    account(id: $accountId) {
      nrql(query: $nrql) {
        results
      }
    }
  }
}
"""

_HEALTH_CHECK_QUERY = "query { actor { user { name } } }"


class NewRelicBackend(BaseBackend):
    """New Relic NerdGraph GraphQL API backend."""

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        account_id: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize New Relic backend.

        Args:
            url: NerdGraph endpoint (e.g. https://api.newrelic.com/graphql,
                or https://api.eu.newrelic.com/graphql for the EU region).
            api_key: A NerdGraph-scoped **User** API key (required) - not
                an Ingest or License key, which cannot query. Sent via the
                non-standard `API-Key` header (not `Authorization`).
            account_id: New Relic account ID (required) - a User API key
                is user-scoped, not account-scoped, so every NRQL query in
                this backend must state which account to query explicitly.
            timeout: Request timeout in seconds.
        """
        super().__init__(url, api_key, timeout)

        if not self.url.startswith("https://"):
            raise ValueError(
                "New Relic backend requires an https:// URL - the API-Key header "
                "must not be sent over plain http"
            )
        if not self.api_key:
            raise ValueError("New Relic backend requires a User API key (BACKEND_API_KEY)")
        if not account_id:
            raise ValueError(
                "New Relic backend requires an account ID (BACKEND_NEWRELIC_ACCOUNT_ID) "
                "since NerdGraph queries are user-scoped not account-scoped - a User API "
                "key alone does not identify which account to query"
            )
        if not account_id.isdigit():
            raise ValueError(
                f"New Relic backend requires a numeric account ID, got {account_id!r} "
                "(BACKEND_NEWRELIC_ACCOUNT_ID)"
            )

        self.account_id = int(account_id)
        self._semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)

    def _create_headers(self) -> dict[str, str]:
        """Create headers for NerdGraph requests.

        Returns:
            Dictionary with the API-Key header NerdGraph requires.
        """
        return {"API-Key": self.api_key or "", "Content-Type": "application/json"}

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling.

        Overrides BaseBackend to disable automatic redirect-following, the
        same rationale as the Datadog backend's own override: `API-Key` is
        a non-standard header that httpx does not strip on cross-origin
        redirects (unlike `Authorization`), so following a redirect to an
        unexpected host would leak the key there.

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
        """Get natively supported operators via NRQL's WHERE clause.

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
        }

    async def search_traces(self, query: TraceQuery) -> list[TraceData]:
        """Search traces by discovering candidate trace IDs then hydrating each.

        NRQL's Span event is span-centric, so this mirrors the Sentry/
        Datadog search-then-hydrate pattern: search for matching spans,
        collect their distinct trace.id values, hydrate each via the
        native get_trace lookup, then re-apply every filter against the
        fully-hydrated trace.

        The initial NRQL search's own rows are kept and overlaid back onto
        the hydrated result (see _enrich_trace_with_search_rows) rather than
        discarded once trace IDs are extracted - get_trace's GraphQL
        hydration path relies on an unverified assumption about the shape
        of distributedTracing.trace's `attributes` field (module docstring
        point 1), and Sentry's own backend hit exactly this kind of gap in
        its own native lookup endpoint on a live account. Overlaying the
        search rows (which came from the same `SELECT * FROM Span` query
        search_spans already trusts) means a wrong shape assumption in
        get_trace degrades gracefully instead of silently dropping
        gen_ai.* attributes across every tool that calls search_traces.

        Args:
            query: Trace query parameters

        Returns:
            List of matching traces with all spans
        """
        all_filters = query.get_all_filters()
        supported_operators = self.get_supported_operators()
        native_filters = [f for f in all_filters if f.operator in supported_operators]

        where_clause = self._build_nrql_where(native_filters)
        start, end = self._time_range(query.start_time, query.end_time)
        rows = await self._run_nrql_search(where_clause, start, end, query.limit * 5)

        rows_by_trace: dict[str, list[dict[str, Any]]] = {}
        trace_ids: list[str] = []
        for row in rows:
            raw_trace_id = row.get("trace.id")
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

        traces: list[TraceData] = []
        for trace_id in trace_ids[:max_to_fetch]:
            try:
                hydrated = await self.get_trace(trace_id)
            except Exception as e:
                logger.warning(f"Failed to fetch trace {trace_id}: {e}")
                continue
            traces.append(self._enrich_trace_with_search_rows(hydrated, rows_by_trace[trace_id]))

        if all_filters:
            traces = FilterEngine.apply_filters(traces, all_filters)

        return traces[: query.limit]

    def _enrich_trace_with_search_rows(
        self, trace: TraceData, rows: list[dict[str, Any]]
    ) -> TraceData:
        """Overlay richer per-span attributes/status from the original NRQL
        search onto a get_trace()-hydrated trace.

        get_trace() stays the source of truth for topology (parent/child
        structure, timing) since it can return spans the initial search
        never matched at all. Only attributes and status are overlaid, and
        only for spans a search row actually matched by span_id - see
        search_traces's docstring for why this is needed.

        Args:
            trace: The trace as parsed by get_trace()
            rows: This trace's raw rows from the original NRQL search

        Returns:
            A new TraceData with enriched spans where a match was found
        """
        parsed_by_span_id: dict[str, SpanData] = {}
        for row in rows:
            parsed = self._parse_newrelic_row(row)
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

        where_clause = self._build_nrql_where(native_filters)
        start, end = self._time_range(query.start_time, query.end_time)
        rows = await self._run_nrql_search(where_clause, start, end, query.limit * 2)

        spans: list[SpanData] = []
        for row in rows:
            span = self._parse_newrelic_row(row)
            if span:
                spans.append(span)

        if client_filters:
            spans = FilterEngine.apply_filters(spans, client_filters)

        return spans[: query.limit]

    async def get_trace(self, trace_id: str) -> TraceData:
        """Get a specific trace by ID via NerdGraph's native distributedTracing lookup.

        Args:
            trace_id: Trace identifier

        Returns:
            Complete trace data with all spans

        Raises:
            ValueError: If no trace/spans are found for the trace ID, or
                the response shape is unexpected
        """
        data = await self._call_nerdgraph(_GET_TRACE_QUERY, {"traceId": trace_id})

        actor = data.get("actor") if isinstance(data, dict) else None
        distributed_tracing = actor.get("distributedTracing") if isinstance(actor, dict) else None
        trace_data = (
            distributed_tracing.get("trace") if isinstance(distributed_tracing, dict) else None
        )
        if not isinstance(trace_data, dict):
            raise ValueError(f"No trace found for {trace_id}")

        raw_spans = trace_data.get("spans")
        if not isinstance(raw_spans, list):
            raise ValueError(
                f"Unexpected New Relic trace response shape for {trace_id} "
                f"(expected spans list, got {type(raw_spans).__name__})"
            )

        spans: list[SpanData] = []
        for raw_span in raw_spans:
            if not isinstance(raw_span, dict):
                continue
            span = self._parse_newrelic_graphql_span(raw_span, trace_id)
            if span and span.trace_id == trace_id:
                spans.append(span)

        if not spans:
            raise ValueError(f"No spans found for trace {trace_id}")

        return self._group_into_trace(trace_id, spans)

    async def list_services(self) -> list[str]:
        """List distinct service names seen in recent Span data.

        Returns:
            List of service names
        """
        start, end = self._time_range(None, None)
        # NRQL, not SQL - no user-controlled value is interpolated here
        # (only this backend's own since_until() output), but ruff's S608
        # heuristic keys off the literal SELECT/FROM keywords regardless.
        nrql = f"SELECT uniques(service.name, 200) AS services FROM Span {self._since_until(start, end)}"  # noqa: S608
        results = await self._run_account_nrql(nrql)
        if not results:
            return []

        row = results[0]
        values = row.get("services") if isinstance(row, dict) else None
        if not isinstance(values, list):
            return []
        return sorted({str(v) for v in values if v})

    async def get_service_operations(self, service_name: str) -> list[str]:
        """Get distinct span names for one service.

        Args:
            service_name: Service name to scope the query to

        Returns:
            List of operation (span) names
        """
        start, end = self._time_range(None, None)
        escaped_service = self._escape_nrql_string(service_name)
        # escaped_service went through _escape_nrql_string's quote/escape
        # convention above, same as every other backend's own native-query
        # builder - see _filter_to_nrql_condition for the equivalent path
        # for MCP-tool-supplied filter values. ruff's S608 heuristic keys
        # off the literal SELECT/FROM/WHERE keywords regardless.
        nrql = f"SELECT uniques(name, 200) AS operations FROM Span WHERE service.name = {escaped_service} {self._since_until(start, end)}"  # noqa: S608
        results = await self._run_account_nrql(nrql)
        if not results:
            return []

        row = results[0]
        values = row.get("operations") if isinstance(row, dict) else None
        if not isinstance(values, list):
            return []
        return sorted({str(v) for v in values if v})

    async def health_check(self) -> HealthCheckResponse:
        """Check backend health via a minimal, data-independent identity query.

        A user-identity query (rather than a data query) is used
        deliberately so a fresh account with zero recent spans still
        reports healthy - health here means "the key authenticates and
        NerdGraph is reachable," not "there is data to query."

        Returns:
            Health status information
        """
        logger.debug("Checking backend health")

        try:
            await self._call_nerdgraph(_HEALTH_CHECK_QUERY, {})
            return HealthCheckResponse(status="healthy", backend="newrelic", url=self.url)
        except Exception as e:
            return HealthCheckResponse(
                status="unhealthy", backend="newrelic", url=self.url, error=str(e)
            )

    # -- internal helpers ---------------------------------------------------

    async def _call_nerdgraph(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """POST a GraphQL query to NerdGraph and return its `data` object.

        Args:
            query: GraphQL query document
            variables: GraphQL variables

        Returns:
            The response's `data` object

        Raises:
            ValueError: If the response carries GraphQL errors, or `data`
                is missing/not an object
            httpx.HTTPError: If the HTTP request itself fails
        """
        async with self._semaphore:
            response = await self.client.post(
                str(self.url), json={"query": query, "variables": variables}
            )
        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected NerdGraph response shape (expected an object, "
                f"got {type(payload).__name__})"
            )

        errors = payload.get("errors")
        if errors:
            raise ValueError(f"NerdGraph query returned errors: {errors}")

        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError(
                f"Unexpected NerdGraph response shape (expected a 'data' object, "
                f"got {type(data).__name__})"
            )
        return data

    async def _run_account_nrql(self, nrql: str) -> list[dict[str, Any]]:
        """Run one account-scoped NRQL query and return its raw result rows.

        Args:
            nrql: Full NRQL query string (already built/escaped by the caller)

        Returns:
            List of raw result row dicts (possibly empty)
        """
        data = await self._call_nerdgraph(_NRQL_QUERY, {"accountId": self.account_id, "nrql": nrql})

        actor = data.get("actor")
        account = actor.get("account") if isinstance(actor, dict) else None
        nrql_result = account.get("nrql") if isinstance(account, dict) else None
        results = nrql_result.get("results") if isinstance(nrql_result, dict) else None

        if not isinstance(results, list):
            logger.warning(
                f"Unexpected NRQL results shape (got {type(results).__name__}); treating as empty"
            )
            return []
        return [r for r in results if isinstance(r, dict)]

    async def _run_nrql_search(
        self, where_clause: str, start: datetime, end: datetime, limit: int
    ) -> list[dict[str, Any]]:
        """Run a `SELECT * FROM Span` search, bounded by NRQL's own row cap.

        Args:
            where_clause: Pre-built, already-escaped WHERE condition (empty
                string means match-all)
            start: Start of the search window
            end: End of the search window
            limit: Target number of rows (clamped to _MAX_NRQL_LIMIT - no
                multi-page pagination in v1, see module docstring)

        Returns:
            List of raw result row dicts
        """
        nrql_limit = min(max(limit, 1), _MAX_NRQL_LIMIT)
        where = f"WHERE {where_clause} " if where_clause else ""
        # where_clause was built by _build_nrql_where/_filter_to_nrql_condition,
        # which reject unsafe field names and escape every value via
        # _format_nrql_value before this point - ruff's S608 heuristic keys
        # off the literal SELECT/FROM/WHERE/LIMIT keywords regardless.
        nrql = f"SELECT * FROM Span {where}{self._since_until(start, end)} LIMIT {nrql_limit}"  # noqa: S608
        return await self._run_account_nrql(nrql)

    def _since_until(self, start: datetime, end: datetime) -> str:
        """Build NRQL's SINCE/UNTIL clause from a resolved time range.

        Args:
            start: Start of the range
            end: End of the range

        Returns:
            "SINCE <start_ms> UNTIL <end_ms>" clause
        """
        return f"SINCE {int(start.timestamp() * 1000)} UNTIL {int(end.timestamp() * 1000)}"

    def _time_range(
        self,
        start_time: datetime | None,
        end_time: datetime | None,
        lookback: timedelta = _DEFAULT_LOOKBACK,
    ) -> tuple[datetime, datetime]:
        """Resolve a query's time range, defaulting to a lookback window.

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

    def _newrelic_field(self, field: str) -> str:
        """Map an internal field name to its NRQL/Span-event column name.

        Args:
            field: Internal field name (e.g. "service.name", "gen_ai.system")

        Returns:
            NRQL column name
        """
        if field == Fields.SERVICE_NAME:
            return "service.name"
        if field == Fields.OPERATION_NAME:
            return "name"
        if field == Fields.DURATION:
            return "duration.ms"
        if field == Fields.STATUS:
            return Status.CODE
        return field

    def _escape_nrql_string(self, value: str) -> str:
        """Escape and quote a value for safe interpolation into an NRQL string literal.

        Always quotes and escapes embedded quotes/backslashes - see module
        docstring point 3.

        Args:
            value: Raw value to interpolate

        Returns:
            A single-quoted, escaped NRQL string literal
        """
        escaped = value.replace("\\", "\\\\").replace("'", "''")
        return f"'{escaped}'"

    def _format_nrql_value(self, value: Any) -> str:
        """Format a filter value for interpolation into an NRQL condition.

        Args:
            value: Filter value to format

        Returns:
            A query-safe NRQL literal token
        """
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int | float):
            return str(value)
        return self._escape_nrql_string(str(value))

    def _filter_to_nrql_condition(self, filter_obj: Filter) -> str | None:
        """Convert a single Filter to an NRQL WHERE condition.

        Args:
            filter_obj: Filter to convert

        Returns:
            NRQL condition string, or None if unsupported/unsafe
        """
        field = self._newrelic_field(filter_obj.field)
        if not _VALID_NRQL_FIELD_RE.match(field):
            logger.warning(f"Rejecting filter with unsafe/invalid NRQL field name: {field!r}")
            return None

        operator = filter_obj.operator
        value = filter_obj.value
        values = filter_obj.values

        if operator == FilterOperator.EQUALS:
            return f"{field} = {self._format_nrql_value(value)}"

        elif operator == FilterOperator.NOT_EQUALS:
            return f"{field} != {self._format_nrql_value(value)}"

        elif operator in (
            FilterOperator.GT,
            FilterOperator.GTE,
            FilterOperator.LT,
            FilterOperator.LTE,
        ):
            if not isinstance(value, int | float) or isinstance(value, bool):
                logger.warning(
                    f"Skipping non-numeric value for {operator.value} on {field!r}: {value!r}"
                )
                return None
            symbol = {
                FilterOperator.GT: ">",
                FilterOperator.GTE: ">=",
                FilterOperator.LT: "<",
                FilterOperator.LTE: "<=",
            }[operator]
            return f"{field} {symbol} {self._format_nrql_value(value)}"

        elif operator == FilterOperator.EXISTS:
            return f"{field} IS NOT NULL"

        elif operator == FilterOperator.NOT_EXISTS:
            return f"{field} IS NULL"

        elif operator == FilterOperator.IN:
            if not values:
                return None
            formatted = ", ".join(self._format_nrql_value(v) for v in values)
            return f"{field} IN ({formatted})"

        logger.warning(f"Unsupported operator for NRQL query: {operator}")
        return None

    def _build_nrql_where(self, filters: list[Filter]) -> str:
        """Build an NRQL WHERE clause body from Filter objects.

        Args:
            filters: List of Filter conditions

        Returns:
            NRQL WHERE condition string (empty string means match-all)
        """
        raw_conditions = [self._filter_to_nrql_condition(f) for f in filters]
        conditions = [c for c in raw_conditions if c is not None]
        return " AND ".join(conditions)

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

    def _parse_newrelic_timestamp(self, value: Any) -> datetime | None:
        """Parse a New Relic timestamp value (epoch milliseconds).

        Args:
            value: Raw timestamp value from a NerdGraph/NRQL response

        Returns:
            Parsed timezone-aware datetime, or None if missing/unparseable
        """
        if value is None or isinstance(value, bool):
            return None
        try:
            if isinstance(value, int | float):
                return datetime.fromtimestamp(float(value) / 1000, tz=UTC)
            if isinstance(value, str) and value:
                return datetime.fromtimestamp(float(value) / 1000, tz=UTC)
        except (ValueError, OverflowError, OSError):
            pass
        logger.warning(f"Could not parse New Relic timestamp: {value!r}")
        return None

    def _extract_extra_attributes(
        self, source: dict[str, Any], structural_fields: frozenset[str]
    ) -> dict[str, Any]:
        """Extract non-structural, JSON-scalar fields for SpanAttributes.

        Args:
            source: Raw row/attributes dict
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

    def _parse_newrelic_row(self, row: dict[str, Any]) -> SpanData | None:
        """Parse a raw `SELECT * FROM Span` row into SpanData.

        Args:
            row: Raw row dict from the NRQL results array

        Returns:
            Parsed SpanData, or None if required fields are missing/invalid
        """
        try:
            span_id = row.get("id")
            trace_id = row.get("trace.id")
            if not span_id or not trace_id:
                return None

            parent_span_id = row.get("parent.id")
            operation = row.get("name")
            service = row.get("service.name") or row.get("entity.name")
            if not operation or not service:
                # Don't fabricate identity: a substituted "unknown" would
                # silently merge structurally-unrelated spans into one fake
                # service/operation bucket downstream.
                logger.warning(f"Rejecting span {span_id}: missing operation name or service name")
                return None

            start_time = self._parse_newrelic_timestamp(row.get("timestamp"))
            if start_time is None:
                logger.warning(f"Rejecting span {span_id}: missing or invalid timestamp")
                return None

            duration_raw = row.get("duration.ms")
            if not isinstance(duration_raw, int | float) or isinstance(duration_raw, bool):
                logger.warning(f"Rejecting span {span_id}: missing or invalid duration.ms")
                return None
            duration_ms = float(duration_raw)
            if duration_ms < 0 or duration_ms > _MAX_REASONABLE_DURATION_MS:
                logger.warning(
                    f"Rejecting span {span_id}: out-of-range duration.ms {duration_ms!r}"
                )
                return None

            status = self._infer_status(row.get(Status.CODE))
            extra = self._extract_extra_attributes(row, _NRQL_STRUCTURAL_FIELDS)

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
            logger.error(f"Error parsing New Relic span row: {e}")
            return None

    def _parse_newrelic_graphql_span(
        self, span: dict[str, Any], requested_trace_id: str
    ) -> SpanData | None:
        """Parse a distributedTracing.trace().spans[] entry into SpanData.

        Args:
            span: Raw span dict from the GraphQL response
            requested_trace_id: The trace_id originally requested, used as
                a fallback if the span's own traceId field is absent

        Returns:
            Parsed SpanData, or None if required fields are missing/invalid
        """
        try:
            span_id = span.get("id")
            if not span_id:
                return None

            attributes = span.get("attributes")
            if not isinstance(attributes, dict):
                # See module docstring point 1 - the shape of this field is
                # unverified; treat anything that isn't a plain object as
                # carrying no usable attributes rather than raising.
                attributes = {}

            operation = span.get("name")
            service = attributes.get("service.name") or attributes.get("entity.name")
            if not operation or not service:
                logger.warning(f"Rejecting span {span_id}: missing operation name or service name")
                return None

            start_time = self._parse_newrelic_timestamp(span.get("timestamp"))
            if start_time is None:
                logger.warning(f"Rejecting span {span_id}: missing or invalid timestamp")
                return None

            duration_raw = span.get("durationMs")
            if not isinstance(duration_raw, int | float) or isinstance(duration_raw, bool):
                logger.warning(f"Rejecting span {span_id}: missing or invalid durationMs")
                return None
            duration_ms = float(duration_raw)
            if duration_ms < 0 or duration_ms > _MAX_REASONABLE_DURATION_MS:
                logger.warning(f"Rejecting span {span_id}: out-of-range durationMs {duration_ms!r}")
                return None

            status = self._infer_status(attributes.get(Status.CODE))
            parent_span_id = span.get("parentId")
            trace_id = span.get("traceId") or requested_trace_id

            extra = self._extract_extra_attributes(
                attributes, _GRAPHQL_ATTRIBUTES_STRUCTURAL_FIELDS
            )

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
            logger.error(f"Error parsing New Relic GraphQL span: {e}")
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

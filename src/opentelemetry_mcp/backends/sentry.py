"""Sentry backend implementation using the Discover/Explore Events API and the
native trace-lookup endpoint.

Unlike Datadog, Sentry does have a trace-level lookup endpoint
(``GET /organizations/{org}/trace/{trace_id}/``), so ``get_trace`` calls it
directly instead of reconstructing a trace from a span search. However,
Sentry's *search* surface (``search_traces``/``search_spans``) still goes
through the same span-centric Discover/Explore "events" endpoint
(``GET /organizations/{org}/events/`` with ``dataset=spans``), so
``search_traces`` mirrors the Datadog/Tempo search-then-hydrate pattern:
discover candidate trace IDs via a span search, then hydrate each one via
``get_trace``.

Schema note: this implementation is grounded in docs.sentry.io (Auth,
Pagination, Rate Limits, the Explore/table-format Events API, and the Span
Properties reference) plus the ``getsentry/sentry`` endpoint source read at
research time, rather than a live Sentry account (none was available). A
number of things are not fully pinned down by the public docs and are
flagged here so a reviewer with a live account (with real gen_ai-instrumented
OTLP traces) can verify before relying on this in production:

1. **Auth is standard.** ``Authorization: Bearer <token>`` is Sentry's
   documented, recommended scheme. Unlike Datadog's non-standard
   ``DD-API-KEY``/``DD-APPLICATION-KEY`` headers, httpx already strips a
   standard ``Authorization`` header on a cross-origin redirect, so this
   backend does *not* override ``client`` to disable redirect-following -
   doing so would just be cargo-culting Datadog's mitigation for a threat
   model that doesn't apply here.
2. **``get_trace`` endpoint choice.** This uses the modern
   ``GET /organizations/{org}/trace/{trace_id}/`` endpoint, whose own
   docstring in Sentry's source says it "replaces"
   ``OrganizationEventsTraceEndpoint`` (the legacy ``/events-trace/{id}/``
   endpoint) - so the legacy endpoint is deliberately not used here. This
   endpoint always queries across every project in the org (a trace can span
   several), and - like the related ``/traces/`` list endpoint - may be
   feature-gated/experimental on some accounts (e.g. older self-hosted
   installs), in which case it would 404. There is no fallback to the legacy
   endpoint; this is a known limitation, not a silent failure (the 404
   propagates as an ``httpx.HTTPStatusError``).
3. **Raw span column names.** Sentry's public "Span Properties" reference
   confirms ``span.op``, ``span.duration``, ``span.status``, and
   ``transaction`` as documented, queryable span properties, but does not
   document the exact ``field=`` column names for a span's own ID, its trace
   ID, or its parent span ID in the table-format Events API response. This
   implementation uses ``id``, ``trace``, and ``parent_span`` - the column
   names Sentry's own Explore UI exposes for these - as a best-effort,
   unverified mapping.
4. **``SerializedTraceItem`` shape.** The exact JSON shape returned by
   ``/trace/{trace_id}/`` is not published. This implementation looks up
   several plausible candidate key names for each field (e.g.
   ``start_timestamp`` or ``precise.start_ts``) and rejects (skips) any item
   missing required identifying/timing data rather than guessing at a
   fabricated value.
5. **Query-value escaping.** Sentry's documented search syntax
   (docs.sentry.io/concepts/search/) shows quoting multi-word values in
   double quotes but does not document an escape sequence for a literal
   quote/backslash *inside* a quoted term. This implementation uses the same
   backslash-escaping convention as the Datadog backend
   (``_escape_sentry_query_value``), unverified against a live account.
6. **No "service" concept.** Sentry has no separate services API; a
   "service" in this codebase's sense maps onto a Sentry *project* slug.
   ``get_service_operations`` tries the (feature-gated) trace-item
   attribute-values endpoint first and falls back to sampling recent spans
   on any failure (missing feature, 404, or unexpected shape).

Someone with a live Sentry account and real gen_ai-instrumented traces should
verify all of the above against actual payloads before relying on this in
production.
"""

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from urllib.parse import quote

from opentelemetry_mcp.attributes import HealthCheckResponse, SpanAttributes
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.backends.filter_engine import FilterEngine
from opentelemetry_mcp.constants import Fields
from opentelemetry_mcp.models import (
    Filter,
    FilterOperator,
    SpanData,
    SpanQuery,
    TraceData,
    TraceQuery,
)

logger = logging.getLogger(__name__)

# Sentry search syntax fields that are documented, queryable span/trace
# properties rather than custom attributes - queried bare, matching Sentry's
# own search syntax (no `@`-style prefix is used, unlike Datadog's custom
# attribute convention).
_FACET_FIELDS = {
    Fields.SERVICE_NAME: "project",
    Fields.OPERATION_NAME: "span.op",
}

# Default lookback window used when a query has no explicit time range.
# Sentry's published retention varies by plan (sentry.io/pricing: 30 days on
# Developer/free, "up to 90 days" on Team/Business, custom on Enterprise) -
# unlike Datadog, where 30 days matched one known account-wide ceiling, there
# is no single ceiling here. 90 days is chosen as a safe default: an account
# on the 30-day tier simply gets an empty tail on the extra window (harmless),
# while a Team/Business account gets its full documented range. Enterprise
# accounts may have a materially longer window; this is not currently
# exposed as a config knob.
_DEFAULT_LOOKBACK = timedelta(days=90)

_MAX_TRACES_TO_HYDRATE = 50

# Safety bound on how many pages _search_events_raw will follow via Sentry's
# Link-header cursor pagination for a single logical search, so a
# pathological query can't loop forever.
_MAX_SEARCH_PAGES = 10

# Sanity upper bound (10 years, in milliseconds) on a parsed span duration.
# A raw duration/timestamp pair from Sentry is never validated against this
# codebase's assumption of non-negative, human-scale durations - a clock-skew
# end-before-start pair or a corrupted/garbage `duration` field could
# otherwise produce a negative or absurdly large `duration_ms` that later
# overflows the `datetime` arithmetic in `_group_into_trace`. Reject the span
# instead of propagating an untrustworthy value.
_MAX_REASONABLE_DURATION_MS = 1000 * 60 * 60 * 24 * 365 * 10

# Columns requested from the table-format Events API for a raw span search.
# See module docstring point 3 for the unverified column-name assumptions.
_SPAN_STRUCTURAL_FIELDS = (
    "id",
    "trace",
    "parent_span",
    "span.op",
    "transaction",
    "project",
    "timestamp",
    "span.duration",
    "span.status",
)

# A small, well-known set of gen_ai.* semantic-convention columns requested
# alongside the structural fields above, so search results carry LLM
# attributes without requiring a caller to know to ask for them. Sentry's
# EAP spans dataset is assumed to expose OTel span attributes as bare,
# directly-queryable/selectable column names (the same convention confirmed
# for `span.op`/`span.duration`/`span.status`), which is itself an
# unverified assumption - see module docstring point 3.
_GEN_AI_FIELDS = (
    "gen_ai.system",
    "gen_ai.request.model",
    "gen_ai.response.model",
    "gen_ai.usage.prompt_tokens",
    "gen_ai.usage.completion_tokens",
    "gen_ai.usage.total_tokens",
    # Needed for list_sessions/get_session_stats (grouped by conversation.id)
    # and get_prompt_version_stats (grouped by prompt.name+version) to see
    # any results at all against this backend - without these, both tools
    # silently returned empty regardless of what data Sentry actually held.
    "gen_ai.conversation.id",
    "gen_ai.prompt.name",
    "gen_ai.prompt.version",
)

_SPAN_SEARCH_FIELDS = list(_SPAN_STRUCTURAL_FIELDS) + list(_GEN_AI_FIELDS)

# A mapped Sentry search field name must look like a plain dotted identifier.
# Filter.field (models.py) is an unvalidated `str` reachable from any MCP
# tool call, and unlike the filter *value* (escaped via
# `_escape_sentry_query_value`), the field name is spliced directly into the
# query string in every operator branch of `_filter_to_sentry_query` - so a
# field like `x) OR (a:b` would inject arbitrary structure into the query.
# Reject anything that doesn't match this allowlist pattern instead.
_VALID_SENTRY_FIELD_RE = re.compile(r"^[A-Za-z0-9_.]+$")

# Keys on a /trace/{id}/ SerializedTraceItem that this backend treats as
# structural (already extracted into named SpanData fields) rather than
# passed through as custom SpanAttributes - see module docstring point 4.
_TRACE_ITEM_STRUCTURAL_FIELDS = frozenset(
    {
        "trace_id",
        "trace",
        "span_id",
        "id",
        "event_id",
        "parent_span_id",
        "parentSpanId",
        "op",
        "transaction",
        "description",
        "project_slug",
        "project",
        "start_timestamp",
        "precise.start_ts",
        "timestamp",
        "end_timestamp",
        "precise.finish_ts",
        "duration",
        "span.duration",
        "status",
        "span.status",
        "errors",
        "occurrences",
        "children",
    }
)


class SentryBackend(BaseBackend):
    """Sentry Discover/Explore Events API + native trace-lookup backend."""

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        org_slug: str | None = None,
        project_slug: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize Sentry backend.

        Args:
            url: Sentry base URL (e.g. https://sentry.io for SaaS, or a
                self-hosted install's base URL). Sentry SaaS also has
                region-specific URLs (e.g. https://us.sentry.io).
            api_key: Sentry auth token with trace/event read permissions
                (required) - sent as a standard Bearer token.
            org_slug: Sentry organization slug (required) - every endpoint
                this backend calls is organization-scoped.
            project_slug: Optional Sentry project slug used to narrow
                queries. When omitted, queries span every project the token
                can access.
            timeout: Request timeout in seconds.
        """
        super().__init__(url, api_key, timeout)

        if not self.url.startswith("https://"):
            raise ValueError(
                "Sentry backend requires an https:// URL - the auth token "
                "must not be sent over plain http"
            )
        if not self.api_key:
            raise ValueError("Sentry backend requires an auth token (BACKEND_API_KEY)")
        if not org_slug:
            raise ValueError(
                "Sentry backend requires an organization slug (BACKEND_SENTRY_ORG) "
                "since every Sentry API endpoint used here is organization-scoped"
            )

        self.org_slug = org_slug
        self.project_slug = project_slug

    def _create_headers(self) -> dict[str, str]:
        """Create headers for Sentry API requests.

        Returns:
            Dictionary with a standard Bearer Authorization header.
        """
        return {"Authorization": f"Bearer {self.api_key or ''}"}

    def get_supported_operators(self) -> set[FilterOperator]:
        """Get natively supported operators via Sentry's search syntax.

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

        Sentry's search surface is span-centric (the Events API), so this
        mirrors the Datadog/Tempo pattern: search spans for distinct
        trace IDs, hydrate each via the native ``get_trace`` lookup, then
        re-apply every filter against the fully-hydrated trace (a
        trace-level filter must hold for the whole reconstructed trace, not
        just the one span that matched the initial search).

        A live account confirmed ``get_trace()``'s own endpoint carries no
        gen_ai.*/OTel custom attributes and no reliable status signal for
        OTLP-ingested spans, unlike this search. Requesting the full
        ``_SPAN_SEARCH_FIELDS`` here (not just ``trace``) and overlaying
        those richer per-span rows onto the hydrated result closes that gap
        for every span the search actually matched - see
        ``_enrich_trace_with_search_rows``.

        Args:
            query: Trace query parameters

        Returns:
            List of matching traces with all spans

        Raises:
            httpx.HTTPError: If the API request fails
        """
        all_filters = query.get_all_filters()
        supported_operators = self.get_supported_operators()
        native_filters = [f for f in all_filters if f.operator in supported_operators]

        sentry_query = self._build_sentry_query(native_filters)
        start, end = self._time_range(query.start_time, query.end_time)

        rows = await self._search_events_raw(
            sentry_query, list(_SPAN_SEARCH_FIELDS), start, end, query.limit * 5
        )

        rows_by_trace: dict[str, list[dict[str, Any]]] = {}
        trace_ids: list[str] = []
        for row in rows:
            trace_id = row.get("trace")
            if not trace_id:
                continue
            trace_id = str(trace_id)
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
        """Overlay richer per-span attributes/status from the original span
        search onto a ``get_trace()``-hydrated trace.

        ``get_trace()`` stays the source of truth for topology (parent/child
        structure, timing) since it can return spans the initial search
        never matched at all. Only attributes and status are overlaid, and
        only for spans a search row actually matched by span_id - see
        ``search_traces``'s docstring for why this is needed.

        Args:
            trace: The trace as parsed by get_trace()
            rows: This trace's raw rows from the original span search

        Returns:
            A new TraceData with enriched spans where a match was found
        """
        parsed_by_span_id: dict[str, SpanData] = {}
        for row in rows:
            parsed = self._parse_sentry_row(row)
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

        Raises:
            httpx.HTTPError: If the API request fails
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

        sentry_query = self._build_sentry_query(native_filters)
        start, end = self._time_range(query.start_time, query.end_time)

        rows = await self._search_events_raw(
            sentry_query, _SPAN_SEARCH_FIELDS, start, end, query.limit * 2
        )

        spans: list[SpanData] = []
        for row in rows:
            span = self._parse_sentry_row(row)
            if span:
                spans.append(span)

        if client_filters:
            spans = FilterEngine.apply_filters(spans, client_filters)

        return spans[: query.limit]

    async def get_trace(self, trace_id: str) -> TraceData:
        """Get a specific trace by ID via Sentry's native trace-lookup endpoint.

        Args:
            trace_id: Trace identifier

        Returns:
            Complete trace data with all spans

        Raises:
            ValueError: If no spans are found for the trace ID, or the
                response shape is unexpected
            httpx.HTTPError: If the API request fails
        """
        start, end = self._time_range(None, None, lookback=_DEFAULT_LOOKBACK)
        # trace_id can originate from an external MCP tool call. It's a URL
        # *path* segment here (not a query-string token), so it needs path
        # escaping rather than the query-value escaper used elsewhere.
        encoded_trace_id = quote(trace_id, safe="")

        response = await self.client.get(
            f"/api/0/organizations/{self.org_slug}/trace/{encoded_trace_id}/",
            params={
                # A trace can span multiple projects, and this endpoint
                # always queries across all of them regardless of project
                # filtering - pass -1 explicitly to make that intent clear.
                "project": "-1",
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        )
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, list):
            raise ValueError(
                f"Unexpected Sentry trace response shape for {trace_id} "
                f"(expected a list, got {type(data).__name__})"
            )

        flat_items = self._flatten_trace_items(data)

        spans: list[SpanData] = []
        for item in flat_items:
            span = self._parse_sentry_trace_item(item, trace_id)
            # Belt-and-suspenders: only keep spans that exactly match the
            # requested trace_id, rather than trusting the endpoint's own
            # scoping unconditionally.
            if span and span.trace_id == trace_id:
                spans.append(span)

        if not spans:
            raise ValueError(f"No spans found for trace {trace_id}")

        return self._group_into_trace(trace_id, spans)

    async def list_services(self) -> list[str]:
        """List all available projects (Sentry's analog of "services").

        Returns:
            List of project slugs

        Raises:
            httpx.HTTPError: If the API request fails
        """
        logger.debug("Listing services (Sentry projects)")

        response = await self.client.get(f"/api/0/organizations/{self.org_slug}/projects/")
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, list):
            logger.warning(
                f"Sentry projects response was not a list (got {type(data).__name__}); "
                "treating as empty"
            )
            return []

        slugs = {
            item["slug"]
            for item in data
            if isinstance(item, dict) and isinstance(item.get("slug"), str) and item["slug"]
        }
        return sorted(slugs)

    async def get_service_operations(self, service_name: str) -> list[str]:
        """Get operations (span operation names) for a project.

        Tries the trace-item attribute-values endpoint first (feature-gated
        on some accounts), falling back to sampling recent spans on any
        failure - see module docstring point 6.

        Args:
            service_name: Project slug

        Returns:
            List of operation names

        Raises:
            httpx.HTTPError: If both the primary and fallback requests fail
        """
        logger.debug(f"Getting operations for service: {service_name}")

        operations = await self._get_operations_via_attribute_values(service_name)
        if operations is not None:
            return operations

        return await self._get_operations_via_sampling(service_name)

    async def health_check(self) -> HealthCheckResponse:
        """Check Sentry backend health via a minimal organization-details call.

        Returns:
            Health status information
        """
        logger.debug("Checking backend health")

        try:
            response = await self.client.get(f"/api/0/organizations/{self.org_slug}/")
            response.raise_for_status()

            return HealthCheckResponse(
                status="healthy",
                backend="sentry",
                url=self.url,
            )
        except Exception as e:
            return HealthCheckResponse(
                status="unhealthy",
                backend="sentry",
                url=self.url,
                error=str(e),
            )

    # -- internal helpers ---------------------------------------------------

    async def _get_operations_via_attribute_values(self, service_name: str) -> list[str] | None:
        """Try the trace-item attribute-values endpoint for span.op values.

        Args:
            service_name: Project slug to scope the query to

        Returns:
            Sorted list of operation names, or None if the endpoint is
            unavailable/unexpected-shaped (caller should fall back)
        """
        try:
            response = await self.client.get(
                f"/api/0/organizations/{self.org_slug}/trace-items/attributes/span.op/values/",
                params={
                    "project": service_name,
                    "dataset": "spans",
                    "per_page": 100,
                },
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.info(
                f"Sentry attribute-values endpoint unavailable ({e}); "
                "falling back to sampling recent spans"
            )
            return None

        # Response shape isn't pinned down by public docs - accept either a
        # bare list of strings or a list of dicts carrying a "value" key,
        # and degrade to the sampling fallback for anything else.
        if not isinstance(data, list):
            logger.info(
                "Sentry attribute-values response was not a list "
                f"(got {type(data).__name__}); falling back to sampling"
            )
            return None

        values: set[str] = set()
        for entry in data:
            if isinstance(entry, str) and entry:
                values.add(entry)
            elif isinstance(entry, dict) and isinstance(entry.get("value"), str) and entry["value"]:
                values.add(entry["value"])

        if not values:
            return None

        return sorted(values)

    async def _get_operations_via_sampling(self, service_name: str) -> list[str]:
        """Fall back to sampling recent spans for distinct span.op values.

        Args:
            service_name: Project slug to scope the query to

        Returns:
            Sorted list of operation names

        Raises:
            httpx.HTTPError: If the API request fails
        """
        start, end = self._time_range(None, None)
        sentry_query = f"project:{self._escape_sentry_query_value(service_name)}"

        rows = await self._search_events_raw(
            sentry_query, ["span.op"], start, end, limit=1000, project=service_name
        )

        operations = {row["span.op"] for row in rows if row.get("span.op")}
        return sorted(operations)

    async def _search_events_raw(
        self,
        sentry_query: str,
        fields: list[str],
        start: datetime,
        end: datetime,
        limit: int,
        project: str | None = None,
    ) -> list[dict[str, Any]]:
        """Call Sentry's table-format Events API, following Link pagination.

        Sentry paginates via a cursor carried in the HTTP ``Link`` response
        header rather than a JSON body field. This follows ``rel="next"``
        links (while their ``results`` flag is ``"true"``) until either
        ``limit`` rows have been collected or pagination stops, bounded by
        ``_MAX_SEARCH_PAGES`` so a pathological query can't loop forever.

        Args:
            sentry_query: Sentry search syntax query string
            fields: Column names to request
            start: Start of the search window
            end: End of the search window
            limit: Target total number of rows to collect
            project: Optional project slug to scope the query to; falls
                back to the backend's configured ``project_slug``, if any

        Returns:
            List of raw row dicts from the ``data`` array

        Raises:
            httpx.HTTPError: If the API request fails
        """
        collected: list[dict[str, Any]] = []
        cursor: str | None = None
        effective_project = project or self.project_slug

        for _ in range(_MAX_SEARCH_PAGES):
            remaining = limit - len(collected)
            if remaining <= 0:
                break

            params: dict[str, Any] = {
                "dataset": "spans",
                "field": fields,
                "query": sentry_query,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "per_page": min(max(remaining, 1), 100),
            }
            if effective_project:
                params["project"] = effective_project
            if cursor:
                params["cursor"] = cursor

            response = await self.client.get(
                f"/api/0/organizations/{self.org_slug}/events/", params=params
            )
            response.raise_for_status()

            data = response.json()

            # A 200 response with an unexpected shape (a scalar/list body or
            # a non-list `data`) would otherwise crash callers that do
            # `row.get(...)` directly - validate defensively instead of
            # trusting the shape.
            if not isinstance(data, dict):
                logger.warning(
                    f"Sentry events response body was not an object "
                    f"(got {type(data).__name__}); treating as empty"
                )
                break

            rows = data.get("data", [])
            if not isinstance(rows, list):
                logger.warning(
                    f"Sentry events response 'data' was not a list "
                    f"(got {type(rows).__name__}); treating as empty"
                )
                rows = []

            collected.extend(row for row in rows if isinstance(row, dict))

            cursor = self._next_cursor_from_links(response.links)
            if not cursor:
                break
        else:
            logger.warning(
                f"Stopped after {_MAX_SEARCH_PAGES} pages with more results available "
                f"(query: {sentry_query!r}); results may be incomplete"
            )

        return collected

    def _next_cursor_from_links(self, links: dict[str | None, dict[str, str]]) -> str | None:
        """Extract the next-page cursor from an httpx-parsed Link header.

        Args:
            links: ``httpx.Response.links`` - a dict keyed by ``rel``

        Returns:
            The next cursor token, or None if there is no further page
        """
        next_link = links.get("next")
        if not isinstance(next_link, dict):
            return None
        if next_link.get("results") != "true":
            return None
        cursor = next_link.get("cursor")
        return cursor if isinstance(cursor, str) and cursor else None

    def _flatten_trace_items(self, items: list[Any]) -> list[dict[str, Any]]:
        """Flatten a nested SerializedTraceItem tree into a flat list.

        Iterative (not recursive) to avoid any risk of hitting Python's
        recursion limit on a pathologically deep tree.

        Args:
            items: Top-level items from the /trace/{id}/ response

        Returns:
            Flat list of every item (root and descendant) that is a dict
        """
        flat: list[dict[str, Any]] = []
        stack: list[Any] = list(items)

        while stack:
            item = stack.pop()
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dict Sentry trace item (got {type(item).__name__})")
                continue
            flat.append(item)
            children = item.get("children")
            if isinstance(children, list):
                stack.extend(children)

        return flat

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

    def _build_sentry_query(self, filters: list[Filter]) -> str:
        """Build a Sentry search syntax query string from Filter objects.

        Args:
            filters: List of Filter conditions

        Returns:
            Sentry search syntax query string (empty string means match-all)
        """
        raw_conditions = [self._filter_to_sentry_query(f) for f in filters]
        conditions = [c for c in raw_conditions if c is not None]
        return " AND ".join(conditions)

    def _sentry_field(self, field: str) -> str:
        """Map an internal field name to its Sentry search syntax name.

        Args:
            field: Internal field name (e.g. "service.name", "gen_ai.system")

        Returns:
            Sentry search field name (queried bare, no `@`-style prefix)
        """
        if field in _FACET_FIELDS:
            return _FACET_FIELDS[field]
        if field == Fields.STATUS:
            return "span.status"
        if field == Fields.DURATION:
            return "span.duration"
        return field

    def _escape_sentry_query_value(self, value: str) -> str:
        """Escape and exact-quote a value for safe interpolation into a query.

        Always quotes and escapes embedded quotes/backslashes. Used for
        every string value interpolated into a Sentry query - filter values
        and project/service names - since any of them can originate from
        external input (e.g. an MCP tool call's argument) and must not be
        able to inject additional query clauses. Unverified against a live
        account (see module docstring point 5).

        Args:
            value: Raw value to interpolate

        Returns:
            A double-quoted, escaped Sentry query term
        """
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def _format_query_value(self, value: Any, is_duration_field: bool) -> str:
        """Format a filter value for interpolation into a Sentry query term.

        Numeric values are written as unquoted literals (with an explicit
        `ms` unit suffix for the duration field, matching Sentry's
        documented duration-unit search syntax); booleans as bare
        true/false tokens; everything else through the escaping quoter.

        Args:
            value: Filter value to format
            is_duration_field: Whether the target field is span.duration

        Returns:
            A query-safe token to interpolate after `field:` / `field:>` etc.
        """
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int | float):
            return f"{value}ms" if is_duration_field else str(value)
        return self._escape_sentry_query_value(str(value))

    def _filter_to_sentry_query(self, filter_obj: Filter) -> str | None:
        """Convert a single Filter to a Sentry search syntax condition.

        Args:
            filter_obj: Filter to convert

        Returns:
            Sentry search condition string, or None if unsupported
        """
        field = self._sentry_field(filter_obj.field)
        if not _VALID_SENTRY_FIELD_RE.match(field):
            logger.warning(f"Rejecting filter with unsafe/invalid Sentry field name: {field!r}")
            return None

        operator = filter_obj.operator
        value = filter_obj.value
        values = filter_obj.values
        is_duration = field == "span.duration"

        if operator == FilterOperator.EQUALS:
            if field == "span.status" and value == "ERROR":
                return "!span.status:ok"
            if field == "span.status" and value == "OK":
                return "span.status:ok"
            return f"{field}:{self._format_query_value(value, is_duration)}"

        elif operator == FilterOperator.NOT_EQUALS:
            if field == "span.status" and value == "ERROR":
                return "span.status:ok"
            return f"!{field}:{self._format_query_value(value, is_duration)}"

        elif operator in (
            FilterOperator.GT,
            FilterOperator.GTE,
            FilterOperator.LT,
            FilterOperator.LTE,
        ):
            # Filter.value_type isn't enforced against the actual Python
            # type of `value` - reject non-numeric operands rather than
            # interpolating them unchecked into a range expression. `bool`
            # is an `int` subclass in Python but not a sensible operand.
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
            return f"{field}:{symbol}{self._format_query_value(value, is_duration)}"

        elif operator == FilterOperator.EXISTS:
            return f"has:{field}"

        elif operator == FilterOperator.NOT_EXISTS:
            return f"!has:{field}"

        elif operator == FilterOperator.IN:
            if not values:
                return None
            # Sentry's `key:[a,b]` list syntax has no documented escaping
            # for special characters within it, so this uses the same
            # OR-of-quoted-terms form as the Datadog backend instead, which
            # goes through the same escaping helper as everything else.
            or_terms = [f"{field}:{self._format_query_value(v, is_duration)}" for v in values]
            return "(" + " OR ".join(or_terms) + ")"

        logger.warning(f"Unsupported operator for Sentry query: {operator}")
        return None

    def _parse_sentry_timestamp(self, value: Any) -> datetime | None:
        """Parse a Sentry timestamp value (epoch seconds or ISO8601 string).

        Args:
            value: Raw timestamp value from a Sentry response

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
        logger.warning(f"Could not parse Sentry timestamp: {value!r}")
        return None

    def _infer_status(
        self, raw_status: Any, has_error_signal: bool
    ) -> Literal["OK", "ERROR", "UNSET"]:
        """Map a Sentry status value onto this codebase's OK/ERROR/UNSET model.

        Unlike Datadog, Sentry has a first-class status concept, so this is
        more direct than Datadog's best-effort inference - but still
        defaults to UNSET rather than guessing OK, since an unrecognized
        value should not silently read as a passing span.

        Args:
            raw_status: The span's status value (e.g. "ok", "internal_error")
            has_error_signal: Whether a separate linked-error signal (e.g. a
                non-empty `errors` list) was present

        Returns:
            "ERROR", "OK", or "UNSET"
        """
        if has_error_signal:
            return "ERROR"
        if isinstance(raw_status, str):
            status = raw_status.lower()
            if status == "ok":
                return "OK"
            if status in ("", "unset", "unknown"):
                return "UNSET"
            return "ERROR"
        return "UNSET"

    def _parse_sentry_row(self, row: dict[str, Any]) -> SpanData | None:
        """Parse a raw table-format Events API row into SpanData.

        Args:
            row: Raw row dict from the `data` array

        Returns:
            Parsed SpanData, or None if required fields are missing/invalid
        """
        try:
            span_id = row.get("id")
            trace_id = row.get("trace")
            if not span_id or not trace_id:
                return None

            parent_span_id = row.get("parent_span")
            operation = row.get("span.op") or row.get("transaction")
            project = row.get("project")
            if not operation or not project:
                # Don't fabricate identity: a substituted "unknown" would
                # silently merge structurally-unrelated spans into one fake
                # service/operation bucket downstream.
                logger.warning(f"Rejecting span {span_id}: missing operation name or project")
                return None

            start_time = self._parse_sentry_timestamp(row.get("timestamp"))
            if start_time is None:
                # Don't fabricate timing data: a substituted "now" start
                # time would silently corrupt ordering/duration for anyone
                # reading this span downstream.
                logger.warning(f"Rejecting span {span_id}: missing or invalid timestamp")
                return None

            duration_raw = row.get("span.duration")
            if not isinstance(duration_raw, int | float) or isinstance(duration_raw, bool):
                logger.warning(f"Rejecting span {span_id}: missing or invalid span.duration")
                return None
            duration_ms = float(duration_raw)
            if duration_ms < 0 or duration_ms > _MAX_REASONABLE_DURATION_MS:
                logger.warning(
                    f"Rejecting span {span_id}: out-of-range span.duration {duration_ms!r}"
                )
                return None

            status = self._infer_status(row.get("span.status"), has_error_signal=False)

            extra = self._extract_extra_attributes(row, _SPAN_STRUCTURAL_FIELDS)

            return SpanData(
                trace_id=str(trace_id),
                span_id=str(span_id),
                parent_span_id=str(parent_span_id) if parent_span_id else None,
                operation_name=str(operation),
                service_name=str(project),
                start_time=start_time,
                duration_ms=duration_ms,
                status=status,
                attributes=SpanAttributes(**extra),
                events=[],
            )
        except Exception as e:
            logger.error(f"Error parsing Sentry span row: {e}")
            return None

    def _parse_sentry_trace_item(
        self, item: dict[str, Any], requested_trace_id: str
    ) -> SpanData | None:
        """Parse a SerializedTraceItem (from /trace/{id}/) into SpanData.

        Args:
            item: Raw trace item dict (see module docstring point 4)
            requested_trace_id: The trace_id originally requested, used only
                to attribute a rejection warning to the right lookup

        Returns:
            Parsed SpanData, or None if required fields are missing/invalid
        """
        try:
            span_id = item.get("span_id") or item.get("id") or item.get("event_id")
            if not span_id:
                return None

            # A live account confirmed this endpoint's response items never
            # carry their own trace_id/trace field - GET /trace/{id}/ is
            # already scoped to one trace by the URL path itself, so Sentry
            # doesn't repeat it per item. Trusting requested_trace_id here
            # is not a fabrication: it's the only trace this response could
            # possibly contain. get_trace()'s own belt-and-suspenders check
            # (`span.trace_id == trace_id`) becomes a no-op as a result, but
            # is left in place rather than removed, in case some other
            # Sentry deployment's response shape ever does include a
            # differing value.
            trace_id = str(item.get("trace_id") or item.get("trace") or requested_trace_id)

            parent_span_id = item.get("parent_span_id") or item.get("parentSpanId")
            operation = item.get("op") or item.get("transaction") or item.get("description")
            project = item.get("project_slug") or item.get("project")
            if not operation or not project:
                # Don't fabricate identity: a substituted "unknown" would
                # silently merge structurally-unrelated items into one fake
                # service/operation bucket downstream.
                logger.warning(
                    f"Rejecting Sentry trace item {span_id}: missing operation name or project"
                )
                return None

            start_time = self._parse_sentry_timestamp(
                self._first_present(item, "start_timestamp", "precise.start_ts", "timestamp")
            )
            if start_time is None:
                logger.warning(
                    f"Rejecting Sentry trace item {span_id}: missing or invalid start timestamp"
                )
                return None

            end_time = self._parse_sentry_timestamp(
                self._first_present(item, "end_timestamp", "precise.finish_ts")
            )
            duration_raw = self._first_present(item, "duration", "span.duration")

            if end_time is not None:
                duration_ms = (end_time - start_time).total_seconds() * 1000
            elif isinstance(duration_raw, int | float) and not isinstance(duration_raw, bool):
                duration_ms = float(duration_raw)
            else:
                # Never fabricate a duration: without an end timestamp or an
                # explicit duration field, there's nothing trustworthy to
                # compute one from.
                logger.warning(
                    f"Rejecting Sentry trace item {span_id}: missing or invalid duration"
                )
                return None

            if duration_ms < 0 or duration_ms > _MAX_REASONABLE_DURATION_MS:
                # Reject clock-skew (end before start) or corrupted/garbage
                # duration values rather than letting them silently corrupt
                # duration-based filtering/aggregation, or overflow the
                # `datetime` arithmetic in `_group_into_trace` downstream.
                logger.warning(
                    f"Rejecting Sentry trace item {span_id}: out-of-range duration {duration_ms!r}"
                )
                return None

            errors = item.get("errors")
            # The exact shape of `errors` isn't published (module docstring
            # point 4) - it may be a list of linked error events, or an
            # integer count. Treat any non-list-but-truthy value (e.g. a
            # non-zero count) as a real error signal too, rather than
            # silently reading it as "no errors" just because it isn't a
            # list.
            has_error_signal = len(errors) > 0 if isinstance(errors, list) else bool(errors)
            status = self._infer_status(
                item.get("status") or item.get("span.status"), has_error_signal
            )

            extra = self._extract_extra_attributes(item, _TRACE_ITEM_STRUCTURAL_FIELDS)

            return SpanData(
                trace_id=trace_id,
                span_id=str(span_id),
                parent_span_id=str(parent_span_id) if parent_span_id else None,
                operation_name=str(operation),
                service_name=str(project),
                start_time=start_time,
                duration_ms=duration_ms,
                status=status,
                attributes=SpanAttributes(**extra),
                events=[],
            )
        except Exception as e:
            logger.error(f"Error parsing Sentry trace item: {e}")
            return None

    def _first_present(self, source: dict[str, Any], *keys: str) -> Any:
        """Return the value of the first key present in ``source`` with a
        non-None value, trying each candidate key in order.

        Unlike chaining ``a or b or c``, this correctly treats a legitimate
        falsy-but-present value (e.g. ``0`` or ``0.0`` - a perfectly valid
        timestamp or duration) as present, rather than skipping to the next
        candidate key.

        Args:
            source: Dict to look up keys in
            *keys: Candidate key names, in priority order

        Returns:
            The first non-None value found, or None if none of the keys are
            present (or all present values are None)
        """
        for key in keys:
            value = source.get(key)
            if value is not None:
                return value
        return None

    def _extract_extra_attributes(
        self, source: dict[str, Any], structural_fields: Any
    ) -> dict[str, Any]:
        """Extract non-structural, JSON-scalar fields for SpanAttributes.

        Args:
            source: Raw row/item dict
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

        # Preserve UNSET rather than defaulting to OK: "no span confirmed an
        # error" is not the same claim as "every span confirmed success."
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

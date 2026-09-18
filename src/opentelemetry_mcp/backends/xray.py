"""AWS X-Ray backend implementation using GetTraceSummaries/BatchGetTraces.

boto3 is fully synchronous; every call to the underlying client is wrapped in
``asyncio.to_thread`` (see ``_call`` below) rather than pulling in
``aioboto3``/``aiobotocore`` as a second, less-mainstream AWS SDK dependency.

Schema/behavior notes, grounded in AWS's own docs (xray-concepts.html,
API_GetTraceSummaries.html, API_BatchGetTraces.html,
xray-api-segmentdocuments.html, xray-console-filters.html,
migrate-xray-to-opentelemetry-python.html) rather than a live account (none
was available in this environment - see the Verification section of the
implementation plan), following the same "grounded in docs, flag for a live
account to confirm" convention already used in sentry.py's own docstring:

1. boto3's HTTP transport is TLS-only by default for every AWS service
   endpoint (SigV4 over https://) unless a caller explicitly passes
   ``use_ssl=False`` via ``botocore.config.Config`` - this backend never
   does, so the "reject non-https" pattern used elsewhere in this codebase
   is satisfied by boto3's own default rather than an explicit check here.
2. This backend never touches ``BaseBackend.client``/``_create_headers``'s
   httpx machinery - ``self.url`` is purely decorative (see config.py's
   ``BackendConfig.url`` docstring and ``_create_backend``'s xray branch);
   the functional endpoint selector is ``aws_region``. All of
   ``_RetryingTransport``'s redirect/retry/SSRF-guard logic is simply
   unused, not bypassed insecurely - there is no httpx client at all.
3. ``FilterExpression`` is a real, string-interpolated mini-DSL (like
   Datadog's span-search syntax and Sentry's Discover syntax) - every value
   spliced into it goes through ``_escape_xray_filter_value``.
4. By default, OpenTelemetry span attributes are converted to X-Ray segment
   **metadata**, not annotations - only annotations are indexed for
   ``FilterExpression`` search. This backend therefore does NOT attempt to
   map arbitrary ``gen_ai.*``/custom fields onto an ``annotation.<key>``
   predicate (doing so would silently return zero matches whenever the
   field isn't actually indexed by the customer's own OTel/ADOT collector
   config, which cannot be detected at runtime) - only a small, genuinely
   structural field set is natively pushed down (service name, duration,
   error/fault/throttle-derived status via ``_NATIVE_TRACE_LEVEL_FIELDS``);
   everything else is applied client-side via ``FilterEngine``, mirroring
   Jaeger's/Tempo's "most filtering is client-side" precedent more than
   Datadog/Sentry's "most operators are native" one.
5. ``BatchGetTraces``' ``Segment.Document`` arrives as a JSON-encoded
   **string** nested inside the already-JSON response body - this double
   encoding is parsed and shape-validated explicitly in
   ``_parse_segment_document``.
6. Segment/subsegment schema has no per-subsegment service-name field of
   its own (only ``name``, which for a subsegment is closer to this
   codebase's ``operation_name``) - ``service_name`` is inherited top-down
   from the nearest top-level-segment ancestor while flattening the tree.
   This may not perfectly reflect X-Ray's own "inferred segment for a
   downstream service" behavior for a subsegment representing a call to a
   different, OTel-instrumented service (which can itself become a
   distinct top-level segment in a real trace) - verify against a live,
   multi-service account before relying on this in production.

Someone with a live AWS account and real gen_ai-instrumented ADOT/OTel
traces should verify all of the above before relying on this in production.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError

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

# AWS X-Ray's documented trace data retention (docs.aws.amazon.com/xray/
# latest/devguide/xray-concepts.html: "Trace data is retained for 30 days") -
# grounded in an explicit documented ceiling, matching Datadog's own default
# lookback value.
_DEFAULT_LOOKBACK = timedelta(days=30)

_MAX_TRACES_TO_HYDRATE = 50

# BatchGetTraces' own hard API limit ("Array Members: Minimum number of 1
# item. Maximum number of 5 items.") - a real service constraint, not a
# convention borrowed from another backend.
_BATCH_GET_TRACES_MAX_IDS = 5

# Safety bound on GetTraceSummaries' NextToken pagination and on
# BatchGetTraces' own per-batch-of-<=5-ids NextToken pagination (segments
# for a large trace can themselves paginate), so a pathological query/trace
# can't loop forever.
_MAX_SEARCH_PAGES = 10
_MAX_BATCH_PAGES = 10

# Sanity upper bound (10 years, in milliseconds) on a parsed segment
# duration - mirrors the same bound used by the Datadog/Sentry backends.
_MAX_REASONABLE_DURATION_MS = 1000 * 60 * 60 * 24 * 365 * 10

# Fields this backend can honestly push down into GetTraceSummaries'
# FilterExpression - see module docstring point 4 for why this is
# deliberately small rather than "every field, @-prefixed" like Datadog/Sentry.
_NATIVE_TRACE_LEVEL_FIELDS = {Fields.SERVICE_NAME, Fields.DURATION, Fields.STATUS}

# Segment-document keys that are structural (never span attributes) -
# excluded when flattening a segment's own top-level scalar fields, mirrors
# Sentry's _TRACE_ITEM_STRUCTURAL_FIELDS convention.
_STRUCTURAL_SEGMENT_FIELDS = frozenset(
    {
        "id",
        "name",
        "trace_id",
        "parent_id",
        "start_time",
        "end_time",
        "in_progress",
        "type",
        "error",
        "fault",
        "throttle",
        "annotations",
        "metadata",
        "subsegments",
    }
)


class XRayBackend(BaseBackend):
    """AWS X-Ray GetTraceSummaries/BatchGetTraces backend."""

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        aws_region: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize the X-Ray backend.

        Args:
            url: Decorative only - see module docstring point 2 and
                config.py's BackendConfig.url docstring. Never dereferenced
                for a live request.
            api_key: Unused - X-Ray uses AWS's own SigV4-signed request
                credentials (boto3's default credential chain: environment
                variables, shared config/credentials files, an assumed
                role, or an instance/task role), never a bearer-token-style
                header. Accepted only to satisfy BaseBackend's constructor
                shape.
            aws_region: AWS region the X-Ray API calls target (e.g.
                "us-east-1") - required, since boto3 has no default that's
                safe to assume for this project.
            timeout: Request timeout in seconds, applied via
                botocore.config.Config's connect/read timeouts.
        """
        super().__init__(url, api_key, timeout)

        if not aws_region:
            raise ValueError(
                "X-Ray backend requires an AWS region (BACKEND_AWS_REGION) - "
                "boto3 has no safe default to assume for this project"
            )
        self.aws_region = aws_region

        botocore_config = BotocoreConfig(
            connect_timeout=self.timeout,
            read_timeout=self.timeout,
            retries={"max_attempts": 3, "mode": "standard"},
        )
        self._xray_client = boto3.client("xray", region_name=aws_region, config=botocore_config)

    def _create_headers(self) -> dict[str, str]:
        """N/A for this backend - X-Ray auth is SigV4 request signing
        applied internally by boto3, not a static header this codebase's
        shared httpx client construction ever uses. Returns an empty dict
        to satisfy the abstract method; BaseBackend.client (the only
        caller) is never invoked by this backend."""
        return {}

    def get_supported_operators(self) -> set[FilterOperator]:
        """Operators natively pushable into GetTraceSummaries' FilterExpression.

        Deliberately narrow (module docstring point 4): a field not in
        _NATIVE_TRACE_LEVEL_FIELDS is never pushed down regardless of
        operator - see _split_filters.
        """
        return {
            FilterOperator.EQUALS,
            FilterOperator.NOT_EQUALS,
            FilterOperator.GT,
            FilterOperator.LT,
            FilterOperator.GTE,
            FilterOperator.LTE,
            FilterOperator.IN,
        }

    async def _call(self, operation: str, **kwargs: Any) -> dict[str, Any]:
        """Run one synchronous boto3 X-Ray client call off the event loop.

        Args:
            operation: boto3 client method name (e.g. "get_trace_summaries")
            **kwargs: Passed through to the boto3 method

        Returns:
            The raw boto3 response dict

        Raises:
            botocore.exceptions.ClientError: Left to propagate bare,
                matching every other backend's convention of letting its
                underlying SDK's exceptions bubble up unwrapped.
        """
        method = getattr(self._xray_client, operation)
        result: dict[str, Any] = await asyncio.to_thread(method, **kwargs)
        return result

    async def close(self) -> None:
        """Close the boto3 client, then delegate to BaseBackend.close() for
        the query-cache clear (its httpx-close half is a no-op here, since
        self._client is never populated by this backend)."""
        self._xray_client.close()
        await super().close()

    # -- public interface ----------------------------------------------

    async def search_traces(self, query: TraceQuery) -> list[TraceData]:
        """Search-then-hydrate: GetTraceSummaries discovers candidate trace
        IDs (with whatever native FilterExpression pushdown applies),
        BatchGetTraces (batched _BATCH_GET_TRACES_MAX_IDS IDs per call -
        X-Ray's own hard API limit) hydrates each into full segment trees,
        then every filter is re-applied client-side against the
        fully-hydrated trace - mirrors DatadogBackend.search_traces/
        SentryBackend.search_traces exactly.
        """
        all_filters = query.get_all_filters()
        filter_expression = self._build_filter_expression(all_filters)
        start, end = self._time_range(query.start_time, query.end_time)

        summaries = await self._get_trace_summaries_raw(
            filter_expression, start, end, query.limit * 5
        )
        trace_ids = self._dedupe_trace_ids(summaries)

        max_to_fetch = min(len(trace_ids), _MAX_TRACES_TO_HYDRATE)
        if len(trace_ids) > max_to_fetch:
            logger.warning(
                f"Limiting trace fetch to {max_to_fetch} out of {len(trace_ids)} "
                f"results to avoid excessive API calls"
            )
        ids_to_fetch = trace_ids[:max_to_fetch]

        segments_by_id, unprocessed = await self._batch_get_traces_raw(ids_to_fetch)
        if unprocessed:
            logger.warning(
                f"X-Ray reported {len(unprocessed)} unprocessed trace IDs: {sorted(unprocessed)}"
            )

        traces: list[TraceData] = []
        for trace_id in ids_to_fetch:
            segments = segments_by_id.get(trace_id)
            if not segments:
                continue
            spans = self._parse_segments_to_spans(trace_id, segments)
            if spans:
                traces.append(self._group_into_trace(trace_id, spans))

        if all_filters:
            traces = FilterEngine.apply_filters(traces, all_filters)

        return traces[: query.limit]

    async def search_spans(self, query: SpanQuery) -> list[SpanData]:
        """No dedicated X-Ray spans API - search traces, flatten to spans,
        apply remaining filters client-side, truncate to query.limit -
        mirrors JaegerBackend.search_spans/TempoBackend.search_spans's
        documented "no dedicated spans endpoint" pattern."""
        all_filters = query.get_all_filters()
        filter_expression = self._build_filter_expression(all_filters)
        start, end = self._time_range(query.start_time, query.end_time)

        summaries = await self._get_trace_summaries_raw(
            filter_expression, start, end, query.limit * 2
        )
        trace_ids = self._dedupe_trace_ids(summaries)
        ids_to_fetch = trace_ids[:_MAX_TRACES_TO_HYDRATE]

        segments_by_id, _unprocessed = await self._batch_get_traces_raw(ids_to_fetch)

        spans: list[SpanData] = []
        for trace_id in ids_to_fetch:
            segments = segments_by_id.get(trace_id)
            if segments:
                spans.extend(self._parse_segments_to_spans(trace_id, segments))

        if all_filters:
            spans = FilterEngine.apply_filters(spans, all_filters)

        return spans[: query.limit]

    async def get_trace(self, trace_id: str) -> TraceData:
        """BatchGetTraces(TraceIds=[trace_id]) directly - no FilterExpression
        involved (BatchGetTraces takes an exact ID list, not a string DSL,
        so there is no query-value-escaping concern for trace_id
        specifically). Checks UnprocessedTraceIds explicitly (an
        X-Ray-specific failure mode) and only trusts segments keyed under
        the exact requested trace_id in the response - re-verifying the
        returned trace's identity rather than assuming a match.
        """
        segments_by_id, unprocessed = await self._batch_get_traces_raw([trace_id])

        if trace_id in unprocessed:
            raise ValueError(f"X-Ray reported trace {trace_id} as unprocessed (BatchGetTraces)")

        segments = segments_by_id.get(trace_id)
        if not segments:
            raise ValueError(f"No trace found for id {trace_id}")

        spans = self._parse_segments_to_spans(trace_id, segments)
        if not spans:
            raise ValueError(f"No parseable segments found for trace {trace_id}")

        return self._group_into_trace(trace_id, spans)

    async def list_services(self) -> list[str]:
        """Two-tier: try GetServiceGraph first (a real, if IAM-permission-
        gated, X-Ray endpoint that returns each service-graph node's Name
        directly - confirmed to be a distinct, separately-permissioned
        action from GetTraceSummaries), falling back to sampling recent
        GetTraceSummaries+BatchGetTraces results on any failure (missing
        permission, unexpected shape, etc.) - mirrors SentryBackend's own
        sampling-fallback shape."""
        start, end = self._time_range(None, None)

        try:
            response = await self._call("get_service_graph", StartTime=start, EndTime=end)
            raw_services = response.get("Services") if isinstance(response, dict) else None
            if isinstance(raw_services, list):
                names: set[str] = set()
                for s in raw_services:
                    name = s.get("Name") if isinstance(s, dict) else None
                    if isinstance(name, str) and name:
                        names.add(name)
                if names:
                    return sorted(names)
        except ClientError as e:
            logger.warning(f"GetServiceGraph failed ({e}); falling back to sampling recent traces")

        return await self._sample_service_names(start, end)

    async def get_service_operations(self, service_name: str) -> list[str]:
        """Sampling fallback only (no dedicated endpoint): GetTraceSummaries
        scoped by the natively-supported service(<name>) FilterExpression
        predicate, BatchGetTraces to hydrate a bounded sample, collect
        distinct segment/subsegment names belonging to that service -
        mirrors DatadogBackend.get_service_operations's shape."""
        start, end = self._time_range(None, None)
        filter_expression = f"service({self._escape_xray_filter_value(service_name)})"

        summaries = await self._get_trace_summaries_raw(filter_expression, start, end, limit=1000)
        trace_ids = self._dedupe_trace_ids(summaries)[:_MAX_TRACES_TO_HYDRATE]

        segments_by_id, _unprocessed = await self._batch_get_traces_raw(trace_ids)

        operations: set[str] = set()
        for trace_id in trace_ids:
            segments = segments_by_id.get(trace_id)
            if not segments:
                continue
            for span in self._parse_segments_to_spans(trace_id, segments):
                if span.service_name == service_name:
                    operations.add(span.operation_name)

        return sorted(operations)

    async def health_check(self) -> HealthCheckResponse:
        """Calls list_services() and wraps success/failure - identical
        shape to every other backend's health_check."""
        try:
            await self.list_services()
            return HealthCheckResponse(status="healthy", backend="xray", url=self.url)
        except Exception as e:
            return HealthCheckResponse(
                status="unhealthy", backend="xray", url=self.url, error=str(e)
            )

    # -- internal helpers: sampling fallback -----------------------------

    async def _sample_service_names(self, start: datetime, end: datetime) -> list[str]:
        """Sample recent traces (no FilterExpression) and collect distinct
        service names from every segment's inherited service_name."""
        summaries = await self._get_trace_summaries_raw(None, start, end, limit=1000)
        trace_ids = self._dedupe_trace_ids(summaries)[:_MAX_TRACES_TO_HYDRATE]

        segments_by_id, _unprocessed = await self._batch_get_traces_raw(trace_ids)

        services: set[str] = set()
        for trace_id in trace_ids:
            segments = segments_by_id.get(trace_id)
            if not segments:
                continue
            for span in self._parse_segments_to_spans(trace_id, segments):
                services.add(span.service_name)

        return sorted(services)

    def _dedupe_trace_ids(self, summaries: list[dict[str, Any]]) -> list[str]:
        """Extract unique trace IDs from raw TraceSummaries items,
        preserving discovery order."""
        trace_ids: list[str] = []
        seen: set[str] = set()
        for summary in summaries:
            trace_id = summary.get("Id")
            if isinstance(trace_id, str) and trace_id and trace_id not in seen:
                seen.add(trace_id)
                trace_ids.append(trace_id)
        return trace_ids

    # -- internal helpers: FilterExpression building --------------------

    def _escape_xray_filter_value(self, value: str) -> str:
        """Always double-quote + escape embedded quotes/backslashes - same
        convention as _escape_dd_query_value/_escape_sentry_query_value,
        used for every value spliced into a FilterExpression string."""
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def _split_filters(self, all_filters: list[Filter]) -> tuple[list[Filter], list[Filter]]:
        """Native filters are gated by BOTH operator membership in
        get_supported_operators() AND field membership in
        _NATIVE_TRACE_LEVEL_FIELDS - see module docstring point 4.
        Everything else (in particular every gen_ai.*/custom attribute)
        goes client-side."""
        supported_operators = self.get_supported_operators()
        native: list[Filter] = []
        client_side: list[Filter] = []
        for f in all_filters:
            if f.operator in supported_operators and f.field in _NATIVE_TRACE_LEVEL_FIELDS:
                native.append(f)
            else:
                client_side.append(f)
        return native, client_side

    def _build_filter_expression(self, all_filters: list[Filter]) -> str | None:
        """Combine _filter_to_xray_expression results with AND. Returns
        None (not "") when there are no native filters, matching
        GetTraceSummaries' own FilterExpression being optional. Every
        filter (native or not) is still re-applied client-side by the
        caller via FilterEngine, matching Datadog/Sentry's "search then
        re-verify everything" convention."""
        native_filters, _client_side = self._split_filters(all_filters)
        parts = [self._filter_to_xray_expression(f) for f in native_filters]
        conditions = [p for p in parts if p is not None]
        if not conditions:
            return None
        return " AND ".join(conditions)

    def _filter_to_xray_expression(self, filter_obj: Filter) -> str | None:
        """Fields.SERVICE_NAME -> service("<escaped value>") (or an OR-group
        of them for IN). Fields.DURATION -> "duration <op> <value/1000>"
        (X-Ray's duration keyword is in SECONDS, unlike this codebase's
        milliseconds - scaled accordingly, mirroring Datadog's own
        ns<->ms scaling pattern). Rejects non-numeric operands (bool is an
        int subclass) for duration. Fields.STATUS EQUALS "ERROR" ->
        "(error = true OR fault = true)"; EQUALS "OK" -> "(error = false
        AND fault = false AND throttle = false)" (never fabricates an
        "ok = true" keyword - not confirmed to exist in X-Ray's documented
        filter syntax). Any other field/operator combination reaching here
        (e.g. GT on service_name) is unsupported and returns None - it is
        still re-applied client-side by the caller.
        """
        field = filter_obj.field
        operator = filter_obj.operator
        value = filter_obj.value

        if field == Fields.SERVICE_NAME:
            if operator == FilterOperator.EQUALS and isinstance(value, str):
                return f"service({self._escape_xray_filter_value(value)})"
            if operator == FilterOperator.NOT_EQUALS and isinstance(value, str):
                return f"NOT service({self._escape_xray_filter_value(value)})"
            if operator == FilterOperator.IN and filter_obj.values:
                or_terms = [
                    f"service({self._escape_xray_filter_value(str(v))})" for v in filter_obj.values
                ]
                return "(" + " OR ".join(or_terms) + ")"
            return None

        if field == Fields.DURATION:
            if not isinstance(value, int | float) or isinstance(value, bool):
                logger.warning(
                    f"Skipping non-numeric duration value for {operator.value}: {value!r}"
                )
                return None
            seconds = value / 1000
            if operator == FilterOperator.EQUALS:
                return f"duration = {seconds}"
            if operator == FilterOperator.NOT_EQUALS:
                return f"duration != {seconds}"
            if operator == FilterOperator.GT:
                return f"duration > {seconds}"
            if operator == FilterOperator.GTE:
                return f"duration >= {seconds}"
            if operator == FilterOperator.LT:
                return f"duration < {seconds}"
            if operator == FilterOperator.LTE:
                return f"duration <= {seconds}"
            return None

        if field == Fields.STATUS:
            if operator == FilterOperator.EQUALS:
                if value == "ERROR":
                    return "(error = true OR fault = true)"
                if value == "OK":
                    return "(error = false AND fault = false AND throttle = false)"
                return None
            if operator == FilterOperator.NOT_EQUALS:
                if value == "ERROR":
                    return "(error = false AND fault = false)"
                if value == "OK":
                    return "(error = true OR fault = true OR throttle = true)"
                return None
            return None

        return None

    # -- internal helpers: GetTraceSummaries/BatchGetTraces pagination ---

    async def _get_trace_summaries_raw(
        self,
        filter_expression: str | None,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Follows NextToken up to _MAX_SEARCH_PAGES, mirroring
        _search_spans_raw (Datadog)/_search_events_raw (Sentry)'s exact
        pagination shape. Validates the response shape before extending
        the collected list."""
        collected: list[dict[str, Any]] = []
        next_token: str | None = None

        for _ in range(_MAX_SEARCH_PAGES):
            if len(collected) >= limit:
                break

            kwargs: dict[str, Any] = {"StartTime": start, "EndTime": end}
            if filter_expression:
                kwargs["FilterExpression"] = filter_expression
            if next_token:
                kwargs["NextToken"] = next_token

            response = await self._call("get_trace_summaries", **kwargs)
            if not isinstance(response, dict):
                logger.warning(
                    f"GetTraceSummaries response was not an object "
                    f"(got {type(response).__name__}); treating as empty"
                )
                break

            summaries = response.get("TraceSummaries", [])
            if not isinstance(summaries, list):
                logger.warning(
                    f"GetTraceSummaries 'TraceSummaries' was not a list "
                    f"(got {type(summaries).__name__}); treating as empty"
                )
                summaries = []
            collected.extend(s for s in summaries if isinstance(s, dict))

            next_token = response.get("NextToken")
            if not isinstance(next_token, str) or not next_token:
                break
        else:
            logger.warning(
                f"Stopped after {_MAX_SEARCH_PAGES} pages with more results available "
                f"(filter: {filter_expression!r}); results may be incomplete"
            )

        return collected

    async def _batch_get_traces_raw(
        self, trace_ids: list[str]
    ) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
        """Batches trace_ids into groups of <= _BATCH_GET_TRACES_MAX_IDS (5,
        X-Ray's own hard limit) per BatchGetTraces call, and within each
        call follows that call's own NextToken up to _MAX_BATCH_PAGES.
        Returns (segments_by_trace_id, unprocessed_trace_ids); the returned
        dict is keyed by each Trace's OWN reported Id from the response
        (not the requested id blindly assumed to match), so a caller
        looking up a specific requested id can only find segments under
        the exact id X-Ray actually returned.
        """
        segments_by_trace_id: dict[str, list[dict[str, Any]]] = {}
        unprocessed: set[str] = set()

        for i in range(0, len(trace_ids), _BATCH_GET_TRACES_MAX_IDS):
            chunk = trace_ids[i : i + _BATCH_GET_TRACES_MAX_IDS]
            next_token: str | None = None

            for _ in range(_MAX_BATCH_PAGES):
                kwargs: dict[str, Any] = {"TraceIds": chunk}
                if next_token:
                    kwargs["NextToken"] = next_token

                response = await self._call("batch_get_traces", **kwargs)
                if not isinstance(response, dict):
                    logger.warning(
                        f"BatchGetTraces response was not an object "
                        f"(got {type(response).__name__}); treating as empty"
                    )
                    break

                raw_traces = response.get("Traces", [])
                if isinstance(raw_traces, list):
                    for raw_trace in raw_traces:
                        if not isinstance(raw_trace, dict):
                            continue
                        trace_id = raw_trace.get("Id")
                        if not isinstance(trace_id, str) or not trace_id:
                            continue
                        raw_segments = raw_trace.get("Segments", [])
                        if not isinstance(raw_segments, list):
                            raw_segments = []
                        segments_by_trace_id.setdefault(trace_id, []).extend(
                            s for s in raw_segments if isinstance(s, dict)
                        )

                raw_unprocessed = response.get("UnprocessedTraceIds", [])
                if isinstance(raw_unprocessed, list):
                    unprocessed.update(u for u in raw_unprocessed if isinstance(u, str))

                next_token = response.get("NextToken")
                if not isinstance(next_token, str) or not next_token:
                    break
            else:
                logger.warning(
                    f"Stopped after {_MAX_BATCH_PAGES} pages of BatchGetTraces "
                    f"for batch {chunk}; segments may be incomplete"
                )

        return segments_by_trace_id, unprocessed

    # -- internal helpers: segment document parsing ----------------------

    def _parse_segment_document(self, raw_document: str) -> dict[str, Any] | None:
        """Parses Segment.Document's JSON-encoded STRING (the double
        encoding - see module docstring point 5) into a dict, validating
        the top-level shape is actually a dict (not a list/scalar) before
        returning it."""
        try:
            parsed = json.loads(raw_document)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Skipping malformed X-Ray segment document: {e}")
            return None
        if not isinstance(parsed, dict):
            logger.warning(
                f"Skipping X-Ray segment document with unexpected shape "
                f"(got {type(parsed).__name__}, expected object)"
            )
            return None
        return parsed

    def _flatten_segment_tree(
        self, root_documents: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], str]]:
        """Iterative (stack-based, not recursive, so a pathologically deep
        subsegment tree can't hit Python's recursion limit) flatten of
        every top-level segment document plus its nested subsegments
        array, carrying each item's INHERITED service name alongside it
        (module docstring point 6): a top-level document's own name
        becomes the service name for itself and every subsegment nested
        beneath it; a subsegment never supplies its own service name."""
        results: list[tuple[dict[str, Any], str]] = []
        stack: list[tuple[dict[str, Any], str]] = []

        for doc in root_documents:
            name = doc.get("name")
            if isinstance(name, str) and name:
                stack.append((doc, name))

        while stack:
            item, service_name = stack.pop()
            results.append((item, service_name))
            subsegments = item.get("subsegments")
            if isinstance(subsegments, list):
                for sub in subsegments:
                    if isinstance(sub, dict):
                        stack.append((sub, service_name))

        return results

    def _parse_segment_item(
        self, item: dict[str, Any], inherited_service_name: str, trace_id: str
    ) -> SpanData | None:
        """Builds a flat attributes dict from annotations + metadata (the
        same nested-object walk() idiom DatadogBackend._extract_semconv_attributes
        already uses) then SpanAttributes(**dict).

        Rejects (returns None, logs) rather than fabricating when:
        - id/name missing (never fabricate identity)
        - start_time missing/non-numeric (never fabricate timing)
        - neither end_time nor in_progress=true is present (never invents
          an end time)
        - resulting duration_ms is negative or exceeds
          _MAX_REASONABLE_DURATION_MS

        Status: ERROR if error or fault is True; explicitly OK only when
        error/fault/throttle are ALL present and all False (a genuine
        "checked, and it's clean" signal); UNSET otherwise - never
        defaults an absent/partial signal to OK.
        """
        segment_id = item.get("id")
        name = item.get("name")
        if (
            not isinstance(segment_id, str)
            or not segment_id
            or not isinstance(name, str)
            or not name
        ):
            logger.warning(f"Skipping X-Ray segment for trace {trace_id}: missing id/name")
            return None

        start_time_raw = item.get("start_time")
        if not isinstance(start_time_raw, int | float) or isinstance(start_time_raw, bool):
            logger.warning(f"Skipping X-Ray segment {segment_id}: missing/invalid start_time")
            return None

        end_time_raw = item.get("end_time")
        in_progress = item.get("in_progress") is True
        if not isinstance(end_time_raw, int | float) or isinstance(end_time_raw, bool):
            if in_progress:
                logger.debug(f"Skipping in-progress X-Ray segment {segment_id} (no end_time yet)")
            else:
                logger.warning(
                    f"Skipping X-Ray segment {segment_id}: no end_time and not in_progress"
                )
            return None

        duration_ms = (end_time_raw - start_time_raw) * 1000
        if duration_ms < 0 or duration_ms > _MAX_REASONABLE_DURATION_MS:
            logger.warning(
                f"Skipping X-Ray segment {segment_id}: implausible duration_ms={duration_ms}"
            )
            return None

        parent_id_raw = item.get("parent_id")
        parent_span_id = parent_id_raw if isinstance(parent_id_raw, str) and parent_id_raw else None

        error = item.get("error")
        fault = item.get("fault")
        throttle = item.get("throttle")
        status: Literal["OK", "ERROR", "UNSET"]
        if error is True or fault is True:
            status = "ERROR"
        elif error is False and fault is False and throttle is False:
            status = "OK"
        else:
            status = "UNSET"

        attrs_dict = self._extract_segment_attributes(item)

        return SpanData(
            trace_id=trace_id,
            span_id=segment_id,
            parent_span_id=parent_span_id,
            operation_name=name,
            service_name=inherited_service_name,
            start_time=datetime.fromtimestamp(start_time_raw, tz=UTC),
            duration_ms=duration_ms,
            status=status,
            attributes=SpanAttributes(**attrs_dict),
        )

    def _extract_segment_attributes(self, item: dict[str, Any]) -> dict[str, Any]:
        """Flatten annotations + metadata into a dotted-key dict suitable
        for SpanAttributes(**dict) - mirrors
        DatadogBackend._extract_semconv_attributes's walk() idiom exactly.
        annotations are indexed/queryable (see module docstring point 4);
        metadata is not, but both are surfaced here identically since this
        backend only cares about presenting attributes, not about
        server-side searchability."""
        flat: dict[str, Any] = {}

        def walk(prefix: str, obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    key = f"{prefix}.{k}" if prefix else k
                    walk(key, v)
            else:
                flat[prefix] = obj

        annotations = item.get("annotations")
        if isinstance(annotations, dict):
            walk("", annotations)
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            walk("", metadata)

        return flat

    def _parse_segments_to_spans(
        self, trace_id: str, raw_segments: list[dict[str, Any]]
    ) -> list[SpanData]:
        """Parse a trace's raw BatchGetTraces Segments list (each with an
        Id and a JSON-encoded Document string) into a flat list of
        SpanData: decode each top-level Document, flatten its subsegment
        tree, parse each resulting item."""
        root_documents: list[dict[str, Any]] = []
        for raw_segment in raw_segments:
            document_str = raw_segment.get("Document")
            if not isinstance(document_str, str):
                continue
            document = self._parse_segment_document(document_str)
            if document is not None:
                root_documents.append(document)

        spans: list[SpanData] = []
        for item, service_name in self._flatten_segment_tree(root_documents):
            span = self._parse_segment_item(item, service_name, trace_id)
            if span is not None:
                spans.append(span)

        return spans

    def _group_into_trace(self, trace_id: str, spans: list[SpanData]) -> TraceData:
        """Identical shape to every other backend's _group_into_trace
        (root = first span with no parent_span_id else spans[0]; ERROR if
        any span has_error, OK if all OK, else UNSET)."""
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

    def _time_range(
        self,
        start_time: datetime | None,
        end_time: datetime | None,
        lookback: timedelta = _DEFAULT_LOOKBACK,
    ) -> tuple[datetime, datetime]:
        """Identical shape to Datadog/Sentry's own _time_range."""
        end = end_time or datetime.now(UTC)
        start = start_time or (end - lookback)
        return start, end

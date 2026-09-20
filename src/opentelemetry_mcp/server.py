"""Opentelemetry MCP Server - Main entry point."""

import asyncio
import json
import logging
import re
import sys
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, Literal, NoReturn

import click
import uvicorn
from fastmcp import FastMCP
from fastmcp.server.http import StarletteWithLifespan
from mcp.types import ToolAnnotations
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from opentelemetry_mcp import __version__
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.backends.datadog import DatadogBackend
from opentelemetry_mcp.backends.honeycomb import HoneycombBackend
from opentelemetry_mcp.backends.jaeger import JaegerBackend
from opentelemetry_mcp.backends.newrelic import NewRelicBackend
from opentelemetry_mcp.backends.sentry import SentryBackend
from opentelemetry_mcp.backends.tempo import TempoBackend
from opentelemetry_mcp.backends.traceloop import TraceloopBackend
from opentelemetry_mcp.backends.xray import XRayBackend
from opentelemetry_mcp.config import BackendConfig, ServerConfig
from opentelemetry_mcp.models import (
    CorrelationResult,
    SearchSpansResult,
    SearchTracesResult,
    TraceDetail,
    TriageResult,
)
from opentelemetry_mcp.observability import (
    McpServerTracingMiddleware,
    configure_metrics,
    configure_tracing,
    install_trace_context_log_filter,
)
from opentelemetry_mcp.tools import (
    compare,
    correlate,
    errors,
    expensive_traces,
    investigate,
    list_llm_tools,
    list_models,
    model_stats,
    prompt_versions,
    search,
    search_spans,
    services,
    sessions,
    slow_traces,
    trace,
    triage,
    usage,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# All 19 tools below only ever query trace/span backends and never mutate
# backend state, so the same read-only/destructive/idempotent/open-world
# annotations apply to every one of them. destructiveHint is meaningful
# only when readOnlyHint is false per the MCP spec, so it carries no
# functional weight here - it's set explicitly (rather than left to its
# true-by-default value) because some MCP directories (e.g. OpenAI's
# ChatGPT Apps submission pipeline) require all applicable hints to be
# present as explicit booleans.
_READ_ONLY_TOOL_ANNOTATIONS = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True
)


def _handle_tool_error(tool_name: str, error: Exception) -> NoReturn:
    """Centralized error handler for tool functions.

    Logs the error with traceback, then re-raises it. Letting the exception
    propagate (rather than swallowing it and returning an error-shaped JSON
    string) is what makes the MCP SDK's lowlevel server report the failure as
    CallToolResult(isError=True), per SEP-2140 - a plain successful return
    would leave isError=False even though the tool call failed.

    Args:
        tool_name: Name of the tool that encountered the error
        error: The exception that was raised
    """
    logger.error(f"Error executing {tool_name}: {error}", exc_info=True)
    raise error


# Global backend instance
_backend: BaseBackend | None = None
# Optional second backend for tools/correlate.py's correlate_trace - only
# ever constructed if config.secondary_backend is set (SECONDARY_BACKEND_*).
_secondary_backend: BaseBackend | None = None
_config: ServerConfig | None = None

# Initialize FastMCP server
mcp = FastMCP("tracehub-mcp", version=__version__)


def _clamp_limit(limit: int) -> int:
    """Cap a tool's requested limit at the server-wide ceiling
    (--max-traces-per-query / MAX_TRACES_PER_QUERY), so a single query can
    never fetch more traces/spans than the operator has allowed, regardless
    of what the calling agent requests. _config is only None before startup
    has finished (in practice, _get_backend() already fails first in that
    case), so there is nothing to clamp against yet.
    """
    if _config is None:
        return limit
    return min(limit, _config.max_traces_per_query)


def _build_backend_from_config(backend_config: BackendConfig) -> BaseBackend:
    """Pure backend-type dispatch: BackendConfig -> concrete BaseBackend
    subclass. Shared by both the primary backend (_create_backend, below)
    and the optional secondary backend (_get_secondary_backend) - a
    secondary backend is just another BackendConfig, with no separate
    construction path of its own. See CLAUDE.md's "Adding New Backends"
    checklist - a new backend type only needs one elif branch here.

    Raises:
        ValueError: If backend type is unsupported
    """
    backend: BaseBackend

    if backend_config.type == "jaeger":
        logger.info(f"Initializing Jaeger backend: {backend_config.url}")
        backend = JaegerBackend(
            url=str(backend_config.url),
            api_key=backend_config.api_key,
            timeout=backend_config.timeout,
        )
    elif backend_config.type == "tempo":
        logger.info(f"Initializing Tempo backend: {backend_config.url}")
        backend = TempoBackend(
            url=str(backend_config.url),
            api_key=backend_config.api_key,
            timeout=backend_config.timeout,
            tempo_instance_id=backend_config.tempo_instance_id,
        )
    elif backend_config.type == "traceloop":
        logger.info(f"Initializing Traceloop backend: {backend_config.url}")
        backend = TraceloopBackend(
            url=str(backend_config.url),
            api_key=backend_config.api_key,
            timeout=backend_config.timeout,
            environments=backend_config.environments,
        )
    elif backend_config.type == "datadog":
        logger.info(f"Initializing Datadog backend: {backend_config.url}")
        backend = DatadogBackend(
            url=str(backend_config.url),
            api_key=backend_config.api_key,
            app_key=backend_config.app_key,
            timeout=backend_config.timeout,
        )
    elif backend_config.type == "sentry":
        logger.info(f"Initializing Sentry backend: {backend_config.url}")
        backend = SentryBackend(
            url=str(backend_config.url),
            api_key=backend_config.api_key,
            org_slug=backend_config.sentry_org,
            project_slug=backend_config.sentry_project,
            timeout=backend_config.timeout,
        )
    elif backend_config.type == "xray":
        logger.info(f"Initializing X-Ray backend: region {backend_config.aws_region}")
        backend = XRayBackend(
            url=str(backend_config.url),
            api_key=backend_config.api_key,
            aws_region=backend_config.aws_region,
            timeout=backend_config.timeout,
        )
    elif backend_config.type == "newrelic":
        logger.info(f"Initializing New Relic backend: account {backend_config.newrelic_account_id}")
        backend = NewRelicBackend(
            url=str(backend_config.url),
            api_key=backend_config.api_key,
            account_id=backend_config.newrelic_account_id,
            timeout=backend_config.timeout,
        )
    elif backend_config.type == "honeycomb":
        logger.info(f"Initializing Honeycomb backend: dataset {backend_config.honeycomb_dataset}")
        backend = HoneycombBackend(
            url=str(backend_config.url),
            api_key=backend_config.api_key,
            dataset=backend_config.honeycomb_dataset,
            timeout=backend_config.timeout,
        )
    else:
        raise ValueError(f"Unsupported backend type: {backend_config.type}")

    return backend


def _create_backend(config: ServerConfig) -> BaseBackend:
    """Create the primary backend instance based on configuration.

    Args:
        config: Server configuration

    Returns:
        Backend instance
    """
    backend = _build_backend_from_config(config.backend)
    # Not a constructor parameter (see BaseBackend.__init__'s own comment):
    # several backend subclasses override __init__ with their own named
    # params, so this is applied uniformly here instead.
    backend.slow_request_threshold_ms = config.slow_request_threshold_ms
    backend.configure_query_cache(config.query_cache_ttl_seconds)
    return backend


async def _get_backend() -> BaseBackend:
    """Get or lazily create backend in the current event loop.

    This ensures the backend is always created within FastMCP's event loop,
    avoiding "Event loop is closed" errors from premature initialization.

    Returns:
        Backend instance

    Raises:
        RuntimeError: If server configuration is not set
    """
    global _backend, _config

    if not _config:
        raise RuntimeError("Server configuration not set")

    # Lazily create backend on first use
    if _backend is None:
        logger.info("Creating backend in current event loop")
        _backend = _create_backend(_config)

        # Perform health check on first initialization
        try:
            health = await _backend.health_check()
            logger.info(f"Backend health check: {health}")
            if health.status != "healthy":
                logger.warning("Backend is not healthy, but continuing...")
        except Exception as e:
            logger.error(f"Backend health check failed: {e}")
            logger.warning("Continuing anyway, requests may fail...")

    return _backend


async def _get_secondary_backend() -> BaseBackend:
    """Get or lazily create the optional secondary backend, for
    tools/correlate.py's correlate_trace. Mirrors _get_backend()'s own
    lazy-init-in-the-current-event-loop pattern and non-fatal health check.

    Returns:
        Secondary backend instance

    Raises:
        RuntimeError: If server configuration is not set
        ValueError: If no secondary backend is configured
            (SECONDARY_BACKEND_TYPE/SECONDARY_BACKEND_URL)
    """
    global _secondary_backend, _config

    if not _config:
        raise RuntimeError("Server configuration not set")

    if _config.secondary_backend is None:
        raise ValueError(
            "No secondary backend configured - set SECONDARY_BACKEND_TYPE and "
            "SECONDARY_BACKEND_URL (and any backend-specific fields, e.g. "
            "SECONDARY_BACKEND_SENTRY_ORG) via environment variables to enable "
            "correlate_trace"
        )

    if _secondary_backend is None:
        logger.info("Creating secondary backend in current event loop")
        _secondary_backend = _build_backend_from_config(_config.secondary_backend)
        _secondary_backend.slow_request_threshold_ms = _config.slow_request_threshold_ms
        _secondary_backend.configure_query_cache(_config.query_cache_ttl_seconds)

        try:
            health = await _secondary_backend.health_check()
            logger.info(f"Secondary backend health check: {health}")
            if health.status != "healthy":
                logger.warning("Secondary backend is not healthy, but continuing...")
        except Exception as e:
            logger.error(f"Secondary backend health check failed: {e}")
            logger.warning("Continuing anyway, requests may fail...")

    return _secondary_backend


async def _run_backend_checks(backend: BaseBackend) -> tuple[bool, dict[str, Any]]:
    """Run health_check + list_services against an already-constructed
    backend. Returns (all_ok, details). The caller owns the backend's
    lifecycle (construct/close) - this never closes it, so it is safe to
    call against the server's own cached _get_backend() instance without
    disrupting subsequent real tool calls that share it.
    """
    all_ok = True
    details: dict[str, Any] = {}

    try:
        health = await backend.health_check()
        details["health_check"] = {"status": health.status, "error": health.error}
        if health.status != "healthy":
            all_ok = False
    except Exception as e:
        details["health_check"] = {"status": "error", "error": str(e)}
        all_ok = False

    try:
        services_found = await backend.list_services()
        details["list_services"] = {"count": len(services_found)}
    except Exception as e:
        details["list_services"] = {"error": str(e)}
        all_ok = False

    return all_ok, details


@mcp.custom_route("/health", methods=["GET"])
async def health_route(request: Request) -> Response:
    """Liveness only - deliberately does no backend I/O, so this reflects
    whether the process itself is up, independent of backend reachability.
    Only reachable over the streamable-http transport; stdio mode has no
    HTTP server for this route to attach to."""
    return JSONResponse({"status": "ok"})


@mcp.custom_route("/ready", methods=["GET"])
async def ready_route(request: Request) -> Response:
    """Readiness - reuses the server's own cached _get_backend() instance
    (never constructs or closes a separate one, unlike `doctor`), so this
    can never break a real tool call that shares the same backend."""
    try:
        backend = await _get_backend()
    except Exception as e:
        return JSONResponse({"status": "not_ready", "error": str(e)}, status_code=503)

    all_ok, details = await _run_backend_checks(backend)
    return JSONResponse(
        {"status": "ready" if all_ok else "not_ready", **details},
        status_code=200 if all_ok else 503,
    )


@mcp.tool(title="Search Traces", annotations=_READ_ONLY_TOOL_ANNOTATIONS)
async def search_traces(
    service_name: str | None = None,
    operation_name: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    min_duration_ms: int | None = None,
    max_duration_ms: int | None = None,
    gen_ai_system: str | None = None,
    gen_ai_request_model: str | None = None,
    gen_ai_response_model: str | None = None,
    has_error: bool | None = None,
    tags: dict[str, str] | None = None,
    filters: list[dict[str, Any]] | None = None,
    limit: int = 100,
) -> SearchTracesResult:
    """Search for OpenTelemetry traces with filters.

    Supports both simple parameters and advanced generic filter system.

    Args:
        service_name: Filter by service name (use filters for advanced queries)
        operation_name: Filter by operation/span name
        start_time: Start time in ISO 8601 format (e.g., 2024-01-01T00:00:00Z)
        end_time: End time in ISO 8601 format
        min_duration_ms: Minimum trace duration in milliseconds
        max_duration_ms: Maximum trace duration in milliseconds
        gen_ai_system: Filter by LLM provider (e.g., openai, anthropic)
        gen_ai_request_model: Filter by requested model name (e.g., gpt-4)
        gen_ai_response_model: Filter by actual model used (e.g., gpt-4-0613)
        has_error: Filter traces with errors
        tags: Additional tag filters as key-value pairs
        filters: Generic filter conditions (advanced) - list of filter objects with:
            - field: Field name in dotted notation (e.g., "gen_ai.usage.prompt_tokens")
            - operator: Comparison operator (equals, not_equals, gt, lt, gte, lte, contains,
                       not_contains, starts_with, ends_with, in, not_in, between, exists, not_exists)
            - value: Single value for most operators
            - values: List of values for "in", "not_in", "between" operators
            - value_type: Type of value(s) - "string", "number", or "boolean"
        limit: Maximum number of traces to return (1-1000, default: 100)

    Returns:
        JSON string with search results

    Filter Examples:
        Find expensive traces:
        {"field": "gen_ai.usage.total_tokens", "operator": "gt", "value": 5000, "value_type": "number"}

        Filter by multiple models:
        {"field": "gen_ai.request.model", "operator": "in", "values": ["gpt-4", "claude-3"], "value_type": "string"}

        Check if attribute exists:
        {"field": "gen_ai.request.temperature", "operator": "exists", "value_type": "number"}

        Find streaming requests:
        {"field": "gen_ai.request.is_streaming", "operator": "equals", "value": true, "value_type": "boolean"}
    """
    try:
        backend = await _get_backend()
        result = await search.search_traces(
            backend,
            service_name=service_name,
            operation_name=operation_name,
            start_time=start_time,
            end_time=end_time,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            gen_ai_system=gen_ai_system,
            gen_ai_request_model=gen_ai_request_model,
            gen_ai_response_model=gen_ai_response_model,
            has_error=has_error,
            tags=tags,
            filters=filters,
            limit=_clamp_limit(limit),
        )
        return result
    except Exception as e:
        return _handle_tool_error("search_traces", e)


@mcp.tool(title="Get Trace", annotations=_READ_ONLY_TOOL_ANNOTATIONS)
async def get_trace(
    trace_id: str, detail_level: Literal["summary", "full"] = "full"
) -> TraceDetail:
    """Get complete trace details by trace ID.

    Returns all spans with attributes, including parsed Opentelemetry data for LLM operations.

    Args:
        trace_id: Trace identifier
        detail_level: "full" (default) returns every attribute/event value
            in full, unchanged from this tool's original behavior. "summary"
            elides known-large gen_ai.* fields (input/output messages, system
            instructions, retrieval documents) and truncates long
            event-attribute values, for callers that don't need full
            prompt/completion bodies.

    Returns:
        TraceDetail with complete trace data
    """
    try:
        backend = await _get_backend()
        result = await trace.get_trace(backend, trace_id=trace_id, detail_level=detail_level)
        return result
    except Exception as e:
        return _handle_tool_error("get_trace", e)


@mcp.tool(title="Triage Trace", annotations=_READ_ONLY_TOOL_ANNOTATIONS)
async def triage_trace(
    trace_id: str, detail_level: Literal["summary", "full"] = "summary"
) -> TriageResult:
    """Synthesize a likely-root-cause diagnosis for a trace, instead of
    returning raw trace data for the caller to re-derive one from every time.

    Computes a critical path (the "Last Finishing Child" chain actually
    responsible for the trace's total latency), ranks spans by self-time
    (latency contribution net of children, top 10), and - when the trace
    contains an error anywhere under any root span - identifies the deepest
    error span in the trace's error chain as the likely root cause. Falls
    back to the highest self-time span as a pure-latency diagnosis when no
    error is present. Deterministic (no LLM call); works against any
    configured backend.

    Args:
        trace_id: Trace identifier
        detail_level: "summary" (default) returns a compact diagnosis only
            - this differs from get_trace's own "full"-by-default, since a
            triage result is already a small synthesized diagnosis rather
            than a raw data dump, so there is no unbounded-by-default
            payload to guard against here. "full" additionally attaches
            the diagnosed root cause's raw error detail (message/type/
            stacktrace) to both the verdict and the matching error_chain
            entry, when the verdict is error-driven.

    Returns:
        TriageResult with a verdict, critical path, top 10 latency
        contributors, and (when present) the error chain.
    """
    try:
        backend = await _get_backend()
        result = await triage.triage_trace(backend, trace_id=trace_id, detail_level=detail_level)
        return result
    except Exception as e:
        return _handle_tool_error("triage_trace", e)


@mcp.tool(title="Correlate Trace", annotations=_READ_ONLY_TOOL_ANNOTATIONS)
async def correlate_trace(trace_id: str) -> CorrelationResult:
    """Try to find the corresponding trace in a second, independently-
    configured backend (e.g. a Datadog trace and its downstream Sentry
    error, joined) - given a trace_id known to the primary backend.

    Tries a direct trace_id match in the secondary backend first
    (confidence "high"); if that fails, falls back to a time-window +
    service-name-overlap heuristic search (confidence "low"). This is a
    best-effort correlation, not a guaranteed join - see the always-present
    `limitations` in the result for why. Requires a secondary backend to be
    configured via SECONDARY_BACKEND_TYPE/SECONDARY_BACKEND_URL (and any
    backend-specific fields) environment variables; raises a clear error
    otherwise.

    Args:
        trace_id: Trace identifier, as known to the primary (already
            configured) backend.

    Returns:
        CorrelationResult with every candidate match found and the fixed
        list of limitations that always apply to cross-backend correlation.
    """
    try:
        backend = await _get_backend()
        secondary = await _get_secondary_backend()
        result = await correlate.correlate_trace(backend, secondary, trace_id=trace_id)
        return result
    except Exception as e:
        return _handle_tool_error("correlate_trace", e)


@mcp.tool(title="Get LLM Usage", annotations=_READ_ONLY_TOOL_ANNOTATIONS)
async def get_llm_usage(
    start_time: str | None = None,
    end_time: str | None = None,
    service_name: str | None = None,
    gen_ai_system: str | None = None,
    gen_ai_request_model: str | None = None,
    gen_ai_response_model: str | None = None,
    limit: int = 1000,
) -> str:
    """Get aggregated LLM usage metrics (token counts) for a time period.

    Provides breakdowns by model and service.

    Args:
        start_time: Start time in ISO 8601 format
        end_time: End time in ISO 8601 format
        service_name: Filter by service name
        gen_ai_system: Filter by LLM provider
        gen_ai_request_model: Filter by requested model name
        gen_ai_response_model: Filter by actual model used
        limit: Maximum traces to analyze (default: 1000)

    Returns:
        JSON string with usage metrics
    """
    try:
        backend = await _get_backend()
        result = await usage.get_llm_usage(
            backend,
            start_time=start_time,
            end_time=end_time,
            service_name=service_name,
            gen_ai_system=gen_ai_system,
            gen_ai_request_model=gen_ai_request_model,
            gen_ai_response_model=gen_ai_response_model,
            limit=_clamp_limit(limit),
        )
        return result
    except Exception as e:
        return _handle_tool_error("get_llm_usage", e)


@mcp.tool(title="List Services", annotations=_READ_ONLY_TOOL_ANNOTATIONS)
async def list_services() -> str:
    """List all available services in the OpenTelemetry backend.

    Returns:
        JSON string with list of services
    """
    try:
        backend = await _get_backend()
        result = await services.list_services(backend)
        return result
    except Exception as e:
        return _handle_tool_error("list_services", e)


@mcp.tool(title="Find Errors", annotations=_READ_ONLY_TOOL_ANNOTATIONS)
async def find_errors(
    start_time: str | None = None,
    end_time: str | None = None,
    service_name: str | None = None,
    limit: int = 100,
) -> str:
    """Find traces with errors.

    Including detailed error messages, stack traces, and LLM-specific error information.

    Args:
        start_time: Start time in ISO 8601 format
        end_time: End time in ISO 8601 format
        service_name: Filter by service name
        limit: Maximum error traces to return (default: 100)

    Returns:
        JSON string with error traces
    """
    try:
        backend = await _get_backend()
        result = await errors.find_errors(
            backend,
            start_time=start_time,
            end_time=end_time,
            service_name=service_name,
            limit=_clamp_limit(limit),
        )
        return result
    except Exception as e:
        return _handle_tool_error("find_errors", e)


@mcp.tool(title="List LLM Models", annotations=_READ_ONLY_TOOL_ANNOTATIONS)
async def list_llm_models(
    start_time: str | None = None,
    end_time: str | None = None,
    service_name: str | None = None,
    gen_ai_system: str | None = None,
    limit: int = 1000,
) -> str:
    """List all LLM models being used with usage statistics.

    Discovers what models are deployed and tracks their usage patterns.

    Args:
        start_time: Start time in ISO 8601 format (e.g., 2024-01-01T00:00:00Z)
        end_time: End time in ISO 8601 format
        service_name: Filter by service name
        gen_ai_system: Filter by LLM provider (e.g., openai, anthropic, cohere)
        limit: Maximum traces to analyze for model discovery (default: 1000)

    Returns:
        JSON string with list of models and their statistics (count, request_count, first_seen, last_seen)
    """
    try:
        backend = await _get_backend()
        result = await list_models.list_models(
            backend,
            start_time=start_time,
            end_time=end_time,
            service_name=service_name,
            gen_ai_system=gen_ai_system,
            limit=_clamp_limit(limit),
        )
        return result
    except Exception as e:
        return _handle_tool_error("list_llm_models", e)


@mcp.tool(title="Get LLM Model Stats", annotations=_READ_ONLY_TOOL_ANNOTATIONS)
async def get_llm_model_stats(
    model_name: str,
    start_time: str | None = None,
    end_time: str | None = None,
    service_name: str | None = None,
) -> str:
    """Get detailed performance statistics for a specific LLM model.

    Analyzes request count, latency percentiles (p50, p95, p99), token usage statistics,
    error rates, and finish reason distributions.

    Args:
        model_name: Model name to analyze (e.g., "gpt-4", "claude-3-opus", "gpt-3.5-turbo")
        start_time: Start time in ISO 8601 format (e.g., 2024-01-01T00:00:00Z)
        end_time: End time in ISO 8601 format
        service_name: Filter by service name

    Returns:
        JSON string with comprehensive model statistics including duration/token percentiles
    """
    try:
        backend = await _get_backend()
        result = await model_stats.get_model_stats(
            backend,
            model_name=model_name,
            start_time=start_time,
            end_time=end_time,
            service_name=service_name,
        )
        return result
    except Exception as e:
        return _handle_tool_error("get_llm_model_stats", e)


@mcp.tool(title="List Sessions", annotations=_READ_ONLY_TOOL_ANNOTATIONS)
async def list_sessions(
    start_time: str | None = None,
    end_time: str | None = None,
    service_name: str | None = None,
    gen_ai_system: str | None = None,
    limit: int = 1000,
) -> sessions.ListSessionsResult:
    """List conversations/sessions grouped by gen_ai.conversation.id.

    Groups spans that carry the gen_ai.conversation.id attribute (a real,
    cross-industry OTel semantic convention for session/conversation grouping)
    to surface per-conversation span counts, token usage, and time bounds -
    useful for understanding multi-turn conversation activity.

    Args:
        start_time: Start time in ISO 8601 format (e.g., 2024-01-01T00:00:00Z)
        end_time: End time in ISO 8601 format
        service_name: Filter by service name
        gen_ai_system: Filter by LLM provider (openai, anthropic, etc.)
        limit: Maximum spans to analyze (default: 1000)

    Returns:
        JSON string with list of sessions and their statistics
    """
    try:
        backend = await _get_backend()
        result = await sessions.list_sessions(
            backend,
            start_time=start_time,
            end_time=end_time,
            service_name=service_name,
            gen_ai_system=gen_ai_system,
            limit=_clamp_limit(limit),
        )
        return result
    except Exception as e:
        return _handle_tool_error("list_sessions", e)


@mcp.tool(title="Get Session Stats", annotations=_READ_ONLY_TOOL_ANNOTATIONS)
async def get_session_stats(
    conversation_id: str,
    start_time: str | None = None,
    end_time: str | None = None,
    service_name: str | None = None,
    limit: int = 1000,
) -> str:
    """Get detailed statistics for a single conversation/session.

    Analyzes span count, distinct services, time bounds, LLM request/success/
    error counts, latency percentiles, and token usage for every span sharing
    the given gen_ai.conversation.id.

    Args:
        conversation_id: The gen_ai.conversation.id to analyze
        start_time: Start time in ISO 8601 format (e.g., 2024-01-01T00:00:00Z)
        end_time: End time in ISO 8601 format
        service_name: Filter by service name
        limit: Maximum spans to analyze (default: 1000)

    Returns:
        JSON string with comprehensive session statistics
    """
    try:
        backend = await _get_backend()
        result = await sessions.get_session_stats(
            backend,
            conversation_id=conversation_id,
            start_time=start_time,
            end_time=end_time,
            service_name=service_name,
            limit=_clamp_limit(limit),
        )
        return result
    except Exception as e:
        return _handle_tool_error("get_session_stats", e)


@mcp.tool(title="Compare Time Windows", annotations=_READ_ONLY_TOOL_ANNOTATIONS)
async def compare_time_windows(
    range_a_start: str | None = None,
    range_a_end: str | None = None,
    range_b_start: str | None = None,
    range_b_end: str | None = None,
    service_name: str | None = None,
    gen_ai_system: str | None = None,
    gen_ai_request_model: str | None = None,
    gen_ai_response_model: str | None = None,
    limit: int = 1000,
) -> str:
    """Compare aggregated LLM usage metrics between two time windows.

    Runs the same usage aggregation for both ranges and returns the delta -
    useful for "this week vs last week" or "before/after a deploy" style
    comparisons of request/token counts.

    Args:
        range_a_start: Range A start time in ISO 8601 format
        range_a_end: Range A end time in ISO 8601 format
        range_b_start: Range B start time in ISO 8601 format
        range_b_end: Range B end time in ISO 8601 format
        service_name: Filter by service name (applied to both ranges)
        gen_ai_system: Filter by LLM provider (applied to both ranges)
        gen_ai_request_model: Filter by requested model name (applied to both ranges)
        gen_ai_response_model: Filter by actual model used (applied to both ranges)
        limit: Maximum number of traces to analyze per range (default: 1000)

    Returns:
        JSON string with range_a, range_b, and a delta summarizing the change
    """
    try:
        backend = await _get_backend()
        result = await compare.compare_time_windows(
            backend,
            range_a_start=range_a_start,
            range_a_end=range_a_end,
            range_b_start=range_b_start,
            range_b_end=range_b_end,
            service_name=service_name,
            gen_ai_system=gen_ai_system,
            gen_ai_request_model=gen_ai_request_model,
            gen_ai_response_model=gen_ai_response_model,
            limit=_clamp_limit(limit),
        )
        return result
    except Exception as e:
        return _handle_tool_error("compare_time_windows", e)


@mcp.tool(title="Investigate Cost Spike", annotations=_READ_ONLY_TOOL_ANNOTATIONS)
async def investigate_cost_spike(
    recent_start: str,
    recent_end: str,
    baseline_start: str | None = None,
    baseline_end: str | None = None,
    service_name: str | None = None,
    gen_ai_system: str | None = None,
    gen_ai_request_model: str | None = None,
    gen_ai_response_model: str | None = None,
    limit: int = 1000,
    top_n: int = 5,
) -> str:
    """Investigate an LLM cost spike: compare a recent window against a
    baseline and rank which models/services contributed most to the change.

    On-request/pull-based analysis, not a push alert - mirrors SigNoz's own
    "investigate telemetry cost" skill. Call this when you suspect (or want
    to check for) a cost increase, rather than polling get_llm_usage by hand.

    Args:
        recent_start: Recent window start time in ISO 8601 format
        recent_end: Recent window end time in ISO 8601 format
        baseline_start: Baseline window start (ISO 8601). If omitted along
            with baseline_end, auto-computed as the same duration
            immediately preceding recent_start.
        baseline_end: Baseline window end (ISO 8601)
        service_name: Filter by service name (applied to both windows)
        gen_ai_system: Filter by LLM provider (applied to both windows)
        gen_ai_request_model: Filter by requested model name (applied to both windows)
        gen_ai_response_model: Filter by actual model used (applied to both windows)
        limit: Maximum number of traces to analyze per window (default: 1000)
        top_n: Maximum ranked contributors to return per breakdown (default: 5, max: 50)

    Returns:
        JSON string with recent/baseline usage, a summary_delta, and
        top_model_contributors/top_service_contributors ranked by cost change
    """
    try:
        backend = await _get_backend()
        result = await investigate.investigate_cost_spike(
            backend,
            recent_start=recent_start,
            recent_end=recent_end,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            service_name=service_name,
            gen_ai_system=gen_ai_system,
            gen_ai_request_model=gen_ai_request_model,
            gen_ai_response_model=gen_ai_response_model,
            limit=_clamp_limit(limit),
            top_n=top_n,
        )
        return result
    except Exception as e:
        return _handle_tool_error("investigate_cost_spike", e)


@mcp.tool(title="Investigate Error Spike", annotations=_READ_ONLY_TOOL_ANNOTATIONS)
async def investigate_error_spike(
    recent_start: str,
    recent_end: str,
    baseline_start: str | None = None,
    baseline_end: str | None = None,
    service_name: str | None = None,
    limit: int = 1000,
    top_n: int = 5,
    min_error_count_increase: int = 3,
    rate_multiplier_threshold: float = 2.0,
) -> str:
    """Investigate an error-rate spike: compare a recent window against a
    baseline and rank which services/models/error types contributed most.

    is_spike requires both an absolute error-count floor and a relative
    rate-multiplier to hold, so a tiny sample (e.g. 1 error becoming 2)
    doesn't read as a spike.

    Args:
        recent_start: Recent window start time in ISO 8601 format
        recent_end: Recent window end time in ISO 8601 format
        baseline_start: Baseline window start (ISO 8601). If omitted along
            with baseline_end, auto-computed as the same duration
            immediately preceding recent_start.
        baseline_end: Baseline window end (ISO 8601)
        service_name: Filter by service name (applied to both windows)
        limit: Maximum number of traces to analyze per window (default: 1000)
        top_n: Maximum ranked contributors to return per breakdown (default: 5, max: 50)
        min_error_count_increase: Minimum absolute error-count increase to
            count as a spike (default: 3)
        rate_multiplier_threshold: Minimum error-rate multiplier (recent /
            baseline) to count as a spike (default: 2.0)

    Returns:
        JSON string with recent/baseline error stats, is_spike, and ranked
        top_service_contributors/top_model_contributors/top_error_type_contributors
    """
    try:
        backend = await _get_backend()
        result = await investigate.investigate_error_spike(
            backend,
            recent_start=recent_start,
            recent_end=recent_end,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            service_name=service_name,
            limit=_clamp_limit(limit),
            top_n=top_n,
            min_error_count_increase=min_error_count_increase,
            rate_multiplier_threshold=rate_multiplier_threshold,
        )
        return result
    except Exception as e:
        return _handle_tool_error("investigate_error_spike", e)


@mcp.tool(title="Get Prompt Version Stats", annotations=_READ_ONLY_TOOL_ANNOTATIONS)
async def get_prompt_version_stats(
    start_time: str | None = None,
    end_time: str | None = None,
    service_name: str | None = None,
    gen_ai_system: str | None = None,
    limit: int = 1000,
) -> str:
    """Get aggregated performance stats grouped by prompt name and version.

    Groups spans by gen_ai.prompt.name + gen_ai.prompt.version, mirroring
    Langfuse's shipped per-prompt Metrics tab. Real-world adoption of these
    two attributes is still thin, so this tool may often return an empty
    list until more instrumentations populate them.

    Args:
        start_time: Start time in ISO 8601 format (e.g., 2024-01-01T00:00:00Z)
        end_time: End time in ISO 8601 format
        service_name: Filter by service name
        gen_ai_system: Filter by LLM provider (openai, anthropic, etc.)
        limit: Maximum spans to analyze (default: 1000)

    Returns:
        JSON string with per-(prompt_name, prompt_version) request counts,
        time bounds, and duration/token percentiles
    """
    try:
        backend = await _get_backend()
        result = await prompt_versions.get_prompt_version_stats(
            backend,
            start_time=start_time,
            end_time=end_time,
            service_name=service_name,
            gen_ai_system=gen_ai_system,
            limit=_clamp_limit(limit),
        )
        return result
    except Exception as e:
        return _handle_tool_error("get_prompt_version_stats", e)


@mcp.tool(title="Get LLM Expensive Traces", annotations=_READ_ONLY_TOOL_ANNOTATIONS)
async def get_llm_expensive_traces(
    limit: int = 10,
    start_time: str | None = None,
    end_time: str | None = None,
    min_tokens: int | None = None,
    service_name: str | None = None,
    gen_ai_request_model: str | None = None,
    gen_ai_response_model: str | None = None,
) -> str:
    """Find traces with highest LLM token usage.

    Useful for cost optimization and identifying inefficient prompts.

    Args:
        limit: Maximum number of traces to return (default: 10)
        start_time: Start time in ISO 8601 format (e.g., 2024-01-01T00:00:00Z)
        end_time: End time in ISO 8601 format
        min_tokens: Minimum token count threshold (only return traces above this)
        service_name: Filter by service name
        gen_ai_request_model: Filter by requested model name (e.g., "gpt-4")
        gen_ai_response_model: Filter by actual model used (e.g., "gpt-4-0613")

    Returns:
        JSON string with top N most expensive traces sorted by total token usage
    """
    try:
        backend = await _get_backend()
        result = await expensive_traces.get_expensive_traces(
            backend,
            limit=_clamp_limit(limit),
            start_time=start_time,
            end_time=end_time,
            min_tokens=min_tokens,
            service_name=service_name,
            gen_ai_request_model=gen_ai_request_model,
            gen_ai_response_model=gen_ai_response_model,
        )
        return result
    except Exception as e:
        return _handle_tool_error("get_llm_expensive_traces", e)


@mcp.tool(title="Get LLM Slow Traces", annotations=_READ_ONLY_TOOL_ANNOTATIONS)
async def get_llm_slow_traces(
    limit: int = 10,
    start_time: str | None = None,
    end_time: str | None = None,
    min_duration_ms: int | None = None,
    service_name: str | None = None,
    gen_ai_request_model: str | None = None,
    gen_ai_response_model: str | None = None,
) -> str:
    """Find slowest LLM traces by duration.

    Useful for performance optimization and identifying latency bottlenecks.

    Args:
        limit: Maximum number of traces to return (default: 10)
        start_time: Start time in ISO 8601 format (e.g., 2024-01-01T00:00:00Z)
        end_time: End time in ISO 8601 format
        min_duration_ms: Minimum duration threshold in milliseconds (only return traces above this)
        service_name: Filter by service name
        gen_ai_request_model: Filter by requested model name (e.g., "gpt-4")
        gen_ai_response_model: Filter by actual model used (e.g., "gpt-4-0613")

    Returns:
        JSON string with top N slowest traces sorted by duration
    """
    try:
        backend = await _get_backend()
        result = await slow_traces.get_slow_traces(
            backend,
            limit=_clamp_limit(limit),
            start_time=start_time,
            end_time=end_time,
            min_duration_ms=min_duration_ms,
            service_name=service_name,
            gen_ai_request_model=gen_ai_request_model,
            gen_ai_response_model=gen_ai_response_model,
        )
        return result
    except Exception as e:
        return _handle_tool_error("get_llm_slow_traces", e)


@mcp.tool(title="Search Spans", annotations=_READ_ONLY_TOOL_ANNOTATIONS)
async def search_spans_tool(
    service_name: str | None = None,
    operation_name: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    min_duration_ms: int | None = None,
    max_duration_ms: int | None = None,
    gen_ai_system: str | None = None,
    gen_ai_request_model: str | None = None,
    gen_ai_response_model: str | None = None,
    has_error: bool | None = None,
    tags: dict[str, str] | None = None,
    filters: list[dict[str, Any]] | None = None,
    limit: int = 100,
) -> SearchSpansResult:
    """Search for individual OpenTelemetry spans with optional filters.

    Unlike search_traces, this returns individual spans rather than grouped traces,
    which is useful for analyzing specific operations or finding spans with certain
    characteristics (e.g., LLM tool calls with traceloop.span.kind == tool).

    Args:
        service_name: Filter by service name
        operation_name: Filter by operation/span name
        start_time: Start time in ISO 8601 format (e.g., 2024-01-01T00:00:00Z)
        end_time: End time in ISO 8601 format
        min_duration_ms: Minimum span duration in milliseconds
        max_duration_ms: Maximum span duration in milliseconds
        gen_ai_system: Filter by LLM provider (e.g., openai, anthropic)
        gen_ai_request_model: Filter by requested model name (e.g., "gpt-4")
        gen_ai_response_model: Filter by actual model used (e.g., "gpt-4-0613")
        has_error: Filter spans with errors
        tags: Additional tag filters as key-value pairs
        filters: Generic filter conditions - list of filter objects with:
            - field: Field name in dotted notation (e.g., "traceloop.span.kind")
            - operator: Comparison operator
            - value: Single value for most operators
            - values: List of values for "in", "not_in", "between" operators
            - value_type: Type of value(s) - "string", "number", or "boolean"
        limit: Maximum number of spans to return (1-1000, default: 100)

    Returns:
        JSON string with span summaries

    Example filter to find LLM tool calls:
        {"field": "traceloop.span.kind", "operator": "equals", "value": "tool", "value_type": "string"}
    """
    try:
        backend = await _get_backend()
        result = await search_spans.search_spans(
            backend,
            service_name=service_name,
            operation_name=operation_name,
            start_time=start_time,
            end_time=end_time,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            gen_ai_system=gen_ai_system,
            gen_ai_request_model=gen_ai_request_model,
            gen_ai_response_model=gen_ai_response_model,
            has_error=has_error,
            tags=tags,
            filters=filters,
            limit=_clamp_limit(limit),
        )
        return result
    except Exception as e:
        return _handle_tool_error("search_spans_tool", e)


@mcp.tool(title="List LLM Tools", annotations=_READ_ONLY_TOOL_ANNOTATIONS)
async def list_llm_tools_tool(
    start_time: str | None = None,
    end_time: str | None = None,
    service_name: str | None = None,
    gen_ai_system: str | None = None,
    limit: int = 1000,
) -> str:
    """List all LLM tools being used by identifying traceloop.span.kind == tool.

    Discovers which tools/functions LLM applications are calling, grouped by tool name
    with usage statistics.

    Args:
        start_time: Start time in ISO 8601 format (e.g., 2024-01-01T00:00:00Z)
        end_time: End time in ISO 8601 format
        service_name: Filter by service name
        gen_ai_system: Filter by LLM provider (openai, anthropic, etc.)
        limit: Maximum spans to analyze (default: 1000)

    Returns:
        JSON string with list of tools and their statistics (usage count, services, first/last seen)
    """
    try:
        backend = await _get_backend()
        result = await list_llm_tools.list_llm_tools(
            backend,
            start_time=start_time,
            end_time=end_time,
            service_name=service_name,
            gen_ai_system=gen_ai_system,
            limit=_clamp_limit(limit),
        )
        return result
    except Exception as e:
        return _handle_tool_error("list_llm_tools_tool", e)


# Every @mcp.tool()-registered function name above. FastMCP's own tool
# storage is async-only (no synchronous listing API), so this is
# maintained by hand - keep it in sync with the registrations above.
_ALL_TOOL_NAMES = frozenset(
    {
        "search_traces",
        "get_trace",
        "get_llm_usage",
        "list_services",
        "find_errors",
        "list_llm_models",
        "get_llm_model_stats",
        "list_sessions",
        "get_session_stats",
        "compare_time_windows",
        "investigate_cost_spike",
        "investigate_error_spike",
        "get_prompt_version_stats",
        "get_llm_expensive_traces",
        "get_llm_slow_traces",
        "search_spans_tool",
        "list_llm_tools_tool",
        "triage_trace",
        "correlate_trace",
    }
)


def _apply_tool_gating(disable_tools: str | None, enabled_tools: str | None) -> None:
    """Remove tools from this server instance per --enabled-tools (allowlist,
    applied first) and --disable-tools (removed on top of whatever the
    allowlist kept), for reduced-trust or multi-tenant deployments.

    Uses FastMCP's own mcp.local_provider.remove_tool() rather than
    hand-rolling tool exclusion.
    """
    if enabled_tools:
        keep = {name.strip() for name in enabled_tools.split(",") if name.strip()}
        unknown = keep - _ALL_TOOL_NAMES
        if unknown:
            raise ValueError(f"Unknown tool name(s) in --enabled-tools: {sorted(unknown)}")
        for name in _ALL_TOOL_NAMES - keep:
            mcp.local_provider.remove_tool(name)
        logger.info(f"Tool allowlist applied via --enabled-tools: {sorted(keep)}")

    if disable_tools:
        disable = {name.strip() for name in disable_tools.split(",") if name.strip()}
        unknown = disable - _ALL_TOOL_NAMES
        if unknown:
            raise ValueError(f"Unknown tool name(s) in --disable-tools: {sorted(unknown)}")
        for name in disable:
            try:
                mcp.local_provider.remove_tool(name)
            except KeyError:
                pass  # already removed by --enabled-tools above
        logger.info(f"Tools disabled via --disable-tools: {sorted(disable)}")


_LOCAL_ORIGIN_PATTERN = re.compile(r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$", re.IGNORECASE)


class OriginValidationMiddleware(BaseHTTPMiddleware):
    """Validate the Origin header on requests to the streamable-http transport.

    fastmcp 3.2.0 built its StreamableHTTPSessionManager without passing
    security_settings through at all, leaving the upstream mcp SDK's own
    DNS-rebinding/Origin protection (mcp.server.transport_security) disabled
    by omission. Re-verified against the real installed fastmcp 4.0.5
    (fastmcp/server/http.py:668-682, not assumed unchanged from 3.2.0): the
    gap is now an explicit, unconditional
    `security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False)`
    - fastmcp's own comment there says it "owns DNS-rebinding protection via
    HostOriginGuardMiddleware... always disable the SDK's own protection so
    the two layers don't double-block." That replacement,
    HostOriginGuardMiddleware, is real but stays opt-in
    (`host_origin_protection: HostOriginProtection = False` by default,
    fastmcp/server/http.py:556) and this codebase's own
    `mcp.http_app(transport=..., middleware=[...])` call never sets it - so
    fastmcp's own native alternative is never actually inserted for this
    deployment, and the underlying SDK protection stays disabled either way.
    This middleware remains fully necessary, unchanged, satisfying the MCP
    spec's (2025-06-18 basic/transports) MUST-requirement for Streamable HTTP
    servers to validate the Origin header regardless of which fastmcp
    version is running.

    A request with no Origin header is allowed through unmodified: Origin can
    be absent for same-origin requests, and most MCP HTTP clients are not
    browsers and never send one in normal use. A request that does send an
    Origin header is only allowed through if it points at a local dev origin
    (127.0.0.1 or localhost, any scheme/port) - blocking the actual attack
    this check exists for: a malicious webpage running in a victim's browser
    using DNS rebinding or a crafted fetch() to reach a locally-bound
    tracehub-mcp HTTP server.

    No separate CORS middleware is configured alongside this Origin
    allowlist. That is deliberate, not an oversight: the documented use
    case for the streamable-http transport (see CLAUDE.md) is non-browser
    MCP clients - CLI tools, agents, and server-to-server callers - which
    never send preflight requests and are unaffected by CORS either way.
    A browser-based MCP client attempting direct cross-origin access would
    be blocked by the browser's own CORS enforcement (no
    Access-Control-Allow-Origin is ever sent), same-origin exemptions
    aside. Do not "fix" this by adding a permissive CORS layer or loosening
    _LOCAL_ORIGIN_PATTERN above - either would widen exactly the
    DNS-rebinding/cross-origin attack surface this middleware exists to
    close.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        origin = request.headers.get("origin")
        if origin and not _LOCAL_ORIGIN_PATTERN.match(origin):
            logger.warning(f"Invalid Origin header: {origin}")
            return Response("Invalid Origin header", status_code=403)
        return await call_next(request)


class _FixedWindowRateLimiter:
    """Fixed-window (start_time, count) rate limiter keyed by an arbitrary
    string, guarded by an asyncio.Lock. No rate-limiting dependency exists
    in this repo (slowapi/asgi-ratelimit/redis/limits: zero matches), so
    this is hand-rolled to match backends/base.py's own precedent of
    hand-rolling retry/SSRF-guarding rather than adding a library.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._clock = clock
        self._buckets: dict[str, tuple[float, int]] = {}
        self._lock = asyncio.Lock()
        self._last_swept = self._clock()

    def _sweep_expired_locked(self, now: float) -> None:
        """Drop bucket entries whose window has already expired. Caller
        must already hold self._lock.

        Without this, _buckets is never evicted or pruned: every distinct
        key (client IP - rate limiting is on by default) that ever hits
        the server adds a permanent entry that's never removed, even long
        after its own window expired - an unbounded-growth risk for a
        long-running process, or trivially for an attacker/bot rotating
        source IPs. Mirrors backends/cache.py's TTLCoalescingCache's own
        `_sweep_expired_locked`: amortized rather than per-call, running at
        most once per window_seconds rather than on every hit(), so this
        stays O(1) amortized rather than O(len(_buckets)) per call.

        Known, accepted limitation (see RateLimitMiddleware's own
        docstring for the full rationale): this sweep - like the limiter
        as a whole - is keyed on request.client.host only. A source
        rotating its IP on every request always starts a brand-new,
        never-before-seen bucket at count 0, so the per-IP cap never
        engages against that source's aggregate volume. That is a
        structural limitation of pure per-IP fixed-window limiting, not
        something this sweep (or a bigger one) can fix - it would need
        additional infra (e.g. a WAF) to address.
        """
        if now - self._last_swept < self._window_seconds:
            return
        self._last_swept = now
        expired = [
            key
            for key, (window_start, _) in self._buckets.items()
            if now - window_start >= self._window_seconds
        ]
        for key in expired:
            del self._buckets[key]

    async def hit(self, key: str) -> bool:
        """Record one hit for `key`. Returns True if within the allowed
        limit, False if this hit breaches it (caller should reject)."""
        now = self._clock()
        async with self._lock:
            self._sweep_expired_locked(now)
            window_start, count = self._buckets.get(key, (now, 0))
            if now - window_start >= self._window_seconds:
                window_start, count = now, 0
            count += 1
            self._buckets[key] = (window_start, count)
            return count <= self._max_requests


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-client-IP fixed-window rate limiter for the streamable-http
    transport. Bare 429 on breach.

    Known v1 limitation: keys on request.client.host only, with no
    X-Forwarded-For parsing - this codebase has no trusted-proxy allowlist
    anywhere to validate that header against, and trusting a client-
    supplied header without one would make the limiter trivially
    bypassable. A direct consequence: a source rotating its IP on every
    request is never capped, since each never-before-seen IP always
    starts its own bucket at count 0 - the per-IP cap only ever bounds a
    single, stable IP's own volume, never a rotating source's aggregate
    volume. This is a known, accepted structural limitation of pure
    per-IP fixed-window limiting, not something fixable at this layer -
    mitigating it would need additional infra (e.g. a WAF) in front of
    this server.
    """

    def __init__(
        self,
        app: Any,
        max_requests: int = 100,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(app)
        self._limiter = _FixedWindowRateLimiter(max_requests, window_seconds, clock)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        if not await self._limiter.hit(client_ip):
            logger.warning(f"Rate limit exceeded for client {client_ip}")
            return Response("Rate limit exceeded", status_code=429)
        return await call_next(request)


_BACKEND_CLOSE_TIMEOUT_SECONDS = 5.0


async def _drain_and_close_backend(drain_seconds: float, close_timeout_seconds: float) -> None:
    """Runs from inside the *inner* ASGI lifespan.shutdown callback that
    _install_shutdown_drain splices onto mcp.http_app()'s app - i.e. from
    inside uvicorn.Server.shutdown(), strictly before capture_signals()'s
    post-serve() finally block restores the default SIGTERM handler and
    re-raises the captured signal (uvicorn/server.py). That ordering is
    what makes this fire reliably on a real SIGTERM, unlike a
    FastMCP(lifespan=...) constructor-kwarg attempt (tried and reverted),
    whose teardown was deferred until `await server.serve()` itself
    returned - which never happens on a real SIGTERM.

    Sleeps `drain_seconds` (uvicorn has already finished draining
    in-flight connections/tasks by this point - this is a final grace
    window, not a substitute for that phase), then closes the shared
    backend HTTP client(s) - primary and, if one was ever lazily created,
    the optional secondary backend too - if ever lazily created. Each
    close is independently bounded by `close_timeout_seconds` because
    uvicorn places no timeout of its own around the lifespan.shutdown
    wait - an unbounded backend.close() could otherwise hang process exit
    forever, and a hanging primary must not also starve the secondary's
    own close attempt. Threaded as a parameter (default
    _BACKEND_CLOSE_TIMEOUT_SECONDS) rather than read directly from the
    module constant so a real signal-timing integration test can force the
    timeout-fallback branch deterministically without monkeypatching module
    internals from outside a subprocess, which is impossible.
    """
    global _backend, _secondary_backend

    if drain_seconds > 0:
        logger.info(
            f"HTTP transport shutting down - draining for {drain_seconds}s "
            "before closing the backend client(s)"
        )
        await asyncio.sleep(drain_seconds)

    backends_to_close = [b for b in (_backend, _secondary_backend) if b is not None]
    _backend, _secondary_backend = None, None

    if not backends_to_close:
        logger.info("HTTP transport shutdown: no backend was ever created, nothing to close")
        return

    async def _close_one(backend_to_close: BaseBackend) -> None:
        try:
            await asyncio.wait_for(backend_to_close.close(), timeout=close_timeout_seconds)
            logger.info("Backend closed during shutdown drain")
        except TimeoutError:
            logger.warning(
                f"Backend close did not finish within {close_timeout_seconds}s "
                "during shutdown, abandoning it"
            )
        except Exception as e:
            logger.warning(f"Error closing backend during shutdown: {e}")

    # Concurrent, not sequential: a hanging primary's close() must not
    # delay the secondary's own close attempt by the full
    # close_timeout_seconds on top of its own - each is bounded
    # independently, so total shutdown latency stays bounded by a single
    # close_timeout_seconds regardless of how many backends were created.
    await asyncio.gather(*(_close_one(b) for b in backends_to_close))


def _install_shutdown_drain(
    app: StarletteWithLifespan, drain_seconds: float, close_timeout_seconds: float
) -> None:
    """Splice _drain_and_close_backend onto the *inner* lifespan
    mcp.http_app() builds, by reassigning the mutable
    app.router.lifespan_context attribute after construction - rather
    than passing FastMCP(lifespan=...) at the constructor.
    FastMCP._lifespan_manager() is ref-counted/reentrant; a
    constructor-level lifespan's teardown only runs on the ref_count==0
    transition, which on the HTTP path used to be the *outer* entrant
    wrapping the whole `await server.serve()` call in run_http_async - and
    that call never returns on a real SIGTERM (see
    _drain_and_close_backend's docstring). Splicing here means
    FastMCP._lifespan_manager() is entered exactly once, by this inner
    lifespan (since main() no longer calls mcp.run()/run_http_async() for
    the HTTP path at all), so there is no second entrant to race with.

    Starlette (this installed version, 1.3.1) has no on_shutdown/
    add_event_handler API - lifespan= is the only mechanism, and
    StarletteWithLifespan.lifespan is literally
    `self.router.lifespan_context`, read fresh on every ASGI lifespan
    dispatch - so this reassignment is guaranteed to take effect.
    """
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def combined_lifespan(app: Starlette) -> AsyncIterator[None]:
        async with original_lifespan(app):
            try:
                yield
            finally:
                await _drain_and_close_backend(drain_seconds, close_timeout_seconds)

    app.router.lifespan_context = combined_lifespan


def _print_config_backend_dict(backend_config: BackendConfig) -> dict[str, Any]:
    """Shape shared by --print-config's "backend"/"secondary_backend" keys.
    Built by hand, not model_dump(): sentry_org/sentry_project/
    tempo_instance_id are marked exclude=True on BackendConfig even though
    they are not secrets (only api_key/app_key are)."""
    return {
        "type": backend_config.type,
        "url": str(backend_config.url),
        "environments": backend_config.environments,
        "timeout": backend_config.timeout,
        "sentry_org": backend_config.sentry_org,
        "sentry_project": backend_config.sentry_project,
        "tempo_instance_id": backend_config.tempo_instance_id,
        "aws_region": backend_config.aws_region,
        "newrelic_account_id": backend_config.newrelic_account_id,
        "honeycomb_dataset": backend_config.honeycomb_dataset,
        "api_key_set": backend_config.api_key is not None,
        "app_key_set": backend_config.app_key is not None,
    }


def _backend_config_options(f: Any) -> Any:
    """Shared backend-override flags, applied to both main (serve) and
    doctor, so an operator can validate a candidate config with doctor
    before committing to it via the same flags main accepts."""
    f = click.option(
        "--backend",
        type=click.Choice(
            ["jaeger", "tempo", "traceloop", "datadog", "sentry", "xray", "newrelic", "honeycomb"]
        ),
        help="Backend type (overrides BACKEND_TYPE env var)",
    )(f)
    f = click.option(
        "--url",
        type=str,
        help="Backend URL (overrides BACKEND_URL env var)",
    )(f)
    f = click.option(
        "--api-key",
        type=str,
        help="API key for backend authentication (overrides BACKEND_API_KEY env var)",
    )(f)
    f = click.option(
        "--app-key",
        type=str,
        help="Application key, required by the Datadog backend in addition to "
        "--api-key (overrides BACKEND_APP_KEY env var)",
    )(f)
    f = click.option(
        "--tempo-instance-id",
        type=str,
        help="Grafana Cloud stack/instance ID, used for Basic Auth with --api-key "
        "instead of Bearer auth (Tempo backend only, required for Grafana "
        "Cloud-hosted Tempo, not needed for self-hosted Tempo; overrides "
        "BACKEND_TEMPO_INSTANCE_ID env var)",
    )(f)
    f = click.option(
        "--sentry-org",
        type=str,
        help="Sentry organization slug, required by the Sentry backend "
        "(overrides BACKEND_SENTRY_ORG env var)",
    )(f)
    f = click.option(
        "--sentry-project",
        type=str,
        help="Sentry project slug, optional for the Sentry backend "
        "(overrides BACKEND_SENTRY_PROJECT env var)",
    )(f)
    f = click.option(
        "--environments",
        type=str,
        help="Comma-separated list of environments for Traceloop backend "
        "(overrides BACKEND_ENVIRONMENTS env var)",
    )(f)
    f = click.option(
        "--aws-region",
        type=str,
        help="AWS region, required by the X-Ray backend (overrides BACKEND_AWS_REGION env var)",
    )(f)
    f = click.option(
        "--newrelic-account-id",
        type=str,
        help="New Relic account ID, required by the New Relic backend since NerdGraph "
        "queries are user-scoped not account-scoped (overrides "
        "BACKEND_NEWRELIC_ACCOUNT_ID env var)",
    )(f)
    f = click.option(
        "--honeycomb-dataset",
        type=str,
        help="Honeycomb dataset slug, required by the Honeycomb backend since the "
        "Query Data API is dataset-scoped (overrides BACKEND_HONEYCOMB_DATASET env var)",
    )(f)
    return f


@click.group(invoke_without_command=True)
@click.pass_context
@_backend_config_options
@click.option(
    "--print-config",
    is_flag=True,
    default=False,
    help="Print the resolved configuration as JSON and exit without starting "
    "the server. Secrets (api_key/app_key) are reported as booleans "
    "(*_set), never their actual value.",
)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "http"]),
    default="stdio",
    envvar="MCP_TRANSPORT",
    help="Transport type: stdio (default) for local/Claude Desktop, http for "
    "network access (overrides MCP_TRANSPORT env var)",
)
@click.option(
    "--host",
    type=str,
    default="0.0.0.0",  # noqa: S104 - HTTP transport is documented for network/Docker deployment, where binding only to loopback would make the exposed port unreachable; pass --host 127.0.0.1 explicitly for a loopback-only server.
    envvar="MCP_HOST",
    help="Host to bind HTTP server to (only for --transport http, default: "
    "0.0.0.0, overrides MCP_HOST env var)",
)
@click.option(
    "--port",
    type=int,
    default=8000,
    envvar="MCP_PORT",
    help="Port for HTTP server (only for --transport http, default: 8000, "
    "overrides MCP_PORT env var)",
)
@click.option(
    "--include-args-in-spans",
    is_flag=True,
    default=False,
    envvar="MCP_INCLUDE_ARGS_IN_SPANS",
    help="Include tool call arguments/results as span attributes when OTel "
    "self-instrumentation is enabled (default: False, since these may "
    "contain sensitive data - overrides MCP_INCLUDE_ARGS_IN_SPANS env var)",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=None,
    help="Logging level (overrides LOG_LEVEL env var, default: INFO)",
)
@click.option(
    "--max-traces-per-query",
    type=click.IntRange(1, 1000),
    default=None,
    help="Maximum traces returned per query, 1-1000 (overrides "
    "MAX_TRACES_PER_QUERY env var, default: 500)",
)
@click.option(
    "--disable-tools",
    type=str,
    default=None,
    help="Comma-separated tool names to remove from this server instance "
    "(e.g. for reduced-trust deployments). Applied after --enabled-tools.",
)
@click.option(
    "--enabled-tools",
    type=str,
    default=None,
    help="Comma-separated allowlist of tool names - every other tool is "
    "removed from this server instance. Combine with --disable-tools to "
    "further narrow the allowlist.",
)
@click.option(
    "--slow-request-threshold-ms",
    type=float,
    default=None,
    help="Log a warning when a backend request takes longer than this many "
    "milliseconds, independent of --log-level (unset: disabled)",
)
@click.option(
    "--rate-limit-max-requests",
    type=int,
    default=100,
    envvar="RATE_LIMIT_MAX_REQUESTS",
    help="Max requests per client IP per --rate-limit-window-seconds on the "
    "HTTP transport (only for --transport http, default: 100, set to 0 "
    "to disable, overrides RATE_LIMIT_MAX_REQUESTS env var)",
)
@click.option(
    "--rate-limit-window-seconds",
    type=float,
    default=60.0,
    envvar="RATE_LIMIT_WINDOW_SECONDS",
    help="Fixed window size in seconds for --rate-limit-max-requests (only "
    "for --transport http, default: 60.0, overrides "
    "RATE_LIMIT_WINDOW_SECONDS env var)",
)
@click.option(
    "--query-cache-ttl-seconds",
    type=float,
    default=None,
    help="Cache backend query results (search_traces/search_spans/get_trace/"
    "list_services/get_service_operations) for this many seconds, with "
    "in-flight request coalescing (unset: disabled, overrides "
    "QUERY_CACHE_TTL_SECONDS env var)",
)
@click.option(
    "--shutdown-drain-seconds",
    type=float,
    default=0.0,
    envvar="SHUTDOWN_DRAIN_SECONDS",
    help="On HTTP transport, wait this many seconds inside the ASGI shutdown "
    "handler - after uvicorn has already finished draining in-flight "
    "connections - before closing the shared backend HTTP client (only "
    "for --transport http, default: 0.0/disabled, overrides "
    "SHUTDOWN_DRAIN_SECONDS env var)",
)
@click.option(
    "--backend-close-timeout-seconds",
    type=float,
    default=_BACKEND_CLOSE_TIMEOUT_SECONDS,
    envvar="BACKEND_CLOSE_TIMEOUT_SECONDS",
    help="On HTTP transport, abandon closing the shared backend HTTP client "
    "during shutdown if it does not finish within this many seconds - "
    "uvicorn places no timeout of its own around this wait (only for "
    f"--transport http, default: {_BACKEND_CLOSE_TIMEOUT_SECONDS}, "
    "overrides BACKEND_CLOSE_TIMEOUT_SECONDS env var)",
)
@click.option(
    "--graceful-shutdown-timeout-seconds",
    type=int,
    default=2,
    envvar="GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS",
    help="On HTTP transport, uvicorn's own bound on waiting for in-flight "
    "connections/tasks to finish on shutdown before cancelling them - "
    "passed straight through as uvicorn.Config(timeout_graceful_shutdown=), "
    "which only accepts whole seconds (only for --transport http, "
    "default: 2, overrides GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS env var)",
)
def main(
    ctx: click.Context,
    backend: str | None,
    url: str | None,
    api_key: str | None,
    app_key: str | None,
    tempo_instance_id: str | None,
    sentry_org: str | None,
    sentry_project: str | None,
    environments: str | None,
    aws_region: str | None,
    newrelic_account_id: str | None,
    honeycomb_dataset: str | None,
    print_config: bool,
    transport: str,
    host: str,
    port: int,
    include_args_in_spans: bool,
    log_level: str | None,
    max_traces_per_query: int | None,
    disable_tools: str | None,
    enabled_tools: str | None,
    slow_request_threshold_ms: float | None,
    rate_limit_max_requests: int,
    rate_limit_window_seconds: float,
    query_cache_ttl_seconds: float | None,
    shutdown_drain_seconds: float,
    backend_close_timeout_seconds: float,
    graceful_shutdown_timeout_seconds: int,
) -> None:
    """Opentelemetry MCP Server - Query OpenTelemetry traces from LLM applications.

    Supports multiple backends: Jaeger, Tempo, Traceloop, Datadog, Sentry, AWS X-Ray,
    New Relic, and Honeycomb.
    Configuration can be provided via environment variables or CLI arguments.

    Transport options:
      - stdio (default): Standard input/output for local use (Claude Desktop)
      - http: HTTP server for network access (remote clients)

    Examples:
      # Run with stdio transport (default, for Claude Desktop)
      tracehub-mcp --backend traceloop

      # Run with HTTP transport for network access
      tracehub-mcp --transport http --port 8000

      # Run with HTTP on specific host/port
      tracehub-mcp --transport http --host 127.0.0.1 --port 9000

      # Validate a candidate config without starting the server
      tracehub-mcp doctor --backend jaeger --url http://localhost:16686

      # Print the resolved config as JSON (secrets redacted to booleans)
      tracehub-mcp --print-config
    """
    if ctx.invoked_subcommand is not None:
        return

    global _config

    try:
        # Load configuration from environment
        _config = ServerConfig.from_env()

        # Set logging level
        logging.getLogger().setLevel(_config.log_level)

        # Apply CLI overrides
        if (
            backend
            or url
            or api_key
            or app_key
            or tempo_instance_id
            or sentry_org
            or sentry_project
            or environments
            or aws_region
            or newrelic_account_id
            or honeycomb_dataset
            or log_level
            or max_traces_per_query is not None
            or slow_request_threshold_ms is not None
            or query_cache_ttl_seconds is not None
        ):
            _config.apply_cli_overrides(
                backend_type=backend,
                backend_url=url,
                api_key=api_key,
                app_key=app_key,
                sentry_org=sentry_org,
                sentry_project=sentry_project,
                tempo_instance_id=tempo_instance_id,
                aws_region=aws_region,
                newrelic_account_id=newrelic_account_id,
                honeycomb_dataset=honeycomb_dataset,
                environments=environments,
                log_level=log_level,
                max_traces_per_query=max_traces_per_query,
                slow_request_threshold_ms=slow_request_threshold_ms,
                query_cache_ttl_seconds=query_cache_ttl_seconds,
            )
            logging.getLogger().setLevel(_config.log_level)

        if print_config:
            # Built by hand, not model_dump(): sentry_org/sentry_project/
            # tempo_instance_id are marked exclude=True on BackendConfig
            # even though they are not secrets (only api_key/app_key are),
            # and transport/host/port/tool-gating are never stored on the
            # config model at all - they only ever exist as this
            # invocation's own CLI args.
            resolved = {
                "backend": _print_config_backend_dict(_config.backend),
                "secondary_backend": (
                    _print_config_backend_dict(_config.secondary_backend)
                    if _config.secondary_backend is not None
                    else None
                ),
                "log_level": _config.log_level,
                "max_traces_per_query": _config.max_traces_per_query,
                "slow_request_threshold_ms": _config.slow_request_threshold_ms,
                "query_cache_ttl_seconds": _config.query_cache_ttl_seconds,
                "transport": transport,
                "host": host,
                "port": port,
                "include_args_in_spans": include_args_in_spans,
                "disable_tools": disable_tools,
                "enabled_tools": enabled_tools,
                "rate_limit_max_requests": rate_limit_max_requests,
                "rate_limit_window_seconds": rate_limit_window_seconds,
                "shutdown_drain_seconds": shutdown_drain_seconds,
                "backend_close_timeout_seconds": backend_close_timeout_seconds,
                "graceful_shutdown_timeout_seconds": graceful_shutdown_timeout_seconds,
            }
            click.echo(json.dumps(resolved, indent=2))
            return

        # Backend will be lazily initialized on first tool call
        # This ensures it's created in FastMCP's event loop, not a separate one

        _apply_tool_gating(disable_tools=disable_tools, enabled_tools=enabled_tools)

        # OTel self-instrumentation is fully opt-in: configure_tracing()/
        # configure_metrics() only return True when OTEL_EXPORTER_OTLP_ENDPOINT
        # is actually set, so there is zero overhead and no dependency on a
        # collector for anyone who has not opted in. configure_metrics()
        # runs before the middleware is constructed so metrics.get_meter()
        # resolves against the real provider immediately, mirroring the
        # existing tracer-resolution ordering.
        tracing_enabled = configure_tracing()
        configure_metrics()
        if tracing_enabled:
            mcp.add_middleware(McpServerTracingMiddleware(include_args=include_args_in_spans))
            install_trace_context_log_filter()

        # Run server with selected transport
        if transport == "http":
            logger.info(f"Starting MCP server with HTTP transport on {host}:{port}")
            logger.info("Using streamable-http transport for better compatibility")
            logger.info(f"Connect clients to: http://{host}:{port}/mcp")
            middleware = [Middleware(OriginValidationMiddleware)]
            if rate_limit_max_requests > 0:
                middleware.append(
                    Middleware(
                        RateLimitMiddleware,
                        max_requests=rate_limit_max_requests,
                        window_seconds=rate_limit_window_seconds,
                    )
                )
            app = mcp.http_app(transport="streamable-http", middleware=middleware)
            _install_shutdown_drain(
                app,
                drain_seconds=shutdown_drain_seconds,
                close_timeout_seconds=backend_close_timeout_seconds,
            )
            uvicorn_config = uvicorn.Config(
                app,
                host=host,
                port=port,
                lifespan="on",
                # Preserves run_http_async's own default (lost by bypassing
                # it) for the connection/task-draining wait uvicorn does
                # BEFORE dispatching lifespan.shutdown - unrelated to the
                # drain feature itself, which fires after this. Exposed as
                # --graceful-shutdown-timeout-seconds so a real signal-timing
                # test can widen it deterministically instead of relying on
                # this project's own default margin.
                timeout_graceful_shutdown=graceful_shutdown_timeout_seconds,
            )
            asyncio.run(uvicorn.Server(uvicorn_config).serve())
        else:
            logger.info(
                f"Starting MCP server with stdio transport using Backend: {_config.backend.type} connected to: {_config.backend.url}"
            )
            mcp.run(transport="stdio")

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


def _print_backend_check_results(details: dict[str, Any], *, prefix: str = "") -> None:
    """Prints doctor's [OK]/[FAIL] health-check + connectivity-probe lines
    for one backend's already-collected _run_backend_checks() details.
    `prefix` (e.g. "Secondary ") distinguishes the secondary backend's own
    lines from the primary's when both are checked in one doctor run."""
    health = details["health_check"]
    if health.get("status") == "healthy":
        click.secho(f"[OK] {prefix}Health check: {health['status']}", fg="green")
    elif health.get("status") == "error":
        click.secho(f"[FAIL] {prefix}Health check raised: {health['error']}", fg="red")
    else:
        click.secho(
            f"[FAIL] {prefix}Health check: {health.get('status')} ({health.get('error')})",
            fg="red",
        )

    list_services_info = details["list_services"]
    if "error" in list_services_info:
        click.secho(
            f"[FAIL] {prefix}Connectivity probe (list_services): {list_services_info['error']}",
            fg="red",
        )
    else:
        click.secho(
            f"[OK] {prefix}Connectivity probe (list_services): "
            f"{list_services_info['count']} service(s)",
            fg="green",
        )


@main.command("doctor")
@_backend_config_options
def doctor(
    backend: str | None,
    url: str | None,
    api_key: str | None,
    app_key: str | None,
    tempo_instance_id: str | None,
    sentry_org: str | None,
    sentry_project: str | None,
    environments: str | None,
    aws_region: str | None,
    newrelic_account_id: str | None,
    honeycomb_dataset: str | None,
) -> None:
    """Run startup diagnostics against the resolved backend config: config
    load, backend construction, health check, and a live read-only
    connectivity probe (list_services). Exits non-zero if any step fails.
    Also validates the optional secondary backend (SECONDARY_BACKEND_*,
    used by correlate_trace) the same way, if one is configured.

    Unlike the server's own lazy backend initialization (which deliberately
    swallows a failed health check and keeps running so requests may still
    work later), doctor surfaces every failure explicitly.
    """
    try:
        config = ServerConfig.from_env()
        if (
            backend
            or url
            or api_key
            or app_key
            or tempo_instance_id
            or sentry_org
            or sentry_project
            or environments
            or aws_region
            or newrelic_account_id
            or honeycomb_dataset
        ):
            config.apply_cli_overrides(
                backend_type=backend,
                backend_url=url,
                api_key=api_key,
                app_key=app_key,
                sentry_org=sentry_org,
                sentry_project=sentry_project,
                tempo_instance_id=tempo_instance_id,
                aws_region=aws_region,
                newrelic_account_id=newrelic_account_id,
                honeycomb_dataset=honeycomb_dataset,
                environments=environments,
            )
        click.secho("[OK] Configuration loaded and validated", fg="green")
    except Exception as e:
        click.secho(f"[FAIL] Configuration: {e}", fg="red")
        sys.exit(1)

    try:
        backend_instance = _create_backend(config)
        click.secho(
            f"[OK] Backend constructed: {config.backend.type} @ {config.backend.url}", fg="green"
        )
    except Exception as e:
        click.secho(f"[FAIL] Backend construction: {e}", fg="red")
        sys.exit(1)

    async def _run_checks() -> int:
        exit_code = 0

        try:
            all_ok, details = await _run_backend_checks(backend_instance)
            _print_backend_check_results(details)
            if not all_ok:
                exit_code = 1
        finally:
            await backend_instance.close()

        # Secondary backend is env-var-only (no CLI override surface, see
        # config.py's ServerConfig.secondary_backend docstring), so there is
        # nothing to apply_cli_overrides here - it's already fully resolved
        # by ServerConfig.from_env() above.
        if config.secondary_backend is not None:
            try:
                secondary_instance = _build_backend_from_config(config.secondary_backend)
                click.secho(
                    f"[OK] Secondary backend constructed: {config.secondary_backend.type} "
                    f"@ {config.secondary_backend.url}",
                    fg="green",
                )
            except Exception as e:
                click.secho(f"[FAIL] Secondary backend construction: {e}", fg="red")
                return 1

            try:
                secondary_all_ok, secondary_details = await _run_backend_checks(secondary_instance)
                _print_backend_check_results(secondary_details, prefix="Secondary ")
                if not secondary_all_ok:
                    exit_code = 1
            finally:
                await secondary_instance.close()

        return exit_code

    sys.exit(asyncio.run(_run_checks()))


if __name__ == "__main__":
    main()

"""Opentelemetry MCP Server - Main entry point."""

import asyncio
import hmac
import ipaddress
import json
import logging
import math
import re
import sys
import time
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Literal, NoReturn

import click
import uvicorn
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth.auth import AccessToken, TokenVerifier
from fastmcp.server.http import StarletteWithLifespan
from mcp.types import ToolAnnotations
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from opentelemetry_mcp import __version__
from opentelemetry_mcp.attributes import HealthCheckResponse
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
from opentelemetry_mcp.security import (
    UserinfoRedactingFilter,
    is_loopback_host,
    redact_url_userinfo,
    redact_userinfo_in_text,
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

# Set up logging. The root stderr handler is built explicitly (rather than
# left to basicConfig) so the userinfo-redacting filter sits on the one
# HANDLER every record passes through - a filter on the root *logger* is
# never consulted for records propagated from child loggers (httpx, fastmcp,
# opentelemetry_mcp.*). basicConfig is a no-op when the root already has
# handlers (pytest), so tests drive _ROOT_LOG_HANDLER directly.
_ROOT_LOG_HANDLER = logging.StreamHandler()
_ROOT_LOG_HANDLER.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
_ROOT_LOG_HANDLER.addFilter(UserinfoRedactingFilter())
logging.basicConfig(level=logging.INFO, handlers=[_ROOT_LOG_HANDLER])
logger = logging.getLogger(__name__)


def _apply_log_level(level: str) -> None:
    """Set the root level and, unless the operator asked for DEBUG, quiet
    httpx's per-request INFO line ("HTTP Request: GET <url> ..."), which
    prints the full request URL - the one URL sink that cannot be redacted
    at creation time. LOG_LEVEL=DEBUG keeps that line for connectivity
    debugging; it then passes through the redacting handler filter like
    every other record."""
    root = logging.getLogger()
    root.setLevel(level)
    httpx_level = logging.NOTSET if root.getEffectiveLevel() <= logging.DEBUG else logging.WARNING
    logging.getLogger("httpx").setLevel(httpx_level)


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


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    """`error` followed by its __cause__ / implicit __context__ links, in
    the order a traceback prints them. Cycle-safe."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__suppress_context__:
            current = None
        else:
            current = current.__context__


def _handle_tool_error(tool_name: str, error: Exception) -> NoReturn:
    """Centralized error handler for tool functions.

    Logs the error with traceback, then re-raises it. Letting the exception
    propagate (rather than swallowing it and returning an error-shaped JSON
    string) is what makes the MCP SDK's lowlevel server report the failure as
    CallToolResult(isError=True), per SEP-2140 - a plain successful return
    would leave isError=False even though the tool call failed.

    The one exception to the identity re-raise: when the error text - or that
    of any exception chained beneath it (__cause__/__context__) - embeds a
    URL userinfo (httpx's HTTPStatusError message carries the request URL
    verbatim, so a credential a bypassed config left in BACKEND_URL would
    otherwise reach the MCP client as tool-error content), a ToolError with
    the redacted text is raised from it instead - still isError=True. The
    chain matters because the identity path ends in fastmcp's generic
    `except Exception`, whose logger.exception() prints every chained cause
    through the 'fastmcp' logger's own RichHandler - that logger does not
    propagate to the root handler, so UserinfoRedactingFilter never sees it.
    fastmcp logs a ToolError with exc_info=False, so nothing re-prints the
    chain.

    Args:
        tool_name: Name of the tool that encountered the error
        error: The exception that was raised
    """
    messages = [str(link) for link in _exception_chain(error)]
    if all(redact_userinfo_in_text(text) == text for text in messages):
        logger.error(f"Error executing {tool_name}: {error}", exc_info=True)
        raise error
    redacted = redact_userinfo_in_text(messages[0])
    # The traceback's final line would repeat the raw message: log without it.
    logger.error(f"Error executing {tool_name}: {redacted}")
    raise ToolError(redacted) from error


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
    (--max-traces-per-query / MAX_TRACES_PER_QUERY). This is the per-tool
    contract (the advertised `limit` is honoured up to the ceiling); the
    enforcement point is the backend boundary - BaseBackend.max_query_limit,
    set from the same field in _create_backend/_get_secondary_backend -
    which clamps every search_traces/search_spans query however a tool or
    helper re-expanded or chose its limit. _config is only None before
    startup has finished (every handler calls _get_backend() first, which
    raises the same error), so fail closed rather than forward an unbounded
    limit.
    """
    if _config is None:
        raise RuntimeError("Server configuration not set")
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
        logger.info(f"Initializing Jaeger backend: {redact_url_userinfo(backend_config.url)}")
        backend = JaegerBackend(
            url=str(backend_config.url),
            api_key=backend_config.api_key,
            timeout=backend_config.timeout,
        )
    elif backend_config.type == "tempo":
        logger.info(f"Initializing Tempo backend: {redact_url_userinfo(backend_config.url)}")
        backend = TempoBackend(
            url=str(backend_config.url),
            api_key=backend_config.api_key,
            timeout=backend_config.timeout,
            tempo_instance_id=backend_config.tempo_instance_id,
        )
    elif backend_config.type == "traceloop":
        logger.info(f"Initializing Traceloop backend: {redact_url_userinfo(backend_config.url)}")
        backend = TraceloopBackend(
            url=str(backend_config.url),
            api_key=backend_config.api_key,
            timeout=backend_config.timeout,
            environments=backend_config.environments,
        )
    elif backend_config.type == "datadog":
        logger.info(f"Initializing Datadog backend: {redact_url_userinfo(backend_config.url)}")
        backend = DatadogBackend(
            url=str(backend_config.url),
            api_key=backend_config.api_key,
            app_key=backend_config.app_key,
            timeout=backend_config.timeout,
        )
    elif backend_config.type == "sentry":
        logger.info(f"Initializing Sentry backend: {redact_url_userinfo(backend_config.url)}")
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
    backend.max_response_bytes = config.backend.max_response_bytes
    backend.max_query_limit = config.max_traces_per_query
    backend.configure_query_cache(config.query_cache_ttl_seconds, config.query_cache_max_entries)
    return backend


def _describe_health(health: HealthCheckResponse) -> str:
    """status plus the redacted error, never the model repr (which carries
    the backend URL and the raw error text)."""
    if health.error:
        return f"{health.status} ({redact_userinfo_in_text(health.error)})"
    return health.status


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
            logger.info(f"Backend health check: {_describe_health(health)}")
            if health.status != "healthy":
                logger.warning("Backend is not healthy, but continuing...")
        except Exception as e:
            logger.error(f"Backend health check failed: {redact_userinfo_in_text(str(e))}")
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
        _secondary_backend.max_response_bytes = _config.secondary_backend.max_response_bytes
        _secondary_backend.max_query_limit = _config.max_traces_per_query
        _secondary_backend.configure_query_cache(
            _config.query_cache_ttl_seconds, _config.query_cache_max_entries
        )

        try:
            health = await _secondary_backend.health_check()
            logger.info(f"Secondary backend health check: {_describe_health(health)}")
            if health.status != "healthy":
                logger.warning("Secondary backend is not healthy, but continuing...")
        except Exception as e:
            logger.error(
                f"Secondary backend health check failed: {redact_userinfo_in_text(str(e))}"
            )
            logger.warning("Continuing anyway, requests may fail...")

    return _secondary_backend


# GET /ready is unauthenticated and every probe costs the backend one
# health_check plus one list_services - on Honeycomb a slot of the process-
# wide 10/min Create Query Result budget that every session's tool calls
# queue behind, on X-Ray up to 222 operator-paid API calls. The route
# therefore memoizes the probe result for a window (success) or a shorter
# one (failure, so recovery is noticed quickly), shares one in-flight check
# between concurrent probes, and answers 503 instead of hanging on a parked
# backend. The window matches the Dockerfile HEALTHCHECK interval;
# READINESS_CACHE_SECONDS / --readiness-cache-seconds tunes it (0 disables
# the memo but keeps the timeout and the single-flight).
_READINESS_MEMO_TTL_SECONDS = 30.0
_READINESS_FAILURE_MEMO_TTL_SECONDS = 5.0
_READINESS_CHECK_TIMEOUT_SECONDS = 10.0

_readiness_cache_seconds: float = _READINESS_MEMO_TTL_SECONDS
_readiness_clock: Callable[[], float] = time.monotonic
# (backend, expires_at, all_ok, details, generation). Keyed on the backend
# object's identity so a recreated or test-patched backend invalidates it;
# `generation` counts completed probes so a caller that queued on the lock
# while a probe ran reuses that probe's result even when the memo window
# is 0 (single-flight without a shared future to cancel).
_readiness_memo: tuple[BaseBackend, float, bool, dict[str, Any], int] | None = None
_readiness_generation = 0
# Stored with the loop it was created in: a contended asyncio.Lock binds to
# its loop and raises RuntimeError when re-acquired under another one (one
# TestClient per test), so a lock from a previous loop is replaced.
_readiness_lock: tuple[asyncio.AbstractEventLoop, asyncio.Lock] | None = None


def _configure_readiness_cache(
    seconds: float, *, clock: Callable[[], float] = time.monotonic
) -> None:
    """Set the /ready memo window (0 disables the memo) and its clock; main()
    applies the operator's value before the HTTP app is built."""
    global _readiness_cache_seconds, _readiness_clock
    _readiness_cache_seconds = seconds
    _readiness_clock = clock
    _reset_readiness_memo()


def _reset_readiness_memo() -> None:
    """Drop the memo AND the lazily created lock - tests drive /ready from a
    fresh event loop each, and a lock contended under a previous loop
    cannot be reused under the next one."""
    global _readiness_memo, _readiness_lock, _readiness_generation
    _readiness_memo = None
    _readiness_lock = None
    _readiness_generation = 0


def _get_readiness_lock() -> asyncio.Lock:
    global _readiness_lock
    loop = asyncio.get_running_loop()
    if _readiness_lock is None or _readiness_lock[0] is not loop:
        _readiness_lock = (loop, asyncio.Lock())
    return _readiness_lock[1]


async def _run_backend_checks_memoized(backend: BaseBackend) -> tuple[bool, dict[str, Any]]:
    """/ready's view of _run_backend_checks: at most one probe per memo
    window per process, one in-flight probe shared by every concurrent
    caller, and a hard deadline that turns a parked backend into a 503.

    Stores and returns the details exactly as _run_backend_checks produced
    them (already redacted there); `doctor` keeps calling the live helper.
    """
    global _readiness_memo, _readiness_generation
    queued_at_generation = _readiness_generation
    async with _get_readiness_lock():
        memo = _readiness_memo
        if memo is not None and memo[0] is backend:
            _, expires_at, all_ok, details, generation = memo
            if expires_at > _readiness_clock() or generation > queued_at_generation:
                return all_ok, details

        try:
            async with asyncio.timeout(_READINESS_CHECK_TIMEOUT_SECONDS):
                all_ok, details = await _run_backend_checks(backend)
        except TimeoutError:
            all_ok = False
            details = {
                "error": f"readiness check timed out after {_READINESS_CHECK_TIMEOUT_SECONDS:g} s"
            }

        ttl = (
            _readiness_cache_seconds
            if all_ok
            else min(_readiness_cache_seconds, _READINESS_FAILURE_MEMO_TTL_SECONDS)
        )
        _readiness_generation += 1
        _readiness_memo = (
            backend,
            _readiness_clock() + ttl,
            all_ok,
            details,
            _readiness_generation,
        )
        return all_ok, details


async def _run_backend_checks(backend: BaseBackend) -> tuple[bool, dict[str, Any]]:
    """Run health_check + list_services against an already-constructed
    backend. Returns (all_ok, details). The caller owns the backend's
    lifecycle (construct/close) - this never closes it, so it is safe to
    call against the server's own cached _get_backend() instance without
    disrupting subsequent real tool calls that share it.

    Every error string is redacted HERE, not by the callers: /ready hands
    the details to unauthenticated peers and doctor prints them, and httpx
    error text embeds the request URL (userinfo included) verbatim.

    The list_services half goes through BaseBackend.list_services_uncached,
    never the cache-wrapped public method: with QUERY_CACHE_TTL_SECONDS set
    a warmed entry would otherwise report "ready" with zero backend requests
    for a whole TTL after the backend failed.
    """
    all_ok = True
    details: dict[str, Any] = {}

    try:
        health = await backend.health_check()
        details["health_check"] = {
            "status": health.status,
            "error": redact_userinfo_in_text(health.error),
        }
        if health.status != "healthy":
            all_ok = False
    except Exception as e:
        details["health_check"] = {"status": "error", "error": redact_userinfo_in_text(str(e))}
        all_ok = False

    try:
        services_found = await backend.list_services_uncached()
        details["list_services"] = {"count": len(services_found)}
    except Exception as e:
        details["list_services"] = {"error": redact_userinfo_in_text(str(e))}
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
    can never break a real tool call that shares the same backend. The
    probe result is memoized and single-flighted by
    _run_backend_checks_memoized because this route is unauthenticated and
    each probe costs the backend real, budgeted work."""
    try:
        backend = await _get_backend()
    except Exception as e:
        return JSONResponse(
            {"status": "not_ready", "error": redact_userinfo_in_text(str(e))}, status_code=503
        )

    all_ok, details = await _run_backend_checks_memoized(backend)
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
        # The heuristic fallback's per-service candidate count is a helper-
        # internal limit with no caller-facing argument; it still goes
        # through the operator ceiling like every other search limit.
        result = await correlate.correlate_trace(
            backend,
            secondary,
            trace_id=trace_id,
            max_candidates_per_service=_clamp_limit(correlate.MAX_CANDIDATES_PER_SERVICE_QUERY),
        )
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
    limit: int = 1000,
) -> str:
    """Get detailed performance statistics for a specific LLM model.

    Analyzes request count, latency percentiles (p50, p95, p99), token usage statistics,
    error rates, and finish reason distributions.

    Args:
        model_name: Model name to analyze (e.g., "gpt-4", "claude-3-opus", "gpt-3.5-turbo")
        start_time: Start time in ISO 8601 format (e.g., 2024-01-01T00:00:00Z)
        end_time: End time in ISO 8601 format
        service_name: Filter by service name
        limit: Maximum traces to analyze (default: 1000, capped by MAX_TRACES_PER_QUERY / --max-traces-per-query)

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
            limit=_clamp_limit(limit),
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


_MIN_HTTP_BEARER_TOKEN_CHARS = 32


class OperatorTokenVerifier(TokenVerifier):
    """Binds every /mcp request on the HTTP transport to the operator's own
    shared bearer token.

    fastmcp 4.0.5 only installs RequireAuthMiddleware around the /mcp route
    when ``FastMCP.auth`` is set (fastmcp/server/http.py, the ``if auth:``
    branch of create_streamable_http_app); with it unset every network peer
    that can reach the port runs all 19 tools under the operator's backend
    credential. This is the smallest AuthProvider that closes that gap for
    the shared-secret case: no OAuth routes, no metadata endpoints, one
    constant-time comparison. fastmcp's own StaticTokenVerifier is
    deliberately not used - its docstring says it must never be used in
    production (it keeps a plaintext token table and compares with ``==``).

    JWT/OIDC resource-server mode (fastmcp's JWTVerifier / RemoteAuthProvider)
    is additive through the same ``mcp.auth`` seam if ever wanted.
    """

    def __init__(self, token: str) -> None:
        super().__init__()
        self._expected = token.encode()

    async def verify_token(self, token: str) -> AccessToken | None:
        if hmac.compare_digest(token.encode(), self._expected):
            return AccessToken(token=token, client_id="operator", scopes=[])
        return None


def _validate_http_listener_auth(
    *, host: str, http_bearer_token: str | None, allow_unauthenticated_http: bool
) -> None:
    """Startup gate for ``--transport http``: serving on a non-loopback
    interface without caller authentication requires an explicit operator
    acknowledgement. Raises click.UsageError (exit code 2, message on
    stderr) - called from main() *before* its ``try:`` so the catch-all
    ``except Exception`` there can never swallow the usage error into a
    logged 'Server error' and exit 1."""
    if http_bearer_token is not None:
        if len(http_bearer_token) < _MIN_HTTP_BEARER_TOKEN_CHARS:
            raise click.UsageError(
                "MCP_HTTP_BEARER_TOKEN / --http-bearer-token must be at least "
                f"{_MIN_HTTP_BEARER_TOKEN_CHARS} characters (got {len(http_bearer_token)}); "
                "generate one with `openssl rand -hex 32`."
            )
        return
    if is_loopback_host(host) or allow_unauthenticated_http:
        return
    raise click.UsageError(
        f"--transport http on a non-loopback host ({host}) exposes every tool to any "
        "network peer with no authentication. Either set MCP_HTTP_BEARER_TOKEN (preferred; "
        "clients then send `Authorization: Bearer <token>`), bind loopback with "
        "`--host 127.0.0.1` behind an authenticating reverse proxy, or pass "
        "--allow-unauthenticated-http (MCP_ALLOW_UNAUTHENTICATED_HTTP=true) to accept "
        "that risk explicitly."
    )


_LOCAL_ORIGIN_PATTERN = re.compile(r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$", re.IGNORECASE)

# Upper bound on how much of a rejected Host/Origin header value reaches the
# log. OriginValidationMiddleware logs before RateLimitMiddleware runs, so
# without a bound one unauthenticated peer could write header-sized,
# attacker-chosen text to the log on every request.
_MAX_LOGGED_HEADER_CHARS = 128


def _parse_csv_option(value: str | None) -> list[str] | None:
    """Comma-separated CLI/env value -> trimmed, non-empty items; None stays None
    so callers can tell "not configured" from "configured empty"."""
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _hostname_from_host_header(value: str) -> str:
    """Lower-cased host part of a Host header value: ``[v6]:port`` -> ``v6``,
    ``name:port`` -> ``name``, a missing/blank header -> ``""``."""
    value = value.strip().lower()
    if not value:
        return ""
    if value.startswith("["):
        end = value.find("]")
        return value[1:end] if end != -1 else value[1:]
    if value.count(":") > 1:
        # Bare IPv6 literal (not valid in a Host header, but never a port).
        return value
    host, _, port = value.rpartition(":")
    if host and port.isdigit():
        return host
    return value


def _is_unspecified_address(hostname: str) -> bool:
    """True for the wildcard binds 0.0.0.0 / :: - a listener bound there has
    no single legitimate name to add to a Host allowlist."""
    try:
        return ipaddress.ip_address(hostname).is_unspecified
    except ValueError:
        return False


class OriginValidationMiddleware(BaseHTTPMiddleware):
    """Validate the Host and Origin headers on requests to the streamable-http
    transport.

    Host rule (DNS-rebinding defence proper). A page served from
    ``attacker.example`` whose name is then re-pointed at 127.0.0.1 issues a
    same-origin GET that carries ``Host: attacker.example:8000`` and - per
    the Fetch standard for same-origin GET/HEAD - no Origin header at all,
    so an Origin-only check lets it reach /ready and /mcp. When the
    legitimate names are known the Host header is therefore checked *first*:
    automatically for a loopback bind (``--host 127.0.0.1`` / ``localhost`` /
    ``::1``, main()'s default), and for any deployment that sets
    ``--allowed-hosts``. Accepted names are every loopback spelling, the
    bound address itself when it is a specific (non-wildcard) address, and
    the ``--allowed-hosts`` entries, compared case-insensitively with the
    port stripped; anything else answers ``421 Misdirected Request`` before
    any route runs. A wildcard bind (``0.0.0.0`` / ``::``, the Dockerfile
    CMD) has no single legitimate name, so it performs no Host validation
    unless ``--allowed-hosts`` names the public hostnames; ``--allowed-hosts
    '*'`` is the explicit opt-out. Constructed with no kwargs (bind_host and
    allowed_hosts both None) the middleware keeps its pre-existing
    Origin-only contract.

    Origin rule. fastmcp 3.2.0 built its StreamableHTTPSessionManager without
    passing security_settings through at all, leaving the upstream mcp SDK's
    own DNS-rebinding/Origin protection (mcp.server.transport_security)
    disabled by omission. Re-verified against the real installed fastmcp
    4.0.5 (fastmcp/server/http.py:668-682, not assumed unchanged from
    3.2.0): the gap is now an explicit, unconditional

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
    This middleware remains fully necessary, satisfying the MCP spec's
    (2025-06-18 basic/transports) MUST-requirement for Streamable HTTP
    servers to validate the Origin header regardless of which fastmcp
    version is running. fastmcp's guard was deliberately not adopted for the
    Host rule either: once given an explicit allowlist it also enforces its
    own Origin policy, which rejects the local-dev Origin allowance below
    whenever the Host is a non-loopback name - the in-repo check keeps one
    allowlist and the Origin semantics byte-for-byte.

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

    Both rejection paths log at WARNING so a misconfigured MCP_ALLOWED_HOSTS
    or client Origin is diagnosable from the server log. This middleware runs
    before RateLimitMiddleware (Starlette applies the middleware list
    outermost-first and main() places this class first), so an
    unauthenticated peer can trigger one such line per request. Accepted
    limitation, bounded rather than reordered: the logged value is cut to
    _MAX_LOGGED_HEADER_CHARS and repr()-escaped, so each line carries at most
    that much attacker-chosen text and can never contain a raw newline.
    """

    def __init__(
        self,
        app: Any,
        bind_host: str | None = None,
        allowed_hosts: Sequence[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._validate_host = allowed_hosts is not None or is_loopback_host(bind_host)
        self._host_wildcard = allowed_hosts is not None and "*" in allowed_hosts
        allowlist: set[str] = set()
        if bind_host:
            bound = _hostname_from_host_header(bind_host)
            if bound and not _is_unspecified_address(bound):
                allowlist.add(bound)
        for name in allowed_hosts or ():
            normalized = _hostname_from_host_header(name)
            if normalized and normalized != "*":
                allowlist.add(normalized)
        self._host_allowlist = frozenset(allowlist)

    def _host_is_allowed(self, request: Request) -> bool:
        hostname = _hostname_from_host_header(request.headers.get("host", ""))
        return is_loopback_host(hostname) or hostname in self._host_allowlist

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if self._validate_host and not self._host_wildcard and not self._host_is_allowed(request):
            host = request.headers.get("host", "")[:_MAX_LOGGED_HEADER_CHARS]
            logger.warning(f"Unexpected Host header: {host!r}")
            return Response("Misdirected Request", status_code=421)
        origin = request.headers.get("origin")
        if origin and not _LOCAL_ORIGIN_PATTERN.match(origin):
            logger.warning(f"Invalid Origin header: {origin[:_MAX_LOGGED_HEADER_CHARS]!r}")
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
        as a whole - is keyed on request.client.host, i.e. the TCP peer
        address (or, only for connections from a proxy listed in
        --forwarded-allow-ips, the right-most untrusted X-Forwarded-For
        entry uvicorn substitutes for it). A source rotating its address
        on every request always starts a brand-new, never-before-seen
        bucket at count 0, so the per-peer cap never engages against that
        source's aggregate volume. That is a structural limitation of pure
        per-peer fixed-window limiting, not something this sweep (or a
        bigger one) can fix - it would need additional infra (e.g. a WAF)
        to address.
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
    """Per-peer fixed-window rate limiter for the streamable-http
    transport. Bare 429 on breach.

    The key is request.client.host - the TCP peer address as the served
    process sees it. This middleware parses no X-Forwarded-For itself, but
    uvicorn's ProxyHeadersMiddleware (defaulted ON by uvicorn, trusting
    127.0.0.1 or FORWARDED_ALLOW_IPS implicitly) rewrites scope["client"]
    from that header for connections it trusts *before* this code runs,
    which would let any loopback peer - a pass-through proxy, an ssh -L /
    socat tunnel, another local user - pick its own key. main() therefore
    builds the uvicorn.Config with proxy_headers disabled unless the
    operator names the proxies via --forwarded-allow-ips /
    FORWARDED_ALLOW_IPS (see _build_uvicorn_config); only then, and only
    for connections from those addresses, does the right-most untrusted
    X-Forwarded-For entry replace the TCP peer as the key.

    Known structural limitation: a source rotating its address on every
    request is never capped, since each never-before-seen address starts
    its own bucket at count 0 - the per-peer cap bounds a single, stable
    peer's own volume, never a rotating source's aggregate. Mitigating
    that would need additional infra (e.g. a WAF) in front of this server.
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


# Parity with the mcp SDK's own DEFAULT_SESSION_IDLE_TIMEOUT (30 min), which
# every other mcp-based streamable-HTTP server already applies; fastmcp 4.0.5
# is the odd one out in overriding it to "never" (see _build_http_app).
_DEFAULT_SESSION_IDLE_TIMEOUT_SECONDS = 1800.0


def _require_finite_seconds(ctx: click.Context, param: click.Parameter, value: float) -> float:
    """Option callback for the duration options (--session-idle-timeout-seconds,
    --readiness-cache-seconds). click.FloatRange(min=0, ...) still admits
    ``inf`` (``inf > 0`` holds) and ``nan``
    (every comparison against it is False), and the mcp SDK's
    StreamableHTTPSessionManager raises ValueError for both only at lifespan
    start - the same late, cryptic 'Server error' / exit 1 the range check
    exists to prevent. Rejecting them here keeps every bad value a usage error
    with exit code 2 before the app is built."""
    if not math.isfinite(value):
        raise click.BadParameter("must be a finite number of seconds")
    return value


def _build_http_app(
    middleware: list[Middleware], *, session_idle_timeout_seconds: float, stateless_http: bool
) -> StarletteWithLifespan:
    """The one place main() turns `mcp` into the streamable-HTTP Starlette app.

    fastmcp 4.0.5's http_app() resolves an unset session_idle_timeout from
    its own ``settings.http_session_idle_timeout``, whose default is None -
    "sessions never expire from inactivity" - and forwards that None into
    the mcp SDK's StreamableHTTPSessionManager, overriding the SDK's own
    1800 s default. With the SDK's 10,000-session cap left in place, one
    peer could then fill the shared session table with bare initialize
    POSTs and make every new client's initialize answer 503 until restart.
    Always passing a finite bound here restores the SDK behaviour; an idle
    session's id answers 404 and spec-compliant clients re-initialize.

    In stateless mode no per-peer state exists, so the timeout is passed as
    None (the SDK ignores it there anyway).
    """
    return mcp.http_app(
        transport="streamable-http",
        middleware=middleware,
        stateless_http=stateless_http,
        session_idle_timeout=None if stateless_http else session_idle_timeout_seconds,
    )


def _build_uvicorn_config(
    app: Any,
    *,
    host: str,
    port: int,
    graceful_shutdown_timeout_seconds: int,
    forwarded_allow_ips: str | None,
) -> uvicorn.Config:
    """The uvicorn.Config main() serves the HTTP app with.

    uvicorn 0.40.0 defaults ``proxy_headers=True`` and, when
    ``forwarded_allow_ips`` is None, resolves the trusted set from the
    FORWARDED_ALLOW_IPS environment variable or "127.0.0.1" - so by default
    any loopback peer's X-Forwarded-For rewrote request.client.host, the
    RateLimitMiddleware key. Passing ``proxy_headers=forwarded_allow_ips is
    not None`` means uvicorn's implicit fallback is never consulted: the
    header is honoured only for connections from the proxies the operator
    named (the CLI option deliberately reuses uvicorn's FORWARDED_ALLOW_IPS
    env var name, so operators who already set it see identical behaviour).
    """
    return uvicorn.Config(
        app,
        host=host,
        port=port,
        lifespan="on",
        # Preserves run_http_async's own default (lost by bypassing it) for
        # the connection/task-draining wait uvicorn does BEFORE dispatching
        # lifespan.shutdown - unrelated to the drain feature itself, which
        # fires after this. Exposed as --graceful-shutdown-timeout-seconds
        # so a real signal-timing test can widen it deterministically
        # instead of relying on this project's own default margin.
        timeout_graceful_shutdown=graceful_shutdown_timeout_seconds,
        proxy_headers=forwarded_allow_ips is not None,
        forwarded_allow_ips=forwarded_allow_ips,
    )


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
        "url": redact_url_userinfo(backend_config.url),
        "environments": backend_config.environments,
        "timeout": backend_config.timeout,
        "max_response_bytes": backend_config.max_response_bytes,
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
    default="127.0.0.1",
    envvar="MCP_HOST",
    help="Host to bind HTTP server to (only for --transport http, default: "
    "127.0.0.1 per the MCP spec's loopback guidance; a non-loopback host such "
    "as 0.0.0.0 additionally requires MCP_HTTP_BEARER_TOKEN or "
    "--allow-unauthenticated-http, overrides MCP_HOST env var)",
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
    "--http-bearer-token",
    type=str,
    default=None,
    envvar="MCP_HTTP_BEARER_TOKEN",
    help="Shared bearer token every /mcp request must present as "
    "`Authorization: Bearer <token>` (only for --transport http; at least "
    f"{_MIN_HTTP_BEARER_TOKEN_CHARS} characters, e.g. `openssl rand -hex 32`). "
    "Prefer the MCP_HTTP_BEARER_TOKEN env var over this flag: argv is visible "
    "to every local user via `ps`. /health and /ready stay unauthenticated.",
)
@click.option(
    "--allow-unauthenticated-http",
    is_flag=True,
    default=False,
    envvar="MCP_ALLOW_UNAUTHENTICATED_HTTP",
    help="Acknowledge serving --transport http on a non-loopback --host with "
    "no MCP_HTTP_BEARER_TOKEN, i.e. every network peer runs every tool under "
    "the operator's backend credential (default: refuse to start; overrides "
    "MCP_ALLOW_UNAUTHENTICATED_HTTP env var)",
)
@click.option(
    "--allowed-hosts",
    type=str,
    default=None,
    envvar="MCP_ALLOWED_HOSTS",
    help="Comma-separated hostnames/IP literals accepted in the Host header "
    "besides loopback names and the bound address; any other Host answers 421 "
    "(DNS-rebinding defence). A loopback --host validates Host automatically; "
    "a network bind only validates when this is set; the single value '*' "
    "disables Host validation (only for --transport http, overrides "
    "MCP_ALLOWED_HOSTS env var)",
)
@click.option(
    "--forwarded-allow-ips",
    type=str,
    default=None,
    envvar="FORWARDED_ALLOW_IPS",
    help="Comma-separated proxy IPs/CIDRs (or '*') whose X-Forwarded-For may "
    "replace the TCP peer address as the rate-limit key. Unset (default): the "
    "header is never trusted, not even from 127.0.0.1 - uvicorn's own implicit "
    "default is deliberately not applied. Same env var uvicorn honours (only "
    "for --transport http, overrides FORWARDED_ALLOW_IPS env var)",
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
    "--trust-client-trace-context/--no-trust-client-trace-context",
    default=None,
    envvar="MCP_TRUST_CLIENT_TRACE_CONTEXT",
    help="When OTel self-instrumentation is enabled, adopt the W3C trace "
    "context a caller places under a request's params._meta as the parent "
    "of this server's own spans (default: on for stdio, whose only caller is "
    "the trusted parent process; off for --transport http, where any peer "
    "could otherwise choose the trace identity the server's spans and "
    "correlated log lines are filed under - overrides "
    "MCP_TRUST_CLIENT_TRACE_CONTEXT env var)",
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
    help="Maximum traces/spans a single backend query may fetch, 1-1000 - "
    "caps every tool's limit, including tool-internal over-fetch, at the "
    "backend boundary (overrides MAX_TRACES_PER_QUERY env var, default: 500)",
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
    "--readiness-cache-seconds",
    type=click.FloatRange(min=0),
    callback=_require_finite_seconds,
    default=_READINESS_MEMO_TTL_SECONDS,
    envvar="READINESS_CACHE_SECONDS",
    help="Memoize GET /ready results for this many seconds (a failed probe "
    f"for at most {_READINESS_FAILURE_MEMO_TTL_SECONDS:g} s) so any number "
    "of unauthenticated probes cost the backend at most one health_check + "
    "list_services per window; concurrent probes share one in-flight check "
    f"and a check answers 503 after {_READINESS_CHECK_TIMEOUT_SECONDS:g} s "
    f"(only for --transport http, default: {_READINESS_MEMO_TTL_SECONDS:g}, "
    "set to 0 to disable the memo, overrides READINESS_CACHE_SECONDS env var)",
)
@click.option(
    "--session-idle-timeout-seconds",
    # min_open exists on click 8.x's FloatRange (verified at runtime); the
    # pinned types-click 7.1.8 stubs predate it, hence the targeted ignore.
    type=click.FloatRange(min=0, min_open=True),  # type: ignore[call-arg]
    callback=_require_finite_seconds,
    default=_DEFAULT_SESSION_IDLE_TIMEOUT_SECONDS,
    envvar="SESSION_IDLE_TIMEOUT_SECONDS",
    help="Close a streamable-HTTP session that has had no request for this "
    "many seconds; its id then answers 404 and the client re-initializes "
    "(only for --transport http, must be a finite number > 0, default: "
    f"{_DEFAULT_SESSION_IDLE_TIMEOUT_SECONDS:g} - the mcp SDK's own default, "
    "overrides SESSION_IDLE_TIMEOUT_SECONDS env var)",
)
@click.option(
    "--stateless-http/--no-stateless-http",
    default=False,
    envvar="MCP_STATELESS_HTTP",
    help="Serve each /mcp request on a fresh transport with no session "
    "tracking at all, so no per-peer state outlives a request - at the cost "
    "of the GET SSE stream used for server-initiated notifications (only for "
    "--transport http, default: stateful, overrides MCP_STATELESS_HTTP env var)",
)
@click.option(
    "--query-cache-ttl-seconds",
    type=float,
    default=None,
    help="Cache backend query results (search_traces/search_spans/get_trace/"
    "list_services/get_service_operations) for this many seconds, with "
    "in-flight request coalescing; must be finite, > 0 and at most 86400 "
    "(unset: disabled, overrides QUERY_CACHE_TTL_SECONDS env var)",
)
@click.option(
    "--query-cache-max-entries",
    type=click.IntRange(1, 100_000),
    default=None,
    help="Upper bound on the number of cached backend query results held at "
    "once; the least-recently-used entry is evicted first, independent of "
    "the TTL (overrides QUERY_CACHE_MAX_ENTRIES env var, default: 1024)",
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
    http_bearer_token: str | None,
    allow_unauthenticated_http: bool,
    allowed_hosts: str | None,
    forwarded_allow_ips: str | None,
    include_args_in_spans: bool,
    trust_client_trace_context: bool | None,
    log_level: str | None,
    max_traces_per_query: int | None,
    disable_tools: str | None,
    enabled_tools: str | None,
    slow_request_threshold_ms: float | None,
    rate_limit_max_requests: int,
    rate_limit_window_seconds: float,
    readiness_cache_seconds: float,
    session_idle_timeout_seconds: float,
    stateless_http: bool,
    query_cache_ttl_seconds: float | None,
    query_cache_max_entries: int | None,
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

      # Run with HTTP transport on loopback (the default host)
      tracehub-mcp --transport http --port 8000

      # Run with HTTP on every interface - requires a caller token
      MCP_HTTP_BEARER_TOKEN=$(openssl rand -hex 32) \\
        tracehub-mcp --transport http --host 0.0.0.0 --port 9000

      # Validate a candidate config without starting the server
      tracehub-mcp doctor --backend jaeger --url http://localhost:16686

      # Print the resolved config as JSON (secrets redacted to booleans)
      tracehub-mcp --print-config
    """
    if ctx.invoked_subcommand is not None:
        return

    global _config

    # Before the try: a click.UsageError raised here reaches click's own
    # handler (exit 2, message on stderr) instead of the catch-all below.
    if transport == "http":
        _validate_http_listener_auth(
            host=host,
            http_bearer_token=http_bearer_token,
            allow_unauthenticated_http=allow_unauthenticated_http,
        )
    elif http_bearer_token is not None:
        logger.warning(
            "MCP_HTTP_BEARER_TOKEN is set but --transport is stdio: the token only "
            "guards the HTTP listener and is ignored for stdio"
        )
    if trust_client_trace_context is None:
        # stdio's only caller is the trusted parent process; an HTTP peer is not.
        trust_client_trace_context = transport == "stdio"

    try:
        try:
            # Load configuration from environment
            _config = ServerConfig.from_env()

            # Set logging level
            _apply_log_level(_config.log_level)

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
                or query_cache_max_entries is not None
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
                    query_cache_max_entries=query_cache_max_entries,
                )
                _apply_log_level(_config.log_level)
        except (ValidationError, ValueError) as e:
            # pydantic's ValidationError echoes input_value='<the URL>' and a
            # traceback would print it again: the redacted message only.
            logger.error(f"Configuration error: {redact_userinfo_in_text(str(e))}")
            sys.exit(1)

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
                "query_cache_max_entries": _config.query_cache_max_entries,
                "transport": transport,
                "host": host,
                "port": port,
                "http_bearer_token_set": http_bearer_token is not None,
                "allow_unauthenticated_http": allow_unauthenticated_http,
                "allowed_hosts": allowed_hosts,
                "forwarded_allow_ips": forwarded_allow_ips,
                "include_args_in_spans": include_args_in_spans,
                "trust_client_trace_context": trust_client_trace_context,
                "disable_tools": disable_tools,
                "enabled_tools": enabled_tools,
                "rate_limit_max_requests": rate_limit_max_requests,
                "rate_limit_window_seconds": rate_limit_window_seconds,
                "readiness_cache_seconds": readiness_cache_seconds,
                "session_idle_timeout_seconds": session_idle_timeout_seconds,
                "stateless_http": stateless_http,
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
        tracing_enabled = configure_tracing(trust_client_trace_context=trust_client_trace_context)
        configure_metrics()
        if tracing_enabled:
            mcp.add_middleware(McpServerTracingMiddleware(include_args=include_args_in_spans))
            install_trace_context_log_filter()

        # Run server with selected transport
        if transport == "http":
            logger.info(f"Starting MCP server with HTTP transport on {host}:{port}")
            logger.info("Using streamable-http transport for better compatibility")
            logger.info(f"Connect clients to: http://{host}:{port}/mcp")
            if http_bearer_token is not None:
                # FastMCP.http_app forwards self.auth at call time, so
                # setting it here is enough for RequireAuthMiddleware to
                # wrap /mcp; /health and /ready are plain custom routes.
                mcp.auth = OperatorTokenVerifier(http_bearer_token)
                logger.info("Bearer-token authentication enabled for /mcp")
            else:
                logger.warning(
                    "Serving /mcp with NO caller authentication on %s:%d - every peer that "
                    "can reach this port runs every tool under the configured backend "
                    "credential",
                    host,
                    port,
                )
            middleware = [
                Middleware(
                    OriginValidationMiddleware,
                    bind_host=host,
                    allowed_hosts=_parse_csv_option(allowed_hosts),
                )
            ]
            if rate_limit_max_requests > 0:
                middleware.append(
                    Middleware(
                        RateLimitMiddleware,
                        max_requests=rate_limit_max_requests,
                        window_seconds=rate_limit_window_seconds,
                    )
                )
            _configure_readiness_cache(readiness_cache_seconds)
            app = _build_http_app(
                middleware,
                session_idle_timeout_seconds=session_idle_timeout_seconds,
                stateless_http=stateless_http,
            )
            _install_shutdown_drain(
                app,
                drain_seconds=shutdown_drain_seconds,
                close_timeout_seconds=backend_close_timeout_seconds,
            )
            uvicorn_config = _build_uvicorn_config(
                app,
                host=host,
                port=port,
                graceful_shutdown_timeout_seconds=graceful_shutdown_timeout_seconds,
                forwarded_allow_ips=forwarded_allow_ips,
            )
            asyncio.run(uvicorn.Server(uvicorn_config).serve())
        else:
            logger.info(
                f"Starting MCP server with stdio transport using Backend: {_config.backend.type} "
                f"connected to: {redact_url_userinfo(_config.backend.url)}"
            )
            mcp.run(transport="stdio")

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except click.ClickException:
        # click.UsageError is an Exception subclass: let click render it
        # (exit 2, message on stderr) rather than logging 'Server error'.
        raise
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
        # pydantic's ValidationError echoes the raw input URL (see main()).
        click.secho(f"[FAIL] Configuration: {redact_userinfo_in_text(str(e))}", fg="red")
        sys.exit(1)

    try:
        backend_instance = _create_backend(config)
        click.secho(
            f"[OK] Backend constructed: {config.backend.type} @ "
            f"{redact_url_userinfo(config.backend.url)}",
            fg="green",
        )
    except Exception as e:
        click.secho(f"[FAIL] Backend construction: {redact_userinfo_in_text(str(e))}", fg="red")
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
                    f"@ {redact_url_userinfo(config.secondary_backend.url)}",
                    fg="green",
                )
            except Exception as e:
                click.secho(
                    f"[FAIL] Secondary backend construction: {redact_userinfo_in_text(str(e))}",
                    fg="red",
                )
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

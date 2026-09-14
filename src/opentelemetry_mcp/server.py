"""Opentelemetry MCP Server - Main entry point."""

import logging
import re
import sys
from typing import Any, NoReturn

import click
from fastmcp import FastMCP
from mcp.types import ToolAnnotations
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from opentelemetry_mcp import __version__
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.backends.datadog import DatadogBackend
from opentelemetry_mcp.backends.jaeger import JaegerBackend
from opentelemetry_mcp.backends.sentry import SentryBackend
from opentelemetry_mcp.backends.tempo import TempoBackend
from opentelemetry_mcp.backends.traceloop import TraceloopBackend
from opentelemetry_mcp.config import ServerConfig
from opentelemetry_mcp.observability import McpServerTracingMiddleware, configure_tracing
from opentelemetry_mcp.tools import (
    compare,
    errors,
    expensive_traces,
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
    usage,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# All 11 tools below only ever query trace/span backends and never mutate
# backend state, so the same read-only/idempotent/open-world annotations
# apply to every one of them.
_READ_ONLY_TOOL_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True, idempotentHint=True, openWorldHint=True
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
_config: ServerConfig | None = None

# Initialize FastMCP server
mcp = FastMCP("tracehub-mcp", version=__version__)


def _create_backend(config: ServerConfig) -> BaseBackend:
    """Create backend instance based on configuration.

    Args:
        config: Server configuration

    Returns:
        Backend instance

    Raises:
        ValueError: If backend type is unsupported
    """
    backend_config = config.backend
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
    else:
        raise ValueError(f"Unsupported backend type: {backend_config.type}")

    # Not a constructor parameter (see BaseBackend.__init__'s own comment):
    # several backend subclasses override __init__ with their own named
    # params, so this is applied uniformly here instead.
    backend.slow_request_threshold_ms = config.slow_request_threshold_ms
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


@mcp.tool(annotations=_READ_ONLY_TOOL_ANNOTATIONS)
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
) -> str:
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
            limit=limit,
        )
        return result
    except Exception as e:
        return _handle_tool_error("search_traces", e)


@mcp.tool(annotations=_READ_ONLY_TOOL_ANNOTATIONS)
async def get_trace(trace_id: str) -> str:
    """Get complete trace details by trace ID.

    Returns all spans with attributes, including parsed Opentelemetry data for LLM operations.

    Args:
        trace_id: Trace identifier

    Returns:
        JSON string with trace details
    """
    try:
        backend = await _get_backend()
        result = await trace.get_trace(backend, trace_id=trace_id)
        return result
    except Exception as e:
        return _handle_tool_error("get_trace", e)


@mcp.tool(annotations=_READ_ONLY_TOOL_ANNOTATIONS)
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
            limit=limit,
        )
        return result
    except Exception as e:
        return _handle_tool_error("get_llm_usage", e)


@mcp.tool(annotations=_READ_ONLY_TOOL_ANNOTATIONS)
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


@mcp.tool(annotations=_READ_ONLY_TOOL_ANNOTATIONS)
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
            limit=limit,
        )
        return result
    except Exception as e:
        return _handle_tool_error("find_errors", e)


@mcp.tool(annotations=_READ_ONLY_TOOL_ANNOTATIONS)
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
            limit=limit,
        )
        return result
    except Exception as e:
        return _handle_tool_error("list_llm_models", e)


@mcp.tool(annotations=_READ_ONLY_TOOL_ANNOTATIONS)
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


@mcp.tool(annotations=_READ_ONLY_TOOL_ANNOTATIONS)
async def list_sessions(
    start_time: str | None = None,
    end_time: str | None = None,
    service_name: str | None = None,
    gen_ai_system: str | None = None,
    limit: int = 1000,
) -> str:
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
            limit=limit,
        )
        return result
    except Exception as e:
        return _handle_tool_error("list_sessions", e)


@mcp.tool(annotations=_READ_ONLY_TOOL_ANNOTATIONS)
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
            limit=limit,
        )
        return result
    except Exception as e:
        return _handle_tool_error("get_session_stats", e)


@mcp.tool(annotations=_READ_ONLY_TOOL_ANNOTATIONS)
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
            limit=limit,
        )
        return result
    except Exception as e:
        return _handle_tool_error("compare_time_windows", e)


@mcp.tool(annotations=_READ_ONLY_TOOL_ANNOTATIONS)
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
            limit=limit,
        )
        return result
    except Exception as e:
        return _handle_tool_error("get_prompt_version_stats", e)


@mcp.tool(annotations=_READ_ONLY_TOOL_ANNOTATIONS)
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
            limit=limit,
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


@mcp.tool(annotations=_READ_ONLY_TOOL_ANNOTATIONS)
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
            limit=limit,
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


@mcp.tool(annotations=_READ_ONLY_TOOL_ANNOTATIONS)
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
) -> str:
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
            limit=limit,
        )
        return result
    except Exception as e:
        return _handle_tool_error("search_spans_tool", e)


@mcp.tool(annotations=_READ_ONLY_TOOL_ANNOTATIONS)
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
            limit=limit,
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
        "get_prompt_version_stats",
        "get_llm_expensive_traces",
        "get_llm_slow_traces",
        "search_spans_tool",
        "list_llm_tools_tool",
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

    fastmcp 3.2.0 builds its StreamableHTTPSessionManager without passing
    security_settings through, which leaves the upstream mcp SDK's own
    DNS-rebinding/Origin protection (mcp.server.transport_security) disabled
    by default. fastmcp exposes no kwarg to pass security_settings through,
    so this middleware restores Origin validation directly, satisfying the
    MCP spec's (2025-06-18 basic/transports) MUST-requirement for Streamable
    HTTP servers to validate the Origin header.

    A request with no Origin header is allowed through unmodified: Origin can
    be absent for same-origin requests, and most MCP HTTP clients are not
    browsers and never send one in normal use. A request that does send an
    Origin header is only allowed through if it points at a local dev origin
    (127.0.0.1 or localhost, any scheme/port) - blocking the actual attack
    this check exists for: a malicious webpage running in a victim's browser
    using DNS rebinding or a crafted fetch() to reach a locally-bound
    tracehub-mcp HTTP server.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        origin = request.headers.get("origin")
        if origin and not _LOCAL_ORIGIN_PATTERN.match(origin):
            logger.warning(f"Invalid Origin header: {origin}")
            return Response("Invalid Origin header", status_code=403)
        return await call_next(request)


@click.command()
@click.option(
    "--backend",
    type=click.Choice(["jaeger", "tempo", "traceloop", "datadog", "sentry"]),
    help="Backend type (overrides BACKEND_TYPE env var)",
)
@click.option(
    "--url",
    type=str,
    help="Backend URL (overrides BACKEND_URL env var)",
)
@click.option(
    "--api-key",
    type=str,
    help="API key for backend authentication (overrides BACKEND_API_KEY env var)",
)
@click.option(
    "--app-key",
    type=str,
    help="Application key, required by the Datadog backend in addition to "
    "--api-key (overrides BACKEND_APP_KEY env var)",
)
@click.option(
    "--tempo-instance-id",
    type=str,
    help="Grafana Cloud stack/instance ID, used for Basic Auth with --api-key "
    "instead of Bearer auth (Tempo backend only, required for Grafana "
    "Cloud-hosted Tempo, not needed for self-hosted Tempo; overrides "
    "BACKEND_TEMPO_INSTANCE_ID env var)",
)
@click.option(
    "--sentry-org",
    type=str,
    help="Sentry organization slug, required by the Sentry backend "
    "(overrides BACKEND_SENTRY_ORG env var)",
)
@click.option(
    "--sentry-project",
    type=str,
    help="Sentry project slug, optional for the Sentry backend "
    "(overrides BACKEND_SENTRY_PROJECT env var)",
)
@click.option(
    "--environments",
    type=str,
    help="Comma-separated list of environments for Traceloop backend (overrides BACKEND_ENVIRONMENTS env var)",
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
def main(
    backend: str | None,
    url: str | None,
    api_key: str | None,
    app_key: str | None,
    tempo_instance_id: str | None,
    sentry_org: str | None,
    sentry_project: str | None,
    environments: str | None,
    transport: str,
    host: str,
    port: int,
    include_args_in_spans: bool,
    log_level: str | None,
    max_traces_per_query: int | None,
    disable_tools: str | None,
    enabled_tools: str | None,
    slow_request_threshold_ms: float | None,
) -> None:
    """Opentelemetry MCP Server - Query OpenTelemetry traces from LLM applications.

    Supports multiple backends: Jaeger, Tempo, Traceloop, Datadog, and Sentry.
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
    """
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
            or log_level
            or max_traces_per_query is not None
            or slow_request_threshold_ms is not None
        ):
            _config.apply_cli_overrides(
                backend_type=backend,
                backend_url=url,
                api_key=api_key,
                app_key=app_key,
                sentry_org=sentry_org,
                sentry_project=sentry_project,
                tempo_instance_id=tempo_instance_id,
                environments=environments,
                log_level=log_level,
                max_traces_per_query=max_traces_per_query,
                slow_request_threshold_ms=slow_request_threshold_ms,
            )
            logging.getLogger().setLevel(_config.log_level)

        # Backend will be lazily initialized on first tool call
        # This ensures it's created in FastMCP's event loop, not a separate one

        _apply_tool_gating(disable_tools=disable_tools, enabled_tools=enabled_tools)

        # OTel self-instrumentation is fully opt-in: configure_tracing() only
        # returns True (and only then do we register the middleware) when
        # OTEL_EXPORTER_OTLP_ENDPOINT is actually set, so there is zero
        # overhead and no dependency on a collector for anyone who has not
        # opted in.
        if configure_tracing():
            mcp.add_middleware(McpServerTracingMiddleware(include_args=include_args_in_spans))

        # Run server with selected transport
        if transport == "http":
            logger.info(f"Starting MCP server with HTTP transport on {host}:{port}")
            logger.info("Using streamable-http transport for better compatibility")
            logger.info(f"Connect clients to: http://{host}:{port}/mcp")
            mcp.run(
                transport="streamable-http",
                host=host,
                port=port,
                middleware=[Middleware(OriginValidationMiddleware)],
            )
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


if __name__ == "__main__":
    main()

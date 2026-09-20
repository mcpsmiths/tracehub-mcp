# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tracehub-mcp is an MCP (Model Context Protocol) server that enables AI agents to query and analyze OpenTelemetry traces from LLM applications. It parses Opentelemetry semantic conventions (the `gen_ai.*` attributes) to enable automated debugging and observability.

**Key Features:**

- Multi-backend support: Jaeger, Grafana Tempo, Traceloop, Datadog, Sentry, AWS X-Ray, New Relic, and Honeycomb
- 19 MCP tools: Core tools + agent-native triage + cross-backend correlation + LLM-oriented discovery and analysis tools + session/prompt-version/time-window aggregation + cost/error spike investigation
- Cost attribution (`cost_usd`) via a vendored litellm pricing table, with an honest `cost_usd_is_partial` flag when a model's price can't be resolved
- Token usage tracking and aggregation across models/services
- Finish reasons tracking for debugging truncated/filtered responses
- Enhanced token calculation supporting all `gen_ai.usage.*` attributes
- Dual transport modes: stdio (Claude Desktop) and HTTP/SSE (network access)
- Opt-in OTel self-instrumentation of the server itself (see Configuration)

## Development Commands

**Package Manager:** This project uses UV (not pip). All commands should use `uv run`.

```bash
# Install dependencies
uv sync

# Run the server (stdio transport for Claude Desktop)
uv run tracehub-mcp

# Run with HTTP transport
uv run tracehub-mcp --transport http --port 8000

# Override backend configuration
uv run tracehub-mcp --backend jaeger --url http://localhost:16686

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=opentelemetry_mcp --cov-report=html

# Run specific test file
uv run pytest tests/test_models.py

# Format code (always run before committing)
uv run ruff format .

# Lint code
uv run ruff check .

# Type check (strict mode, must pass)
uv run mypy src/
```

## Architecture

### Backend Abstraction Pattern

All trace storage backends implement the `BaseBackend` abstract interface in [opentelemetry_mcp/backends/base.py](opentelemetry_mcp/backends/base.py):

```python
class BaseBackend(ABC):
    @abstractmethod
    async def search_traces(self, query: TraceQuery) -> list[TraceData]: ...
    @abstractmethod
    async def search_spans(self, query: SpanQuery) -> list[SpanData]: ...
    @abstractmethod
    async def get_trace(self, trace_id: str) -> TraceData: ...
    @abstractmethod
    async def list_services(self) -> list[str]: ...
    @abstractmethod
    async def get_service_operations(self, service: str) -> list[str]: ...
    @abstractmethod
    async def health_check(self) -> bool: ...
```

Concrete implementations:

- [backends/jaeger.py](opentelemetry_mcp/backends/jaeger.py) - Jaeger backend
- [backends/tempo.py](opentelemetry_mcp/backends/tempo.py) - Grafana Tempo backend
- [backends/traceloop.py](opentelemetry_mcp/backends/traceloop.py) - Traceloop backend
- [backends/datadog.py](opentelemetry_mcp/backends/datadog.py) - Datadog backend
- [backends/sentry.py](opentelemetry_mcp/backends/sentry.py) - Sentry backend
- [backends/xray.py](opentelemetry_mcp/backends/xray.py) - AWS X-Ray backend (boto3/SigV4, not httpx - see its module docstring)
- [backends/newrelic.py](opentelemetry_mcp/backends/newrelic.py) - New Relic backend (NerdGraph GraphQL API - built without a live account to verify against, see its module docstring for exactly what's unverified)
- [backends/honeycomb.py](opentelemetry_mcp/backends/honeycomb.py) - Honeycomb backend (Query Data API - built without a live account to verify against; the programmatic Query Data API is documented as Enterprise-plan exclusive, see its module docstring)

### Tool-Based Architecture

Each MCP capability is implemented as a separate tool module in [opentelemetry_mcp/tools/](opentelemetry_mcp/tools/):

**Core Tools:**

- [tools/search.py](opentelemetry_mcp/tools/search.py) - Search traces with filters
- [tools/search_spans.py](opentelemetry_mcp/tools/search_spans.py) - Search individual spans with filters
- [tools/trace.py](opentelemetry_mcp/tools/trace.py) - Get detailed trace by ID (`detail_level`: `"full"` default returns everything unbounded; `"summary"` elides large gen_ai.\* message/document fields and truncates long event attributes)
- [tools/triage.py](opentelemetry_mcp/tools/triage.py) - `triage_trace`: synthesize a likely-root-cause diagnosis (critical path, latency-contribution ranking, error chain) instead of returning raw trace data - composes [tools/span_tree.py](opentelemetry_mcp/tools/span_tree.py)'s tree-walking utilities, which no backend or other tool builds elsewhere
- [tools/correlate.py](opentelemetry_mcp/tools/correlate.py) - `correlate_trace`: best-effort cross-backend correlation against a second, independently-configured backend (`SECONDARY_BACKEND_*`) - direct `trace_id` match first, else a time-window/service-overlap heuristic; also reuses `tools/span_tree.py` to compare error-driven root causes across both traces
- [tools/usage.py](opentelemetry_mcp/tools/usage.py) - Aggregate token usage metrics
- [tools/services.py](opentelemetry_mcp/tools/services.py) - List available services
- [tools/errors.py](opentelemetry_mcp/tools/errors.py) - Find traces with errors

**LLM-Oriented Tools (Discovery & Analysis):**

- [tools/list_models.py](opentelemetry_mcp/tools/list_models.py) - List all models in use with statistics
- [tools/model_stats.py](opentelemetry_mcp/tools/model_stats.py) - Performance stats for a specific model
- [tools/expensive_traces.py](opentelemetry_mcp/tools/expensive_traces.py) - Find highest token usage traces
- [tools/slow_traces.py](opentelemetry_mcp/tools/slow_traces.py) - Find slowest LLM traces
- [tools/list_llm_tools.py](opentelemetry_mcp/tools/list_llm_tools.py) - List LLM tools used (via traceloop.span.kind == tool)
- [tools/sessions.py](opentelemetry_mcp/tools/sessions.py) - `list_sessions`/`get_session_stats`: group spans by `gen_ai.conversation.id`
- [tools/compare.py](opentelemetry_mcp/tools/compare.py) - `compare_time_windows`: diff aggregated usage between two time ranges
- [tools/investigate.py](opentelemetry_mcp/tools/investigate.py) - `investigate_cost_spike`/`investigate_error_spike`: compare a recent window against a baseline and rank which models/services/error types contributed most to the change
- [tools/prompt_versions.py](opentelemetry_mcp/tools/prompt_versions.py) - `get_prompt_version_stats`: group spans by `gen_ai.prompt.name`/`gen_ai.prompt.version`

**Two valid return patterns exist.** Returning a JSON string was never actually an MCP protocol
requirement - it was this codebase's original convention, from before FastMCP auto-derived
`outputSchema`/`structuredContent` from a tool function's return-type annotation. Most tools
still use the original pattern; `search_traces`, `search_spans_tool`, `list_sessions`, `get_trace`,
and `triage_trace` were converted to (or, for the latter two, built directly on) the newer pattern
(see [models.py](opentelemetry_mcp/models.py)'s `SearchTracesResult`/`SearchSpansResult`/
`TraceDetail`/`TriageResult` and [tools/sessions.py](opentelemetry_mcp/tools/sessions.py)'s
`ListSessionsResult`) so MCP clients get real structured output instead of a JSON string wrapped
in `{"result": "<json string>"}`.

```python
# Pattern A (most tools): return a JSON string
return json.dumps({"result": data})

# Pattern B (search_traces, search_spans_tool, list_sessions, get_trace, triage_trace): return a typed model
return SearchTracesResult(count=len(summaries), traces=summaries)
```

Whichever pattern a given tool already uses, keep using it - don't mix a `-> str` return
annotation with a raw dict return (that part of the original rule still holds: `return {"result":
data}` from a `-> str`-annotated tool breaks the protocol).

### Key Components

- [server.py](opentelemetry_mcp/server.py) - FastMCP application, CLI interface, tool handlers
- [config.py](opentelemetry_mcp/config.py) - Pydantic configuration models
- [models.py](opentelemetry_mcp/models.py) - Core data models (SpanData, TraceData, UsageMetrics)
- [attributes.py](opentelemetry_mcp/attributes.py) - Strongly-typed OpenTelemetry attribute models

## Configuration

**Environment Variables** (see [.env.example](.env.example)):

- `BACKEND_TYPE` - Required: `jaeger`, `tempo`, `traceloop`, `datadog`, `sentry`, `xray`, `newrelic`, or `honeycomb`
- `BACKEND_URL` - Required: Backend API endpoint (decorative-only placeholder for X-Ray - see config.py's `BackendConfig.url` docstring)
- `BACKEND_API_KEY` - Optional: Authentication key (unused by X-Ray, which authenticates via boto3's standard AWS credential chain)
- `BACKEND_APP_KEY` - Required for Datadog backend only: Application key, in addition to `BACKEND_API_KEY`
- `BACKEND_SENTRY_ORG` - Required for Sentry backend only: Organization slug
- `BACKEND_SENTRY_PROJECT` - Optional for Sentry backend: Project slug to narrow queries
- `BACKEND_AWS_REGION` - Required for X-Ray backend only: AWS region (e.g. `us-east-1`)
- `BACKEND_NEWRELIC_ACCOUNT_ID` - Required for New Relic backend only: numeric account ID (NerdGraph queries are user-scoped, not account-scoped)
- `BACKEND_HONEYCOMB_DATASET` - Required for Honeycomb backend only: dataset slug (the Query Data API is dataset-scoped)
- `BACKEND_TIMEOUT` - Optional: Request timeout (default: 30s)
- `BACKEND_TEMPO_INSTANCE_ID` - Optional for Tempo backend: Grafana Cloud stack/instance ID (Basic Auth with `BACKEND_API_KEY` instead of Bearer auth)
- `SECONDARY_BACKEND_TYPE` / `SECONDARY_BACKEND_URL` / ... - Optional, opt-in: a second, independently-configured backend for `correlate_trace` (cross-backend correlation) - every `BACKEND_*` variable above has a `SECONDARY_BACKEND_*` equivalent. Env-var only (no CLI override), see `config.py`'s `BackendConfig.from_env_optional`
- `LOG_LEVEL` - Optional: Logging level (default: INFO)
- `MAX_TRACES_PER_QUERY` - Optional: Result limit (default: 500, 1-1000)
- `SLOW_REQUEST_THRESHOLD_MS` - Optional: Logs a WARNING for any backend request slower than this, independent of `LOG_LEVEL` (default: unset/disabled)
- `MCP_TRANSPORT` / `MCP_HOST` / `MCP_PORT` - Optional: Env-var equivalents of `--transport`/`--host`/`--port` (see CLI flags below)
- `MCP_INCLUDE_ARGS_IN_SPANS` - Optional: Env-var equivalent of `--include-args-in-spans` (default: `false`). Known credential shapes are redacted before export (`observability._redact_secrets`) - a pattern match, not a guarantee.
- `OTEL_EXPORTER_OTLP_ENDPOINT` - Optional: Enables opt-in OTel self-instrumentation of tool calls when set (unset by default - no TracerProvider is configured and no middleware is registered, so there is zero overhead if you don't opt in)
- `OTEL_SERVICE_NAME` - Optional: Service name reported in self-instrumentation spans (default: `tracehub-mcp`)
- `RATE_LIMIT_MAX_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS` - Optional: Env-var equivalents of `--rate-limit-max-requests`/`--rate-limit-window-seconds` (HTTP transport only, defaults: `100`/`60.0`)
- `SHUTDOWN_DRAIN_SECONDS` - Optional: Env-var equivalent of `--shutdown-drain-seconds` (HTTP transport only, default: `0.0`/disabled)
- `BACKEND_CLOSE_TIMEOUT_SECONDS` - Optional: Env-var equivalent of `--backend-close-timeout-seconds` (HTTP transport only, default: `5.0`)
- `GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS` - Optional: Env-var equivalent of `--graceful-shutdown-timeout-seconds` (HTTP transport only, default: `2`)

**CLI Flags** (all mirror an env var above where noted; run `tracehub-mcp --help` for the full list):

- `--log-level [DEBUG|INFO|WARNING|ERROR]` - Overrides `LOG_LEVEL`
- `--max-traces-per-query <1-1000>` - Overrides `MAX_TRACES_PER_QUERY`
- `--disable-tools <name1,name2,...>` - Removes the named tools from this server instance (applied after `--enabled-tools`)
- `--enabled-tools <name1,name2,...>` - Allowlist: every tool not named here is removed from this server instance
- `--slow-request-threshold-ms <float>` - Overrides `SLOW_REQUEST_THRESHOLD_MS`
- `--include-args-in-spans` - Overrides `MCP_INCLUDE_ARGS_IN_SPANS`
- `--rate-limit-max-requests <int>` / `--rate-limit-window-seconds <float>` - HTTP transport only. Per-client-IP fixed-window rate limit (defaults: 100 requests / 60s window). Set `--rate-limit-max-requests 0` to disable.
- `--shutdown-drain-seconds <float>` - HTTP transport only. On shutdown, wait this many seconds inside the ASGI shutdown handler (after uvicorn has already finished draining in-flight connections) before closing the shared backend HTTP client (default: `0.0`/disabled).
- `--backend-close-timeout-seconds <float>` - HTTP transport only. Abandon closing the shared backend HTTP client during shutdown if it does not finish within this many seconds - uvicorn places no timeout of its own around this wait (default: `5.0`).
- `--graceful-shutdown-timeout-seconds <int>` - HTTP transport only. uvicorn's own bound on waiting for in-flight connections/tasks to finish on shutdown before cancelling them - passed straight through as `uvicorn.Config(timeout_graceful_shutdown=)`, which only accepts whole seconds (default: `2`).
- `--print-config` - Print the resolved configuration as JSON and exit without starting the server (secrets reported as `*_set` booleans, never their actual value)

**`tracehub-mcp doctor`** - Subcommand (not a flag) that runs startup diagnostics against the resolved backend config: config load, backend construction, `health_check()`, and a real read-only connectivity probe (`list_services`). Also validates the optional secondary backend (`SECONDARY_BACKEND_*`) the same way, if one is configured. Prints `[OK]`/`[FAIL]` per step and exits non-zero on any failure - unlike the server's own lazy backend initialization, which deliberately swallows a failed health check and keeps running. Accepts the same `--backend`/`--url`/etc. override flags as the main command, so a candidate config can be validated before committing to it (the secondary backend has no such override flags - it's resolved from env vars only).

**HTTP Endpoints** (streamable-http transport only, registered via `FastMCP.custom_route` - not applicable to stdio): `GET /health` - liveness only, always `200 {"status": "ok"}`, no backend I/O. `GET /ready` - readiness, reuses the server's own cached backend (never closes it), `200` when `health_check`/`list_services` both succeed, `503` with details otherwise.

**Configuration Precedence:** CLI args > environment variables > defaults

**Known third-party egress:** FastMCP itself (not this project's code) checks PyPI for a newer FastMCP release every 12h when it prints the startup banner. Disable with `FASTMCP_CHECK_FOR_UPDATES=off`, or suppress the banner entirely with `FASTMCP_SHOW_SERVER_BANNER=false` (both are FastMCP's own env vars, not tracehub-mcp's). See `.venv/.../fastmcp/utilities/version_check.py` / `settings.py` for the implementation.

## Opentelemetry Semantic Conventions

The server parses both current and legacy Opentelemetry conventions:

**Primary (gen_ai.\*):**

- `gen_ai.system` - LLM provider (e.g., "openai", "anthropic")
- `gen_ai.request.model` - Model name (e.g., "gpt-4", "claude-3-opus")
- `gen_ai.usage.prompt_tokens` / `gen_ai.usage.input_tokens` - Input tokens
- `gen_ai.usage.completion_tokens` / `gen_ai.usage.output_tokens` - Output tokens
- `gen_ai.response.finish_reasons` - Completion reasons

**Legacy (llm.\*):** Supported for backward compatibility

- `llm.system`, `llm.request.model`, etc.

**Token Naming Variations:**

- OpenAI: `prompt_tokens`, `completion_tokens`
- Anthropic: `input_tokens`, `output_tokens`

Parse attributes using: `LLMSpanAttributes.from_span(span_data)`

## Key Development Patterns

### 1. Type Safety

- Full type annotations required (MyPy strict mode)
- Use Pydantic models for all data structures
- Type checking must pass before committing

### 2. Async-First

- All backend operations are async
- Use `async with` for backend context managers
- Always `await` backend method calls

### 3. Pydantic Models

- Use for data validation and serialization
- Serialize with `model_dump(mode="json")` for JSON output
- Models automatically handle validation and type conversion

### 4. Error Handling

- Tools must let exceptions propagate (raise `ValueError`/pydantic `ValidationError` for bad input,
  let backend exceptions bubble up as-is) rather than catching them and returning `{"error": ...}`
  JSON. `server.py`'s shared `_handle_tool_error` logs then re-raises; letting the exception
  reach the MCP SDK's lowlevel server is what makes it build `CallToolResult(isError=True)` per
  SEP-2140. A tool function that swallows an exception and returns a normal JSON string produces
  `isError=False` even though the call failed - a client checking the spec-correct `isError` flag
  would see it as a success.
- Backend health checks are non-blocking
- Server starts even if backend is initially unhealthy

### 5. Adding New Backends

1. Create new file in [opentelemetry_mcp/backends/](opentelemetry_mcp/backends/)
2. Extend `BaseBackend` class
3. Implement all abstract methods
4. Add a branch to the `_create_backend` factory in [server.py](opentelemetry_mcp/server.py)
   (not config.py - config.py only holds the config *schema*). Any field the backend needs
   beyond `url`/`api_key`/`timeout` (e.g. Sentry's `org_slug`, X-Ray's `aws_region`) is read
   unconditionally from `BackendConfig` regardless of `type` - there is no per-backend-type
   *validation* at the config layer; "required for this backend" is enforced inside the
   concrete backend's own `__init__` via a `ValueError`, not in config.py
5. Add the new backend name to every hardcoded backend-name literal site (must update in
   lockstep): [config.py](opentelemetry_mcp/config.py)'s `BackendConfig.type` `Literal` (and its
   `from_env`/`apply_cli_overrides` inline validation lists),
   [attributes.py](opentelemetry_mcp/attributes.py)'s `HealthCheckResponse.backend` `Literal`, and
   [server.py](opentelemetry_mcp/server.py)'s `click.Choice` for the `--backend` CLI flag

### 6. Adding New Tools

1. Create new module in [opentelemetry_mcp/tools/](opentelemetry_mcp/tools/)
2. Implement tool function that takes backend and returns JSON string
3. Register in [server.py](opentelemetry_mcp/server.py) using `@mcp.tool()`

## LLM-Oriented Tools

The server provides specialized tools optimized for LLM observability and cost optimization:

### `list_llm_models` - Model Discovery

Lists all LLM models being used with usage statistics.

**Use Cases:**

- Discover shadow AI usage (unauthorized models)
- Track model adoption across services
- Identify deprecated models still in use

**Parameters:**

- `start_time`, `end_time` - Time range filter
- `service_name` - Filter by service
- `gen_ai_system` - Filter by provider (openai, anthropic, etc.)
- `limit` - Max traces to analyze (default: 1000)

**Returns:** List of models with `request_count`, `first_seen`, `last_seen`

**Example:** "What models is my production service using?"

### `get_llm_model_stats` - Performance Analysis

Get detailed performance statistics for a specific model including latency percentiles, token usage, error rates, and finish reason distributions.

**Use Cases:**

- Compare model performance (gpt-4 vs claude-3-opus)
- Identify problematic models with high error rates
- Analyze finish reasons to detect truncation issues

**Parameters:**

- `model_name` - Model to analyze (required)
- `start_time`, `end_time` - Time range filter
- `service_name` - Filter by service

**Returns:**

- Request/success/error counts and rates
- Duration percentiles (mean, median, p50, p95, p99)
- Token usage percentiles (prompt, completion, total)
- Finish reasons breakdown

**Example:** "How is gpt-4 performing this week?"

### `get_llm_expensive_traces` - Cost Optimization

Find traces with highest token usage for cost optimization.

**Use Cases:**

- Identify inefficient prompts consuming excessive tokens
- Find runaway requests with huge context windows
- Prioritize optimization efforts on high-cost operations

**Parameters:**

- `limit` - Number of traces to return (default: 10)
- `start_time`, `end_time` - Time range filter
- `min_tokens` - Minimum token threshold
- `service_name`, `gen_ai_request_model`, `gen_ai_response_model` - Filters

**Returns:** Top N traces sorted by total tokens with breakdown (prompt/completion/total)

**Example:** "Show me the 10 most expensive requests today"

### `get_llm_slow_traces` - Performance Optimization

Find slowest LLM traces by duration to identify latency bottlenecks.

**Use Cases:**

- Identify slow operations affecting user experience
- Debug timeout issues
- Optimize critical path operations

**Parameters:**

- `limit` - Number of traces to return (default: 10)
- `start_time`, `end_time` - Time range filter
- `min_duration_ms` - Minimum duration threshold
- `service_name`, `gen_ai_request_model`, `gen_ai_response_model` - Filters

**Returns:** Top N traces sorted by duration with token counts and model info

**Example:** "What are the slowest LLM calls in the last hour?"

### `list_sessions` / `get_session_stats` - Conversation/Session Analysis

Group spans by the `gen_ai.conversation.id` attribute - a real, cross-industry
OTel semantic convention (Weave, LangWatch, Sentry, Google ADK, Azure SDK,
OpenLit, Dynatrace) for session/conversation grouping.

**Use Cases:**

- Understand multi-turn conversation activity without correlating spans by hand
- Track per-conversation token usage and cost
- Drill into a single conversation's request/success/error breakdown

**Parameters (`list_sessions`):**

- `start_time`, `end_time` - Time range filter
- `service_name` - Filter by service name
- `gen_ai_system` - Filter by LLM provider
- `limit` - Maximum spans to analyze (default: 1000)

**Returns (`list_sessions`):** One row per conversation - `conversation_id`, `span_count`, `services`, `first_seen`/`last_seen`, `total_tokens` - sorted by span count descending.

**Parameters (`get_session_stats`):** `conversation_id` (required), plus `start_time`, `end_time`, `service_name`, `limit`.

**Returns (`get_session_stats`):** Span/service counts, LLM request/success/error counts and rates, duration and token percentiles, finish reasons - for every span sharing that conversation ID.

**Example:** "What happened in conversation abc-123?"

### `compare_time_windows` - Period-over-Period Comparison

Runs the same usage aggregation for two time ranges and returns the delta -
composition over `get_llm_usage`, not new aggregation logic.

**Use Cases:**

- "This week vs last week" or "before/after a deploy" comparisons
- Spot a regression in request volume or token usage between two periods

**Parameters:**

- `range_a_start`/`range_a_end`, `range_b_start`/`range_b_end` - Time range filters (ISO 8601)
- `service_name`, `gen_ai_system`, `gen_ai_request_model`, `gen_ai_response_model` - Filters (applied to both ranges)
- `limit` - Maximum traces to analyze per range (default: 1000)

**Returns:** `range_a` and `range_b` (each the full `get_llm_usage` shape) plus a `delta` with `change`/`percent_change` per metric (`percent_change` is `null`, not an error, when the range A value is 0).

**Example:** "Compare this week's token usage to last week's"

### `investigate_cost_spike` / `investigate_error_spike` - On-Demand Anomaly Investigation

Agent-invoked, on-request analysis (not a push alert) - mirrors SigNoz's
shipped "investigate telemetry cost" skill. Compares a recent window against
a baseline (auto-computed as the same duration immediately preceding the
recent window if not given explicitly) and ranks which models/services/error
types contributed most to the change, instead of just diffing top-line
totals like `compare_time_windows`.

**Use Cases:**

- "Why did our LLM bill spike this week?" - ranks which model/service is responsible
- "Is this error rate increase a real spike?" - `investigate_error_spike`'s `is_spike` requires both an absolute count floor and a relative rate multiplier, so a tiny sample (1 error becoming 2) doesn't false-positive

**Parameters (`investigate_cost_spike`):** `recent_start`/`recent_end` (required), `baseline_start`/`baseline_end` (optional), `service_name`, `gen_ai_system`, `gen_ai_request_model`, `gen_ai_response_model`, `limit`, `top_n` (default 5, max 50).

**Parameters (`investigate_error_spike`):** `recent_start`/`recent_end` (required), `baseline_start`/`baseline_end` (optional), `service_name`, `limit`, `top_n`, `min_error_count_increase` (default 3), `rate_multiplier_threshold` (default 2.0).

**Returns:** `recent`/`baseline` stats, a delta, and ranked `top_model_contributors`/`top_service_contributors` (`investigate_cost_spike`) or `top_service_contributors`/`top_model_contributors`/`top_error_type_contributors` with sample messages (`investigate_error_spike`).

**Example:** "Did our OpenAI costs spike in the last hour?"

### `get_prompt_version_stats` - Prompt Version Performance

Groups spans by `gen_ai.prompt.name` + `gen_ai.prompt.version`, mirroring
Langfuse's shipped per-prompt Metrics tab. Real-world adoption of these two
attributes is still thin (early-stage OTel GenAI semconv), so this tool may
often return an empty list until more instrumentations populate them - that
is expected, not a bug.

**Use Cases:**

- Compare latency/token cost across prompt versions before promoting one to production

**Parameters:**

- `start_time`, `end_time` - Time range filter
- `service_name` - Filter by service name
- `gen_ai_system` - Filter by LLM provider
- `limit` - Maximum spans to analyze (default: 1000)

**Returns:** One row per `(prompt_name, prompt_version)` pair with request count, time bounds, and duration/token percentiles, sorted by request count descending.

**Example:** "How does prompt v2 compare to v1?"

## Enhanced Token Calculation

The server uses an intelligent token calculation strategy that searches for ALL `gen_ai.usage.*` attributes:

1. **Primary:** Use explicit `gen_ai.usage.total_tokens` if present
2. **Fallback 1:** Sum all numeric `gen_ai.usage.*` attributes found
3. **Fallback 2:** Calculate as `prompt_tokens + completion_tokens`

This ensures compatibility with custom token metrics beyond standard prompt/completion/total (e.g., `gen_ai.usage.cached_tokens`, `gen_ai.usage.reasoning_tokens`).

## Finish Reasons Support

The `gen_ai.response.finish_reasons` attribute is now fully supported for debugging:

**Common Finish Reasons:**

- `stop` - Normal completion
- `length` - Hit max_tokens limit (response truncated)
- `content_filter` - Content policy violation
- `tool_calls` / `function_call` - Model requested tool/function execution

**Use Cases:**

- Identify truncated responses: Filter for `finish_reason = "length"`
- Debug content filter issues: Find traces with `finish_reason = "content_filter"`
- Analyze tool usage: Count `finish_reason = "tool_calls"`

**Supported in:**

- `get_llm_model_stats` - Finish reason distribution breakdown
- Parsed in `LLMSpanAttributes` for all LLM spans

## Generic Filter System

The `search_traces` tool supports a powerful generic filter system in addition to legacy simple parameters.

### Filter Structure

Each filter is an object with:

- **field**: Field name in dotted notation (e.g., `"gen_ai.usage.prompt_tokens"`)
- **operator**: Comparison operator (see below)
- **value**: Single value for most operators
- **values**: List of values for `in`, `not_in`, `between` operators
- **value_type**: Type of value(s) - `"string"`, `"number"`, or `"boolean"`

### Supported Operators

**String Operators:**

- `equals`, `not_equals` - Exact string match/mismatch
- `contains`, `not_contains` - Substring match/mismatch
- `starts_with`, `ends_with` - Prefix/suffix match
- `in`, `not_in` - Match against list of values

**Number Operators:**

- `equals`, `not_equals` - Exact numeric match/mismatch
- `gt`, `lt`, `gte`, `lte` - Greater than, less than, greater/less or equal
- `between` - Range check (requires 2 values)
- `in`, `not_in` - Match against list of values

**Boolean Operators:**

- `equals`, `not_equals` - Boolean match/mismatch

**Existence Operators:**

- `exists`, `not_exists` - Check if attribute exists (no value needed)

### Filter Examples

**Find expensive traces (>5000 tokens):**

```json
{
  "field": "gen_ai.usage.total_tokens",
  "operator": "gt",
  "value": 5000,
  "value_type": "number"
}
```

**Filter by multiple models:**

```json
{
  "field": "gen_ai.request.model",
  "operator": "in",
  "values": ["gpt-4", "gpt-4-turbo", "claude-3-opus"],
  "value_type": "string"
}
```

**Find traces with temperature between 0.7 and 1.0:**

```json
{
  "field": "gen_ai.request.temperature",
  "operator": "between",
  "values": [0.7, 1.0],
  "value_type": "number"
}
```

**Find streaming requests:**

```json
{
  "field": "gen_ai.request.is_streaming",
  "operator": "equals",
  "value": true,
  "value_type": "boolean"
}
```

**Check if attribute exists:**

```json
{
  "field": "gen_ai.request.temperature",
  "operator": "exists",
  "value_type": "number"
}
```

**Filter by service name containing "prod":**

```json
{
  "field": "service.name",
  "operator": "contains",
  "value": "prod",
  "value_type": "string"
}
```

### Hybrid Filtering

The server uses a hybrid filtering strategy:

1. **Native Filtering**: Filters are pushed to the backend when supported
2. **Client-Side Filtering**: Unsupported filters are applied after fetching

**Backend Support:**

| Backend             | Supported Operators                                                                       | Notes                               |
| ------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------- |
| **Tempo (TraceQL)** | equals, not_equals, gt, lt, gte, lte, contains (regex), in (OR logic), exists, not_exists | -                                   |
| **Traceloop**       | equals, not_equals, gt, lt, gte, lte                                                      | -                                   |
| **Datadog**         | Most operators via span-search syntax                                                     | Field names validated against an allowlist before being spliced into the query |
| **Sentry**          | Most operators via Discover search syntax                                                 | Same field-name allowlisting as Datadog |
| **Jaeger**          | equals (via tags only)                                                                    | **Requires service_name parameter** |
| **AWS X-Ray**       | equals, not_equals, gt, lt, gte, lte, in - only on `service.name`/`duration`/`status`      | `gen_ai.*` and every other field is never natively filterable via `FilterExpression` - by default OTel attributes land in unindexed segment *metadata*, not indexed *annotations* (undetectable at runtime) - always applied client-side |

### Combining Filters

Multiple filters are combined with AND logic. Examples:

**Find expensive OpenAI traces:**

```json
{
  "filters": [
    {
      "field": "gen_ai.system",
      "operator": "equals",
      "value": "openai",
      "value_type": "string"
    },
    {
      "field": "gen_ai.usage.total_tokens",
      "operator": "gt",
      "value": 5000,
      "value_type": "number"
    }
  ]
}
```

**Find errors in production:**

```json
{
  "filters": [
    {
      "field": "service.name",
      "operator": "contains",
      "value": "prod",
      "value_type": "string"
    },
    {
      "field": "status",
      "operator": "equals",
      "value": "ERROR",
      "value_type": "string"
    }
  ]
}
```

### Backward Compatibility

Legacy simple parameters (service_name, gen_ai_request_model, gen_ai_response_model, etc.) still work and are automatically converted to filters internally. You can mix legacy parameters with explicit filters.

### Backend-Specific Requirements

**Jaeger Backend - service_name Required:**

The Jaeger backend requires the `service_name` parameter for both `search_traces` and `search_spans` operations. This is because:

- Jaeger's API is optimized for per-service queries
- Querying all services would be very expensive (especially for spans)
- Users can easily discover services first using `list_services()`

**Example workflow:**

```python
# 1. First, list available services
services = list_services()

# 2. Then query a specific service
traces = search_traces(service_name="my-service")
spans = search_spans(service_name="my-service")
```

**Error message if service_name is missing:**

```
ValueError: Jaeger backend requires 'service_name' parameter.
Use list_services() to see available services, then specify one with service_name parameter.
```

**Other backends (Tempo, Traceloop):** service_name is optional - they support querying across all services.

## Span Search and LLM Tools Discovery

### `search_spans` - Individual Span Search

Unlike `search_traces` which returns grouped traces, `search_spans` returns individual spans. This is useful for analyzing specific operations or finding spans with certain characteristics.

**Key Features:**

- Search for individual spans across all traces
- Apply filters to span-level attributes
- Useful for finding specific LLM operations (e.g., tool calls, specific model usage)

**Use Cases:**

- Find all LLM tool calls: Filter by `traceloop.span.kind == tool`
- Find all spans using a specific model
- Identify spans with specific attributes or errors
- Analyze individual operation performance

**Example - Find LLM tool calls:**

```python
{
  "filters": [
    {
      "field": "traceloop.span.kind",
      "operator": "equals",
      "value": "tool",
      "value_type": "string"
    }
  ],
  "limit": 100
}
```

**Returns:** List of `SpanSummary` objects containing:

- `trace_id`, `span_id`, `parent_span_id`
- `operation_name`, `service_name`
- `start_time`, `duration_ms`, `status`
- `is_llm_span`, `gen_ai_system`
- `total_tokens` (for LLM spans)

### `list_llm_tools` - Discover LLM Tool Usage

Automatically discovers which tools/functions LLM applications are calling by identifying spans with `traceloop.span.kind == tool`.

**Key Features:**

- Groups tool calls by tool name
- Shows usage statistics per tool
- Tracks which services use each tool
- Provides first/last seen timestamps

**Use Cases:**

- Discover what tools your LLM agents are using
- Track tool adoption across services
- Identify most/least used tools
- Monitor tool usage patterns over time

**Parameters:**

- `start_time`, `end_time` - Time range filter
- `service_name` - Filter by service
- `gen_ai_system` - Filter by LLM provider
- `limit` - Max spans to analyze (default: 1000)

**Returns:** List of tools with:

- `tool_name` - Name of the tool/function
- `usage_count` - Number of times called
- `services` - List of services using this tool
- `first_seen`, `last_seen` - Time range of usage

**Example Query:** "What tools is my production service using?"

**Example Response:**

```json
{
  "count": 5,
  "total_calls": 127,
  "tools": [
    {
      "tool_name": "search_database",
      "usage_count": 45,
      "services": ["chatbot-service", "qa-service"],
      "first_seen": "2024-01-01T10:00:00Z",
      "last_seen": "2024-01-02T15:30:00Z"
    },
    {
      "tool_name": "web_search",
      "usage_count": 38,
      "services": ["research-agent"],
      "first_seen": "2024-01-01T09:00:00Z",
      "last_seen": "2024-01-02T16:00:00Z"
    }
  ]
}
```

## Testing

- Use pytest with async support (pytest-asyncio)
- Fixtures defined in [tests/conftest.py](tests/conftest.py)
- Mock backend responses for tool tests
- All tests must pass before merging

## Python Version

- Minimum: Python 3.11
- CI uses: Python 3.13
- Version file: [.python-version](.python-version)

<!-- mcp-name: io.github.mcpsmiths/tracehub-mcp -->

# tracehub-mcp

[![CI](https://github.com/mcpsmiths/tracehub-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/mcpsmiths/tracehub-mcp/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mcpsmiths/tracehub-mcp/graph/badge.svg)](https://codecov.io/gh/mcpsmiths/tracehub-mcp)
[![PyPI](https://img.shields.io/pypi/v/tracehub-mcp.svg)](https://pypi.org/project/tracehub-mcp/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![mcpsmiths/tracehub-mcp MCP server](https://glama.ai/mcp/servers/mcpsmiths/tracehub-mcp/badges/score.svg)](https://glama.ai/mcp/servers/mcpsmiths/tracehub-mcp)

Also listed on the [official MCP registry](https://registry.modelcontextprotocol.io/v0/servers?search=tracehub) as `io.github.mcpsmiths/tracehub-mcp`.

**Give your AI assistant a direct line into your observability backend.** tracehub-mcp is an MCP (Model Context Protocol) server that lets Claude, Cursor, Windsurf, Gemini CLI, or any MCP client query OpenTelemetry traces from your LLM/GenAI application and reason about them — find expensive calls, debug errors, compare model performance, track token usage — without you copy-pasting trace JSON into a chat window.

It speaks OpenTelemetry's `gen_ai.*` semantic conventions natively, so it understands prompts, completions, token usage, and finish reasons as first-class concepts, not just generic span attributes.

tracehub-mcp started as a fork of [traceloop/opentelemetry-mcp-server](https://github.com/traceloop/opentelemetry-mcp-server) (Apache 2.0) — full attribution and fork history are in [NOTICE](NOTICE). It's grown into a 5-backend, security-hardened server maintained independently under [mcpsmiths](https://github.com/mcpsmiths); see [What's Different From Upstream](#whats-different-from-upstream) below for the parts that are new here.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Supported Backends](#supported-backends)
- [What's Different From Upstream](#whats-different-from-upstream)
- [Installation](#installation)
- [Configuration](#configuration)
- [Security Considerations](#security-considerations)
- [MCP Client Setup](#mcp-client-setup)
- [Tools Reference](#tools-reference)
- [Generic Filter System](#generic-filter-system)
- [Example Queries](#example-queries)
- [Common Workflows](#common-workflows)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)
- [Support](#support)

---

## Quick Start

tracehub-mcp is on PyPI. No install step needed — `uvx` fetches and runs it in one shot:

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "tracehub-mcp": {
      "command": "uvx",
      "args": ["tracehub-mcp"],
      "env": {
        "BACKEND_TYPE": "jaeger",
        "BACKEND_URL": "http://localhost:16686"
      }
    }
  }
}
```

Or from Claude Code directly:

```bash
claude mcp add tracehub-mcp -e BACKEND_TYPE=jaeger -e BACKEND_URL=http://localhost:16686 -- uvx tracehub-mcp
```

**That's it.** Ask your assistant: _"Show me traces with errors from the last hour."_

See [MCP Client Setup](#mcp-client-setup) for Cursor, Windsurf, VS Code, and Gemini CLI, and [Installation](#installation) for `pip`/`pipx`/from-source alternatives.

---

## Supported Backends

- **[Jaeger](https://www.jaegertracing.io/)** — local/self-hosted, the most common open-source trace backend. No auth required.
- **[Grafana Tempo](https://grafana.com/oss/tempo/)** — local or Grafana Cloud, TraceQL-native search.
- **[Traceloop](https://www.traceloop.com/)** — cloud LLM observability platform, API-key auth.
- **[Datadog](https://www.datadoghq.com/)** — cloud APM, requires an API key *and* an Application key.
- **[Sentry](https://sentry.io/)** — cloud or self-hosted, requires an auth token and an organization slug.

All five implement the same `BaseBackend` interface, so every MCP tool works identically regardless of which one you point the server at. See [Configuration](#configuration) for per-backend setup.

---

## What's Different From Upstream

Upstream `opentelemetry-mcp-server` shipped Jaeger, Tempo, and Traceloop. tracehub-mcp adds **Datadog and Sentry** as full backends — not thin wrappers, but complete implementations of every tool (search, span search, trace hydration, service discovery, health checks). Along the way, all five backends — including the three inherited from upstream — were hardened to a consistent bar:

- **HTTPS-only enforcement** on cloud backends. Datadog and Sentry both refuse to start against a plain `http://` URL, because their auth is a bearer token / API+App key pair that has no business going out over plaintext.
- **Query-injection-safe escaping.** Every value spliced into a Datadog span-search query or a Sentry Discover query is escaped and exact-quoted; field *names* (which are less obviously untrusted, since they come from the MCP tool's `filters` parameter) are validated against an allowlist pattern before being spliced into the query string, closing off structural injection through a crafted field name.
- **Bounded pagination** on every backend that paginates via cursor (Datadog, Sentry) — a search stops at the requested `limit` or when the backend stops returning a continuation cursor, whichever comes first, so a single tool call can't degrade into an unbounded crawl.
- **Exact-ID re-verification.** Where a backend's search API can return neighbors instead of an exact match (notably Datadog's trace reconstruction from grouped spans), every result is re-checked against the exact ID that was asked for before being returned.
- **No fabricated data on malformed responses, in the backends we built.** Datadog and Sentry reject a span outright — rather than substituting a placeholder like `now()` for a missing timestamp or a literal `"unknown"` for a missing `service_name`/`operation_name` — since a fabricated value would silently corrupt trace ordering, duration aggregation, and any tool that groups by service or operation. (The three backends inherited from upstream — Jaeger, Tempo, Traceloop — predate this discipline and haven't been retrofitted; that's deliberate scope discipline, not an oversight, mirroring this project's own precedent of not reaching into shared/inherited code without full regression coverage for it.)

All of this is backed by **458 passing tests (2 skipped, zero regressions)**, a clean `ruff check` and `mypy --strict` run, and two rounds of adversarial CodeRabbit review on the new backends.

---

## Installation

tracehub-mcp is [on PyPI](https://pypi.org/project/tracehub-mcp/). Pick whichever of these your workflow already uses — they're equivalent.

### Option 1: `uvx` (no install step)

```bash
uvx tracehub-mcp --backend jaeger --url http://localhost:16686
```

This is what the [Quick Start](#quick-start) config above uses — `uv` fetches the package and runs the `tracehub-mcp` entry point in one shot, nothing left behind on disk between runs.

### Option 2: `pip` / `pipx`

```bash
pipx install tracehub-mcp
# or: pip install tracehub-mcp

tracehub-mcp --backend jaeger --url http://localhost:16686
```

### Option 3: Clone and run from source

```bash
git clone https://github.com/mcpsmiths/tracehub-mcp.git
cd tracehub-mcp
uv sync

uv run tracehub-mcp --backend jaeger --url http://localhost:16686
```

Use this if you're developing locally, want to pin to a specific commit, or want the dev tooling installed (`uv sync --group dev`).

### Option 4: Docker

```bash
docker run --rm -p 8000:8000 \
  -e BACKEND_TYPE=jaeger -e BACKEND_URL=http://host.docker.internal:16686 \
  ghcr.io/mcpsmiths/tracehub-mcp:latest
```

Runs HTTP transport by default (the image's `CMD`); clients connect to `http://localhost:8000/mcp`. This is also the form to use for MCP clients whose config takes a `command`/`args` pair pointing at `docker` directly (Cursor, Windsurf) instead of a local binary.

**Prerequisites:** Python 3.11+, plus [uv](https://github.com/astral-sh/uv) for Options 1 and 3; Docker for Option 4.

---

## Configuration

Configuration comes from environment variables, CLI flags, or both. **Precedence: CLI arguments > environment variables > defaults.**

```bash
# .env (see .env.example)
BACKEND_TYPE=jaeger
BACKEND_URL=http://localhost:16686
```

```bash
# Equivalent via CLI flags
tracehub-mcp --backend jaeger --url http://localhost:16686
```

### All Configuration Options

| Variable                 | Type    | Default  | Description                                                          |
| ------------------------ | ------- | -------- | ---------------------------------------------------------------------|
| `BACKEND_TYPE`           | string  | `jaeger` | Backend type: `jaeger`, `tempo`, `traceloop`, `datadog`, or `sentry`  |
| `BACKEND_URL`            | URL     | -        | Backend API endpoint (required)                                     |
| `BACKEND_API_KEY`        | string  | -        | API key/auth token (required for Traceloop, Datadog, and Sentry)     |
| `BACKEND_APP_KEY`        | string  | -        | Application key (Datadog only, in addition to `BACKEND_API_KEY`)     |
| `BACKEND_TEMPO_INSTANCE_ID` | string | -     | Grafana Cloud stack/instance ID (Tempo only, enables Basic Auth in addition to `BACKEND_API_KEY`) |
| `BACKEND_SENTRY_ORG`     | string  | -        | Organization slug (required for Sentry)                             |
| `BACKEND_SENTRY_PROJECT` | string  | -        | Project slug (optional for Sentry, narrows queries to one project)   |
| `BACKEND_ENVIRONMENTS`   | string  | `prd`    | Comma-separated environments (Traceloop only)                        |
| `BACKEND_TIMEOUT`        | float   | `30`     | Request timeout in seconds                                           |
| `LOG_LEVEL`              | string  | `INFO`   | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` (`--log-level`)    |
| `MAX_TRACES_PER_QUERY`   | integer | `500`    | Server-wide ceiling (1-1000, `--max-traces-per-query`) - caps every tool's `limit` argument before it reaches a backend query, regardless of what the calling agent requests |
| `SLOW_REQUEST_THRESHOLD_MS` | float | unset  | Logs a WARNING for any backend request slower than this, independent of `LOG_LEVEL` (`--slow-request-threshold-ms`) |
| `MCP_TRANSPORT` / `MCP_HOST` / `MCP_PORT` | string/int | `stdio`/`0.0.0.0`/`8000` | Env-var equivalents of `--transport`/`--host`/`--port` |
| `MCP_INCLUDE_ARGS_IN_SPANS` | bool | `false` | Include tool call arguments/results as OTel span attributes when self-instrumentation is enabled below - off by default since they may contain sensitive data. Known credential shapes (Bearer tokens, `api_key=`/`secret=`/`password=`-style fields, AWS/GitHub/common vendor key prefixes) are redacted before export, but this is a pattern match, not a guarantee - trace_id/span_id and other legitimate trace data are deliberately left untouched (`--include-args-in-spans`) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | URL | unset | Enables opt-in OTel self-instrumentation of tool calls when set; unset means zero overhead (no TracerProvider configured, no middleware registered) |
| `OTEL_SERVICE_NAME`      | string  | `tracehub-mcp` | Service name reported in self-instrumentation spans                |

Every backend-related CLI flag has a matching env var (`--backend`/`BACKEND_TYPE`, `--url`/`BACKEND_URL`, `--api-key`/`BACKEND_API_KEY`, `--app-key`/`BACKEND_APP_KEY`, `--tempo-instance-id`/`BACKEND_TEMPO_INSTANCE_ID`, `--sentry-org`/`BACKEND_SENTRY_ORG`, `--sentry-project`/`BACKEND_SENTRY_PROJECT`, `--environments`/`BACKEND_ENVIRONMENTS`). `--disable-tools <name1,name2,...>` / `--enabled-tools <name1,name2,...>` (CLI-only, no env var) remove/allowlist tools for reduced-trust deployments - `--enabled-tools` is applied first, `--disable-tools` on top of whatever it kept. Run `tracehub-mcp --help` for the full list.

### Backend-Specific Setup

<details>
<summary><b>Jaeger</b></summary>

```bash
BACKEND_TYPE=jaeger
BACKEND_URL=http://localhost:16686
```

No API key required. **`search_traces` and `search_spans_tool` both require a `service_name` parameter** — Jaeger's API is optimized for per-service queries, so querying across all services isn't supported. Discover service names first with `list_services`.

</details>

<details>
<summary><b>Grafana Tempo</b></summary>

```bash
BACKEND_TYPE=tempo
BACKEND_URL=http://localhost:3200
```

No API key required for a local/self-hosted install. Search uses [TraceQL](https://grafana.com/docs/tempo/latest/traceql/) under the hood; `service_name` is optional.

For **Grafana Cloud**-hosted Tempo, also set `BACKEND_TEMPO_INSTANCE_ID` to the stack's instance ID and `BACKEND_API_KEY` to a Cloud Access Policy token scoped to `traces:read` — Grafana Cloud requires Basic Auth (instance ID as username, token as password) instead of self-hosted Tempo's Bearer-token auth:

```bash
BACKEND_TYPE=tempo
BACKEND_URL=https://tempo-prod-XX-prod-XX-XXXX.grafana.net
BACKEND_TEMPO_INSTANCE_ID=your_stack_instance_id
BACKEND_API_KEY=your_cloud_access_policy_token
```

</details>

<details>
<summary><b>Traceloop</b></summary>

```bash
BACKEND_TYPE=traceloop
BACKEND_URL=https://api.traceloop.com
BACKEND_API_KEY=your_api_key_here
```

The API key encodes project information — the backend always uses a project slug of `"default"`, and Traceloop resolves the actual project/environment from the key itself.

</details>

<details>
<summary><b>Datadog</b></summary>

```bash
BACKEND_TYPE=datadog
# US site (default): https://api.datadoghq.com
# EU site:            https://api.datadoghq.eu
BACKEND_URL=https://api.datadoghq.com
BACKEND_API_KEY=your_api_key_here
BACKEND_APP_KEY=your_application_key_here
```

Datadog requires **both** an API key and an Application key — a single key is not enough for span/trace queries, even though ingestion only needs the API key. Trace search uses [Datadog's span search query syntax](https://docs.datadoghq.com/logs/explorer/search_syntax/) rather than TraceQL or Jaeger-style tag params, and traces are reconstructed from grouped spans since Datadog has no trace-level lookup endpoint. The backend also refuses a plain `http://` URL — see [What's Different From Upstream](#whats-different-from-upstream).

> **Troubleshooting:** a `403` from the Datadog API almost always means the Application key (not the API key) is missing or invalid. If you're on the EU site, double check `BACKEND_URL` is `https://api.datadoghq.eu`, not the US default.

</details>

<details>
<summary><b>Sentry</b></summary>

```bash
BACKEND_TYPE=sentry
# SaaS (may be region-specific, e.g. https://us.sentry.io):
BACKEND_URL=https://sentry.io
BACKEND_API_KEY=your_auth_token_here
BACKEND_SENTRY_ORG=your-org-slug
# Optional: narrow queries to one project
BACKEND_SENTRY_PROJECT=your-project-slug
```

Sentry requires **both** an auth token *and* an organization slug — every endpoint this backend calls is organization-scoped. Trace search uses [Sentry's search syntax](https://docs.sentry.io/concepts/search/) against the Discover/Explore Events API. Unlike Datadog, Sentry does have a native trace-lookup endpoint, so `get_trace` calls it directly instead of reconstructing a trace from spans — `search_traces` still discovers candidate trace IDs via a span search first, since Sentry's search surface is itself span-centric. Like Datadog, this backend refuses a plain `http://` URL.

> **Troubleshooting:** a `403`/`401` from the Sentry API almost always means the auth token is missing, invalid, or lacks the necessary scopes. A `404` on an org-scoped endpoint usually means the organization slug is wrong. For a self-hosted install, `BACKEND_URL` should be the install's own base URL, not `https://sentry.io`. Some of the tracing endpoints this backend depends on are newer/experimental on Sentry's side and may not be available on every plan or self-hosted version — see the module docstring in [backends/sentry.py](src/opentelemetry_mcp/backends/sentry.py) for specifics.

</details>

### Transport Modes

```bash
# stdio (default) — local use, Claude Desktop, single process
uvx tracehub-mcp
tracehub-mcp                      # pipx/pip install
uv run tracehub-mcp               # from-source install

# HTTP — remote access, multiple clients, network deployment, sample applications
uvx tracehub-mcp --transport http --host 0.0.0.0 --port 8000
tracehub-mcp --transport http --host 0.0.0.0 --port 8000              # pipx/pip install
uv run tracehub-mcp --transport http --host 0.0.0.0 --port 8000       # from-source install
```

With HTTP transport, clients connect to `http://<host>:<port>/mcp` (streamable-HTTP, for compatibility across MCP clients).

---

## Security Considerations

Trace and span data returned by this server — attribute values, error messages, operation names — comes from whatever application your observability backend is instrumenting, not from tracehub-mcp itself. That makes it fundamentally the same category of untrusted external content as a webpage or a file, even though it's a trusted server (this one) handing it back to your MCP client.

- **Treat backend data as untrusted input.** An LLM client consuming trace/span data from tracehub-mcp should apply the same caution it would to any other external tool output — a span attribute or error message is application data to reason about, not an instruction to follow, no matter how it's phrased.
- **This is a known MCP risk category, not a tracehub-mcp-specific one.** OWASP's GenAI Security Project covers it in their [Practical Guide for Secure MCP Server Development](https://genai.owasp.org/resource/a-practical-guide-for-secure-mcp-server-development/), and Anthropic's own engineering guidance, [How We Contain Claude](https://www.anthropic.com/engineering/how-we-contain-claude), states plainly that tool output is an attack surface even when the tool itself is trusted.
- **Practical implication:** if you're querying traces from an application that processes untrusted user input (e.g. a customer-facing chatbot), be aware that adversarial content a user fed into that application could end up in a span attribute this server returns — and from there, in your LLM client's context.

---

## MCP Client Setup

Every example below uses `uvx tracehub-mcp` (no install step). Swap in `tracehub-mcp` (pip/pipx install) or `uv run tracehub-mcp` (from-source, `--directory /absolute/path/to/tracehub-mcp`) if you installed it a different way — see [Installation](#installation).

<details>
<summary><b>Claude Desktop</b></summary>

Config file location:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

**Jaeger** (no auth):

```json
{
  "mcpServers": {
    "tracehub-mcp": {
      "command": "uvx",
      "args": ["tracehub-mcp"],
      "env": {
        "BACKEND_TYPE": "jaeger",
        "BACKEND_URL": "http://localhost:16686"
      }
    }
  }
}
```

**Datadog** (API key + App key):

```json
{
  "mcpServers": {
    "tracehub-mcp": {
      "command": "uvx",
      "args": ["tracehub-mcp"],
      "env": {
        "BACKEND_TYPE": "datadog",
        "BACKEND_URL": "https://api.datadoghq.com",
        "BACKEND_API_KEY": "your_api_key_here",
        "BACKEND_APP_KEY": "your_application_key_here"
      }
    }
  }
}
```

**Sentry** (auth token + org slug):

```json
{
  "mcpServers": {
    "tracehub-mcp": {
      "command": "uvx",
      "args": ["tracehub-mcp"],
      "env": {
        "BACKEND_TYPE": "sentry",
        "BACKEND_URL": "https://sentry.io",
        "BACKEND_API_KEY": "your_auth_token_here",
        "BACKEND_SENTRY_ORG": "your-org-slug"
      }
    }
  }
}
```

If you're running from a clone instead, the bundled wrapper script gives easy backend switching during local dev:

```json
{
  "mcpServers": {
    "tracehub-mcp": {
      "command": "/absolute/path/to/tracehub-mcp/start_locally.sh"
    }
  }
}
```

(the script ships Jaeger/Traceloop/Tempo blocks only, with Jaeger active by default — to switch, comment out the active block and uncomment the one you want; Datadog/Sentry aren't in the script, so add their `export` lines manually).

</details>

<details>
<summary><b>Claude Code</b></summary>

```bash
claude mcp add tracehub-mcp -e BACKEND_TYPE=jaeger -e BACKEND_URL=http://localhost:16686 -- uvx tracehub-mcp
```

Datadog/Sentry work the same way — add more `-e KEY=value` flags for each backend's required env vars (see [Backend-Specific Setup](#backend-specific-setup)). Then:

```bash
claude mcp list
claude "Show me traces with errors from the last hour"
```

</details>

<details>
<summary><b>Cursor</b></summary>

`.cursor/mcp.json` (project-level) or your global Cursor MCP config:

```json
{
  "mcpServers": {
    "tracehub-mcp": {
      "command": "uvx",
      "args": ["tracehub-mcp"],
      "env": {
        "BACKEND_TYPE": "jaeger",
        "BACKEND_URL": "http://localhost:16686"
      }
    }
  }
}
```

</details>

<details>
<summary><b>Windsurf</b></summary>

`~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "tracehub-mcp": {
      "command": "uvx",
      "args": ["tracehub-mcp"],
      "env": {
        "BACKEND_TYPE": "jaeger",
        "BACKEND_URL": "http://localhost:16686"
      }
    }
  }
}
```

</details>

<details>
<summary><b>VS Code (GitHub Copilot)</b></summary>

`.vscode/mcp.json` in your workspace — note the top-level key is `servers`, not `mcpServers`, and stdio servers need no `"type"` field:

```json
{
  "servers": {
    "tracehub-mcp": {
      "command": "uvx",
      "args": ["tracehub-mcp"],
      "env": {
        "BACKEND_TYPE": "jaeger",
        "BACKEND_URL": "http://localhost:16686"
      }
    }
  }
}
```

</details>

<details>
<summary><b>Gemini CLI</b></summary>

Config file: `~/.gemini/config.json`, same JSON shape as Claude Desktop above. Then:

```bash
gemini "Analyze token usage for gpt-4 requests today"
```

</details>

---

## Tools Reference

tracehub-mcp exposes **15 MCP tools**:

| Tool                       | Description                                       | Use Case                           |
| --------------------------- | -------------------------------------------------- | ----------------------------------- |
| `search_traces`            | Search traces with simple params or advanced filters | Find specific requests or patterns |
| `search_spans_tool`        | Search individual spans (not grouped into traces) | Find LLM tool calls, specific ops  |
| `get_trace`                | Get complete trace details by trace ID            | Deep-dive into a single trace      |
| `get_llm_usage`             | Aggregate token usage metrics                     | Track costs and usage trends       |
| `list_services`            | List available services                           | Discover what's instrumented       |
| `find_errors`               | Find traces with errors                           | Debug failures quickly             |
| `list_llm_models`          | Discover models in use, with usage stats          | Track model adoption, shadow AI    |
| `get_llm_model_stats`      | Latency/token percentiles + finish reasons for one model | Compare model efficiency    |
| `get_llm_expensive_traces` | Find highest token-usage traces                   | Cost optimization                  |
| `get_llm_slow_traces`      | Find slowest traces by duration                   | Latency debugging                  |
| `list_llm_tools_tool`      | Discover LLM tool/function calls (`traceloop.span.kind == tool`) | Track agent tool usage |
| `list_sessions`            | Group spans by `gen_ai.conversation.id`           | Understand multi-turn conversation activity |
| `get_session_stats`        | Detailed stats for one conversation ID            | Drill into a single conversation   |
| `compare_time_windows`     | Diff aggregated usage between two time ranges     | "This week vs last week" comparisons |
| `get_prompt_version_stats` | Group spans by `gen_ai.prompt.name`/`.version`    | Compare prompt versions before promoting one |

### Backend Support Matrix

| Feature          | Jaeger | Tempo | Traceloop | Datadog | Sentry |
| ---------------- | :----: | :---: | :-------: | :-----: | :----: |
| Search traces    |   ✓    |   ✓   |     ✓     |   ✓†    |   ✓‡   |
| Search spans     |  ✓\*   |   ✓   |     ✓     |    ✓    |   ✓    |
| Get trace by ID  |   ✓    |   ✓   |     ✓     |   ✓†    |   ✓    |
| Advanced filters |   ✓    |   ✓   |     ✓     |    ✓    |    ✓    |
| Error traces     |   ✓    |   ✓   |     ✓     |    ✓    |    ✓    |
| All LLM tools    |   ✓    |   ✓   |     ✓     |    ✓    |    ✓    |

<sub>\* Jaeger requires the `service_name` parameter for span search.</sub><br>
<sub>† Datadog has no trace-level API; traces are reconstructed by searching spans and grouping by `trace_id`, with every result re-verified against the exact ID requested.</sub><br>
<sub>‡ Sentry does have a native trace-lookup endpoint (unlike Datadog), so `get_trace` calls it directly; `search_traces` still discovers candidate trace IDs via a span search first, since Sentry's search surface is itself span-centric.</sub>

### Key Tool Details

**`search_traces`**

```python
{
  "service_name": "my-app",
  "start_time": "2024-01-01T00:00:00Z",
  "end_time": "2024-01-01T23:59:59Z",
  "gen_ai_system": "openai",
  "gen_ai_request_model": "gpt-4",
  "min_duration_ms": 1000,
  "has_error": false,
  "limit": 50
}
```

Parameters: `service_name`, `operation_name`, `start_time`/`end_time` (ISO 8601), `min_duration_ms`/`max_duration_ms`, `gen_ai_system`, `gen_ai_request_model`, `gen_ai_response_model`, `has_error`, `tags`, `filters` (see [Generic Filter System](#generic-filter-system)), `limit` (1-1000, default 100). Returns trace summaries with token counts.

**`get_trace`**

```python
{ "trace_id": "abc123def456" }
```

Returns the full trace tree: all spans with attributes, parsed OpenTelemetry `gen_ai.*` data for LLM spans, per-span token usage, and error information.

**`get_llm_usage`**

```python
{
  "start_time": "2024-01-01T00:00:00Z",
  "end_time": "2024-01-01T23:59:59Z",
  "service_name": "my-app",
  "gen_ai_system": "openai",
  "limit": 1000
}
```

Returns aggregated prompt/completion/total tokens, broken down by model and by service, plus request counts.

**`list_services`** — no parameters. Returns the list of instrumented service names.

**`find_errors`**

```python
{
  "start_time": "2024-01-15T14:00:00Z",
  "service_name": "my-app",
  "limit": 50
}
```

Returns error messages, error types, truncated stack traces, and LLM-specific error info.

**`list_llm_models` / `get_llm_model_stats` / `get_llm_expensive_traces` / `get_llm_slow_traces` / `list_llm_tools_tool` / `search_spans_tool` / `list_sessions` / `get_session_stats` / `compare_time_windows` / `get_prompt_version_stats`** are documented in detail, with worked examples, in [CLAUDE.md](CLAUDE.md) — this README covers the shape every tool shares; CLAUDE.md is the fuller reference for exact parameters and response fields on the LLM-analysis tools.

---

## Generic Filter System

`search_traces` and `search_spans_tool` both accept a `filters` list in addition to (or instead of) their simple named parameters, for advanced queries. Each filter is:

```json
{
  "field": "gen_ai.usage.total_tokens",
  "operator": "gt",
  "value": 5000,
  "value_type": "number"
}
```

- **`field`** — dotted attribute name, e.g. `gen_ai.usage.prompt_tokens`, `traceloop.span.kind`, `service.name`
- **`operator`** — see table below
- **`value`** — single value (most operators) or **`values`** — list (for `in`, `not_in`, `between`)
- **`value_type`** — `"string"`, `"number"`, or `"boolean"`

| Category    | Operators                                                            |
| ----------- | --------------------------------------------------------------------- |
| String      | `equals`, `not_equals`, `contains`, `not_contains`, `starts_with`, `ends_with`, `in`, `not_in` |
| Number      | `equals`, `not_equals`, `gt`, `lt`, `gte`, `lte`, `between`, `in`, `not_in` |
| Boolean     | `equals`, `not_equals`                                                |
| Existence   | `exists`, `not_exists` (no value needed)                              |

Multiple filters combine with **AND** logic. Legacy simple parameters (`service_name`, `gen_ai_request_model`, etc.) still work and are converted to filters internally — mix and match freely.

The server uses a **hybrid filtering strategy**: filters are pushed to the backend's native query language when supported (TraceQL for Tempo, span-search syntax for Datadog, Discover syntax for Sentry), and applied client-side afterward for anything the backend can't express natively.

| Backend             | Native filter support                                                | Notes                          |
| -------------------- | ---------------------------------------------------------------------- | ------------------------------- |
| **Tempo** (TraceQL)  | `equals`, `not_equals`, `gt`, `lt`, `gte`, `lte`, `contains` (regex), `in` (OR), `exists`, `not_exists` | — |
| **Traceloop**        | `equals`, `not_equals`, `gt`, `lt`, `gte`, `lte`                       | —                                |
| **Datadog**          | Most operators via span-search syntax                                 | Field names validated against an allowlist before being spliced into the query |
| **Sentry**           | Most operators via Discover search syntax                             | Same field-name allowlisting as Datadog |
| **Jaeger**           | `equals` (via tags only)                                               | **Requires `service_name`**    |

Example — expensive OpenAI traces:

```json
{
  "filters": [
    { "field": "gen_ai.system", "operator": "equals", "value": "openai", "value_type": "string" },
    { "field": "gen_ai.usage.total_tokens", "operator": "gt", "value": 5000, "value_type": "number" }
  ]
}
```

For the full semantic-convention attribute list (`gen_ai.*` vs legacy `llm.*`, token-naming variants across providers, finish-reason values, and the token-calculation fallback chain), see [CLAUDE.md](CLAUDE.md#opentelemetry-semantic-conventions).

---

## Example Queries

### Find Expensive OpenAI Operations

**Ask:** _"Show me OpenAI traces from the last hour that took longer than 5 seconds"_

**Tool call:** `search_traces`

```json
{
  "service_name": "my-app",
  "gen_ai_system": "openai",
  "min_duration_ms": 5000,
  "start_time": "2024-01-15T10:00:00Z",
  "limit": 20
}
```

**Response:**

```json
{
  "traces": [
    {
      "trace_id": "abc123...",
      "service_name": "my-app",
      "operation_name": "chat.completions",
      "status": "OK",
      "duration_ms": 8250,
      "span_count": 3,
      "llm_span_count": 1,
      "total_tokens": 4523,
      "has_errors": false
    }
  ],
  "count": 1
}
```

### Analyze Token Usage by Model

**Ask:** _"How many tokens did we use for each model today?"_

**Tool call:** `get_llm_usage`

```json
{
  "start_time": "2024-01-15T00:00:00Z",
  "end_time": "2024-01-15T23:59:59Z",
  "service_name": "my-app"
}
```

**Response:**

```json
{
  "period": { "start_time": "2024-01-15T00:00:00Z", "end_time": "2024-01-15T23:59:59Z" },
  "filters": { "service_name": "my-app" },
  "summary": {
    "total_requests": 487,
    "total_prompt_tokens": 82140,
    "total_completion_tokens": 43290,
    "total_tokens": 125430
  },
  "by_model": {
    "gpt-4": { "requests": 156, "prompt_tokens": 58300, "completion_tokens": 26900, "total_tokens": 85200 },
    "gpt-3.5-turbo": { "requests": 331, "prompt_tokens": 23840, "completion_tokens": 16390, "total_tokens": 40230 }
  },
  "by_service": {
    "my-app": { "requests": 487, "prompt_tokens": 82140, "completion_tokens": 43290, "total_tokens": 125430 }
  }
}
```

### Find Traces with Errors

**Ask:** _"Show me all errors from the last hour"_

**Tool call:** `find_errors`

```json
{
  "start_time": "2024-01-15T14:00:00Z",
  "service_name": "my-app",
  "limit": 10
}
```

**Response:**

```json
{
  "count": 1,
  "error_traces": [
    {
      "trace_id": "def456...",
      "service_name": "my-app",
      "operation_name": "chat.completions",
      "start_time": "2024-01-15T14:23:15Z",
      "duration_ms": 1200,
      "status": "ERROR",
      "span_count": 2,
      "llm_span_count": 1,
      "total_tokens": 310,
      "has_errors": true,
      "error_spans": [
        {
          "span_id": "span789...",
          "operation_name": "chat.completions",
          "service_name": "my-app",
          "status": "ERROR",
          "error_message": "RateLimitError: Too many requests",
          "error_type": "openai.error.RateLimitError",
          "is_llm_error": true,
          "llm_provider": "openai",
          "llm_model": "gpt-4"
        }
      ]
    }
  ]
}
```

### Compare Model Performance

**Ask:** _"What's the performance difference between GPT-4 and Claude?"_

**Tool call 1:** `get_llm_model_stats` for `gpt-4`

```json
{ "model_name": "gpt-4", "start_time": "2024-01-15T00:00:00Z" }
```

**Tool call 2:** `get_llm_model_stats` for `claude-3-opus`

```json
{ "model_name": "claude-3-opus-20240229", "start_time": "2024-01-15T00:00:00Z" }
```

### Investigate High Token Usage

**Ask:** _"Which requests used the most tokens today?"_

**Tool call:** `get_llm_expensive_traces`

```json
{ "limit": 10, "start_time": "2024-01-15T00:00:00Z", "min_tokens": 5000 }
```

---

## Common Workflows

### Cost Optimization

1. `get_llm_expensive_traces` — find the highest-token requests
2. `get_llm_usage` — see which models are costing the most
3. `get_trace` on a specific `trace_id` — inspect the exact prompt/response

### Performance Debugging

1. `get_llm_slow_traces` — identify latency outliers
2. `find_errors` — check for failure patterns
3. `get_llm_model_stats` — check finish-reason distribution for truncation

### Model Adoption Tracking

1. `list_llm_models` — see every model actually being called
2. `get_llm_model_stats` per model — compare performance
3. Scan `list_llm_models` results for unexpected models/services (shadow AI)

---

## Development

```bash
git clone https://github.com/mcpsmiths/tracehub-mcp.git
cd tracehub-mcp
uv sync --group dev   # pulls in pytest, mypy, ruff, etc. for local iteration

# Tests (458 passed, 2 skipped at time of writing)
uv run pytest

# With coverage
uv run pytest --cov=opentelemetry_mcp --cov-report=html

# Format, lint, type-check
uv run ruff format .
uv run ruff check .
uv run mypy src/
```

CI (`.github/workflows/ci.yml`) runs Ruff and mypy (strict) on every push, plus the full pytest suite.

---

## Troubleshooting

**Backend connection issues:**

```bash
curl http://localhost:16686/api/services   # Jaeger
curl http://localhost:3200/api/search/tags  # Tempo
```

**Authentication errors:** confirm your key is set —

```bash
export BACKEND_API_KEY=your_key_here
# or: tracehub-mcp --api-key your_key_here
```

**No traces found:**
- Check the time range (use recent timestamps)
- Verify service names with `list_services`
- Try searching without filters first

**Token usage shows zero:**
- Confirm your traces have OpenTelemetry `gen_ai.*` (or legacy `llm.*`) instrumentation
- Inspect raw span attributes with `get_trace`

**Datadog/Sentry-specific issues:** see the troubleshooting notes under each backend in [Configuration](#configuration).

---

## Roadmap

Two ideas are deliberately **not** built yet — they're being deferred until v0.1 ships and gets real usage feedback, rather than guessed at up front:

- **Cross-backend correlation** — querying multiple configured backends in a single call and correlating results across them (e.g. a Datadog trace and its downstream Sentry error, joined).
- **Agent-native triage** — tools that flag a likely root cause rather than just returning raw trace data, so an agent can act on a diagnosis instead of re-deriving one from a trace dump every time.

Beyond that, the next backends under research (in order, not yet started): **Grafana Cloud**, **New Relic**, **Honeycomb**, **AWS X-Ray**.

Carried over from upstream's older roadmap and still pending, re-prioritized behind the above rather than dropped: cost calculation with built-in pricing tables, model performance comparison tools, prompt pattern analysis, MCP resources for common queries, a caching layer for frequent queries, and SigNoz/ClickHouse backend support.

None of the above is shipped. Everything documented elsewhere in this README is.

---

## Contributing

Contributions are welcome. Before opening a PR, make sure:

1. All tests pass: `uv run pytest`
2. Code is formatted: `uv run ruff format .`
3. No linting errors: `uv run ruff check .`
4. Type checking passes: `uv run mypy src/`

## License

Apache License 2.0 — see [LICENSE](LICENSE). This project is a fork of [traceloop/opentelemetry-mcp-server](https://github.com/traceloop/opentelemetry-mcp-server); full attribution and the fork relationship are documented in [NOTICE](NOTICE).

## Support

- **Issues & feature requests:** [github.com/mcpsmiths/tracehub-mcp/issues](https://github.com/mcpsmiths/tracehub-mcp/issues)
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **Community:** [Traceloop Community Slack](https://traceloop.com/slack) (upstream's community channel; general OpenTelemetry/GenAI-tracing questions welcome)
- **Related:** [Model Context Protocol](https://modelcontextprotocol.io/) · [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) · [traceloop/opentelemetry-mcp-server](https://github.com/traceloop/opentelemetry-mcp-server) (upstream)

# Privacy Policy

tracehub-mcp is an open-source MCP server that you run yourself — as a local process, a
Docker container, or a self-managed HTTP deployment. There is no tracehub-mcp-operated
service, account, or backend. This policy explains what data the software touches and
where it goes.

## Data collection

**tracehub-mcp collects no data on its own.** It has no built-in analytics, telemetry, or
crash reporting that sends anything to its maintainers ([mcpsmiths](https://github.com/mcpsmiths))
or any third party. There is no tracking of any kind baked into the software.

## What data the software handles, and where it goes

- **Backend credentials** (`BACKEND_API_KEY`, `BACKEND_APP_KEY`, etc.) are read from
  environment variables you provide and used only to authenticate to the observability
  backend *you* configure (`BACKEND_URL`). They are never logged, never sent anywhere
  else, and never leave the process — redaction is applied on the one code path that
  could otherwise echo them back (see [Secret Flow](#secret-flow) below).
- **Trace and span data** returned by tools comes from whatever application your own
  observability backend is instrumenting. tracehub-mcp queries that backend using your
  credentials and returns the results to your MCP client (Claude, Cursor, etc.) — it does
  not store, cache, or forward that data anywhere beyond that single request/response.
- **Network calls** the running server makes are limited to: (1) the backend URL you
  configure (Jaeger/Tempo/Traceloop/Datadog/Sentry), and (2) if you enable
  `--transport http`, binding its own listening port for MCP clients to connect to. No
  other outbound network calls happen during normal operation. (Maintainer-only dev
  scripts under `scripts/`, such as the pricing-table sync script, are never invoked by
  the running server and are not part of its runtime behavior.)
- **Self-instrumentation is opt-in and points wherever *you* choose.** Setting
  `OTEL_EXPORTER_OTLP_ENDPOINT` turns on OpenTelemetry tracing of the server's own tool
  calls; those spans go to the OTLP endpoint you configure, not to tracehub-mcp's
  maintainers. It is disabled by default. When enabled, tool call arguments/results
  recorded on these spans pass through a redaction filter for known credential shapes
  before export (`observability._redact_secrets`) — a pattern match, not a guarantee, so
  treat any self-instrumentation collector as seeing the same data your MCP client sees.

## Secret Flow

Secret environment variables (API keys, tokens) are read once at startup, used to build
auth headers for your configured backend, and are never written to logs, never included
in tool output, and never transmitted to any host other than the backend URL you
configured.

## Third parties

tracehub-mcp itself has no third-party data-sharing relationships. Whatever backend you
configure (Datadog, Sentry, Traceloop, etc.) has its own privacy policy governing the
data you send it — that is a relationship between you and that vendor, not something
tracehub-mcp mediates or is party to.

## Data retention

tracehub-mcp itself retains nothing — it is a stateless request/response proxy between
your MCP client and your observability backend. Retention of trace/span data is governed
entirely by your backend's own retention settings.

## Contact

Questions about this policy or the software's data handling: open an issue at
[github.com/mcpsmiths/tracehub-mcp/issues](https://github.com/mcpsmiths/tracehub-mcp/issues).

## Changes to this policy

Material changes to this policy will be reflected in this file's git history and noted
in [CHANGELOG.md](CHANGELOG.md).

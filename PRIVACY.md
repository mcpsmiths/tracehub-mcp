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
  backend(s) *you* configure (`BACKEND_URL`, and `SECONDARY_BACKEND_URL` if set; for AWS
  X-Ray, the AWS X-Ray API). Each backend receives only its own credential. They are
  never logged, never sent anywhere else, and never leave the process — redaction is
  applied on the one code path that could otherwise echo them back (see
  [Secret Flow](#secret-flow) below). `BACKEND_URL`
  itself may not carry a credential: a `user:password@host` URL is rejected at startup,
  and as defence in depth every URL and backend error text that is logged, printed
  (`--print-config`, `doctor`) or returned (tool errors) is stripped of any userinfo
  component first. The unauthenticated `GET /ready` never returns backend text at all:
  a failing probe's `503` body names only an error class (`backend unreachable`,
  `backend timeout`, `backend error <status>`, `backend unhealthy`) and the redacted
  detail is written to the server log once per fresh probe.
- **Server logs** are the only place request data is written, and only in bounded form.
  The HTTP transport keeps uvicorn's per-request access log off, because it would record
  every request target - a token a misconfigured client puts in the query string
  included; the server's own lines carry rejected `Host`/`Origin` values cut and
  repr-quoted, and every record written by the root log handler is stripped of URL
  credentials and of control characters before it reaches the terminal or a log
  collector. Nothing is logged to any destination other than the process's stderr.
- **Trace and span data** returned by tools comes from whatever application your own
  observability backend is instrumenting. tracehub-mcp queries that backend using your
  credentials and returns the results to your MCP client (Claude, Cursor, etc.) — it does
  not persist or forward that data anywhere beyond that single request/response. The one
  exception is the opt-in query cache (`QUERY_CACHE_TTL_SECONDS`, unset by default): when
  enabled, query results are held in the server process's memory for up to the configured
  TTL (at most one day), capped at `QUERY_CACHE_MAX_ENTRIES` results (default 1024, least
  recently used evicted first), shared by every MCP session of that process and discarded
  when it exits. tracehub-mcp's own code never writes trace data or credentials to disk;
  the only file written on its behalf is FastMCP's version-check cache described under
  [Third-party egress](#third-party-egress-fastmcp-update-check).
- **Network calls** the running server makes are limited to: (1) the backend you
  configure (see [Where requests go, per backend](#where-requests-go-per-backend)),
  plus the optional second backend of `SECONDARY_BACKEND_*` if you set one, (2) if you
  enable `--transport http`, binding its own listening port for MCP clients to connect
  to, (3) if you set `OTEL_EXPORTER_OTLP_ENDPOINT`, the OTLP endpoint you chose (see
  below), and (4) under the stdio transport, FastMCP's own update check (see
  [Third-party egress](#third-party-egress-fastmcp-update-check)). No other outbound
  network calls happen during normal operation. (Maintainer-only dev scripts under
  `scripts/`, such as the pricing-table sync script, are never invoked by the running
  server and are not part of its runtime behavior.)
- **Self-instrumentation is opt-in and points wherever *you* choose.** Setting
  `OTEL_EXPORTER_OTLP_ENDPOINT` turns on OpenTelemetry tracing of the server's own tool
  calls; those spans go to the OTLP endpoint you configure, not to tracehub-mcp's
  maintainers. It is disabled by default. When enabled, tool call arguments/results
  recorded on these spans pass through a redaction filter for known credential shapes
  before export (`observability._redact_secrets`) — a pattern match, not a guarantee, so
  treat any self-instrumentation collector as seeing the same data your MCP client sees.

## Where requests go, per backend

tracehub-mcp has no built-in destination for any backend: every HTTP backend sends its
requests to the `BACKEND_URL` you configure (and `SECONDARY_BACKEND_URL` for the optional
second backend used by `correlate_trace` and checked by `doctor`). The tracehub-mcp
maintainers operate none of them. A request to any other origin is refused: the
backend client is pinned to the configured origin (scheme, host and port), so a backend
response that redirects elsewhere is not followed.

| `BACKEND_TYPE` | Where requests go | Credential sent |
| --- | --- | --- |
| `jaeger` | Your Jaeger query service at `BACKEND_URL` (user-configured; the docs use `http://localhost:16686`) | Optional `BACKEND_API_KEY`, as an `Authorization: Bearer` header |
| `tempo` | Your Grafana Tempo at `BACKEND_URL` (user-configured; self-hosted or Grafana Cloud) | `BACKEND_API_KEY` as `Authorization: Bearer`, or `Authorization: Basic` built from `BACKEND_TEMPO_INSTANCE_ID` and the key for Grafana Cloud |
| `traceloop` | The Traceloop API at `BACKEND_URL` (user-configured; the code docs give `https://api.traceloop.com` as the example) | `BACKEND_API_KEY` as `Authorization: Bearer` |
| `datadog` | The Datadog site API at `BACKEND_URL` (user-configured, for example `https://api.datadoghq.com`; `https://` is required) | `DD-API-KEY` from `BACKEND_API_KEY` and `DD-APPLICATION-KEY` from `BACKEND_APP_KEY` |
| `sentry` | Sentry at `BACKEND_URL` (user-configured: `https://sentry.io`, a regional URL, or your self-hosted install; `https://` is required), on organization-scoped paths for `BACKEND_SENTRY_ORG` | `BACKEND_API_KEY` as `Authorization: Bearer` |
| `xray` | The AWS X-Ray API for the region in `BACKEND_AWS_REGION`, through `boto3` with SigV4 signing. `BACKEND_URL` is a placeholder this backend does not contact | Your standard AWS credential chain as resolved by `boto3`; `BACKEND_API_KEY` is unused. Depending on how you configured AWS, that chain may itself call AWS credential endpoints |
| `newrelic` | The New Relic NerdGraph endpoint at `BACKEND_URL` (user-configured, for example `https://api.newrelic.com/graphql` or the EU endpoint; `https://` is required), for the account in `BACKEND_NEWRELIC_ACCOUNT_ID` | `BACKEND_API_KEY` as an `API-Key` header |
| `honeycomb` | The Honeycomb API at `BACKEND_URL` (user-configured, for example `https://api.honeycomb.io` or the EU instance `https://api.eu1.honeycomb.io`; `https://` is required), for the dataset in `BACKEND_HONEYCOMB_DATASET` | `BACKEND_API_KEY` as an `X-Honeycomb-Team` header |

What each request carries is the query the MCP client asked for (time range, filters,
trace IDs, service names) and the credential in the last column. Spans fetched back
are returned to your MCP client, not stored.

## Third-party egress: FastMCP update check

tracehub-mcp is built on [FastMCP](https://gofastmcp.com). Under the stdio transport the
server calls FastMCP's `run`, which by default prints a startup banner; before printing
it, FastMCP checks `https://pypi.org/pypi/fastmcp/json` for a newer FastMCP release (a
2-second request, cached for 12 hours in a `version_cache.json` file in FastMCP's own
user-data directory). That request is made by FastMCP, not by tracehub-mcp's code, and
goes to PyPI, not to the maintainers. Set `FASTMCP_CHECK_FOR_UPDATES=off` to disable the
check, or `FASTMCP_SHOW_SERVER_BANNER=false` to suppress the banner (and with it the
check). The HTTP transport builds its own application and does not print the banner.

## Secret Flow

Secret environment variables (API keys, tokens) are read once at startup, used to build
auth headers for your configured backend, and are never written to logs, never included
in tool output, and never transmitted to any host other than the backend URL you
configured (or, for AWS X-Ray, to the AWS X-Ray API for your configured region) and, if
you configure one, the `SECONDARY_BACKEND_URL` (each backend receives only its own
credential).

## Third parties

tracehub-mcp itself has no third-party data-sharing relationships. Whatever backend you
configure (Jaeger, Tempo, Traceloop, Datadog, Sentry, AWS X-Ray, New Relic or Honeycomb)
has its own privacy policy governing the
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

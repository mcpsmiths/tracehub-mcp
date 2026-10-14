## v0.13.0 (2026-10-14)

### BREAKING CHANGE

- GET /ready now memoizes its probe result for
READINESS_CACHE_SECONDS (default 30 s; a failed probe for 5 s) and may
report a state up to that old; a probe that has not answered after 10 s
returns 503. Set READINESS_CACHE_SECONDS=0 / --readiness-cache-seconds 0
to disable the memo (the timeout and in-flight sharing remain). A
Honeycomb query whose Create Query Result slot would not free within
BACKEND_TIMEOUT now fails with HoneycombRateLimitedError instead of
parking past the timeout; raise BACKEND_TIMEOUT to allow longer waits.
- QUERY_CACHE_TTL_SECONDS / --query-cache-ttl-seconds
above 86400 or non-finite is now a startup error (previously accepted);
the cache evicts least-recently-used results beyond 1024 entries
(QUERY_CACHE_MAX_ENTRIES / --query-cache-max-entries raises it, up to
100000). Values at or below one day and default-sized caches are
unaffected.
- credentials embedded in BACKEND_URL (user:password@host)
are rejected at startup; move them to BACKEND_API_KEY (and
BACKEND_TEMPO_INSTANCE_ID for Basic auth). httpx request lines are logged
only at LOG_LEVEL=DEBUG.
- backend responses larger than BACKEND_MAX_RESPONSE_BYTES
(default 64 MiB decoded) are rejected; raise the variable if legitimate
large trace queries are refused.
- cross-host redirects from a backend are no longer
followed; set BACKEND_URL to the host that actually serves the API.
- a manual mcp-registry-publish dispatch no longer accepts a
`ref`; it publishes server.json from refs/tags/v<version> only.
- a manual docker-publish dispatch now requires the `tag`
input naming an existing vX.Y.Z tag and no longer moves :latest unless
`also_tag_latest` is set; dispatching from a branch is refused.
- a search_traces/search_spans call with more than 50
filters, more than 50 tags, an in/not_in filter with more than 100
values, or a filter field name longer than 256 characters is now rejected
with a validation error before any backend request. There is no opt-out;
callers that need wider queries must split them.
- a filter dict with an unknown key (e.g. logical_operator),
an empty field name, a non-numeric/non-finite operand under value_type
"number", a bool under "number", or anything other than a bool or the
strings "true"/"false" under "boolean" is now rejected with a validation
error instead of being silently ignored or compared as a no-match. Numeric
strings under "number" ("5000") and scalars under "string" (500 -> "500")
keep working. Tempo GT/LT/GTE/LTE filters declared with value_type
"string" are no longer pushed to TraceQL; they are applied client-side.
- with OTEL_EXPORTER_OTLP_ENDPOINT set, HTTP deployments no
longer parent the server's spans under a caller's params._meta.traceparent
by default; set MCP_TRUST_CLIENT_TRACE_CONTEXT=true (or
--trust-client-trace-context) to restore that behaviour for trusted
callers. stdio deployments are unaffected.
- X-Forwarded-For from 127.0.0.1 is no longer honoured by
default. A same-host reverse proxy that previously relied on uvicorn's
implicit loopback trust to key clients individually must now set
FORWARDED_ALLOW_IPS=127.0.0.1 (or --forwarded-allow-ips); otherwise all of
its clients share one rate-limit bucket keyed on the proxy's address.
- a loopback-bound HTTP listener (the new default) now
answers 421 to any request whose Host header is not a loopback name or the
bound address. Deployments that reach a 127.0.0.1-bound server through a
reverse proxy that preserves the public Host must set
MCP_ALLOWED_HOSTS=<those names> (or `*` to disable Host validation). The
0.0.0.0 shape (Dockerfile CMD) is unchanged unless MCP_ALLOWED_HOSTS is set.
- streamable-HTTP sessions idle for 1800 s are now closed
(their id answers 404 and the client must re-initialize) where previously
they lived until the client deleted them or the process stopped. Operators
needing longer-lived idle sessions raise --session-idle-timeout-seconds /
SESSION_IDLE_TIMEOUT_SECONDS.
- --host / MCP_HOST now defaults to 127.0.0.1. --transport
http on a non-loopback host refuses to start (exit code 2) unless
MCP_HTTP_BEARER_TOKEN is set or --allow-unauthenticated-http is passed; the
Docker image (CMD --host 0.0.0.0) therefore needs -e MCP_HTTP_BEARER_TOKEN
or -e MCP_ALLOW_UNAUTHENTICATED_HTTP=true. Clients of a token-protected
server must send `Authorization: Bearer <token>` on every /mcp request.

### Fix

- **verify**: read the archive from the main checkout, timeout pre-#41 server starts, glob the allowlist
- **security**: never report an empty list_services probe error text
- **security**: cut percent-encoded matrix parameters from access-log targets
- **security**: refuse a Host with two or more trailing dots at the request surface
- **security**: treat a non-list Jaeger services payload as an error, not a character count
- **security**: trim only ASCII whitespace around Host values on both sides
- **security**: refuse an allowlist entry with more than one trailing dot
- **security**: cut fragments, matrix parameters and encoded separators from access-log targets
- **security**: rebuild the refused credentialed URL from its parsed parts, never from the text
- **security**: never report an empty health-check error text
- **security**: strip exactly one trailing dot from a loopback Host name
- **security**: name the secondary backend's own variables in its configuration errors
- **security**: anchor the gitleaks fixture allowlist per line, not at the file start
- **security**: keep a bracketed Host with a trailing dot out of the loopback test
- **security**: strip the query string of every request line in a pre-rendered access line
- **security**: make uvicorn's access log an explicit opt-in (--access-log / MCP_ACCESS_LOG)
- **security**: refuse a Host header with text after an IPv6 literal's bracket
- **security**: redact a truncated authority only when its visible head carries a colon
- **security**: bound the backend-chosen host in the origin-refusal text
- **security**: name the option, not pydantic's validator, in configuration errors
- **security**: accept underscores in Host allowlist hostname labels
- **security**: scan a pre-rendered access line linearly and bound its length
- **security**: release a cancelled leader's slot before the finisher can be joined
- **security**: bound and escape the exported span status description
- **security**: ship the image with FastMCP's PyPI update check off
- **security**: allowlist the bearer-auth test fixture for gitleaks' generic rule
- **security**: escape control characters in every root-handler log message
- **security**: redact the truncated authority of a pydantic input_value echo
- **security**: pin backend requests to the configured origin, not the host
- **security**: bound triage chains and name a root-less parent cycle
- **security**: finish a coalesced computation for followers when the leader is cancelled
- **security**: inflate backend bodies incrementally under the byte cap
- **security**: spend one retry and redirect budget per logical request
- **security**: bound every backend request by one total deadline
- **security**: normalise Host allowlist names and IPv6 literals on both sides
- **security**: answer /ready failures with an error class, log the detail
- **security**: apply LOG_LEVEL in doctor so httpx request lines stay at DEBUG
- **security**: validate host, proxy and rate-limit options at startup
- **security**: refuse an empty or non-presentable HTTP bearer token
- **security**: keep query-string tokens out of uvicorn's access log
- **security**: refuse credentialed URLs before pydantic can echo them
- **security**: reject dot-segment Honeycomb result ids before polling
- **security**: fullmatch Honeycomb columns; share limiter per credential
- **security**: probe readiness with live list_services, never the cache
- **security**: bound X-Ray boto3 calls to a private 8-thread executor
- **security**: memoize /ready and cap Honeycomb limiter parking
- **security**: clamp get_llm_model_stats and correlate fallback limits
- **security**: clamp every backend search limit at the backend boundary
- **security**: bound the query cache by entry count and TTL ceiling
- **security**: shield coalesced cache followers from cancellation
- **security**: address review for T2 - cap Content-Length digits
- **security**: address review for T2 - MCPB opt-out for the response cap
- **security**: address review for T2 - judge the whole exception chain
- **security**: address review for T2 - harden userinfo redaction
- **security**: reject BACKEND_URL credentials; redact URL userinfo
- **security**: cap honoured Retry-After at the 10 s retry budget
- **security**: cap decoded backend response size at the transport
- **security**: pin backend requests to the configured host
- **security**: address review for T7 - pin the TruffleHog scanner version
- **security**: address review for T7 - describe only the release controls that exist
- **security**: pin the MCPB launcher to its release and exclude local artifacts
- **security**: scan whole history weekly and detect Traceloop keys
- **security**: bind MCP Registry publishes to the release tag and identity
- **security**: publish GHCR :latest only from an existing release tag
- **security**: pin uv, hatchling and its build closure for the release build
- **security**: bind the PyPI and registry publishers to the release env
- **security**: gate the RELEASE_PAT job on the release environment
- **security**: build the PyPI release from the bump commit, not main
- **security**: bump releases with the hash-pinned cz on the runner
- **security**: hash-pin commitizen in the release job
- **security**: address review for T6 - clamp degenerate durations in ms
- **security**: address review for T6 - one error chain, counted ties
- **security**: cap correlate_trace secondary fan-out at 10 services
- **security**: cap correlate_trace heuristic search window at 24 h
- **security**: make error-chain triage linear in span count
- **security**: address review for T5 - precheck path segments before counting or echoing
- **security**: address review for T5 - reject NUMBER operands beyond float range
- **security**: reject dot-segment ids before building backend URL paths
- **security**: keep span attribute values out of filter-engine warnings
- **security**: compile filter operands once per apply_filters call
- **security**: cap filter values, filters and tags per search query
- **security**: gate non-string Filter operands in TraceQL rendering
- **security**: address review for T1 - reject inf/nan idle timeouts up front
- **security**: address review for T1 - bound Host/Origin rejection log lines
- **security**: own the span sampler; ignore HTTP peers' _meta context
- **security**: bound the tools/call metric label to registered tools
- **security**: trust X-Forwarded-For only from --forwarded-allow-ips
- **security**: validate Host for loopback binds and --allowed-hosts
- **security**: bound idle streamable-HTTP sessions to 1800 s
- **security**: require a bearer token on /mcp; default host loopback
- **release**: bump uv.lock's self-pin in the same commit and keep git noise out of the notes
- sync uv.lock's self-referential version pin to 0.12.3

### Refactor

- **security**: remove the host-only redirect pin from _RetryingTransport

## v0.12.3 (2026-10-10)

### Fix

- **deps**: bump pyjwt to 2.15.1 and urllib3 to 2.8.0 to clear new pip-audit advisories
- **ci**: add explicit restrictive top-level permissions alongside job-level scoping
- **ci**: scope GITHUB_TOKEN permissions to the job that actually needs them
- **ci**: stop Codecov's missing-token failure from blocking every Dependabot PR
- **ci**: align security.yml's Trivy config with ci.yml to stop nightly alert flip-flop
- correct stale backend counts found by a full end-to-end verify
- sync uv.lock's self-referential version pin to 0.12.2
- add codespell false-positive for a deliberate CONTAINS-filter test value
- shorten server.json description under the registry's 100-char limit

## v0.12.2 (2026-09-20)

### Fix

- sync server.json/mcpb manifest with the real 8-backend list

## v0.12.1 (2026-09-20)

### Fix

- **ci**: skip the smoke-test's execution check on non-executable arches
- **ci**: pin the smoke-test container's --platform explicitly
- **ci**: scope the Trivy gate to actionable findings, fix 2 real CVEs
- **ci**: stop the smoke-test step misreading a failed curl as success
- **tests**: resolve mypy errors CI's stricter `mypy .` check caught
- resolve 46 findings from a full-repo security/correctness audit

## v0.12.0 (2026-09-20)

### Feat

- **backends**: add Honeycomb (Query Data API) as an 8th backend
- **backends**: add New Relic (NerdGraph) as a 7th backend
- **tools**: add correlate_trace for best-effort cross-backend correlation
- **tools**: add triage_trace for agent-native root-cause diagnosis

### Fix

- **mcp-registry**: shorten server.json description under the registry's 100-char limit

## v0.11.0 (2026-09-19)

### Feat

- **deps**: upgrade FastMCP 3.2 -> 4.0.5
- **backends**: add AWS X-Ray backend
- **server**: add real graceful shutdown/drain for the HTTP transport

### Fix

- **ci**: pin anyio for a real CVE and disable rate limiting in the conformance job

## v0.10.0 (2026-09-18)

### Feat

- **observability**: add log-to-trace correlation
- **observability**: add opt-in OTel metrics self-instrumentation
- **backends**: add TTL cache + request coalescing for backend queries
- **server**: add per-IP rate limiting on the HTTP transport

## v0.9.0 (2026-09-17)

### Feat

- **server**: add /health and /ready HTTP endpoints
- **trace**: add detail_level to get_trace, convert to typed output

## v0.8.2 (2026-09-17)

### Feat

- **server**: add distinct human-readable title to all 17 tools

### Fix

- **jaeger**: fix typo in null-data comments flagged by codespell
- **jaeger**: handle explicit null data field from list_services/get_service_operations

## v0.8.1 (2026-09-16)

### Fix

- **server**: set destructiveHint explicitly on all 17 read-only tools

## v0.8.0 (2026-09-16)

### Feat

- **trace**: expose span events generically in get_trace output
- **attributes**: add typed field for gen_ai.retrieval.documents
- **attributes**: add typed fields for gen_ai.input.messages/output.messages

### Fix

- **tempo**: parse OTLP array-of-objects attributes correctly

## v0.7.0 (2026-09-16)

### Feat

- **cli**: add doctor subcommand and --print-config flag
- **tools**: add investigate_cost_spike and investigate_error_spike
- **usage**: add cost_usd attribution to get_llm_usage
- **security**: block SSRF to cloud metadata endpoints on BACKEND_URL

## v0.6.0 (2026-09-15)

### Feat

- **tools**: structured output for search_traces, search_spans_tool, list_sessions
- **attributes**: resolve OTel gen_ai.system -> gen_ai.provider.name rename
- **backends**: retry 429 responses with Retry-After-aware backoff

### Fix

- **sentry**: enrich search_traces from search rows, kill redundant get_trace re-fetch
- **sentry**: get_trace no longer rejects every item for a missing trace_id
- **datadog**: correct the custom-attributes key and add real status field support
- **gitignore**: cover .env.<suffix> variants, not just the bare .env file
- **backends**: restore Datadog retry transport, fix NOT_EQUALS status filter, unbreak 2 Sentry tools
- **observability**: redact known credential shapes from span attributes
- **backends**: strip query strings from slow-request log messages
- **server**: actually enforce max_traces_per_query
- **ci**: bump conformance job to Node 22, unblocking the pipeline
- **ci**: match literal strings, not regex, in mcp-registry-publish.yml
- **docker**: replace the no-op HEALTHCHECK with a real one
- **ci**: retry mcp-publisher publish on transient PyPI-lag 404

## v0.5.0 (2026-09-14)

### Feat

- **cli**: add log-level, tool-gating, slow-request, and env-var parity flags
- **observability**: add opt-in OTel self-instrumentation for tool calls
- **models**: surface score.*/evaluation.* attributes in search_spans_tool
- **tools**: add get_prompt_version_stats tool
- **tools**: add compare_time_windows tool
- **tools**: add gen_ai.conversation.id typed field and session tools
- add MCPB bundle source for Smithery publishing

### Fix

- **docker**: stop the default container from crashing on startup
- **tools**: stop swallowing exceptions into fake-success error JSON
- **server**: report tool failures via CallToolResult isError per SEP-2140
- **ci**: poll for PyPI availability in the release smoke test, not a fixed sleep
- **server**: report tracehub-mcp own version in the MCP handshake

## v0.4.0 (2026-09-13)

### Feat

- **attributes**: add gen_ai.system_instructions field
- **backends**: warn on non-local HTTP, Grafana Cloud Tempo auth, retry backoff

### Fix

- **ci**: use a PAT for the release workflow push, not the default token
- **security**: validate Origin header on HTTP transport, mark tools read-only
- **filters**: reject NaN in numeric filters, never raise on malformed finish_reasons

## v0.3.0 (2026-09-13)

### Feat

- publish the Docker image to GHCR on release

### Fix

- stop inlining GitHub Actions expressions into shell script text
- drop the fake auto-succeed visibility step in docker-publish

## v0.2.3 (2026-09-13)

### Fix

- stop the release workflow crashing on a docs-only dispatch

## v0.2.2 (2026-09-12)

### Fix

- use model_validate instead of dict-splat in the new attribute tests
- stop silently dropping spans with array-valued attributes

## v0.2.1 (2026-09-12)

### Fix

- correct 6 bugs found during the test-coverage pass
- attach GitHub Release to the actual tag_format-produced tag

### Perf

- fetch traces concurrently instead of one at a time

## v0.2.0 (2026-09-12)

### Feat

- add Sentry backend (search, get_trace, list_services, health_check)
- add Datadog backend (search/get_trace/list_services/health_check)

### Fix

- correct Commitizen version drift and release.yml git identity
- reject spans missing service/operation identity in Datadog backend
- reject unescaped filter field names in Datadog query builder
- remove hardcoded Traceloop API key from start_locally.sh
- complete the tracehub-mcp rename, remove old-org CI dependencies
- declare explicit hatch wheel package path
- reject entries with non-dict attributes; dedupe test fake-client stubs
- harden _search_spans_raw against a fully malformed response envelope
- address round-2 CodeRabbit findings (query escaping, response validation, get_trace limit)
- address CodeRabbit review findings on the Datadog backend
- **deps**: resolve 2 Dependabot CVEs (pydantic-settings, vcrpy) (#48)
- **deps**: resolve 26 Dependabot CVEs via dependency upgrades (#47)
- **deps**: resolve 12 Dependabot CVEs via dependency upgrades (#34)
- **deps**: upgrade cryptography and fastmcp for security alerts + bumps (#26)

## v0.2.2 (2026-02-08)

### Fix

- **ci**: add GH_TOKEN for release creation (#24)

## v0.2.1 (2026-02-08)

### Fix

- **CI**: add security-events permission for Trivy and clean up unused tokens (#23)

## v0.2.0 (2025-11-17)

### Feat

- **release**: Add configuration for Commitizen and GitHub Actions for release (#8)
- Initial commit of OpenTelemetry MCP Server (#2)

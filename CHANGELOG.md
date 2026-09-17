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

## Unreleased

### Fix

- **`--access-log` / `MCP_ACCESS_LOG` (HTTP transport, default off).** uvicorn's per-request access line was switched off in this round because it logs the full request target, token-in-query-string included; operators who want request lines can opt back in, and the query string is stripped from every access line either way.

- **security**: refuse credentialed URLs before pydantic can echo them <!-- TA1 -->
- **security**: keep query-string tokens out of uvicorn's access log <!-- TA2 -->
- **security**: refuse an empty or non-presentable HTTP bearer token <!-- TA3 -->
- **security**: validate host, proxy and rate-limit options at startup <!-- TA4 -->
- **security**: apply LOG_LEVEL in doctor so httpx request lines stay at DEBUG <!-- TA5 -->
- **security**: answer /ready failures with an error class, log the detail <!-- TA6 -->
- **security**: normalise Host allowlist names and IPv6 literals on both sides <!-- TA7 -->
- **security**: bound every backend request by one total deadline <!-- TB1 -->
- **security**: spend one retry and redirect budget per logical request <!-- TB2 -->
- **security**: inflate backend bodies incrementally under the byte cap <!-- TB3 -->
- **security**: finish a coalesced computation for followers when the leader is cancelled <!-- TB4 -->
- **security**: bound triage chains and name a root-less parent cycle <!-- TB5 -->
- **security**: pin backend requests to the configured origin, not the host <!-- TB6 -->
- **security**: redact the truncated authority of a pydantic input_value echo <!-- TC1 -->
- **security**: escape control characters in every root-handler log message <!-- TC2 -->
- **security**: allowlist the bearer-auth test fixture for gitleaks' generic rule <!-- TC3 -->
- **ci**: install the release toolchain from wheels only <!-- TC4 -->
- **ci**: cross-check the tree version before publishing to the MCP Registry <!-- TC5 -->
- **ci**: fail the smoke test on an empty platform; pass the token by name <!-- TC6 -->
- **ci**: treat a re-dispatch with no new commits as the documented no-op <!-- TC7 -->
- **security**: ship the image with FastMCP's PyPI update check off <!-- TC8 -->
- **security**: bound and escape the exported span status description <!-- TC9 -->
- **docs**: document the round-2 hardening and the operator-facing gaps <!-- TC10 -->
- **security**: release a cancelled leader's slot before the finisher can be joined <!-- IF2 -->
- **security**: scan a pre-rendered access line linearly and bound its length <!-- IF3 -->
- **security**: accept underscores in Host allowlist hostname labels <!-- IF4 -->
- **security**: name the option, not pydantic's validator, in configuration errors <!-- IF5 -->
- **security**: bound the backend-chosen host in the origin-refusal text <!-- IF6 -->
- **security**: redact a truncated authority only when its visible head carries a colon <!-- IF7 -->
- **security**: refuse a Host header with text after an IPv6 literal's bracket <!-- IF9 -->

### Refactor

- **security**: remove the host-only redirect pin from _RetryingTransport <!-- IF1 -->

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

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

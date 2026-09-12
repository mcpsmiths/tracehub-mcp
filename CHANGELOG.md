## [Unreleased]

### Changed

- Forked from traceloop/opentelemetry-mcp-server (Apache 2.0) and renamed to tracehub-mcp, now maintained under github.com/mcpsmiths/tracehub-mcp. See NOTICE for full attribution.
- Added a Datadog backend (search, get_trace, list_services, health_check), fully hardened per CodeRabbit review.

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

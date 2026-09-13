# Security Policy

## Supported Versions

tracehub-mcp is pre-1.0 (`0.x`, [Semantic Versioning](https://semver.org/) with `major_version_zero`).
Only the latest published `0.x` release on [PyPI](https://pypi.org/project/tracehub-mcp/) is supported.
There are no backported security fixes to older `0.x` releases — please upgrade to the latest
release before reporting an issue.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, use GitHub's [Private Vulnerability Reporting](https://github.com/mcpsmiths/tracehub-mcp/security/advisories/new)
(Security tab → "Report a vulnerability"). This opens a private advisory visible only to the
maintainers until a fix is ready.

Include, where relevant:

- A description of the vulnerability and its potential impact
- Steps to reproduce (a minimal repro is ideal)
- The affected version(s)
- Whether it requires a specific backend (Jaeger, Tempo, Traceloop, Datadog, Sentry) to trigger

We aim to acknowledge new reports within a few days. Since this is an independently maintained
open-source project rather than a funded security team, response and fix timelines are
best-effort, not contractual.

## Supply Chain Notes

For anyone auditing this project as a dependency:

- All GitHub Actions third-party steps are pinned to a full commit SHA (not a mutable tag) across
  every workflow.
- CI runs [`ruff`](https://docs.astral.sh/ruff/) with the `S` (flake8-bandit) ruleset and
  [`pip-audit`](https://pypi.org/project/pip-audit/) against the resolved lockfile on every push.
- Published container images (`ghcr.io/mcpsmiths/tracehub-mcp`) include build provenance
  attestation and an SBOM via `docker/build-push-action`.
- Releases are published to PyPI via GitHub Actions using
  [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC), not a long-lived API
  token.

## Scope

This policy covers the tracehub-mcp server code itself (this repository). It does not cover:

- The observability backends it connects to (Jaeger, Grafana Tempo, Traceloop, Datadog, Sentry) —
  report vulnerabilities in those systems to their respective maintainers.
- MCP clients that connect to this server (Claude Desktop, Cursor, etc.).

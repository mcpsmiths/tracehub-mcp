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
- The release toolchain is version- and hash-pinned in-tree: the release job installs Commitizen
  only from `.github/requirements-release.txt` (`pip install --require-hashes --no-deps`); `uv` is
  pinned to the same version as the Dockerfile in every workflow; the build backend (`hatchling`)
  and its build-time dependencies are exact-pinned in `pyproject.toml` and hash-bound through
  `.github/requirements-build.txt` (`uv build --build-constraints ... --require-hashes`). Nothing
  is resolved from PyPI at release time, and the checkout that receives the release token does not
  persist it to disk.
- CI runs [`ruff`](https://docs.astral.sh/ruff/) with the `S` (flake8-bandit) ruleset and
  [`pip-audit`](https://pypi.org/project/pip-audit/) against the resolved lockfile on every push.
- Published container images (`ghcr.io/mcpsmiths/tracehub-mcp`) include build provenance
  attestation and an SBOM via `docker/build-push-action`.
- Releases are published to PyPI via GitHub Actions using
  [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC), not a long-lived API
  token.

### Release authorization

Releases are cut only by dispatching `release.yml` from `main`. The jobs that hold release
credentials declare the `release` GitHub environment, whose deployment-branch policy admits only
`main`: GitHub refuses such a job before its first step when the run's ref is anything else, which
is the control that holds even against a branch carrying an edited copy of the workflow. The
`RELEASE_PAT` used to push the bump commit and tag (a fine-grained token scoped to this repository
with Contents read/write) is stored as a secret of that environment, is handed to the checkout
without being persisted, and reaches the push step only as a step-level variable. The PyPI trusted
publisher for `tracehub-mcp` and the MCP Registry publish job mint their OIDC tokens only from jobs
running under the same environment, so neither can be fed an unreviewed build from another ref.
Only repository administrators can dispatch the workflow; required reviewers on the environment
are off while the project has a single maintainer.

## Scope

Trace and span data returned by this server is treated as untrusted input by design — it
originates from whatever application your observability backend is instrumenting, not from this
project. See [Security Considerations](README.md#security-considerations) in the README for the
full explanation; this document's scope is the vulnerability reporting process below.

This policy covers the tracehub-mcp server code itself (this repository). It does not cover:

- The observability backends it connects to (Jaeger, Grafana Tempo, Traceloop, Datadog, Sentry) —
  report vulnerabilities in those systems to their respective maintainers.
- MCP clients that connect to this server (Claude Desktop, Cursor, etc.).

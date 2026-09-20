# Multi-stage build for OpenTelemetry MCP Server

# Stage 1: Builder with official UV image
#
# MUST track the same Python minor version as .python-version (3.13) and
# the runtime stage's base image below - a venv built here is not
# relocatable across Python minor versions, and the two stages' base images
# are bumped independently by Dependabot. A prior desync (this stage stayed
# on 3.13 while the runtime stage below was bumped to 3.14 by a Docker
# minor/patch-only Dependabot group) silently shipped a container whose
# entrypoint crashed with "ModuleNotFoundError: No module named
# 'opentelemetry_mcp'" on every single run, across 4 published releases -
# `docker build` and the CI Trivy scan both still succeed in that state,
# since neither actually invokes the venv's own interpreter. Bumping this
# stage to 3.14 to match is NOT a safe alternative fix without a dedicated
# upgrade: .python-version pins exactly 3.13, and `uv sync --frozen` with
# UV_PYTHON_DOWNLOADS=0 fails outright under a 3.14-only interpreter search
# path ("No interpreter found for Python 3.13"). See the CI workflow's
# build-test job for the real-container smoke test added to catch this
# class of bug going forward.
#
# Pinned by immutable @sha256 digest (in addition to the tag), mirroring
# every third-party step in .github/workflows/*.yml being SHA-pinned there:
# a mutable tag can be re-pushed (registry compromise or an upstream
# re-tag) and a subsequent build would then silently pull a different
# image with no build-time detection. The tag is kept alongside the digest
# (not replaced) so Dependabot's "docker" ecosystem entry in
# .github/dependabot.yml can keep bumping both together - a digest-only
# reference has no version for Dependabot to track. Digest verified
# 2026-09-20 via `docker buildx imagetools inspect
# ghcr.io/astral-sh/uv:0.12.15-python3.13-trixie-slim` (reads the registry
# API only, no image layers pulled); re-verify the same way after any tag
# bump and update the digest below to match.
FROM ghcr.io/astral-sh/uv:0.12.15-python3.13-trixie-slim@sha256:3ba6b26a3424b592f2dd630450caa526d107597e1cc38f8964a154d4123952b0 AS builder

# Enable bytecode compilation for faster startup and use copy mode
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Disable Python downloads to use system interpreter
ENV UV_PYTHON_DOWNLOADS=0

# Set working directory
WORKDIR /app

# Install dependencies first (cached layer) - this layer is cached between builds
# Uses bind mounts to avoid copying files into intermediate layers
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync --frozen --no-install-project --no-dev

# Copy application code and install project
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Stage 2: Runtime - Minimal production image
#
# MUST track the same Python minor version as the builder stage above - see
# that stage's comment for why.
#
# Pinned by immutable @sha256 digest alongside the tag - see the builder
# stage's FROM comment above for the full rationale (registry-compromise /
# re-tag protection, and why the tag stays so Dependabot can still bump
# this). Digest verified 2026-09-20 via `docker buildx imagetools inspect
# python:3.13-slim-trixie`; re-verify the same way after any tag bump.
FROM python:3.13-slim-trixie@sha256:8d9d0b8bcf6506481eae4907c18f5e3e7902e629f5f6d684f9e7c32e85e3ddf0 AS runtime

# Upgrade packages to get latest security patches
RUN apt-get update && \
    apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*

# The base image's SYSTEM pip vendors its own bundled copy of msgpack
# (pip/_vendor/msgpack) at a version with a known HIGH CVE
# (GHSA-6v7p-g79w-8964), and the base image's system setuptools has its own
# HIGH CVE (CVE-2025-47273) - neither is fixable via `apt-get upgrade`
# above, since both were installed by the upstream base image's own
# ensurepip step, not apt/dpkg. This app never invokes system pip/
# setuptools at runtime - it runs entirely from the uv-managed venv copied
# in below via its own compiled `tracehub-mcp` console-script entrypoint -
# so removing them outright resolves both CVEs and reduces the final
# image's attack surface, rather than trying to patch build-time-only
# tooling that ships in the production image for no reason.
RUN python3 -m pip uninstall -y pip setuptools wheel

# Copy the entire app with virtual environment from builder
COPY --from=builder /app /app

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 mcpuser && \
    chown -R mcpuser:mcpuser /app

# Switch to non-root user
USER mcpuser

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose port for HTTP transport
EXPOSE 8000

# Environment variables (can be overridden at runtime)
# Note: BACKEND_TYPE/BACKEND_URL are intentionally left unset here (not set
# to "") so config.py's own os.getenv(key, default) fallback applies -
# os.getenv only returns its default when the var is truly unset, not when
# it is set to an empty string, so declaring them here with empty values
# previously made every container crash on startup with "Invalid
# BACKEND_TYPE: .". BACKEND_API_KEY should be provided at runtime via:
#   - docker run -e BACKEND_API_KEY=secret
#   - Docker Compose environment files
#   - Kubernetes secrets
#   - .env files mounted at runtime
ENV BACKEND_TIMEOUT="30" \
    LOG_LEVEL="INFO" \
    MAX_TRACES_PER_QUERY="500"

# Health check - see docker_healthcheck.py (already present, copied in via
# COPY --from=builder /app /app above) for what "healthy" means here and
# why stdio-transport deployments should disable this check.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["python", "docker_healthcheck.py"]

# Default command: Run server in HTTP transport mode
# Override with docker run command or docker-compose for different configurations
ENTRYPOINT ["tracehub-mcp"]
CMD ["--transport", "http", "--host", "0.0.0.0", "--port", "8000"]

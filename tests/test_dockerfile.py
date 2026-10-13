"""Tests for the Dockerfile's base-image pinning and runtime environment.

Regression coverage for the finding that both FROM lines (builder:
ghcr.io/astral-sh/uv, runtime: python) were pinned by mutable tag only, not
an immutable @sha256 digest - unlike every third-party step in this repo's
.github/workflows/*.yml files, which are all SHA-pinned. A re-pushed tag
(registry compromise, or an upstream re-tag) could otherwise be pulled
silently on a subsequent build with no build-time detection.

Also pins the image's ENV defaults that keep the shipped container from
making egress this project did not ask for (FastMCP's PyPI update check).
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile"

# Matches a FROM line's image reference, capturing the "@sha256:<64 hex>"
# digest pin if present (group 1 is None when no digest is pinned).
_FROM_LINE_RE = re.compile(
    r"^FROM\s+(\S+?)(@sha256:[0-9a-f]{64})?(?:\s+AS\s+\S+)?\s*$", re.MULTILINE
)


def _from_lines() -> list[tuple[str, str]]:
    """Returns (image, digest) pairs. `digest` is "" (not None) when a FROM
    line has no @sha256 pin - re.findall returns an empty string, never
    None, for a group that didn't participate in the match."""
    text = DOCKERFILE_PATH.read_text()
    return list(_FROM_LINE_RE.findall(text))


def test_dockerfile_has_exactly_two_from_lines() -> None:
    """Sanity check that the regex above is actually matching this
    multi-stage Dockerfile's two stages, so the assertions below aren't
    silently vacuous."""
    assert len(_from_lines()) == 2


def test_all_from_lines_pinned_by_immutable_digest() -> None:
    """Every FROM line must pin an immutable @sha256:<digest> in addition
    to its mutable tag, so a re-pushed tag can never be pulled silently."""
    from_lines = _from_lines()

    unpinned = [image for image, digest in from_lines if not digest]
    assert not unpinned, (
        f"FROM line(s) missing an @sha256 digest pin: {unpinned!r} - a mutable "
        "tag alone lets a re-pushed/compromised registry tag be pulled with "
        "no build-time detection."
    )


def test_builder_stage_digest_pin() -> None:
    """The builder stage's uv image must be pinned by digest alongside its
    tag (tag kept so Dependabot's docker ecosystem entry can still bump it -
    see the Dockerfile's own comment above this FROM line)."""
    from_lines = dict(_from_lines())

    pinned_image = next(
        (image for image in from_lines if image.startswith("ghcr.io/astral-sh/uv:")),
        None,
    )
    assert pinned_image is not None, "builder FROM line must keep its human-readable tag"
    assert from_lines[pinned_image], "builder FROM line must pin an @sha256 digest"


def test_runtime_stage_digest_pin() -> None:
    """The runtime stage's python image must be pinned by digest alongside
    its tag, for the same reason as the builder stage above."""
    from_lines = dict(_from_lines())

    pinned_image = next(
        (image for image in from_lines if image.startswith("python:3.13-slim-trixie")),
        None,
    )
    assert pinned_image is not None, "runtime FROM line must keep its human-readable tag"
    assert from_lines[pinned_image], "runtime FROM line must pin an @sha256 digest"


def _env_assignments() -> dict[str, str]:
    """Every KEY=VALUE set by an ENV instruction in the runtime stage, with
    backslash continuations folded and surrounding quotes removed. Only the
    stage after the last FROM is the shipped image."""
    text = DOCKERFILE_PATH.read_text()
    runtime_stage = text.rsplit("\nFROM ", 1)[1]
    folded = re.sub(r"\\\n\s*", " ", runtime_stage)
    assignments: dict[str, str] = {}
    for line in folded.splitlines():
        if not line.startswith("ENV "):
            continue
        for key, value in re.findall(r'(\w+)=("[^"]*"|\S+)', line[len("ENV ") :]):
            assignments[key] = value.strip('"')
    return assignments


def test_runtime_env_parser_sees_the_documented_defaults() -> None:
    """Sanity check for the parser above so the assertion below cannot pass
    vacuously: the three long-standing defaults must be visible to it."""
    env = _env_assignments()
    assert env["LOG_LEVEL"] == "INFO" and env["BACKEND_TIMEOUT"] == "30", env
    assert env["MAX_TRACES_PER_QUERY"] == "500", env


def test_runtime_image_disables_fastmcps_pypi_update_check_and_banner() -> None:
    """In stdio mode the shipped image contacted https://pypi.org/pypi/fastmcp/json
    at startup (runtime verification, docker-cli 29): FastMCP's own update
    check, which README and CLAUDE.md tell operators to disable with
    FASTMCP_CHECK_FOR_UPDATES=off / FASTMCP_SHOW_SERVER_BANNER=false - but
    the Dockerfile's ENV block set neither, so every container made that
    egress unless the operator repeated the advice. The values are the ones
    fastmcp.settings accepts (Literal "off"; a boolean "false")."""
    env = _env_assignments()
    assert env.get("FASTMCP_CHECK_FOR_UPDATES") == "off", env
    assert env.get("FASTMCP_SHOW_SERVER_BANNER") == "false", env

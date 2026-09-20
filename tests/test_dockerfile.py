"""Tests for the Dockerfile's base-image pinning.

Regression coverage for the finding that both FROM lines (builder:
ghcr.io/astral-sh/uv, runtime: python) were pinned by mutable tag only, not
an immutable @sha256 digest - unlike every third-party step in this repo's
.github/workflows/*.yml files, which are all SHA-pinned. A re-pushed tag
(registry compromise, or an upstream re-tag) could otherwise be pulled
silently on a subsequent build with no build-time detection.
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

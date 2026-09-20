"""Tests for .dockerignore's environment-file exclusion patterns.

Regression coverage for the finding that .dockerignore only excluded the
two specifically-named dotenv entries (.env, .env.local, .env.*.local) and
not the broader .env.* pattern already used by .gitignore - so an arbitrary
dotenv file (e.g. .env.datadog, a real credential file that can exist in
this repo's working tree) would be copied into the Docker build context via
the Dockerfile's `COPY . /app` and remain extractable from a pushed image
layer even after the file is later deleted from disk.
"""

import fnmatch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERIGNORE_PATH = REPO_ROOT / ".dockerignore"


def _read_patterns() -> list[str]:
    lines = DOCKERIGNORE_PATH.read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def _is_excluded(filename: str, patterns: list[str]) -> bool:
    """Mimics .dockerignore's glob-style matching for a single top-level
    filename (no directory components involved for any pattern checked here)."""
    return any(fnmatch.fnmatch(filename, pattern) for pattern in patterns)


def test_dockerignore_excludes_arbitrary_dotenv_variants() -> None:
    """A dotenv file that isn't literally named .env/.env.local/*.local
    (e.g. .env.datadog) must still be excluded from the Docker build
    context - otherwise it gets copied into the image via `COPY . /app`."""
    patterns = _read_patterns()

    assert _is_excluded(".env.datadog", patterns), (
        ".dockerignore must exclude arbitrary .env.* variants, not just "
        "the specifically-named .env/.env.local/.env.*.local entries, or a "
        "credential file like .env.datadog leaks into the built image."
    )


def test_dockerignore_has_broad_env_pattern_matching_gitignore() -> None:
    """.dockerignore's env exclusion should mirror .gitignore's convention
    of a broad `.env.*` pattern as the primary safeguard, kept alongside
    (not instead of) the specific entries in case exceptions are needed."""
    patterns = _read_patterns()

    assert ".env.*" in patterns
    assert ".env" in patterns
    assert ".env.local" in patterns

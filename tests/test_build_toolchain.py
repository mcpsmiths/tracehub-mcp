"""The PyPI release artifact must be a function of reviewed repository content only.

`uv build` in release.yml used to run with `astral-sh/setup-uv` at
`version: "latest"`, `[build-system] requires = ["hatchling"]` (no bound) and no
constraint or hash flags, so the uv binary, the build backend and its
build-time dependencies were all resolved at publish time. These tests pin the
toolchain in-tree: one uv version shared with the Dockerfile, an exact hatchling
pin that `[tool.uv] build-constraint-dependencies` and the hash-bound
.github/requirements-build.txt must both agree with, and the `uv build` flags
that make any upstream change fail the build instead of shipping.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import yaml
from packaging.requirements import Requirement

from tests.test_workflow_policy import _hashed_requirement_lines

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
REQUIREMENTS_BUILD = REPO_ROOT / ".github" / "requirements-build.txt"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
RELEASE_YML = WORKFLOWS_DIR / "release.yml"
DOCKERFILE = REPO_ROOT / "Dockerfile"

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
_DOCKERFILE_UV_RE = re.compile(r"^FROM ghcr\.io/astral-sh/uv:(\d+\.\d+\.\d+)-", re.MULTILINE)


def _pyproject() -> dict[str, Any]:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def _exact_pin(spec: str) -> tuple[str, str]:
    """Return (name, version) for a requirement that is exactly one `==` pin."""
    requirement = Requirement(spec)
    specifiers = list(requirement.specifier)
    assert len(specifiers) == 1 and specifiers[0].operator == "==", (
        f"{spec!r} is not an exact `==` pin"
    )
    return requirement.name.lower(), specifiers[0].version


def _build_constraint_pins() -> dict[str, str]:
    tool_uv = _pyproject()["tool"]["uv"]
    assert "build-constraint-dependencies" in tool_uv, (
        "[tool.uv] build-constraint-dependencies is missing from pyproject.toml"
    )
    return dict(_exact_pin(spec) for spec in tool_uv["build-constraint-dependencies"])


def _requirements_build_pins() -> dict[str, str]:
    assert REQUIREMENTS_BUILD.is_file(), (
        ".github/requirements-build.txt is missing - generate it with "
        "`uv pip compile --generate-hashes --python-version 3.13`"
    )
    pins: dict[str, str] = {}
    for line in _hashed_requirement_lines(REQUIREMENTS_BUILD):
        assert "--hash=sha256:" in line, f"unhashed requirement: {line!r}"
        name, version = _exact_pin(line.split(" ", 1)[0])
        pins[name] = version
    assert pins, "requirements-build.txt declares no requirements"
    return pins


def test_build_system_requires_are_exact_pins() -> None:
    requires = _pyproject()["build-system"]["requires"]
    pins = dict(_exact_pin(spec) for spec in requires)
    assert "hatchling" in pins, pins


def test_uv_build_constraints_agree_with_the_build_system_pin() -> None:
    build_system = dict(_exact_pin(spec) for spec in _pyproject()["build-system"]["requires"])
    constraints = _build_constraint_pins()
    assert constraints["hatchling"] == build_system["hatchling"], (constraints, build_system)


def test_requirements_build_is_hash_pinned_and_matches_pyproject() -> None:
    """Dependabot bumps the hashed file but never edits pyproject.toml, so the
    two must be asserted equal or the release would fail closed (good) at
    the first drift - and this test tells the maintainer why."""
    file_pins = _requirements_build_pins()
    build_system = dict(_exact_pin(spec) for spec in _pyproject()["build-system"]["requires"])
    assert file_pins["hatchling"] == build_system["hatchling"], (file_pins, build_system)
    assert file_pins == _build_constraint_pins(), (
        "[tool.uv] build-constraint-dependencies must list exactly the pins of "
        ".github/requirements-build.txt"
    )


def test_every_setup_uv_step_pins_the_dockerfile_uv_version() -> None:
    """One uv version for the whole project: Dependabot's docker ecosystem
    bumps the Dockerfile's tag+digest, and this test forces every workflow's
    setup-uv to follow (Dependabot never edits `with:` inputs)."""
    dockerfile_versions = set(_DOCKERFILE_UV_RE.findall(DOCKERFILE.read_text(encoding="utf-8")))
    assert len(dockerfile_versions) == 1, dockerfile_versions
    (expected,) = dockerfile_versions

    seen = 0
    offenders: list[str] = []
    for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_name, job in workflow["jobs"].items():
            for step in job.get("steps", []):
                if not str(step.get("uses", "")).startswith("astral-sh/setup-uv"):
                    continue
                seen += 1
                version = str(step.get("with", {}).get("version", ""))
                if not _SEMVER_RE.match(version) or version != expected:
                    offenders.append(f"{path.name}:{job_name}: version={version!r}")
    assert seen > 0, "no astral-sh/setup-uv step found"
    assert offenders == [], f"expected uv {expected} everywhere: {offenders}"


def test_release_build_uses_the_hashed_build_constraints() -> None:
    workflow = yaml.safe_load(RELEASE_YML.read_text(encoding="utf-8"))
    runs = [
        str(step.get("run", ""))
        for step in workflow["jobs"]["build-and-publish"]["steps"]
        if "uv build" in str(step.get("run", ""))
    ]
    assert len(runs) == 1, runs
    assert "--build-constraints .github/requirements-build.txt" in runs[0], runs[0]
    assert "--require-hashes" in runs[0], runs[0]

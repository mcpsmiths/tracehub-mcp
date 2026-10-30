"""Tests for pyproject.toml's dependency lower bounds vs. uv.lock.

Regression coverage for the finding that several production dependencies
(mcp, opentelemetry-sdk, opentelemetry-exporter-otlp-proto-http, boto3) had
wide, largely-unconstrained lower bounds (e.g. "opentelemetry-sdk>=1.30")
relying entirely on uv.lock for reproducibility within this repo's own
`uv sync --frozen` workflow. If this package is ever installed outside that
path (`pip install` from a wheel/sdist pulling only pyproject.toml
constraints, or a downstream project vendoring it without the lockfile),
the resolver could select an older point release predating a since-patched
CVE, even though no such CVE was identified for the versions this project
actually tests against.
"""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
UV_LOCK_PATH = REPO_ROOT / "uv.lock"

# The specific dependencies the finding called out as needing a tightened
# floor - not every dependency in the project (several others already have
# a tight lower bound, e.g. pydantic>=2.12, or are intentionally left loose
# per their own comment, e.g. anyio).
CHECKED_PACKAGES = [
    "mcp",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp-proto-http",
    "boto3",
]


def _pyproject_dependency_specifiers() -> dict[str, Requirement]:
    with PYPROJECT_PATH.open("rb") as f:
        data = tomllib.load(f)
    requirements = [Requirement(dep) for dep in data["project"]["dependencies"]]
    return {req.name: req for req in requirements}


def _uv_lock_versions() -> dict[str, str]:
    with UV_LOCK_PATH.open("rb") as f:
        data = tomllib.load(f)
    return {pkg["name"]: pkg["version"] for pkg in data["package"]}


def _lower_bound(requirement: Requirement) -> Version | None:
    """Extract the ">=" lower bound from a requirement's specifier set,
    if any."""
    for spec in requirement.specifier:
        if spec.operator == ">=":
            return Version(spec.version)
    return None


def test_checked_dependencies_have_a_lower_bound() -> None:
    """Each dependency this finding covers must declare an explicit ">="
    floor at all - a dependency with no lower bound at all is the most
    permissive case the finding warns about."""
    specifiers = _pyproject_dependency_specifiers()

    for package in CHECKED_PACKAGES:
        assert package in specifiers, f"{package} is missing from pyproject.toml dependencies"
        assert _lower_bound(specifiers[package]) is not None, (
            f"{package} has no '>=' lower bound in pyproject.toml"
        )


def test_dependency_floors_match_locked_versions() -> None:
    """The declared '>=' floor for each checked dependency must be at
    least as high as the version actually locked (and tested against) in
    uv.lock. Before the fix, e.g. opentelemetry-sdk's floor was ">=1.30"
    while uv.lock had 1.44.0 locked - a fresh resolve outside this repo's
    `uv sync --frozen` workflow could then select anything from 1.30 up,
    including versions this project has never run its test suite against."""
    specifiers = _pyproject_dependency_specifiers()
    locked_versions = _uv_lock_versions()

    for package in CHECKED_PACKAGES:
        floor = _lower_bound(specifiers[package])
        assert floor is not None

        locked_version_str = locked_versions.get(package)
        assert locked_version_str is not None, f"{package} not found in uv.lock"
        locked_version = Version(locked_version_str)

        assert floor >= locked_version, (
            f"{package}'s pyproject.toml floor ({floor}) is lower than the "
            f"version actually locked in uv.lock ({locked_version}) - a fresh "
            "resolve outside `uv sync --frozen` could select an untested, "
            "potentially-vulnerable older release."
        )


# First fastmcp release that is clean of GHSA-7xgj-8392-5q4r and GHSA-vqqp-x5qm-5qp4
# (fixed in 4.1.2) AND whose server extra requires certifi>=2026.10.19
# (GHSA-6q48-6c7w-9gw2 / PYSEC-2026-4320; 4.1.2 does not declare that floor, 4.1.3 does).
FASTMCP_FIRST_PATCHED = Version("4.1.3")
FASTMCP_VULNERABLE_RELEASES = ["4.0.5", "4.1.0", "4.1.1", "4.1.2"]


def test_fastmcp_range_excludes_every_vulnerable_release() -> None:
    """A fresh resolve outside `uv sync --frozen` (pip install of the wheel, a
    downstream project without our lockfile) must not be able to pick a fastmcp
    that carries a published advisory, directly or through its certifi floor.
    The upper bound stays below the next minor so an untested 4.2 is never
    selected silently."""
    fastmcp = _pyproject_dependency_specifiers()["fastmcp"]

    floor = _lower_bound(fastmcp)
    assert floor is not None
    assert floor >= FASTMCP_FIRST_PATCHED
    for vulnerable in FASTMCP_VULNERABLE_RELEASES:
        assert not fastmcp.specifier.contains(vulnerable), vulnerable
    assert fastmcp.specifier.contains(FASTMCP_FIRST_PATCHED)
    assert not fastmcp.specifier.contains("4.2.0")

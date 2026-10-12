"""Static policy checks for the release and publish GitHub Actions workflows.

The release pipeline (release.yml -> docker-publish.yml / mcp-registry-publish.yml)
is the last point where this project's published artifacts are decided, so the
properties below are pinned as tests rather than left to review: every byte that
runs while RELEASE_PAT or a PyPI OIDC token is reachable must come from reviewed,
pinned content, and every publishing job must build exactly the tree the release
names. The workflow files are parsed with yaml.safe_load; note that under YAML 1.1
the top-level `on` key parses as the boolean True.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
RELEASE_YML = WORKFLOWS_DIR / "release.yml"
DEPENDABOT_YML = REPO_ROOT / ".github" / "dependabot.yml"
REQUIREMENTS_RELEASE = REPO_ROOT / ".github" / "requirements-release.txt"
SECURITY_MD = REPO_ROOT / "SECURITY.md"

Step = dict[str, Any]
Job = dict[str, Any]


def _load_yaml(path: Path) -> dict[Any, Any]:
    # Keys are not `str` only: the top-level `on` key parses as the boolean True.
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), path
    return data


def _jobs(workflow: dict[Any, Any]) -> dict[str, Job]:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    return jobs


def _steps(job: Job) -> list[Step]:
    steps = job.get("steps", [])
    assert isinstance(steps, list)
    return steps


def _run(step: Step) -> str:
    run = step.get("run", "")
    return run if isinstance(run, str) else ""


def _uses(step: Step) -> str:
    uses = step.get("uses", "")
    return uses if isinstance(uses, str) else ""


def _with(step: Step) -> dict[str, Any]:
    with_block = step.get("with", {})
    return with_block if isinstance(with_block, dict) else {}


def _step_index(steps: list[Step], predicate: Any) -> int:
    matches = [i for i, step in enumerate(steps) if predicate(step)]
    assert len(matches) == 1, f"expected exactly one matching step, found {len(matches)}"
    return matches[0]


def _hashed_requirement_lines(path: Path) -> list[str]:
    """Join a pip requirements file into logical lines (continuations folded).

    Comment and blank lines are dropped, so each returned entry is one
    requirement together with all of its `--hash=` options.
    """
    logical: list[str] = []
    current = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            current += line[:-1].strip() + " "
            continue
        logical.append((current + line).strip())
        current = ""
    assert not current, f"{path} ends inside a continuation"
    return logical


def _pinned_name(logical_line: str) -> str:
    requirement = logical_line.split(" ", 1)[0]
    assert "==" in requirement, f"not an exact pin: {logical_line!r}"
    return requirement.split("==", 1)[0].lower()


# --- bump-version: commitizen installed only from the hash-pinned file ----------


def test_release_jobs_only_pip_install_with_hashes() -> None:
    """Every `pip install` in a job that can hold RELEASE_PAT or publish must
    install only from .github/requirements-release.txt with --require-hashes.
    The `permissions: {}` PyPI smoke test (test-installation) is the one
    legitimate bare install: it exercises the just-published package."""
    jobs = _jobs(_load_yaml(RELEASE_YML))
    offenders: list[str] = []
    for job_name, job in jobs.items():
        if job_name == "test-installation":
            continue
        for step in _steps(job):
            run = _run(step)
            if "pip install" not in run:
                continue
            if "--require-hashes" not in run or "-r .github/requirements-release.txt" not in run:
                offenders.append(f"{job_name}: {step.get('name', '<unnamed>')}")
    assert offenders == [], offenders


def test_requirements_release_pins_commitizen_with_hashes() -> None:
    assert REQUIREMENTS_RELEASE.is_file(), (
        ".github/requirements-release.txt is missing - generate it with "
        "`uv pip compile --generate-hashes --python-version 3.13`"
    )
    lines = _hashed_requirement_lines(REQUIREMENTS_RELEASE)
    assert lines, "requirements-release.txt declares no requirements"
    names = {_pinned_name(line) for line in lines}
    assert "commitizen" in names, names
    for line in lines:
        assert "--hash=sha256:" in line, f"unhashed requirement: {line!r}"


def test_release_pat_checkout_does_not_persist_credentials() -> None:
    """actions/checkout writes its token into .git/config unless told not to;
    every later step in the job (including anything installed at run time)
    could read it from there."""
    jobs = _jobs(_load_yaml(RELEASE_YML))
    pat_checkouts = [
        step
        for job in jobs.values()
        for step in _steps(job)
        if _uses(step).startswith("actions/checkout")
        and "secrets.RELEASE_PAT" in str(_with(step).get("token", ""))
    ]
    assert pat_checkouts, "no checkout step receives RELEASE_PAT - did the job change shape?"
    for step in pat_checkouts:
        assert _with(step).get("persist-credentials") is False, step


def test_dependabot_tracks_the_hashed_github_requirements() -> None:
    updates = _load_yaml(DEPENDABOT_YML)["updates"]
    pip_dirs = {u["directory"] for u in updates if u["package-ecosystem"] == "pip"}
    assert "/.github" in pip_dirs, pip_dirs


def test_security_md_describes_the_hash_pinned_release_toolchain() -> None:
    text = SECURITY_MD.read_text(encoding="utf-8")
    assert "requirements-release.txt" in text or "hash-pinned" in text


def test_workflow_on_key_parses_as_boolean_true_under_yaml_1_1() -> None:
    """Documents the yaml.safe_load quirk every test in this module relies on."""
    workflow = _load_yaml(RELEASE_YML)
    assert True in workflow and "on" not in workflow


def test_no_secret_is_interpolated_into_a_run_body() -> None:
    """`${{ secrets.* }}` inlined into a run body is written into the script
    text itself; secrets must only reach steps through `with:` or `env:`."""
    pattern = re.compile(r"\$\{\{\s*secrets\.")
    offenders = [
        f"{path.name}:{job_name}:{step.get('name', '<unnamed>')}"
        for path in sorted(WORKFLOWS_DIR.glob("*.yml"))
        for job_name, job in _jobs(_load_yaml(path)).items()
        for step in _steps(job)
        if pattern.search(_run(step))
    ]
    assert offenders == [], offenders

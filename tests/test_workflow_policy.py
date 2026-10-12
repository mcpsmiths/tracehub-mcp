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
CZ_TOML = REPO_ROOT / ".cz.toml"
SYNC_HOOK = REPO_ROOT / "scripts" / "sync_uv_lock_version.py"

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


# --- bump-version: no Docker action resolving code at run time --------------------


def _bump_version_steps() -> list[Step]:
    return _steps(_jobs(_load_yaml(RELEASE_YML))["bump-version"])


def test_release_no_longer_uses_commitizen_action() -> None:
    """commitizen-action is `FROM python:3.10-alpine` (mutable tag) and runs
    `pip install commitizen` at step start - two run-time-resolved code
    sources inside the job that holds RELEASE_PAT."""
    offenders = [
        _uses(step)
        for job in _jobs(_load_yaml(RELEASE_YML)).values()
        for step in _steps(job)
        if _uses(step).startswith("commitizen-tools/commitizen-action")
    ]
    assert offenders == [], offenders


def test_cz_bump_runs_on_the_runner_from_the_hashed_install() -> None:
    steps = _bump_version_steps()
    install_idx = _step_index(steps, lambda s: "requirements-release.txt" in _run(s))
    identity_idx = _step_index(steps, lambda s: "git config user.name" in _run(s))
    cz_idx = _step_index(steps, lambda s: s.get("id") == "cz")
    cz_run = _run(steps[cz_idx])
    assert "cz --no-raise 21" in cz_run, cz_run
    assert "version=" in cz_run and "GITHUB_OUTPUT" in cz_run, cz_run
    assert "uses" not in steps[cz_idx], "the bump must be a run: step, not an action"
    assert install_idx < cz_idx, "cz must be installed (hash-pinned) before it runs"
    assert identity_idx < cz_idx, "git identity must be configured before the bump commit"


def test_bump_commit_and_tag_are_pushed_with_a_per_step_header() -> None:
    steps = _bump_version_steps()
    cz_idx = _step_index(steps, lambda s: s.get("id") == "cz")
    push_idx = _step_index(steps, lambda s: "push origin HEAD:main" in _run(s))
    push_run = _run(steps[push_idx])
    assert push_idx > cz_idx
    assert "http.https://github.com/.extraheader" in push_run, push_run
    assert "refs/tags/v" in push_run, "exactly one tag is pushed, never --tags"
    assert "--tags" not in push_run, push_run
    assert steps[push_idx].get("env", {}).get("RELEASE_PAT") == "${{ secrets.RELEASE_PAT }}"


def test_release_pat_reaches_only_the_checkout_token_and_the_push_step_env() -> None:
    """The PAT may be handed to actions/checkout (clone only, not persisted)
    and to the push step as a step-level env var; nowhere else, and never
    inline in a run body."""
    locations: list[tuple[str, str]] = []
    for job_name, job in _jobs(_load_yaml(RELEASE_YML)).items():
        for step in _steps(job):
            assert "secrets.RELEASE_PAT" not in _run(step), (job_name, step.get("name"))
            for key, value in _with(step).items():
                if "secrets.RELEASE_PAT" in str(value):
                    locations.append(("with", key))
            env = step.get("env", {})
            for key, value in env.items() if isinstance(env, dict) else []:
                if "secrets.RELEASE_PAT" in str(value):
                    locations.append(("env", key))
    assert sorted(locations) == [("env", "RELEASE_PAT"), ("with", "token")], locations


def test_pre_bump_hook_docs_no_longer_describe_the_action_container() -> None:
    """The hook now runs on the ubuntu runner under setup-python 3.13."""
    cz_text = CZ_TOML.read_text(encoding="utf-8")
    hook_text = SYNC_HOOK.read_text(encoding="utf-8")
    assert "container" not in cz_text and "python:3.10-alpine" not in cz_text
    assert "python:3.10-alpine" not in hook_text and "INSIDE commitizen-action" not in hook_text


# --- build-and-publish: build exactly the bump commit the release names -----------


def _build_and_publish_steps() -> list[Step]:
    return _steps(_jobs(_load_yaml(RELEASE_YML))["build-and-publish"])


def test_bump_version_exports_the_bump_commit_sha() -> None:
    job = _jobs(_load_yaml(RELEASE_YML))["bump-version"]
    assert job["outputs"]["bump_sha"] == "${{ steps.bump_sha.outputs.sha }}", job["outputs"]
    steps = _steps(job)
    idx = _step_index(steps, lambda s: s.get("id") == "bump_sha")
    run = _run(steps[idx])
    assert "sha=$(git rev-parse HEAD)" in run and "GITHUB_OUTPUT" in run, run
    cz_idx = _step_index(steps, lambda s: s.get("id") == "cz")
    assert idx > cz_idx, "the SHA must be recorded after the bump commit exists"


def test_build_and_publish_checks_out_the_bump_commit_not_main() -> None:
    """`ref: main` is whatever main is when the job starts - anything merged
    between the bump push and the checkout (observed 251 s / 373 s windows on
    past releases) would ship under the bumped version number."""
    steps = _build_and_publish_steps()
    checkout = steps[0]
    assert _uses(checkout).startswith("actions/checkout"), checkout
    assert _with(checkout).get("ref") == "${{ needs.bump-version.outputs.bump_sha }}", checkout
    assert _with(checkout).get("persist-credentials") is False, checkout
    for job_name, job in _jobs(_load_yaml(RELEASE_YML)).items():
        for step in _steps(job):
            if _uses(step).startswith("actions/checkout"):
                assert _with(step).get("ref") != "main", (job_name, step)


def test_build_and_publish_verifies_tag_and_version_before_building() -> None:
    steps = _build_and_publish_steps()
    checkout_idx = _step_index(steps, lambda s: _uses(s).startswith("actions/checkout"))
    verify_idx = _step_index(
        steps,
        lambda s: "refs/tags/v${NEW_VERSION}^{commit}" in _run(s) and "pyproject.toml" in _run(s),
    )
    build_idx = _step_index(steps, lambda s: "uv build" in _run(s))
    publish_idx = _step_index(steps, lambda s: _uses(s).startswith("pypa/gh-action-pypi-publish"))
    assert checkout_idx < verify_idx < build_idx < publish_idx
    verify_run = _run(steps[verify_idx])
    assert "$BUMP_SHA" in verify_run, "HEAD must be compared with the recorded bump SHA"
    env = steps[verify_idx].get("env", {})
    assert env.get("BUMP_SHA") == "${{ needs.bump-version.outputs.bump_sha }}", env
    assert env.get("NEW_VERSION") == "${{ needs.bump-version.outputs.new_version }}", env

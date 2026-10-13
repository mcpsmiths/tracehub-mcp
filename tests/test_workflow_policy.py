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

import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
RELEASE_YML = WORKFLOWS_DIR / "release.yml"
DOCKER_PUBLISH_YML = WORKFLOWS_DIR / "docker-publish.yml"
MCP_REGISTRY_YML = WORKFLOWS_DIR / "mcp-registry-publish.yml"
CI_YML = WORKFLOWS_DIR / "ci.yml"
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


def _triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    triggers = workflow[True]  # the `on` key, see the module docstring
    assert isinstance(triggers, dict)
    return triggers


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


def test_hash_pinned_install_refuses_the_sdist_fallback() -> None:
    """--require-hashes alone is not fail-closed: when only the WHEEL hash of
    an entry is wrong pip falls back to the still-hashed sdist and runs an
    isolated build whose backend (uv_build, unpinned) it fetches from PyPI at
    that moment - exit 0 (runtime verification T7 fix1/03-04). With
    --only-binary=:all: the same file fails with "THESE PACKAGES DO NOT MATCH
    THE HASHES" (exit 1), and nothing is ever built next to RELEASE_PAT."""
    steps = _bump_version_steps()
    install = steps[_step_index(steps, lambda s: "requirements-release.txt" in _run(s))]
    run = _run(install)
    assert "--only-binary=:all:" in run, run
    assert "--require-hashes" in run and "--no-deps" in run, run
    # YAML comments are not part of the parsed step: read the raw text and
    # require the step's comment block to explain the sdist fallback it closes.
    lines = RELEASE_YML.read_text(encoding="utf-8").splitlines()
    (idx,) = [
        i for i, line in enumerate(lines) if "pip install" in line and "--only-binary=:all:" in line
    ]
    assert any("sdist" in line for line in lines[idx - 14 : idx]), lines[idx - 14 : idx]


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


# --- release authorization: environment-gated jobs ----------------------------------


def _environment_name(job: Job) -> str | None:
    env = job.get("environment")
    if isinstance(env, dict):
        name = env.get("name")
        return name if isinstance(name, str) else None
    return env if isinstance(env, str) else None


def _job_mentions(job: Job, needle: str) -> bool:
    return needle in yaml.safe_dump(job)


def test_bump_version_is_gated_on_the_release_environment_from_main() -> None:
    """The environment's deployment-branch policy (main only) is the control
    that survives a writer's branch copy of this file; the `if` is defence in
    depth against an accidental off-main dispatch of the unmodified file."""
    job = _jobs(_load_yaml(RELEASE_YML))["bump-version"]
    assert _environment_name(job) == "release", job.get("environment")
    assert "github.ref == 'refs/heads/main'" in str(job.get("if", "")), job.get("if")


def test_every_job_touching_release_pat_declares_the_release_environment() -> None:
    offenders = [
        f"{path.name}:{job_name}"
        for path in sorted(WORKFLOWS_DIR.glob("*.yml"))
        for job_name, job in _jobs(_load_yaml(path)).items()
        if _job_mentions(job, "secrets.RELEASE_PAT") and _environment_name(job) != "release"
    ]
    assert offenders == [], offenders


def test_security_md_documents_release_authorization() -> None:
    text = SECURITY_MD.read_text(encoding="utf-8")
    assert "Release authorization" in text
    assert "`release`" in text and "RELEASE_PAT" in text


def test_build_and_publish_is_gated_on_the_release_environment_from_main() -> None:
    """PyPI's trusted publisher is bound to this environment name; a token
    minted outside it carries no `environment` claim and PyPI rejects it."""
    job = _jobs(_load_yaml(RELEASE_YML))["build-and-publish"]
    assert _environment_name(job) == "release", job.get("environment")
    condition = str(job.get("if", ""))
    assert "github.ref == 'refs/heads/main'" in condition, condition
    assert "already_released == 'false'" in condition, condition


def _mints_a_publishing_token(job: Job) -> bool:
    permissions = job.get("permissions", {})
    if not (isinstance(permissions, dict) and permissions.get("id-token") == "write"):
        return False
    return any(
        _uses(step).startswith("pypa/gh-action-pypi-publish") or "mcp-publisher" in _run(step)
        for step in _steps(job)
    )


def test_every_oidc_publishing_job_declares_the_release_environment() -> None:
    publishing_jobs = {
        f"{path.name}:{job_name}": job
        for path in sorted(WORKFLOWS_DIR.glob("*.yml"))
        for job_name, job in _jobs(_load_yaml(path)).items()
        if _mints_a_publishing_token(job)
    }
    assert set(publishing_jobs) == {
        "mcp-registry-publish.yml:publish-mcp",
        "release.yml:build-and-publish",
    }, set(publishing_jobs)
    offenders = [
        name for name, job in publishing_jobs.items() if _environment_name(job) != "release"
    ]
    assert offenders == [], offenders


def test_security_md_binds_the_pypi_publisher_to_the_environment() -> None:
    text = SECURITY_MD.read_text(encoding="utf-8")
    section = text.split("### Release authorization", 1)[1].split("\n## ", 1)[0]
    assert "PyPI" in section and "MCP Registry" in section, section


def test_security_md_release_authorization_claims_only_controls_that_exist() -> None:
    """GitHub has no per-workflow dispatch permission: any collaborator with
    write access can run workflow_dispatch, and the environment's branch policy
    restricts the ref, not the actor. The section must not claim an actor
    restriction, must name the tag ruleset that GHCR tag integrity depends on,
    and must say that the environment/PyPI/ruleset settings are configured by
    the maintainer rather than by this repository's files."""
    text = SECURITY_MD.read_text(encoding="utf-8")
    section = text.split("### Release authorization", 1)[1].split("\n## ", 1)[0]
    prose = " ".join(section.split())  # the file is hard-wrapped; phrases may span lines
    assert "administrators can dispatch" not in prose, prose
    assert "write access can dispatch" in prose, prose
    assert "protect-release-tags" in prose, prose
    assert "inert until they are configured" in prose, prose


# --- docker-publish: :latest and :<X.Y.Z> only from the tree behind refs/tags/vX.Y.Z --

IMAGE = "ghcr.io/mcpsmiths/tracehub-mcp"


def _docker_publish_steps() -> list[Step]:
    return _steps(_jobs(_load_yaml(DOCKER_PUBLISH_YML))["build-and-push"])


def _parse_github_output(text: str) -> dict[str, str]:
    """Parse the `key=value` and `key<<DELIM ... DELIM` forms of $GITHUB_OUTPUT."""
    parsed: dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if "<<" in line:
            key, delimiter = line.split("<<", 1)
            body: list[str] = []
            i += 1
            while lines[i] != delimiter:
                body.append(lines[i])
                i += 1
            parsed[key] = "\n".join(body)
        else:
            key, _, value = line.partition("=")
            parsed[key] = value
        i += 1
    return parsed


def _replay_resolve_release_tag(tmp_path: Path, **env: str) -> tuple[int, dict[str, str]]:
    steps = _docker_publish_steps()
    run = _run(steps[_step_index(steps, lambda s: s.get("id") == "version")])
    script = tmp_path / "resolve.sh"
    script.write_text(run, encoding="utf-8")
    output = tmp_path / "github_output"
    output.write_text("", encoding="utf-8")
    bash = shutil.which("bash")
    assert bash is not None
    proc = subprocess.run(  # noqa: S603 - replaying our own workflow step under test
        [bash, "-eo", "pipefail", str(script)],
        env={"PATH": os.environ["PATH"], "GITHUB_OUTPUT": str(output), **env},
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, _parse_github_output(output.read_text(encoding="utf-8"))


def test_docker_publish_dispatch_requires_an_existing_release_tag() -> None:
    inputs = _triggers(_load_yaml(DOCKER_PUBLISH_YML))["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"tag", "also_tag_latest"}, inputs
    assert inputs["tag"]["required"] is True and inputs["tag"]["type"] == "string", inputs["tag"]
    assert inputs["also_tag_latest"]["type"] == "boolean", inputs["also_tag_latest"]
    assert inputs["also_tag_latest"]["default"] is False, inputs["also_tag_latest"]


def test_docker_publish_image_tags_come_only_from_the_resolved_release() -> None:
    steps = _docker_publish_steps()
    build = steps[_step_index(steps, lambda s: _uses(s).startswith("docker/build-push-action"))]
    assert _with(build)["tags"] == "${{ steps.version.outputs.tags }}", _with(build)
    assert _with(build).get("push") is True and _with(build).get("sbom") is True, _with(build)
    assert _with(build).get("provenance") == "mode=max", _with(build)
    for step in steps:
        assert ":latest" not in yaml.safe_dump(_with(step)), step


def test_docker_publish_resolves_then_checks_out_then_guards_the_tag() -> None:
    steps = _docker_publish_steps()
    version_idx = _step_index(steps, lambda s: s.get("id") == "version")
    checkout_idx = _step_index(steps, lambda s: _uses(s).startswith("actions/checkout"))
    login_idx = _step_index(steps, lambda s: _uses(s).startswith("docker/login-action"))
    guard_idx = _step_index(steps, lambda s: "refs/tags/v${TAG}^{commit}" in _run(s))
    build_idx = _step_index(steps, lambda s: _uses(s).startswith("docker/build-push-action"))
    assert version_idx < checkout_idx < guard_idx < build_idx
    assert version_idx < login_idx, "a non-release ref must fail before GHCR login"
    checkout = _with(steps[checkout_idx])
    assert checkout.get("ref") == "${{ steps.version.outputs.ref }}", checkout
    assert checkout.get("persist-credentials") is False and checkout.get("fetch-depth") == 0


def test_resolve_release_tag_push_of_a_release_tag_publishes_version_and_latest(
    tmp_path: Path,
) -> None:
    code, out = _replay_resolve_release_tag(
        tmp_path, GITHUB_EVENT_NAME="push", GITHUB_REF="refs/tags/v0.12.3"
    )
    assert code == 0, out
    assert out["tag"] == "0.12.3" and out["ref"] == "refs/tags/v0.12.3", out
    assert out["latest"] == "true", out
    assert out["tags"] == f"{IMAGE}:0.12.3\n{IMAGE}:latest", out


def test_resolve_release_tag_dispatch_moves_latest_only_when_asked(tmp_path: Path) -> None:
    code, out = _replay_resolve_release_tag(
        tmp_path,
        GITHUB_EVENT_NAME="workflow_dispatch",
        GITHUB_REF="refs/heads/main",
        TAG_INPUT="v0.12.3",
        ALSO_LATEST="false",
    )
    assert code == 0, out
    assert out["ref"] == "refs/tags/v0.12.3" and out["latest"] == "false", out
    assert out["tags"] == f"{IMAGE}:0.12.3", out

    code, out = _replay_resolve_release_tag(
        tmp_path,
        GITHUB_EVENT_NAME="workflow_dispatch",
        GITHUB_REF="refs/heads/main",
        TAG_INPUT="v0.12.3",
        ALSO_LATEST="true",
    )
    assert code == 0 and out["tags"] == f"{IMAGE}:0.12.3\n{IMAGE}:latest", out


def test_resolve_release_tag_refuses_non_release_refs(tmp_path: Path) -> None:
    """The audit's sandbox cases: a dispatch naming a branch and a push of
    main used to publish `manual-<run_id>` plus :latest; both now fail closed
    with nothing written to $GITHUB_OUTPUT."""
    for env in (
        {
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REF": "refs/heads/feature/unreviewed",
            "TAG_INPUT": "refs/heads/feature/unreviewed",
            "ALSO_LATEST": "true",
        },
        {
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REF": "refs/heads/main",
            "TAG_INPUT": "",
            "ALSO_LATEST": "true",
        },
        {"GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main"},
        {"GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/tags/v1.2"},
        {"GITHUB_EVENT_NAME": "schedule", "GITHUB_REF": "refs/heads/main"},
    ):
        code, out = _replay_resolve_release_tag(tmp_path, **env)
        assert code == 1, (env, out)
        assert out == {}, (env, out)


# --- every publishing job builds an explicit, immutable release ref ------------------

_RELEASE_REF_RE = re.compile(
    r"^\$\{\{ .*(bump_sha|steps\.version\.outputs\.ref|format\('refs/tags/v\{0\}'|inputs\.ref).* \}\}$"
)


def _publishes_an_artifact(job: Job) -> bool:
    for step in _steps(job):
        if _uses(step).startswith("pypa/gh-action-pypi-publish"):
            return True
        if _uses(step).startswith("docker/build-push-action") and _with(step).get("push") is True:
            return True
        if "mcp-publisher publish" in _run(step):
            return True
    return False


def test_every_publishing_job_checks_out_an_explicit_release_ref() -> None:
    publishing = {
        f"{path.name}:{job_name}": job
        for path in sorted(WORKFLOWS_DIR.glob("*.yml"))
        for job_name, job in _jobs(_load_yaml(path)).items()
        if _publishes_an_artifact(job)
    }
    assert set(publishing) == {
        "docker-publish.yml:build-and-push",
        "mcp-registry-publish.yml:publish-mcp",
        "release.yml:build-and-publish",
    }, set(publishing)
    for name, job in publishing.items():
        checkouts = [s for s in _steps(job) if _uses(s).startswith("actions/checkout")]
        assert checkouts, f"{name}: no checkout step"
        for checkout in checkouts:
            ref = str(_with(checkout).get("ref", ""))
            assert _RELEASE_REF_RE.match(ref), f"{name}: checkout ref {ref!r} is not a release ref"


# --- mcp-registry-publish: the entry for version X is tag vX's server.json ---------


def _publish_mcp_steps() -> list[Step]:
    return _steps(_jobs(_load_yaml(MCP_REGISTRY_YML))["publish-mcp"])


def test_mcp_registry_dispatch_exposes_only_the_version_input() -> None:
    """The dispatch path is bound to refs/tags/v<version>; `ref` stays a
    workflow_call input because release.yml passes the tag it just made."""
    triggers = _triggers(_load_yaml(MCP_REGISTRY_YML))
    assert set(triggers["workflow_dispatch"]["inputs"]) == {"version"}, triggers[
        "workflow_dispatch"
    ]
    assert set(triggers["workflow_call"]["inputs"]) == {"ref", "version"}, triggers["workflow_call"]


def test_mcp_registry_checks_out_the_release_tag() -> None:
    steps = _publish_mcp_steps()
    checkout = _with(steps[_step_index(steps, lambda s: _uses(s).startswith("actions/checkout"))])
    ref = str(checkout.get("ref", ""))
    assert "format('refs/tags/v{0}', inputs.version)" in ref and "inputs.ref ||" in ref, ref
    assert checkout.get("persist-credentials") is False and checkout.get("fetch-depth") == 0


def test_mcp_registry_asserts_identity_and_tag_before_publishing() -> None:
    steps = _publish_mcp_steps()
    sync_idx = _step_index(steps, lambda s: ".packages[0].version = $v" in _run(s))
    guard_idx = _step_index(steps, lambda s: "refs/tags/v${NEW_VERSION}^{commit}" in _run(s))
    assert_idx = _step_index(steps, lambda s: "scripts/check_server_json.py" in _run(s))
    poll_idx = _step_index(steps, lambda s: "pypi.org/pypi/" in _run(s))
    publish_idx = _step_index(steps, lambda s: "mcp-publisher publish" in _run(s))
    assert guard_idx < publish_idx
    assert sync_idx < assert_idx < poll_idx < publish_idx
    assert_run = _run(steps[assert_idx])
    assert '--expect-version "$NEW_VERSION"' in assert_run and "server.json" in assert_run, (
        assert_run
    )
    assert steps[assert_idx].get("env", {}).get("NEW_VERSION") == "${{ inputs.version }}"


def _embedded_python(run: str) -> str:
    """The body of the one `python3 - ... <<'PY' ... PY` heredoc in a run script."""
    lines = run.splitlines()
    start = _step_index(lines, lambda line: "<<'PY'" in line) + 1
    end = _step_index(lines, lambda line: line.strip() == "PY")
    return textwrap.dedent("\n".join(lines[start:end]))


def _replay_version_guard(tmp_path: Path, run: str, declared: str, version: str) -> tuple[int, str]:
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "tracehub-mcp"\nversion = "{declared}"\n', encoding="utf-8"
    )
    proc = subprocess.run(  # noqa: S603 - replaying our own workflow step under test
        [sys.executable, "-I", "-", version],
        input=_embedded_python(run),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout


def test_mcp_registry_guard_cross_checks_the_tree_version_before_the_sync(
    tmp_path: Path,
) -> None:
    """The tag guard alone accepts a tag v0.12.8 that points at the 0.12.3
    tree (runtime verification T7 fix7-8/02, case R5 exit 0, where
    docker-publish's D5 failed), and because the jq sync rewrites both
    server.json version fields to the input before check_server_json runs,
    the version clause there is self-fulfilling. The guard must therefore
    compare pyproject.toml's version with the tag BEFORE the sync, exactly
    like docker-publish.yml's guard, and fail with a ::error:: annotation."""
    steps = _publish_mcp_steps()
    guard_idx = _step_index(steps, lambda s: "refs/tags/v${NEW_VERSION}^{commit}" in _run(s))
    sync_idx = _step_index(steps, lambda s: ".packages[0].version = $v" in _run(s))
    assert guard_idx < sync_idx
    guard = _run(steps[guard_idx])
    assert "pyproject.toml" in guard and "::error::" in guard, guard
    assert steps[guard_idx].get("env", {}).get("NEW_VERSION") == "${{ inputs.version }}"

    code, out = _replay_version_guard(tmp_path, guard, declared="0.12.3", version="0.12.8")
    assert code == 1 and "::error::pyproject.toml declares 0.12.3" in out, (code, out)
    code, out = _replay_version_guard(tmp_path, guard, declared="0.12.8", version="0.12.8")
    assert code == 0 and out == "", (code, out)


def test_mcp_registry_pypi_poll_targets_the_asserted_identifier() -> None:
    steps = _publish_mcp_steps()
    poll_run = _run(steps[_step_index(steps, lambda s: "pypi.org/pypi/" in _run(s))])
    assert "pypi.org/pypi/tracehub-mcp/" not in poll_run, "poll URL must not be hardcoded"
    assert "jq -r '.packages[0].identifier' server.json" in poll_run, poll_run
    assert "pypi.org/pypi/${identifier}/${NEW_VERSION}/json" in poll_run, poll_run


# --- ci.yml smoke test: never vacuously green, token off the docker argv ----------------


def _smoke_test_step() -> Step:
    steps = [
        step
        for job in _jobs(_load_yaml(CI_YML)).values()
        for step in _steps(job)
        if str(step.get("name", "")).startswith("Smoke-test the built image")
    ]
    assert len(steps) == 1, [s.get("name") for s in steps]
    return steps[0]


def _replay_smoke_step(
    tmp_path: Path, *, target_platform: str, docker_arch: str | None
) -> tuple[int, str, list[str]]:
    """Run the smoke step's script with a stub `docker` on PATH (or none) and
    return (exit code, stdout, the docker sub-commands the stub saw)."""
    script = tmp_path / "smoke.sh"
    script.write_text(_run(_smoke_test_step()), encoding="utf-8")
    calls = tmp_path / "docker-calls.log"
    calls.write_text("", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    if docker_arch is not None:
        stub = bin_dir / "docker"
        stub.write_text(
            f'#!/bin/sh\necho "$1" >> "{calls}"\n[ "$1" = info ] && echo "{docker_arch}"\nexit 0\n',
            encoding="utf-8",
        )
        stub.chmod(0o755)
    bash = shutil.which("bash")
    assert bash is not None
    proc = subprocess.run(  # noqa: S603 - replaying our own workflow step under test
        [bash, "-eo", "pipefail", str(script)],
        env={
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "TARGET_PLATFORM": target_platform,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, calls.read_text(encoding="utf-8").split()


def test_smoke_step_fails_closed_when_the_platform_mapping_is_missing(tmp_path: Path) -> None:
    """With TARGET_PLATFORM empty the old case statement printed
    "::warning::Skipping smoke-test execution ... cannot execute a  image" and
    exited 0 (runtime verification, T1): drop the `env:` mapping and the step
    passes without ever starting the container. An empty platform is a
    workflow bug, not an infrastructure limitation, and must fail the job
    before docker is consulted."""
    step = _smoke_test_step()
    assert step.get("env", {}).get("TARGET_PLATFORM") == "${{ matrix.platform }}", step.get("env")

    code, out, docker_calls = _replay_smoke_step(tmp_path, target_platform="", docker_arch=None)
    assert code == 1, (code, out)
    assert "::error::" in out and "TARGET_PLATFORM" in out, out
    assert "::warning::" not in out, out
    assert docker_calls == [], docker_calls


def test_smoke_step_still_skips_loudly_on_a_runner_that_cannot_execute_the_image(
    tmp_path: Path,
) -> None:
    """The legitimate skip (arm64 image on an x86_64 host with no QEMU) keeps
    its warning and exit 0, and never reaches `docker run`."""
    code, out, docker_calls = _replay_smoke_step(
        tmp_path, target_platform="arm64", docker_arch="x86_64"
    )
    assert code == 0, (code, out)
    assert "::warning::Skipping smoke-test execution" in out and "arm64" in out, out
    assert docker_calls == ["info"], docker_calls


def test_smoke_step_hands_docker_the_token_by_name_not_on_the_argv() -> None:
    """Passing the generated value inline in the `-e NAME=VALUE` form put it on
    the docker CLI's argv, visible to every process on the runner via ps.
    Generate it into the step's own environment instead and let docker read
    it by name (`-e MCP_HTTP_BEARER_TOKEN`), the form README's own docker
    example uses."""
    run = _run(_smoke_test_step())
    assert re.search(r"^\s*export MCP_HTTP_BEARER_TOKEN\b", run, re.MULTILINE), run
    flag_lines = [
        line
        for line in run.splitlines()
        if "-e MCP_HTTP_BEARER_TOKEN" in line and not line.lstrip().startswith("#")
    ]
    assert flag_lines, run
    for line in flag_lines:
        assert line.rstrip(" \\").endswith("MCP_HTTP_BEARER_TOKEN"), line

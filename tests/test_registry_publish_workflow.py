"""Hardening of .github/workflows/mcp-registry-publish.yml (issue #56).

The registry's own PyPI lookup lags pypi.org and fails in several shapes, so the publish step
has to retry every error the registry labels transient (and only those), stay idempotent when
the entry for the version is already live, and prove the served listing afterwards. These tests
pin each of those elements in the workflow text and then run the publish and read-back step
scripts for real, with a stub ``mcp-publisher``, a stub ``curl`` and an instant ``sleep``, so
nothing here touches a network or a registry.

Scope note: the trigger, environment, OIDC login, tag guards and identity assertion are pinned by
tests/test_workflow_policy.py; this file owns the publish-step and read-back behaviour.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "mcp-registry-publish.yml"

JOB = "publish-mcp"
PUBLISH_STEP = "Validate and publish server.json"
READ_BACK_STEP = "Confirm the published entry"
INSTALL_STEP = "Install mcp-publisher (pinned, SHA256-verified)"
POLL_STEP = "Wait for PyPI to serve the release with the ownership marker"

SERVER_NAME = "io.github.mcpsmiths/tracehub-mcp"
VERSION = "9.9.9"
ENTRY_URL = (
    "https://registry.modelcontextprotocol.io/v0.1/servers/"
    f"io.github.mcpsmiths%2Ftracehub-mcp/versions/{VERSION}"
)
RETRY_ATTEMPTS = 8
READ_BACK_ATTEMPTS = 10

# The registry's own message texts, internal/validators/registries/pypi.go at registry v1.8.1
# (identifiers substituted). The first five are the shapes it labels transient: version not
# yet visible (:151), PyPI 429 (:135), PyPI 5xx (:137), inconclusive probe (:155) and the
# wrapped transport failure (:79).
TRANSIENT_ERRORS = [
    (
        "PyPI package 'tracehub-mcp' exists, but version '9.9.9' was not found (status: 404). "
        "A newly published release can take a moment to appear on PyPI. Wait and retry, or "
        "publish version '9.9.9' before registering it"
    ),
    (
        "PyPI rate-limited the metadata request for package 'tracehub-mcp' (status: 429). "
        "Likely transient, retry later"
    ),
    (
        "PyPI upstream error fetching metadata for package 'tracehub-mcp' (status: 503). "
        "Likely transient, retry later"
    ),
    (
        "PyPI could not confirm package 'tracehub-mcp' version '9.9.9' (version status: 404, "
        "package check inconclusive). Likely transient, retry later"
    ),
    'failed to fetch package metadata from PyPI: Get "https://pypi.org/pypi/tracehub-mcp/json": '
    "dial tcp: i/o timeout",
]
# Final errors: retrying any of these only wastes minutes (database.go:19 duplicate; pypi.go
# :107 ownership; :153 unknown package; :139 other metadata status; :160 version unknown).
FINAL_ERRORS = [
    "invalid version: cannot publish duplicate version",
    (
        "PyPI package 'tracehub-mcp' ownership validation failed. The server name "
        "'io.github.mcpsmiths/tracehub-mcp' must appear as 'mcp-name: "
        "io.github.mcpsmiths/tracehub-mcp' in the package README"
    ),
    "PyPI package 'someone-elses-package' not found (status: 404)",
    "PyPI package 'tracehub-mcp' metadata fetch failed (status: 403)",
    "PyPI package 'tracehub-mcp' version '9.9.9' not found (status: 404)",
]


# --- helpers ---------------------------------------------------------------------------------


def _load() -> dict[Any, Any]:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _job() -> dict[str, Any]:
    job: dict[str, Any] = _load()["jobs"][JOB]
    return job


def _step(name: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = [s for s in _job()["steps"] if s.get("name") == name]
    assert len(matches) == 1, f"expected exactly one step named {name!r}, found {len(matches)}"
    return matches[0]


def _script(name: str) -> str:
    run = _step(name).get("run")
    assert isinstance(run, str), f"step {name!r} has no run script"
    return run


def _step_names() -> list[str]:
    return [str(s.get("name", "")) for s in _job()["steps"]]


def _retry_pattern() -> re.Pattern[str]:
    match = re.search(r"grep -qE '([^']+)'", _script(PUBLISH_STEP))
    assert match, "publish step has no `grep -qE '<pattern>'` transient-error filter"
    return re.compile(match.group(1))


class _Sandbox:
    """A scratch runner: stub binaries on PATH, a fixture server.json, an events log."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.work = root / "work"
        self.bin = root / "bin"
        self.state = root / "state"
        for directory in (self.work, self.bin, self.state):
            directory.mkdir()
        manifest = {
            "name": SERVER_NAME,
            "version": VERSION,
            "packages": [
                {"registryType": "pypi", "identifier": "tracehub-mcp", "version": VERSION}
            ],
        }
        (self.work / "server.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        self._write_exec(
            self.work / "mcp-publisher",
            """#!/bin/bash
d="$STUB_DIR"
case "$1" in
  validate)
    echo validate >> "$d/events"
    if [ -f "$d/validate_fails" ]; then echo "validation failed: bad server.json"; exit 1; fi
    exit 0 ;;
  login)
    echo login >> "$d/events"
    if [ -f "$d/login_fails" ]; then echo "login failed: oidc unreachable"; exit 1; fi
    exit 0 ;;
  publish)
    echo publish >> "$d/events"
    n=$(( $(cat "$d/publish_count" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "$d/publish_count"
    f="$d/publish_$n"
    [ -f "$f" ] || f="$d/publish_last"
    code="$(head -n 1 "$f")"
    tail -n +2 "$f"
    exit "$code" ;;
esac
exit 64
""",
        )
        self._write_exec(
            self.bin / "sleep",
            '#!/bin/bash\necho "sleep $*" >> "$STUB_DIR/events"\nexit 0\n',
        )
        # Serves the registry entry from state/entry_body when present, else fails like
        # `curl -f` on a 404 (exit 22). Logs the URL it was asked for.
        self._write_exec(
            self.bin / "curl",
            """#!/bin/bash
d="$STUB_DIR"
url=""
for arg in "$@"; do url="$arg"; done
echo "curl $url" >> "$d/events"
n=$(( $(cat "$d/curl_count" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$d/curl_count"
if [ -f "$d/entry_body" ] && [ "$n" -gt "$(cat "$d/entry_absent_for" 2>/dev/null || echo 0)" ]; then
  cat "$d/entry_body"
  exit 0
fi
exit 22
""",
        )

    @staticmethod
    def _write_exec(path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def publish_attempts(self, *attempts: tuple[int, str]) -> None:
        """Script `mcp-publisher publish`: one (exit code, output) per call, last one repeats."""
        for index, (code, output) in enumerate(attempts, start=1):
            (self.state / f"publish_{index}").write_text(f"{code}\n{output}\n", encoding="utf-8")
        code, output = attempts[-1]
        (self.state / "publish_last").write_text(f"{code}\n{output}\n", encoding="utf-8")

    def entry_live(self, body: str | None = None, *, absent_for_first: int = 0) -> None:
        (self.state / "entry_body").write_text(body or _entry_body(), encoding="utf-8")
        (self.state / "entry_absent_for").write_text(str(absent_for_first), encoding="utf-8")

    def run(self, script: str) -> subprocess.CompletedProcess[str]:
        # GitHub Actions executes a `run:` block with `bash -e {0}`; pipefail added is the
        # stricter form (every pipeline in these steps exits the same way under both).
        script_path = self.root / "step.sh"
        script_path.write_text(script, encoding="utf-8")
        bash = shutil.which("bash")
        assert bash is not None
        env = {
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "HOME": str(self.root),
            "NEW_VERSION": VERSION,
            "STUB_DIR": str(self.state),
        }
        return subprocess.run(  # noqa: S603 - replaying the workflow's own step text
            [bash, "--noprofile", "--norc", "-eo", "pipefail", str(script_path)],
            cwd=self.work,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

    def events(self) -> list[str]:
        path = self.state / "events"
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def _entry_body(
    *,
    version: str = VERSION,
    package_version: str = VERSION,
    identifier: str = "tracehub-mcp",
    registry_type: str = "pypi",
    status: str = "active",
) -> str:
    return json.dumps(
        {
            "server": {
                "name": SERVER_NAME,
                "version": version,
                "packages": [
                    {
                        "registryType": registry_type,
                        "identifier": identifier,
                        "version": package_version,
                    }
                ],
            },
            "_meta": {"io.modelcontextprotocol.registry/official": {"status": status}},
        }
    )


@pytest.fixture
def sandbox(tmp_path: Path) -> _Sandbox:
    return _Sandbox(tmp_path)


# --- static pins -----------------------------------------------------------------------------


def test_retry_filter_matches_every_transient_error_and_no_final_error() -> None:
    pattern = _retry_pattern()
    for message in TRANSIENT_ERRORS:
        assert pattern.search(message), f"transient registry error would not be retried: {message}"
    for message in FINAL_ERRORS:
        assert not pattern.search(message), f"final registry error would be retried: {message}"


def test_retry_budget_is_eight_attempts_30s_apart_without_a_trailing_sleep() -> None:
    script = _script(PUBLISH_STEP)
    assert f"for attempt in $(seq 1 {RETRY_ATTEMPTS}); do" in script
    assert "sleep 30" in script
    assert f'echo "Attempt ${{attempt}}/{RETRY_ATTEMPTS}:' in script
    assert f"still failing after {RETRY_ATTEMPTS} attempts" in script
    assert "seq 1 5" not in script and "/5:" not in script and "after 5 attempts" not in script
    assert f'if [ "$attempt" -lt {RETRY_ATTEMPTS} ]; then' in script


def test_publish_step_probes_the_versioned_entry_url_before_giving_up() -> None:
    script = _script(PUBLISH_STEP)
    assert "registry.modelcontextprotocol.io/v0.1/servers/${encoded}/versions/${NEW_VERSION}" in (
        script
    )
    assert "@uri" in script, "the server name must be URL-encoded (it contains a slash)"
    assert "already published" in script


def test_read_back_step_follows_the_publish_step_and_requires_an_active_listing() -> None:
    names = _step_names()
    assert names.index(PUBLISH_STEP) < names.index(READ_BACK_STEP)
    script = _script(READ_BACK_STEP)
    assert "io.modelcontextprotocol.registry/official" in script
    assert '.status == "active"' in script
    assert "$s.version == $v" in script
    assert "$s.packages[0].version == $v" in script
    assert 'registryType == "pypi"' in script
    assert "$s.packages[0].identifier == $id" in script
    assert "mcp-publisher" not in script, "the read-back must not publish anything"


def test_header_documents_the_token_bound_and_the_recovery_path() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    header = text.split("\non:\n", 1)[0]
    assert "5 minutes" in header and "logs in again" in header
    assert "dispatch mcp-registry-publish.yml from main" in header
    assert "idempotent" in header


def test_job_has_a_timeout_that_covers_the_designed_worst_case() -> None:
    # PyPI poll 40 x 20 s (13.3 min) + 8 publish attempts (3.5 min of waits) + read-back
    # 10 x 15 s (2.25 min) is about 20 minutes; anything past 30 is a stalled connection.
    assert _job().get("timeout-minutes") == 30


def test_every_curl_in_the_job_is_time_bounded() -> None:
    offenders = []
    for step in _job()["steps"]:
        run = step.get("run")
        if not isinstance(run, str):
            continue
        for line in run.splitlines():
            if re.search(r"\bcurl\b", line) and "--max-time" not in line:
                offenders.append(f"{step.get('name')}: {line.strip()}")
    assert offenders == [], offenders


def test_mcp_publisher_pin_and_digest_are_the_verified_pair() -> None:
    # Same v1.8.1 linux_amd64 pair cost-guard-mcp verified against the archive's sigstore
    # bundle on 2026-10-15; a bump must change both lines deliberately.
    script = _script(INSTALL_STEP)
    assert 'version="1.8.1"' in script
    assert (
        'expected_sha256="a06c9096dcb9727c13555b6be26c7effa707b01f06a4c561ba7a3635443cf2cc"'
        in script
    )
    assert "sha256sum --check --strict" in script


def test_publish_authority_is_unchanged_least_privilege_and_environment_gated() -> None:
    workflow = _load()
    assert workflow["permissions"] == {"contents": "read"}
    job = _job()
    assert job["environment"] == "release"
    assert job["permissions"] == {"id-token": "write", "contents": "read"}
    login = _script("Authenticate to the MCP Registry (GitHub OIDC)")
    assert login.strip() == "./mcp-publisher login github-oidc"


def test_no_workflow_expression_is_interpolated_into_a_run_script() -> None:
    # Untrusted inputs (version, ref) reach shell only through `env:`, never through ${{ }}
    # inside a run block.
    offenders = [
        str(step.get("name"))
        for step in _job()["steps"]
        if isinstance(step.get("run"), str) and "${{" in step["run"]
    ]
    assert offenders == [], offenders
    for name in (POLL_STEP, PUBLISH_STEP, READ_BACK_STEP):
        assert _step(name).get("env", {}).get("NEW_VERSION") == "${{ inputs.version }}", name


# --- the publish step, executed -----------------------------------------------------------------


def test_first_attempt_success_publishes_once_without_waiting(sandbox: _Sandbox) -> None:
    sandbox.publish_attempts((0, "published"))
    completed = sandbox.run(_script(PUBLISH_STEP))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert sandbox.events() == ["validate", "publish"]


def test_validate_failure_stops_before_any_publish(sandbox: _Sandbox) -> None:
    (sandbox.state / "validate_fails").write_text("", encoding="utf-8")
    sandbox.publish_attempts((0, "published"))
    completed = sandbox.run(_script(PUBLISH_STEP))
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert sandbox.events() == ["validate"]


def test_registry_404_then_success_retries_once_after_30s(sandbox: _Sandbox) -> None:
    sandbox.publish_attempts((1, TRANSIENT_ERRORS[0]), (0, "published"))
    completed = sandbox.run(_script(PUBLISH_STEP))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert sandbox.events() == ["validate", "publish", "sleep 30", "login", "publish"]
    assert "Attempt 1/8" in completed.stdout


@pytest.mark.parametrize("message", TRANSIENT_ERRORS, ids=[f"transient_{i}" for i in range(5)])
def test_each_transient_registry_error_is_retried_then_succeeds(
    sandbox: _Sandbox, message: str
) -> None:
    sandbox.publish_attempts((1, message), (1, message), (0, "published"))
    completed = sandbox.run(_script(PUBLISH_STEP))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert sandbox.events() == [
        "validate",
        "publish",
        "sleep 30",
        "login",
        "publish",
        "sleep 30",
        "login",
        "publish",
    ]


def test_non_transient_failure_with_the_entry_live_is_an_idempotent_success(
    sandbox: _Sandbox,
) -> None:
    sandbox.publish_attempts((1, FINAL_ERRORS[0]))
    sandbox.entry_live()
    completed = sandbox.run(_script(PUBLISH_STEP))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "already published" in completed.stdout
    assert f"{SERVER_NAME} {VERSION}" in completed.stdout
    assert sandbox.events() == ["validate", "publish", f"curl {ENTRY_URL}"]


@pytest.mark.parametrize("message", FINAL_ERRORS, ids=[f"final_{i}" for i in range(5)])
def test_non_transient_failure_with_the_entry_absent_fails_at_once(
    sandbox: _Sandbox, message: str
) -> None:
    sandbox.publish_attempts((1, message))
    completed = sandbox.run(_script(PUBLISH_STEP))
    assert completed.returncode == 1, completed.stdout + completed.stderr
    # One publish, one probe of the versioned entry URL, no waiting.
    assert sandbox.events() == ["validate", "publish", f"curl {ENTRY_URL}"]


def test_exhausting_eight_transient_attempts_fails_with_no_trailing_sleep(
    sandbox: _Sandbox,
) -> None:
    sandbox.publish_attempts((1, TRANSIENT_ERRORS[1]))
    completed = sandbox.run(_script(PUBLISH_STEP))
    assert completed.returncode == 1, completed.stdout + completed.stderr
    events = sandbox.events()
    assert events.count("publish") == RETRY_ATTEMPTS
    assert events.count("sleep 30") == RETRY_ATTEMPTS - 1
    assert events[-1] == "publish", "no sleep may follow the final attempt"
    assert "still failing after 8 attempts" in completed.stdout + completed.stderr


def test_registry_token_is_minted_fresh_before_every_attempt_after_the_first(
    sandbox: _Sandbox,
) -> None:
    # The saved registry token lives 5 minutes and `publish` never refreshes it, while seven
    # 30 s waits plus the failed publishes outlast that: attempts 2..8 each need a new login,
    # made after the wait so the token is youngest when it is used.
    sandbox.publish_attempts((1, TRANSIENT_ERRORS[1]))
    completed = sandbox.run(_script(PUBLISH_STEP))
    assert completed.returncode == 1, completed.stdout + completed.stderr
    events = sandbox.events()
    assert events[:2] == ["validate", "publish"], "attempt 1 uses the login step's token"
    assert events.count("login") == RETRY_ATTEMPTS - 1
    expected_tail = [
        event for _ in range(RETRY_ATTEMPTS - 1) for event in ("sleep 30", "login", "publish")
    ]
    assert events[2:] == expected_tail


def test_a_failed_re_login_does_not_stop_the_retry_loop(sandbox: _Sandbox) -> None:
    (sandbox.state / "login_fails").write_text("", encoding="utf-8")
    sandbox.publish_attempts((1, TRANSIENT_ERRORS[0]), (0, "published"))
    completed = sandbox.run(_script(PUBLISH_STEP))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert sandbox.events() == ["validate", "publish", "sleep 30", "login", "publish"]
    assert "::warning::registry re-login failed before attempt 2" in completed.stdout


def test_exhaustion_error_tells_the_operator_how_to_recover(sandbox: _Sandbox) -> None:
    sandbox.publish_attempts((1, TRANSIENT_ERRORS[1]))
    completed = sandbox.run(_script(PUBLISH_STEP))
    assert completed.returncode == 1, completed.stdout + completed.stderr
    error_line = next(
        line for line in completed.stdout.splitlines() if "still failing after 8 attempts" in line
    )
    assert error_line.startswith("::error::")
    assert "PyPI and the GitHub Release are already live" in error_line
    assert f"mcp-registry-publish.yml from main with version={VERSION}" in error_line
    assert "idempotent" in error_line


def test_a_transient_run_ending_in_a_final_error_stops_there(sandbox: _Sandbox) -> None:
    sandbox.publish_attempts((1, TRANSIENT_ERRORS[0]), (1, FINAL_ERRORS[3]))
    completed = sandbox.run(_script(PUBLISH_STEP))
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert sandbox.events() == [
        "validate",
        "publish",
        "sleep 30",
        "login",
        "publish",
        f"curl {ENTRY_URL}",
    ]


# --- the read-back step, executed ---------------------------------------------------------------


def test_read_back_accepts_an_active_entry_at_the_expected_version(sandbox: _Sandbox) -> None:
    sandbox.entry_live()
    completed = sandbox.run(_script(READ_BACK_STEP))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "status active" in completed.stdout
    assert sandbox.events() == [f"curl {ENTRY_URL}"]


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"status": "deprecated"}, id="status_deprecated"),
        pytest.param({"status": "deleted"}, id="status_deleted"),
        pytest.param({"version": "9.9.8"}, id="top_level_version_differs"),
        pytest.param({"package_version": "9.9.8"}, id="package_version_differs"),
        pytest.param({"identifier": "someone-elses-package"}, id="identifier_differs"),
        pytest.param({"registry_type": "npm"}, id="registry_type_not_pypi"),
    ],
)
def test_read_back_fails_unless_the_served_entry_is_active_and_correct(
    sandbox: _Sandbox, overrides: dict[str, str]
) -> None:
    sandbox.entry_live(_entry_body(**overrides))
    completed = sandbox.run(_script(READ_BACK_STEP))
    assert completed.returncode != 0, completed.stdout + completed.stderr


def test_read_back_fails_when_the_status_metadata_is_missing(sandbox: _Sandbox) -> None:
    sandbox.entry_live('{"server": {"name": "x", "version": "9.9.9", "packages": []}}')
    completed = sandbox.run(_script(READ_BACK_STEP))
    assert completed.returncode != 0, completed.stdout + completed.stderr


def test_read_back_waits_for_an_entry_that_is_not_readable_yet(sandbox: _Sandbox) -> None:
    sandbox.entry_live(absent_for_first=2)
    completed = sandbox.run(_script(READ_BACK_STEP))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert sandbox.events() == [
        f"curl {ENTRY_URL}",
        "sleep 15",
        f"curl {ENTRY_URL}",
        "sleep 15",
        f"curl {ENTRY_URL}",
    ]


def test_read_back_gives_up_after_ten_unreadable_attempts_with_no_trailing_sleep(
    sandbox: _Sandbox,
) -> None:
    completed = sandbox.run(_script(READ_BACK_STEP))
    assert completed.returncode == 1, completed.stdout + completed.stderr
    events = sandbox.events()
    assert len([e for e in events if e.startswith("curl ")]) == READ_BACK_ATTEMPTS
    assert events.count("sleep 15") == READ_BACK_ATTEMPTS - 1
    assert events[-1].startswith("curl "), "no sleep may follow the final attempt"
    assert "was not readable after 10 attempts" in completed.stdout + completed.stderr


def test_read_back_gives_the_operator_a_recovery_path_when_the_entry_is_unreadable(
    sandbox: _Sandbox,
) -> None:
    completed = sandbox.run(_script(READ_BACK_STEP))
    assert completed.returncode == 1, completed.stdout + completed.stderr
    error_line = next(
        line for line in completed.stdout.splitlines() if "not readable after 10 attempts" in line
    )
    assert error_line.startswith("::error::")
    assert "PyPI and the GitHub Release are already live" in error_line
    assert f"mcp-registry-publish.yml from main with version={VERSION}" in error_line


def test_read_back_mismatch_explains_that_a_re_dispatch_cannot_change_the_record(
    sandbox: _Sandbox,
) -> None:
    sandbox.entry_live(_entry_body(status="deprecated"))
    completed = sandbox.run(_script(READ_BACK_STEP))
    assert completed.returncode == 1, completed.stdout + completed.stderr
    error_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("::error::")
    )
    assert "but not as an active pypi/tracehub-mcp/9.9.9 entry" in error_line
    assert "cannot change a served record" in error_line

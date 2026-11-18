"""Behaviour of release.yml's "Test PyPI Installation" step.

The step polls ``pip install tracehub-mcp==<new version>`` for up to ten attempts,
because PyPI's index can lag the upload that the previous job just made. For the
0.13.2 release all ten attempts failed and the log said only "not yet available on
PyPI": the step sent pip's stderr to /dev/null, so nothing said why, and after the
last attempt the script fell through to ``python -c "import ..."`` and failed with a
``ModuleNotFoundError`` that named neither the version nor the cause.

These tests run the step's real ``run:`` script under ``bash -e`` (what the runner does
for a step without ``shell:``) with fake ``pip``, ``sleep``, ``python`` and
``tracehub-mcp`` executables first on ``PATH``, so no network is used and no real
waiting happens. They pin that pip's own reason is shown for every failed attempt (the
last five lines of its stderr), that giving up is an explicit error, and that text
coming back from the index cannot start a GitHub workflow command.

The runner turns a log line into a command when, after trimming leading whitespace, it
starts with ``::`` (the current form) or contains ``##[`` anywhere (the legacy form),
and it reads the step's output line by line splitting on ``\\n``, ``\\r`` and
``\\r\\n``. The injection test therefore parses the output the same way instead of
looking only at ``\\n``-separated lines.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import sys
from pathlib import Path

import pytest
import yaml

RELEASE_YML = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "release.yml"
STEP_NAME = "Wait for package availability and test installation"
ATTEMPTS = 10
NEW_VERSION = "9.9.9"
# Executables the step's own script and the fakes use besides bash builtins.
_TOOLS = ("seq", "tail", "tr", "cut", "awk", "cat", "touch")

pytestmark = pytest.mark.skipif(
    sys.platform == "win32"
    or shutil.which("bash") is None
    or any(shutil.which(tool) is None for tool in _TOOLS),
    reason="runs the step's bash script and needs bash plus " + ", ".join(_TOOLS),
)

# Fake executables. State lives in $FAKE_STATE so each test can inspect what ran.
_PIP = """#!/usr/bin/env bash
count_file="$FAKE_STATE/pip-calls"
n=$(( $(cat "$count_file" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$count_file"
if [ "$n" -ge "$FAKE_PIP_SUCCEEDS_ON" ]; then
  echo "Successfully installed tracehub-mcp-$NEW_VERSION"
  touch "$FAKE_STATE/installed"
  exit 0
fi
i=1
while [ "$i" -le "${FAKE_PIP_NOISE_LINES:-0}" ]; do
  echo "noise-line-$i" >&2
  i=$(( i + 1 ))
done
echo "ERROR: Could not find a version that satisfies the requirement tracehub-mcp==$NEW_VERSION (from versions: none)" >&2
echo "ERROR: No matching distribution found for tracehub-mcp==$NEW_VERSION" >&2
if [ -n "${FAKE_PIP_EXTRA_STDERR:-}" ]; then
  printf '%s\\n' "$FAKE_PIP_EXTRA_STDERR" >&2
fi
exit 1
"""
_SLEEP = """#!/usr/bin/env bash
echo "$@" >> "$FAKE_STATE/sleeps"
"""
_PYTHON = """#!/usr/bin/env bash
if [ -e "$FAKE_STATE/installed" ]; then
  touch "$FAKE_STATE/python-ran"
  echo "Installed version: $NEW_VERSION"
  exit 0
fi
echo "ModuleNotFoundError: No module named 'opentelemetry_mcp'" >&2
exit 1
"""
# What BSD awk does in a UTF-8 locale when cut leaves a truncated multibyte character:
# it aborts with a non-zero status after reading its input.
_FAILING_AWK = """#!/usr/bin/env bash
cat > /dev/null
echo "awk: towc: multibyte conversion failure" >&2
exit 2
"""
_TRACEHUB = """#!/usr/bin/env bash
touch "$FAKE_STATE/help-ran"
echo "usage: tracehub-mcp [OPTIONS]"
"""


def _install_check_script() -> str:
    data = yaml.safe_load(RELEASE_YML.read_text(encoding="utf-8"))
    steps = data["jobs"]["test-installation"]["steps"]
    matches = [step for step in steps if step.get("name") == STEP_NAME]
    assert len(matches) == 1, f"expected exactly one step named {STEP_NAME!r}"
    return str(matches[0]["run"])


def _logical_lines(output: str) -> list[str]:
    """Split the way the runner reads a step's output: on \\n, \\r and \\r\\n."""
    return re.split(r"\r\n|\r|\n", output)


class _Run:
    def __init__(self, returncode: int | None, output: str, state: Path) -> None:
        self.returncode = returncode
        self.output = output
        self.state = state

    def ran(self, marker: str) -> bool:
        return (self.state / marker).exists()

    def sleeps(self) -> int:
        sleeps = self.state / "sleeps"
        return len(sleeps.read_text().splitlines()) if sleeps.exists() else 0


async def _run_step(
    tmp_path: Path,
    *,
    succeeds_on: int,
    extra_stderr: str = "",
    noise_lines: int = 0,
    runner_temp_mode: str = "set",
    failing_awk: bool = False,
) -> _Run:
    bash = shutil.which("bash")
    assert bash is not None
    bin_dir = tmp_path / "bin"
    state = tmp_path / "state"
    runner_temp = tmp_path / "runner-temp"
    for directory in (bin_dir, state, runner_temp):
        directory.mkdir()
    fakes = [
        ("pip", _PIP),
        ("sleep", _SLEEP),
        ("python", _PYTHON),
        ("tracehub-mcp", _TRACEHUB),
    ]
    if failing_awk:
        fakes.append(("awk", _FAILING_AWK))
    for name, body in fakes:
        exe = bin_dir / name
        exe.write_text(body)
        exe.chmod(0o755)
    script = tmp_path / "step.sh"
    script.write_text(_install_check_script())
    tool_dirs = {str(Path(str(shutil.which(tool))).parent) for tool in (*_TOOLS, "bash")}
    env = {
        "PATH": ":".join([str(bin_dir), *sorted(tool_dirs)]),
        "HOME": str(tmp_path),
        "NEW_VERSION": NEW_VERSION,
        "FAKE_STATE": str(state),
        "FAKE_PIP_SUCCEEDS_ON": str(succeeds_on),
        "FAKE_PIP_EXTRA_STDERR": extra_stderr,
        "FAKE_PIP_NOISE_LINES": str(noise_lines),
    }
    if runner_temp_mode == "set":
        env["RUNNER_TEMP"] = str(runner_temp)
    elif runner_temp_mode == "empty":
        env["RUNNER_TEMP"] = ""
    proc = await asyncio.create_subprocess_exec(
        bash,
        "-e",
        str(script),
        env=env,
        cwd=tmp_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    output = stdout.decode("utf-8", "replace") + stderr.decode("utf-8", "replace")
    return _Run(proc.returncode, output, state)


async def test_a_later_attempt_succeeds_and_each_failed_attempt_shows_pips_reason(
    tmp_path: Path,
) -> None:
    run = await _run_step(tmp_path, succeeds_on=3)

    assert run.returncode == 0, run.output
    assert "Installed on attempt 3" in run.output
    assert run.output.count("not yet available on PyPI") == 2
    # Two failed attempts, and pip's own reason is printed for each of them: the
    # "Could not find a version ..." line is the one that says what the index had.
    assert run.output.count("Could not find a version that satisfies") == 2, run.output
    assert run.output.count("No matching distribution found") == 2, run.output
    assert run.sleeps() == 2
    assert run.ran("python-ran") and run.ran("help-ran")


async def test_only_the_last_five_lines_of_pips_stderr_are_shown(tmp_path: Path) -> None:
    # 8 noise lines + pip's 2 ERROR lines = 10 stderr lines; one failed attempt.
    run = await _run_step(tmp_path, succeeds_on=2, noise_lines=8)

    assert run.returncode == 0, run.output
    for kept in (6, 7, 8):
        assert f"noise-line-{kept}" in run.output
    for dropped in (1, 2, 3, 4, 5):
        assert f"noise-line-{dropped}" not in run.output
    assert "Could not find a version that satisfies" in run.output
    assert "No matching distribution found" in run.output


async def test_giving_up_is_an_explicit_error_that_keeps_pips_last_reason(
    tmp_path: Path,
) -> None:
    run = await _run_step(tmp_path, succeeds_on=ATTEMPTS + 100)

    assert run.returncode == 1, run.output
    assert run.output.count("not yet available on PyPI") == ATTEMPTS
    assert run.output.count("Could not find a version that satisfies") == ATTEMPTS, run.output
    assert run.output.count("No matching distribution found") == ATTEMPTS, run.output
    assert "::error::" in run.output and f"tracehub-mcp=={NEW_VERSION}" in run.output
    assert "could not be installed" in run.output
    # It stops there: the import and --help must not run against a missing package,
    # which is what produced a bare ModuleNotFoundError for 0.13.2.
    assert not run.ran("python-ran") and not run.ran("help-ran")
    assert "ModuleNotFoundError" not in run.output


async def test_a_failing_diagnostic_does_not_end_the_retry_loop(tmp_path: Path) -> None:
    # Under bash -e a failing last pipeline stage would abort the whole step after the
    # first failed attempt; the diagnostic printing must never decide the outcome.
    run = await _run_step(tmp_path, succeeds_on=3, failing_awk=True)

    assert run.returncode == 0, run.output
    assert "Installed on attempt 3" in run.output
    assert run.output.count("not yet available on PyPI") == 2
    assert run.ran("python-ran") and run.ran("help-ran")


async def test_a_very_long_pip_line_is_cut_instead_of_repeated_in_full(tmp_path: Path) -> None:
    run = await _run_step(tmp_path, succeeds_on=ATTEMPTS + 100, extra_stderr="LONG-" + "x" * 5000)

    assert "LONG-" in run.output, "the start of the line must still be shown"
    # 500 characters of pip's line plus the "    pip: " prefix, never the whole 5000.
    assert max(len(line) for line in _logical_lines(run.output)) <= 500 + len("    pip: ")


@pytest.mark.parametrize("mode", ["unset", "empty"])
async def test_a_missing_runner_temp_fails_fast_without_blaming_pypi(
    tmp_path: Path, mode: str
) -> None:
    run = await _run_step(tmp_path, succeeds_on=1, runner_temp_mode=mode)

    assert run.returncode != 0, run.output
    assert "RUNNER_TEMP" in run.output
    # pip is never called and the log never claims the package is missing from PyPI.
    assert not (run.state / "pip-calls").exists()
    assert "not yet available on PyPI" not in run.output


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            "MARK-1 ::error::injected by the index\n::set-output name=x::MARK-1b",
            id="line-start-command",
        ),
        pytest.param(
            "MARK-2 first half\r::error::MARK-2b second half after a carriage return",
            id="command-after-carriage-return",
        ),
        pytest.param(
            "MARK-3 first half\r\n::error::MARK-3b after crlf",
            id="command-after-crlf",
        ),
        pytest.param("MARK-4 ERROR: x ##[error]MARK-4b boom", id="legacy-command-mid-line"),
        pytest.param("MARK-5 ####[error]MARK-5b stacked", id="legacy-command-stacked-hashes"),
        pytest.param(
            "MARK-7 ##[error]first ##[error]MARK-7b second on the same line",
            id="legacy-command-twice-on-one-line",
        ),
        pytest.param("   ::warning::MARK-6 after leading spaces", id="leading-whitespace"),
    ],
)
async def test_text_from_the_index_cannot_start_a_workflow_command(
    tmp_path: Path, payload: str
) -> None:
    run = await _run_step(tmp_path, succeeds_on=ATTEMPTS + 100, extra_stderr=payload)

    marker = payload.split()[0]
    assert marker in run.output, "pip's stderr must be shown"
    own_failure_line = f"::error::tracehub-mcp=={NEW_VERSION} "
    commands = []
    for line in _logical_lines(run.output):
        # The runner trims leading whitespace before looking for "::" and finds the
        # legacy "##[" form at any column.
        assert "##[" not in line, line
        if line.lstrip().startswith("::"):
            commands.append(line)
    # The only workflow command in the whole output is the step's own failure line.
    assert len(commands) == 1, run.output
    assert commands[0].startswith(own_failure_line), commands

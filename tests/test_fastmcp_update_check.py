"""FastMCP's PyPI update check must be OFF in a default tracehub-mcp run.

FastMCP (not this project) prints a banner at stdio start-up and, while doing
so, GETs https://pypi.org/pypi/fastmcp/json unless ``FASTMCP_CHECK_FOR_UPDATES``
says otherwise. That is egress to a host the operator never configured, from a
tool that is otherwise only meant to talk to its configured backend.

These tests observe BEHAVIOUR, not the line that switches the check off: the
real console-script entry point (``[project.scripts]`` in pyproject.toml) runs
in a subprocess under a Python audit hook (``sys.addaudithook``) that records
every ``socket.getaddrinfo`` / ``socket.gethostbyname`` / ``socket.connect``
aimed at a non-loopback host and RAISES inside the hook, so no real connection
is ever attempted. The subprocess environment is built from scratch (nothing is
inherited), so no FASTMCP_* variable, proxy, credential or real home directory
can leak in, and FastMCP's version cache (``<home>/version_cache.json``) lands
in a temp dir.

The opt-in test (``FASTMCP_CHECK_FOR_UPDATES=stable``) is the control: it
proves this harness does see the pypi.org lookup when it happens, so the
default-run test passing cannot be vacuous.
"""

import asyncio
import json
import os
import socket
import sys
import tomllib
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
_GENEROUS_TIMEOUT_SECONDS = 120.0

# Only printed by FastMCP's startup banner (fastmcp/utilities/cli.py). Seeing it
# in the default run proves the banner - the code path that performs the update
# check - was reached, i.e. the harness reproduces FastMCP's default banner
# behaviour rather than a run where the banner was silently off.
_BANNER_MARKER = "gofastmcp.com"

# Installed first in the child, before the package is imported, so import-time
# egress is seen too. argv: <src dir> <audit log path> <module:attr entry point>.
# What it runs afterwards is exactly what the generated console script does:
# `sys.exit(main())` with the entry point named in pyproject.toml.
_AUDIT_BOOTSTRAP = r"""
import importlib
import ipaddress
import json
import sys

src, log_path, entry = sys.argv[1:4]
sys.path.insert(0, src)

_HOST_EVENTS = {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}


def _is_local(host):
    if host is None or host == "":
        return True  # getaddrinfo(None, port): the local/wildcard address
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host.split("%")[0])
    except ValueError:
        return False
    return (getattr(address, "ipv4_mapped", None) or address).is_loopback


def _hook(event, args):
    if event in _HOST_EVENTS:
        host = args[0]
    elif event == "socket.connect":
        address = args[1]
        if not isinstance(address, tuple):
            return  # AF_UNIX path: not a network destination
        host = address[0]
    else:
        return
    if _is_local(host):
        return
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(json.dumps({"event": event, "host": str(host)}) + "\n")
    # An OSError subclass, so a caller that treats network failure as
    # non-fatal (FastMCP's update check does) carries on, while no connection
    # is ever made.
    raise ConnectionRefusedError(f"{event} {host!r} blocked by the test audit hook")


sys.addaudithook(_hook)
module_name, _, attr = entry.partition(":")
sys.argv = ["tracehub-mcp"]
sys.exit(getattr(importlib.import_module(module_name), attr)())
"""

# Reports what the settings object looks like once server.py has been imported.
_SETTINGS_PROBE = r"""
import json
import sys

sys.path.insert(0, sys.argv[1])
import opentelemetry_mcp.server
import fastmcp

settings = fastmcp.settings
print(json.dumps({
    "check_for_updates": settings.check_for_updates,
    "show_server_banner": settings.show_server_banner,
    "explicitly_set": sorted(settings.model_fields_set),
}))
"""

_INITIALIZE_REQUESTS = (
    json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "update-check-test", "version": "0"},
            },
        }
    )
    + "\n"
    + json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
    + "\n"
)


def _console_script_entry() -> str:
    """The `module:attr` the installed `tracehub-mcp` console script runs."""
    pyproject = tomllib.loads((_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(pyproject["project"]["scripts"]["tracehub-mcp"])


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _clean_env(root: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """An environment built from nothing: a throwaway HOME and XDG tree (so
    FastMCP's version cache never touches the real home), a backend that points
    at a loopback port nobody listens on, python-dotenv's .env walk disabled
    (config.py calls load_dotenv() at import), and no FASTMCP_*, proxy, AWS or
    OTel variable."""
    home = root / "home"
    scratch = root / "scratch"
    for directory in (home, scratch):
        directory.mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": os.defpath,
        "HOME": str(home),
        "TMPDIR": str(scratch),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_DATA_HOME": str(home / ".local" / "share"),
        "XDG_STATE_HOME": str(home / ".local" / "state"),
        "PYTHON_DOTENV_DISABLED": "1",
        "BACKEND_TYPE": "jaeger",
        "BACKEND_URL": f"http://127.0.0.1:{_unused_loopback_port()}",
    }
    env.update(extra or {})
    return env


async def _run_child(
    args: list[str], env: dict[str, str], cwd: Path, stdin_text: str = ""
) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=cwd,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(stdin_text.encode()), timeout=_GENEROUS_TIMEOUT_SECONDS
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return proc.returncode or 0, stdout.decode(), stderr.decode()


async def _run_stdio_server(
    root: Path, extra_env: dict[str, str] | None = None
) -> tuple[list[dict[str, str]], str, str]:
    """Start the real entry point on stdio, send `initialize` (so the server is
    provably up and serving, past the banner), close stdin, and return the
    non-loopback lookups the audit hook blocked plus the server's stdout/stderr."""
    audit_log = root / "non-loopback-attempts.jsonl"
    returncode, stdout, stderr = await _run_child(
        [
            sys.executable,
            "-I",
            "-c",
            _AUDIT_BOOTSTRAP,
            str(_SRC),
            str(audit_log),
            _console_script_entry(),
        ],
        _clean_env(root, extra_env),
        cwd=root,
        stdin_text=_INITIALIZE_REQUESTS,
    )
    assert returncode == 0, f"server exited {returncode}\n--- stderr ---\n{stderr}"
    responses = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    initialize = next((r for r in responses if r.get("id") == 1), None)
    assert initialize is not None and "result" in initialize, (
        f"no initialize result on stdout: {stdout!r}\n--- stderr ---\n{stderr}"
    )
    assert initialize["result"]["serverInfo"]["name"] == "tracehub-mcp", initialize
    attempts = (
        [json.loads(line) for line in audit_log.read_text(encoding="utf-8").splitlines()]
        if audit_log.exists()
        else []
    )
    return attempts, stdout, stderr


async def test_default_stdio_run_makes_no_non_loopback_lookup(tmp_path: Path) -> None:
    attempts, _stdout, stderr = await _run_stdio_server(tmp_path)

    # Non-vacuity: FastMCP's banner (the only place the update check runs) was
    # printed, because nothing in this run turned it off.
    assert _BANNER_MARKER in stderr, stderr
    assert attempts == [], (
        "a default run reached for a non-loopback host (FastMCP's PyPI update check "
        f"should be off by default): {attempts}"
    )


async def test_opt_in_update_check_still_reaches_pypi(tmp_path: Path) -> None:
    attempts, _stdout, stderr = await _run_stdio_server(
        tmp_path, {"FASTMCP_CHECK_FOR_UPDATES": "stable"}
    )

    assert _BANNER_MARKER in stderr, stderr
    hosts = {attempt["host"] for attempt in attempts}
    assert "pypi.org" in hosts, (
        "an operator who sets FASTMCP_CHECK_FOR_UPDATES=stable must keep FastMCP's "
        f"update check (and this harness must be able to see it): {attempts}"
    )


@pytest.mark.parametrize(
    ("env_var", "dotenv_line", "expected_check", "expected_explicitly_set"),
    [
        pytest.param(None, None, "off", ["check_for_updates"], id="unset-defaults-off"),
        pytest.param("stable", None, "stable", ["check_for_updates"], id="env-stable-kept"),
        pytest.param(
            "prerelease", None, "prerelease", ["check_for_updates"], id="env-prerelease-kept"
        ),
        pytest.param("off", None, "off", ["check_for_updates"], id="env-off-kept"),
        pytest.param(
            None,
            "FASTMCP_CHECK_FOR_UPDATES=prerelease",
            "prerelease",
            ["check_for_updates"],
            id="dotenv-prerelease-kept",
        ),
    ],
)
async def test_default_is_off_and_the_operators_own_choice_is_kept(
    tmp_path: Path,
    env_var: str | None,
    dotenv_line: str | None,
    expected_check: str,
    expected_explicitly_set: list[str],
) -> None:
    """Importing server.py switches the check off only when the operator said
    nothing - through the environment OR through the .env file FastMCP's own
    settings read from the working directory - and touches no other setting
    (the banner stays as FastMCP decides)."""
    if dotenv_line is not None:
        (tmp_path / ".env").write_text(dotenv_line + "\n", encoding="utf-8")
    extra = {"FASTMCP_CHECK_FOR_UPDATES": env_var} if env_var is not None else None
    returncode, stdout, stderr = await _run_child(
        [sys.executable, "-I", "-c", _SETTINGS_PROBE, str(_SRC)],
        _clean_env(tmp_path, extra),
        cwd=tmp_path,
    )
    assert returncode == 0, stderr
    observed = json.loads(stdout.splitlines()[-1])

    assert observed == {
        "check_for_updates": expected_check,
        "show_server_banner": True,
        "explicitly_set": expected_explicitly_set,
    }, observed

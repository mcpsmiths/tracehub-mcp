"""Tests for docker_healthcheck.py - the Docker HEALTHCHECK script used by
the Dockerfile's streamable-http transport (default CMD).

Regression coverage for the finding that this script probed /mcp (process
liveness only - ANY HTTP response, even a 4xx, counted as healthy; only a
refused/timed-out connection failed it) instead of /ready (server.py's
ready_route, which runs a real health_check() + list_services() probe
against the configured trace backend and returns 503 when that probe
fails). Before the fix, a sustained backend outage never made the
container's own HEALTHCHECK report unhealthy.
"""

import email.message
import importlib.util
import urllib.error
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HEALTHCHECK_PATH = REPO_ROOT / "docker_healthcheck.py"


def _run_healthcheck() -> ModuleType:
    """Load and execute docker_healthcheck.py fresh, the same way `python
    docker_healthcheck.py` (the Dockerfile's HEALTHCHECK CMD) does - the
    probe runs as a side effect of module-level code, not a callable, so
    each test needs its own isolated execution rather than a cached
    import."""
    spec = importlib.util.spec_from_file_location("docker_healthcheck_under_test", HEALTHCHECK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_healthcheck_probes_ready_endpoint_not_mcp() -> None:
    """The script must hit /ready (real backend health, per server.py's
    ready_route), not /mcp (process liveness only)."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value = MagicMock()
        _run_healthcheck()

    requested_url = mock_urlopen.call_args[0][0]
    assert requested_url == "http://localhost:8000/ready"


def test_healthcheck_fails_on_503_not_ready() -> None:
    """A 503 from /ready (backend probe failed, per ready_route returning
    503 on health_check()/list_services() failure) must be treated as
    unhealthy - i.e. the script must exit(1)."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "http://localhost:8000/ready",
            503,
            "Service Unavailable",
            email.message.Message(),
            None,
        )
        with pytest.raises(SystemExit) as exc_info:
            _run_healthcheck()

    assert exc_info.value.code == 1


def test_healthcheck_succeeds_on_200_ready() -> None:
    """A 200 from /ready (backend healthy) must not raise or exit - this
    is the only response that should leave the container marked healthy."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value = MagicMock()
        _run_healthcheck()  # must not raise SystemExit

    mock_urlopen.assert_called_once()


def test_healthcheck_fails_on_connection_refused() -> None:
    """A refused/timed-out connection must still be treated as unhealthy
    (unchanged behavior from before the fix)."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        with pytest.raises(SystemExit) as exc_info:
            _run_healthcheck()

    assert exc_info.value.code == 1

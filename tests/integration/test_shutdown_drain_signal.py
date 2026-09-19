"""Real-subprocess, real-SIGTERM re-validation of the graceful-shutdown
lifespan splice (_install_shutdown_drain, server.py).

This is the exact class of test that would have caught the original bug: a
constructor-level FastMCP(lifespan=...) attempt whose teardown never fired
on a real SIGTERM (uvicorn's capture_signals() restores the default signal
handler and re-raises the captured SIGTERM before Server.serve() itself
ever returns, killing the process first) - tried and reverted in favor of
the current app.router.lifespan_context splice (commit 1d4d1c6).
tests/test_shutdown_drain.py's TestClient-based tests drive the ASGI
lifespan.shutdown message directly and prove the teardown is wired to the
right lifespan object, but can never reproduce a real OS signal killing the
process before that teardown gets a chance to run - only a real subprocess
and a real signal can.

Not VCR-based like the other tests in this directory - a slow-backend HTTP
stub (_slow_backend_stub.py) with a deliberate artificial delay is used
instead, since exact response TIMING (keeping a real tool call provably
in-flight across the SIGTERM boundary) is the entire point, and a cassette
replay can't control that.

A real, separate finding from building this test (documented here rather
than silently worked around): uvicorn's Server.shutdown() calls
connection.shutdown() on every open connection immediately on SIGTERM
(uvicorn/server.py:272-274) - this lets the in-flight response finish
sending (confirmed via the server's own access log: the tools/call request
gets a real 200 OK even when it started before SIGTERM), but the
subsequent connection closure is something the MCP streamable-http client's
own SSE-stream reader treats as a connection error rather than a graceful
end-of-response signal. That is a real interaction between uvicorn and the
mcp/httpx2 client transport, not a defect in this project's own
_drain_and_close_backend/_install_shutdown_drain - so this test verifies
success via server-side log evidence (the backend call and the tool
response both completing, in the right order, after shutdown began) rather
than asserting the client itself receives a clean, error-free result.
"""

import asyncio
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator

import httpx
import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from tests.integration._slow_backend_stub import SlowBackendStub

pytestmark = pytest.mark.integration


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def _wait_for_ready(port: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/ready", timeout=1.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError as e:
            last_error = e
        time.sleep(0.1)
    raise TimeoutError(f"Server on port {port} never became ready: {last_error}")


def _spawn_server(port: int, backend_url: str, **extra_flags: str) -> subprocess.Popen[str]:
    args = [
        sys.executable,
        "-m",
        "opentelemetry_mcp.server",
        "--transport",
        "http",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--backend",
        "jaeger",
        "--url",
        backend_url,
    ]
    for flag, value in extra_flags.items():
        args.extend([flag, value])
    # All args are this file's own hardcoded literals plus sys.executable -
    # nothing here originates from external/untrusted input.
    return subprocess.Popen(  # noqa: S603
        args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )


def _terminate_if_still_running(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)


@pytest.fixture
def slow_backend() -> Iterator[SlowBackendStub]:
    with SlowBackendStub(delay_seconds=1.5) as stub:
        yield stub


class TestShutdownDrainSignal:
    async def test_in_flight_backend_call_completes_across_a_real_sigterm(
        self, slow_backend: SlowBackendStub
    ) -> None:
        port = _free_port()
        proc = _spawn_server(
            port,
            slow_backend.url,
            **{
                "--shutdown-drain-seconds": "3",
                "--backend-close-timeout-seconds": "2",
                "--graceful-shutdown-timeout-seconds": "10",
            },
        )
        try:
            _wait_for_ready(port)

            transport = StreamableHttpTransport(f"http://127.0.0.1:{port}/mcp")
            client = Client(transport)
            await client.__aenter__()  # type: ignore[no-untyped-call]
            try:
                # Fire the call and hand control back immediately (no
                # await here) so SIGTERM below is provably sent while the
                # 1.5s-delayed backend call is genuinely in flight, not
                # after it already returned.
                call_task = asyncio.ensure_future(
                    client.call_tool("search_traces", {"service_name": "stub-service", "limit": 5})
                )
                proc.send_signal(signal.SIGTERM)
                sigterm_sent_at = time.monotonic()
                # See module docstring: the connection-closure interaction
                # with the MCP streamable-http client's own SSE reader is a
                # separate, real finding, not what this test verifies -
                # only the server-side evidence below is asserted on.
                try:
                    await asyncio.wait_for(call_task, timeout=8)
                except (Exception, TimeoutError):
                    pass
            finally:
                try:
                    await client.__aexit__(None, None, None)  # type: ignore[no-untyped-call]
                except Exception:
                    pass

            exit_code = proc.wait(timeout=15)
            output = proc.stdout.read() if proc.stdout else ""
        finally:
            _terminate_if_still_running(proc)

        # -signal.SIGTERM (-15), not 0: capture_signals() (uvicorn) restores
        # the default SIGTERM disposition and re-raises the captured signal
        # once teardown completes, so this process always dies via the OS's
        # default signal termination rather than a clean sys.exit(0) - by
        # design, confirmed by this exact mechanism during the original
        # graceful-shutdown work (commit 1d4d1c6).
        assert exit_code == -signal.SIGTERM, output

        shutdown_idx = output.index("Shutting down")
        backend_call_idx = output.index("/api/traces")
        tool_response_idx = output.rindex('"POST /mcp HTTP/1.1" 200 OK')
        drain_idx = output.index("shutting down - draining")
        closed_idx = output.index("Backend closed during shutdown drain")

        assert shutdown_idx < backend_call_idx, (
            "the backend call must have started (or at least completed) "
            "after shutdown began - otherwise this test isn't proving "
            "anything was actually in flight at the SIGTERM boundary"
        )
        assert backend_call_idx < tool_response_idx, (
            "the backend call must complete before the tool call's own "
            "200 OK response is sent - confirms the in-flight operation "
            "was not abandoned mid-flight"
        )
        assert tool_response_idx < drain_idx < closed_idx, (
            "drain must start only after the in-flight request is fully "
            "handled, and the close-confirmation log must come after that"
        )
        assert time.monotonic() - sigterm_sent_at < 10, (
            "process took unexpectedly long to reach the point where "
            "logs could be read back - possible hang"
        )

    async def test_hung_backend_close_still_exits_within_the_configured_bound(self) -> None:
        """The fallback path: if backend.close() hangs past
        --backend-close-timeout-seconds, the process must still exit within
        drain+close_timeout+margin rather than hanging forever - this is
        what --backend-close-timeout-seconds (Step 9) exists to make
        deterministically testable at all."""
        with SlowBackendStub(delay_seconds=30) as hung_backend:
            port = _free_port()
            proc = _spawn_server(
                port,
                hung_backend.url,
                **{
                    "--shutdown-drain-seconds": "0",
                    "--backend-close-timeout-seconds": "1",
                    "--graceful-shutdown-timeout-seconds": "2",
                },
            )
            try:
                _wait_for_ready(port)
                start = time.monotonic()
                proc.send_signal(signal.SIGTERM)
                exit_code = proc.wait(timeout=10)
                elapsed = time.monotonic() - start
            finally:
                _terminate_if_still_running(proc)

            # -signal.SIGTERM (-15), not 0 - see the other test's comment
            # for why this process always dies via signal termination.
            assert exit_code == -signal.SIGTERM
            assert elapsed < 8, (
                f"Process took {elapsed:.1f}s to exit - the "
                "--backend-close-timeout-seconds fallback should bound this "
                "well below the hung backend's 30s delay"
            )

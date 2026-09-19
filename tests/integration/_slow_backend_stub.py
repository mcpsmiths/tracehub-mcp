"""A minimal Jaeger-shaped HTTP stub for the real-SIGTERM shutdown-drain
integration test (test_shutdown_drain_signal.py).

Not VCR-based like every other integration test in this directory -
this test needs to control exact response TIMING (a deliberate artificial
delay on /api/traces) to keep a real tool call provably in flight across a
real SIGTERM, which a cassette replay cannot do. Uses only the stdlib
http.server rather than pulling in a second async HTTP framework for a
one-off test fixture.
"""

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import TracebackType


class _Handler(BaseHTTPRequestHandler):
    delay_seconds: float = 0.0

    def log_message(self, format: str, *args: object) -> None:
        """Silence stdlib's default per-request stderr logging - noisy in
        test output and irrelevant to what this fixture verifies."""

    def do_GET(self) -> None:  # noqa: N802 - stdlib-mandated method name
        if self.path.startswith("/api/services"):
            body = json.dumps({"data": ["stub-service"]}).encode()
        elif self.path.startswith("/api/traces"):
            if self.delay_seconds > 0:
                time.sleep(self.delay_seconds)
            body = json.dumps({"data": []}).encode()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class SlowBackendStub:
    """Context manager owning a background HTTP server thread serving a
    Jaeger-shaped /api/services (instant) and /api/traces (configurably
    delayed) - just enough surface for JaegerBackend's health_check/
    list_services/search_traces to succeed against it."""

    def __init__(self, delay_seconds: float = 0.0) -> None:
        handler_cls = type("_BoundHandler", (_Handler,), {"delay_seconds": delay_seconds})
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        # Always bound to 127.0.0.1 (see __init__) - hardcoded rather than
        # read back from server_address, whose typeshed type is str | bytes
        # for the AF_UNIX case this server never uses.
        port = self._server.server_address[1]
        return f"http://127.0.0.1:{port}"

    def __enter__(self) -> "SlowBackendStub":
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._server.shutdown()
        self._server.server_close()

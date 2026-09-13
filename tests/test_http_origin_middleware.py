"""Tests for OriginValidationMiddleware, the Origin-header guard wired into
the streamable-http transport (see server.py's main() and the class
docstring for the fastmcp 3.2.0 gap this middleware closes).

This wraps only the middleware itself around a trivial downstream handler -
spinning up the full MCP server here would be slow and out of scope.
"""

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from opentelemetry_mcp.server import OriginValidationMiddleware


async def _downstream_handler(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


def _build_client() -> TestClient:
    app = Starlette(
        routes=[Route("/", _downstream_handler)],
        middleware=[Middleware(OriginValidationMiddleware)],
    )
    return TestClient(app)


def test_missing_origin_header_passes_through() -> None:
    """Non-browser MCP clients typically send no Origin header at all;
    those requests must reach the downstream handler unmodified."""
    client = _build_client()

    response = client.get("/")

    assert response.status_code == 200
    assert response.text == "ok"


def test_local_dev_origin_passes_through() -> None:
    """A same-machine dev client (e.g. a browser-based MCP inspector on
    localhost) must be allowed through."""
    client = _build_client()

    response = client.get("/", headers={"origin": "http://localhost:5173"})

    assert response.status_code == 200
    assert response.text == "ok"


def test_untrusted_origin_is_rejected() -> None:
    """A malicious webpage's Origin must be rejected with 403, without
    reaching the downstream handler (DNS-rebinding protection)."""
    client = _build_client()

    response = client.get("/", headers={"origin": "https://evil.example.com"})

    assert response.status_code == 403
    assert response.text == "Invalid Origin header"

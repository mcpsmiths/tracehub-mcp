"""Raw streamable-HTTP request helpers shared by the HTTP-listener tests.

The in-memory ``fastmcp.Client`` bypasses the Starlette app entirely, so
anything that lives at the HTTP layer - bearer-token enforcement, Host/Origin
validation, session lifetime, per-peer rate limiting, ``params._meta`` trace
context - can only be exercised by speaking the wire protocol through
``starlette.testclient.TestClient``. These helpers keep that protocol detail
(the mandatory ``Accept`` pair, the ``mcp-session-id`` round trip, the
``notifications/initialized`` handshake, SSE framing of responses) in one
place so each test file states only what it is actually asserting.

Not a test module itself (no ``test_`` prefix), so pytest never collects it.
"""

import json
from typing import Any

from starlette.testclient import TestClient

# starlette 1.x's TestClient speaks httpx2 when it is installed and falls back
# to httpx otherwise; mirror that so the response annotations below match
# whatever TestClient actually returns.
try:
    import httpx2 as httpx
except ImportError:  # pragma: no cover - httpx2 is pinned transitively
    import httpx  # type: ignore[no-redef]

MCP_PATH = "/mcp"
# The SDK rejects a POST whose Accept header does not list both media types.
ACCEPT_HEADERS = {"Accept": "application/json, text/event-stream"}
PROTOCOL_VERSION = "2025-06-18"


def initialize_request(request_id: int = 1) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "initialize",
        "params": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "tracehub-mcp-tests", "version": "0"},
        },
    }


def sse_json(response: httpx.Response) -> dict[str, Any]:
    """Return the first JSON-RPC message carried by an SSE (or plain JSON)
    response body."""
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        body: dict[str, Any] = response.json()
        return body
    for line in response.text.splitlines():
        if line.startswith("data:"):
            message: dict[str, Any] = json.loads(line[len("data:") :].strip())
            return message
    raise AssertionError(f"no SSE data frame in response body: {response.text!r}")


def post_initialize(client: TestClient, *, headers: dict[str, str] | None = None) -> httpx.Response:
    """POST a bare initialize request (no session id) and return the raw
    response, so callers can assert on status/headers of the handshake itself."""
    return client.post(
        MCP_PATH, json=initialize_request(), headers={**ACCEPT_HEADERS, **(headers or {})}
    )


def open_session(client: TestClient, *, headers: dict[str, str] | None = None) -> str:
    """Run the full initialize + notifications/initialized handshake and
    return the server-issued ``mcp-session-id``."""
    extra = headers or {}
    response = post_initialize(client, headers=extra)
    assert response.status_code == 200, (response.status_code, response.text)
    session_id: str | None = response.headers.get("mcp-session-id")
    assert session_id, "initialize response carried no mcp-session-id"
    initialized = client.post(
        MCP_PATH,
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        headers={**ACCEPT_HEADERS, "mcp-session-id": session_id, **extra},
    )
    assert initialized.status_code == 202, (initialized.status_code, initialized.text)
    return session_id


def post_request(
    client: TestClient,
    session_id: str | None,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    request_id: int = 2,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """POST one JSON-RPC request inside an already-open session (or with no
    session header when ``session_id`` is None) and return the raw response."""
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    session_headers = {"mcp-session-id": session_id} if session_id is not None else {}
    return client.post(
        MCP_PATH,
        json=message,
        headers={**ACCEPT_HEADERS, **session_headers, **(headers or {})},
    )


def call_tool(
    client: TestClient,
    session_id: str,
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    meta: dict[str, Any] | None = None,
    request_id: int = 2,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    params: dict[str, Any] = {"name": name, "arguments": arguments or {}}
    if meta is not None:
        params["_meta"] = meta
    return post_request(
        client, session_id, "tools/call", params, request_id=request_id, headers=headers
    )

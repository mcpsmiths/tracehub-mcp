"""Tests for OriginValidationMiddleware, the Origin- and Host-header guard
wired into the streamable-http transport (see server.py's main() and the
class docstring for the fastmcp 3.2.0 gap this middleware closes).

This wraps only the middleware itself around a trivial downstream handler -
spinning up the full MCP server here would be slow and out of scope.
"""

from collections.abc import Sequence

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from opentelemetry_mcp.server import OriginValidationMiddleware, _hostname_from_host_header

ALL_INTERFACES = "0.0.0.0"  # noqa: S104 - the network bind shape under test


async def _downstream_handler(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


def _build_client(
    *,
    bind_host: str | None = None,
    allowed_hosts: Sequence[str] | None = None,
    reached: list[str] | None = None,
) -> TestClient:
    """`reached` (when given) records the Host header of every request that
    made it past the middleware to the downstream handler."""

    async def recording_handler(request: Request) -> PlainTextResponse:
        if reached is not None:
            reached.append(request.headers.get("host", ""))
        return PlainTextResponse("ok")

    middleware = (
        Middleware(OriginValidationMiddleware)
        if bind_host is None and allowed_hosts is None
        else Middleware(
            OriginValidationMiddleware, bind_host=bind_host, allowed_hosts=allowed_hosts
        )
    )
    app = Starlette(routes=[Route("/", recording_handler)], middleware=[middleware])
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


class TestHostValidation:
    """A DNS-rebound page's follow-up GET arrives with the attacker's name in
    Host and (per the Fetch standard for same-origin GETs) no Origin at all,
    so an Origin-only check lets it through. When the legitimate names are
    known - a loopback bind, or an operator allowlist - the Host header is
    checked first and anything else answers 421 before any route runs."""

    def test_loopback_bind_rejects_a_foreign_host_without_origin(self) -> None:
        reached: list[str] = []
        client = _build_client(bind_host="127.0.0.1", reached=reached)

        response = client.get("/", headers={"host": "attacker.example:8000"})

        assert response.status_code == 421
        assert response.text == "Misdirected Request"
        assert reached == []

    def test_loopback_bind_accepts_every_loopback_spelling(self) -> None:
        client = _build_client(bind_host="127.0.0.1")

        for host in ("localhost:8000", "127.0.0.1:8000", "[::1]:8000", "localhost", "127.0.0.1"):
            response = client.get("/", headers={"host": host})
            assert response.status_code == 200, host

    def test_host_is_checked_before_origin(self) -> None:
        client = _build_client(bind_host="127.0.0.1")

        response = client.get(
            "/",
            headers={"host": "attacker.example:8000", "origin": "http://attacker.example:8000"},
        )

        assert response.status_code == 421

    def test_network_bind_without_allowlist_performs_no_host_validation(self) -> None:
        """The documented 0.0.0.0 shape (Dockerfile CMD) has unknown
        legitimate names, so its contract is unchanged until the operator
        names them with --allowed-hosts."""
        client = _build_client(bind_host=ALL_INTERFACES)

        response = client.get("/", headers={"host": "attacker.example:8000"})

        assert response.status_code == 200

    def test_network_bind_with_allowlist_enforces_it_plus_loopback(self) -> None:
        client = _build_client(bind_host=ALL_INTERFACES, allowed_hosts=["tracehub.example"])

        assert client.get("/", headers={"host": "tracehub.example:8000"}).status_code == 200
        assert client.get("/", headers={"host": "TraceHub.Example"}).status_code == 200
        assert client.get("/", headers={"host": "other.example:8000"}).status_code == 421
        assert client.get("/", headers={"host": "localhost:8000"}).status_code == 200

    def test_specific_bind_address_is_always_allowed(self) -> None:
        client = _build_client(bind_host="10.0.0.5", allowed_hosts=["tracehub.example"])

        assert client.get("/", headers={"host": "10.0.0.5:8000"}).status_code == 200
        assert client.get("/", headers={"host": "other.example"}).status_code == 421

    def test_wildcard_allowlist_opts_out_of_host_validation(self) -> None:
        client = _build_client(bind_host="127.0.0.1", allowed_hosts=["*"])

        response = client.get("/", headers={"host": "attacker.example:8000"})

        assert response.status_code == 200

    def test_no_kwargs_keeps_the_pre_existing_origin_only_contract(self) -> None:
        reached: list[str] = []
        client = _build_client(reached=reached)

        response = client.get("/", headers={"host": "attacker.example:8000"})

        assert response.status_code == 200
        assert reached == ["attacker.example:8000"]


class TestHostnameFromHostHeader:
    def test_strips_port_and_lowercases(self) -> None:
        assert _hostname_from_host_header("TraceHub.Example:8000") == "tracehub.example"
        assert _hostname_from_host_header("127.0.0.1:8000") == "127.0.0.1"
        assert _hostname_from_host_header("localhost") == "localhost"

    def test_ipv6_literals(self) -> None:
        assert _hostname_from_host_header("[::1]:8000") == "::1"
        assert _hostname_from_host_header("[::1]") == "::1"
        assert _hostname_from_host_header("::1") == "::1"

    def test_missing_header_is_empty(self) -> None:
        assert _hostname_from_host_header("") == ""
        assert _hostname_from_host_header("   ") == ""

"""Tests for OriginValidationMiddleware, the Origin- and Host-header guard
wired into the streamable-http transport (see server.py's main() and the
class docstring for the fastmcp 3.2.0 gap this middleware closes).

This wraps only the middleware itself around a trivial downstream handler -
spinning up the full MCP server here would be slow and out of scope.
"""

import logging
from collections.abc import Sequence

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from opentelemetry_mcp.server import (
    _MAX_LOGGED_HEADER_CHARS,
    OriginValidationMiddleware,
    _hostname_from_host_header,
)

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


class TestBracketedHostTrailingText:
    """Runtime verification prober-1 F7: '[::1]evil.example' was accepted
    because everything after the closing ']' was ignored. After a ']' only
    ':<port>' or the end of the value is a Host header; anything else - and
    an unclosed '[' - is malformed and answers 421 like every other refused
    Host."""

    def test_text_after_the_closing_bracket_is_refused_on_a_loopback_bind(self) -> None:
        client = _build_client(bind_host="127.0.0.1")

        assert client.get("/", headers={"host": "[::1]evil.example"}).status_code == 421
        assert client.get("/", headers={"host": "[::1]:8000"}).status_code == 200
        assert client.get("/", headers={"host": "[::1]"}).status_code == 200

    def test_malformed_bracketed_values_never_match_an_allowlist_entry(self) -> None:
        client = _build_client(bind_host=ALL_INTERFACES, allowed_hosts=["[2001:db8::1]"])

        for host in (
            "[2001:db8::1]evil.example",
            "[2001:db8::1]:abc",
            "[2001:db8::1]:",
            "[2001:db8::1",
        ):
            assert client.get("/", headers={"host": host}).status_code == 421, host
        assert client.get("/", headers={"host": "[2001:db8::1]:8443"}).status_code == 200

    def test_hostname_from_host_header_keeps_a_malformed_bracketed_value_whole(self) -> None:
        assert _hostname_from_host_header("[::1]evil.example") == "[::1]evil.example"
        assert _hostname_from_host_header("[::1]:abc") == "[::1]:abc"
        assert _hostname_from_host_header("[::1]:") == "[::1]:"
        assert _hostname_from_host_header("[::1") == "[::1"
        assert _hostname_from_host_header("[::1]:8000") == "::1"
        assert _hostname_from_host_header("[::1]") == "::1"


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


class TestRejectionLogging:
    """Both rejection paths log at WARNING *before* RateLimitMiddleware can
    throttle the peer (this class is first in main()'s middleware list and
    Starlette applies the list outermost-first), so an unauthenticated peer
    gets one log line per request. Each line must therefore be bounded no
    matter how large the offending header is, and repr()-quoted so it can
    never carry raw control characters."""

    def test_unexpected_host_log_line_is_bounded_and_quoted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _build_client(bind_host="127.0.0.1")
        oversized = "evil.example" + "a" * (50 * _MAX_LOGGED_HEADER_CHARS)
        with caplog.at_level(logging.WARNING, logger="opentelemetry_mcp.server"):
            response = client.get("/", headers={"host": oversized})

        assert response.status_code == 421
        [message] = [
            r.getMessage() for r in caplog.records if "Unexpected Host header" in r.getMessage()
        ]
        assert "'evil.example" in message  # repr()-quoted, prefix preserved for diagnosis
        assert len(message) <= len("Unexpected Host header: ''") + _MAX_LOGGED_HEADER_CHARS

    def test_invalid_origin_log_line_is_bounded_and_quoted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _build_client()
        oversized = "https://evil.example/" + "a" * (50 * _MAX_LOGGED_HEADER_CHARS)
        with caplog.at_level(logging.WARNING, logger="opentelemetry_mcp.server"):
            response = client.get("/", headers={"origin": oversized})

        assert response.status_code == 403
        [message] = [
            r.getMessage() for r in caplog.records if "Invalid Origin header" in r.getMessage()
        ]
        assert "'https://evil.example/" in message
        assert len(message) <= len("Invalid Origin header: ''") + _MAX_LOGGED_HEADER_CHARS

    def test_short_values_are_logged_whole(self, caplog: pytest.LogCaptureFixture) -> None:
        """Positive control for the bound: a realistic header is not cut."""
        client = _build_client(bind_host="127.0.0.1")
        with caplog.at_level(logging.WARNING, logger="opentelemetry_mcp.server"):
            client.get("/", headers={"host": "attacker.example:8000"})

        assert "Unexpected Host header: 'attacker.example:8000'" in caplog.text


class TestHostAllowlistNormalisation:
    """Prober-1 F7: a trailing dot was stripped for loopback names
    ('localhost.' -> 200, via is_loopback_host) but not for allowlisted
    names ('verify.local.' -> 421), and an allowlisted IPv6 literal was
    compared as a string, so its expanded spelling answered 421. Both sides
    now go through the same normalisation: one trailing dot removed and
    IPv6 literals in their RFC 5952 compressed form."""

    def test_trailing_dot_fqdn_spelling_matches_an_allowlisted_name(self) -> None:
        client = _build_client(bind_host=ALL_INTERFACES, allowed_hosts=["verify.local"])

        assert client.get("/", headers={"host": "verify.local.:8000"}).status_code == 200
        assert client.get("/", headers={"host": "Verify.Local."}).status_code == 200
        # Only one trailing dot is the FQDN spelling; more is not a name.
        assert client.get("/", headers={"host": "verify.local.."}).status_code == 421

    def test_allowlist_entry_with_a_trailing_dot_matches_the_bare_name(self) -> None:
        client = _build_client(bind_host=ALL_INTERFACES, allowed_hosts=["verify.local."])

        assert client.get("/", headers={"host": "verify.local:8000"}).status_code == 200

    def test_loopback_trailing_dot_still_passes(self) -> None:
        client = _build_client(bind_host="127.0.0.1")

        assert client.get("/", headers={"host": "localhost.:8000"}).status_code == 200

    def test_every_spelling_of_an_allowlisted_ipv6_literal_matches(self) -> None:
        client = _build_client(bind_host=ALL_INTERFACES, allowed_hosts=["[2001:db8::1]:8443"])

        for host in (
            "[2001:db8::1]:8000",
            "[2001:0db8:0000:0000:0000:0000:0000:0001]:8000",
            "[2001:DB8:0:0:0:0:0:1]",
            "[2001:db8:0:0::1]",
        ):
            assert client.get("/", headers={"host": host}).status_code == 200, host
        assert client.get("/", headers={"host": "[2001:db8::2]:8000"}).status_code == 421

    def test_expanded_allowlist_entry_matches_the_compressed_header(self) -> None:
        client = _build_client(
            bind_host=ALL_INTERFACES, allowed_hosts=["[2001:0db8:0000:0000:0000:0000:0000:0001]"]
        )

        assert client.get("/", headers={"host": "[2001:db8::1]:8000"}).status_code == 200

    def test_hostname_from_host_header_normalises_both_shapes(self) -> None:
        assert _hostname_from_host_header("Verify.Local.:8000") == "verify.local"
        assert _hostname_from_host_header("verify.local..") == "verify.local."
        assert _hostname_from_host_header("[2001:0DB8::0001]:80") == "2001:db8::1"
        assert _hostname_from_host_header("2001:0db8:0:0:0:0:0:1") == "2001:db8::1"
        # Not an address: left alone (and so never equal to a real literal).
        assert _hostname_from_host_header("[not:an:address]") == "not:an:address"

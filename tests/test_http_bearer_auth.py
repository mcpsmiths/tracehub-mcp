"""Tests for the streamable-HTTP listener's caller authentication: the
OperatorTokenVerifier that binds every /mcp request to the operator's shared
bearer token, the startup gate in main() that refuses to serve HTTP on a
non-loopback interface without it, and the fastmcp RequireAuthMiddleware
wiring that protects /mcp while leaving /health and /ready (the Docker
HEALTHCHECK probe) reachable.

Drives the real Starlette app FastMCP assembles - exactly as main() builds
it - through starlette's TestClient with raw JSON-RPC bodies (see
tests/streamable_http_client.py), because the in-memory fastmcp.Client never
touches the HTTP layer where authentication is enforced.

The fixture credential is assembled from fragments (the pattern
test_observability.py already uses) so this file's own static content has no
contiguous secret-shaped literal for a scanner to flag.
"""

# server.py imports hmac as a module and calls hmac.compare_digest through it,
# so `server.hmac` exists at runtime but mypy's strict --no-implicit-reexport
# treats it as undefined - same friction test_server.py's docstring documents.
# mypy: disable-error-code="attr-defined"

import asyncio
import hmac
import io
import json
import logging
import threading
import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import click
import httpx
import pytest
import uvicorn
from click.testing import CliRunner
from starlette.middleware import Middleware
from starlette.testclient import TestClient
from uvicorn.logging import AccessFormatter

from opentelemetry_mcp import server
from opentelemetry_mcp.attributes import HealthCheckResponse
from opentelemetry_mcp.server import (
    AccessLogQueryStringFilter,
    OperatorTokenVerifier,
    OriginValidationMiddleware,
    RateLimitMiddleware,
    _build_uvicorn_config,
)
from tests.streamable_http_client import open_session, post_initialize, post_request, sse_json

# 46 characters, comfortably over the 32-character floor.
TEST_TOKEN = "-".join(["unit", "test", "operator", "credential", "0123456789abcdef"])
# 48 characters a peer might present in the query string instead of the header.
QUERY_DUMMY = "-".join(["query", "string", "dummy", "bearer", "value", "access", "log", "probe"])
SHORT_VALUE = "-".join(["too", "short"])
# The test subject is exactly the all-interfaces bind the gate exists to guard.
ALL_INTERFACES = "0.0.0.0"  # noqa: S104


@pytest.fixture(autouse=True)
def _reset_mcp_auth() -> Generator[None]:
    """server.mcp is a module-level singleton; its .auth attribute is what
    main() mutates to install the verifier, so every test here must leave
    it as it found it (None) or later HTTP tests would start answering 401."""
    server.mcp.auth = None
    yield
    server.mcp.auth = None


@pytest.fixture
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BACKEND_TYPE", "jaeger")
    monkeypatch.setenv("BACKEND_URL", "http://localhost:16686")
    for name in (
        "BACKEND_API_KEY",
        "BACKEND_APP_KEY",
        "BACKEND_SENTRY_ORG",
        "BACKEND_SENTRY_PROJECT",
        "BACKEND_ENVIRONMENTS",
        "MCP_HTTP_BEARER_TOKEN",
        "MCP_ALLOW_UNAUTHENTICATED_HTTP",
        "MCP_HOST",
        "MCP_TRANSPORT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)


def _build_app_as_main_does() -> TestClient:
    app = server.mcp.http_app(
        transport="streamable-http",
        middleware=[
            Middleware(OriginValidationMiddleware),
            Middleware(RateLimitMiddleware, max_requests=100, window_seconds=60.0),
        ],
    )
    return TestClient(app)


def _healthy_backend() -> AsyncMock:
    fake_backend = AsyncMock()
    fake_backend.health_check = AsyncMock(
        return_value=HealthCheckResponse(status="healthy", backend="jaeger", url="http://x")
    )
    fake_backend.list_services = AsyncMock(return_value=["svc-a"])
    # _run_backend_checks probes through the never-cached list_services_uncached.
    fake_backend.list_services_uncached = fake_backend.list_services
    return fake_backend


def _bearer(value: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {value}"}


class TestOperatorTokenVerifier:
    async def test_exact_token_yields_operator_access_token(self) -> None:
        verifier = OperatorTokenVerifier(TEST_TOKEN)

        access = await verifier.verify_token(TEST_TOKEN)

        assert access is not None
        assert access.client_id == "operator"
        assert access.scopes == []

    async def test_token_differing_in_last_character_is_rejected(self) -> None:
        verifier = OperatorTokenVerifier(TEST_TOKEN)

        assert await verifier.verify_token(TEST_TOKEN[:-1] + "X") is None
        assert await verifier.verify_token("") is None

    async def test_comparison_is_constant_time(self) -> None:
        """The whole point of a dedicated verifier over fastmcp's
        StaticTokenVerifier (whose docstring forbids production use) is the
        constant-time compare - pin that hmac.compare_digest is the primitive."""
        verifier = OperatorTokenVerifier(TEST_TOKEN)
        with patch.object(server.hmac, "compare_digest", wraps=hmac.compare_digest) as spy:
            await verifier.verify_token(TEST_TOKEN)

        spy.assert_called_once()


class TestMcpRouteRequiresBearerToken:
    def test_initialize_without_authorization_is_401_with_challenge(self) -> None:
        server.mcp.auth = OperatorTokenVerifier(TEST_TOKEN)

        with _build_app_as_main_does() as client:
            response = post_initialize(client)

        assert response.status_code == 401
        assert response.headers["www-authenticate"].startswith("Bearer")
        assert "mcp-session-id" not in response.headers

    def test_wrong_token_is_401_and_exact_token_opens_a_session(self) -> None:
        server.mcp.auth = OperatorTokenVerifier(TEST_TOKEN)

        with _build_app_as_main_does() as client:
            wrong = post_initialize(client, headers=_bearer(TEST_TOKEN + "x"))
            assert wrong.status_code == 401
            assert "mcp-session-id" not in wrong.headers

            session_id = open_session(client, headers=_bearer(TEST_TOKEN))
            listed = post_request(client, session_id, "tools/list", headers=_bearer(TEST_TOKEN))

        assert listed.status_code == 200
        assert len(sse_json(listed)["result"]["tools"]) == 19

    def test_health_and_ready_stay_reachable_without_a_token(self) -> None:
        """RequireAuthMiddleware wraps only the /mcp route; custom routes are
        appended as plain routes, so docker_healthcheck.py (GET /ready, no
        Authorization header) keeps working with auth configured."""
        server.mcp.auth = OperatorTokenVerifier(TEST_TOKEN)

        with patch.object(server, "_get_backend", AsyncMock(return_value=_healthy_backend())):
            with _build_app_as_main_does() as client:
                health = client.get("/health")
                ready = client.get("/ready")

        assert health.status_code == 200
        assert ready.status_code == 200


@pytest.mark.usefixtures("_clean_env")
class TestMainHttpListenerGate:
    def test_network_bind_without_token_or_opt_out_is_a_usage_error(self) -> None:
        runner = CliRunner()
        with patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls:
            mock_server_cls.return_value.serve = AsyncMock()
            result = runner.invoke(server.main, ["--transport", "http", "--host", ALL_INTERFACES])

        assert result.exit_code == 2, result.output
        assert "MCP_HTTP_BEARER_TOKEN" in result.output
        assert "--allow-unauthenticated-http" in result.output
        mock_server_cls.return_value.serve.assert_not_awaited()

    def test_explicit_opt_out_serves_unauthenticated_on_a_network_bind(self) -> None:
        runner = CliRunner()
        with patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls:
            mock_server_cls.return_value.serve = AsyncMock()
            result = runner.invoke(
                server.main,
                ["--transport", "http", "--host", ALL_INTERFACES, "--allow-unauthenticated-http"],
            )

        assert result.exit_code == 0, result.output
        mock_server_cls.return_value.serve.assert_awaited_once()
        assert server.mcp.auth is None

    def test_plain_http_transport_defaults_to_loopback_and_needs_no_token(self) -> None:
        runner = CliRunner()
        with patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls:
            mock_server_cls.return_value.serve = AsyncMock()
            result = runner.invoke(server.main, ["--transport", "http"])

        assert result.exit_code == 0, result.output
        config_arg = mock_server_cls.call_args.args[0]
        assert config_arg.host == "127.0.0.1"
        mock_server_cls.return_value.serve.assert_awaited_once()
        assert server.mcp.auth is None

    def test_token_from_env_installs_the_verifier_for_a_network_bind(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MCP_HTTP_BEARER_TOKEN", TEST_TOKEN)
        runner = CliRunner()
        with patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls:
            mock_server_cls.return_value.serve = AsyncMock()
            result = runner.invoke(server.main, ["--transport", "http", "--host", ALL_INTERFACES])

        assert result.exit_code == 0, result.output
        mock_server_cls.return_value.serve.assert_awaited_once()
        assert isinstance(server.mcp.auth, OperatorTokenVerifier)

    def test_short_token_is_rejected_with_a_usage_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MCP_HTTP_BEARER_TOKEN", SHORT_VALUE)
        runner = CliRunner()
        with patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls:
            mock_server_cls.return_value.serve = AsyncMock()
            result = runner.invoke(server.main, ["--transport", "http", "--host", ALL_INTERFACES])

        assert result.exit_code == 2, result.output
        assert "32" in result.output
        mock_server_cls.return_value.serve.assert_not_awaited()
        assert server.mcp.auth is None

    def test_token_with_stdio_transport_logs_a_warning_and_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("MCP_HTTP_BEARER_TOKEN", TEST_TOKEN)
        runner = CliRunner()
        with (
            patch.object(server.mcp, "run") as mock_run,
            caplog.at_level("WARNING", logger="opentelemetry_mcp.server"),
        ):
            result = runner.invoke(server.main, ["--transport", "stdio"])

        assert result.exit_code == 0, result.output
        mock_run.assert_called_once_with(transport="stdio")
        assert server.mcp.auth is None
        assert any("MCP_HTTP_BEARER_TOKEN" in r.getMessage() for r in caplog.records)
        assert TEST_TOKEN not in caplog.text

    def test_print_config_reports_token_presence_never_its_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MCP_HTTP_BEARER_TOKEN", TEST_TOKEN)
        runner = CliRunner()
        result = runner.invoke(server.main, ["--print-config", "--transport", "http"])

        assert result.exit_code == 0, result.output
        assert TEST_TOKEN not in result.output
        resolved = json.loads(result.output)
        assert resolved["http_bearer_token_set"] is True
        assert resolved["allow_unauthenticated_http"] is False
        assert resolved["host"] == "127.0.0.1"


def _uvicorn_access_record(target: str, status: int = 401) -> logging.LogRecord:
    """A record shaped exactly like uvicorn 0.40's h11/httptools access line:
    '%s - "%s %s HTTP/%s" %d' with (client_addr, method, target, version, status)."""
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:53061", "POST", target, "1.1", status),
        exc_info=None,
    )


def _rendered_access_record(line: str) -> logging.LogRecord:
    """An access record whose line is already formatted (args consumed), as a
    differently configured access logger may hand it over."""
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=line,
        args=None,
        exc_info=None,
    )


class TestAccessLogQueryStringFilter:
    """uvicorn's access log writes the full request target, so a bearer token
    a client put in the query string (?access_token=...) landed verbatim in
    the server log at the default level even though /mcp answered 401
    (runtime verification prober-1 L01). main() disables the access log;
    this filter is what keeps a re-enabled one credential-free."""

    def test_strips_the_query_string_from_uvicorns_record_shape(self) -> None:
        record = _uvicorn_access_record(f"/mcp?access_token={QUERY_DUMMY}")

        assert AccessLogQueryStringFilter().filter(record) is True

        assert record.args == ("127.0.0.1:53061", "POST", "/mcp", "1.1", 401)
        assert record.getMessage() == '127.0.0.1:53061 - "POST /mcp HTTP/1.1" 401'
        assert QUERY_DUMMY not in record.getMessage()

    def test_uvicorns_own_access_formatter_still_renders_the_record(self) -> None:
        """The args tuple must keep uvicorn's five-field shape: its
        AccessFormatter unpacks record.args positionally."""
        record = _uvicorn_access_record(f"/mcp?token={QUERY_DUMMY}&x=1")
        AccessLogQueryStringFilter().filter(record)

        rendered = AccessFormatter(
            fmt='%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            use_colors=False,
        ).format(record)

        assert '"POST /mcp HTTP/1.1"' in rendered
        assert QUERY_DUMMY not in rendered

    @pytest.mark.parametrize(
        ("target", "kept"),
        [
            # uvicorn quotes the path, so a fragment, an encoded '?' or a matrix parameter's
            # ';' arrives as %23 / %3F / %3B; the raw '#' and ';' rows cover a custom access
            # format that logs the unquoted target.
            ("/health%23access_token%3D{tok}", "/health"),
            ("/mcp%3Faccess_token%3D{tok}", "/mcp"),
            ("/mcp#access_token={tok}", "/mcp"),
            ("/health;access_token={tok}", "/health"),
            # Final verification pass: these spellings were logged verbatim because the pattern
            # knew ';' but not the %3B uvicorn actually renders.
            ("/health%3Baccess_token%3D{tok}", "/health"),
            ("/mcp%3Baccess_token%3D{tok}", "/mcp"),
            ("/mcp%3Bx%3D1%3Baccess_token%3D{tok}", "/mcp"),
            ("/mcp%3baccess_token%3d{tok}", "/mcp"),
            ("/mcp?x=1#access_token={tok}", "/mcp"),
        ],
    )
    def test_cuts_fragments_matrix_parameters_and_encoded_separators_too(
        self, target: str, kept: str
    ) -> None:
        # Delta verification probe P5b: nine access lines carried a token placed in a
        # fragment, a percent-encoded '?', or a matrix parameter because only '?' was cut.
        record = _uvicorn_access_record(target.format(tok=QUERY_DUMMY))
        AccessLogQueryStringFilter().filter(record)

        assert isinstance(record.args, tuple)
        assert record.args[2] == kept
        assert QUERY_DUMMY not in record.getMessage()

    def test_bounds_the_logged_path(self) -> None:
        # A 12 KiB path produced a 12 KiB access line; the 8 KiB drop applied only to
        # pre-rendered records.
        record = _uvicorn_access_record("/health/" + "p" * 12000)
        AccessLogQueryStringFilter().filter(record)

        assert isinstance(record.args, tuple)
        path = record.args[2]
        assert isinstance(path, str)
        assert len(path) <= 2048 + len("...[cut]")
        assert path.endswith("...[cut]")

    def test_leaves_a_target_without_a_query_string_alone(self) -> None:
        record = _uvicorn_access_record("/mcp", status=200)
        AccessLogQueryStringFilter().filter(record)

        assert record.args == ("127.0.0.1:53061", "POST", "/mcp", "1.1", 200)

    def test_strips_the_query_string_from_a_pre_rendered_line(self) -> None:
        """A differently configured access logger may hand over the line
        already formatted (args consumed); the quoted request line is still
        recognisable and still stripped."""
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=f'127.0.0.1:1 - "GET /mcp?bearer={QUERY_DUMMY}&a=b HTTP/1.1" 401',
            args=None,
            exc_info=None,
        )
        AccessLogQueryStringFilter().filter(record)

        assert record.getMessage() == '127.0.0.1:1 - "GET /mcp HTTP/1.1" 401'

    @pytest.mark.parametrize(
        ("target", "kept"),
        [
            ("/mcp%3Baccess_token%3D{tok}", "/mcp"),
            ("/mcp%3Faccess_token%3D{tok}", "/mcp"),
            ("/health%23access_token%3D{tok}", "/health"),
            ("/health;access_token={tok}", "/health"),
            ("/mcp#access_token={tok}", "/mcp"),
        ],
    )
    def test_cuts_every_separator_from_a_pre_rendered_line(self, target: str, kept: str) -> None:
        # The fallback used to cut at '?' only; it now stops at the same separators as the
        # tuple path, and only inside the request target.
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=f'127.0.0.1:1 - "GET {target.format(tok=QUERY_DUMMY)} HTTP/1.1" 404',
            args=None,
            exc_info=None,
        )
        AccessLogQueryStringFilter().filter(record)

        assert record.getMessage() == f'127.0.0.1:1 - "GET {kept} HTTP/1.1" 404'
        assert QUERY_DUMMY not in record.getMessage()

    def test_strips_the_target_query_even_with_tokens_before_the_request_line(self) -> None:
        """A custom access format may put a referer (with its own '?') or a user
        agent (even one that says 'HTTP/1.1') before the request line; the
        TARGET's query string is still the one removed. The first version of the
        linear scan cut only at the first '?' in the line (round-2 review)."""
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=(
                '127.0.0.1:1 - "http://ref.example/?r=1" "evil HTTP/1.1" '
                f'"GET /mcp?token={QUERY_DUMMY} HTTP/1.1" 401'
            ),
            args=None,
            exc_info=None,
        )
        AccessLogQueryStringFilter().filter(record)

        assert QUERY_DUMMY not in record.getMessage()
        assert '"GET /mcp HTTP/1.1" 401' in record.getMessage()

    def test_never_raises_on_an_unformattable_record(self) -> None:
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="%s %s",
            args=("only-one",),
            exc_info=None,
        )

        assert AccessLogQueryStringFilter().filter(record) is True

    def test_a_64_kib_adversarial_pre_rendered_line_is_dropped_in_linear_time(self) -> None:
        """Round-2 review of this filter: the regex fallback was quadratic -
        16 KiB of "?" with no " HTTP/" took 2.2 s, in the logging caller.
        The fallback is a linear scan, and a line longer than 8 KiB is not a
        request line any HTTP parser accepted, so it is dropped outright and
        nothing in it - this dummy included - can reach a handler."""
        record = _rendered_access_record("?" * (64 * 1024) + f"?token={QUERY_DUMMY}")

        start = time.perf_counter()
        kept = AccessLogQueryStringFilter().filter(record)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.05, elapsed
        assert kept is False

    def test_a_pre_rendered_line_over_the_bound_is_dropped_with_its_query_string(self) -> None:
        line = f'127.0.0.1:1 - "GET /mcp?token={QUERY_DUMMY}&pad={"a" * 9000} HTTP/1.1" 401'
        record = _rendered_access_record(line)

        assert AccessLogQueryStringFilter().filter(record) is False

    def test_a_pre_rendered_line_at_the_bound_is_still_rewritten(self) -> None:
        prefix = f'127.0.0.1:1 - "GET /mcp?token={QUERY_DUMMY}&pad='
        suffix = ' HTTP/1.1" 401'
        line = prefix + "a" * (8 * 1024 - len(prefix) - len(suffix)) + suffix
        assert len(line) == 8 * 1024
        record = _rendered_access_record(line)

        assert AccessLogQueryStringFilter().filter(record) is True
        assert record.getMessage() == '127.0.0.1:1 - "GET /mcp HTTP/1.1" 401'

    def test_many_question_marks_under_the_bound_are_scanned_once(self) -> None:
        """8 KiB of "?" with no " HTTP/" after it: nothing to cut, kept as is,
        and still fast - the old regex re-scanned the line from every "?"."""
        record = _rendered_access_record("?" * (8 * 1024))

        start = time.perf_counter()
        kept = AccessLogQueryStringFilter().filter(record)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.05, elapsed
        assert kept is True
        assert record.getMessage() == "?" * (8 * 1024)

    @pytest.mark.parametrize(
        "line",
        [
            '127.0.0.1:1 - "GET /mcp HTTP/1.1" 200',  # no "?" at all
            '127.0.0.1:1 - "GET /mcp HTTP/1.1" 401 ?trailing',  # no whitespace after the "?"
            '127.0.0.1:1 - "GET /mcp?x=1 y" 401',  # whitespace, but no "HTTP/" after it
            "127.0.0.1:1 - ?x=1   ",  # whitespace to the end of the line
        ],
        ids=["no-query", "no-whitespace", "no-http-after", "whitespace-to-end"],
    )
    def test_a_question_mark_outside_the_request_line_is_left_alone(self, line: str) -> None:
        record = _rendered_access_record(line)

        assert AccessLogQueryStringFilter().filter(record) is True
        assert record.getMessage() == line


class TestUvicornConfigAccessLog:
    def test_access_log_is_off_and_the_filter_guards_a_re_enabled_one(self) -> None:
        config = _build_uvicorn_config(
            server.mcp.http_app(transport="streamable-http"),
            host="127.0.0.1",
            port=8000,
            graceful_shutdown_timeout_seconds=2,
            forwarded_allow_ips=None,
        )

        assert config.access_log is False
        access_logger = logging.getLogger("uvicorn.access")
        assert access_logger.handlers == []
        assert any(isinstance(f, AccessLogQueryStringFilter) for f in access_logger.filters)


@contextmanager
def _served(app: Any) -> Iterator[str]:
    """Run `app` under a real uvicorn.Server built exactly as main() builds
    it, on an OS-assigned loopback port, in a background thread."""
    config = _build_uvicorn_config(
        app,
        host="127.0.0.1",
        port=0,
        graceful_shutdown_timeout_seconds=1,
        forwarded_allow_ips=None,
    )
    uvicorn_server = uvicorn.Server(config)
    thread = threading.Thread(target=lambda: asyncio.run(uvicorn_server.serve()), daemon=True)
    thread.start()
    deadline = time.monotonic() + 15.0
    while not uvicorn_server.started:
        assert thread.is_alive(), "uvicorn exited before it started serving"
        assert time.monotonic() < deadline, "uvicorn did not start within 15 s"
        time.sleep(0.01)
    port = uvicorn_server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=15.0)


class TestQueryStringTokenNeverReachesTheServerLog:
    """Prober-1 L01 / judge 'BROKE CONTROL': the served process answered
    POST /mcp?access_token=<token> with 401 and a clean body, but uvicorn's
    default access log (INFO, enabled by default, never configured by
    server.py) wrote the whole request target:
    'INFO: 127.0.0.1:53061 - "POST /mcp?access_token=<token> HTTP/1.1" 401'.
    Driven through a real uvicorn.Server because starlette's TestClient
    never reaches uvicorn's protocol layer where that line is written."""

    @pytest.mark.parametrize("name", ["access_token", "token", "bearer", "authorization"])
    def test_query_string_token_is_refused_and_absent_from_every_log_sink(
        self, name: str, caplog: pytest.LogCaptureFixture, capfd: pytest.CaptureFixture[str]
    ) -> None:
        server.mcp.auth = OperatorTokenVerifier(TEST_TOKEN)
        app = server.mcp.http_app(transport="streamable-http")
        re_enabled = io.StringIO()
        re_enabled_handler = logging.StreamHandler(re_enabled)
        access_logger = logging.getLogger("uvicorn.access")

        with caplog.at_level(logging.DEBUG), _served(app) as base_url:
            body = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
            headers = {"Accept": "application/json, text/event-stream"}
            default = httpx.post(
                f"{base_url}/mcp", params={name: QUERY_DUMMY}, json=body, headers=headers
            )
            # An operator who re-enables the access log (any handler on
            # uvicorn.access turns it back on per connection) must still
            # never see the query string.
            access_logger.addHandler(re_enabled_handler)
            try:
                re_enabled_response = httpx.post(
                    f"{base_url}/mcp", params={name: QUERY_DUMMY}, json=body, headers=headers
                )
            finally:
                access_logger.removeHandler(re_enabled_handler)

        for response in (default, re_enabled_response):
            assert response.status_code == 401
            assert response.headers["www-authenticate"].startswith("Bearer")
            assert QUERY_DUMMY not in response.text

        out, err = capfd.readouterr()
        assert QUERY_DUMMY not in out
        assert QUERY_DUMMY not in err
        assert QUERY_DUMMY not in caplog.text
        re_enabled_lines = re_enabled.getvalue()
        assert '"POST /mcp HTTP/1.1" 401' in re_enabled_lines
        assert QUERY_DUMMY not in re_enabled_lines
        assert "?" not in re_enabled_lines


# Values that pass the 32-character floor but that no HTTP client can present
# byte-exactly in an Authorization header (h11 trims OWS; header bytes are
# latin-1), so accepting them locked the deployment out silently.
ALL_SPACES = " " * 32
NON_ASCII = "ü" * 32
INNER_WHITESPACE = TEST_TOKEN[:20] + "\t" + TEST_TOKEN[21:]
# Exactly 48 printable-ASCII characters, the shape the probes used.
FORTY_EIGHT = "-".join(["forty", "eight", "char", "ascii", "operator", "value", "0123456789"])


def _invoke_http(args: list[str]) -> tuple[Any, Any]:
    runner = CliRunner()
    with patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls:
        mock_server_cls.return_value.serve = AsyncMock()
        result = runner.invoke(server.main, ["--transport", "http", *args])
    return result, mock_server_cls.return_value.serve


@pytest.mark.usefixtures("_clean_env")
class TestEmptyBearerTokenEnvIsAUsageError:
    """Prober-1 F2: click resolves an environment variable whose value is
    empty as unset, so MCP_HTTP_BEARER_TOKEN= (a blank line in a compose
    file or .env) silently started a loopback listener with NO auth and
    only the generic WARNING, while --http-bearer-token '' exited 2 with
    '(got 0)'. Both spellings must take the same path."""

    def test_empty_env_var_on_loopback_exits_2_naming_the_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MCP_HTTP_BEARER_TOKEN", "")

        result, serve = _invoke_http([])

        assert result.exit_code == 2, result.output
        assert "MCP_HTTP_BEARER_TOKEN" in result.output
        assert "(got 0)" in result.output
        serve.assert_not_awaited()
        assert server.mcp.auth is None

    def test_empty_env_var_on_a_network_bind_gets_the_same_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MCP_HTTP_BEARER_TOKEN", "")

        result, serve = _invoke_http(["--host", ALL_INTERFACES])

        assert result.exit_code == 2, result.output
        assert "(got 0)" in result.output
        serve.assert_not_awaited()

    def test_empty_flag_is_unchanged_control(self) -> None:
        result, serve = _invoke_http(["--http-bearer-token", ""])

        assert result.exit_code == 2, result.output
        assert "(got 0)" in result.output
        serve.assert_not_awaited()

    def test_unset_env_var_still_serves_loopback_unauthenticated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Positive control: absent (not empty) keeps the documented
        loopback default."""
        monkeypatch.delenv("MCP_HTTP_BEARER_TOKEN", raising=False)

        result, serve = _invoke_http([])

        assert result.exit_code == 0, result.output
        serve.assert_awaited_once()
        assert server.mcp.auth is None


@pytest.mark.usefixtures("_clean_env")
class TestBearerTokenMustBePrintableAscii:
    """Prober-1 F4: 32 spaces and 32 non-ASCII code points passed the
    length check, auth was 'enabled', and no client could ever present the
    value - a silent lockout. Such a value is a usage error, and the
    message never echoes it."""

    @pytest.mark.parametrize("value", [ALL_SPACES, NON_ASCII, INNER_WHITESPACE])
    def test_env_var_value_is_refused_without_being_echoed(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("MCP_HTTP_BEARER_TOKEN", value)

        result, serve = _invoke_http(["--host", ALL_INTERFACES])

        assert result.exit_code == 2, result.output
        assert "MCP_HTTP_BEARER_TOKEN" in result.output
        assert "printable ASCII" in result.output
        assert value not in result.output
        serve.assert_not_awaited()
        assert server.mcp.auth is None

    @pytest.mark.parametrize("value", [ALL_SPACES, NON_ASCII, INNER_WHITESPACE])
    def test_flag_value_is_refused_without_being_echoed(self, value: str) -> None:
        result, serve = _invoke_http(["--host", ALL_INTERFACES, "--http-bearer-token", value])

        assert result.exit_code == 2, result.output
        assert "printable ASCII" in result.output
        assert value not in result.output
        serve.assert_not_awaited()

    def test_48_char_ascii_value_still_starts_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert len(FORTY_EIGHT) == 48
        monkeypatch.setenv("MCP_HTTP_BEARER_TOKEN", FORTY_EIGHT)

        result, serve = _invoke_http(["--host", ALL_INTERFACES])

        assert result.exit_code == 0, result.output
        serve.assert_awaited_once()
        assert isinstance(server.mcp.auth, OperatorTokenVerifier)

    def test_48_char_ascii_value_still_starts_from_flag(self) -> None:
        result, serve = _invoke_http(["--host", ALL_INTERFACES, "--http-bearer-token", FORTY_EIGHT])

        assert result.exit_code == 0, result.output
        serve.assert_awaited_once()

    def test_every_token68_character_is_accepted(self) -> None:
        """The documented recipe (`openssl rand -hex 32`) yields 64 hex
        characters; every RFC 6750 token68 character is printable ASCII."""
        for value in ("0f" * 32, "abc-._~+/" * 4 + "=="):
            server._validate_http_listener_auth(
                host=ALL_INTERFACES, http_bearer_token=value, allow_unauthenticated_http=False
            )

    @pytest.mark.parametrize("value", [ALL_SPACES, NON_ASCII, INNER_WHITESPACE, "x" * 31 + "\n"])
    def test_validator_refuses_the_value_with_a_static_message(self, value: str) -> None:
        with pytest.raises(click.UsageError) as exc_info:
            server._validate_http_listener_auth(
                host="127.0.0.1", http_bearer_token=value, allow_unauthenticated_http=False
            )

        assert "printable ASCII" in str(exc_info.value)
        assert value not in str(exc_info.value)

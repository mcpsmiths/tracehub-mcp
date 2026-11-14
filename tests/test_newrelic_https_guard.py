"""Pin the New Relic backend's https-only credential guard.

NerdGraph authenticates with a non-standard ``API-Key`` header. Sending it
over plain http would put the key on the wire in cleartext, so
``NewRelicBackend.__init__`` refuses any URL that is not ``https://`` before
a client (and therefore any request) can exist. Each of the other
header-keyed backends (Datadog, Sentry, Honeycomb) already pins its own
guard; this module is the New Relic equivalent.

A credential embedded in the URL itself (``user:password@host``) is refused
one layer earlier, at ``BackendConfig``; the second class below pins that
for ``type="newrelic"`` so the two refusals together cover the cleartext
credential surface.
"""

import httpx
import pytest
from pydantic import HttpUrl, ValidationError

from opentelemetry_mcp import server
from opentelemetry_mcp.backends.newrelic import NewRelicBackend
from opentelemetry_mcp.config import BackendConfig, ServerConfig

FAKE_USER_KEY = "fake-newrelic-userkey-1"
FAKE_ACCOUNT_ID = "1234567"
# Dummy marker standing in for a password; nothing real.
VFY_MARKER = "VFY-NR-4d2a"

_HTTPS_ERROR = "requires an https:// URL"


def _build(url: str) -> NewRelicBackend:
    return NewRelicBackend(url=url, api_key=FAKE_USER_KEY, account_id=FAKE_ACCOUNT_ID)


class TestHttpsOnlyGuard:
    @pytest.mark.parametrize(
        "url",
        [
            "http://api.newrelic.com/graphql",
            "http://api.eu.newrelic.com/graphql",
            "http://localhost:8080/graphql",
            "ftp://api.newrelic.com/graphql",
            "api.newrelic.com/graphql",
            "//api.newrelic.com/graphql",
            "",
        ],
        ids=["http", "http-eu", "http-localhost", "ftp", "no-scheme", "scheme-relative", "empty"],
    )
    def test_non_https_url_is_refused_with_the_documented_error(self, url: str) -> None:
        with pytest.raises(ValueError, match=_HTTPS_ERROR) as exc_info:
            _build(url)

        assert "must not be sent over plain http" in str(exc_info.value)

    def test_refusal_never_echoes_the_api_key(self) -> None:
        with pytest.raises(ValueError, match=_HTTPS_ERROR) as exc_info:
            _build("http://api.newrelic.com/graphql")

        assert FAKE_USER_KEY not in str(exc_info.value)

    def test_refused_url_issues_no_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sent: list[httpx.Request] = []

        async def _record_send(
            self: httpx.AsyncClient, request: httpx.Request, **kwargs: object
        ) -> httpx.Response:
            sent.append(request)
            return httpx.Response(200, json={})

        monkeypatch.setattr(httpx.AsyncClient, "send", _record_send)

        with pytest.raises(ValueError, match=_HTTPS_ERROR):
            _build("http://api.newrelic.com/graphql")

        assert sent == []

    @pytest.mark.parametrize(
        "url",
        ["https://api.newrelic.com/graphql", "https://api.eu.newrelic.com/graphql"],
        ids=["us", "eu"],
    )
    def test_https_url_is_accepted_and_sends_the_key_only_to_that_origin(self, url: str) -> None:
        backend = _build(url)

        assert backend.url == url
        client = backend.client
        assert client.base_url.scheme == "https"
        assert client.headers["API-Key"] == FAKE_USER_KEY

    def test_guard_is_not_satisfied_by_https_appearing_later_in_the_url(self) -> None:
        # A prefix check, not a substring check: an http URL whose query or
        # path merely mentions https:// must still be refused.
        with pytest.raises(ValueError, match=_HTTPS_ERROR):
            _build("http://api.newrelic.com/graphql?next=https://api.newrelic.com/graphql")


class TestServerPathRefusesCleartextCredentials:
    """The same refusals as seen by an operator: BACKEND_URL goes through
    BackendConfig and then server._create_backend."""

    def test_create_backend_refuses_an_http_backend_url(self) -> None:
        config = ServerConfig(
            backend=BackendConfig(
                type="newrelic",
                url=HttpUrl("http://api.newrelic.com/graphql"),
                api_key=FAKE_USER_KEY,
                newrelic_account_id=FAKE_ACCOUNT_ID,
            )
        )

        with pytest.raises(ValueError, match=_HTTPS_ERROR):
            server._create_backend(config)

    def test_create_backend_accepts_an_https_backend_url(self) -> None:
        config = ServerConfig(
            backend=BackendConfig(
                type="newrelic",
                url=HttpUrl("https://api.newrelic.com/graphql"),
                api_key=FAKE_USER_KEY,
                newrelic_account_id=FAKE_ACCOUNT_ID,
            )
        )

        backend = server._create_backend(config)

        assert isinstance(backend, NewRelicBackend)
        assert backend.url.startswith("https://")

    def test_url_with_embedded_credentials_is_refused_without_echoing_the_password(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            BackendConfig(
                type="newrelic",
                url=HttpUrl(f"https://u:{VFY_MARKER}@api.newrelic.com/graphql"),
                api_key=FAKE_USER_KEY,
                newrelic_account_id=FAKE_ACCOUNT_ID,
            )

        (error,) = exc_info.value.errors()
        assert "must not embed credentials" in error["msg"]
        assert VFY_MARKER not in error["msg"]

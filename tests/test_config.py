"""Tests for BackendConfig URL validation."""

import logging

import pytest
from pydantic import HttpUrl

from opentelemetry_mcp.config import BackendConfig


def test_non_local_http_url_logs_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """A plain http:// URL to a non-local host is a real interception risk
    (CVE-2025-6514) and must be warned about, but not blocked - some
    deployments (private VPC, Docker Compose internal network) genuinely
    have no TLS available."""
    with caplog.at_level(logging.WARNING):
        config = BackendConfig(type="jaeger", url=HttpUrl("http://example.com:16686"))

    assert config.url.scheme == "http"
    assert any(
        "http" in record.message.lower() and "example.com" in record.message
        for record in caplog.records
    )


def test_localhost_http_url_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    """localhost has no real MITM risk, so it must stay silent - this is
    also the documented default (BACKEND_URL=http://localhost:16686)."""
    with caplog.at_level(logging.WARNING):
        BackendConfig(type="jaeger", url=HttpUrl("http://localhost:16686"))

    assert caplog.records == []


def test_loopback_ip_http_url_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    """127.0.0.1 is equivalent to localhost for this purpose."""
    with caplog.at_level(logging.WARNING):
        BackendConfig(type="jaeger", url=HttpUrl("http://127.0.0.1:16686"))

    assert caplog.records == []


def test_https_url_never_warns(caplog: pytest.LogCaptureFixture) -> None:
    """https:// is always safe, local or not."""
    with caplog.at_level(logging.WARNING):
        BackendConfig(type="jaeger", url=HttpUrl("https://example.com:16686"))
        BackendConfig(type="jaeger", url=HttpUrl("https://localhost:16686"))

    assert caplog.records == []

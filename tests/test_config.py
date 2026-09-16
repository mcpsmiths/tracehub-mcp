"""Tests for BackendConfig URL validation."""

import logging

import pytest
from pydantic import HttpUrl, ValidationError

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


def test_other_127_x_x_x_loopback_address_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    """The entire 127.0.0.0/8 range is loopback, not just 127.0.0.1."""
    with caplog.at_level(logging.WARNING):
        BackendConfig(type="jaeger", url=HttpUrl("http://127.0.0.5:16686"))

    assert caplog.records == []


def test_aws_imds_ip_is_rejected() -> None:
    """169.254.169.254 (AWS/Azure/GCP/DigitalOcean/Oracle instance metadata)
    is never a legitimate trace backend and must be hard-blocked."""
    with pytest.raises(ValidationError, match="metadata"):
        BackendConfig(type="jaeger", url=HttpUrl("http://169.254.169.254/latest/meta-data/"))


def test_aws_ecs_task_metadata_ip_is_rejected() -> None:
    with pytest.raises(ValidationError, match="metadata"):
        BackendConfig(type="jaeger", url=HttpUrl("http://169.254.170.2/v2/metadata"))


def test_aws_imds_ipv6_is_rejected() -> None:
    with pytest.raises(ValidationError, match="metadata"):
        BackendConfig(type="jaeger", url=HttpUrl("http://[fd00:ec2::254]/latest/meta-data/"))


def test_alibaba_metadata_ip_is_rejected() -> None:
    with pytest.raises(ValidationError, match="metadata"):
        BackendConfig(type="jaeger", url=HttpUrl("http://100.100.100.200/latest/meta-data/"))


def test_gcp_metadata_hostname_is_rejected() -> None:
    with pytest.raises(ValidationError, match="metadata"):
        BackendConfig(
            type="jaeger", url=HttpUrl("http://metadata.google.internal/computeMetadata/v1/")
        )


def test_ipv4_mapped_ipv6_metadata_is_rejected() -> None:
    with pytest.raises(ValidationError, match="metadata"):
        BackendConfig(type="jaeger", url=HttpUrl("http://[::ffff:169.254.169.254]/"))


def test_private_ip_self_hosted_backend_is_still_allowed() -> None:
    """A private IP is a completely normal self-hosted deployment (VPC,
    Docker Compose network) and must never be blocked - only cloud
    metadata addresses are always illegitimate."""
    config = BackendConfig(type="jaeger", url=HttpUrl("http://10.0.0.5:16686"))
    assert config.url.host == "10.0.0.5"


def test_https_url_never_warns(caplog: pytest.LogCaptureFixture) -> None:
    """https:// is always safe, local or not."""
    with caplog.at_level(logging.WARNING):
        BackendConfig(type="jaeger", url=HttpUrl("https://example.com:16686"))
        BackendConfig(type="jaeger", url=HttpUrl("https://localhost:16686"))

    assert caplog.records == []

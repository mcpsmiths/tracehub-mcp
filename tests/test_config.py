"""Tests for BackendConfig URL validation."""

import logging

import pytest
from pydantic import HttpUrl, ValidationError

from opentelemetry_mcp.backends.base import DEFAULT_MAX_RESPONSE_BYTES
from opentelemetry_mcp.config import BackendConfig, ServerConfig


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


def test_azure_wireserver_ip_is_rejected() -> None:
    """168.63.129.16 (Azure WireServer) is a second metadata-class address
    on every Azure VM, distinct from 169.254.169.254."""
    with pytest.raises(ValidationError, match="metadata"):
        BackendConfig(type="jaeger", url=HttpUrl("http://168.63.129.16/machine?comp=goalstate"))


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


class TestMaxResponseBytes:
    """BACKEND_MAX_RESPONSE_BYTES / SECONDARY_BACKEND_MAX_RESPONSE_BYTES bound
    the DECODED size of any single backend response (enforced in
    backends.base._RetryingTransport); parsed like BACKEND_TIMEOUT."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BACKEND_TYPE", "jaeger")
        monkeypatch.setenv("BACKEND_URL", "http://localhost:16686")
        monkeypatch.delenv("BACKEND_MAX_RESPONSE_BYTES", raising=False)
        monkeypatch.setenv("SECONDARY_BACKEND_TYPE", "jaeger")
        monkeypatch.setenv("SECONDARY_BACKEND_URL", "http://localhost:16687")
        monkeypatch.delenv("SECONDARY_BACKEND_MAX_RESPONSE_BYTES", raising=False)

    def test_default_is_the_transport_default(self) -> None:
        assert BackendConfig.from_env().max_response_bytes == DEFAULT_MAX_RESPONSE_BYTES
        assert DEFAULT_MAX_RESPONSE_BYTES == 64 * 1024 * 1024

    def test_env_override_is_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BACKEND_MAX_RESPONSE_BYTES", "1048576")

        assert BackendConfig.from_env().max_response_bytes == 1048576

    def test_secondary_backend_reads_its_own_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BACKEND_MAX_RESPONSE_BYTES", "1048576")
        monkeypatch.setenv("SECONDARY_BACKEND_MAX_RESPONSE_BYTES", "2097152")

        secondary = BackendConfig.from_env_optional()

        assert secondary is not None
        assert secondary.max_response_bytes == 2097152

    def test_unparsable_value_warns_and_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("BACKEND_MAX_RESPONSE_BYTES", "lots")

        with caplog.at_level(logging.WARNING):
            config = BackendConfig.from_env()

        assert config.max_response_bytes == DEFAULT_MAX_RESPONSE_BYTES
        assert any(
            "BACKEND_MAX_RESPONSE_BYTES" in r.message and "lots" in r.message
            for r in caplog.records
        )

    @pytest.mark.parametrize("value", [0, -1, 1024**3 + 1])
    def test_out_of_range_value_is_rejected(self, value: int) -> None:
        with pytest.raises(ValidationError, match="max_response_bytes"):
            BackendConfig(
                type="jaeger", url=HttpUrl("http://localhost:16686"), max_response_bytes=value
            )


# Dummy markers standing in for a credential; nothing real.
VFY_MARKER = "VFY-PW-7c1e"
VFY_MARKER_A = "VFY-A1-9f3b"
VFY_MARKER_B = "VFY-B2-9f3b"


class TestUserinfoInBackendUrl:
    """A credential embedded in BACKEND_URL (user:password@host) is rejected
    at the config boundary: str(url) is logged, printed by --print-config
    and doctor, and echoed by httpx error text into /ready and tool errors,
    so the only safe place for a backend credential is a dedicated field."""

    def test_user_and_password_are_rejected_without_echoing_the_password(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            BackendConfig(type="jaeger", url=HttpUrl(f"https://u:{VFY_MARKER}@h.test"))

        (error,) = exc_info.value.errors()
        assert "BACKEND_URL" in error["msg"]
        assert "BACKEND_API_KEY" in error["msg"]
        assert "BACKEND_TEMPO_INSTANCE_ID" in error["msg"]
        assert VFY_MARKER not in error["msg"]

    def test_username_only_userinfo_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not embed credentials"):
            BackendConfig(type="jaeger", url=HttpUrl("https://svc-account@h.test"))

    def test_apply_cli_overrides_rejects_userinfo_without_echoing_the_password(self) -> None:
        config = ServerConfig(backend=BackendConfig(type="jaeger", url=HttpUrl("http://localhost")))

        with pytest.raises(ValueError, match="must not embed credentials") as exc_info:
            config.apply_cli_overrides(backend_url=f"https://u:{VFY_MARKER}@h.test")

        assert VFY_MARKER not in str(exc_info.value)
        assert str(config.backend.url) == "http://localhost/"

    def test_secondary_backend_url_with_userinfo_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SECONDARY_BACKEND_TYPE", "jaeger")
        monkeypatch.setenv("SECONDARY_BACKEND_URL", f"https://u:{VFY_MARKER}@h.test")

        with pytest.raises(ValidationError, match="must not embed credentials"):
            BackendConfig.from_env_optional()

    def test_rejection_logs_nothing_containing_the_password(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A userinfo URL is refused before the plain-HTTP warning or the
        metadata check could stringify it."""
        with caplog.at_level(logging.DEBUG), pytest.raises(ValidationError):
            BackendConfig(type="jaeger", url=HttpUrl(f"http://u:{VFY_MARKER}@169.254.169.254/"))

        assert VFY_MARKER not in caplog.text

    def test_metadata_rejection_names_the_host_not_the_whole_url(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            BackendConfig(type="jaeger", url=HttpUrl("http://169.254.169.254/latest/meta-data/"))

        (error,) = exc_info.value.errors()
        assert "169.254.169.254" in error["msg"]
        assert "/latest/meta-data/" not in error["msg"]

    def test_plain_http_warning_names_the_host_not_the_whole_url(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            BackendConfig(type="jaeger", url=HttpUrl("http://example.com:16686/some/path?k=v"))

        (record,) = [r for r in caplog.records if "plain HTTP" in r.message]
        assert "example.com" in record.message
        assert "/some/path" not in record.message
        assert "k=v" not in record.message

    def test_credential_fields_are_hidden_from_repr(self) -> None:
        config = BackendConfig(
            type="datadog",
            url=HttpUrl("https://api.datadoghq.com"),
            api_key=VFY_MARKER_A,
            app_key=VFY_MARKER_B,
        )

        assert VFY_MARKER_A not in repr(config)
        assert VFY_MARKER_B not in repr(config)
        assert VFY_MARKER_A not in str(config)

"""Tests for BackendConfig URL validation."""

import logging

import pytest
from pydantic import HttpUrl, ValidationError

from opentelemetry_mcp.backends.base import DEFAULT_MAX_RESPONSE_BYTES
from opentelemetry_mcp.config import BackendConfig, ServerConfig, _reject_url_userinfo


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

        # A plain ValueError: the env path refuses the raw string before
        # pydantic can echo it (see TestUserinfoIsRefusedBeforePydanticParsesTheUrl).
        with pytest.raises(ValueError, match="must not embed credentials") as exc_info:
            BackendConfig.from_env_optional()

        assert VFY_MARKER not in str(exc_info.value)

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
        assert record.message.startswith(
            "BACKEND_URL host 'example.com' is reached over plain HTTP"
        )
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


# Fragments only - no contiguous credential-shaped literal in this file.
LONG_USER = "judgeuser"
LONG_PASSWORD = "".join(["Qz7Vk", "Mx9Rt", "LpWnY", "bHdFg", "Js"])


def _assert_credential_free(text: str) -> None:
    assert LONG_USER not in text
    for start in range(len(LONG_PASSWORD) - 3):
        assert LONG_PASSWORD[start : start + 4] not in text, LONG_PASSWORD[start : start + 4]


class TestUserinfoIsRefusedBeforePydanticParsesTheUrl:
    """pydantic-core echoes a ValidationError's input_value truncated to
    24 + "..." + 23 characters, which for a credentialed URL over ~50 chars
    removes the "@" the textual redaction needs and leaves the password
    prefix in the one message guaranteed to be printed. The raw env/CLI
    value is therefore checked with urllib.parse.urlsplit first and refused
    with a plain ValueError that quotes only the redacted URL; a credential
    never becomes a ValidationError input at all."""

    @pytest.mark.parametrize(
        "tail",
        [
            "tempo.internal.example.com/tempo",
            "127.0.0.1:18199/a/longer/base/path",
            "127.0.0.1:18164/?x=1#frag",
            "[2001:db8::1]:3200/tempo/api/search?limit=20#top",
        ],
    )
    def test_backend_url_env_raises_a_plain_value_error_with_the_redacted_url(
        self, monkeypatch: pytest.MonkeyPatch, tail: str
    ) -> None:
        monkeypatch.setenv("BACKEND_TYPE", "jaeger")
        monkeypatch.setenv("BACKEND_URL", f"https://{LONG_USER}:{LONG_PASSWORD}@{tail}")

        with pytest.raises(ValueError, match="must not embed credentials") as exc_info:
            BackendConfig.from_env()

        assert not isinstance(exc_info.value, ValidationError)
        text = str(exc_info.value)
        assert text.startswith("BACKEND_URL ")
        assert f"https://[REDACTED]@{tail}" in text
        _assert_credential_free(text)

    def test_secondary_backend_url_env_names_its_own_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SECONDARY_BACKEND_TYPE", "jaeger")
        monkeypatch.setenv(
            "SECONDARY_BACKEND_URL",
            f"https://{LONG_USER}:{LONG_PASSWORD}@observability.corp.example.com:443/tempo",
        )

        with pytest.raises(ValueError, match="must not embed credentials") as exc_info:
            BackendConfig.from_env_optional()

        assert not isinstance(exc_info.value, ValidationError)
        assert str(exc_info.value).startswith("SECONDARY_BACKEND_URL ")
        _assert_credential_free(str(exc_info.value))

    def test_cli_url_override_refuses_before_the_type_adapter_can_echo_it(self) -> None:
        """A credentialed URL pydantic cannot even parse used to surface as
        the TypeAdapter's ValidationError, input_value included."""
        config = ServerConfig(backend=BackendConfig(type="jaeger", url=HttpUrl("http://localhost")))
        malformed = f"https://{LONG_USER}:{LONG_PASSWORD}@h.test:notaport/very/long/path/here"

        with pytest.raises(ValueError, match="must not embed credentials") as exc_info:
            config.apply_cli_overrides(backend_url=malformed)

        assert not isinstance(exc_info.value, ValidationError)
        _assert_credential_free(str(exc_info.value))
        assert str(config.backend.url) == "http://localhost/"

    def test_malformed_ipv6_authority_with_userinfo_is_still_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """urlsplit raises for an unbalanced bracket; the textual shape
        check must still keep the raw value away from pydantic."""
        monkeypatch.setenv("BACKEND_TYPE", "jaeger")
        monkeypatch.setenv("BACKEND_URL", f"https://{LONG_USER}:{LONG_PASSWORD}@[2001:db8::1/x")

        with pytest.raises(ValueError, match="must not embed credentials") as exc_info:
            BackendConfig.from_env()

        assert not isinstance(exc_info.value, ValidationError)
        _assert_credential_free(str(exc_info.value))

    def test_password_with_a_literal_at_sign_is_fully_redacted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BACKEND_TYPE", "jaeger")
        monkeypatch.setenv(
            "BACKEND_URL", f"https://{LONG_USER}:p@{LONG_PASSWORD}@tempo.internal.example.com/x"
        )

        with pytest.raises(ValueError, match="must not embed credentials") as exc_info:
            BackendConfig.from_env()

        _assert_credential_free(str(exc_info.value))
        assert "p@" not in str(exc_info.value)

    def test_credential_free_urls_still_reach_pydantic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Positive controls: a normal URL loads, and a malformed one
        without userinfo still gets pydantic's own diagnosis."""
        monkeypatch.setenv("BACKEND_TYPE", "jaeger")
        monkeypatch.setenv("BACKEND_URL", "https://tempo.internal.example.com:3200/tempo")
        assert str(BackendConfig.from_env().url) == "https://tempo.internal.example.com:3200/tempo"

        monkeypatch.setenv("BACKEND_URL", "not a url at all")
        with pytest.raises(ValidationError):
            BackendConfig.from_env()


def test_secondary_backend_field_errors_name_the_secondary_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Delta verification (TA, I_extras): with only SECONDARY_BACKEND_TIMEOUT=-5 set the error
    # read "BackendConfig: BACKEND_TIMEOUT: ..." - the PRIMARY backend's variable, because the
    # secondary BackendConfig is validated standalone and pydantic titles it "BackendConfig".
    for var in ("BACKEND_TYPE", "BACKEND_URL", "SECONDARY_BACKEND_TYPE", "SECONDARY_BACKEND_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("BACKEND_TYPE", "jaeger")
    monkeypatch.setenv("BACKEND_URL", "http://127.0.0.1:18299")
    monkeypatch.setenv("SECONDARY_BACKEND_TYPE", "jaeger")
    monkeypatch.setenv("SECONDARY_BACKEND_URL", "http://127.0.0.1:18298")
    monkeypatch.setenv("SECONDARY_BACKEND_TIMEOUT", "-5")

    with pytest.raises(ValueError, match="SECONDARY_BACKEND_TIMEOUT") as excinfo:
        ServerConfig.from_env()

    text = str(excinfo.value)
    assert "SECONDARY_BACKEND_TIMEOUT: Input should be greater than 0" in text
    assert "(--" not in text  # the secondary backend has env vars only, no flags
    assert "BackendConfig" not in text and "input_value" not in text


@pytest.mark.parametrize(
    "raw",
    [
        "http://dummyuser:Zq7v Xk9pLm2tR@127.0.0.1:18299/a/longer/base/path",
        "http://dummyuser: Zq7vXk9pLm2tR@127.0.0.1:18299/a/longer/base/path",
        "http://dummyuser:Zq7vXk9pLm2tR@127.0.0.1:18299/a/longer/base/path?x=1#f",
    ],
)
def test_reject_url_userinfo_never_echoes_the_credential(raw: str) -> None:
    # Delta verification probe P1b: a password containing whitespace stopped the textual
    # redactor (its userinfo run is whitespace-free) and the FULL credential was echoed on
    # every surface. The shown URL is now rebuilt from the parsed parts.
    with pytest.raises(ValueError, match="BACKEND_URL must not embed credentials") as excinfo:
        _reject_url_userinfo(raw, source="BACKEND_URL")
    text = str(excinfo.value)
    for secret in ("dummyuser", "Zq7v", "Xk9pLm2tR"):
        assert secret not in text
    assert "[REDACTED]@127.0.0.1:18299/a/longer/base/path" in text

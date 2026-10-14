"""Tests for ServerConfig's from_env parsing and apply_cli_overrides logic
(log_level, max_traces_per_query, slow_request_threshold_ms).

Phase 4 of the post-launch plan: these three fields previously only had
env-var support (log_level, max_traces_per_query) or did not exist yet
(slow_request_threshold_ms) - apply_cli_overrides did not touch
ServerConfig-level fields at all, only BackendConfig ones.
"""

import logging

import pytest
from pydantic import HttpUrl, ValidationError

from opentelemetry_mcp.config import BackendConfig, ServerConfig


def _server_config() -> ServerConfig:
    return ServerConfig(backend=BackendConfig(type="jaeger", url=HttpUrl("http://localhost:16686")))


class TestFromEnvSlowRequestThreshold:
    def test_unset_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SLOW_REQUEST_THRESHOLD_MS", raising=False)

        config = ServerConfig.from_env()

        assert config.slow_request_threshold_ms is None

    def test_parses_a_valid_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLOW_REQUEST_THRESHOLD_MS", "500")

        config = ServerConfig.from_env()

        assert config.slow_request_threshold_ms == 500.0

    def test_invalid_value_disables_rather_than_raising(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("SLOW_REQUEST_THRESHOLD_MS", "not-a-number")

        with caplog.at_level(logging.WARNING):
            config = ServerConfig.from_env()

        assert config.slow_request_threshold_ms is None
        assert any("SLOW_REQUEST_THRESHOLD_MS" in r.message for r in caplog.records)


class TestApplyCliOverridesLogLevel:
    def test_valid_log_level_is_applied_uppercased(self) -> None:
        config = _server_config()

        config.apply_cli_overrides(log_level="debug")

        assert config.log_level == "DEBUG"

    def test_invalid_log_level_raises(self) -> None:
        config = _server_config()

        with pytest.raises(ValueError, match="Invalid log level"):
            config.apply_cli_overrides(log_level="not-a-level")

    def test_none_leaves_existing_log_level_unchanged(self) -> None:
        config = _server_config()
        config.log_level = "WARNING"

        config.apply_cli_overrides(log_level=None)

        assert config.log_level == "WARNING"


class TestApplyCliOverridesMaxTracesPerQuery:
    def test_valid_value_is_applied(self) -> None:
        config = _server_config()

        config.apply_cli_overrides(max_traces_per_query=250)

        assert config.max_traces_per_query == 250

    def test_zero_raises(self) -> None:
        config = _server_config()

        with pytest.raises(ValueError, match="Invalid max_traces_per_query"):
            config.apply_cli_overrides(max_traces_per_query=0)

    def test_above_max_raises(self) -> None:
        config = _server_config()

        with pytest.raises(ValueError, match="Invalid max_traces_per_query"):
            config.apply_cli_overrides(max_traces_per_query=1001)

    def test_boundary_values_are_accepted(self) -> None:
        config = _server_config()

        config.apply_cli_overrides(max_traces_per_query=1)
        assert config.max_traces_per_query == 1

        config.apply_cli_overrides(max_traces_per_query=1000)
        assert config.max_traces_per_query == 1000


class TestApplyCliOverridesUrl:
    def test_valid_url_override_is_applied(self) -> None:
        config = _server_config()

        config.apply_cli_overrides(backend_url="https://jaeger.internal:16686")

        assert str(config.backend.url) == "https://jaeger.internal:16686/"

    def test_cli_url_override_also_rejects_metadata_ip(self) -> None:
        """Regression for a real bypass: a bare TypeAdapter(HttpUrl) sets the
        attribute directly, skipping BackendConfig.validate_url entirely -
        the CVE-2025-6514 warning and the metadata-endpoint block must both
        still apply when the URL comes from a CLI/env override, not just at
        initial BackendConfig construction."""
        config = _server_config()

        with pytest.raises(ValueError, match="metadata"):
            config.apply_cli_overrides(backend_url="http://169.254.169.254/latest/meta-data/")

    def test_cli_url_override_still_warns_on_plain_http_non_local(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = _server_config()

        with caplog.at_level(logging.WARNING):
            config.apply_cli_overrides(backend_url="http://example.com:16686")

        assert any("example.com" in r.message for r in caplog.records)

    def test_none_leaves_existing_url_unchanged(self) -> None:
        config = _server_config()

        config.apply_cli_overrides(backend_url=None)

        assert str(config.backend.url) == "http://localhost:16686/"


class TestSecondaryBackendFromEnv:
    """BackendConfig.from_env_optional() (used by ServerConfig.from_env()
    for correlate_trace's optional secondary backend) - unlike the always-
    required primary backend's from_env(), an unset SECONDARY_BACKEND_TYPE
    must produce None, not a jaeger/localhost default."""

    def test_unset_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SECONDARY_BACKEND_TYPE", raising=False)

        config = ServerConfig.from_env()

        assert config.secondary_backend is None

    def test_blank_type_is_treated_the_same_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression: .env.example ships `SECONDARY_BACKEND_TYPE=` (an
        empty string once dotenv loads it, not an absent key) - it must
        not be treated as "opted in with an empty type," which would raise
        and crash server startup for anyone who leaves the example as-is."""
        monkeypatch.setenv("SECONDARY_BACKEND_TYPE", "")
        monkeypatch.delenv("SECONDARY_BACKEND_URL", raising=False)

        config = ServerConfig.from_env()

        assert config.secondary_backend is None

    def test_whitespace_only_type_is_treated_the_same_as_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SECONDARY_BACKEND_TYPE", "   ")
        monkeypatch.delenv("SECONDARY_BACKEND_URL", raising=False)

        config = ServerConfig.from_env()

        assert config.secondary_backend is None

    def test_parses_a_full_config_when_type_is_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SECONDARY_BACKEND_TYPE", "sentry")
        monkeypatch.setenv("SECONDARY_BACKEND_URL", "https://sentry.io")
        monkeypatch.setenv("SECONDARY_BACKEND_API_KEY", "sec-key")
        monkeypatch.setenv("SECONDARY_BACKEND_SENTRY_ORG", "sec-org")
        monkeypatch.setenv("SECONDARY_BACKEND_TIMEOUT", "12")

        config = ServerConfig.from_env()

        assert config.secondary_backend is not None
        assert config.secondary_backend.type == "sentry"
        assert str(config.secondary_backend.url) == "https://sentry.io/"
        assert config.secondary_backend.api_key == "sec-key"
        assert config.secondary_backend.sentry_org == "sec-org"
        assert config.secondary_backend.timeout == 12.0

    def test_invalid_type_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SECONDARY_BACKEND_TYPE", "not-a-real-backend")
        monkeypatch.setenv("SECONDARY_BACKEND_URL", "https://example.com")

        with pytest.raises(ValueError, match="Invalid SECONDARY_BACKEND_TYPE"):
            ServerConfig.from_env()

    def test_type_set_without_url_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SECONDARY_BACKEND_TYPE", "sentry")
        monkeypatch.delenv("SECONDARY_BACKEND_URL", raising=False)

        with pytest.raises(ValueError, match="SECONDARY_BACKEND_URL is required"):
            ServerConfig.from_env()

    def test_invalid_timeout_falls_back_to_default_rather_than_raising(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("SECONDARY_BACKEND_TYPE", "sentry")
        monkeypatch.setenv("SECONDARY_BACKEND_URL", "https://sentry.io")
        monkeypatch.setenv("SECONDARY_BACKEND_TIMEOUT", "not-a-number")

        with caplog.at_level(logging.WARNING):
            config = ServerConfig.from_env()

        assert config.secondary_backend is not None
        assert config.secondary_backend.timeout == 30.0
        assert any("SECONDARY_BACKEND_TIMEOUT" in r.message for r in caplog.records)

    def test_does_not_affect_the_primary_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BACKEND_TYPE", "jaeger")
        monkeypatch.setenv("BACKEND_URL", "http://localhost:16686")
        monkeypatch.setenv("SECONDARY_BACKEND_TYPE", "datadog")
        monkeypatch.setenv("SECONDARY_BACKEND_URL", "https://api.datadoghq.com")
        monkeypatch.setenv("SECONDARY_BACKEND_API_KEY", "sec-api-key")
        monkeypatch.setenv("SECONDARY_BACKEND_APP_KEY", "sec-app-key")

        config = ServerConfig.from_env()

        assert config.backend.type == "jaeger"
        assert config.secondary_backend is not None
        assert config.secondary_backend.type == "datadog"
        assert config.secondary_backend.app_key == "sec-app-key"


class TestApplyCliOverridesSlowRequestThreshold:
    def test_valid_value_is_applied(self) -> None:
        config = _server_config()

        config.apply_cli_overrides(slow_request_threshold_ms=500.0)

        assert config.slow_request_threshold_ms == 500.0

    def test_zero_raises(self) -> None:
        config = _server_config()

        with pytest.raises(ValueError, match="Invalid slow_request_threshold_ms"):
            config.apply_cli_overrides(slow_request_threshold_ms=0.0)

    def test_negative_raises(self) -> None:
        config = _server_config()

        with pytest.raises(ValueError, match="Invalid slow_request_threshold_ms"):
            config.apply_cli_overrides(slow_request_threshold_ms=-100.0)

    def test_none_leaves_existing_value_unchanged(self) -> None:
        config = _server_config()
        config.slow_request_threshold_ms = 200.0

        config.apply_cli_overrides(slow_request_threshold_ms=None)

        assert config.slow_request_threshold_ms == 200.0


class TestFromEnvQueryCache:
    """QUERY_CACHE_TTL_SECONDS used to accept any positive float (inf, 1e12)
    and the cache had no entry bound; both are now bounded at the config
    layer, with the same warn-and-default parse pattern the sibling fields
    use for unparsable text and a ValidationError for out-of-range values
    (exactly like MAX_TRACES_PER_QUERY=5000 today)."""

    def test_max_entries_defaults_to_1024(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("QUERY_CACHE_MAX_ENTRIES", raising=False)

        config = ServerConfig.from_env()

        assert config.query_cache_max_entries == 1024

    def test_max_entries_parses_a_valid_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QUERY_CACHE_MAX_ENTRIES", "256")

        config = ServerConfig.from_env()

        assert config.query_cache_max_entries == 256

    def test_max_entries_invalid_text_warns_and_keeps_the_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("QUERY_CACHE_MAX_ENTRIES", "lots")

        with caplog.at_level(logging.WARNING):
            config = ServerConfig.from_env()

        assert config.query_cache_max_entries == 1024
        assert any("QUERY_CACHE_MAX_ENTRIES" in r.message for r in caplog.records)

    @pytest.mark.parametrize("value", ["0", "100001"])
    def test_max_entries_out_of_range_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("QUERY_CACHE_MAX_ENTRIES", value)

        with pytest.raises(ValidationError):
            ServerConfig.from_env()

    @pytest.mark.parametrize("value", ["inf", "1e12", "86401", "nan"])
    def test_ttl_non_finite_or_above_one_day_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("QUERY_CACHE_TTL_SECONDS", value)

        with pytest.raises(ValidationError):
            ServerConfig.from_env()

    def test_ttl_at_the_one_day_ceiling_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QUERY_CACHE_TTL_SECONDS", "86400")

        config = ServerConfig.from_env()

        assert config.query_cache_ttl_seconds == 86400.0


class TestApplyCliOverridesQueryCache:
    @pytest.mark.parametrize("ttl", [1e12, float("inf"), float("nan"), 86400.5])
    def test_ttl_non_finite_or_above_one_day_raises(self, ttl: float) -> None:
        config = _server_config()

        with pytest.raises(ValueError, match="Invalid query_cache_ttl_seconds"):
            config.apply_cli_overrides(query_cache_ttl_seconds=ttl)

    def test_ttl_within_bounds_is_applied(self) -> None:
        config = _server_config()

        config.apply_cli_overrides(query_cache_ttl_seconds=86400.0)

        assert config.query_cache_ttl_seconds == 86400.0

    def test_max_entries_is_applied(self) -> None:
        config = _server_config()

        config.apply_cli_overrides(query_cache_max_entries=50)

        assert config.query_cache_max_entries == 50

    @pytest.mark.parametrize("max_entries", [0, -1, 100_001])
    def test_max_entries_out_of_range_raises(self, max_entries: int) -> None:
        config = _server_config()

        with pytest.raises(ValueError, match="Invalid query_cache_max_entries"):
            config.apply_cli_overrides(query_cache_max_entries=max_entries)

    def test_none_leaves_max_entries_unchanged(self) -> None:
        config = _server_config()
        config.query_cache_max_entries = 99

        config.apply_cli_overrides(query_cache_max_entries=None)

        assert config.query_cache_max_entries == 99

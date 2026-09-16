"""Tests for ServerConfig's from_env parsing and apply_cli_overrides logic
(log_level, max_traces_per_query, slow_request_threshold_ms).

Phase 4 of the post-launch plan: these three fields previously only had
env-var support (log_level, max_traces_per_query) or did not exist yet
(slow_request_threshold_ms) - apply_cli_overrides did not touch
ServerConfig-level fields at all, only BackendConfig ones.
"""

import logging

import pytest
from pydantic import HttpUrl

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

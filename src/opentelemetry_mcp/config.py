"""Configuration management for Opentelemetry MCP Server."""

import logging
import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl, TypeAdapter, field_validator

from opentelemetry_mcp.security import is_cloud_metadata_host, is_loopback_host

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class BackendConfig(BaseModel):
    """Configuration for OpenTelemetry trace backend."""

    type: Literal["jaeger", "tempo", "traceloop", "datadog", "sentry", "xray"]
    url: HttpUrl
    api_key: str | None = Field(default=None, exclude=True)
    app_key: str | None = Field(
        default=None, exclude=True, description="Datadog Application key (Datadog backend only)"
    )
    sentry_org: str | None = Field(
        default=None, exclude=True, description="Sentry organization slug (Sentry backend only)"
    )
    sentry_project: str | None = Field(
        default=None,
        exclude=True,
        description="Sentry project slug, optional (Sentry backend only)",
    )
    tempo_instance_id: str | None = Field(
        default=None,
        exclude=True,
        description="Grafana Cloud stack/instance ID for Basic Auth (Tempo backend only, "
        "used instead of Bearer auth when set - required for Grafana Cloud-hosted Tempo, "
        "not needed for self-hosted Tempo)",
    )
    aws_region: str | None = Field(
        default=None, exclude=True, description="AWS region (X-Ray backend only)"
    )
    timeout: float = Field(default=30.0, gt=0, le=300)
    environments: list[str] = Field(default_factory=lambda: ["prd"])

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: HttpUrl) -> HttpUrl:
        """Validate URL scheme and reject cloud metadata endpoints.

        A private/loopback address (e.g. a self-hosted Jaeger on a VPC or
        Docker Compose network) is a completely normal, intended deployment
        and stays allowed - only cloud instance-metadata endpoints are
        blocked outright, since those are never a legitimate trace-backend
        location and exist purely as an SSRF-driven credential-theft target
        (the vulnerability class behind GHSA-65h7/GHSA-v6ph in similar MCP
        adapter tooling).
        """
        if v.scheme not in ["http", "https"]:
            raise ValueError("URL must use http or https scheme")
        if is_cloud_metadata_host(v.host):
            raise ValueError(
                f"BACKEND_URL '{v}' points at a cloud instance-metadata endpoint. "
                "This is never a legitimate trace-backend location and is blocked "
                "to prevent SSRF-based credential theft."
            )
        if v.scheme == "http" and not is_loopback_host(v.host):
            logger.warning(
                f"BACKEND_URL '{v}' uses plain HTTP to a non-local host. "
                "This is vulnerable to network interception (see CVE-2025-6514). "
                "Use https:// unless this backend is only reachable over a "
                "trusted private network (e.g. a VPC or Docker Compose network)."
            )
        return v

    @classmethod
    def from_env(cls) -> "BackendConfig":
        """Load configuration from environment variables."""
        backend_type = os.getenv("BACKEND_TYPE", "jaeger")
        backend_url = os.getenv("BACKEND_URL", "http://localhost:16686")
        if backend_type not in ["jaeger", "tempo", "traceloop", "datadog", "sentry", "xray"]:
            raise ValueError(
                f"Invalid BACKEND_TYPE: {backend_type}. "
                "Must be one of: jaeger, tempo, traceloop, datadog, sentry, xray"
            )

        # Parse environments from comma-separated string
        environments_str = os.getenv("BACKEND_ENVIRONMENTS", "prd")
        environments = [env.strip() for env in environments_str.split(",") if env.strip()]

        # Parse timeout with validation
        timeout_str = os.getenv("BACKEND_TIMEOUT", "30")
        try:
            timeout = float(timeout_str)
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid BACKEND_TIMEOUT value '{timeout_str}': {e}. Using default: 30")
            timeout = 30.0

        return cls(
            type=backend_type,  # type: ignore
            url=backend_url,  # type: ignore
            api_key=os.getenv("BACKEND_API_KEY"),
            app_key=os.getenv("BACKEND_APP_KEY"),
            sentry_org=os.getenv("BACKEND_SENTRY_ORG"),
            sentry_project=os.getenv("BACKEND_SENTRY_PROJECT"),
            tempo_instance_id=os.getenv("BACKEND_TEMPO_INSTANCE_ID"),
            aws_region=os.getenv("BACKEND_AWS_REGION"),
            timeout=timeout,
            environments=environments,
        )


class ServerConfig(BaseModel):
    """MCP Server configuration."""

    backend: BackendConfig
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    max_traces_per_query: int = Field(default=500, ge=1, le=1000)
    slow_request_threshold_ms: float | None = Field(default=None, gt=0)
    query_cache_ttl_seconds: float | None = Field(default=None, gt=0)

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load server configuration from environment variables."""
        log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR")
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = (
            log_level_str if log_level_str in valid_levels else "INFO"  # type: ignore[assignment]
        )

        # Parse max_traces_per_query with validation
        max_traces_str = os.getenv("MAX_TRACES_PER_QUERY", "500")
        try:
            max_traces_per_query = int(max_traces_str)
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Invalid MAX_TRACES_PER_QUERY value '{max_traces_str}': {e}. Using default: 500"
            )
            max_traces_per_query = 500

        # Parse slow_request_threshold_ms with validation (optional, unset by default)
        slow_request_threshold_ms: float | None = None
        slow_request_threshold_str = os.getenv("SLOW_REQUEST_THRESHOLD_MS")
        if slow_request_threshold_str:
            try:
                slow_request_threshold_ms = float(slow_request_threshold_str)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid SLOW_REQUEST_THRESHOLD_MS value "
                    f"'{slow_request_threshold_str}': {e}. Slow-request logging disabled."
                )

        # Parse query_cache_ttl_seconds with validation (optional, unset by default)
        query_cache_ttl_seconds: float | None = None
        query_cache_ttl_str = os.getenv("QUERY_CACHE_TTL_SECONDS")
        if query_cache_ttl_str:
            try:
                query_cache_ttl_seconds = float(query_cache_ttl_str)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid QUERY_CACHE_TTL_SECONDS value "
                    f"'{query_cache_ttl_str}': {e}. Query caching disabled."
                )

        return cls(
            backend=BackendConfig.from_env(),
            log_level=log_level,
            max_traces_per_query=max_traces_per_query,
            slow_request_threshold_ms=slow_request_threshold_ms,
            query_cache_ttl_seconds=query_cache_ttl_seconds,
        )

    def apply_cli_overrides(
        self,
        backend_type: str | None = None,
        backend_url: str | None = None,
        api_key: str | None = None,
        app_key: str | None = None,
        sentry_org: str | None = None,
        sentry_project: str | None = None,
        tempo_instance_id: str | None = None,
        aws_region: str | None = None,
        environments: str | None = None,
        log_level: str | None = None,
        max_traces_per_query: int | None = None,
        slow_request_threshold_ms: float | None = None,
        query_cache_ttl_seconds: float | None = None,
    ) -> None:
        """Apply CLI argument overrides to configuration."""
        if slow_request_threshold_ms is not None:
            if slow_request_threshold_ms <= 0:
                raise ValueError(
                    f"Invalid slow_request_threshold_ms: {slow_request_threshold_ms}. "
                    "Must be greater than 0"
                )
            self.slow_request_threshold_ms = slow_request_threshold_ms

        if query_cache_ttl_seconds is not None:
            if query_cache_ttl_seconds <= 0:
                raise ValueError(
                    f"Invalid query_cache_ttl_seconds: {query_cache_ttl_seconds}. "
                    "Must be greater than 0"
                )
            self.query_cache_ttl_seconds = query_cache_ttl_seconds

        if log_level:
            log_level_upper = log_level.upper()
            if log_level_upper not in ("DEBUG", "INFO", "WARNING", "ERROR"):
                raise ValueError(
                    f"Invalid log level: {log_level}. Must be one of: DEBUG, INFO, WARNING, ERROR"
                )
            self.log_level = log_level_upper  # type: ignore[assignment]

        if max_traces_per_query is not None:
            if not (1 <= max_traces_per_query <= 1000):
                raise ValueError(
                    f"Invalid max_traces_per_query: {max_traces_per_query}. Must be between 1 and 1000"
                )
            self.max_traces_per_query = max_traces_per_query

        if backend_type:
            if backend_type not in ["jaeger", "tempo", "traceloop", "datadog", "sentry", "xray"]:
                raise ValueError(
                    f"Invalid backend type: {backend_type}. "
                    "Must be one of: jaeger, tempo, traceloop, datadog, sentry, xray"
                )
            self.backend.type = backend_type  # type: ignore

        if backend_url:
            # Route through the same validator BackendConfig construction
            # uses - a bare TypeAdapter validates the HttpUrl *type* only,
            # bypassing validate_url's scheme/metadata/CVE-2025-6514 checks
            # entirely, since direct attribute assignment on an already
            # constructed model does not re-run field validators here.
            parsed = TypeAdapter(HttpUrl).validate_python(backend_url)
            self.backend.url = BackendConfig.validate_url(parsed)

        if api_key:
            self.backend.api_key = api_key

        if app_key:
            self.backend.app_key = app_key

        if sentry_org:
            self.backend.sentry_org = sentry_org

        if sentry_project:
            self.backend.sentry_project = sentry_project

        if tempo_instance_id:
            self.backend.tempo_instance_id = tempo_instance_id

        if aws_region:
            self.backend.aws_region = aws_region

        if environments:
            self.backend.environments = [
                env.strip() for env in environments.split(",") if env.strip()
            ]

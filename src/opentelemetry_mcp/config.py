"""Configuration management for Opentelemetry MCP Server."""

import logging
import math
import os
from typing import Literal
from urllib.parse import urlsplit

from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl, TypeAdapter, field_validator

from opentelemetry_mcp.backends.base import DEFAULT_MAX_RESPONSE_BYTES
from opentelemetry_mcp.backends.cache import DEFAULT_QUERY_CACHE_MAX_ENTRIES
from opentelemetry_mcp.security import (
    is_cloud_metadata_host,
    is_loopback_host,
    redact_userinfo_in_text,
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def _reject_url_userinfo(raw_url: str, *, source: str) -> None:
    """Refuse a URL carrying a userinfo component BEFORE pydantic parses it.

    BackendConfig.validate_url rejects the same URLs, but by then the raw
    string is pydantic's input: a ValidationError's str() appends
    input_value=<the URL> truncated by pydantic-core to 24 + "..." + 23
    characters, and for a URL longer than ~50 characters that cut removes
    the "@" the textual redaction (security.redact_userinfo_in_text) keys
    on - so the password's leading characters survived in the startup error
    on every surface (runtime verification P7b). Judging the raw env/CLI
    string here means a credentialed URL never becomes a ValidationError
    input at all; the only text that leaves is the redacted URL. `source`
    names the variable or flag the value came from.
    """
    try:
        parts = urlsplit(raw_url)
        has_userinfo = bool(parts.username or parts.password)
    except ValueError:
        # Malformed authority (an unbalanced IPv6 bracket): pydantic would
        # echo it too, so fall through to the textual shape check alone.
        has_userinfo = False
    redacted = redact_userinfo_in_text(raw_url)
    if has_userinfo or redacted != raw_url:
        raise ValueError(
            f"{source} must not embed credentials (user:password@host); got {redacted!r}. "
            "Use BACKEND_API_KEY (Bearer) or BACKEND_TEMPO_INSTANCE_ID + BACKEND_API_KEY "
            "(Basic) so the secret is never stringified"
        )


class BackendConfig(BaseModel):
    """Configuration for OpenTelemetry trace backend."""

    type: Literal[
        "jaeger", "tempo", "traceloop", "datadog", "sentry", "xray", "newrelic", "honeycomb"
    ]
    url: HttpUrl
    api_key: str | None = Field(default=None, exclude=True, repr=False)
    app_key: str | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Datadog Application key (Datadog backend only)",
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
    newrelic_account_id: str | None = Field(
        default=None,
        exclude=True,
        description="New Relic account ID, required since NerdGraph queries are "
        "user-scoped not account-scoped (New Relic backend only)",
    )
    honeycomb_dataset: str | None = Field(
        default=None,
        exclude=True,
        description="Honeycomb dataset slug, required since the Query Data API is "
        "dataset-scoped (Honeycomb backend only)",
    )
    timeout: float = Field(default=30.0, gt=0, le=300)
    max_response_bytes: int = Field(
        default=DEFAULT_MAX_RESPONSE_BYTES,
        ge=1,
        le=1024**3,
        description="Ceiling on the decoded size of any single backend response, in bytes; "
        "an oversized body is refused at the transport before it is buffered or parsed",
    )
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
        if v.username or v.password:
            # Never keep a credential inside the URL: str(url) is logged at
            # startup, printed by --print-config and doctor, and echoed by
            # httpx's own error text into /ready bodies and tool errors.
            # Checked first so neither message below can stringify it.
            raise ValueError(
                "BACKEND_URL must not embed credentials (user:password@host). Use "
                "BACKEND_API_KEY (Bearer) or BACKEND_TEMPO_INSTANCE_ID + BACKEND_API_KEY "
                "(Basic) so the secret is never stringified"
            )
        if is_cloud_metadata_host(v.host):
            raise ValueError(
                f"BACKEND_URL host '{v.host}' is a cloud instance-metadata endpoint. "
                "This is never a legitimate trace-backend location and is blocked "
                "to prevent SSRF-based credential theft."
            )
        if v.scheme == "http" and not is_loopback_host(v.host):
            logger.warning(
                f"BACKEND_URL host '{v.host}' is reached over plain HTTP from a non-local "
                "host. This is vulnerable to network interception (see CVE-2025-6514). "
                "Use https:// unless this backend is only reachable over a "
                "trusted private network (e.g. a VPC or Docker Compose network)."
            )
        return v

    @classmethod
    def from_env(cls) -> "BackendConfig":
        """Load the primary backend's configuration from BACKEND_* env vars."""
        return cls._from_env_with_prefix(
            prefix="BACKEND_",
            type_default="jaeger",
            url_default="http://localhost:16686",
        )

    @classmethod
    def from_env_optional(cls, prefix: str = "SECONDARY_BACKEND_") -> "BackendConfig | None":
        """Load an optional secondary backend's configuration from
        `{prefix}*` env vars (SECONDARY_BACKEND_* by default), for
        cross-backend correlation (tools/correlate.py). Returns None if
        `{prefix}TYPE` is unset OR blank, so an unconfigured secondary
        backend is a valid, opt-in-by-default state - unlike the always-
        required primary backend, this must not silently fall back to
        jaeger/localhost. Blank (not just absent) is treated the same as
        unset because .env.example ships `SECONDARY_BACKEND_TYPE=` with no
        value, matching every other optional field's convention in that
        file - `dotenv.load_dotenv()` sets that to an empty string, not
        leaving the key absent, so `os.getenv(...) is None` alone would
        incorrectly treat "the user left the example's blank default in
        place" as "the user opted in with an empty type."
        """
        if not (os.getenv(f"{prefix}TYPE") or "").strip():
            return None
        return cls._from_env_with_prefix(prefix=prefix, type_default=None, url_default=None)

    @classmethod
    def _from_env_with_prefix(
        cls, *, prefix: str, type_default: str | None, url_default: str | None
    ) -> "BackendConfig":
        """Shared env-var parsing behind from_env()/from_env_optional() -
        every BackendConfig field, read from `{prefix}<FIELD>`."""
        backend_type = os.getenv(f"{prefix}TYPE", type_default)
        backend_url = os.getenv(f"{prefix}URL", url_default)
        if backend_type not in [
            "jaeger",
            "tempo",
            "traceloop",
            "datadog",
            "sentry",
            "xray",
            "newrelic",
            "honeycomb",
        ]:
            raise ValueError(
                f"Invalid {prefix}TYPE: {backend_type}. "
                "Must be one of: jaeger, tempo, traceloop, datadog, sentry, xray, newrelic, honeycomb"
            )
        if not backend_url:
            raise ValueError(f"{prefix}URL is required when {prefix}TYPE is set")
        _reject_url_userinfo(backend_url, source=f"{prefix}URL")

        # Parse environments from comma-separated string
        environments_str = os.getenv(f"{prefix}ENVIRONMENTS", "prd")
        environments = [env.strip() for env in environments_str.split(",") if env.strip()]

        # Parse timeout with validation
        timeout_str = os.getenv(f"{prefix}TIMEOUT", "30")
        try:
            timeout = float(timeout_str)
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid {prefix}TIMEOUT value '{timeout_str}': {e}. Using default: 30")
            timeout = 30.0

        # Parse max_response_bytes with validation (optional; range-checked by
        # the Field so an out-of-range value fails fast like a bad timeout)
        max_response_bytes = DEFAULT_MAX_RESPONSE_BYTES
        max_response_bytes_str = os.getenv(f"{prefix}MAX_RESPONSE_BYTES")
        if max_response_bytes_str:
            try:
                max_response_bytes = int(max_response_bytes_str)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid {prefix}MAX_RESPONSE_BYTES value '{max_response_bytes_str}': {e}. "
                    f"Using default: {DEFAULT_MAX_RESPONSE_BYTES}"
                )

        return cls(
            type=backend_type,  # type: ignore
            url=backend_url,  # type: ignore
            api_key=os.getenv(f"{prefix}API_KEY"),
            app_key=os.getenv(f"{prefix}APP_KEY"),
            sentry_org=os.getenv(f"{prefix}SENTRY_ORG"),
            sentry_project=os.getenv(f"{prefix}SENTRY_PROJECT"),
            tempo_instance_id=os.getenv(f"{prefix}TEMPO_INSTANCE_ID"),
            aws_region=os.getenv(f"{prefix}AWS_REGION"),
            newrelic_account_id=os.getenv(f"{prefix}NEWRELIC_ACCOUNT_ID"),
            honeycomb_dataset=os.getenv(f"{prefix}HONEYCOMB_DATASET"),
            timeout=timeout,
            max_response_bytes=max_response_bytes,
            environments=environments,
        )


# Ceilings for the opt-in query cache. One day mirrors BACKEND_TIMEOUT's own
# `le` pattern; 100000 entries is far above any deployment's distinct query
# count but still a bound a single peer cannot push past.
MAX_QUERY_CACHE_TTL_SECONDS = 86400.0
MAX_QUERY_CACHE_MAX_ENTRIES = 100_000


class ServerConfig(BaseModel):
    """MCP Server configuration."""

    backend: BackendConfig
    # Opt-in only, for tools/correlate.py's correlate_trace - unlike
    # `backend`, there is deliberately no CLI override surface for this in
    # v1 (SECONDARY_BACKEND_* env vars only); see the Roadmap/README note
    # on cross-backend correlation for why this stays a smaller surface
    # than the primary backend's.
    secondary_backend: BackendConfig | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    max_traces_per_query: int = Field(default=500, ge=1, le=1000)
    slow_request_threshold_ms: float | None = Field(default=None, gt=0)
    # Bounded above (one day) and finite: an inf TTL meant "never expire"
    # and never opened the cache's expiry sweep, so nothing was ever evicted
    # from the process-wide cache every MCP session shares.
    query_cache_ttl_seconds: float | None = Field(
        default=None, gt=0, le=MAX_QUERY_CACHE_TTL_SECONDS, allow_inf_nan=False
    )
    # LRU entry bound on the query cache, independent of the TTL - the
    # aggregate cap on what distinct tool arguments can make the process
    # retain (backends/cache.py's DEFAULT_QUERY_CACHE_MAX_ENTRIES).
    query_cache_max_entries: int = Field(
        default=DEFAULT_QUERY_CACHE_MAX_ENTRIES, ge=1, le=MAX_QUERY_CACHE_MAX_ENTRIES
    )

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

        # Parse query_cache_max_entries with validation (same warn-and-default
        # pattern; out-of-range values are rejected by the field bounds below)
        query_cache_max_entries = DEFAULT_QUERY_CACHE_MAX_ENTRIES
        query_cache_max_entries_str = os.getenv("QUERY_CACHE_MAX_ENTRIES")
        if query_cache_max_entries_str:
            try:
                query_cache_max_entries = int(query_cache_max_entries_str)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid QUERY_CACHE_MAX_ENTRIES value "
                    f"'{query_cache_max_entries_str}': {e}. "
                    f"Using default: {DEFAULT_QUERY_CACHE_MAX_ENTRIES}"
                )

        return cls(
            backend=BackendConfig.from_env(),
            secondary_backend=BackendConfig.from_env_optional(),
            log_level=log_level,
            max_traces_per_query=max_traces_per_query,
            slow_request_threshold_ms=slow_request_threshold_ms,
            query_cache_ttl_seconds=query_cache_ttl_seconds,
            query_cache_max_entries=query_cache_max_entries,
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
        newrelic_account_id: str | None = None,
        honeycomb_dataset: str | None = None,
        environments: str | None = None,
        log_level: str | None = None,
        max_traces_per_query: int | None = None,
        slow_request_threshold_ms: float | None = None,
        query_cache_ttl_seconds: float | None = None,
        query_cache_max_entries: int | None = None,
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
            # isfinite first: nan compares False against every bound below.
            if (
                not math.isfinite(query_cache_ttl_seconds)
                or query_cache_ttl_seconds <= 0
                or query_cache_ttl_seconds > MAX_QUERY_CACHE_TTL_SECONDS
            ):
                raise ValueError(
                    f"Invalid query_cache_ttl_seconds: {query_cache_ttl_seconds}. "
                    f"Must be greater than 0 and at most {MAX_QUERY_CACHE_TTL_SECONDS:g}"
                )
            self.query_cache_ttl_seconds = query_cache_ttl_seconds

        if query_cache_max_entries is not None:
            if not (1 <= query_cache_max_entries <= MAX_QUERY_CACHE_MAX_ENTRIES):
                raise ValueError(
                    f"Invalid query_cache_max_entries: {query_cache_max_entries}. "
                    f"Must be between 1 and {MAX_QUERY_CACHE_MAX_ENTRIES}"
                )
            self.query_cache_max_entries = query_cache_max_entries

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
            if backend_type not in [
                "jaeger",
                "tempo",
                "traceloop",
                "datadog",
                "sentry",
                "xray",
                "newrelic",
                "honeycomb",
            ]:
                raise ValueError(
                    f"Invalid backend type: {backend_type}. "
                    "Must be one of: jaeger, tempo, traceloop, datadog, sentry, xray, newrelic, "
                    "honeycomb"
                )
            self.backend.type = backend_type  # type: ignore

        if backend_url:
            # Route through the same validator BackendConfig construction
            # uses - a bare TypeAdapter validates the HttpUrl *type* only,
            # bypassing validate_url's scheme/metadata/CVE-2025-6514 checks
            # entirely, since direct attribute assignment on an already
            # constructed model does not re-run field validators here.
            # The userinfo check runs on the raw string first: the TypeAdapter's
            # own ValidationError would echo a malformed credentialed URL.
            _reject_url_userinfo(backend_url, source="--url")
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

        if newrelic_account_id:
            self.backend.newrelic_account_id = newrelic_account_id

        if honeycomb_dataset:
            self.backend.honeycomb_dataset = honeycomb_dataset

        if environments:
            self.backend.environments = [
                env.strip() for env in environments.split(",") if env.strip()
            ]

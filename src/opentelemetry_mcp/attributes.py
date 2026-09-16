"""Strongly-typed span attribute models following OpenTelemetry semantic conventions."""

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Note: We import constants for documentation but must use string literals for Pydantic aliases
# due to mypy strict mode requirements
from .constants import GenAI

# OTel GenAI semantic conventions renamed gen_ai.system to gen_ai.provider.name
# in semconv v1.37.0 - confirmed against the real release changelog, not
# assumed. Because SpanAttributes uses extra="allow" (not "ignore"/"forbid"),
# a span carrying only the new name doesn't raise or get dropped - it
# silently lands in extra_attributes instead of the typed gen_ai_system
# field, which is what is_llm_span/LLMSpanAttributes.from_span/FilterEngine's
# client-side filtering all actually read. _normalize_gen_ai_provider_rename
# below closes that gap once, at the model level, for every backend.
_GEN_AI_PROVIDER_NAME_KEY = "gen_ai.provider.name"


class SpanAttributes(BaseModel):
    """
    Strongly-typed span attributes following OpenTelemetry semantic conventions.

    Supports both gen_ai.* (OpenTelemetry standard) and llm.* (legacy Traceloop) naming conventions
    through field aliases. The primary access pattern uses gen_ai.* attributes.

    The model allows extra fields through ConfigDict(extra='allow') to support additional
    unknown attributes that may be present in span data.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # LLM System and Model
    gen_ai_system: str | None = Field(None, alias="gen_ai.system")
    gen_ai_request_model: str | None = Field(None, alias="gen_ai.request.model")
    gen_ai_response_model: str | None = Field(None, alias="gen_ai.response.model")
    gen_ai_operation_name: str | None = Field(None, alias="gen_ai.operation.name")

    # Request Parameters
    gen_ai_request_temperature: float | None = Field(None, alias="gen_ai.request.temperature")
    gen_ai_request_top_p: float | None = Field(None, alias="gen_ai.request.top_p")
    gen_ai_request_max_tokens: int | None = Field(None, alias="gen_ai.request.max_tokens")
    gen_ai_request_is_streaming: bool | None = Field(None, alias="gen_ai.request.is_streaming")

    # Response Attributes
    gen_ai_response_finish_reasons: list[str] | None = Field(
        None, alias="gen_ai.response.finish_reasons"
    )
    gen_ai_system_instructions: list[dict[str, str]] | None = Field(
        None, alias="gen_ai.system_instructions"
    )
    # dict[str, Any], not dict[str, str] like system_instructions above -
    # OTel message objects are {"role": ..., "parts": [...]}, where "parts"
    # is itself a nested list, so a str-only value type would reject real
    # message data. Explicit Any is fine under mypy strict (it forbids
    # implicit Any, not this).
    gen_ai_input_messages: list[dict[str, Any]] | None = Field(None, alias="gen_ai.input.messages")
    gen_ai_output_messages: list[dict[str, Any]] | None = Field(
        None, alias="gen_ai.output.messages"
    )
    gen_ai_retrieval_documents: list[dict[str, Any]] | None = Field(
        None, alias="gen_ai.retrieval.documents"
    )

    # Conversation and prompt identity (OTel GenAI semconv, Development status)
    gen_ai_conversation_id: str | None = Field(None, alias="gen_ai.conversation.id")
    gen_ai_prompt_name: str | None = Field(None, alias="gen_ai.prompt.name")
    gen_ai_prompt_version: str | None = Field(None, alias="gen_ai.prompt.version")

    # Usage Metrics (gen_ai.* format)
    gen_ai_usage_prompt_tokens: int | None = Field(None, alias="gen_ai.usage.prompt_tokens")
    gen_ai_usage_input_tokens: int | None = Field(None, alias="gen_ai.usage.input_tokens")
    gen_ai_usage_completion_tokens: int | None = Field(None, alias="gen_ai.usage.completion_tokens")
    gen_ai_usage_output_tokens: int | None = Field(None, alias="gen_ai.usage.output_tokens")
    gen_ai_usage_total_tokens: int | None = Field(None, alias="gen_ai.usage.total_tokens")

    # Legacy llm.* attributes (for backward compatibility with Traceloop)
    llm_vendor: str | None = Field(None, alias="llm.vendor")
    llm_request_model: str | None = Field(None, alias="llm.request.model")
    llm_response_finish_reasons: list[str] | None = Field(None, alias="llm.response.finish_reasons")
    llm_usage_prompt_tokens: int | None = Field(None, alias="llm.usage.prompt_tokens")
    llm_usage_input_tokens: int | None = Field(None, alias="llm.usage.input_tokens")
    llm_usage_completion_tokens: int | None = Field(None, alias="llm.usage.completion_tokens")
    llm_usage_output_tokens: int | None = Field(None, alias="llm.usage.output_tokens")
    llm_usage_total_tokens: int | None = Field(None, alias="llm.usage.total_tokens")

    # OpenTelemetry Standard Attributes
    service_name: str | None = Field(None, alias="service.name")
    otel_status_code: str | None = Field(None, alias="otel.status_code")
    error: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_gen_ai_provider_rename(cls, data: Any) -> Any:
        """Fall back to gen_ai.provider.name when gen_ai.system is absent.

        A field_validator on gen_ai_system can't see this: it only ever
        receives the value already routed to that field by its own alias,
        never a sibling key elsewhere in the input. This is inherently a
        cross-key concern, so it needs a model-level, before-validation
        look at the whole input instead - same "never raise, best-effort
        normalize" spirit as this class's two field-level coercers above,
        just scoped to the mapping rather than one field.

        gen_ai.system stays canonical (preferred when both are present)
        since it's what the rest of this codebase already keys off; this
        only fills the gap when it's missing, and removes the raw
        gen_ai.provider.name key so the same value doesn't also duplicate
        into extra_attributes once it has been consumed as the fallback.
        """
        if not isinstance(data, dict):
            return data
        has_system = data.get("gen_ai.system") is not None or data.get("gen_ai_system") is not None
        if has_system:
            return data
        provider_name = data.get(GenAI.PROVIDER_NAME)
        if provider_name is None:
            return data
        return {
            **{k: v for k, v in data.items() if k != GenAI.PROVIDER_NAME},
            "gen_ai.system": provider_name,
        }

    @field_validator("gen_ai_response_finish_reasons", "llm_response_finish_reasons", mode="before")
    @classmethod
    def _coerce_finish_reasons(cls, value: Any) -> Any:
        """Coerce finish_reasons into a real list before type validation.

        Backends that store OTel attributes as flat tags (Jaeger) or that
        serialize OTLP arrayValue by hand (Tempo) hand this field a string
        instead of a list - a JSON-encoded array ('["stop"]') or, in
        Tempo's case, a Python repr of the raw arrayValue structure. Without
        this, list[str] validation raises, SpanAttributes(**attrs)
        construction fails, and the calling backend's per-span parser
        silently drops the whole span rather than losing just this field.
        """
        if value is None or isinstance(value, list):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                except (json.JSONDecodeError, ValueError):
                    parsed = None
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            return [part.strip() for part in stripped.split(",") if part.strip()]
        # Any other type (int, dict, bool, ...) can't be interpreted as a list
        # of finish reasons. Coercing to None - dropping just this field - is
        # a strictly smaller loss than the ValidationError raising here would
        # cause, per this function's own never-raise contract above.
        return None

    @field_validator("gen_ai_system_instructions", mode="before")
    @classmethod
    def _coerce_system_instructions(cls, value: Any) -> Any:
        """Coerce system_instructions into a real list before type validation.

        gen_ai.system_instructions is emitted by instrumentation (e.g. the
        Vercel AI SDK's OTel integration) as its own JSON array of objects
        shaped like {"type": "text", "content": ...}, distinct from
        gen_ai.input.messages. Just like finish_reasons above, backends that
        can't preserve a real list/array structure - Jaeger's flat-tag model,
        Tempo's hand-rolled OTLP parser - hand this field a JSON-encoded
        string instead of a list. Without this, list[dict[str, str]]
        validation raises, SpanAttributes(**attrs) construction fails, and
        the calling backend's per-span parser silently drops the whole span
        rather than losing just this field.
        """
        if value is None or isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value.strip())
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if isinstance(parsed, list):
                if all(isinstance(item, dict) for item in parsed):
                    return parsed
                # A JSON array of non-dict items (e.g. plain strings) doesn't
                # conform to list[dict[str, str]] - wrap each element so the
                # data survives instead of raising.
                return [{"type": "text", "content": str(item)} for item in parsed]
            # Either not valid JSON at all, or valid JSON that isn't a list
            # (e.g. a bare string or number) - never drop the underlying
            # string data just because it didn't arrive as a clean JSON
            # array. Mirrors _coerce_finish_reasons's comma-split fallback
            # in spirit: wrap the whole string as a single instruction.
            return [{"type": "text", "content": value}]
        # Any other type (int, dict, bool, ...) can't be interpreted as a
        # list of system instructions. Coercing to None - dropping just this
        # field - is a strictly smaller loss than the ValidationError raising
        # here would cause, per this function's own never-raise contract
        # above.
        return None

    @field_validator("gen_ai_input_messages", "gen_ai_output_messages", mode="before")
    @classmethod
    def _coerce_messages(cls, value: Any) -> Any:
        """Coerce input/output messages into a real list before type
        validation - same never-raise shape as _coerce_system_instructions
        above, since Jaeger/Tempo can both hand this field a JSON-encoded
        string instead of a real list."""
        if value is None or isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value.strip())
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if isinstance(parsed, list):
                if all(isinstance(item, dict) for item in parsed):
                    return parsed
                # A JSON array of non-dict items doesn't conform to
                # list[dict[str, Any]] - wrap each element so the data
                # survives instead of raising.
                return [{"role": "unknown", "content": str(item)} for item in parsed]
            # Either not valid JSON at all, or valid JSON that isn't a list
            # - never drop the underlying string data just because it
            # didn't arrive as a clean JSON array.
            return [{"role": "unknown", "content": value}]
        # Any other type (int, dict, bool, ...) can't be interpreted as a
        # list of messages. Coercing to None - dropping just this field -
        # is a strictly smaller loss than the ValidationError raising here
        # would cause, per this function's own never-raise contract above.
        return None

    @field_validator("gen_ai_retrieval_documents", mode="before")
    @classmethod
    def _coerce_retrieval_documents(cls, value: Any) -> Any:
        """Coerce retrieval documents into a real list before type
        validation - same never-raise shape as _coerce_messages above."""
        if value is None or isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value.strip())
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if isinstance(parsed, list):
                if all(isinstance(item, dict) for item in parsed):
                    return parsed
                return [{"content": str(item)} for item in parsed]
            return [{"content": value}]
        return None

    def to_dict(self) -> dict[str, str | int | float | bool]:
        """
        Convert to dictionary representation with dotted keys.

        Returns only non-None values with their original dotted notation (e.g., "gen_ai.system").
        """
        result: dict[str, str | int | float | bool] = {}

        # Map field names back to their aliases for proper serialization
        for field_name, field_info in self.__class__.model_fields.items():
            value = getattr(self, field_name)
            if value is not None:
                # Use the alias if available, otherwise use field name
                key = field_info.alias or field_name
                result[key] = value

        # Add extra fields that were allowed through ConfigDict(extra='allow')
        if hasattr(self, "__pydantic_extra__") and self.__pydantic_extra__:
            for key, value in self.__pydantic_extra__.items():
                if value is not None:
                    result[key] = value

        return result

    @property
    def extra_attributes(self) -> dict[str, str | int | float | bool]:
        """Attributes not covered by any typed field above - e.g. score.*/
        evaluation.*-shaped attributes an instrumentation added that this
        model does not explicitly define - surfaced via
        ConfigDict(extra='allow'). Unlike to_dict(), this excludes the
        typed fields, so callers that already expose those separately
        (e.g. SpanSummary's gen_ai_system/total_tokens) do not duplicate
        them."""
        if not self.__pydantic_extra__:
            return {}
        return {k: v for k, v in self.__pydantic_extra__.items() if v is not None}

    def get(
        self, key: str, default: str | int | float | bool | None = None
    ) -> str | int | float | bool | None:
        """
        Get attribute value by key, supporting both dotted notation and field names.

        This method provides backward compatibility with dict-style access patterns.

        Args:
            key: Attribute key (e.g., "gen_ai.system" or "gen_ai_system")
            default: Default value if key not found

        Returns:
            Attribute value or default
        """
        # Try direct field access first (underscore notation)
        field_name = key.replace(".", "_")
        if hasattr(self, field_name):
            value = getattr(self, field_name)
            if value is not None:
                # Cast to ensure the return type matches the signature
                return value  # type: ignore[no-any-return]

        # Try extra fields (dotted notation)
        if (
            hasattr(self, "__pydantic_extra__")
            and self.__pydantic_extra__
            and key in self.__pydantic_extra__
        ):
            return self.__pydantic_extra__[key]  # type: ignore[no-any-return]

        return default

    def __getitem__(self, key: str) -> str | int | float | bool:
        """
        Get attribute value using subscript notation for backward compatibility.

        Args:
            key: Attribute key (e.g., "gen_ai.system")

        Returns:
            Attribute value

        Raises:
            KeyError: If key not found
        """
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value


class SpanEvent(BaseModel):
    """
    Strongly-typed span event structure.

    Represents an event that occurred during span execution, such as prompt content
    or completion results in LLM operations.
    """

    model_config = ConfigDict(extra="allow")

    name: str = Field(
        ...,
        description=f"Event name (e.g., '{GenAI.EVENT_CONTENT_PROMPT}', '{GenAI.EVENT_CONTENT_COMPLETION}')",
    )
    timestamp: int = Field(..., description="Unix timestamp in nanoseconds")
    attributes: dict[str, str | int | float | bool] = Field(
        default_factory=dict, description="Event attributes with typed values"
    )


class HealthCheckResponse(BaseModel):
    """Health check response from backend systems."""

    status: Literal["healthy", "unhealthy"] = Field(..., description="Health status of the backend")
    backend: Literal["jaeger", "tempo", "traceloop", "datadog", "sentry"] = Field(
        ..., description="Backend type"
    )
    url: str = Field(..., description="Backend URL")
    error: str | None = Field(default=None, description="Error message if unhealthy")

    # Backend-specific fields
    service_count: int | None = Field(
        default=None, description="Number of services available (Jaeger)"
    )
    project_id: str | None = Field(default=None, description="Project ID (Traceloop)")

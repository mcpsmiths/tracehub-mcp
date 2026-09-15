"""Tests for SpanAttributes, focused on the finish_reasons coercion bug.

Regression coverage for a real bug found via a live end-to-end dry run: a
span with gen_ai.response.finish_reasons set was silently dropped from
every query result on the Jaeger and Tempo backends, with no error
surfaced anywhere. Root cause: gen_ai_response_finish_reasons is typed
list[str], but Jaeger's tag model and Tempo's hand-rolled OTLP parser both
handed it a string instead of a real list, which raised a pydantic
ValidationError inside SpanAttributes(**attrs) construction - and both
backends' per-span parsers caught that broadly and silently returned None,
indistinguishable from "no span".

Uses model_validate(dict) rather than SpanAttributes(**dict) - the
dict-splat form defeats mypy's keyword-argument analysis for a model with
this many optional aliased fields (every existing SpanAttributes(**...)
call site in src/ works around the same issue with a pre-typed
intermediate variable or a `# type: ignore[arg-type]`); model_validate is
the pattern the rest of this test suite already uses for the same reason.
"""

from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from opentelemetry_mcp.attributes import SpanAttributes


def test_finish_reasons_accepts_a_real_list_unchanged() -> None:
    attrs = SpanAttributes.model_validate({"gen_ai.response.finish_reasons": ["stop", "length"]})
    assert attrs.gen_ai_response_finish_reasons == ["stop", "length"]


def test_finish_reasons_accepts_none() -> None:
    attrs = SpanAttributes.model_validate({})
    assert attrs.gen_ai_response_finish_reasons is None


def test_finish_reasons_parses_jaeger_style_json_encoded_string() -> None:
    """Jaeger's tag model has no array type - the OTLP collector encodes a
    list attribute as a JSON string when writing it as a tag. Construction
    must not raise, and the value must come out as a real list."""
    attrs = SpanAttributes.model_validate({"gen_ai.response.finish_reasons": '["stop"]'})
    assert attrs.gen_ai_response_finish_reasons == ["stop"]


def test_finish_reasons_parses_multi_value_json_encoded_string() -> None:
    attrs = SpanAttributes.model_validate({"gen_ai.response.finish_reasons": '["stop", "length"]'})
    assert attrs.gen_ai_response_finish_reasons == ["stop", "length"]


def test_finish_reasons_falls_back_to_comma_split_for_a_plain_string() -> None:
    attrs = SpanAttributes.model_validate({"gen_ai.response.finish_reasons": "stop,length"})
    assert attrs.gen_ai_response_finish_reasons == ["stop", "length"]


def test_finish_reasons_does_not_raise_on_unparseable_garbage() -> None:
    """Must never raise regardless of input shape - a raised exception here
    is exactly what caused the whole span to be silently dropped upstream."""
    attrs = SpanAttributes.model_validate({"gen_ai.response.finish_reasons": "not json but fine"})
    assert attrs.gen_ai_response_finish_reasons == ["not json but fine"]


def test_legacy_llm_response_finish_reasons_gets_the_same_coercion() -> None:
    attrs = SpanAttributes.model_validate({"llm.response.finish_reasons": '["stop"]'})
    assert attrs.llm_response_finish_reasons == ["stop"]


@given(
    value=st.one_of(
        st.none(),
        st.text(),
        st.lists(st.text()),
        st.integers(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.booleans(),
        st.dictionaries(st.text(), st.integers()),
        st.tuples(st.integers(), st.integers()),
    )
)
def test_finish_reasons_coercion_never_raises(value: object) -> None:
    """Property pinning _coerce_finish_reasons's own stated contract: no
    matter what shape this field arrives in, construction must succeed and
    the field must end up either None or a real list[str] - never propagate
    a ValidationError, since that is exactly what silently dropped whole
    spans upstream (see module docstring)."""
    attrs = SpanAttributes.model_validate({"gen_ai.response.finish_reasons": value})
    result = attrs.gen_ai_response_finish_reasons
    assert result is None or (
        isinstance(result, list) and all(isinstance(item, str) for item in result)
    )


def test_system_instructions_accepts_a_real_list_unchanged() -> None:
    instructions = [{"type": "text", "content": "You are a helpful assistant."}]
    attrs = SpanAttributes.model_validate({"gen_ai.system_instructions": instructions})
    assert attrs.gen_ai_system_instructions == instructions


def test_system_instructions_accepts_none() -> None:
    attrs = SpanAttributes.model_validate({})
    assert attrs.gen_ai_system_instructions is None


def test_system_instructions_parses_jaeger_style_json_encoded_string() -> None:
    """Jaeger's tag model has no array-of-objects type - the OTLP collector
    encodes a list attribute as a JSON string when writing it as a tag.
    Construction must not raise, and the value must come out as a real
    list of dicts."""
    attrs = SpanAttributes.model_validate(
        {"gen_ai.system_instructions": '[{"type": "text", "content": "Be concise."}]'}
    )
    assert attrs.gen_ai_system_instructions == [{"type": "text", "content": "Be concise."}]


def test_system_instructions_wraps_json_encoded_list_of_plain_strings() -> None:
    """Some instrumentation may hand this field a JSON array of plain
    strings rather than {"type", "content"} objects - each element must be
    wrapped so it still conforms to list[dict[str, str]]."""
    attrs = SpanAttributes.model_validate(
        {"gen_ai.system_instructions": '["Be concise.", "Answer in English."]'}
    )
    assert attrs.gen_ai_system_instructions == [
        {"type": "text", "content": "Be concise."},
        {"type": "text", "content": "Answer in English."},
    ]


def test_system_instructions_wraps_unparseable_string_as_single_instruction() -> None:
    """Must never raise regardless of input shape - a raised exception here
    is exactly what caused the whole span to be silently dropped upstream
    for the analogous finish_reasons field."""
    attrs = SpanAttributes.model_validate(
        {"gen_ai.system_instructions": "You are a helpful assistant."}
    )
    assert attrs.gen_ai_system_instructions == [
        {"type": "text", "content": "You are a helpful assistant."}
    ]


def test_system_instructions_wraps_json_scalar_string() -> None:
    """Valid JSON that isn't a list (e.g. a bare JSON string or number)
    must still be wrapped rather than raising or being dropped."""
    attrs = SpanAttributes.model_validate({"gen_ai.system_instructions": "42"})
    assert attrs.gen_ai_system_instructions == [{"type": "text", "content": "42"}]


@given(
    value=st.one_of(
        st.none(),
        st.text(),
        st.lists(st.dictionaries(st.text(), st.text())),
        st.integers(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.booleans(),
        st.dictionaries(st.text(), st.integers()),
        st.tuples(st.integers(), st.integers()),
    )
)
def test_system_instructions_coercion_never_raises(value: object) -> None:
    """Property pinning _coerce_system_instructions's own stated contract:
    no matter what shape this field arrives in, construction must succeed
    and the field must end up either None or a real list[dict] - never
    propagate a ValidationError, since that is exactly what silently
    dropped whole spans upstream for the analogous finish_reasons field
    (see module docstring)."""
    attrs = SpanAttributes.model_validate({"gen_ai.system_instructions": value})
    result = attrs.gen_ai_system_instructions
    assert result is None or (
        isinstance(result, list) and all(isinstance(item, dict) for item in result)
    )


def test_conversation_id_parses_via_alias() -> None:
    attrs = SpanAttributes.model_validate({"gen_ai.conversation.id": "conv-123"})
    assert attrs.gen_ai_conversation_id == "conv-123"


def test_prompt_name_and_version_parse_via_alias() -> None:
    attrs = SpanAttributes.model_validate(
        {"gen_ai.prompt.name": "summarize", "gen_ai.prompt.version": "3"}
    )
    assert attrs.gen_ai_prompt_name == "summarize"
    assert attrs.gen_ai_prompt_version == "3"


def test_conversation_id_and_prompt_fields_default_to_none() -> None:
    attrs = SpanAttributes.model_validate({})
    assert attrs.gen_ai_conversation_id is None
    assert attrs.gen_ai_prompt_name is None
    assert attrs.gen_ai_prompt_version is None


class TestGenAiProviderNameRename:
    """OTel semconv v1.37.0 renamed gen_ai.system to gen_ai.provider.name.
    Without the model_validator these tests exercise, a span carrying only
    the new name would silently land in extra_attributes instead of the
    typed gen_ai_system field that is_llm_span/LLMSpanAttributes.from_span/
    FilterEngine's client-side filtering all actually read - not a
    hypothetical, since the rename already shipped upstream over a year
    ago and any instrumentation library could adopt it at any time."""

    def test_gen_ai_system_alone_still_works(self) -> None:
        """Control: the existing, canonical name is unaffected."""
        attrs = SpanAttributes.model_validate({"gen_ai.system": "openai"})
        assert attrs.gen_ai_system == "openai"

    def test_gen_ai_provider_name_alone_populates_gen_ai_system(self) -> None:
        attrs = SpanAttributes.model_validate({"gen_ai.provider.name": "openai"})
        assert attrs.gen_ai_system == "openai"

    def test_gen_ai_system_wins_when_both_present(self) -> None:
        attrs = SpanAttributes.model_validate(
            {"gen_ai.system": "anthropic", "gen_ai.provider.name": "openai"}
        )
        assert attrs.gen_ai_system == "anthropic"

    def test_neither_name_present_defaults_to_none(self) -> None:
        attrs = SpanAttributes.model_validate({"gen_ai.request.model": "gpt-4"})
        assert attrs.gen_ai_system is None

    def test_provider_name_used_as_fallback_does_not_duplicate_into_extras(self) -> None:
        """Once consumed as the fallback, the raw key should not also
        reappear in extra_attributes - mirroring
        test_extra_attributes_excludes_typed_fields's shape for the
        canonical name."""
        attrs = SpanAttributes.model_validate({"gen_ai.provider.name": "openai"})
        assert "gen_ai.provider.name" not in attrs.extra_attributes

    def test_kwargs_construction_style_also_works(self) -> None:
        """Backends construct via SpanAttributes(**extra), not
        model_validate(dict) - confirm the same normalization applies to
        that call style too. Typed as dict[str, Any] (matching every real
        backend call site, e.g. datadog.py's
        SpanAttributes(**self._extract_semconv_attributes(...))) rather
        than a narrower inline-literal type mypy would strictly check the
        ** spread against per-field."""
        extra: dict[str, Any] = {"gen_ai.provider.name": "openai", "gen_ai.request.model": "gpt-4"}
        attrs = SpanAttributes(**extra)
        assert attrs.gen_ai_system == "openai"


def test_extra_attributes_surfaces_score_and_evaluation_shaped_fields() -> None:
    attrs = SpanAttributes.model_validate(
        {"gen_ai.system": "openai", "score.relevance": 0.92, "evaluation.passed": True}
    )
    assert attrs.extra_attributes == {"score.relevance": 0.92, "evaluation.passed": True}


def test_extra_attributes_excludes_typed_fields() -> None:
    """gen_ai.system is a typed field - it must not also appear in
    extra_attributes (that would duplicate what to_dict()/typed access
    already expose)."""
    attrs = SpanAttributes.model_validate({"gen_ai.system": "openai", "score.relevance": 0.5})
    assert "gen_ai.system" not in attrs.extra_attributes
    assert "gen_ai_system" not in attrs.extra_attributes


def test_extra_attributes_is_empty_dict_when_none_present() -> None:
    attrs = SpanAttributes.model_validate({"gen_ai.system": "openai"})
    assert attrs.extra_attributes == {}


def test_extra_attributes_excludes_none_values() -> None:
    attrs = SpanAttributes.model_validate({"score.relevance": None})
    assert attrs.extra_attributes == {}

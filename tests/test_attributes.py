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

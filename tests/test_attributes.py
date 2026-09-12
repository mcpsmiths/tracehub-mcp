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
"""

from opentelemetry_mcp.attributes import SpanAttributes


def test_finish_reasons_accepts_a_real_list_unchanged() -> None:
    attrs = SpanAttributes(**{"gen_ai.response.finish_reasons": ["stop", "length"]})
    assert attrs.gen_ai_response_finish_reasons == ["stop", "length"]


def test_finish_reasons_accepts_none() -> None:
    attrs = SpanAttributes()
    assert attrs.gen_ai_response_finish_reasons is None


def test_finish_reasons_parses_jaeger_style_json_encoded_string() -> None:
    """Jaeger's tag model has no array type - the OTLP collector encodes a
    list attribute as a JSON string when writing it as a tag. Construction
    must not raise, and the value must come out as a real list."""
    attrs = SpanAttributes(**{"gen_ai.response.finish_reasons": '["stop"]'})
    assert attrs.gen_ai_response_finish_reasons == ["stop"]


def test_finish_reasons_parses_multi_value_json_encoded_string() -> None:
    attrs = SpanAttributes(**{"gen_ai.response.finish_reasons": '["stop", "length"]'})
    assert attrs.gen_ai_response_finish_reasons == ["stop", "length"]


def test_finish_reasons_falls_back_to_comma_split_for_a_plain_string() -> None:
    attrs = SpanAttributes(**{"gen_ai.response.finish_reasons": "stop,length"})
    assert attrs.gen_ai_response_finish_reasons == ["stop", "length"]


def test_finish_reasons_does_not_raise_on_unparseable_garbage() -> None:
    """Must never raise regardless of input shape - a raised exception here
    is exactly what caused the whole span to be silently dropped upstream."""
    attrs = SpanAttributes(**{"gen_ai.response.finish_reasons": "not json but fine"})
    assert attrs.gen_ai_response_finish_reasons == ["not json but fine"]


def test_legacy_llm_response_finish_reasons_gets_the_same_coercion() -> None:
    attrs = SpanAttributes(**{"llm.response.finish_reasons": '["stop"]'})
    assert attrs.llm_response_finish_reasons == ["stop"]

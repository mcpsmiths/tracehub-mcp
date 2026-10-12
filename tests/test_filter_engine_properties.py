"""Property-based tests for FilterEngine, previously covered by zero tests.

Focused on the parts of filter_engine.py where a plausible input shape (NaN,
reversed BETWEEN bounds, an attribute missing from every span) can silently
produce the wrong boolean instead of raising - failures that unit tests
built from hand-picked examples are unlikely to stumble onto.
"""

import math
from datetime import datetime
from typing import Any
from unittest.mock import patch

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from opentelemetry_mcp.attributes import SpanAttributes
from opentelemetry_mcp.backends.filter_engine import FilterEngine
from opentelemetry_mcp.models import Filter, FilterOperator, FilterType, SpanData, TraceData

ARBITRARY_VALUE = st.one_of(
    st.none(),
    st.text(),
    st.integers(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.booleans(),
)

NON_NONE_VALUE = st.one_of(
    st.text(),
    st.integers(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.booleans(),
)

SINGLE_VALUE_OPERATORS = [
    FilterOperator.EQUALS,
    FilterOperator.NOT_EQUALS,
    FilterOperator.GT,
    FilterOperator.LT,
    FilterOperator.GTE,
    FilterOperator.LTE,
    FilterOperator.CONTAINS,
    FilterOperator.NOT_CONTAINS,
    FilterOperator.STARTS_WITH,
    FilterOperator.ENDS_WITH,
]


def _raw_filter(**fields: Any) -> Filter:
    """Build a Filter with the model-level operand normalisation bypassed.

    The Filter model now rejects/coerces operands that disagree with their
    declared value_type, so a validated Filter can no longer carry e.g. a
    str under NUMBER. The engine's never-raises contract must still hold for
    such un-normalised filters (fail closed), so these property tests feed
    them in via model_construct."""
    return Filter.model_construct(**fields)


def _make_span(index: int, attrs: dict[str, object] | None = None) -> SpanData:
    return SpanData(
        trace_id="t1",
        span_id=f"s{index}",
        operation_name="op",
        service_name="svc",
        start_time=datetime.now(),
        duration_ms=10,
        attributes=SpanAttributes.model_validate(attrs or {}),
    )


def _make_trace(spans_attrs: list[dict[str, object]]) -> TraceData:
    spans = [_make_span(i, a) for i, a in enumerate(spans_attrs)]
    return TraceData(
        trace_id="t1",
        spans=spans,
        start_time=datetime.now(),
        duration_ms=10,
        service_name="svc",
        root_operation="op",
    )


@given(
    value_type=st.sampled_from(list(FilterType)),
    operator=st.sampled_from(SINGLE_VALUE_OPERATORS),
    actual=ARBITRARY_VALUE,
    expected=NON_NONE_VALUE,
)
def test_compare_value_never_raises_for_single_value_operators(
    value_type: FilterType,
    operator: FilterOperator,
    actual: str | int | float | bool | None,
    expected: str | int | float | bool,
) -> None:
    """_compare_value's contract is to catch type-conversion failures and
    return False (filter_engine.py:290-292) rather than propagate - any
    input shape must come back as a plain bool, for every operator/type
    combination, including ones that are semantically mismatched
    (e.g. CONTAINS against value_type=NUMBER)."""
    filt = _raw_filter(field="x", operator=operator, value=expected, value_type=value_type)
    result = FilterEngine._compare_value(actual, filt)
    assert isinstance(result, bool)


@given(
    value_type=st.sampled_from(list(FilterType)),
    actual=ARBITRARY_VALUE,
    values=st.lists(NON_NONE_VALUE, min_size=1, max_size=5),
)
def test_compare_value_never_raises_for_in_operators(
    value_type: FilterType,
    actual: str | int | float | bool | None,
    values: list[str | int | float | bool],
) -> None:
    filt = _raw_filter(field="x", operator=FilterOperator.IN, values=values, value_type=value_type)
    result = FilterEngine._compare_value(actual, filt)
    assert isinstance(result, bool)


@given(
    value_type=st.sampled_from(list(FilterType)),
    actual=ARBITRARY_VALUE,
    values=st.lists(NON_NONE_VALUE, min_size=2, max_size=2),
)
def test_compare_value_never_raises_for_between(
    value_type: FilterType,
    actual: str | int | float | bool | None,
    values: list[str | int | float | bool],
) -> None:
    filt = _raw_filter(
        field="x", operator=FilterOperator.BETWEEN, values=values, value_type=value_type
    )
    result = FilterEngine._compare_value(actual, filt)
    assert isinstance(result, bool)


def test_gt_lte_both_reject_nan_actual() -> None:
    """Regression pin for the NaN fix (filter_engine.py:224-232): before it,
    NaN compared False against everything, so GT and LTE silently AGREED on
    False instead of being complements - a NaN-valued attribute passed
    neither a 'greater than' nor a 'less than or equal to' filter, rather
    than being rejected once like any other malformed numeric value."""
    gt_filter = Filter(
        field="x", operator=FilterOperator.GT, value=5.0, value_type=FilterType.NUMBER
    )
    lte_filter = Filter(
        field="x", operator=FilterOperator.LTE, value=5.0, value_type=FilterType.NUMBER
    )
    assert FilterEngine._compare_value(math.nan, gt_filter) is False
    assert FilterEngine._compare_value(math.nan, lte_filter) is False


def test_gt_lte_both_reject_nan_expected() -> None:
    gt_filter = Filter(
        field="x", operator=FilterOperator.GT, value=math.nan, value_type=FilterType.NUMBER
    )
    lte_filter = Filter(
        field="x", operator=FilterOperator.LTE, value=math.nan, value_type=FilterType.NUMBER
    )
    assert FilterEngine._compare_value(5.0, gt_filter) is False
    assert FilterEngine._compare_value(5.0, lte_filter) is False


@given(
    actual=st.floats(allow_nan=False, allow_infinity=False),
    expected=st.floats(allow_nan=False, allow_infinity=False),
)
def test_gt_lte_are_complements_for_non_nan_numbers(actual: float, expected: float) -> None:
    """For any two real (non-NaN) numbers, exactly one of '>' and '<=' holds.
    This is the property the NaN fix restores for normal inputs - NaN is the
    deliberate, documented exception (both tests above)."""
    gt_filter = Filter(
        field="x", operator=FilterOperator.GT, value=expected, value_type=FilterType.NUMBER
    )
    lte_filter = Filter(
        field="x", operator=FilterOperator.LTE, value=expected, value_type=FilterType.NUMBER
    )
    gt = FilterEngine._compare_value(actual, gt_filter)
    lte = FilterEngine._compare_value(actual, lte_filter)
    assert gt != lte


@given(
    high=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    low=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    actual=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_between_with_reversed_bounds_never_matches(high: float, low: float, actual: float) -> None:
    """BETWEEN (filter_engine.py:251-254) does one chained comparison
    values[0] <= actual <= values[1] with no bounds normalization - if
    values arrive as [high, low] instead of [low, high], no actual value can
    ever satisfy it. This pins that as the current, documented behavior:
    callers must supply BETWEEN values in [low, high] order."""
    assume(high > low)
    filt = Filter(
        field="x", operator=FilterOperator.BETWEEN, values=[high, low], value_type=FilterType.NUMBER
    )
    assert FilterEngine._compare_value(actual, filt) is False


@given(
    operator=st.sampled_from(
        [FilterOperator.NOT_EQUALS, FilterOperator.NOT_CONTAINS, FilterOperator.NOT_IN]
    )
)
def test_negative_operator_does_not_vacuously_match_when_field_absent(
    operator: FilterOperator,
) -> None:
    """_matches_filter short-circuits to False whenever no span carries the
    field at all (filter_engine.py:98-99), before the ALL-vs-ANY branch even
    runs. So a negative ('all spans must satisfy') operator does NOT
    vacuously match a trace where the field never appears - unlike the usual
    'every element of an empty set satisfies any predicate' rule one might
    expect from an ALL-based operator."""
    trace = _make_trace([{}, {}])
    value = None if operator == FilterOperator.NOT_IN else "x"
    values = ["x"] if operator == FilterOperator.NOT_IN else None
    filt = Filter(
        field="custom.marker",
        operator=operator,
        value=value,
        values=values,
        value_type=FilterType.STRING,
    )
    assert FilterEngine._matches_filter(trace, filt) is False


# -- Compile-once expected values ---------------------------------------------
# Audit: src/opentelemetry_mcp/backends/filter_engine.py:FilterEngine._compare_value:per-call-expected-values-rebuild-linear-scan
#
# apply_filters must cost O(items x filters) comparisons plus one O(values)
# normalisation per filter: expected values are converted once per
# apply_filters call and IN/NOT_IN membership is O(1), instead of rebuilding
# and linearly scanning the converted values list for every compared span.

MEMBER_STRING_VALUE = st.one_of(
    st.text(),
    st.integers(min_value=-(2**63), max_value=2**63),
    st.floats(allow_nan=False, allow_infinity=True),
    st.booleans(),
)
MEMBER_NUMBER_VALUE = st.one_of(
    st.integers(min_value=-(2**63), max_value=2**63),
    st.floats(allow_nan=False, allow_infinity=True),
)


def test_compile_runs_once_per_filter_per_apply_filters_call() -> None:
    """F filters over N items compile exactly F times (not N x F), for both
    the SpanData and the TraceData (span-level field) paths."""
    spans = [_make_span(i, {"custom.n": i}) for i in range(100)]
    filters = [
        Filter(
            field="custom.n", operator=FilterOperator.GTE, value=0, value_type=FilterType.NUMBER
        ),
        Filter(
            field="custom.n",
            operator=FilterOperator.IN,
            values=list(range(100)),
            value_type=FilterType.NUMBER,
        ),
        Filter(
            field="service.name",
            operator=FilterOperator.EQUALS,
            value="svc",
            value_type=FilterType.STRING,
        ),
    ]

    with patch.object(FilterEngine, "_compile", wraps=FilterEngine._compile) as compile_spy:
        matched_spans = FilterEngine.apply_filters(spans, filters)
    assert len(matched_spans) == 100
    assert compile_spy.call_count == len(filters)

    trace = _make_trace([{"custom.n": i} for i in range(100)])
    with patch.object(FilterEngine, "_compile", wraps=FilterEngine._compile) as compile_spy:
        matched_traces = FilterEngine.apply_filters([trace], filters)
    assert matched_traces == [trace]
    assert compile_spy.call_count == len(filters)


@settings(deadline=None, derandomize=True)
@given(
    actual=MEMBER_STRING_VALUE,
    values=st.lists(MEMBER_STRING_VALUE, min_size=1, max_size=8),
    negate=st.booleans(),
)
def test_string_in_membership_matches_linear_reference(
    actual: str | int | float | bool, values: list[str | int | float | bool], negate: bool
) -> None:
    """The set-based IN/NOT_IN must agree with the old linear scan's
    semantics: str()-coerce both sides and compare for equality."""
    operator = FilterOperator.NOT_IN if negate else FilterOperator.IN
    filt = _raw_filter(field="x", operator=operator, values=values, value_type=FilterType.STRING)
    reference = any(str(actual) == str(v) for v in values)
    assert FilterEngine._compare_value(actual, filt) is (reference != negate)


@settings(deadline=None, derandomize=True)
@given(
    actual=MEMBER_NUMBER_VALUE,
    values=st.lists(MEMBER_NUMBER_VALUE, min_size=1, max_size=8),
    negate=st.booleans(),
)
def test_number_in_membership_matches_linear_reference(
    actual: int | float, values: list[int | float], negate: bool
) -> None:
    operator = FilterOperator.NOT_IN if negate else FilterOperator.IN
    filt = _raw_filter(field="x", operator=operator, values=values, value_type=FilterType.NUMBER)
    reference = any(float(actual) == float(v) for v in values)
    assert FilterEngine._compare_value(actual, filt) is (reference != negate)


def test_invalid_expected_value_yields_no_matches_without_raising() -> None:
    """A filter whose expected values cannot be converted to its declared
    type fails closed for every item (as the per-call engine did) instead of
    raising, and a BOOLEAN IN stays on the unknown-operator False path."""
    spans = [_make_span(i, {"custom.n": i}) for i in range(5)]

    bad_number_in = _raw_filter(
        field="custom.n",
        operator=FilterOperator.IN,
        values=["not-a-number"],
        value_type=FilterType.NUMBER,
    )
    assert FilterEngine.apply_filters(spans, [bad_number_in]) == []

    boolean_in = _raw_filter(
        field="custom.n", operator=FilterOperator.IN, values=[True], value_type=FilterType.BOOLEAN
    )
    assert FilterEngine.apply_filters(spans, [boolean_in]) == []

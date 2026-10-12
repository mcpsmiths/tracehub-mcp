"""Client-side filter engine for post-query filtering of traces and spans."""

import logging
import math
from dataclasses import dataclass
from typing import Any, TypeVar, overload

from opentelemetry_mcp.models import Filter, FilterOperator, FilterType, SpanData, TraceData

logger = logging.getLogger(__name__)

T = TypeVar("T", TraceData, SpanData)

# Dedup key for the per-span type-conversion WARNING: (field, value_type,
# actual Python type name). One apply_filters call over a 1000-span window
# with one mistyped filter emits one record, not one per span.
_ConversionWarningKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class _CompiledFilter:
    """A Filter whose expected operand(s) have been converted to the declared
    value_type exactly once, so comparing it against N span values costs N
    conversions of the span side only (and O(1) IN/NOT_IN membership) rather
    than rebuilding and linearly scanning the converted values list per span.

    ``invalid`` is set when the expected operand(s) could not be converted;
    such a filter evaluates to False for every item, as the per-call engine
    did, after a single payload-free WARNING at compile time.
    """

    filter: Filter
    invalid: bool = False
    # NUMBER
    expected_number: float | None = None
    number_members: frozenset[float] | None = None
    between: tuple[float, float] | None = None
    # BOOLEAN
    expected_bool: bool | None = None
    # STRING
    expected_str: str = ""
    str_members: frozenset[str] | None = None


class FilterEngine:
    """Client-side filter engine for applying filters to traces and spans."""

    @overload
    @staticmethod
    def apply_filters(items: list[TraceData], filters: list[Filter]) -> list[TraceData]: ...

    @overload
    @staticmethod
    def apply_filters(items: list[SpanData], filters: list[Filter]) -> list[SpanData]: ...

    @staticmethod
    def apply_filters(
        items: list[TraceData] | list[SpanData], filters: list[Filter]
    ) -> list[TraceData] | list[SpanData]:
        """Apply all filters to traces or spans and return matching items.

        Args:
            items: List of traces or spans to filter
            filters: List of Filter conditions (combined with AND logic)

        Returns:
            Filtered list of items matching all conditions
        """
        if not filters:
            return items

        # Convert every filter's expected operand(s) exactly once per call;
        # the per-item loop below only ever converts the item's own value.
        compiled_filters = [FilterEngine._compile(f) for f in filters]
        # Call-local accumulator so a mistyped filter warns once per call.
        warned: set[_ConversionWarningKey] = set()

        filtered_items: list[TraceData] | list[SpanData]
        if isinstance(items[0] if items else None, TraceData):
            filtered_items = []
            for item in items:
                if FilterEngine._matches_all_filters(item, compiled_filters, warned):
                    filtered_items.append(item)  # type: ignore[arg-type]
        else:
            filtered_items = []
            for item in items:
                if FilterEngine._matches_all_filters(item, compiled_filters, warned):
                    filtered_items.append(item)  # type: ignore[arg-type]

        logger.debug(
            f"Client-side filtering: {len(items)} items -> {len(filtered_items)} after applying {len(filters)} filters"
        )
        return filtered_items

    @staticmethod
    def _matches_all_filters(
        item: TraceData | SpanData,
        filters: list[_CompiledFilter] | list[Filter],
        warned: set[_ConversionWarningKey] | None = None,
    ) -> bool:
        """Check if an item (trace or span) matches all filters (AND logic).

        Args:
            item: Trace or span to check
            filters: List of (compiled) filters to apply
            warned: Per-call dedup set for type-conversion warnings

        Returns:
            True if item matches all filters
        """
        for filter_obj in filters:
            if not FilterEngine._matches_filter(item, filter_obj, warned):
                return False
        return True

    @staticmethod
    def _matches_filter(
        item: TraceData | SpanData,
        filter_obj: Filter | _CompiledFilter,
        warned: set[_ConversionWarningKey] | None = None,
    ) -> bool:
        """Check if an item (trace or span) matches a single filter.

        Args:
            item: Trace or span to check
            filter_obj: Filter to apply (a raw Filter is compiled on the spot)
            warned: Per-call dedup set for type-conversion warnings

        Returns:
            True if item matches the filter
        """
        compiled = (
            filter_obj
            if isinstance(filter_obj, _CompiledFilter)
            else FilterEngine._compile(filter_obj)
        )
        field = compiled.filter.field
        operator = compiled.filter.operator

        # Get values from item (may check multiple spans for traces, single span for spans)
        values = FilterEngine._get_field_values(item, field)

        # Handle existence operators
        if operator == FilterOperator.EXISTS:
            return len(values) > 0
        if operator == FilterOperator.NOT_EXISTS:
            return len(values) == 0

        # Handle empty values
        if not values:
            return False

        # For negative operators (NOT_EQUALS, NOT_IN, NOT_CONTAINS), require ALL values to satisfy the predicate
        # For positive operators, check if ANY value matches (OR logic across spans)
        negative_ops = {
            FilterOperator.NOT_EQUALS,
            FilterOperator.NOT_CONTAINS,
            FilterOperator.NOT_IN,
        }

        if operator in negative_ops:
            return all(FilterEngine._compare_compiled(value, compiled, warned) for value in values)

        return any(FilterEngine._compare_compiled(value, compiled, warned) for value in values)

    @staticmethod
    def _get_field_values(item: TraceData | SpanData, field: str) -> list[Any]:
        """Extract values for a field from trace/span.

        For TraceData: Supports both trace-level fields and span-level fields (checks all spans).
        For SpanData: Extracts field directly from the span.

        Args:
            item: Trace or span to extract from
            field: Field name in dotted notation

        Returns:
            List of values found (may be empty, or contain multiple values from trace spans)
        """
        values: list[Any] = []

        # Handle SpanData
        if isinstance(item, SpanData):
            span_values = FilterEngine._get_span_field_values(item, field)
            return [v for v in span_values if v is not None]

        # Handle TraceData
        trace = item

        # Check trace-level fields first
        if field == "trace_id":
            values.append(trace.trace_id)
        elif field == "service.name":
            values.append(trace.service_name)
        elif field == "name" or field == "operation_name":
            values.append(trace.root_operation)
        elif field == "duration":
            values.append(trace.duration_ms)
        elif field == "status":
            values.append(trace.status)
        elif field == "span_count":
            values.append(len(trace.spans))
        elif field == "llm_span_count":
            values.append(len(trace.llm_spans))
        elif field == "total_tokens":
            values.append(trace.total_llm_tokens)
        elif field == "has_errors":
            values.append(trace.has_errors)
        else:
            # Check span-level attributes (collect from all spans)
            for span in trace.spans:
                span_values = FilterEngine._get_span_field_values(span, field)
                values.extend(span_values)

        return [v for v in values if v is not None]

    @staticmethod
    def _get_span_field_values(span: Any, field: str) -> list[Any]:
        """Extract values for a field from a span.

        Args:
            span: SpanData object
            field: Field name in dotted notation

        Returns:
            List of values found in this span
        """
        values: list[Any] = []

        # Check span-level fields
        if field == "span_id":
            values.append(span.span_id)
        elif field == "parent_span_id":
            if span.parent_span_id:
                values.append(span.parent_span_id)
        elif field == "service.name":
            values.append(span.service_name)
        elif field == "name" or field == "operation_name":
            values.append(span.operation_name)
        elif field == "duration":
            values.append(span.duration_ms)
        elif field == "status":
            values.append(span.status)
        else:
            # Check span attributes using dotted notation
            attr_value = span.attributes.get(field)
            if attr_value is not None:
                values.append(attr_value)

        return [v for v in values if v is not None]

    @staticmethod
    def _compile(filter_obj: Filter) -> _CompiledFilter:
        """Convert a Filter's expected operand(s) to its declared value_type once.

        NUMBER: value -> float (NaN rejected), values -> frozenset[float] plus
        the BETWEEN bounds when exactly two were given; BOOLEAN: value -> bool
        (IN/NOT_IN stay on the unknown-operator path); STRING: value -> str,
        values -> frozenset[str]. A conversion failure marks the filter
        invalid (False for every item) with one WARNING that names the filter,
        never the operand.
        """
        value = filter_obj.value
        values = filter_obj.values
        try:
            if filter_obj.value_type == FilterType.NUMBER:
                expected_number = float(value) if value is not None else None
                # NaN compares False against everything, including itself, so
                # GT/LTE (and LT/GTE) would silently agree on False instead of
                # being complements - reject it once here like any other
                # malformed numeric operand.
                if expected_number is not None and math.isnan(expected_number):
                    raise ValueError("NaN is not a comparable numeric value")
                numbers = [float(v) for v in values] if values is not None else None
                between: tuple[float, float] | None = None
                if (
                    filter_obj.operator == FilterOperator.BETWEEN
                    and numbers is not None
                    and len(numbers) == 2
                ):
                    between = (numbers[0], numbers[1])
                return _CompiledFilter(
                    filter=filter_obj,
                    expected_number=expected_number,
                    number_members=frozenset(numbers) if numbers is not None else None,
                    between=between,
                )
            if filter_obj.value_type == FilterType.BOOLEAN:
                return _CompiledFilter(
                    filter=filter_obj,
                    expected_bool=bool(value) if value is not None else None,
                )
            return _CompiledFilter(
                filter=filter_obj,
                expected_str=str(value) if value is not None else "",
                str_members=frozenset(str(v) for v in values) if values is not None else None,
            )
        except (ValueError, TypeError):
            logger.warning(
                "Type conversion failed for the expected value(s) of filter on field %r "
                "(operator=%s, value_type=%s); the filter matches nothing",
                filter_obj.field,
                filter_obj.operator.value,
                filter_obj.value_type.value,
            )
            return _CompiledFilter(filter=filter_obj, invalid=True)

    @staticmethod
    def _compare_value(actual: Any, filter_obj: Filter) -> bool:
        """Compare an actual value against a filter condition.

        Thin adapter over _compile + _compare_compiled for callers holding a
        raw Filter; apply_filters compiles each filter once up front instead.

        Args:
            actual: Actual value from trace/span
            filter_obj: Filter with operator and expected value(s)

        Returns:
            True if actual value matches the filter condition
        """
        return FilterEngine._compare_compiled(actual, FilterEngine._compile(filter_obj))

    @staticmethod
    def _compare_compiled(
        actual: Any,
        compiled: _CompiledFilter,
        warned: set[_ConversionWarningKey] | None = None,
    ) -> bool:
        """Compare an actual value against a compiled filter condition.

        Only ``actual`` is converted here; the expected side was converted by
        _compile. Any conversion failure returns False rather than raising,
        after a WARNING that names the filter and the offending value's
        Python type - never the value itself, which is span data (a prompt,
        a message list) that PRIVACY.md promises stays out of the server's
        own logs. ``warned`` deduplicates that WARNING per apply_filters
        call; when None (direct callers), every failure warns.
        """
        if compiled.invalid:
            return False

        filter_obj = compiled.filter
        operator = filter_obj.operator

        try:
            if filter_obj.value_type == FilterType.NUMBER:
                actual_num = float(actual)
                # See _compile for why NaN is rejected rather than compared.
                if math.isnan(actual_num):
                    raise ValueError("NaN is not a comparable numeric value")
                expected_num = compiled.expected_number

                # Apply numeric operators
                if operator == FilterOperator.EQUALS:
                    return actual_num == expected_num
                elif operator == FilterOperator.NOT_EQUALS:
                    return actual_num != expected_num
                elif operator == FilterOperator.GT:
                    return actual_num > expected_num if expected_num is not None else False
                elif operator == FilterOperator.LT:
                    return actual_num < expected_num if expected_num is not None else False
                elif operator == FilterOperator.GTE:
                    return actual_num >= expected_num if expected_num is not None else False
                elif operator == FilterOperator.LTE:
                    return actual_num <= expected_num if expected_num is not None else False
                elif operator == FilterOperator.IN:
                    return actual_num in (compiled.number_members or frozenset())
                elif operator == FilterOperator.NOT_IN:
                    return actual_num not in (compiled.number_members or frozenset())
                elif operator == FilterOperator.BETWEEN:
                    if compiled.between is not None:
                        low, high = compiled.between
                        return low <= actual_num <= high
                    return False

            elif filter_obj.value_type == FilterType.BOOLEAN:
                actual_bool = bool(actual)
                expected_bool = compiled.expected_bool

                if operator == FilterOperator.EQUALS:
                    return actual_bool == expected_bool
                elif operator == FilterOperator.NOT_EQUALS:
                    return actual_bool != expected_bool

            else:  # STRING
                actual_str = str(actual)
                expected_str = compiled.expected_str

                # String operators
                if operator == FilterOperator.EQUALS:
                    return actual_str == expected_str
                elif operator == FilterOperator.NOT_EQUALS:
                    return actual_str != expected_str
                elif operator == FilterOperator.CONTAINS:
                    return expected_str in actual_str
                elif operator == FilterOperator.NOT_CONTAINS:
                    return expected_str not in actual_str
                elif operator == FilterOperator.STARTS_WITH:
                    return actual_str.startswith(expected_str)
                elif operator == FilterOperator.ENDS_WITH:
                    return actual_str.endswith(expected_str)
                elif operator == FilterOperator.IN:
                    return actual_str in (compiled.str_members or frozenset())
                elif operator == FilterOperator.NOT_IN:
                    return actual_str not in (compiled.str_members or frozenset())

        except (ValueError, TypeError):
            actual_type = type(actual).__name__
            key: _ConversionWarningKey = (
                filter_obj.field,
                filter_obj.value_type.value,
                actual_type,
            )
            if warned is None or key not in warned:
                if warned is not None:
                    warned.add(key)
                logger.warning(
                    "Type conversion failed for field %r (operator=%s, value_type=%s, "
                    "actual_type=%s); the value does not match this filter",
                    filter_obj.field,
                    operator.value,
                    filter_obj.value_type.value,
                    actual_type,
                )
            return False

        logger.warning(f"Unknown operator: {operator}")
        return False

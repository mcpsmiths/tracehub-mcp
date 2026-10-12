"""Tests for safe_path_segment, the shared guard for interpolating an
untrusted identifier into exactly one URL path segment.

Audit: src/opentelemetry_mcp/backends/sentry.py:SentryBackend.get_trace:dot-segment-trace-id
"""

import pytest

from opentelemetry_mcp.backends.url_path import MAX_PATH_SEGMENT_CHARS, safe_path_segment


@pytest.mark.parametrize("value", ["", ".", ".."])
def test_dot_and_empty_segments_are_rejected(value: str) -> None:
    """quote() leaves "." unencoded (RFC 3986 unreserved), so these survive
    percent-encoding as whole segments and httpx's base_url merge then
    removes them - ".." retargets the request at the parent path."""
    with pytest.raises(ValueError, match="Invalid trace_id"):
        safe_path_segment(value, "trace_id")


def test_slash_is_percent_encoded() -> None:
    assert safe_path_segment("a/b") == "a%2Fb"


def test_embedded_dot_segments_are_neutralised_by_encoding() -> None:
    # Not a whole-segment dot reference, so it stays inside the one segment.
    assert safe_path_segment("evil/../x") == "evil%2F..%2Fx"


def test_plain_identifier_passes_through_unchanged() -> None:
    assert safe_path_segment("abc123") == "abc123"


def test_length_limit() -> None:
    assert safe_path_segment("x" * MAX_PATH_SEGMENT_CHARS) == "x" * MAX_PATH_SEGMENT_CHARS
    with pytest.raises(ValueError, match="Invalid identifier"):
        safe_path_segment("x" * (MAX_PATH_SEGMENT_CHARS + 1))


def test_error_names_what_was_invalid_and_not_the_overlong_value() -> None:
    long_value = "y" * (MAX_PATH_SEGMENT_CHARS + 1)
    with pytest.raises(ValueError) as excinfo:
        safe_path_segment(long_value, "service_name")
    assert "service_name" in str(excinfo.value)
    assert long_value not in str(excinfo.value)

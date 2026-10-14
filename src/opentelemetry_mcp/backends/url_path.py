"""Safe interpolation of an untrusted identifier into one URL path segment.

Every backend splices a trace id (a tool argument, or an id read from the
backend's own search rows) or a service name into a request *path*, e.g.
``/api/traces/{trace_id}``. ``urllib.parse.quote(value, safe="")`` encodes
``/`` and ``?`` but leaves ``.`` unencoded (RFC 3986 unreserved), so the
whole-segment values ``"."`` and ``".."`` - and the empty segment - survive
percent-encoding intact, and httpx removes dot segments when it merges the
relative request path onto the client's ``base_url``: ``".."`` collapses the
segment *and its parent*, so ``/api/traces/..`` is sent as ``/api/``. A tool
argument could thereby select which same-host endpoint the operator's
credential is presented to. ``safe_path_segment`` rejects those values (and
anything implausibly long for an identifier) before any URL is built.
"""

from urllib.parse import quote

# Generous for any trace id / service name / result id this server handles
# (OTel trace ids are 32 hex characters); bounds the URL a caller can make
# the server emit under the operator's credential.
MAX_PATH_SEGMENT_CHARS = 256

# Whole-segment values httpx's base_url merge removes or collapses.
_DOT_SEGMENTS = frozenset({"", ".", ".."})


def safe_path_segment(value: str, what: str = "identifier") -> str:
    """Percent-encode ``value`` for use as exactly one URL path segment.

    Args:
        value: The untrusted identifier to interpolate
        what: Name of the identifier for the error message (e.g. "trace_id")

    Returns:
        ``value`` percent-encoded with ``safe=""`` (so it can only ever vary
        the one segment it is spliced into)

    Raises:
        ValueError: If ``value`` is ``""``, ``"."`` or ``".."`` (which httpx
            would collapse into a different endpoint) or exceeds
            MAX_PATH_SEGMENT_CHARS
    """
    if value in _DOT_SEGMENTS:
        raise ValueError(f"Invalid {what}: {value!r} is not a valid URL path segment")
    if len(value) > MAX_PATH_SEGMENT_CHARS:
        raise ValueError(
            f"Invalid {what}: {len(value)} characters exceeds the "
            f"{MAX_PATH_SEGMENT_CHARS}-character limit for a URL path segment"
        )
    return quote(value, safe="")

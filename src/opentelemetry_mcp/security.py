"""Shared host-classification helpers for SSRF-relevant URL validation.

Two callers with different trust levels share these helpers:

- BackendConfig.validate_url checks the operator-set BACKEND_URL once at
  startup. Ordinary private/loopback addresses (a self-hosted Jaeger/Tempo on
  a VPC or Docker Compose network) are a normal deployment and stay allowed;
  only cloud instance-metadata endpoints - never a legitimate trace-backend
  location - are refused.
- backends.base._RetryingTransport checks every hop httpx is about to send,
  including redirect targets *chosen by the backend's response*. That input
  is not operator-controlled, so the classifier must judge a host by the
  address the platform resolver will actually connect to: libc's inet_aton
  accepts decimal (2852039166), hex (0xa9fea9fe), octal and short-form
  (169.254.43518) IPv4 spellings that httpx keeps as plain hostnames, and all
  of them reach 169.254.169.254. is_cloud_metadata_host canonicalises those
  spellings before the denylist comparison. DNS aliases of a metadata address
  are not resolved here - the transport's host pin (allowed_host) is the
  layer that refuses them, by refusing every host but the configured one.

The module also owns the URL-userinfo redaction helpers. BackendConfig
rejects `user:password@host` in BACKEND_URL outright, so a validated config
never carries a credential in its URL; redact_url_userinfo /
redact_userinfo_in_text / UserinfoRedactingFilter are the defence in depth
for every sink that stringifies a URL or a backend exception (httpx's
HTTPStatusError text embeds the request URL verbatim) in case one does.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from collections.abc import Mapping
from typing import overload

import httpx

# Cloud instance-metadata endpoints. 169.254.169.254 is shared by AWS, Azure,
# GCP, DigitalOcean, and Oracle Cloud; the others are provider-specific
# variants (AWS ECS task metadata, AWS IMDSv2 over IPv6, Alibaba Cloud, the
# Azure WireServer that every Azure VM also exposes).
_CLOUD_METADATA_IPS = frozenset(
    {
        "169.254.169.254",
        "169.254.170.2",
        "fd00:ec2::254",
        "100.100.100.200",
        "168.63.129.16",
    }
)

_CLOUD_METADATA_HOSTNAMES = frozenset({"metadata.google.internal"})


def _normalize_host(host: str) -> str:
    """Strip the enclosing brackets pydantic's HttpUrl.host keeps on IPv6
    literals (e.g. "[fd00:ec2::254]") - ipaddress.ip_address rejects them."""
    host = host.strip().rstrip(".")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host


def _canonical_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """The address the platform resolver would connect `host` to, or None
    for a real hostname.

    ipaddress.ip_address accepts only the canonical dotted-quad / RFC 4291
    spellings; libc (inet_aton, and getaddrinfo for anything that is not a
    canonical literal) also accepts decimal, hex, octal and 1-3 part IPv4
    forms. Canonicalise exactly as the resolver will, so a denylist
    comparison cannot be sidestepped by re-spelling the same address.
    """
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    try:
        return ipaddress.IPv4Address(socket.inet_aton(host))
    except OSError:
        return None


def is_cloud_metadata_host(host: str | None) -> bool:
    """True if `host` is a known cloud instance-metadata address.

    Handles IP literals in every spelling the resolver accepts (see
    _canonical_ip), IPv4-mapped IPv6 forms like `::ffff:169.254.169.254`,
    and the one metadata hostname (GCP) that isn't reachable via a fixed IP
    alone.
    """
    if not host:
        return False
    host = _normalize_host(host).lower()
    addr = _canonical_ip(host)
    if addr is None:
        return host in _CLOUD_METADATA_HOSTNAMES
    if isinstance(addr, ipaddress.IPv6Address):
        mapped = addr.ipv4_mapped
        if mapped is not None:
            addr = mapped
    return str(addr) in _CLOUD_METADATA_IPS


def is_loopback_host(host: str | None) -> bool:
    """True if `host` is loopback (127.0.0.0/8, ::1, or "localhost").

    Broader than an exact-string match against "127.0.0.1" - any address in
    127.0.0.0/8 is equally loopback-safe.
    """
    if not host:
        return False
    host = _normalize_host(host).lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


# "<scheme>://" then a run with no "/", "?", "#" or whitespace, then "@": the
# userinfo component of a URL embedded anywhere in free text. "@" itself is
# NOT excluded from the run - WHATWG (and so pydantic's HttpUrl) accept a
# literal "@" inside the password (`https://u:p@ss@h.test/` is host h.test,
# password p%40ss), so the match must run greedily to the LAST "@" before
# the authority ends; stopping at the first would leave the password's tail.
# The host and path after the "@" are kept so an error stays actionable.
_URL_USERINFO_RE = re.compile(r"(?<=://)[^/?#\s]+@")
_REDACTED_USERINFO = "[REDACTED]@"


def redact_url_userinfo(url: object) -> str:
    """str(url) with any user:password@ component removed.

    Accepts anything whose str() is a URL (pydantic HttpUrl, httpx.URL, str).
    A string httpx cannot parse falls back to the textual redaction so the
    result is never the raw input.
    """
    text = str(url)
    try:
        return str(httpx.URL(text).copy_with(username=None, password=None))
    except httpx.InvalidURL:
        return _URL_USERINFO_RE.sub(_REDACTED_USERINFO, text)


@overload
def redact_userinfo_in_text(text: str) -> str: ...


@overload
def redact_userinfo_in_text(text: None) -> None: ...


def redact_userinfo_in_text(text: str | None) -> str | None:
    """Replace the userinfo of every URL embedded in `text` with
    "[REDACTED]@", keeping scheme, host and path. For exception messages
    (httpx error text, pydantic's input_value echo) and log lines."""
    if text is None:
        return None
    return _URL_USERINFO_RE.sub(_REDACTED_USERINFO, text)


def _safe_redacted_text(value: object) -> str:
    """str(value) with URL userinfo redacted. A value whose __str__ raises
    becomes a type-name placeholder: this runs inside a logging filter,
    which must never raise."""
    try:
        text = str(value)
    except Exception:
        text = f"<unprintable {type(value).__name__}>"
    return redact_userinfo_in_text(text)


def _redact_log_args(args: object) -> tuple[object, ...] | Mapping[str, object] | None:
    """record.args with every value redacted individually, preserving the
    tuple-or-mapping shape %-formatting expects."""
    if args is None:
        return None
    if isinstance(args, Mapping):
        return {key: _safe_redacted_text(value) for key, value in args.items()}
    if isinstance(args, tuple):
        return tuple(_safe_redacted_text(value) for value in args)
    return (_safe_redacted_text(args),)


class UserinfoRedactingFilter(logging.Filter):
    """Handler-level last line of defence: rewrites a record so neither its
    message nor its formatted traceback carries a URL userinfo component.

    Attach to HANDLERS (server.py's root stderr handler), not to the root
    logger - logger-level filters are never consulted for records that
    propagate up from child loggers (httpx, opentelemetry_mcp.*). Loggers
    that do not propagate at all are outside its reach: fastmcp's installs
    its own RichHandlers with propagate=False at import, and uvicorn's
    default log config does the same - server._handle_tool_error is what
    keeps userinfo out of the records fastmcp itself emits. Creation-time
    redaction at every known sink remains the primary control; this
    catches the sinks nobody named, such as a third-party logger echoing an
    exception. Mutating the record is how logging filters work: every
    handler after this one sees the redacted record too.

    Never raises. Handler.handle() runs filters OUTSIDE emit()'s
    try/except, so an exception here would propagate into the logging
    caller: a malformed %-format call anywhere (uvicorn, starlette,
    httpcore, the OTel SDK, this code) would then crash the request or
    startup path it occurs in, where without the filter Handler.handleError
    would merely print a "--- Logging error ---" notice.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        self._redact_message(record)
        self._redact_traceback(record)
        return True

    @staticmethod
    def _redact_message(record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            # Mismatched %-format arity or type. The Formatter will fail the
            # same way inside emit(), and handleError then prints
            # "Message: %r / Arguments: %s" - so redact the format string and
            # each argument separately and let that notice be credential-free.
            record.msg = _safe_redacted_text(record.msg)
            record.args = _redact_log_args(record.args)
            return
        redacted = redact_userinfo_in_text(message)
        if redacted != message:
            record.msg = redacted
            record.args = None

    @staticmethod
    def _redact_traceback(record: logging.LogRecord) -> None:
        if not record.exc_info:
            return
        try:
            exc_text = record.exc_text or logging.Formatter().formatException(record.exc_info)
        except Exception:
            # A traceback that cannot be rendered cannot be proven
            # credential-free, and raising here would hit the caller: drop it.
            exc_text = "<traceback unavailable>"
        record.exc_text = redact_userinfo_in_text(exc_text)
        # Formatter.format() re-renders exc_info only when exc_text is
        # unset, but other formatters may not; drop exc_info so the
        # redacted cache is the only thing that can print.
        record.exc_info = None

"""Shared host-classification helpers for SSRF-relevant URL validation.

BACKEND_URL is operator-set once at server startup, not per-request attacker
input, so this module deliberately stays narrow: it distinguishes cloud
instance-metadata endpoints (never a legitimate trace-backend location) from
ordinary private/loopback addresses (a completely normal self-hosted
Jaeger/Tempo deployment). It does not attempt full runtime SSRF prevention
(no DNS-pinning, no per-request re-validation, no protection against
decimal/octal/hex-encoded IP literals that require a resolved connection to
catch) - that class of protection is overkill for an operator-configured
value and is intentionally out of scope.
"""

from __future__ import annotations

import ipaddress

# Cloud instance-metadata endpoints. 169.254.169.254 is shared by AWS, Azure,
# GCP, DigitalOcean, and Oracle Cloud; the others are provider-specific
# variants (AWS ECS task metadata, AWS IMDSv2 over IPv6, Alibaba Cloud).
_CLOUD_METADATA_IPS = frozenset(
    {
        "169.254.169.254",
        "169.254.170.2",
        "fd00:ec2::254",
        "100.100.100.200",
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


def is_cloud_metadata_host(host: str | None) -> bool:
    """True if `host` is a known cloud instance-metadata address.

    Handles bare IP literals (including IPv4-mapped IPv6 forms like
    `::ffff:169.254.169.254`) and the one metadata hostname (GCP) that isn't
    reachable via a fixed IP alone.
    """
    if not host:
        return False
    host = _normalize_host(host).lower()
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
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

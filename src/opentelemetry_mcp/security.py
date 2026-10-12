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
"""

from __future__ import annotations

import ipaddress
import socket

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

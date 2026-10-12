"""Tests for the host-classification helpers in security.py."""

import pytest

from opentelemetry_mcp.security import is_cloud_metadata_host, is_loopback_host


class TestIsCloudMetadataHost:
    @pytest.mark.parametrize(
        "host",
        [
            "169.254.169.254",
            "169.254.170.2",
            "fd00:ec2::254",
            "[fd00:ec2::254]",
            "100.100.100.200",
            "metadata.google.internal",
            "METADATA.GOOGLE.INTERNAL.",
            "::ffff:169.254.169.254",
        ],
    )
    def test_canonical_spellings_are_blocked(self, host: str) -> None:
        assert is_cloud_metadata_host(host) is True

    @pytest.mark.parametrize(
        "host",
        [
            "2852039166",  # decimal
            "0xa9fea9fe",  # hex
            "169.254.43518",  # short form (3 parts)
            "0xa9.0xfe.0xa9.0xfe",  # hex-dotted
            "0251.0376.0251.0376",  # octal
            "168.63.129.16",  # Azure WireServer
        ],
    )
    def test_encoded_spellings_and_unlisted_metadata_hosts_are_blocked(self, host: str) -> None:
        """httpx keeps these as plain hostnames and libc's resolver maps them
        to 169.254.169.254 - a backend-chosen redirect hop must be judged by
        the address the resolver will actually connect to."""
        assert is_cloud_metadata_host(host) is True

    @pytest.mark.parametrize(
        "host",
        ["10.0.0.5", "backend.test", "localhost", "127.0.0.1", "::1", "192.168.1.20", "", None],
    )
    def test_ordinary_hosts_are_not_blocked(self, host: str | None) -> None:
        assert is_cloud_metadata_host(host) is False


class TestIsLoopbackHost:
    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "127.0.0.5", "::1", "[::1]"])
    def test_loopback(self, host: str) -> None:
        assert is_loopback_host(host) is True

    @pytest.mark.parametrize("host", ["10.0.0.5", "backend.test", "", None])
    def test_not_loopback(self, host: str | None) -> None:
        assert is_loopback_host(host) is False

"""Tests for the list_services / get_service_operations tool implementations.

The backend is mocked directly (``AsyncMock(spec=BaseBackend)``) since these
tool functions only call two BaseBackend methods and then reshape the result
into JSON - there is no HTTP layer to fake here (that belongs to the
backend-level tests, e.g. test_datadog.py).
"""

import json
from unittest.mock import AsyncMock

from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.tools.services import get_service_operations, list_services


def _backend() -> AsyncMock:
    return AsyncMock(spec=BaseBackend)


class TestListServices:
    """Test list_services()."""

    async def test_happy_path_returns_count_and_sorted_services(self) -> None:
        backend = _backend()
        backend.list_services.return_value = ["zebra-service", "alpha-service", "beta-service"]

        result = json.loads(await list_services(backend))

        assert result == {
            "count": 3,
            "services": ["alpha-service", "beta-service", "zebra-service"],
        }

    async def test_input_order_does_not_affect_output_order(self) -> None:
        """The tool must sort the backend's result, not merely echo it -
        exercise this with an already-reverse-sorted input to prove the
        sort is real rather than accidentally matching input order."""
        backend = _backend()
        backend.list_services.return_value = ["c", "b", "a"]

        result = json.loads(await list_services(backend))

        assert result["services"] == ["a", "b", "c"]

    async def test_empty_service_list(self) -> None:
        backend = _backend()
        backend.list_services.return_value = []

        result = json.loads(await list_services(backend))

        assert result == {"count": 0, "services": []}

    async def test_duplicate_service_names_are_not_deduplicated(self) -> None:
        """sorted() does not dedupe - count and the returned list both
        reflect the raw (possibly duplicated) backend result as-is."""
        backend = _backend()
        backend.list_services.return_value = ["svc-a", "svc-a", "svc-b"]

        result = json.loads(await list_services(backend))

        assert result["count"] == 3
        assert result["services"] == ["svc-a", "svc-a", "svc-b"]

    async def test_backend_exception_is_caught_and_returns_error_json(self) -> None:
        backend = _backend()
        backend.list_services.side_effect = RuntimeError("connection refused")

        output = await list_services(backend)
        result = json.loads(output)

        assert result == {"error": "Failed to list services: connection refused"}

    async def test_backend_exception_does_not_propagate(self) -> None:
        """The tool must never raise - callers only ever see JSON."""
        backend = _backend()
        backend.list_services.side_effect = ValueError("backend unreachable")

        # Must not raise.
        output = await list_services(backend)

        assert json.loads(output)["error"].startswith("Failed to list services:")


class TestGetServiceOperations:
    """Test get_service_operations()."""

    async def test_happy_path_returns_service_name_count_and_sorted_operations(self) -> None:
        backend = _backend()
        backend.get_service_operations.return_value = ["update", "create", "delete"]

        result = json.loads(await get_service_operations(backend, "checkout-service"))

        assert result == {
            "service_name": "checkout-service",
            "count": 3,
            "operations": ["create", "delete", "update"],
        }
        backend.get_service_operations.assert_awaited_once_with("checkout-service")

    async def test_empty_operations_list(self) -> None:
        backend = _backend()
        backend.get_service_operations.return_value = []

        result = json.loads(await get_service_operations(backend, "idle-service"))

        assert result == {"service_name": "idle-service", "count": 0, "operations": []}

    async def test_duplicate_operation_names_are_not_deduplicated(self) -> None:
        backend = _backend()
        backend.get_service_operations.return_value = ["op-a", "op-a"]

        result = json.loads(await get_service_operations(backend, "svc"))

        assert result["count"] == 2
        assert result["operations"] == ["op-a", "op-a"]

    async def test_backend_exception_is_caught_and_includes_service_name(self) -> None:
        backend = _backend()
        backend.get_service_operations.side_effect = RuntimeError("timeout")

        result = json.loads(await get_service_operations(backend, "payments-service"))

        assert result == {"error": "Failed to get operations for service payments-service: timeout"}

    async def test_backend_exception_does_not_propagate(self) -> None:
        backend = _backend()
        backend.get_service_operations.side_effect = ConnectionError("dropped")

        output = await get_service_operations(backend, "svc")

        assert json.loads(output)["error"].startswith("Failed to get operations for service svc:")

    async def test_unknown_service_name_still_reflected_in_error_message(self) -> None:
        """A service name the backend doesn't recognize isn't validated here -
        it's passed straight through, including into the error message if
        the backend raises for it."""
        backend = _backend()
        backend.get_service_operations.side_effect = KeyError("no such service")

        result = json.loads(await get_service_operations(backend, "does-not-exist"))

        assert "does-not-exist" in result["error"]

"""List services and operations tool implementation."""

import json

from opentelemetry_mcp.backends.base import BaseBackend


async def list_services(backend: BaseBackend) -> str:
    """List all available services in the backend.

    Args:
        backend: Backend instance to query

    Returns:
        JSON string with list of service names
    """
    services = await backend.list_services()

    result = {"count": len(services), "services": sorted(services)}

    return json.dumps(result, indent=2)


async def get_service_operations(backend: BaseBackend, service_name: str) -> str:
    """Get all operations for a specific service.

    Args:
        backend: Backend instance to query
        service_name: Service name to query

    Returns:
        JSON string with list of operation names
    """
    operations = await backend.get_service_operations(service_name)

    result = {
        "service_name": service_name,
        "count": len(operations),
        "operations": sorted(operations),
    }

    return json.dumps(result, indent=2)

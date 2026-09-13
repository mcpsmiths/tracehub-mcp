"""Thin launcher for the published ``tracehub-mcp`` PyPI package.

This file is intentionally NOT the server implementation. The real server
(all backends, tools, and business logic) lives entirely in the
``tracehub-mcp`` package published on PyPI:
https://pypi.org/project/tracehub-mcp/

``uv run`` (invoked by this bundle's manifest.json) resolves the dependency
declared in the sibling ``pyproject.toml`` and installs the real package
from PyPI into an ephemeral environment before this script runs.

This stub exists only because the MCPB manifest schema's
``server.entry_point`` field requires a path to a bundled Python file, even
for the ``uv`` server type. No tracehub-mcp source is vendored here.
"""

from opentelemetry_mcp.server import main

if __name__ == "__main__":
    main()

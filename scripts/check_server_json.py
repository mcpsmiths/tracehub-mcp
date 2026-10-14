"""Assert server.json's identity before it is published to the MCP Registry.

mcp-registry-publish.yml publishes whatever server.json the checked-out ref
carries. The registry's github-oidc grant for io.github.mcpsmiths/* is
repo-scoped, so the last in-repo point where "the entry published for version
X is this project's own distribution at version X" can be enforced is right
before `mcp-publisher publish`. This gate fails closed unless:

- `.name` is this project's registry name,
- `.packages` holds exactly one package, of registryType pypi, whose identifier
  is this project's PyPI distribution,
- `.version` and `.packages[0].version` both equal the version being released.

Stdlib only (runs on the ubuntu runner's python3 with nothing installed).

Usage (from the workflow):
    python3 scripts/check_server_json.py --expect-version "$NEW_VERSION" server.json
Exit status 0 with no output when every check passes, 1 with one line per
violation on stderr otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXPECTED_SERVER_NAME = "io.github.mcpsmiths/tracehub-mcp"
EXPECTED_PACKAGE_IDENTIFIER = "tracehub-mcp"
EXPECTED_REGISTRY_TYPE = "pypi"


def check_server_json(doc: dict[str, Any], expected_version: str) -> list[str]:
    """Return every identity violation in `doc` (empty when it is publishable).

    Missing or malformed keys are reported as violations, never raised, so the
    workflow step prints the whole picture instead of a traceback.
    """
    violations: list[str] = []

    name = doc.get("name")
    if name != EXPECTED_SERVER_NAME:
        violations.append(f".name is {name!r}, expected {EXPECTED_SERVER_NAME!r}")

    version = doc.get("version")
    if version != expected_version:
        violations.append(f".version is {version!r}, expected {expected_version!r}")

    packages = doc.get("packages")
    if not isinstance(packages, list) or not packages:
        violations.append(f".packages must contain exactly one package, found {packages!r}")
        return violations
    if len(packages) != 1:
        violations.append(f".packages must contain exactly one package, found {len(packages)}")

    package = packages[0]
    if not isinstance(package, dict):
        violations.append(f".packages[0] must be an object, found {package!r}")
        return violations

    registry_type = package.get("registryType")
    if registry_type != EXPECTED_REGISTRY_TYPE:
        violations.append(
            f".packages[0].registryType is {registry_type!r}, expected {EXPECTED_REGISTRY_TYPE!r}"
        )

    identifier = package.get("identifier")
    if identifier != EXPECTED_PACKAGE_IDENTIFIER:
        violations.append(
            f".packages[0].identifier is {identifier!r}, expected {EXPECTED_PACKAGE_IDENTIFIER!r}"
        )

    package_version = package.get("version")
    if package_version != expected_version:
        violations.append(
            f".packages[0].version is {package_version!r}, expected {expected_version!r}"
        )

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--expect-version", required=True, help="the version being released, e.g. 0.12.3"
    )
    parser.add_argument("path", type=Path, help="path to server.json")
    args = parser.parse_args(argv)

    try:
        doc = json.loads(args.path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"check_server_json: cannot read {args.path}: {exc}", file=sys.stderr)
        return 1
    if not isinstance(doc, dict):
        print(f"check_server_json: {args.path} is not a JSON object", file=sys.stderr)
        return 1

    violations = check_server_json(doc, args.expect_version)
    for violation in violations:
        print(f"check_server_json: {violation}", file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())

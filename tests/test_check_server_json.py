"""scripts/check_server_json.py binds the MCP Registry entry to this project's identity.

mcp-registry-publish.yml publishes whatever server.json the checked-out ref
carries; before this gate the only validation was a semver regex on the version
input and `jq` rewriting the two version fields, so a ref whose server.json
named a different PyPI distribution (or a different server name) would have been
published under this project's registry namespace, and the PyPI ownership poll
hardcoded `tracehub-mcp` regardless. These tests pin the pure checker and its CLI.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_server_json.py"
SERVER_JSON = REPO_ROOT / "server.json"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_server_json", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _committed_doc() -> dict[str, Any]:
    doc = json.loads(SERVER_JSON.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


def _release_version() -> str:
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        version = tomllib.load(f)["project"]["version"]
    assert isinstance(version, str)
    return version


def test_committed_server_json_matches_the_expected_identity() -> None:
    script = _load_script()
    doc = _committed_doc()
    assert script.check_server_json(doc, _release_version()) == []
    assert doc["name"] == script.EXPECTED_SERVER_NAME
    assert doc["packages"][0]["identifier"] == script.EXPECTED_PACKAGE_IDENTIFIER
    assert doc["packages"][0]["registryType"] == script.EXPECTED_REGISTRY_TYPE


def test_substituted_package_identifier_is_exactly_one_violation() -> None:
    """The audit's case C: same server name, different PyPI distribution -
    the old gate (semver regex + jq) let this through."""
    script = _load_script()
    doc = _committed_doc()
    doc["packages"][0]["identifier"] = "tracehub-mcp-evil"
    violations = script.check_server_json(doc, _release_version())
    assert len(violations) == 1, violations
    assert "packages[0].identifier" in violations[0] and "tracehub-mcp-evil" in violations[0]


def test_foreign_server_name_is_a_violation() -> None:
    script = _load_script()
    doc = _committed_doc()
    doc["name"] = "io.github.mcpsmiths/not-ours"
    violations = script.check_server_json(doc, _release_version())
    assert len(violations) == 1 and ".name" in violations[0], violations


def test_every_identity_field_is_checked_independently() -> None:
    script = _load_script()
    doc = _committed_doc()
    doc["packages"].append(dict(doc["packages"][0]))
    doc["packages"][0]["registryType"] = "npm"
    doc["version"] = "0.0.1"
    doc["packages"][0]["version"] = "0.0.2"
    violations = script.check_server_json(doc, _release_version())
    joined = "\n".join(violations)
    assert len(violations) == 4, violations
    assert "exactly one package" in joined
    assert "registryType" in joined and "'npm'" in joined
    assert ".version is '0.0.1'" in joined
    assert "packages[0].version is '0.0.2'" in joined


@pytest.mark.parametrize(
    "doc",
    [
        {},
        {"name": "io.github.mcpsmiths/tracehub-mcp"},
        {"name": "io.github.mcpsmiths/tracehub-mcp", "version": "0.12.3", "packages": "nope"},
        {"name": "io.github.mcpsmiths/tracehub-mcp", "version": "0.12.3", "packages": []},
        {"name": "io.github.mcpsmiths/tracehub-mcp", "version": "0.12.3", "packages": ["str"]},
    ],
)
def test_missing_or_malformed_keys_produce_violations_not_exceptions(doc: dict[str, Any]) -> None:
    script = _load_script()
    violations = script.check_server_json(doc, "0.12.3")
    assert violations, doc
    assert all(isinstance(v, str) for v in violations)


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - our own script, fixed arguments
        [sys.executable, str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_exits_zero_silently_for_the_committed_file() -> None:
    result = _run_cli("--expect-version", _release_version(), str(SERVER_JSON))
    assert result.returncode == 0, result.stderr
    assert result.stdout == "" and result.stderr == ""


def test_cli_exits_one_and_prints_every_violation_for_a_tampered_file(tmp_path: Path) -> None:
    doc = _committed_doc()
    doc["name"] = "io.github.mcpsmiths/not-ours"
    doc["packages"][0]["identifier"] = "tracehub-mcp-evil"
    tampered = tmp_path / "server.json"
    tampered.write_text(json.dumps(doc), encoding="utf-8")
    result = _run_cli("--expect-version", _release_version(), str(tampered))
    assert result.returncode == 1
    assert ".name" in result.stderr and "packages[0].identifier" in result.stderr, result.stderr


def test_cli_exits_one_for_the_wrong_expected_version() -> None:
    result = _run_cli("--expect-version", "9.9.9", str(SERVER_JSON))
    assert result.returncode == 1
    assert ".version" in result.stderr and "packages[0].version" in result.stderr, result.stderr


def test_cli_exits_one_for_unreadable_or_non_object_input(tmp_path: Path) -> None:
    assert _run_cli("--expect-version", "0.12.3", str(tmp_path / "missing.json")).returncode == 1
    not_object = tmp_path / "list.json"
    not_object.write_text("[]", encoding="utf-8")
    assert _run_cli("--expect-version", "0.12.3", str(not_object)).returncode == 1

"""The release bump must leave every version site consistent, uv.lock included.

Commitizen rewrites .cz.toml's `version_files`, but uv.lock carries the
project's own version in its `[[package]]` entry too and cannot be listed
there (many packages share the `version = "..."` line shape). Two releases in
a row (0.12.2, 0.12.3) shipped with uv.lock one version behind and needed a
follow-up chore commit. scripts/sync_uv_lock_version.py runs as a Commitizen
pre-bump hook to close that gap; these tests pin both the hook's behaviour and
its wiring, and fail CI if the two files ever disagree again.
"""

import importlib.util
import json
import shutil
import subprocess
import tomllib
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
from packaging.requirements import Requirement

import opentelemetry_mcp

REPO_ROOT = Path(__file__).resolve().parent.parent
HOOK_PATH = REPO_ROOT / "scripts" / "sync_uv_lock_version.py"


def _load_hook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sync_uv_lock_version", HOOK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _project_lock_version(lock_text: str, name: str) -> str:
    lock = tomllib.loads(lock_text)
    versions = [str(p["version"]) for p in lock["package"] if p["name"] == name]
    assert len(versions) == 1, versions
    return versions[0]


def test_uv_lock_self_version_matches_pyproject() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    lock_text = (REPO_ROOT / "uv.lock").read_text()
    assert (
        _project_lock_version(lock_text, pyproject["project"]["name"])
        == (pyproject["project"]["version"])
    ), "uv.lock's own [[package]] version drifted from pyproject.toml - run `uv lock`"


def test_pre_bump_hook_is_wired_into_cz_toml() -> None:
    cz = tomllib.loads((REPO_ROOT / ".cz.toml").read_text())
    hooks = cz["tool"]["commitizen"]["pre_bump_hooks"]
    assert any("scripts/sync_uv_lock_version.py" in h for h in hooks), hooks
    assert HOOK_PATH.is_file()


_SAMPLE_LOCK = """\
version = 1

[[package]]
name = "anyio"
version = "0.12.2"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "tracehub-mcp"
version = "0.12.2"
source = { editable = "." }
dependencies = [
    { name = "anyio" },
]

[[package]]
name = "zzz"
version = "0.12.2"
source = { registry = "https://pypi.org/simple" }
"""


def test_hook_rewrites_only_the_projects_own_block() -> None:
    hook = _load_hook()
    updated = hook.sync_self_version(_SAMPLE_LOCK, "tracehub-mcp", "0.12.3")

    assert _project_lock_version(updated, "tracehub-mcp") == "0.12.3"
    # Other packages that happen to share the old version string are untouched.
    assert _project_lock_version(updated, "anyio") == "0.12.2"
    assert _project_lock_version(updated, "zzz") == "0.12.2"
    # Nothing else moved - the diff is exactly one line.
    changed = [
        (a, b)
        for a, b in zip(_SAMPLE_LOCK.splitlines(), updated.splitlines(), strict=True)
        if a != b
    ]
    assert changed == [('version = "0.12.2"', 'version = "0.12.3"')]


def test_hook_is_idempotent() -> None:
    hook = _load_hook()
    once = hook.sync_self_version(_SAMPLE_LOCK, "tracehub-mcp", "0.12.3")
    assert hook.sync_self_version(once, "tracehub-mcp", "0.12.3") == once


@pytest.mark.parametrize("name", ["not-in-lock", "anyio"])
def test_hook_refuses_unless_exactly_one_project_block(name: str) -> None:
    # "anyio" exists but is not an editable project entry shaped like ours;
    # the regex still finds exactly one block for it, so use a doubled lock
    # to prove the "more than one" branch and a missing name for "zero".
    hook = _load_hook()
    if name == "not-in-lock":
        with pytest.raises(ValueError, match="found 0"):
            hook.sync_self_version(_SAMPLE_LOCK, name, "9.9.9")
    else:
        with pytest.raises(ValueError, match="found 2"):
            hook.sync_self_version(_SAMPLE_LOCK + "\n" + _SAMPLE_LOCK, name, "9.9.9")


def test_hook_main_reads_new_version_from_commitizen_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook = _load_hook()
    lock = tmp_path / "uv.lock"
    lock.write_text(_SAMPLE_LOCK)
    monkeypatch.chdir(tmp_path)

    monkeypatch.delenv("CZ_PRE_NEW_VERSION", raising=False)
    assert hook.main() == 2, "must refuse to guess when Commitizen's env var is absent"
    assert lock.read_text() == _SAMPLE_LOCK

    monkeypatch.setenv("CZ_PRE_NEW_VERSION", "0.12.3")
    assert hook.main() == 0
    assert _project_lock_version(lock.read_text(), "tracehub-mcp") == "0.12.3"
    assert hook.main() == 0, "second run is a no-op, not an error"


# --- MCPB launcher: the bundle labelled X runs tracehub-mcp==X and ships only its sources --

MCPB_DIR = REPO_ROOT / "mcpb"
LAUNCHER_FILES = {"manifest.json", "pyproject.toml", "src/server.py"}


def _mcpb_launcher_pin() -> str:
    launcher = tomllib.loads((MCPB_DIR / "pyproject.toml").read_text())
    deps = [Requirement(d) for d in launcher["project"]["dependencies"]]
    (dep,) = [d for d in deps if d.name == "tracehub-mcp"]
    specs = list(dep.specifier)
    assert len(specs) == 1 and specs[0].operator == "==", (
        f"mcpb/pyproject.toml must pin tracehub-mcp exactly, got {dep!s}"
    )
    return specs[0].version


def test_mcpb_launcher_pin_and_manifest_match_release_version() -> None:
    """A bare `tracehub-mcp` dependency let a co-located stale uv.lock (or
    simply the newest PyPI release) decide what a 0.12.3-labelled bundle ran."""
    root = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["version"]
    manifest = json.loads((MCPB_DIR / "manifest.json").read_text())["version"]
    launcher = tomllib.loads((MCPB_DIR / "pyproject.toml").read_text())["project"]["version"]
    assert root == manifest == launcher == _mcpb_launcher_pin() == opentelemetry_mcp.__version__


def test_mcpbignore_excludes_local_build_artifacts() -> None:
    """`mcpb pack` archives everything under mcpb/ except its built-in
    patterns and this file - it never reads .gitignore."""
    path = MCPB_DIR / ".mcpbignore"
    assert path.is_file(), "mcpb/.mcpbignore is missing"
    entries = {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert {"uv.lock", ".venv/", "__pycache__/", "*.pyc", ".env", ".env.*"} <= entries, entries


def test_cz_bumps_the_mcpb_launcher_pin() -> None:
    cz = tomllib.loads((REPO_ROOT / ".cz.toml").read_text())
    assert "mcpb/pyproject.toml:tracehub-mcp==" in cz["tool"]["commitizen"]["version_files"]


@pytest.mark.skipif(shutil.which("mcpb") is None, reason="@anthropic-ai/mcpb CLI not installed")
def test_mcpb_pack_ships_only_the_tracked_launcher_files(tmp_path: Path) -> None:
    """Pack a copy of mcpb/ with the artifacts the documented command used to
    archive planted next to the sources; none may reach the bundle."""
    bundle_src = tmp_path / "mcpb"
    shutil.copytree(
        MCPB_DIR, bundle_src, ignore=shutil.ignore_patterns(".venv", "uv.lock", "__pycache__")
    )
    (bundle_src / "uv.lock").write_text("version = 1\n")
    (bundle_src / ".venv").mkdir()
    (bundle_src / ".venv" / "pyvenv.cfg").write_text("home = /Users/someone/.local/bin\n")
    (bundle_src / "src" / "__pycache__").mkdir()
    (bundle_src / "src" / "__pycache__" / "server.cpython-313.pyc").write_bytes(b"\x00")
    (bundle_src / ".env").write_text("BACKEND_API_KEY=not-a-real-key\n")
    (bundle_src / ".env.datadog").write_text("BACKEND_APP_KEY=not-a-real-key\n")
    mcpb = shutil.which("mcpb")
    assert mcpb is not None
    bundle = tmp_path / "check.mcpb"
    result = subprocess.run(  # noqa: S603 - fixed arguments, scratch copy of mcpb/
        [mcpb, "pack", str(bundle_src), str(bundle)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with zipfile.ZipFile(bundle) as archive:
        entries = {name for name in archive.namelist() if not name.endswith("/")}
    assert entries == LAUNCHER_FILES, entries

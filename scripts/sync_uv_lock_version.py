"""Commitizen pre-bump hook: keep uv.lock's self-referential version in step.

`cz bump` rewrites every path in .cz.toml's `version_files`, but uv.lock also
carries this project's own version in its `[[package]]` entry - and uv.lock
cannot be a `version_files` entry, because dozens of other packages in it have
their own `version = "..."` line. Left alone, `uv lock` reports
"Updated tracehub-mcp vOLD -> vNEW" after every release and a chore commit has
to follow (it did for 0.12.2 and 0.12.3). This hook rewrites just that one
block so the bump commit (`git commit -a`) carries the lockfile too and the
release tag points at a fully consistent tree.

Runs INSIDE commitizen-action's container (python:3.10-alpine, no uv), with
the new version in CZ_PRE_NEW_VERSION - so: stdlib only, no tomllib (3.11+),
no `uv lock`. Exits non-zero on anything unexpected: a failed hook aborts the
bump before the commit, which is the loud failure we want.

Usage (from .cz.toml):  pre_bump_hooks = ["python3 scripts/sync_uv_lock_version.py"]
Manual check:           CZ_PRE_NEW_VERSION=1.2.3 python3 scripts/sync_uv_lock_version.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

PROJECT_NAME = "tracehub-mcp"
LOCK_PATH = Path("uv.lock")


def sync_self_version(lock_text: str, project_name: str, new_version: str) -> str:
    """Return lock_text with only the project's own [[package]] version replaced.

    Matches the `version = "..."` line that directly follows the project's
    `name = "..."` line inside a `[[package]]` table, and nothing else - other
    packages that happen to share the old version string are untouched.
    Raises ValueError unless exactly one such block exists.
    """
    pattern = re.compile(
        r'^(\[\[package\]\]\nname = "' + re.escape(project_name) + r'"\nversion = ")([^"\n]*)(")',
        re.MULTILINE,
    )
    matches = pattern.findall(lock_text)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one [[package]] block for {project_name!r} in uv.lock, "
            f"found {len(matches)}"
        )
    return pattern.sub(lambda m: f"{m.group(1)}{new_version}{m.group(3)}", lock_text, count=1)


def main() -> int:
    new_version = os.environ.get("CZ_PRE_NEW_VERSION")
    if not new_version:
        print("sync_uv_lock_version: CZ_PRE_NEW_VERSION is not set", file=sys.stderr)
        return 2
    if not LOCK_PATH.is_file():
        print(f"sync_uv_lock_version: {LOCK_PATH} not found in {Path.cwd()}", file=sys.stderr)
        return 2
    original = LOCK_PATH.read_text(encoding="utf-8")
    try:
        updated = sync_self_version(original, PROJECT_NAME, new_version)
    except ValueError as exc:
        print(f"sync_uv_lock_version: {exc}", file=sys.stderr)
        return 1
    if updated == original:
        print(f"sync_uv_lock_version: uv.lock already at {new_version}", file=sys.stderr)
        return 0
    LOCK_PATH.write_text(updated, encoding="utf-8")
    print(f"sync_uv_lock_version: uv.lock {PROJECT_NAME} -> {new_version}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

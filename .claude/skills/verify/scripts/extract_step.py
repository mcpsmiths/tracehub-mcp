#!/usr/bin/env python3
"""Print the verbatim `run: |` body of one GitHub Actions step (no PyYAML needed).

    extract_step.py WORKFLOW.yml 'name: <substring of the step name>' > step.sh
    GITHUB_REF_NAME=v0.0.4 GITHUB_SHA=$(git rev-parse v0.0.4) bash -e step.sh   # GHA runs `bash -e {0}`

Run the extracted script only inside a throwaway fixture repo under your evidence dir, never
in a real checkout.
"""

from __future__ import annotations

import sys


def extract(workflow: str, needle: str) -> str:
    with open(workflow, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    hits = [n for n, line in enumerate(lines) if needle in line]
    if not hits:
        raise SystemExit(f"no line containing {needle!r} in {workflow}")
    j = hits[0] + 1
    while lines[j].strip() != "run: |":
        j += 1
    indent = len(lines[j]) - len(lines[j].lstrip())
    body: list[str] = []
    for line in lines[j + 1 :]:
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        body.append(line)
    margin = min(len(line) - len(line.lstrip()) for line in body if line.strip())
    return "\n".join(line[margin:] for line in body).rstrip("\n") + "\n"


if __name__ == "__main__":
    sys.stdout.write(extract(sys.argv[1], sys.argv[2]))

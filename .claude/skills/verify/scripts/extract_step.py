#!/usr/bin/env python3
"""Print the verbatim `run: |` body of one GitHub Actions step (no PyYAML needed).

    extract_step.py WORKFLOW.yml 'name: <substring of the step name>' > step.sh
    GITHUB_REF_NAME=v0.0.4 GITHUB_SHA=$(git rev-parse v0.0.4) bash -e step.sh   # GHA runs `bash -e {0}`

The matched line is echoed to stderr as `file:line: - name: ...` - check it is the step you meant
(workflows repeat names such as `Setup Python`). A step without a `run: |` block (a `uses:` step, a
single-line or folded `run:`) exits 1 instead of handing back the NEXT step's body.

Run the extracted script only inside a throwaway fixture repo under your evidence dir, never
in a real checkout.
"""

from __future__ import annotations

import re
import sys

# The next list item of the steps sequence (`- name:` / `- uses:` / `- id:` / `- run:` ...) ends the step.
STEP_START = re.compile(r"^\s*-\s+\w+:")


def extract(workflow: str, needle: str) -> str:
    with open(workflow, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    hits = [n for n, line in enumerate(lines) if needle in line]
    if not hits:
        raise SystemExit(f"no line containing {needle!r} in {workflow}")
    first = hits[0]
    print(f"extract_step: {workflow}:{first + 1}: {lines[first].strip()}", file=sys.stderr)
    if len(hits) > 1:
        others = ", ".join(str(n + 1) for n in hits[1:])
        print(
            f"extract_step: {needle!r} also matches line(s) {others}; using the first",
            file=sys.stderr,
        )
    j = first + 1
    while j < len(lines) and lines[j].strip() != "run: |" and not STEP_START.match(lines[j]):
        j += 1
    if j >= len(lines) or lines[j].strip() != "run: |":
        raise SystemExit(
            f"step at {workflow}:{first + 1} has no `run: |` block "
            "(a `uses:` step, or a single-line / folded `run:`) - refusing to return another step's body"
        )
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

"""Secret detection must cover the credential class this repository already leaked.

start_locally.sh carried a Traceloop API key (`tl_` + 32 lowercase
alphanumerics) at c5fd220; HEAD is clean but the blob is reachable from every
ref and every tag. Neither GitHub's native scanning (no partner pattern for
`tl_` keys) nor TruffleHog's built-in detectors (none for Traceloop) would ever
flag it, the workflow only scanned the pushed/PR commit range, and no pre-commit
hook detected secrets at all. These tests pin the detection controls a
maintainer can land independently of the owner's rotate/rewrite decision: a
weekly whole-history scan, a custom TruffleHog detector and a gitleaks rule
for the pattern, a pinned gitleaks pre-commit hook, and a clean tracked tree.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SECRET_SCAN_YML = REPO_ROOT / ".github" / "workflows" / "secret-scan.yml"
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
TRUFFLEHOG_CONFIG = REPO_ROOT / ".trufflehog.yaml"
GITLEAKS_CONFIG = REPO_ROOT / ".gitleaks.toml"

TRACELOOP_KEY_PATTERN = "tl_[a-z0-9]{32}"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _secret_scan() -> dict[Any, Any]:
    data = yaml.safe_load(SECRET_SCAN_YML.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _only_step_using(job: dict[str, Any], action: str) -> dict[str, Any]:
    steps = [s for s in job["steps"] if str(s.get("uses", "")).startswith(action)]
    assert len(steps) == 1, steps
    step = steps[0]
    assert isinstance(step, dict)
    return step


def _trufflehog_step(job: dict[str, Any]) -> dict[str, Any]:
    return _only_step_using(job, "trufflesecurity/trufflehog@")


def _checkout_step(job: dict[str, Any]) -> dict[str, Any]:
    return _only_step_using(job, "actions/checkout@")


def _jobs_by_role() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (diff-scoped job, whole-history job) by their `if` conditions."""
    jobs = _secret_scan()["jobs"]
    history = [j for j in jobs.values() if "schedule" in str(j.get("if", ""))]
    diff = [j for j in jobs.values() if "pull_request" in str(j.get("if", ""))]
    assert len(history) == 1 and len(diff) == 1, list(jobs)
    return diff[0], history[0]


def test_secret_scan_also_runs_weekly_and_on_demand() -> None:
    triggers = _secret_scan()[True]  # the `on` key parses as True under YAML 1.1
    assert {"push", "pull_request", "schedule", "workflow_dispatch"} <= set(triggers), triggers
    assert triggers["schedule"][0]["cron"], triggers["schedule"]


def test_whole_history_job_scans_every_commit_with_the_custom_detector() -> None:
    diff_job, history_job = _jobs_by_role()
    condition = str(history_job["if"])
    assert "schedule" in condition and "workflow_dispatch" in condition, condition

    assert _checkout_step(history_job)["with"]["fetch-depth"] == 0
    assert _checkout_step(history_job)["with"]["persist-credentials"] is False
    step = _trufflehog_step(history_job)
    # Empty base AND head: the action then runs `--since-commit '' --branch ''`,
    # i.e. the whole history of every fetched branch, not an event-derived range.
    assert step["with"]["base"] == "" and step["with"]["head"] == "", step["with"]
    extra = str(step["with"]["extra_args"])
    assert "--results=verified,unknown,unverified" in extra, extra
    assert "--config .trufflehog.yaml" in extra, extra
    assert step["uses"] == _trufflehog_step(diff_job)["uses"], "both jobs pin the same version"


def test_diff_job_keeps_verified_unknown_and_loads_the_custom_detector() -> None:
    diff_job, _ = _jobs_by_role()
    assert _checkout_step(diff_job)["with"]["fetch-depth"] == 0
    extra = str(_trufflehog_step(diff_job)["with"]["extra_args"])
    assert "--results=verified,unknown" in extra and "unverified" not in extra, extra
    assert "--config .trufflehog.yaml" in extra, extra


def test_trufflehog_config_declares_the_traceloop_detector() -> None:
    assert TRUFFLEHOG_CONFIG.is_file(), ".trufflehog.yaml is missing"
    config = yaml.safe_load(TRUFFLEHOG_CONFIG.read_text(encoding="utf-8"))
    detectors = {d["name"]: d for d in config["detectors"]}
    detector = detectors["traceloop-api-key"]
    assert "tl_" in detector["keywords"], detector
    assert any(TRACELOOP_KEY_PATTERN in str(rx) for rx in detector["regex"].values()), detector


def test_gitleaks_config_extends_the_defaults_with_the_traceloop_rule() -> None:
    assert GITLEAKS_CONFIG.is_file(), ".gitleaks.toml is missing"
    config = tomllib.loads(GITLEAKS_CONFIG.read_text(encoding="utf-8"))
    assert config["extend"]["useDefault"] is True, config.get("extend")
    rules = {r["id"]: r for r in config["rules"]}
    rule = rules["traceloop-api-key"]
    assert TRACELOOP_KEY_PATTERN in rule["regex"], rule
    assert "tl_" in rule["keywords"], rule


def test_pre_commit_declares_a_frozen_sha_pinned_gitleaks_hook() -> None:
    text = PRE_COMMIT_CONFIG.read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    repos = [r for r in config["repos"] if str(r.get("repo", "")).endswith("gitleaks/gitleaks")]
    assert len(repos) == 1, "no gitleaks repo in .pre-commit-config.yaml"
    repo = repos[0]
    assert "gitleaks" in {h["id"] for h in repo["hooks"]}, repo["hooks"]
    rev = str(repo["rev"])
    assert _SHA_RE.match(rev), f"rev must be a full commit SHA, got {rev!r}"
    rev_lines = [line for line in text.splitlines() if rev in line]
    assert len(rev_lines) == 1 and re.search(r"#\s*frozen:\s*v\d", rev_lines[0]), rev_lines
    # pre-commit.ci cannot build golang hooks; CI's own trufflehog job covers it.
    assert "gitleaks" in config["ci"]["skip"], config["ci"]


@pytest.mark.skipif(shutil.which("gitleaks") is None, reason="gitleaks binary not installed")
def test_gitleaks_rule_detects_a_fake_key_and_default_rules_do_not(tmp_path: Path) -> None:
    """Positive and negative control for the rule itself. The fake key is
    assembled at run time so no tracked file ever contains a match."""
    gitleaks = shutil.which("gitleaks")
    assert gitleaks is not None
    (tmp_path / "config.env").write_text(
        "export BACKEND_API_KEY=" + "tl_" + "a1" * 16 + "\n", encoding="utf-8"
    )

    def scan(*extra: str) -> int:
        return subprocess.run(  # noqa: S603 - fixed arguments, our own fixture
            [gitleaks, "dir", str(tmp_path), "--no-banner", "--redact", "--exit-code", "7", *extra],
            capture_output=True,
            text=True,
            check=False,
        ).returncode

    assert scan("--config", str(GITLEAKS_CONFIG)) == 7, "the traceloop rule must fire"
    assert scan() == 0, "gitleaks' default rules alone do not know this key shape"


def test_tracked_tree_contains_no_traceloop_key_shaped_string() -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git not installed")
    result = subprocess.run(  # noqa: S603 - fixed arguments; never prints a match
        [git, "grep", "-c", "-E", TRACELOOP_KEY_PATTERN, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 128:
        pytest.skip("not running inside a git checkout")
    # exit 1 = no match; exit 0 = at least one tracked file matches (count only)
    assert result.returncode == 1, f"tl_ key-shaped string in tracked files: {result.stdout}"

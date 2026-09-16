"""One-off script to refresh the vendored litellm model-pricing table.

Not part of the shipped package - run directly from the repo root:

    uv run python scripts/sync_model_prices.py

Fetches litellm's model_prices_and_context_window.json verbatim, writes it
byte-identical into src/opentelemetry_mcp/pricing/model_prices.json, and
records provenance (source URL, commit SHA, timestamp) in the sibling
metadata.json. Review the diff before committing - see
src/opentelemetry_mcp/pricing/README.md for the full refresh process.
"""

import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

_SOURCE_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)
_COMMITS_API_URL = "https://api.github.com/repos/BerriAI/litellm/commits/main"
_PRICING_DIR = Path(__file__).resolve().parent.parent / "src/opentelemetry_mcp/pricing"


def sync() -> None:
    with urllib.request.urlopen(_SOURCE_URL, timeout=30) as response:  # noqa: S310
        raw = response.read()

    # Validate it's real JSON before writing anything.
    json.loads(raw)

    with urllib.request.urlopen(_COMMITS_API_URL, timeout=30) as response:  # noqa: S310
        commit_sha = json.loads(response.read())["sha"]

    (_PRICING_DIR / "model_prices.json").write_bytes(raw)
    metadata = {
        "source_url": _SOURCE_URL,
        "source_ref": commit_sha,
        "synced_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (_PRICING_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Synced model_prices.json to litellm@{commit_sha[:12]}")


if __name__ == "__main__":
    sync()

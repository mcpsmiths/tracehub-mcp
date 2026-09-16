"""Per-token USD pricing lookups against the vendored litellm pricing table.

Resolution is a best-effort approximation, not a billing-grade lookup -
litellm's own upstream history shows even a maintained, exact-match pricing
table gets model-ID variants wrong sometimes (a missed dated suffix, a
missing provider prefix). Callers must treat an unresolvable model as
"unknown", not "free" - see UsageMetrics.cost_usd_is_partial in models.py.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources

from pydantic import BaseModel

_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")


class ModelPriceEntry(BaseModel):
    """The subset of litellm's per-model pricing schema this project uses.

    litellm's real file carries many more fields (context window sizes,
    capability flags, etc.) - pydantic's default "ignore extra" behavior
    means those pass through harmlessly without needing to be modeled here.
    """

    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    cache_read_input_token_cost: float | None = None
    cache_creation_input_token_cost: float | None = None
    litellm_provider: str | None = None
    mode: str | None = None


@lru_cache(maxsize=1)
def _load_prices() -> dict[str, ModelPriceEntry]:
    """Load the vendored pricing table once per process.

    "sample_spec" is litellm's own documentation placeholder entry (its
    values are literal descriptive strings, not real numbers) - excluded
    so it can never be accidentally resolved as a real model's price.
    """
    raw = resources.files("opentelemetry_mcp.pricing").joinpath("model_prices.json").read_text()
    data = json.loads(raw)
    return {
        key: ModelPriceEntry.model_validate(value)
        for key, value in data.items()
        if key != "sample_spec" and isinstance(value, dict)
    }


def resolve_price(
    model: str | None,
    provider: str | None = None,
    prices: dict[str, ModelPriceEntry] | None = None,
) -> ModelPriceEntry | None:
    """Resolve a (provider, model) pair to a pricing entry.

    Resolution order: exact "{provider}/{model}" -> exact "model" -> the
    date-suffix-stripped model name -> the lexicographically latest sibling
    entry sharing that same stripped prefix (a newer/older dated release of
    the same model family than the vendored table happens to have). Returns
    None when nothing resolves - never fabricates a price.

    `prices` is injectable so tests stay hermetic and immune to future
    price-table refreshes; it defaults to the real vendored table.
    """
    if not model:
        return None
    table = prices if prices is not None else _load_prices()

    if provider:
        qualified = f"{provider}/{model}"
        if qualified in table:
            return table[qualified]

    if model in table:
        return table[model]

    base = _DATE_SUFFIX_RE.sub("", model)
    if base != model and base in table:
        return table[base]

    siblings = sorted(
        key for key in table if key.startswith(base + "-") and _DATE_SUFFIX_RE.search(key)
    )
    if siblings:
        return table[siblings[-1]]

    return None


def compute_cost_usd(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    model: str | None,
    provider: str | None = None,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    prices: dict[str, ModelPriceEntry] | None = None,
) -> float | None:
    """Compute a span's cost in USD, or None if the model's price can't be
    resolved or the resolved entry has no per-token rates at all."""
    entry = resolve_price(model, provider=provider, prices=prices)
    if entry is None:
        return None
    if entry.input_cost_per_token is None and entry.output_cost_per_token is None:
        return None

    cost = 0.0
    cost += prompt_tokens * (entry.input_cost_per_token or 0.0)
    cost += completion_tokens * (entry.output_cost_per_token or 0.0)
    cost += cache_read_tokens * (entry.cache_read_input_token_cost or 0.0)
    cost += cache_creation_tokens * (entry.cache_creation_input_token_cost or 0.0)
    return round(cost, 6)

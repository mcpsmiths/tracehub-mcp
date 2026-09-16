# Pricing data

`model_prices.json` is a vendored, byte-identical copy of
[litellm's](https://github.com/BerriAI/litellm) MIT-licensed
`model_prices_and_context_window.json` - a maintained, per-token USD pricing
table keyed by model name (with a `litellm_provider` field for
disambiguating identically-named models across providers). See
`NOTICE` for the license attribution.

`metadata.json` tracks provenance (`source_url`, `source_ref`, `synced_at`)
separately from the price file itself, so `model_prices.json` stays cleanly
diffable against upstream on every refresh.

## Refreshing

There is deliberately **no automated CI bot** for this in v1 - litellm's own
automated pricing-updater has failed silently for over a month in the past,
so a documented manual refresh with a visible `synced_at` marker is the
safer choice for now:

```bash
uv run python scripts/sync_model_prices.py
git diff src/opentelemetry_mcp/pricing/
```

Review the diff, then commit both `model_prices.json` and `metadata.json`
together.

## Accuracy

Resolution (`lookup.py`'s `resolve_price`) is a best-effort approximation,
not a billing-grade lookup. A model that can't be resolved contributes
nothing to `cost_usd` and flips `UsageMetrics.cost_usd_is_partial` to
`True`, so `get_llm_usage`'s reported total is always an honest floor,
never a silently-wrong "exact" number.

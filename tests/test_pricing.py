"""Tests for the vendored litellm pricing table and its lookup helpers.

resolve_price/compute_cost_usd tests use an injected fixture table (via the
`prices=` parameter), not the real vendored file, so they stay hermetic and
immune to future price-table refreshes. TestVendoredFileIntegrity and
TestComputeCostUsdMatchesHandCalculation are the two tests that deliberately
DO load the real file, to guard against a future refresh corrupting it and
to hand-verify the cost math against a real, current price.
"""

from opentelemetry_mcp.pricing.lookup import (
    ModelPriceEntry,
    _load_prices,
    compute_cost_usd,
    resolve_price,
)


def _fixture_prices() -> dict[str, ModelPriceEntry]:
    return {
        "gpt-4": ModelPriceEntry(
            input_cost_per_token=0.00003,
            output_cost_per_token=0.00006,
            litellm_provider="openai",
            mode="chat",
        ),
        "claude-3-7-sonnet-20250219": ModelPriceEntry(
            input_cost_per_token=0.000003,
            output_cost_per_token=0.000015,
            cache_read_input_token_cost=0.0000003,
            cache_creation_input_token_cost=0.00000375,
            litellm_provider="anthropic",
            mode="chat",
        ),
        "bedrock/anthropic.claude-3-7-sonnet-20250219": ModelPriceEntry(
            input_cost_per_token=0.000004,
            output_cost_per_token=0.00002,
            litellm_provider="bedrock",
        ),
        "no-rates-model": ModelPriceEntry(litellm_provider="mystery"),
    }


class TestResolvePrice:
    def test_exact_match(self) -> None:
        entry = resolve_price("gpt-4", prices=_fixture_prices())

        assert entry is not None
        assert entry.input_cost_per_token == 0.00003

    def test_provider_disambiguated_match_takes_precedence(self) -> None:
        entry = resolve_price(
            "anthropic.claude-3-7-sonnet-20250219", provider="bedrock", prices=_fixture_prices()
        )

        assert entry is not None
        assert entry.input_cost_per_token == 0.000004

    def test_falls_back_to_bare_model_when_provider_prefix_has_no_entry(self) -> None:
        entry = resolve_price(
            "claude-3-7-sonnet-20250219", provider="anthropic", prices=_fixture_prices()
        )

        assert entry is not None
        assert entry.input_cost_per_token == 0.000003

    def test_dated_suffix_falls_back_to_sibling_dated_entry(self) -> None:
        """The vendored table may have a different dated release than a
        span reports - fall back to whichever dated sibling exists sharing
        the same stripped prefix."""
        entry = resolve_price("claude-3-7-sonnet-20250301", prices=_fixture_prices())

        assert entry is not None
        assert entry.input_cost_per_token == 0.000003

    def test_unresolvable_model_returns_none(self) -> None:
        assert resolve_price("totally-unknown-model-xyz", prices=_fixture_prices()) is None

    def test_none_model_returns_none(self) -> None:
        assert resolve_price(None, prices=_fixture_prices()) is None


class TestComputeCostUsd:
    def test_hand_computed_dollar_amount(self) -> None:
        cost = compute_cost_usd(
            prompt_tokens=1000,
            completion_tokens=500,
            model="gpt-4",
            prices=_fixture_prices(),
        )

        assert cost == round(1000 * 0.00003 + 500 * 0.00006, 6)

    def test_unpriced_model_returns_none(self) -> None:
        cost = compute_cost_usd(
            prompt_tokens=100,
            completion_tokens=50,
            model="totally-unknown-model-xyz",
            prices=_fixture_prices(),
        )

        assert cost is None

    def test_entry_with_no_rates_at_all_returns_none(self) -> None:
        cost = compute_cost_usd(
            prompt_tokens=100,
            completion_tokens=50,
            model="no-rates-model",
            prices=_fixture_prices(),
        )

        assert cost is None

    def test_cache_tokens_contribute_to_cost(self) -> None:
        cost = compute_cost_usd(
            prompt_tokens=100,
            completion_tokens=50,
            model="claude-3-7-sonnet-20250219",
            cache_read_tokens=1000,
            cache_creation_tokens=200,
            prices=_fixture_prices(),
        )

        entry = _fixture_prices()["claude-3-7-sonnet-20250219"]
        expected = (
            100 * entry.input_cost_per_token  # type: ignore[operator]
            + 50 * entry.output_cost_per_token  # type: ignore[operator]
            + 1000 * entry.cache_read_input_token_cost  # type: ignore[operator]
            + 200 * entry.cache_creation_input_token_cost  # type: ignore[operator]
        )
        assert cost == round(expected, 6)

    def test_none_model_returns_none(self) -> None:
        cost = compute_cost_usd(
            prompt_tokens=100, completion_tokens=50, model=None, prices=_fixture_prices()
        )

        assert cost is None


class TestVendoredFileIntegrity:
    """Loads the REAL vendored file - guards against a future refresh
    silently corrupting it."""

    def test_real_file_parses_and_has_a_sane_entry_count(self) -> None:
        prices = _load_prices()

        assert len(prices) > 1000

    def test_sample_spec_placeholder_is_excluded(self) -> None:
        """litellm's own documentation placeholder entry has literal
        descriptive-string values, not real numbers - it must never be
        resolvable as a real model's price."""
        prices = _load_prices()

        assert "sample_spec" not in prices

    def test_well_known_models_are_present(self) -> None:
        prices = _load_prices()

        assert "gpt-4" in prices
        assert prices["gpt-4"].litellm_provider == "openai"


class TestComputeCostUsdMatchesHandCalculation:
    """Hand-verification check: pick one real vendored entry, hand-multiply
    its rate against a synthetic span, and confirm compute_cost_usd (using
    the REAL table, not a fixture) matches to 6 decimal places."""

    def test_gpt_4_cost_matches_hand_calculation(self) -> None:
        entry = resolve_price("gpt-4")
        assert entry is not None
        assert entry.input_cost_per_token is not None
        assert entry.output_cost_per_token is not None

        prompt_tokens, completion_tokens = 1234, 567
        hand_calculated = round(
            prompt_tokens * entry.input_cost_per_token
            + completion_tokens * entry.output_cost_per_token,
            6,
        )

        cost = compute_cost_usd(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, model="gpt-4"
        )

        assert cost == hand_calculated

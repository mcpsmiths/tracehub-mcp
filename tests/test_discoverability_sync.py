"""Tests that the repo's discoverability surfaces (server.json's MCP
Registry listing, mcpb/manifest.json's Desktop Extension bundle) actually
advertise every backend this project supports.

Regression coverage for a real, live gap found in production: server.json
was last updated around when X-Ray shipped and was never touched again for
either New Relic or Honeycomb - CLAUDE.md's own "Adding New Backends"
step 5 (the hardcoded backend-name literal sites that must update in
lockstep) never listed server.json/mcpb/manifest.json as one of them, so
three consecutive backend additions (X-Ray, New Relic, Honeycomb) shipped
without anyone noticing these two files never got updated. This was live
on the actual MCP Registry (registry.modelcontextprotocol.io) at the time
this was found: `BACKEND_TYPE`'s choices there only listed 6 of the 8
backends this server actually supports, meaning anyone discovering this
server through the official registry could not see or configure New Relic
or Honeycomb at all.
"""

import json
from pathlib import Path
from typing import get_args

from opentelemetry_mcp.config import BackendConfig

REPO_ROOT = Path(__file__).resolve().parent.parent

_EXPECTED_BACKENDS = set(get_args(BackendConfig.model_fields["type"].annotation))


def test_expected_backends_is_not_accidentally_empty() -> None:
    # Sanity check on the extraction itself, so a typing/pydantic API
    # change silently returning an empty set doesn't make every other test
    # in this file vacuously pass.
    assert len(_EXPECTED_BACKENDS) >= 6
    assert "jaeger" in _EXPECTED_BACKENDS


def test_server_json_backend_type_choices_match_config() -> None:
    server_json = json.loads((REPO_ROOT / "server.json").read_text())
    env_vars = server_json["packages"][0]["environmentVariables"]
    backend_type_var = next(v for v in env_vars if v["name"] == "BACKEND_TYPE")

    assert set(backend_type_var["choices"]) == _EXPECTED_BACKENDS


def test_server_json_description_mentions_every_backend() -> None:
    server_json = json.loads((REPO_ROOT / "server.json").read_text())
    description = server_json["description"].lower()

    # Not a strict name match (the description uses display names like
    # "X-Ray"/"New Relic", not the config Literal's "xray"/"newrelic") -
    # just confirms every backend's own vendor name appears somewhere.
    display_names = {
        "jaeger": "jaeger",
        "tempo": "tempo",
        "traceloop": "traceloop",
        "datadog": "datadog",
        "sentry": "sentry",
        "xray": "x-ray",
        "newrelic": "new relic",
        "honeycomb": "honeycomb",
    }
    for backend in _EXPECTED_BACKENDS:
        assert display_names[backend] in description, (
            f"server.json's description is missing the {backend} backend"
        )


def test_server_json_description_fits_the_registrys_100_char_limit() -> None:
    # Regression test: this has already broken a live release once before
    # (commit e35ad7d, when X-Ray was added) and broke it again when New
    # Relic/Honeycomb were added - the MCP Registry's publish endpoint
    # rejects server.json with a 422 ("expected length <= 100") if
    # `description` exceeds 100 characters, which only surfaces as a CI
    # failure during the release workflow, not during normal development.
    server_json = json.loads((REPO_ROOT / "server.json").read_text())
    description = server_json["description"]
    assert len(description) <= 100, (
        f"server.json's description is {len(description)} chars, over the MCP "
        "Registry's 100-char limit - the publish step will fail with a 422"
    )


def test_mcpb_manifest_backend_type_description_mentions_every_backend() -> None:
    manifest = json.loads((REPO_ROOT / "mcpb" / "manifest.json").read_text())
    description = manifest["user_config"]["backend_type"]["description"].lower()

    for backend in _EXPECTED_BACKENDS:
        assert backend in description, (
            f"mcpb/manifest.json's backend_type description is missing {backend!r}"
        )


def test_mcpb_manifest_env_mapping_covers_every_backend_specific_field() -> None:
    manifest = json.loads((REPO_ROOT / "mcpb" / "manifest.json").read_text())
    env = manifest["server"]["mcp_config"]["env"]

    # Every backend-specific required field (per config.py's own exclusive
    # fields, not the shared type/url/api_key) must have a env mapping so a
    # Desktop Extension user can actually configure that backend.
    backend_specific_env_vars = {
        "BACKEND_AWS_REGION",  # xray
        "BACKEND_NEWRELIC_ACCOUNT_ID",  # newrelic
        "BACKEND_HONEYCOMB_DATASET",  # honeycomb
        "BACKEND_SENTRY_ORG",  # sentry
        "BACKEND_APP_KEY",  # datadog
        "BACKEND_TEMPO_INSTANCE_ID",  # tempo (Grafana Cloud)
    }
    missing = backend_specific_env_vars - set(env.keys())
    assert not missing, f"mcpb/manifest.json's mcp_config.env is missing: {missing}"

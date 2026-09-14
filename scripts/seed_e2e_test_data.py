"""One-off script to seed realistic gen_ai-shaped OTel trace data into a real
Datadog or Sentry account, for end-to-end verification of tracehub-mcp's
tools against a live backend rather than mocks.

Not part of the shipped package - run directly from the repo root:

    uv run python scripts/seed_e2e_test_data.py datadog
    uv run python scripts/seed_e2e_test_data.py sentry --dsn https://<key>@o<org>.ingest.sentry.io/<project>

Datadog needs an Agent with OTLP enabled reachable at localhost:4318 (see
CLAUDE.md/README for how this project's E2E setup runs one in Docker).
Sentry has no generic OTLP intake - the endpoint and x-sentry-auth header
are both derived from the DSN, per
https://docs.sentry.io/concepts/otlp/direct/traces/.

Every span is its own single-span trace (no parent/child nesting) - this
project's tools operate on gen_ai attributes and cross-span aggregation, not
multi-span trace reconstruction, so nesting would add complexity without
exercising anything these tools actually read.
"""

import argparse
import os
import random
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

SERVICES = ["checkout-service", "recommendation-service", "support-chatbot"]
MODELS = [
    ("openai", "gpt-4"),
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-3-5-sonnet-20241022"),
]
CONVERSATIONS = ["conv-seed-001", "conv-seed-002", "conv-seed-003", "conv-seed-004"]
PROMPT_VERSIONS = [("summarize", "1"), ("summarize", "2"), ("classify", "1")]


def _sentry_otlp_config(dsn: str) -> tuple[str, dict[str, str]]:
    """Derive the raw OTLP traces endpoint + auth header from a Sentry DSN.

    Sentry has no generic OTLP intake - unlike Datadog, the endpoint URL
    itself is project-specific, and auth rides a custom header rather than
    the standard Authorization header. Both are fully determined by the DSN,
    so this needs no separate credential.
    """
    parsed = urlparse(dsn)
    public_key = parsed.username
    project_id = parsed.path.lstrip("/")
    host = parsed.hostname  # o<orgId>.ingest.sentry.io
    if not public_key or not project_id or not host:
        raise ValueError(f"malformed Sentry DSN: {dsn!r}")
    endpoint = f"https://{host}/api/{project_id}/integration/otlp/v1/traces"
    headers = {"x-sentry-auth": f"sentry sentry_key={public_key}"}
    return endpoint, headers


def _build_exporter(target: str, dsn: str | None) -> OTLPSpanExporter:
    if target == "datadog":
        return OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
    if target == "sentry":
        if not dsn:
            raise ValueError("sentry target requires --dsn or SENTRY_DSN env var")
        endpoint, headers = _sentry_otlp_config(dsn)
        return OTLPSpanExporter(endpoint=endpoint, headers=headers)
    raise ValueError(f"unknown target: {target!r}")


def seed(target: str, count: int, dsn: str | None) -> None:
    exporter = _build_exporter(target, dsn)
    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "tracehub-mcp-seed"}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    tracer = provider.get_tracer("tracehub-mcp-seed-script")

    now = datetime.now(UTC)
    error_count = 0
    slow_count = 0

    for i in range(count):
        service = random.choice(SERVICES)
        system, model = random.choice(MODELS)
        conversation_id = random.choice(CONVERSATIONS)
        prompt_name, prompt_version = random.choice(PROMPT_VERSIONS)
        is_error = random.random() < 0.15
        is_slow = random.random() < 0.15
        duration_ms = random.uniform(2000, 6000) if is_slow else random.uniform(50, 800)
        prompt_tokens = random.randint(50, 2000)
        completion_tokens = random.randint(20, 1000)
        error_count += is_error
        slow_count += is_slow

        # Alternate between "this hour" and "the hour before" so
        # compare_time_windows has two genuinely different windows to diff,
        # rather than one lump of data with no separation.
        window_offset = timedelta(hours=1) if i % 2 == 0 else timedelta(hours=2)
        start_time = now - window_offset - timedelta(seconds=random.uniform(0, 1800))
        end_time = start_time + timedelta(milliseconds=duration_ms)

        span = tracer.start_span(
            "chat_completion",
            attributes={
                "service.name": service,
                "gen_ai.system": system,
                "gen_ai.request.model": model,
                "gen_ai.response.model": model,
                "gen_ai.usage.prompt_tokens": prompt_tokens,
                "gen_ai.usage.completion_tokens": completion_tokens,
                "gen_ai.usage.total_tokens": prompt_tokens + completion_tokens,
                "gen_ai.conversation.id": conversation_id,
                "gen_ai.prompt.name": prompt_name,
                "gen_ai.prompt.version": prompt_version,
            },
            start_time=int(start_time.timestamp() * 1_000_000_000),
        )
        if is_error:
            span.set_status(Status(StatusCode.ERROR, "upstream model call failed"))
            span.record_exception(RuntimeError("simulated LLM provider error"))
        span.end(end_time=int(end_time.timestamp() * 1_000_000_000))

    provider.shutdown()  # flushes the BatchSpanProcessor before the script exits
    print(
        f"Seeded {count} spans to {target} "
        f"({error_count} errors, {slow_count} slow, "
        f"{len(CONVERSATIONS)} conversations, {len(PROMPT_VERSIONS)} prompt versions, "
        f"split across 2 one-hour windows)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=["datadog", "sentry"])
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument(
        "--dsn", default=os.environ.get("SENTRY_DSN"), help="Sentry DSN (sentry target only)"
    )
    args = parser.parse_args()
    seed(args.target, args.count, args.dsn)

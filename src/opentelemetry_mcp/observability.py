"""OTel self-instrumentation for tracehub-mcp itself.

Implements the official OTel GenAI semantic conventions for MCP
(open-telemetry/semantic-conventions-genai, docs/gen-ai/mcp.md, status
"Development") for the server's own tools/call handling, mirroring
grafana/mcp-grafana's real, shipped precedent for this exact pattern: a
dedicated observability module, standard OTEL_* env var configuration, and
an --include-args-in-spans opt-in flag defaulting to False.

Fully opt-in / no-op when OTEL_EXPORTER_OTLP_ENDPOINT is not set: no
TracerProvider is configured and callers should skip registering the
instrumentation middleware entirely, so there is zero overhead and no
dependency on having a collector running for anyone who has not opted in.
"""

import logging
import os
import re
import time
from collections.abc import Iterable
from typing import Any

from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import Counter, Histogram, Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, ParentBased, Sampler
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer

from opentelemetry_mcp.security import escape_log_control_characters, redact_userinfo_in_text

logger = logging.getLogger(__name__)

_TRACER_NAME = "opentelemetry_mcp"
_METER_NAME = "opentelemetry_mcp"
_MAX_RESULT_ATTRIBUTE_CHARS = 2000
# The one label a tools/call may carry when its name does not resolve to a
# registered, enabled tool (see McpServerTracingMiddleware._bounded_tool_label).
_UNKNOWN_TOOL_LABEL = "unknown"
# Hard cap on a label before the registry is even consulted - no real tool
# name approaches this, and it bounds the lookup cost for a hostile peer.
_MAX_TOOL_NAME_LABEL_CHARS = 128
# Fixed size of an exported span status description (see
# _bounded_status_description): long enough for any backend error text this
# server produces (an httpx status line with its redacted URL), and small
# enough that a peer-chosen tool name echoed by fastmcp's NotFoundError
# cannot make the exported span grow with the request.
_MAX_STATUS_DESCRIPTION_CHARS = 256

# The operator, not the caller, decides whether this server's own spans are
# recorded. The SDK's default (parentbased_always_on) delegates a remote
# parent whose trace-flags are 00 to ALWAYS_OFF, so a caller-supplied
# traceparent could silently drop the server's span; this keeps root and
# remote-parent-not-sampled on ALWAYS_ON while still following a sampled
# parent's decision. Only used when OTEL_TRACES_SAMPLER is unset - an
# operator's own SDK-level choice always wins (see configure_tracing).
_OPERATOR_OWNED_SAMPLER: Sampler = ParentBased(root=ALWAYS_ON, remote_parent_not_sampled=ALWAYS_ON)

# Field-name fragments that signal the value next to them is a credential,
# not application data. Built up from parts rather than one literal list
# so this reads clearly as *detection* vocabulary, not an assignment.
_CREDENTIAL_FIELD_NAME_PARTS = [
    "api" + "_key",
    "app" + "_key",
    "sec" + "ret",
    "pass" + "word",
    "passwd",
    "to" + "ken",
    "auth",
]

# Deliberately narrow: only patterns with a clear contextual marker (a
# known credential prefix, or a field-name fragment immediately before the
# value). A generic "any long hex/base64 string" pattern would also catch
# trace_id/span_id - the very thing this tool exists to surface - and
# redact the actual answer to the user's query instead of a real secret.
_SECRET_PATTERNS = [
    # Authorization: Bearer <value> / bare "Bearer <value>"
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
    # <field name>=<value> or "<field name>": "<value>" pairs, where the
    # field name fragment itself signals a credential.
    re.compile(
        r"(?P<field>['\"]?(?:" + "|".join(_CREDENTIAL_FIELD_NAME_PARTS) + r")['\"]?"
        r"\s*[:=]\s*)(?P<quote>['\"]?)(?P<value>[^\s'\",}&]+)(?P=quote)",
        re.IGNORECASE,
    ),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key ID
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),  # GitHub tokens (ghp_/gho_/ghu_/ghs_/ghr_)
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # a common vendor secret-key prefix shape
]


def _redact_secrets(text: str) -> str:
    """Replace anything matching a known credential shape with [REDACTED],
    for the opt-in --include-args-in-spans/gen_ai.tool.call.arguments and
    .result span attributes and for the exported span status. Applied
    before truncation so a match near the truncation boundary cannot end
    up half-redacted, half-exposed.

    URL userinfo (httpx error text embeds the request URL verbatim) goes
    through security.redact_userinfo_in_text - the one implementation every
    other sink uses, so the two cannot drift - not a second pattern here."""
    text = redact_userinfo_in_text(text)
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            text = pattern.sub(lambda m: f"{m.group('field')}[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    return text


def _bounded_status_description(error: Exception) -> str:
    """str(error) as an exported span status description: credential shapes
    redacted, every control character repr-escaped, and cut to
    _MAX_STATUS_DESCRIPTION_CHARS with a trailing "...".

    _bounded_tool_label keeps a peer-chosen tools/call name out of the span
    name and the metric label, but fastmcp's own NotFoundError text
    ("Unknown tool: '<name>'") still carried that name verbatim into the
    status description - 5016 characters for a 5000-character name, raw NUL
    / newline / ESC included (runtime verification of PR #41, probe-4 P5b).
    The same bound applies to every failure, backend errors included, so no
    exception text can size the exported span. Redaction runs first so a
    credential near the cut cannot be half-exposed; the escape runs before
    the cut so that no raw control character can survive it - the cut may
    still fall inside an escape's TEXT (leaving a trailing "\\x1" before
    the "..."), which is cosmetic and accepted: cutting on an escape
    boundary would buy nothing the collector can act on.
    """
    text = escape_log_control_characters(_redact_secrets(str(error)))
    if len(text) > _MAX_STATUS_DESCRIPTION_CHARS:
        text = text[: _MAX_STATUS_DESCRIPTION_CHARS - 3] + "..."
    return text


def configure_tracing(
    service_name: str | None = None, *, trust_client_trace_context: bool = False
) -> bool:
    """Configure a real OTLP-exporting TracerProvider, but only if
    OTEL_EXPORTER_OTLP_ENDPOINT is actually set in the environment.

    Sampling is the operator's decision, not the caller's: unless the
    operator set OTEL_TRACES_SAMPLER themselves (in which case the SDK's
    own resolution of it wins untouched), the provider gets
    ``_OPERATOR_OWNED_SAMPLER`` - ParentBased with both root and
    remote-parent-not-sampled on ALWAYS_ON - so a remote parent carrying
    trace-flags=00 can never turn this server's own tools/call span into a
    non-recording one (the SDK default would, via its ALWAYS_OFF delegate).

    Also sets FastMCP's own ``telemetry_mode`` (fastmcp>=4.0's public,
    documented API - ``fastmcp.telemetry``) so FastMCP stops creating its
    own native spans for the lifetime of this process. Confirmed via a real
    end-to-end reproduction with an in-memory exporter (not just source
    reading) that without this, FastMCP's own middleware seam span PLUS its
    own internal ``server_span()`` both survive alongside this codebase's
    own McpServerTracingMiddleware span for the *same* logical tools/call
    operation - three SERVER-kind spans exported for one tool call. Both
    non-native modes suppress those spans; they differ in what they do with
    the W3C trace context a caller can place under a request's
    ``params._meta``:

    - ``trust_client_trace_context=False`` (the default, and main()'s
      choice for ``--transport http``): ``"off"`` - FastMCP neither
      extracts nor attaches the caller's ``_meta`` context, so an
      unauthenticated network peer cannot choose the trace identity
      (trace_id, parent span, tracestate) this server's span and its
      correlated log lines are filed under.
    - ``trust_client_trace_context=True`` (main()'s default for stdio,
      whose only caller is the trusted parent process):
      ``"propagation_only"`` - the caller's ``_meta`` context is attached
      as the parent, continuing the caller's distributed trace exactly as
      before.

    A narrower, context-manager-scoped fix (``suppress_fastmcp_telemetry()``
    around just this codebase's own ``call_next()``) was tried first and
    confirmed insufficient: FastMCP's middleware seam span opens *before*
    any user middleware runs, so nothing scoped inside this codebase's own
    middleware can reach back far enough to suppress it - this has to be
    set process-wide, at the point self-instrumentation is enabled.

    One accepted, deliberate tradeoff: FastMCP's own server_span() is the
    only place ``enduser.id``/``enduser.scope`` (auth) attributes would
    have been set - suppressing it means those never appear on any span,
    including for the bearer-token principal server.OperatorTokenVerifier
    establishes on the HTTP transport.

    Returns:
        True if tracing was configured, False if it was skipped (no
        endpoint configured). Callers must only register the
        instrumentation middleware when this returns True.
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False

    resource = Resource.create(
        {SERVICE_NAME: service_name or os.getenv("OTEL_SERVICE_NAME") or "tracehub-mcp"}
    )
    if os.getenv("OTEL_TRACES_SAMPLER"):
        # Leave sampler unset so the SDK resolves the operator's override.
        provider = TracerProvider(resource=resource)
    else:
        provider = TracerProvider(resource=resource, sampler=_OPERATOR_OWNED_SAMPLER)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)

    import fastmcp

    fastmcp.settings.telemetry_mode = "propagation_only" if trust_client_trace_context else "off"

    logger.info(
        f"OTel self-instrumentation enabled, exporting spans to {endpoint} "
        f"(client trace context {'trusted' if trust_client_trace_context else 'ignored'})"
    )
    return True


def configure_metrics(service_name: str | None = None) -> bool:
    """Direct structural analog of configure_tracing: OTLPMetricExporter ->
    PeriodicExportingMetricReader -> MeterProvider -> set_meter_provider(),
    gated behind the same OTEL_EXPORTER_OTLP_ENDPOINT check - no new env
    var needed, since the OTLP spec's per-signal endpoint override still
    falls back to this general one.

    Returns:
        True if metrics were configured, False if skipped (no endpoint
        configured).
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False

    resource = Resource.create(
        {SERVICE_NAME: service_name or os.getenv("OTEL_SERVICE_NAME") or "tracehub-mcp"}
    )
    reader = PeriodicExportingMetricReader(OTLPMetricExporter())
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)
    logger.info(f"OTel metrics self-instrumentation enabled, exporting metrics to {endpoint}")
    return True


_TRACE_CONTEXT_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - "
    "[trace_id=%(trace_id)s span_id=%(span_id)s] - %(message)s"
)


class TraceContextLogFilter(logging.Filter):
    """Stamps the 3 standardized field names from OTel's "Trace Context in
    non-OTLP Log Formats" spec onto every LogRecord: the real hex ids when
    a span is recording, or the spec's own all-zero placeholder otherwise -
    never omits the fields, so a formatter referencing them never
    KeyErrors.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid and span.is_recording():
            record.trace_id = f"{ctx.trace_id:032x}"
            record.span_id = f"{ctx.span_id:016x}"
            record.trace_flags = f"{ctx.trace_flags:02x}"
        else:
            record.trace_id = "0" * 32
            record.span_id = "0" * 16
            record.trace_flags = "00"
        return True


def install_trace_context_log_filter(handlers: Iterable[logging.Handler] | None = None) -> None:
    """Attach TraceContextLogFilter and upgrade formatters on the given
    handlers (default: the root logger's own, i.e. whatever
    logging.basicConfig wired up in server.py). Call only when
    configure_tracing() returned True - see server.py's main() gating."""
    target_handlers = list(handlers) if handlers is not None else logging.getLogger().handlers
    log_filter = TraceContextLogFilter()
    formatter = logging.Formatter(_TRACE_CONTEXT_LOG_FORMAT)
    for handler in target_handlers:
        handler.addFilter(log_filter)
        handler.setFormatter(formatter)


class McpServerTracingMiddleware(Middleware):
    """FastMCP middleware implementing the OTel GenAI semantic conventions
    for MCP tool calls: mcp.server spans, SERVER kind, required
    mcp.method.name, conditionally-required error.type/gen_ai.tool.name/
    gen_ai.operation.name.

    gen_ai.tool.call.arguments/gen_ai.tool.call.result are opt-in only, per
    the spec's own guidance that they may contain sensitive information.
    """

    def __init__(
        self,
        include_args: bool = False,
        tracer: Tracer | None = None,
        tool_call_duration: Histogram | None = None,
        tool_call_counter: Counter | None = None,
    ) -> None:
        self.include_args = include_args
        # Accepting an explicit tracer/instruments (rather than always
        # resolving them via trace.get_tracer()/metrics.get_meter() here)
        # lets tests inject ones bound to an in-memory exporter/reader
        # without touching the process-global TracerProvider/MeterProvider,
        # which OTel only allows setting once per process.
        self._tracer = tracer or trace.get_tracer(_TRACER_NAME)
        meter: Meter = metrics.get_meter(_METER_NAME)
        self._tool_call_duration = tool_call_duration or meter.create_histogram(
            name="mcp.server.tool.call.duration",
            unit="s",
            description="Duration of MCP tools/call requests handled by this server.",
        )
        self._tool_call_counter = tool_call_counter or meter.create_counter(
            name="mcp.server.tool.call.count",
            unit="1",
            description="Count of MCP tools/call requests, labeled by tool name and error status.",
        )

    @staticmethod
    async def _bounded_tool_label(context: MiddlewareContext[Any]) -> str:
        """The tools/call name as a metric label / span name, bounded by the
        server's own tool registry plus one fixed sentinel.

        fastmcp runs the middleware chain *before* it resolves the tool, so
        a peer-chosen string that matches no registered tool still reaches
        on_call_tool; echoing it verbatim into gen_ai.tool.name made every
        distinct unknown name a new, never-evicted aggregation in the
        opentelemetry-sdk (no cardinality limit) and two new exported
        series. The same registry lookup fastmcp performs afterwards
        (FastMCP.get_tool, async, honours disables and session transforms)
        decides whether the raw name is used; anything else - a non-string,
        an over-long name, no fastmcp context (unit tests, odd transports),
        a dead weakref, or any lookup error - collapses to the sentinel.
        Fail-closed by construction: it does not depend on which exception
        fastmcp raises for an unknown tool.
        """
        raw = getattr(context.message, "name", None)
        if not isinstance(raw, str) or len(raw) > _MAX_TOOL_NAME_LABEL_CHARS:
            return _UNKNOWN_TOOL_LABEL
        fastmcp_context = context.fastmcp_context
        if fastmcp_context is None:
            return _UNKNOWN_TOOL_LABEL
        try:
            tool = await fastmcp_context.fastmcp.get_tool(raw)
        except Exception:
            return _UNKNOWN_TOOL_LABEL
        return raw if tool is not None else _UNKNOWN_TOOL_LABEL

    async def on_call_tool(
        self, context: MiddlewareContext[Any], call_next: CallNext[Any, Any]
    ) -> Any:
        """Wrap a tools/call request in a spec-shaped mcp.server span."""
        tool_name = await self._bounded_tool_label(context)
        span_name = f"tools/call {tool_name}"

        attributes: dict[str, str] = {
            "mcp.method.name": "tools/call",
            "gen_ai.tool.name": tool_name,
            "gen_ai.operation.name": "execute_tool",
        }
        session_id = self._session_id(context)
        if session_id is not None:
            attributes["mcp.session.id"] = session_id

        if self.include_args:
            arguments = getattr(context.message, "arguments", None)
            if arguments is not None:
                attributes["gen_ai.tool.call.arguments"] = _redact_secrets(str(arguments))[
                    :_MAX_RESULT_ATTRIBUTE_CHARS
                ]

        start_time = time.perf_counter()
        # record_exception/set_status_on_exception=False: the SDK's own
        # default behavior on an exception propagating out of this `with`
        # block is to add an "exception" span event AND overwrite the
        # span's status - both built from the raw, unredacted
        # str(exception) - which would silently undo _record_error's own
        # redacted Status() below the moment the caught exception is
        # re-raised. _record_error already sets error.type and a redacted
        # Status for every failure, so the SDK's own duplicate, unredacted
        # write must be disabled rather than just outrun.
        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.SERVER,
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                result = await call_next(context)
            except Exception as e:
                self._record_error(span, e)
                self._record_tool_call_metrics(tool_name, start_time, error=True)
                raise
            if self.include_args:
                span.set_attribute(
                    "gen_ai.tool.call.result",
                    _redact_secrets(str(result))[:_MAX_RESULT_ATTRIBUTE_CHARS],
                )
            self._record_tool_call_metrics(tool_name, start_time, error=False)
            return result

    def _record_tool_call_metrics(self, tool_name: str, start_time: float, *, error: bool) -> None:
        """Safe to call unconditionally, even when configure_metrics() was
        never invoked - OTel's API-level no-op meter/instruments make this
        free, exactly like self._tracer already works today when tracing
        is disabled."""
        attrs = {"gen_ai.tool.name": tool_name, "error": str(error).lower()}
        self._tool_call_duration.record(time.perf_counter() - start_time, attrs)
        self._tool_call_counter.add(1, attrs)

    @staticmethod
    def _record_error(span: Span, error: Exception) -> None:
        span.set_attribute("error.type", type(error).__name__)
        # Unlike gen_ai.tool.call.arguments/.result, this runs
        # unconditionally on every failed tool call regardless of
        # include_args - so the exception message must always be
        # redacted, not just gated behind that flag, since a backend can
        # raise with a credential-shaped string embedded in its message.
        span.set_status(Status(StatusCode.ERROR, _bounded_status_description(error)))

    @staticmethod
    def _session_id(context: MiddlewareContext[Any]) -> str | None:
        fastmcp_context = context.fastmcp_context
        if fastmcp_context is None:
            return None
        try:
            return fastmcp_context.session_id
        except RuntimeError:
            return None

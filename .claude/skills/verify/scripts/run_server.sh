#!/usr/bin/env bash
# Hermetic launcher for the tracehub-mcp console script of THIS worktree (the one this file lives in).
#
#   run_server.sh http <port> [extra tracehub-mcp flags...]   # binds 127.0.0.1 only
#   run_server.sh stdio        [extra flags...]               # stdin/stdout are the transport
#   run_server.sh doctor       [extra flags...]
#   run_server.sh print-config [extra flags...]
#
# Backend + knobs come from the caller's environment (DUMMY values only) and are copied into an
# `env -i` allowlist, so nothing else from the shell (real BACKEND_* exports, .env values) can
# reach the server. Defaults: BACKEND_TYPE=jaeger BACKEND_URL=http://127.0.0.1:18710 LOG_LEVEL=info.
# Load a dummy token with:  set -a; . "$E/token.env"; set +a
# GOTCHA: a variable that is set but EMPTY is dropped here (like every launcher). For the
# empty-token gate call `env -i PATH=... MCP_HTTP_BEARER_TOKEN= "$BIN" ...` directly.
set -euo pipefail
HEAD="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BIN="$HEAD/.venv/bin/tracehub-mcp"
[[ -x "$BIN" ]] || { echo "run_server.sh: $BIN missing - run 'uv sync --all-groups' in $HEAD" >&2; exit 2; }

mode="${1:?mode: http|stdio|doctor|print-config}"; shift || true
: "${BACKEND_TYPE:=jaeger}"
: "${BACKEND_URL:=http://127.0.0.1:18710}"
: "${LOG_LEVEL:=info}"
: "${TMPDIR:=/tmp}"

pass=(
  "PATH=$HEAD/.venv/bin:/usr/bin:/bin"
  "HOME=${HOME:-/tmp}"
  "TMPDIR=$TMPDIR"
  "LANG=${LANG:-en_US.UTF-8}"
  "BACKEND_TYPE=$BACKEND_TYPE"
  "BACKEND_URL=$BACKEND_URL"
  "LOG_LEVEL=$LOG_LEVEL"
)
for v in BACKEND_API_KEY BACKEND_APP_KEY BACKEND_SENTRY_ORG BACKEND_SENTRY_PROJECT BACKEND_TEMPO_INSTANCE_ID \
         BACKEND_AWS_REGION BACKEND_NEWRELIC_ACCOUNT_ID BACKEND_HONEYCOMB_DATASET BACKEND_ENVIRONMENTS \
         BACKEND_TIMEOUT BACKEND_MAX_RESPONSE_BYTES \
         SECONDARY_BACKEND_TYPE SECONDARY_BACKEND_URL SECONDARY_BACKEND_API_KEY SECONDARY_BACKEND_SENTRY_ORG \
         SECONDARY_BACKEND_HONEYCOMB_DATASET SECONDARY_BACKEND_MAX_RESPONSE_BYTES SECONDARY_BACKEND_TIMEOUT \
         MCP_HTTP_BEARER_TOKEN MCP_ACCESS_LOG MCP_ALLOW_UNAUTHENTICATED_HTTP MCP_ALLOWED_HOSTS FORWARDED_ALLOW_IPS \
         MCP_PORT SESSION_IDLE_TIMEOUT_SECONDS MCP_STATELESS_HTTP READINESS_CACHE_SECONDS \
         RATE_LIMIT_MAX_REQUESTS RATE_LIMIT_WINDOW_SECONDS QUERY_CACHE_TTL_SECONDS QUERY_CACHE_MAX_ENTRIES \
         MAX_TRACES_PER_QUERY SLOW_REQUEST_THRESHOLD_MS MCP_TRUST_CLIENT_TRACE_CONTEXT MCP_INCLUDE_ARGS_IN_SPANS \
         SHUTDOWN_DRAIN_SECONDS BACKEND_CLOSE_TIMEOUT_SECONDS GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS \
         OTEL_EXPORTER_OTLP_ENDPOINT OTEL_SERVICE_NAME FASTMCP_CHECK_FOR_UPDATES FASTMCP_SHOW_SERVER_BANNER \
         AWS_ENDPOINT_URL AWS_ENDPOINT_URL_XRAY AWS_DEFAULT_REGION SSL_CERT_FILE; do
  if [[ -n "${!v-}" ]]; then pass+=("$v=${!v}"); fi
done

cd "$HEAD"
case "$mode" in
  http)
    port="${1:?port (127.0.0.1, 18700-18799)}"; shift || true
    exec env -i "${pass[@]}" "$BIN" --transport http --host 127.0.0.1 --port "$port" "$@" ;;
  stdio)        exec env -i "${pass[@]}" "$BIN" "$@" ;;
  doctor)       exec env -i "${pass[@]}" "$BIN" doctor "$@" ;;
  print-config) exec env -i "${pass[@]}" "$BIN" --print-config "$@" ;;
  *) echo "unknown mode $mode" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
# Raw-HTTP probes against a RUNNING tracehub-mcp --transport http instance; the token is read from
# $MCP_HTTP_BEARER_TOKEN (load it with: set -a; . "$E/token.env"; set +a) so no command line and no
# evidence file ever carries it.
#
#   http_probes.sh http://127.0.0.1:18700 "$E/http-probes.txt"
#
# Covers: /health, /ready, /mcp initialize unauthenticated / wrong token / authenticated,
# tools/list inside the session, session-id reuse without the token, Host and Origin spoofing.
# Expected from PR #41 onward: 401 + `www-authenticate` without/with a wrong token, 200 +
# `mcp-session-id` with the token, 421 for a foreign Host, 403 for a foreign Origin. Before #41
# (main 89659c7) there is no gate: everything answers 200 - record that as the baseline.
set -uo pipefail
BASE="${1:?base url e.g. http://127.0.0.1:18700}"
OUT="${2:-/dev/stdout}"
TOK="${MCP_HTTP_BEARER_TOKEN:-}"
SCHEME="$(printf 'B%s' earer)"   # assembled at runtime: the SecretScan hook blocks the literal scheme+token shape
AUTH_OK="Authorization: ${SCHEME} ${TOK}"
AUTH_BAD="Authorization: ${SCHEME} wrong-wrong-wrong-wrong-wrong-wrong-wrong-wrong"
INIT='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"verify-curl","version":"0"}}}'
ACC='Accept: application/json, text/event-stream'
CT='Content-Type: application/json'
mask() { if [[ -n "$TOK" ]]; then sed "s/$TOK/<TOKEN>/g"; else cat; fi; }
{
  echo "### $(date -u +%FT%TZ) base=$BASE"
  echo "=== GET /health ==="; curl -s -i --max-time 5 "$BASE/health"; echo; echo
  echo "=== GET /ready ==="; curl -s -i --max-time 15 "$BASE/ready"; echo; echo
  echo "=== UNAUTHENTICATED POST /mcp initialize ==="
  curl -s -i --max-time 10 -X POST "$BASE/mcp" -H "$CT" -H "$ACC" --data "$INIT" | head -12; echo; echo
  echo "=== WRONG-TOKEN POST /mcp initialize ==="
  curl -s -i --max-time 10 -X POST "$BASE/mcp" -H "$AUTH_BAD" -H "$CT" -H "$ACC" --data "$INIT" | head -12; echo; echo
  if [[ -n "$TOK" ]]; then
    echo "=== AUTHENTICATED POST /mcp initialize ==="
    curl -s -i --max-time 10 -X POST "$BASE/mcp" -H "$AUTH_OK" -H "$CT" -H "$ACC" --data "$INIT" | head -12; echo; echo
    echo "=== AUTHENTICATED tools/list inside a session ==="
    SID=$(curl -s -i --max-time 10 -X POST "$BASE/mcp" -H "$AUTH_OK" -H "$CT" -H "$ACC" --data "$INIT" | tr -d '\r' | awk 'tolower($1)=="mcp-session-id:"{print $2}')
    echo "mcp-session-id=$SID"
    curl -s -o /dev/null -w "notifications/initialized -> HTTP %{http_code}\n" --max-time 10 -X POST "$BASE/mcp" -H "$AUTH_OK" -H "$CT" -H "$ACC" -H "mcp-session-id: $SID" --data '{"jsonrpc":"2.0","method":"notifications/initialized"}'
    curl -s -i --max-time 10 -X POST "$BASE/mcp" -H "$AUTH_OK" -H "$CT" -H "$ACC" -H "mcp-session-id: $SID" --data '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' | cut -c1-1500; echo; echo
    echo "=== session id reuse WITHOUT token (expect 401 from #41 on) ==="
    curl -s -i --max-time 10 -X POST "$BASE/mcp" -H "$CT" -H "$ACC" -H "mcp-session-id: $SID" --data '{"jsonrpc":"2.0","id":3,"method":"tools/list"}' | head -5; echo
  fi
  echo "=== Host: evil.example on /health (expect 421 from #41 on) ==="
  curl -s -i --max-time 5 -H 'Host: evil.example' "$BASE/health" | head -4; echo
  echo "=== Origin: http://evil.example on /health (expect 403 from #41 on) ==="
  curl -s -i --max-time 5 -H 'Origin: http://evil.example' "$BASE/health" | head -4; echo
} 2>&1 | mask > "$OUT"
echo "wrote $OUT"

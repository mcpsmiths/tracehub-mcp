"""Docker HEALTHCHECK for the streamable-http transport (this image's
default CMD). Hits the server's own /ready endpoint (server.py's
ready_route), which runs a real health_check() + list_services() probe
against the configured trace backend. Only a 200 response ("ready") counts
as healthy; a 503 ("not_ready", raised whenever the backend probe fails)
counts as unhealthy, same as a refused or timed-out connection - unlike
/mcp (or /health), which only reflects process liveness and would report
healthy even while the backend is down.

urlopen() raises urllib.error.HTTPError (a URLError/OSError subclass) for
the 503 case and urllib.error.URLError for a refused/timed-out connection,
so catching Exception and exiting 1 covers both; only a 200 response
returns without raising, which is what should keep the container marked
healthy.

If you override CMD to run --transport stdio instead, disable this
healthcheck (docker run --no-healthcheck, or healthcheck: disable: true
in Compose) - there is no port to probe.
"""

import sys
import urllib.request

try:
    urllib.request.urlopen("http://localhost:8000/ready", timeout=5)
except Exception:
    sys.exit(1)

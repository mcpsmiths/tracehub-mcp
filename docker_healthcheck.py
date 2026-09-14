"""Docker HEALTHCHECK for the streamable-http transport (this image's
default CMD). Any real HTTP response (even the 406 a bare GET to the
/mcp endpoint returns) counts as healthy; only a refused or timed-out
connection does not.

If you override CMD to run --transport stdio instead, disable this
healthcheck (docker run --no-healthcheck, or healthcheck: disable: true
in Compose) - there is no port to probe.
"""

import sys
import urllib.error
import urllib.request

try:
    urllib.request.urlopen("http://localhost:8000/mcp", timeout=5)
except urllib.error.HTTPError:
    pass
except Exception:
    sys.exit(1)

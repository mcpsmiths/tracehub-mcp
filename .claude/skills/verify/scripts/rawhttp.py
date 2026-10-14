#!/usr/bin/env python3
"""Send ONE raw HTTP/1.1 request over a plain TCP socket to 127.0.0.1:PORT and print the reply.

    rawhttp.py PORT 'GET /health HTTP/1.1' 'Host: evil.example'
    rawhttp.py PORT 'GET /health HTTP/1.1' --no-host                       # no Host header at all
    rawhttp.py PORT --post-json '/mcp?access_token=__TOKEN__' '<json-rpc body>' [HEADER ...]

Use it where curl normalises the wire away (odd Host spellings, duplicate headers, control bytes,
`#`/`%3F` in the target). Everything is latin-1 so bytes 0x80-0xff can be sent. The literal
__TOKEN__ in any argument is replaced by $MCP_HTTP_BEARER_TOKEN, and the token is masked as
<TOKEN> in BOTH the echoed request and the response - a token never appears on a command line
or in the evidence.
"""

from __future__ import annotations

import os
import socket
import sys


def main() -> int:
    port = int(sys.argv[1])
    token = os.environ.get("MCP_HTTP_BEARER_TOKEN", "")
    args = [a.replace("__TOKEN__", token) for a in sys.argv[2:]]
    body = b""
    if args and args[0] == "--post-json":
        target, body_text, *headers = args[1:]
        body = body_text.encode("latin-1")
        lines = [
            f"POST {target} HTTP/1.1",
            "Content-Type: application/json",
            "Accept: application/json, text/event-stream",
            f"Content-Length: {len(body)}",
            *headers,
        ]
    else:
        lines = args
    if "--no-host" in lines:
        lines.remove("--no-host")
    elif not any(h.lower().startswith("host") for h in lines[1:]):
        lines.insert(1, "Host: 127.0.0.1")
    lines.append("Connection: close")
    raw = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body

    def mask(text: str) -> str:
        return text.replace(token, "<TOKEN>") if token else text

    sys.stdout.write(">>> " + mask(repr(raw))[:600] + "\n")
    with socket.create_connection(("127.0.0.1", port), timeout=15) as sock:
        sock.sendall(raw)
        chunks = []
        try:
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        except TimeoutError:
            chunks.append(b"<<recv timeout>>")
    text = mask(b"".join(chunks).decode("latin-1"))
    sys.stdout.write("<<< " + text[:1500] + ("\n<<< ...truncated\n" if len(text) > 1500 else "\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Drive an MCP stdio server over raw newline-delimited JSON-RPC - no SDK needed.

    stdio_rpc.py [OPTS] list-tools           -- CMD [ARG ...]
    stdio_rpc.py [OPTS] call TOOL JSON_ARGS  -- CMD [ARG ...]
    stdio_rpc.py [OPTS] raw MESSAGES.jsonl   -- CMD [ARG ...]   (lines sent verbatim: bare NaN etc.)

OPTS: --env K=V (repeatable)   --env-file FILE (K=V lines; 0600 dummy-credential files go here)
      --stderr FILE (server stderr, appended; default inherit)   --timeout S (per response, 60)
      --idle S (raw mode: stop after S quiet seconds, 2)   --no-handshake

The child inherits ONLY PATH HOME TMPDIR LANG LC_ALL PYTHONUNBUFFERED from this shell plus
--env/--env-file. Output: one JSON document on stdout (handshake, every message received in
order, exit code). The environment itself is never printed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import IO, Any

INHERIT = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "PYTHONUNBUFFERED")
INIT = {
    "jsonrpc": "2.0",
    "id": "init",
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "verify-skill", "version": "0"},
    },
}
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}


def build_env(pairs: list[str] | None, env_file: str | None) -> dict[str, str]:
    env = {k: os.environ[k] for k in INHERIT if k in os.environ}
    env.setdefault("PYTHONUNBUFFERED", "1")
    if env_file:
        with open(env_file, encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if line and not line.startswith("#"):
                    key, _, value = line.partition("=")
                    env[key.strip()] = value
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--env expects K=V, got {pair!r}")
        env[key] = value
    return env


class Server:
    def __init__(self, cmd: list[str], env: dict[str, str], stderr: IO[str] | None) -> None:
        self.proc = subprocess.Popen(  # noqa: S603 - the command IS the thing under test
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=env,
            text=True,
            bufsize=1,
        )
        self.lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None  # noqa: S101
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def send_text(self, text: str) -> None:
        assert self.proc.stdin is not None  # noqa: S101
        self.proc.stdin.write(text.rstrip("\n") + "\n")
        self.proc.stdin.flush()

    def send(self, msg: dict[str, Any]) -> None:
        self.send_text(json.dumps(msg))

    def recv(self, timeout: float) -> Any:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            return None
        if line is None:
            return None
        try:
            return json.loads(line)
        except ValueError:
            return {"_unparsed": line.rstrip("\n")}

    def wait_for(self, msg_id: Any, timeout: float) -> tuple[list[Any], Any]:
        deadline = time.monotonic() + timeout
        seen: list[Any] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return seen, None
            msg = self.recv(remaining)
            if msg is None:
                return seen, None
            seen.append(msg)
            if isinstance(msg, dict) and msg.get("id") == msg_id:
                return seen, msg

    def drain(self, idle: float) -> list[Any]:
        out: list[Any] = []
        while True:
            msg = self.recv(idle)
            if msg is None:
                return out
            out.append(msg)

    def stop(self) -> int | None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        return self.proc.returncode


def main() -> int:
    if "--" not in sys.argv:
        raise SystemExit(__doc__)
    split = sys.argv.index("--")
    ap = argparse.ArgumentParser(usage=__doc__)
    ap.add_argument("mode", choices=["list-tools", "call", "raw"])
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--env", action="append")
    ap.add_argument("--env-file")
    ap.add_argument("--stderr")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--idle", type=float, default=2.0)
    ap.add_argument("--no-handshake", action="store_true")
    args = ap.parse_args(sys.argv[1:split])
    cmd = sys.argv[split + 1 :]
    if not cmd:
        raise SystemExit("missing server command after --")

    out: dict[str, Any] = {"command": cmd, "mode": args.mode, "started": time.time()}
    with contextlib.ExitStack() as stack:
        errlog = (
            stack.enter_context(open(args.stderr, "a", encoding="utf-8")) if args.stderr else None
        )
        srv = Server(cmd, build_env(args.env, args.env_file), errlog)
        try:
            if not args.no_handshake:
                srv.send(INIT)
                seen, resp = srv.wait_for("init", args.timeout)
                out["handshake"] = {"response": resp, "other": [m for m in seen if m is not resp]}
                if resp is None:
                    out["error"] = "no initialize response (server exited or timed out)"
                    return 1
                srv.send(INITIALIZED)
            if args.mode == "list-tools":
                srv.send({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
                seen, resp = srv.wait_for(1, args.timeout)
                out["responses"] = seen
                tools = resp.get("result", {}).get("tools") if isinstance(resp, dict) else None
                if tools is not None:
                    out["tool_count"] = len(tools)
                    out["tool_names"] = sorted(t["name"] for t in tools)
            elif args.mode == "call":
                tool, args_json = args.rest[0], (args.rest[1] if len(args.rest) > 1 else "{}")
                params = {"name": tool, "arguments": json.loads(args_json)}
                srv.send({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params})
                seen, resp = srv.wait_for(1, args.timeout)
                out["responses"] = seen
                if isinstance(resp, dict):
                    out["is_error"] = resp.get("result", {}).get("isError", "error" in resp)
            elif args.mode == "raw":
                with open(args.rest[0], encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            srv.send_text(line)
                out["responses"] = srv.drain(args.idle)
            return 0
        finally:
            out["exit_code"] = srv.stop()
            print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())

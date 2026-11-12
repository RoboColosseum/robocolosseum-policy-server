#!/usr/bin/env python3
"""Local relay: Colosseum Policy Server -> a remote colosseum_server.py.

Colosseum Policy Server only accepts loopback model endpoints (http://127.0.0.1:<port>).
Run this on the Policy Server machine and set the model's `endpoint` to
http://127.0.0.1:<port>. It forwards GET /healthz and POST /act to the remote model and
reuses one connection between requests when the remote allows it. Python 3 standard
library only.

    # Tailscale (private tailnet, no credential):
    python3 policy_server_relay.py --remote http://colosseum-so101:9101 --port 9201
    # ngrok (public, basic auth required):
    export COLOSSEUM_TUNNEL_AUTH='user:password'     # credential from the model host
    python3 policy_server_relay.py --remote https://<name>.ngrok-free.dev --port 9201
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit


class Relay(BaseHTTPRequestHandler):
    def forward(self, method, body=None):
        server = self.server
        headers = {"Authorization": server.auth} if server.auth else {}
        if body is not None:
            headers["Content-Type"] = self.headers.get("Content-Type", "application/json")
        for attempt in (1, 2):  # the kept-alive connection may have been closed remotely
            try:
                if server.conn is None:
                    server.conn = server.conn_class(server.host, timeout=server.remote_timeout)
                server.conn.request(method, self.path, body=body, headers=headers)
                response = server.conn.getresponse()
                return response.status, response.read()
            except (http.client.HTTPException, OSError):
                if server.conn is not None:
                    server.conn.close()
                server.conn = None
                if attempt == 2:
                    raise

    def relay(self, method, path):
        if self.path != path:
            return self.reply(404, json.dumps({"error": "not found"}).encode())
        start = time.perf_counter()
        try:
            body = self.rfile.read(int(self.headers["Content-Length"])) if method == "POST" else None
            status, data = self.forward(method, body)
        except Exception as exc:
            status, data = 502, json.dumps({"error": f"relay: {type(exc).__name__}: {exc}"}).encode()
        print(f"{method} {path} -> {status} in {time.perf_counter() - start:.3f}s", file=sys.stderr)
        self.reply(status, data)

    def reply(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.relay("GET", "/healthz")

    def do_POST(self):
        self.relay("POST", "/act")

    def log_message(self, *args):  # one timing line per request is printed instead
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--remote", required=True,
                        help="model URL: http://<tailnet-host>:<port> or https://<name>.ngrok-free.dev")
    parser.add_argument("--port", type=int, default=9201, help="local port on 127.0.0.1 (default 9201)")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    remote = urlsplit(args.remote)
    if remote.scheme not in ("http", "https") or not remote.hostname or remote.path.strip("/"):
        sys.exit("--remote must be a bare URL, e.g. http://colosseum-so101:9101")
    credential = os.environ.get("COLOSSEUM_TUNNEL_AUTH", "")
    if remote.scheme == "https" and ":" not in credential:
        sys.exit("https (ngrok) needs COLOSSEUM_TUNNEL_AUTH='user:password' from the model host")

    server = HTTPServer(("127.0.0.1", args.port), Relay)
    server.host, server.remote_timeout, server.conn = remote.netloc, args.timeout, None
    server.conn_class = http.client.HTTPSConnection if remote.scheme == "https" else http.client.HTTPConnection
    server.auth = "Basic " + base64.b64encode(credential.encode()).decode() if credential else ""
    print(f"Relaying http://127.0.0.1:{args.port} -> {args.remote}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

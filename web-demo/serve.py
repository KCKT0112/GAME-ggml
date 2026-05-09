#!/usr/bin/env python3
"""Dev server for the GAME ggml web demo.

Sends Cross-Origin-Opener-Policy + Cross-Origin-Embedder-Policy headers so
the page is cross-origin-isolated — required for WebGPU + SharedArrayBuffer.
Also sets a long cache lifetime on the .gguf files so IndexedDB-like
caching by the browser works on reload.

Usage:
    python serve.py              # port 8080 in the current directory
    python serve.py --port 9000
"""
from __future__ import annotations

import argparse
import http.server
import pathlib
import socketserver


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm":    "application/wasm",
        ".gguf":    "application/octet-stream",
        ".js":      "application/javascript",
        ".mjs":     "application/javascript",
    }

    def end_headers(self) -> None:
        # Cross-origin isolation (required by WebGPU + threads).
        self.send_header("Cross-Origin-Opener-Policy",   "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        # Aggressive cache for the huge binaries so reloading is snappy.
        path = pathlib.Path(self.path.split("?", 1)[0])
        if path.suffix in {".wasm", ".gguf"}:
            self.send_header("Cache-Control", "public, max-age=3600")
        super().end_headers()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--bind", type=str, default="127.0.0.1")
    args = ap.parse_args()

    with socketserver.TCPServer((args.bind, args.port), Handler) as httpd:
        print(f"serving http://{args.bind}:{args.port}/")
        print("open http://127.0.0.1:{p}/index.html in a WebGPU-capable browser".format(p=args.port))
        print("  (Chrome 113+ / Safari 18+ / Edge 113+ / Firefox Nightly)")
        print()
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

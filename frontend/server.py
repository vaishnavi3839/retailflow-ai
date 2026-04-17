"""
RetailFlow AI — Frontend Dev Server
Serves the static HTML dashboard on http://localhost:3000.
Run this alongside the FastAPI backend (port 8000).

Usage:
    python server.py
"""

import http.server
import os
import socketserver

PORT = 3000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def log_message(self, fmt, *args):  # Quieter logging
        print(f"  {self.address_string()} — {fmt % args}")


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"\n  RetailFlow AI — Dashboard Server")
        print(f"  ─────────────────────────────────")
        print(f"  URL  : http://localhost:{PORT}")
        print(f"  Dir  : {DIRECTORY}")
        print(f"\n  Ctrl+C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped.")

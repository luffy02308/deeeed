"""
serve.py — Static file server with no-cache headers.
Replaces 'python -m http.server 8081' so the browser always
loads the latest JS files without caching issues.

Usage: python serve.py
"""
import http.server
import os

FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "frontend"
)
PORT = 8081


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    """Serves files with Cache-Control: no-store so browser always reloads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def end_headers(self):
        # Prevent browser from caching any file
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        # Only log non-200 responses to keep terminal clean
        if args and str(args[1]) != "200":
            super().log_message(fmt, *args)


if __name__ == "__main__":
    os.chdir(FRONTEND_DIR)
    print(f"[Frontend] Serving {FRONTEND_DIR}")
    print(f"[Frontend] http://localhost:{PORT}/pages/index.html")
    print(f"[Frontend] Cache-Control: no-store (browser always reloads JS)")
    print(f"[Frontend] Press Ctrl+C to stop.")
    with http.server.HTTPServer(("", PORT), NoCacheHandler) as httpd:
        httpd.serve_forever()

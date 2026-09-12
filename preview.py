"""Dependency-free UI preview. Does not connect to OpenAI or Neo4j."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from fashion_how_graphdb.web_service import catalog_preview, configuration, image_index


class PreviewHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path in ("/api/config", "/api/preview"):
            data = configuration(preview=True) if path.endswith("config") else catalog_preview()
            self.send_data(json.dumps(data, ensure_ascii=False).encode(), "application/json")
            return
        if path == "/":
            file = ROOT / "web" / "index.html"
        elif path.startswith("/assets/"):
            base = (ROOT / "web").resolve()
            file = (base / path.removeprefix("/assets/")).resolve()
            if not file.is_relative_to(base):
                self.send_error(404)
                return
        elif path.startswith("/images/"):
            file = image_index().get(path.removeprefix("/images/"))
        else:
            file = None
        if file is None or not file.is_file():
            self.send_error(404)
            return
        self.send_data(file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")

    def do_POST(self):
        self.send_data(json.dumps({"detail": "This is a catalog preview. Run python web_app.py with the server dependencies and credentials to enable live search."}).encode(), "application/json", 503)

    def send_data(self, data, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type + ("; charset=utf-8" if content_type.startswith("text/") or content_type == "application/json" else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    print("Catalog preview: http://127.0.0.1:7860 (live retrieval is disabled)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 7860), PreviewHandler).serve_forever()

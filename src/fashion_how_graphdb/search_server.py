"""Local web UI server for the Fashion-How GraphDB search pipeline."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from .cypher import (
    DEFAULT_NEO4J_DATABASE,
    DEFAULT_NEO4J_PASSWORD,
    DEFAULT_NEO4J_URI,
    DEFAULT_NEO4J_USER,
)
from .search import (
    DEFAULT_EMBEDDING_MODEL,
    build_search_cypher,
    extract_search_filters,
    needs_soft_rerank,
    rerank_search_results,
    search_catalog,
    soft_candidate_limit,
)
from .vlm import default_model, default_top_detail_model

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = ROOT / "fashion-how" / "image"
HTML_PATH = Path(__file__).with_name("search_viewer.html")
DEFAULT_MODELS = [
    default_model(),
    "gpt-5.4-mini",
    default_top_detail_model(),
    "gpt-4.1-mini",
    "gpt-4.1",
]


class SearchServer:
    def __init__(
        self,
        *,
        image_dir: Path,
        neo4j_uri: str | None,
        neo4j_user: str,
        neo4j_password: str | None,
        neo4j_database: str | None,
        embedding_model: str,
    ) -> None:
        self.image_dir = image_dir
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        self.embedding_model = embedding_model
        self._openai_client: Any | None = None
        self._neo4j_driver: Any | None = None
        self._image_index: dict[str, Path] | None = None

    def openai_client(self) -> Any:
        if self._openai_client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "Missing dependency: pip install openai "
                    f"(python={sys.executable})"
                ) from exc
            self._openai_client = OpenAI()
        return self._openai_client

    def neo4j_driver(self) -> Any:
        if not self.neo4j_uri:
            raise RuntimeError("NEO4J_URI is not set.")
        if not self.neo4j_password:
            raise RuntimeError("NEO4J_PASSWORD is not set.")
        if self._neo4j_driver is None:
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:
                raise RuntimeError(
                    "Missing dependency: pip install neo4j "
                    f"(python={sys.executable})"
                ) from exc
            self._neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
            )
        return self._neo4j_driver

    def image_index(self) -> dict[str, Path]:
        if self._image_index is None:
            self._image_index = {
                path.name: path
                for path in self.image_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            }
        return self._image_index

    def close(self) -> None:
        if self._neo4j_driver is not None:
            self._neo4j_driver.close()


def make_handler(app: SearchServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "FashionHowSearch/0.1"

        def do_GET(self) -> None:
            try:
                if self.path in {"/", "/search", "/search_viewer.html"}:
                    self._send_file(HTML_PATH, "text/html; charset=utf-8")
                    return
                if self.path == "/api/catalog":
                    self._send_json(
                        {
                            "catalog": search_catalog(),
                            "default_model": default_model(),
                            "models": list(dict.fromkeys(DEFAULT_MODELS)),
                            "runtime": runtime_diagnostics(),
                        }
                    )
                    return
                if self.path == "/api/health":
                    self._send_json(runtime_diagnostics())
                    return
                if self.path.startswith("/image/"):
                    self._send_image(unquote(self.path.removeprefix("/image/")))
                    return
                self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json(
                    {"error": exc.__class__.__name__, "message": str(exc)},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def do_POST(self) -> None:
            try:
                if self.path != "/api/search":
                    self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                    return
                payload = self._read_json()
                query = str(payload.get("query") or "").strip()
                if not query:
                    self._send_json(
                        {"error": "bad_request", "message": "query is required"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                model = str(payload.get("model") or default_model())
                limit = int(payload.get("limit") or 20)
                min_confidence = payload.get("min_confidence")
                if min_confidence in {"", None}:
                    min_confidence = None
                else:
                    min_confidence = float(min_confidence)
                min_score = payload.get("min_score")
                if min_score in {"", None}:
                    min_score = None
                else:
                    min_score = float(min_score)

                extraction = extract_search_filters(
                    query,
                    client=app.openai_client(),
                    model=model,
                )
                soft_rerank = needs_soft_rerank(extraction)
                cypher, params = build_search_cypher(
                    extraction,
                    limit=soft_candidate_limit(limit) if soft_rerank else limit,
                    min_confidence=min_confidence,
                    min_score=None if soft_rerank else min_score,
                )

                driver = app.neo4j_driver()
                with (
                    driver.session(database=app.neo4j_database)
                    if app.neo4j_database
                    else driver.session()
                ) as session:
                    rows = session.run(cypher, **params)
                    raw_results = [dict(row["result"]) for row in rows]
                results = rerank_search_results(
                    raw_results,
                    extraction,
                    client=app.openai_client() if soft_rerank else None,
                    embedding_model=app.embedding_model,
                    limit=limit,
                    min_score=min_score if soft_rerank else None,
                )

                self._send_json(
                    {
                        "query": query,
                        "model": model,
                        "extraction": extraction,
                        "cypher": cypher,
                        "params": params,
                        "results": results,
                    }
                )
            except Exception as exc:
                self._send_json(
                    {"error": exc.__class__.__name__, "message": str(exc)},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8")
            if not raw:
                return {}
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("Expected JSON object")
            return data

        def _send_file(self, path: Path, content_type: str | None = None) -> None:
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_image(self, image_file: str) -> None:
            image_name = Path(image_file).name
            path = app.image_index().get(image_name)
            if path is None:
                self._send_json(
                    {"error": "not_found", "message": f"Image not found: {image_name}"},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._send_file(path, mime)

        def _send_json(
            self,
            payload: dict[str, Any],
            *,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[search-ui] {self.address_string()} - {format % args}")

    return Handler


def runtime_diagnostics() -> dict[str, Any]:
    return {
        "python": sys.executable,
        "python_version": sys.version,
        "cwd": str(Path.cwd()),
        "openai": _module_status("openai"),
        "neo4j": _module_status("neo4j"),
    }


def _module_status(module_name: str) -> dict[str, Any]:
    try:
        module = __import__(module_name)
    except Exception as exc:
        return {
            "available": False,
            "error": exc.__class__.__name__,
            "message": str(exc),
        }
    return {
        "available": True,
        "version": getattr(module, "__version__", None),
        "file": getattr(module, "__file__", None),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a local web UI for Fashion-How GraphDB search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--image_dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--neo4j_uri", default=DEFAULT_NEO4J_URI)
    parser.add_argument("--neo4j_user", default=DEFAULT_NEO4J_USER)
    parser.add_argument("--neo4j_password", default=DEFAULT_NEO4J_PASSWORD)
    parser.add_argument("--neo4j_database", default=DEFAULT_NEO4J_DATABASE)
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")
    if not args.image_dir.exists():
        raise SystemExit(f"Image directory not found: {args.image_dir}")

    app = SearchServer(
        image_dir=args.image_dir,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
        embedding_model=args.embedding_model,
    )
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Fashion-How search UI: http://{args.host}:{args.port}")
    print(f"Python: {sys.executable}")
    print(f"OpenAI package: {runtime_diagnostics()['openai']}")
    try:
        httpd.serve_forever()
    finally:
        app.close()


if __name__ == "__main__":
    main()

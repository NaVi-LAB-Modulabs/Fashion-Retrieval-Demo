"""Fashion Search web UI and JSON API. Run with `python web_app.py`."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fashion_how_graphdb.web_service import (
    catalog_preview, configuration, image_index, retrieve, validate_request,
)

app = FastAPI(title="Fashion Search API", version="1.0.0")


@app.middleware("http")
async def response_headers(request: Request, call_next: Any) -> Any:
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/config")
def config() -> dict[str, Any]:
    return configuration()


@app.get("/api/preview")
def preview() -> dict[str, Any]:
    return catalog_preview()


@app.get("/images/{filename}", include_in_schema=False)
def image(filename: str) -> FileResponse:
    path = image_index().get(filename)
    if path is None:
        raise HTTPException(404, "Image not found")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})


@app.post("/api/search")
def search(payload: dict[str, Any]) -> Any:
    try:
        validate_request(payload)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not configuration()["ready"]:
        raise HTTPException(503, "Search is not configured. Set the OpenAI and Neo4j environment variables on the server.")
    try:
        return retrieve(payload)
    except Exception:
        # Keep credentials, provider responses and database connection details off the client.
        logging.getLogger(__name__).warning("Retrieval failed; check the configured providers.")
        return JSONResponse(status_code=502, content={
            "detail": "Search could not finish. Check the server's OpenAI and Neo4j connections, then try again.",
        })


app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")

if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))

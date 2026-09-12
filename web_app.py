"""Fashion Search web UI and JSON API. Run with `python web_app.py`."""

from __future__ import annotations

import logging
import json
import sys
from uuid import uuid4
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fashion_how_graphdb.web_service import (
    catalog_preview, configuration, image_index, retrieve, validate_request,
)
from fashion_how_graphdb.hf_images import fetch_image, resolve_images
from fashion_how_graphdb.diagnostics import failure_details

app = FastAPI(title="Fashion Search API", version="1.0.0")


@app.middleware("http")
async def response_headers(request: Request, call_next: Any) -> Any:
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.url.path.startswith("/api/") and not request.url.path.startswith("/api/image/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html", headers={"Cache-Control": "no-cache"})


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
    except Exception as exc:
        # Keep credentials, provider responses and database connection details off the client.
        details = failure_details(exc)
        error_id = uuid4().hex[:12]
        logging.getLogger(__name__).error("Retrieval failed error_id=%s diagnostics=%s", error_id, json.dumps(details))
        return JSONResponse(status_code=502, content={
            "detail": f"Search failed during {details['stage']}. Reference: {error_id}. Check the server logs for this reference.",
            "error_id": error_id,
            "stage": details["stage"],
        })


@app.post("/api/images")
def images(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return resolve_images(payload)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/image/{item_id}.jpg", include_in_schema=False)
def hf_image(item_id: str, refresh: bool = False) -> Response:
    try:
        payload, content_type = fetch_image(item_id, refresh=refresh)
    except KeyError as exc:
        logging.getLogger(__name__).warning("HF image not found item_id=%r", item_id)
        raise HTTPException(404, "Image not found") from exc
    except TimeoutError as exc:
        logging.getLogger(__name__).warning("HF image proxy timed out item_id=%s", item_id)
        raise HTTPException(504, "Image provider timed out") from exc
    except ValueError as exc:
        logging.getLogger(__name__).warning(
            "HF image proxy rejected item_id=%r reason=%s", item_id, str(exc)
        )
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "HF image proxy unavailable item_id=%s type=%s", item_id, type(exc).__name__
        )
        raise HTTPException(502, "Image provider unavailable") from exc
    return Response(payload, media_type=content_type,
                    headers={"Cache-Control": "public, max-age=3600"})


app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")

if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))

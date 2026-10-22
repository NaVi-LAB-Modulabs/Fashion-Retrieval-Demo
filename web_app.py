"""FashioNavi web UI and JSON API. Run with `python web_app.py`."""

from __future__ import annotations

import logging
import json
import sys
from contextlib import asynccontextmanager
from threading import Thread
from uuid import uuid4
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fashion_how_graphdb import baseline
from fashion_how_graphdb.web_service import (
    catalog_preview, configuration, image_index, retrieve, retrieve_baseline,
    validate_baseline_request, validate_request,
)
from fashion_how_graphdb.diagnostics import failure_details
from fashion_how_graphdb.style_references import reference_catalog
from fashion_how_graphdb import clip_onnx
from fashion_how_graphdb.diagnostics import retrieval_stage


@asynccontextmanager
async def lifespan(_: FastAPI) -> Any:
    # Download and load CLIP in the background so the first baseline search does not wait for it.
    if configuration()["baseline_ready"]:
        Thread(target=baseline.clip_encoder, daemon=True).start()
    yield


app = FastAPI(title="FashioNavi API", version="1.0.0", lifespan=lifespan)


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
    return FileResponse(ROOT / "web" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/config")
def config() -> dict[str, Any]:
    return configuration()


@app.get("/api/preview")
def preview() -> dict[str, Any]:
    return catalog_preview()


@app.get("/api/style-references")
def style_reference_catalog() -> dict[str, Any]:
    return reference_catalog()


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
        return failure_response(exc)


@app.post("/api/baseline")
def baseline_search(payload: dict[str, Any]) -> Any:
    try:
        validate_baseline_request(payload)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not configuration()["baseline_ready"]:
        raise HTTPException(503, "The CLIP baseline is not enabled on this server. Configure Neo4j.")
    try:
        return retrieve_baseline(payload)
    except Exception as exc:
        return failure_response(exc)


def failure_response(exc: Exception) -> JSONResponse:
    # Keep credentials, provider responses and database connection details off the client.
    details = failure_details(exc)
    error_id = uuid4().hex[:12]
    logging.getLogger(__name__).error("Retrieval failed error_id=%s diagnostics=%s", error_id, json.dumps(details))
    return JSONResponse(status_code=502, content={
        "detail": f"Search failed during {details['stage']}. Reference: {error_id}. Check the server logs for this reference.",
        "error_id": error_id,
        "stage": details["stage"],
    })


@app.post("/api/clip/search")
def clip_search(payload: dict[str, Any]) -> Any:
    try:
        clip_onnx.validate_request(payload)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not clip_onnx.configuration()["available"]:
        raise HTTPException(503, "Build the CLIP catalog index before searching.")
    try:
        with retrieval_stage("CLIP ONNX retrieval"):
            return clip_onnx.retrieve(payload)
    except Exception as exc:
        details = failure_details(exc)
        error_id = uuid4().hex[:12]
        logging.getLogger(__name__).error("CLIP failed error_id=%s diagnostics=%s", error_id, json.dumps(details))
        return JSONResponse(status_code=502, content={
            "detail": f"CLIP search failed. Reference: {error_id}. Check the server logs for this reference.",
            "error_id": error_id, "stage": details["stage"],
        })


app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")

if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))

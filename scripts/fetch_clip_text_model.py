"""Build step (Vercel runs it via pyproject's tool.vercel.scripts): fetch the CLIP text model.

The model (about 250 MB) is too large for git, so a deployment downloads it from
CLIP_TEXT_MODEL_URL and checks it against CLIP_TEXT_MODEL_SHA256. Set
CLIP_TEXT_MODEL_TOKEN when the URL needs a bearer token (for example a private
Hugging Face repository). Without a URL this step does nothing and the baseline stays off.

    python scripts/fetch_clip_text_model.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "models" / "clip_text_fp16w.onnx"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, expected_sha256: str, target: Path, token: str = "") -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as file:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                digest.update(chunk)
                file.write(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"checksum mismatch: got {digest.hexdigest()}, expected {expected_sha256}")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def main() -> int:
    url = os.getenv("CLIP_TEXT_MODEL_URL", "").strip()
    expected = os.getenv("CLIP_TEXT_MODEL_SHA256", "").strip().lower()
    if TARGET.is_file():
        if expected and sha256(TARGET) != expected:
            print(f"{TARGET.name} does not match CLIP_TEXT_MODEL_SHA256.", file=sys.stderr)
            return 1
        print(f"{TARGET.name} is present; skipping download.")
        return 0
    if not url:
        print("CLIP_TEXT_MODEL_URL is not set; the CLIP baseline will be disabled.")
        return 0
    if not expected:
        print("Set CLIP_TEXT_MODEL_SHA256 together with CLIP_TEXT_MODEL_URL.", file=sys.stderr)
        return 1
    try:
        fetch(url, expected, TARGET, os.getenv("CLIP_TEXT_MODEL_TOKEN", "").strip())
    except Exception as exc:  # a configured baseline that cannot be fetched should fail the build
        print(f"Could not fetch the CLIP text model: {exc}", file=sys.stderr)
        return 1
    print(f"Fetched {TARGET.name} ({TARGET.stat().st_size / 1e6:.0f} MB).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

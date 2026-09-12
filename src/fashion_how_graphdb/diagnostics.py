"""Search diagnostics without logging credentials, prompts or provider bodies."""

from contextlib import contextmanager
import re
import traceback


class RetrievalFailure(RuntimeError):
    def __init__(self, stage, cause):
        super().__init__(f"Search failed during {stage}")
        self.stage = stage
        self.cause = cause


@contextmanager
def retrieval_stage(stage):
    try:
        yield
    except Exception as exc:
        raise RetrievalFailure(stage, exc) from exc


def failure_details(exc):
    cause = exc.cause if isinstance(exc, RetrievalFailure) else exc
    metadata = getattr(cause, "provider_error", {}) or {}
    result = {
        "stage": exc.stage if isinstance(exc, RetrievalFailure) else "response",
        "error_type": type(cause).__name__,
    }
    for key, value in {
        "provider_type": metadata.get("type"),
        "provider_stage": metadata.get("stage"),
        "code": metadata.get("code", getattr(cause, "code", None)),
        "param": metadata.get("param", getattr(cause, "param", None)),
        "status": metadata.get("status_code", getattr(cause, "status_code", None)),
    }.items():
        if isinstance(value, (str, int)) and re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", str(value)):
            result[key] = value
    # Frame names/line numbers identify code failures without traceback messages or locals.
    result["frames"] = [f"{frame.name}:{frame.lineno}" for frame in traceback.extract_tb(cause.__traceback__)[-6:]]
    return result

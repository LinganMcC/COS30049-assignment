"""HTTP endpoints.

    POST /api/detect      analyse a passage           (concurrency guarded)
    GET  /api/model-info  model, features, test metrics and limits (read-only)
    GET  /api/health      liveness check used by the front end and the DoS demo
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool

from ..config import get_settings
from ..errors import ApiError
from ..schemas import (
    DetectRequest,
    DetectResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
)

router = APIRouter(prefix="/api", tags=["detector"])

ERRORS = {
    422: {"model": ErrorResponse, "description": "Input failed validation (too short, too long, wrong field)"},
    413: {"model": ErrorResponse, "description": "Request body larger than the limit"},
    503: {"model": ErrorResponse, "description": "Server busy or model unavailable; retry shortly"},
    500: {"model": ErrorResponse, "description": "Unexpected server error (no internal details)"},
}


def _detector(request: Request):
    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        raise ApiError(503, "The detector is not available right now.")
    return detector


@router.post(
    "/detect",
    response_model=DetectResponse,
    responses=ERRORS,
    summary="Score a passage for AI-generated writing",
    description=(
        "Splits the passage into sentences, scores each one with the Assignment 2 "
        "Logistic Regression model, and pools them into one document score with a "
        "confidence range. The submitted text is not stored or logged."
    ),
)
async def detect(payload: DetectRequest, request: Request):
    settings = get_settings()
    detector = _detector(request)

    if not settings.dos_guard_enabled:
        # UNDEFENDED path, kept for the before/after demo only:
        # Assignment 2's backtracking regex, run directly on the event loop, so
        # one slow request freezes every other request until it finishes.
        return _run(detector, payload.text, linear_regex=False)

    # Defence A: denial of service.
    #   (a) cap simultaneous analyses; reject the excess at once with 503,
    #   (b) run the CPU-bound work in a worker thread so the event loop keeps
    #       answering other requests,
    #   (c) the linear regex removes the expensive input altogether.
    guard = request.app.state.analysis_guard
    if not guard.try_acquire():
        raise ApiError(
            503,
            "The detector is busy with other checks. Please try again in a few seconds.",
            code="busy",
            headers={"Retry-After": "2"},
        )
    try:
        return await run_in_threadpool(_run, detector, payload.text, linear_regex=True)
    finally:
        guard.release()


def _run(detector, text: str, *, linear_regex: bool) -> dict:
    try:
        return detector.analyse(text, linear_regex=linear_regex)
    except ValueError as exc:
        if str(exc) == "no_sentences":
            raise ApiError(
                422,
                "We could not find any sentences with letters in this text.",
                code="validation_error",
            ) from None
        raise


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    responses={503: ERRORS[503], 500: ERRORS[500]},
    summary="Model details, features and evaluation results",
    description="Everything the About page shows, taken from the frozen model and its Assignment 2 evaluation files.",
)
async def model_info(request: Request):
    settings = get_settings()
    info = _detector(request).info()
    info["limits"] = {
        "min_words": settings.min_words,
        "max_words": settings.max_words,
        "max_chars": settings.max_chars,
    }
    return info


@router.get("/health", response_model=HealthResponse, summary="Is the API up and the model loaded?")
async def health(request: Request):
    return {"status": "ok", "model_loaded": getattr(request.app.state, "detector", None) is not None}

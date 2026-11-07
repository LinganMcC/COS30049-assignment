"""Create the FastAPI application.

Run from the backend folder:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .errors import register_error_handlers
from .ml.detector import Detector
from .routers import detect
from .security import (
    AnalysisGuard,
    BodySizeLimitMiddleware,
    SecurityHeadersMiddleware,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("inkprint")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model ONCE at start-up, not on every request."""
    settings = get_settings()
    app.state.detector = Detector.load(
        settings.model_path,
        settings.model_config_path,
        evasion_defence=settings.evasion_defence_enabled,
    )
    app.state.analysis_guard = AnalysisGuard(settings.max_concurrent_analyses)
    logger.info(
        "Defences: dos_guard=%s evasion=%s | CORS origin=%s",
        settings.dos_guard_enabled,
        settings.evasion_defence_enabled,
        settings.frontend_origin,
    )
    yield
    app.state.detector = None


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Inkprint API",
        description=(
            "Detects AI-generated English writing sentence by sentence, using the "
            "COS30049 Assignment 2 Logistic Regression model. Every error uses the "
            "same JSON shape: `{\"error\": {\"code\", \"message\", \"fields\"}}`."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.show_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.show_docs else None,
    )

    # The middleware added LAST runs FIRST. Order on the way in:
    # CORS -> security headers -> body size limit -> request log -> routes.
    # CORS is outermost so even an early 413 carries the CORS header and the
    # browser can show "too large" instead of a generic network error.
    @app.middleware("http")
    async def request_log(request: Request, call_next):
        # Method, path, status and time only. The submitted text is never logged.
        start = time.perf_counter()
        response = await call_next(request)
        logger.info(
            "%s %s -> %s (%.0f ms)",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - start) * 1000,
        )
        return response

    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],  # exactly one origin, never "*"
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        expose_headers=["Retry-After"],
        max_age=600,
    )

    register_error_handlers(app)
    app.include_router(detect.router)
    return app


app = create_app()

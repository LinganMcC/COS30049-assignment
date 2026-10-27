"""FastAPI application: wiring only - settings, middleware, routes, model loading.

Run from backend/:
    uvicorn app.main:app --port 8000 --no-server-header
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.errors import register_error_handlers
from app.ml.service import DetectorService
from app.routes import analyse, meta
from app.security.body_limit import BodyLimitMiddleware
from app.security.concurrency import InferenceGate
from app.security.headers import SecurityHeadersMiddleware

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at start-up: the model is loaded here, never per request."""
    settings = get_settings()
    app.state.detector = DetectorService(settings.model_dir, settings.normalise_input,
                                         settings.torch_threads)
    app.state.gate = InferenceGate(settings.inference_workers, settings.inference_queue,
                                   settings.inference_timeout_seconds)
    yield
    app.state.gate.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title='AI-Generated Content Detector API',
        description='Sentence-level AI-generated text detection with a transformer trained '
                    'from scratch. Every error uses the same JSON format: '
                    '`{"error": {"code", "message", "details"}}`.',
        version='1.0.0',
        lifespan=lifespan,
        # /docs can be switched off outside development (ENABLE_DOCS=false).
        docs_url='/docs' if settings.enable_docs else None,
        redoc_url=None,
        openapi_url='/openapi.json' if settings.enable_docs else None,
    )
    register_error_handlers(app)

    # Starlette runs the LAST added middleware first, so requests pass through:
    # security headers -> CORS -> body limit -> route. Headers and CORS are
    # therefore also applied to 413 rejections.
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],   # our front end only, never '*'
        allow_methods=['GET', 'POST'],
        allow_headers=['Content-Type'],
        expose_headers=['Retry-After'],
    )
    if settings.security_headers:
        app.add_middleware(SecurityHeadersMiddleware)

    app.include_router(analyse.router)
    app.include_router(meta.router)
    return app


app = create_app()

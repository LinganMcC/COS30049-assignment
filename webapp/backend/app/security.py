"""Security controls that sit in front of the model.

    AnalysisGuard             Defence A (2.7 Advanced, denial of service):
                              at most N analyses at once, 503 for the rest
    SecurityHeadersMiddleware part of Defence B (2.7 Intermediate, misconfiguration):
                              browser security headers on every response
    BodySizeLimitMiddleware   security hygiene (2.3): oversized bodies get 413
                              before they are parsed
"""

from __future__ import annotations

import threading

from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .errors import error_response



# ---------------------------------------------------------------------------
# Defence A: concurrency guard for the expensive model call
# ---------------------------------------------------------------------------
class AnalysisGuard:
    """At most `capacity` analyses run at once; extra requests get 503 at once.

    Rejecting immediately (instead of queueing) means a flood cannot build a
    backlog that keeps the server busy long after the attack stops, and the
    event loop stays free for every other request.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._slots = threading.BoundedSemaphore(capacity)

    def try_acquire(self) -> bool:
        return self._slots.acquire(blocking=False)

    def release(self) -> None:
        self._slots.release()


# ---------------------------------------------------------------------------
# Pure ASGI middleware (does not buffer the body itself)
# ---------------------------------------------------------------------------
class BodySizeLimitMiddleware:
    """Return 413 if the request body is larger than max_bytes.

    Checks Content-Length first, then counts the bytes actually received, so a
    chunked upload with no Content-Length cannot get round the limit.
    """

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        too_large = error_response(413, f"Request body is larger than {self.max_bytes // 1024} KB.")
        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_bytes:
                    await too_large(scope, receive, send)
                    return
            except ValueError:
                await error_response(400, "Invalid Content-Length header.")(scope, receive, send)
                return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # Raised inside the app, so the shared error handler turns it
                    # into the standard 413 JSON response.
                    raise StarletteHTTPException(413, f"Request body is larger than {self.max_bytes // 1024} KB.")
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except StarletteHTTPException as exc:
            if exc.status_code != 413 or response_started:
                raise
            await too_large(scope, receive, send)

class JsonCORSMiddleware(CORSMiddleware):
    """CORSMiddleware, except a rejected preflight uses the API's JSON error format.

    Starlette answers a disallowed preflight with plain text ("Disallowed CORS
    origin"). The decision and the CORS headers are left exactly as Starlette
    made them; only the body changes.
    """

    def preflight_response(self, request_headers: Headers) -> Response:
        response = super().preflight_response(request_headers)
        if response.status_code != 400:
            return response
        kept = {
            name: value
            for name, value in response.headers.items()
            if name not in ("content-type", "content-length")
        }
        return error_response(
            400,
            "This origin, method or header is not allowed to call the API.",
            code="cors_rejected",
            headers=kept,
        )

class SecurityHeadersMiddleware:
    """Standard defensive headers. API responses also forbid caching, because
    a result describes the user's own text."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        is_api = scope.get("path", "").startswith("/api/")

        async def add_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                extra = {
                    b"x-content-type-options": b"nosniff",
                    b"x-frame-options": b"DENY",
                    b"referrer-policy": b"no-referrer",
                    b"cross-origin-resource-policy": b"same-site",
                    b"permissions-policy": b"camera=(), microphone=(), geolocation=()",
                }
                if is_api:
                    extra[b"content-security-policy"] = b"default-src 'none'; frame-ancestors 'none'"
                    extra[b"cache-control"] = b"no-store"
                existing = {k.lower() for k, _ in headers}
                headers.extend((k, v) for k, v in extra.items() if k not in existing)
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, add_headers)

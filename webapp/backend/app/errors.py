"""One JSON error format for every failure: {"error": {"code", "message", "fields"}}.

No handler returns a stack trace, an exception message from library code, or
the user's input. Unexpected errors are logged on the server only.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("inkprint.errors")

_STATUS_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    500: "internal_error",
    503: "service_unavailable",
}


class ApiError(Exception):
    """Raise this from any route to return a clean error response."""

    def __init__(self, status_code: int, message: str, *, code: str | None = None, headers: dict | None = None):
        self.status_code = status_code
        self.message = message
        self.code = code or _STATUS_CODES.get(status_code, "error")
        self.headers = headers or {}


def error_response(status_code: int, message: str, *, code: str | None = None,
                   fields: list[dict] | None = None, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code or _STATUS_CODES.get(status_code, "error"),
                "message": message,
                "fields": fields or [],
            }
        },
        headers=headers,
    )


def _field_name(loc: tuple) -> str:
    parts = [str(p) for p in loc if p not in ("body",)]
    return ".".join(parts) or "body"


def _clean_message(err: dict) -> str:
    message = str(err.get("msg", "Invalid value"))
    # Pydantic prefixes custom validator messages with "Value error, ".
    return message.removeprefix("Value error, ")


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def handle_api_error(_: Request, exc: ApiError):
        return error_response(exc.status_code, exc.message, code=exc.code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(_: Request, exc: RequestValidationError):
        fields = []
        for err in exc.errors():
            if err.get("type") == "json_invalid":
                return error_response(400, "Request body is not valid JSON.", code="invalid_json")
            fields.append({"field": _field_name(tuple(err.get("loc", ()))), "message": _clean_message(err)})
        message = fields[0]["message"] if len(fields) == 1 else "Some fields are invalid."
        return error_response(422, message, fields=fields)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(_: Request, exc: StarletteHTTPException):
        messages = {
            404: "This address does not exist.",
            405: "This method is not allowed here.",
        }
        message = messages.get(exc.status_code) or (exc.detail if isinstance(exc.detail, str) else "Request failed.")
        return error_response(exc.status_code, message, headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):
        # Log type and route only: the exception text could quote user input.
        logger.error("Unhandled %s on %s %s", type(exc).__name__, request.method, request.url.path)
        return error_response(500, "Something went wrong on our side. Please try again.")

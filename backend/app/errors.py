"""One JSON error format for every failure, expected or not.

Every error response has this shape:
    {"error": {"code": "input_too_long", "message": "...", "details": [...]}}

The front end only needs one parser, and no handler ever returns a stack
trace, file path or exception text - unexpected errors are logged on the
server and the client gets a generic message.
"""
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger(__name__)


class AppError(Exception):
    """An error we expect and can explain to the user."""

    def __init__(self, status_code: int, code: str, message: str, headers: dict | None = None):
        self.status_code, self.code, self.message, self.headers = status_code, code, message, headers


def error_response(status_code: int, code: str, message: str,
                   details: list | None = None, headers: dict | None = None) -> JSONResponse:
    body = {'error': {'code': code, 'message': message, 'details': details or []}}
    return JSONResponse(status_code=status_code, content=body, headers=headers)


# Status codes Starlette raises itself (unknown route, wrong method...).
_HTTP_CODES = {404: 'not_found', 405: 'method_not_allowed', 413: 'payload_too_large',
               415: 'unsupported_media_type'}


def _describe(err: dict) -> dict:
    """Turn one Pydantic error into {'field', 'message'} for the user."""
    if err['type'] == 'json_invalid':   # loc is a character offset, not a field
        return {'field': 'body', 'message': 'The body is not valid JSON.'}
    field = '.'.join(str(p) for p in err['loc'] if p != 'body') or 'body'
    return {'field': field, 'message': err['msg'].removeprefix('Value error, ')}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error(_: Request, exc: AppError):
        return error_response(exc.status_code, exc.code, exc.message, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        # Report which field failed and why, but never echo the input back -
        # it could be 8,000 characters of someone's essay.
        details = [_describe(err) for err in exc.errors()]
        return error_response(422, 'invalid_input', 'The request is not valid.', details)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException):
        code = _HTTP_CODES.get(exc.status_code, 'http_error')
        message = exc.detail if isinstance(exc.detail, str) else 'Request failed.'
        return error_response(exc.status_code, code, message)

    @app.exception_handler(Exception)
    async def unexpected_error(_: Request, exc: Exception):
        log.exception('Unhandled error')   # full trace goes to the server log only
        return error_response(500, 'internal_error',
                              'Something went wrong on our side. Please try again.')

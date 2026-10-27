"""Reject oversized request bodies before they are read into memory.

FastAPI has no body size limit. Pydantic's max_length only runs after the whole
body has been received and parsed as JSON, so without this a client could send
hundreds of megabytes and the server would buffer all of it first.

Two checks, because Content-Length can be missing (chunked encoding) or false:
    - a declared Content-Length over the limit is refused straight away
    - otherwise the body is read here, chunk by chunk, and refused the moment
      it passes the limit; a body within the limit is replayed to the route
Buffering is safe because the limit is small (32 KB by default).
"""
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.errors import error_response


class BodyLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http' or scope['method'] in ('GET', 'HEAD', 'OPTIONS'):
            await self.app(scope, receive, send)
            return

        declared = dict(scope['headers']).get(b'content-length')
        if declared is not None and (not declared.isdigit() or int(declared) > self.max_bytes):
            await self._reject(scope, receive, send)
            return

        body = bytearray()
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            body += message.get('body', b'')
            if len(body) > self.max_bytes:
                await self._reject(scope, receive, send)
                return
            if not message.get('more_body', False):
                break

        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
            return await receive()   # afterwards, only a disconnect can arrive

        await self.app(scope, replay_receive, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = error_response(
            413, 'payload_too_large',
            f'The request body is larger than {self.max_bytes // 1024} KB.')
        await response(scope, receive, send)

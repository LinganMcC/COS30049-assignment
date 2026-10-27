"""Security headers on every response (misconfiguration defence).

FastAPI sets none of these by default, and OWASP ZAP flags each one missing.
The API only ever returns JSON, so its Content-Security-Policy can forbid
everything; /docs needs Swagger UI's script and stylesheet from jsDelivr.
"""
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

API_CSP = "default-src 'none'; frame-ancestors 'none'"
DOCS_CSP = ("default-src 'none'; script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
            "style-src https://cdn.jsdelivr.net; img-src https://fastapi.tiangolo.com data:; "
            "connect-src 'self'; frame-ancestors 'none'")
DOCS_PATHS = ('/docs', '/redoc')

COMMON_HEADERS = {
    'X-Content-Type-Options': 'nosniff',      # never guess a different content type
    'X-Frame-Options': 'DENY',                # no clickjacking via iframes
    'Referrer-Policy': 'no-referrer',
    'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Resource-Policy': 'same-site',
    'Cache-Control': 'no-store',              # results contain the user's text (Req 11)
}


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        csp = DOCS_CSP if scope['path'].startswith(DOCS_PATHS) else API_CSP

        async def send_with_headers(message: Message) -> None:
            if message['type'] == 'http.response.start':
                headers = MutableHeaders(scope=message)
                for name, value in COMMON_HEADERS.items():
                    headers[name] = value
                headers['Content-Security-Policy'] = csp
            await send(message)

        await self.app(scope, receive, send_with_headers)

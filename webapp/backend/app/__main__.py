"""Start the API:  python -m app            (add --reload while developing)

Part of Defence B (misconfiguration): uvicorn's defaults are tightened here.
  server_header=False  responses don't advertise the server software
  proxy_headers=False  X-Forwarded-For sent by clients is not trusted
"""

import argparse
import sys

import uvicorn

SUPPORTED = (3, 15)


def main() -> None:
    if sys.version_info[:2] != SUPPORTED:
        print(
            f"Warning: this project is pinned for Python {SUPPORTED[0]}.{SUPPORTED[1]}; "
            f"you are running {sys.version.split()[0]}. If installing or starting fails, "
            "recreate the virtual environment with Python 3.15 (see README section 1).",
            file=sys.stderr,
        )
    parser = argparse.ArgumentParser(description="Run the Inkprint API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        proxy_headers=False,
        server_header=False,
    )


if __name__ == "__main__":
    main()

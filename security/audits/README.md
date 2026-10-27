# Audits (misconfiguration and vulnerable dependencies)

Save each report here twice, `before/` and `after/` the fixes, so the video and
README can show what changed. Every finding is either fixed or justified in the
main README.

## Python dependencies

```bash
cd backend
python -m pip install -r requirements-dev.txt
pip-audit -r requirements.txt -f markdown -o ../security/audits/after/pip-audit.md
```

## JavaScript dependencies

```bash
cd frontend
npm audit --json > ../security/audits/after/npm-audit.json
```

## OWASP ZAP baseline scan

Scan the built front end (it serves the strict CSP) and the API. Docker is the
simplest way to run ZAP; on macOS, `host.docker.internal` reaches your machine.

```bash
cd frontend && npm run build && npm run preview      # terminal 1
cd backend && uvicorn app.main:app --port 8000 --no-server-header   # terminal 2

docker run --rm -v "$PWD/security/audits/after:/zap/wrk" ghcr.io/zaproxy/zaproxy:stable \
  zap-baseline.py -t http://host.docker.internal:5173 -r zap-frontend.html
docker run --rm -v "$PWD/security/audits/after:/zap/wrk" ghcr.io/zaproxy/zaproxy:stable \
  zap-api-scan.py -t http://host.docker.internal:8000/openapi.json -f openapi -r zap-api.html
```

For the "before" scan, start the back end with `SECURITY_HEADERS=false` and the
front end with `npm run dev` (no CSP).

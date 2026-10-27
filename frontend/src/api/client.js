// The only place the front end talks to the back end.
// Every failure - HTTP error, network down, timeout - becomes an ApiError with
// the same fields as the back end's error JSON, so components handle one shape.

const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '');
const TIMEOUT_MS = 20_000;

export class ApiError extends Error {
  constructor(status, code, message, details = [], retryAfter = null) {
    super(message);
    this.status = status;          // 0 when the server could not be reached
    this.code = code;
    this.details = details;
    this.retryAfter = retryAfter;  // seconds, sent with 503 "server busy"
  }
}

async function request(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  let response;
  try {
    response = await fetch(`${API_URL}${path}`, { ...options, signal: controller.signal });
  } catch (err) {
    throw err.name === 'AbortError'
      ? new ApiError(0, 'timeout', 'The server took too long to respond. Please try again.')
      : new ApiError(0, 'network_error', 'Could not reach the server. Check it is running and try again.');
  } finally {
    clearTimeout(timer);
  }

  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const err = body?.error;
    const retryAfter = Number(response.headers.get('Retry-After')) || null;
    throw new ApiError(
      response.status,
      err?.code ?? 'http_error',
      err?.message ?? `The server returned an error (${response.status}).`,
      err?.details ?? [],
      retryAfter,
    );
  }
  if (body === null) {
    throw new ApiError(response.status, 'bad_response', 'The server sent a response we could not read.');
  }
  return body;
}

export function getConfig() {
  return request('/api/config');
}

export function analyseText(text) {
  return request('/api/analyse', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
}

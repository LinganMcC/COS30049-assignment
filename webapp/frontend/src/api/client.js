// The only file that talks to the back end.
import axios from 'axios'

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const http = axios.create({
  baseURL: API_BASE_URL,
  timeout: 20000,
  headers: { 'Content-Type': 'application/json' },
})

/**
 * Turn any failure into one shape the pages can show:
 *   { kind, status, code, message, fields, retryAfter }
 * The back end always sends {"error": {"code", "message", "fields"}}.
 */
export function toApiError(err) {
  if (axios.isCancel(err)) return { kind: 'cancelled' }
  if (err.code === 'ECONNABORTED') {
    return { kind: 'timeout', message: 'The detector took too long to answer. Try again.' }
  }
  if (!err.response) {
    return {
      kind: 'network',
      message: `Nothing answered at ${API_BASE_URL}. Check the back end is running, then try again.`,
    }
  }
  const { status, data, headers } = err.response
  const body = data && data.error ? data.error : {}
  return {
    kind: 'http',
    status,
    code: body.code || 'error',
    message: body.message || 'The detector could not handle this request.',
    fields: Array.isArray(body.fields) ? body.fields : [],
    retryAfter: Number(headers?.['retry-after']) || null,
  }
}

export async function detectText(text, { signal } = {}) {
  try {
    const { data } = await http.post('/api/detect', { text }, { signal })
    return data
  } catch (err) {
    throw toApiError(err)
  }
}

export async function getModelInfo({ signal } = {}) {
  try {
    const { data } = await http.get('/api/model-info', { signal })
    return data
  } catch (err) {
    throw toApiError(err)
  }
}

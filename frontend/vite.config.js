import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// Security headers for the front end itself (misconfiguration defence).
// The strict CSP is applied to `vite preview` (the built app, which is what
// OWASP ZAP scans). The dev server can't use it: React Fast Refresh injects
// an inline script that `script-src 'self'` would block.
const baseHeaders = {
  'X-Content-Type-Options': 'nosniff',
  'X-Frame-Options': 'DENY',
  'Referrer-Policy': 'no-referrer',
  'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
};

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), 'VITE_');
  const apiOrigin = new URL(env.VITE_API_URL || 'http://localhost:8000').origin;
  const csp = [
    "default-src 'self'",
    "script-src 'self'",
    // Recharts positions tooltips with inline style attributes.
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data:",
    `connect-src 'self' ${apiOrigin}`,
    "frame-ancestors 'none'",
    "base-uri 'none'",
    "form-action 'none'",
  ].join('; ');

  return {
    plugins: [react()],
    server: { port: 5173, strictPort: true, headers: baseHeaders },
    preview: {
      port: 5173,
      strictPort: true,
      headers: { ...baseHeaders, 'Content-Security-Policy': csp },
    },
    // Recharts makes the bundle ~650 KB; fine for this app, so raise the warning limit.
    build: { sourcemap: false, chunkSizeWarningLimit: 800 },
  };
});

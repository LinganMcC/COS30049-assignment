import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Port 5173 is fixed because the back end's CORS allows exactly this origin.
export default defineConfig({
  plugins: [react()],
  server: { host: 'localhost', port: 5173, strictPort: true },
  preview: { host: 'localhost', port: 5173, strictPort: true },
})

import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'
import { defineConfig } from 'vite'

// HTTPS so a phone on the same Wi-Fi can open the camera; API/WS proxied to FastAPI on :8000.
export default defineConfig({
  plugins: [react(), basicSsl()],
  server: {
    host: true,
    proxy: {
      '/api': 'http://localhost:8000',
      '/clips': 'http://localhost:8000',
      '/pictures': 'http://localhost:8000',
      '/ws': { target: 'ws://localhost:8000', ws: true },
    },
  },
})

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import basicSsl from '@vitejs/plugin-basic-ssl'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    basicSsl()
  ],
  server: {
    host: '0.0.0.0', // Listen on all network interfaces
    https: true,     // Enable HTTPS
    port: 8080,      // Specify a port
    proxy: {
      // Proxy requests from /api to the backend server
      '/api': {
        target: 'http://127.0.0.1:8000', // Your backend server
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''), // Remove /api prefix
      },
    }
  }
})

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import type { PluginOption } from 'vite';

// Función para dividir chunks manualmente
const manualChunks = (id: string): string | undefined => {
  if (id.includes('node_modules/react') || id.includes('node_modules/react-dom')) {
    return 'vendor';
  }
  if (id.includes('node_modules/lucide-react')) {
    return 'icons';
  }
  return undefined;
};

export default defineConfig({
  plugins: [react() as PluginOption],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://crackguard-backend:5000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    minify: 'terser',
    rollupOptions: {
      output: {
        manualChunks,
      },
    },
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
      },
      format: {
        comments: false,
      },
    },
  },
});
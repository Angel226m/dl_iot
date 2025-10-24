import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Función para dividir chunks manualmente
const manualChunks = (id: string) => {
  if (id.includes('node_modules/react') || id.includes('node_modules/react-dom')) {
    return 'vendor';
  }
  if (id.includes('node_modules/lucide-react')) {
    return 'icons';
  }
  return null;
};

export default defineConfig({
  plugins: [react()],
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
    } as any, // 👈 Solución: forzamos el tipo para evitar el error de sobrecarga
  },
});

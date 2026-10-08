import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
export default defineConfig({
  plugins: [react()],
  base: './',
  server: { host: '127.0.0.1', port: 8081, strictPort: true },
  test: { include: ['src/**/*.test.ts'], environment: 'node' },
});

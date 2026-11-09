import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 30000,
  expect: { timeout: 5000 },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 2 : 2,
  reporter: [['list'], ['html', { outputFolder: 'playwright-report', open: 'never' }]],
  outputDir: 'test-results',
  use: {
    baseURL: 'http://127.0.0.1:8081',
    browserName: 'chromium',
    launchOptions: { executablePath: process.env.CHROME_PATH ?? (process.env.CI ? undefined : '/bin/google-chrome'), args: ['--no-sandbox'] },
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'iphone-390x844', use: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 1 } },
    { name: 'android-412x915', use: { viewport: { width: 412, height: 915 }, isMobile: true, hasTouch: true, deviceScaleFactor: 1 } },
  ],
  webServer: {
    command: 'npm run dev',
    url: 'http://127.0.0.1:8081',
    reuseExistingServer: !process.env.CI,
    timeout: 60000,
  },
});

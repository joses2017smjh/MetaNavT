import { chromium } from '@playwright/test';
import { mkdir, copyFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const output = resolve('demo');
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH || '/bin/google-chrome',
  args: ['--no-sandbox'],
});
const context = await browser.newContext({
  viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true,
  deviceScaleFactor: 2,
  recordVideo: { dir: '/tmp/agent-field-video', size: { width: 390, height: 844 } },
});
const page = await context.newPage();
const errors = [];
page.on('pageerror', error => errors.push(error.message));
const pause = () => page.waitForTimeout(1600);
const capture = async (name) => {
  await pause();
  await page.screenshot({ path: `${output}/${name}.png` });
};
try {
  await page.goto(process.env.DEMO_URL || 'http://127.0.0.1:8081');
  await page.getByRole('heading', { name: 'Your agents. One field.' }).waitFor();
  await page.evaluate(() => document.fonts.ready);
  await capture('01-home');
  await page.getByRole('button', { name: 'Research', exact: true }).click();
  await page.getByRole('button', { name: 'Learning rate', exact: false }).click();
  await page.getByRole('heading', { name: 'Follow the evidence' }).waitFor();
  await capture('02-research');
  await page.locator('.source-card').filter({ has: page.getByRole('heading', { name: 'run_047.yaml', exact: true }) }).click();
  await page.getByRole('dialog', { name: 'Source detail' }).waitFor();
  await capture('03-source');
  await page.getByRole('button', { name: 'Save source', exact: true }).click();
  await pause();
  await page.getByRole('button', { name: 'Close detail' }).click();
  await page.getByRole('button', { name: 'Ask agent about this question' }).click();
  await page.locator('.agent-result').waitFor();
  await page.locator('.agent-panel').evaluate(el => window.scrollTo({ top: window.scrollY + el.getBoundingClientRect().top - 60, behavior: 'instant' }));
  await capture('09-research-agent');
  await page.getByText('View research agent trace', { exact: false }).click();
  await capture('10-research-agent-trace');
  await page.getByRole('button', { name: 'Match Lab', exact: true }).click();
  await page.getByRole('button', { name: 'Explore fixture report' }).click();
  await page.getByRole('heading', { name: 'Three possible outcomes' }).waitFor();
  await page.locator('.probability-card').evaluate(el => window.scrollTo({ top: window.scrollY + el.getBoundingClientRect().top - 100, behavior: 'instant' }));
  await capture('04-match');
  await page.getByRole('button', { name: 'Agent trace', exact: false }).click();
  await capture('05-agent-trace');
  await page.getByRole('button', { name: 'Close detail' }).click();
  await page.getByRole('button', { name: 'Save report', exact: true }).click();
  await pause();
  await page.getByLabel('Choose a fixture').selectOption('approval-required');
  await page.getByRole('button', { name: 'Explore fixture report' }).click();
  await page.getByText('Operator review required', { exact: true }).waitFor();
  await page.locator('.approval-card').evaluate(el => window.scrollTo({ top: window.scrollY + el.getBoundingClientRect().top - 150, behavior: 'instant' }));
  await capture('06-operator-review');
  await page.getByRole('button', { name: 'Saved', exact: true }).click();
  await capture('07-saved');
  await page.getByRole('button', { name: 'Connections', exact: true }).click();
  await capture('08-connections');
  await context.close();
  await copyFile(await page.video().path(), `${output}/feature-demo.webm`);
  await writeFile(`${output}/capture.json`, JSON.stringify({
    captured_at: new Date().toISOString(),
    scope: 'Actual app in Chromium at a 390×844 mobile viewport; not a native-device recording.',
    mode: 'Bundled offline synthetic fixtures',
    steps: ['Research search', 'Inspect source and byte span', 'Save source', 'Match probabilities and expected goals', 'Recorded agent tool trace', 'Operator-review pause', 'Persistent saved library', 'Connection settings', 'Read-only scripted research agent and cited excerpts'],
    page_errors: errors,
  }, null, 2) + '\n');
  if (errors.length) throw new Error(`Demo had browser errors: ${errors.join('; ')}`);
} finally { await browser.close(); }

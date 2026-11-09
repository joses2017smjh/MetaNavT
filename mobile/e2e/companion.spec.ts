import { expect, test, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
const researchCapture = JSON.parse(readFileSync(resolve(process.cwd(), '../app/mobile/demo-response.json'), 'utf8'));

async function connect(page: Page, token = '') {
  await page.getByRole('button', { name: 'Connections', exact: true }).click();
  await page.getByLabel('Connect services', { exact: false }).check();
  await page.getByLabel('MetaNavT service URL').fill('https://research.example.test');
  await page.getByLabel('Soccer mobile adapter URL').fill('https://soccer.example.test');
  await page.getByLabel(/Soccer access token/).fill(token);
  await page.getByRole('button', { name: 'Save workspace settings' }).click();
  await expect(page.getByRole('status')).toContainText('Connections saved');
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

test('research sources save once and persist through a reload', async ({ page }) => {
  await page.getByRole('button', { name: 'Research', exact: true }).click();
  await page.getByRole('button', { name: 'Learning rate', exact: true }).click();
  await expect(page.getByText('Follow the evidence', { exact: true })).toBeVisible();
  await page.locator('.source-card').filter({ has: page.getByRole('heading', { name: 'run_047.yaml', exact: true }) }).click();
  const detail = page.getByRole('dialog', { name: 'Source detail' });
  await expect(detail).toBeVisible();
  await expect(detail).toContainText('learning_rate: 0.0003');
  await expect(detail).toContainText('SYNTHETIC FIXTURE');
  await detail.getByRole('button', { name: 'Save source' }).click();
  await detail.getByRole('button', { name: 'Save source' }).click();
  await expect(page.getByRole('status')).toContainText('already saved');
  await page.getByRole('button', { name: 'Close detail' }).click();
  await page.getByRole('button', { name: 'Saved', exact: true }).click();
  await expect(page.locator('.saved-card')).toHaveCount(1);
  await page.reload();
  await page.getByRole('button', { name: 'Saved', exact: true }).click();
  await expect(page.locator('.saved-card')).toHaveCount(1);
  await page.getByRole('button', { name: /RESEARCH SOURCE.*configs\/run_047.yaml/ }).click();
  await expect(page.getByRole('dialog', { name: 'Saved source' })).toContainText('learning_rate: 0.0003');
});

test('complete soccer report shows probabilities, expected goals and all 11 agent calls', async ({ page }) => {
  await page.getByRole('button', { name: 'Match Lab', exact: true }).click();
  await page.getByRole('button', { name: 'Explore fixture report' }).click();
  await expect(page.getByText('Three possible outcomes', { exact: true })).toBeVisible();
  await expect(page.locator('.probability-values strong')).toHaveText(['48%', '28%', '24%']);
  await expect(page.locator('.xg-values strong')).toHaveText(['1.50', '0.83']);
  await expect(page.getByText('Synthetic model · no real-match accuracy claim')).toBeVisible();
  await page.getByRole('button', { name: /11 recorded tool calls.*Agent trace/ }).click();
  const trace = page.getByRole('dialog', { name: 'Agent trace' });
  await expect(trace).toContainText('RECORDED FIXTURE RUN');
  await expect(trace.locator('.trace-list li')).toHaveCount(11);
  await expect(trace).toContainText('data');
});

test('operator-review fixture displays the pause without a forecast or approval action', async ({ page }) => {
  await page.getByRole('button', { name: 'Match Lab', exact: true }).click();
  await page.getByLabel('Choose a fixture').selectOption('approval-required');
  await page.getByRole('button', { name: 'Explore fixture report' }).click();
  await expect(page.getByText('Operator review required', { exact: true })).toBeVisible();
  await expect(page.getByText('Three possible outcomes', { exact: true })).toHaveCount(0);
  await expect(page.locator('.probability-card')).toHaveCount(0);
  await expect(page.getByRole('button', { name: /approve|execute/i })).toHaveCount(0);
  await page.getByRole('button', { name: /Agent trace/ }).click();
  await expect(page.getByRole('dialog', { name: 'Agent trace' })).toContainText('does not approve or execute');
});

test('connection settings persist URLs and never persist the session token', async ({ page }) => {
  const token = 'PRIVATE_MOBILE_SESSION_ABC123';
  await connect(page, token);
  await page.getByLabel('Research agent URL', { exact: false }).fill('https://agent.example.test');
  await page.getByLabel('Research agent token', { exact: false }).fill('PRIVATE_AGENT_SESSION_DEF456');
  await page.getByRole('button', { name: 'Save workspace settings' }).click();
  const state = await page.evaluate(() => ({ ...localStorage }));
  expect(JSON.stringify(state)).not.toContain(token);
  expect(JSON.stringify(state)).not.toContain('PRIVATE_AGENT_SESSION_DEF456');
  expect(state['agent-field.settings.v1']).toContain('https://research.example.test');
  expect(state['agent-field.settings.v1']).toContain('https://agent.example.test');
  await page.reload();
  await page.getByRole('button', { name: 'Connections', exact: true }).click();
  await expect(page.getByLabel('MetaNavT service URL')).toHaveValue('https://research.example.test');
  await expect(page.getByLabel(/Soccer access token/)).toHaveValue('');
  await expect(page.getByLabel('Research agent token', { exact: false })).toHaveValue('');
});

test('offline research agent searches, inspects, cites and saves real fixture excerpts', async ({ page }) => {
  await page.getByRole('button', { name: 'Research', exact: true }).click();
  await page.getByRole('button', { name: 'Ask agent about this question' }).click();
  const panel = page.getByRole('region', { name: 'Research agent' });
  await expect(panel).toContainText('SCRIPTED AGENT DEMO');
  await expect(panel.locator('.agent-answer')).toContainText('learning_rate: 0.0003');
  await panel.getByRole('button', { name: /S\d+ · run_047.yaml/ }).click();
  const source = page.getByRole('dialog', { name: 'Agent source' });
  await expect(source).toContainText('SYNTHETIC FIXTURE');
  await expect(source).toContainText('learning_rate: 0.0003');
  await source.getByRole('button', { name: 'Save source' }).click();
  await source.getByRole('button', { name: 'Save source' }).click();
  await expect(page.getByRole('status')).toContainText('already saved');
  await page.getByRole('button', { name: 'Close detail' }).click();
  await panel.locator('summary').click();
  await expect(panel.locator('.trace-list li')).toHaveCount(4);
  await expect(panel.locator('.trace-list li').first()).toContainText('search research');
  await expect(panel.locator('.trace-list li').nth(1)).toContainText('inspect source');
  await expect(panel).toContainText('No language model runs');
  await page.getByRole('button', { name: 'Saved', exact: true }).click();
  await expect(page.locator('.saved-card')).toHaveCount(1);
});

test('research agent abstains without evidence and rejects invalid questions', async ({ page }) => {
  await page.getByRole('button', { name: 'Research', exact: true }).click();
  await page.getByLabel('Ask your research files').fill('zzunknownunique987');
  await page.getByRole('button', { name: 'Ask agent about this question' }).click();
  const panel = page.getByRole('region', { name: 'Research agent' });
  await expect(panel).toContainText('abstained');
  await expect(panel.locator('.agent-citations button')).toHaveCount(0);
  await page.getByLabel('Ask your research files').fill('   ');
  await page.getByRole('button', { name: 'Ask agent about this question' }).click();
  await expect(panel.getByRole('alert')).toContainText('between 1 and 1,000');
  await expect(panel.locator('.agent-result')).toHaveCount(0);
});

test('connected research agent rejects an incompatible response without showing a claim', async ({ page }) => {
  await page.route('https://agent.example.test/agent/ask', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'complete', answer: 'Unverified success.' }) });
  });
  await connect(page);
  await page.getByLabel('Research agent URL', { exact: false }).fill('https://agent.example.test');
  await page.getByRole('button', { name: 'Save workspace settings' }).click();
  await page.getByRole('button', { name: 'Research', exact: true }).click();
  await page.getByRole('button', { name: 'Ask agent about this question' }).click();
  const panel = page.getByRole('region', { name: 'Research agent' });
  await expect(panel.getByRole('alert')).toContainText('incompatible or unverified');
  await expect(panel.locator('.agent-answer')).toHaveCount(0);
});

test('malformed and HTTP-error responses clear previously displayed research evidence', async ({ page }) => {
  let attempts = 0;
  await page.route('https://research.example.test/api/retrieve/', async (route) => {
    attempts++;
    if (attempts === 1) await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(researchCapture.examples[0].response) });
    else if (attempts === 2) await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'ok', hits: [] }) });
    else await route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'retrieval unavailable' }) });
  });
  await connect(page);
  await page.getByRole('button', { name: 'Research', exact: true }).click();
  await page.getByRole('button', { name: 'Search research' }).click();
  await expect(page.locator('.source-card')).toHaveCount(3);
  await page.getByRole('button', { name: 'Search research' }).click();
  await expect(page.getByRole('alert')).toContainText('incompatible');
  await expect(page.locator('.source-card')).toHaveCount(0);
  await page.getByRole('button', { name: 'Search research' }).click();
  await expect(page.getByRole('alert')).toContainText('HTTP 503');
  await expect(page.locator('.source-card')).toHaveCount(0);
});

test('corrupted persisted state recovers to the demo without a broken screen', async ({ page }) => {
  await page.evaluate(() => {
    localStorage.setItem('agent-field.settings.v1', '{bad json');
    localStorage.setItem('agent-field.saved.v1', JSON.stringify({ unexpected: true }));
  });
  await page.reload();
  await expect(page.getByText('Demo', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Saved', exact: true }).click();
  await expect(page.getByText('Worth keeping.', { exact: true })).toBeVisible();
});

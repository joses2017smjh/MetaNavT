import { expect, test } from '@playwright/test';

test('actual research-agent demo service returns inspected citations through HTTP', async ({ page }) => {
  test.skip(!process.env.AGENT_E2E_URL, 'Optional companion service contract check: set AGENT_E2E_URL.');
  const url = process.env.AGENT_E2E_URL!.replace(/\/+$/, '');
  let actualResponse: Record<string, any> | undefined;
  // The production client asks a configured live agent. This explicit test
  // harness rewrites only mode to demo to exercise actual server tools and
  // HTTP contracts without installing or claiming a language model.
  await page.route(`${url}/agent/ask`, async (route) => {
    const request = route.request().postDataJSON();
    const response = await route.fetch({ postData: { ...request, mode: 'demo' } });
    expect(response.status()).toBe(200);
    actualResponse = await response.json();
    await route.fulfill({ response });
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Connections', exact: true }).click();
  await page.getByLabel('Connect services', { exact: false }).check();
  await page.getByLabel('MetaNavT service URL').fill('https://unused-research.example.test');
  await page.getByLabel('Soccer mobile adapter URL').fill('https://unused-soccer.example.test');
  await page.getByLabel('Research agent URL', { exact: false }).fill(url);
  await page.getByRole('button', { name: 'Save workspace settings' }).click();
  await page.getByRole('button', { name: 'Research', exact: true }).click();
  await page.getByLabel('Ask your research files').fill('What is the learning rate in run_047.yaml?');
  await page.getByRole('button', { name: 'Ask agent about this question' }).click();
  const panel = page.getByRole('region', { name: 'Research agent' });
  await expect(panel.locator('.agent-answer')).toContainText('learning_rate: 0.0003 [S1]');
  expect(actualResponse?.mode).toBe('demo');
  expect(actualResponse?.synthetic).toBe(true);
  expect(actualResponse?.model).toBe('scripted-extractive-demo-v1');
  expect(actualResponse?.citation_check.passed).toBe(true);
  expect(actualResponse?.trace.map((step) => step.tool)).toEqual(['search_research', 'inspect_source']);
  await panel.getByRole('button', { name: 'S1 · run_047.yaml' }).click();
  await expect(page.getByRole('dialog', { name: 'Agent source' })).toContainText('learning_rate: 0.0003');
  await expect(page.getByRole('dialog', { name: 'Agent source' })).toContainText('SYNTHETIC FIXTURE');
  await page.getByRole('button', { name: 'Close detail' }).click();
  await panel.locator('summary').click();
  await expect(panel.locator('.trace-list li')).toHaveCount(2);
  await expect(panel).toContainText('No LLM inference');
});

test('connected app searches MetaNavT and runs the actual soccer workflow and review boundary', async ({ page }) => {
  test.skip(!process.env.META_E2E_URL || !process.env.SOCCER_E2E_URL || !process.env.SOCCER_E2E_TOKEN,
    'Optional real-service check: run python e2e/run-connected.py with both repository paths.');
  test.setTimeout(60000);
  await page.goto('/');
  await page.getByRole('button', { name: 'Connections', exact: true }).click();
  await page.getByLabel('Connect services', { exact: false }).check();
  await page.getByLabel('MetaNavT service URL').fill(process.env.META_E2E_URL!);
  await page.getByLabel('Soccer mobile adapter URL').fill(process.env.SOCCER_E2E_URL!);
  await page.getByLabel(/Soccer access token/).fill(process.env.SOCCER_E2E_TOKEN!);
  await page.getByRole('button', { name: 'Test connections' }).click();
  await expect(page.getByRole('status')).toContainText('responded');
  await page.getByRole('button', { name: 'Save workspace settings' }).click();
  await page.getByRole('button', { name: 'Research', exact: true }).click();
  await page.getByLabel('Ask your research files').fill('What is the learning rate in run_047.yaml?');
  const retrievalResponse = page.waitForResponse((response) => response.url().endsWith('/api/retrieve/') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Search research' }).click();
  const retrieved = await retrievalResponse;
  expect(retrieved.status()).toBe(200);
  expect((await retrieved.json()).retrieval_mode).toBe('mobile_demo_hash');
  await page.locator('.source-card').filter({ has: page.getByRole('heading', { name: 'run_047.yaml', exact: true }) }).click();
  await expect(page.getByRole('dialog', { name: 'Source detail' })).toContainText('learning_rate: 0.0003');
  await page.getByRole('button', { name: 'Close detail' }).click();

  await page.getByRole('button', { name: 'Match Lab', exact: true }).click();
  await page.getByLabel('Your matchup').fill('Predict Arsenal vs Man City on 2026-07-18');
  const soccerResponse = page.waitForResponse((response) => response.url().endsWith('/mobile/predict') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Analyze matchup' }).click();
  const predicted = await soccerResponse;
  expect(predicted.status()).toBe(200);
  const payload = await predicted.json();
  expect(payload.mode).toBe('live');
  expect(payload.status).toBe('complete');
  expect(payload.provenance.kind).toBe('gateway');
  expect(payload.provenance.model_version).toBe('v0-mobile-demo');
  expect(payload.tool_calls).toHaveLength(11);
  await expect(page.getByText('Three possible outcomes', { exact: true })).toBeVisible();
  const outcome = payload.prediction.match_outcome;
  await expect(page.locator('.probability-values strong')).toHaveText(
    [outcome.home, outcome.draw, outcome.away].map((probability) => `${Math.round(probability * 100)}%`));
  await expect(page.locator('.xg-values strong')).toHaveText(
    [payload.prediction.expected_goals.home, payload.prediction.expected_goals.away].map((goals) => goals.toFixed(2)));
  await page.getByRole('button', { name: /11 recorded tool calls.*Agent trace/ }).click();
  await expect(page.getByRole('dialog', { name: 'Agent trace' })).toContainText('CONNECTED GATEWAY');
  await expect(page.getByRole('dialog', { name: 'Agent trace' }).locator('.trace-list li')).toHaveCount(11);
  await page.getByRole('button', { name: 'Close detail' }).click();

  await page.getByLabel('Your matchup').fill('Arsenal vs Man City on 2026-07-18 — any value bets?');
  const reviewResponse = page.waitForResponse((response) => response.url().endsWith('/mobile/predict') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Analyze matchup' }).click();
  const review = await reviewResponse;
  expect(review.status()).toBe(200);
  const pending = await review.json();
  expect(pending.status).toBe('pending_approval');
  expect(pending.prediction).toBeNull();
  await expect(page.getByText('Operator review required', { exact: true })).toBeVisible();
  await expect(page.locator('.probability-card')).toHaveCount(0);
  expect(await page.evaluate(() => JSON.stringify({ ...localStorage }))).not.toContain(process.env.SOCCER_E2E_TOKEN!);
});

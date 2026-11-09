import { createHash } from 'node:crypto';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { agentSchema, askAgent, runAgentDemo } from './agent';
import { requestJson } from './client';
import serviceCapture from '../../../doc/research-agent-demo.json';

vi.mock('./client', () => ({ requestJson: vi.fn() }));
beforeEach(() => vi.mocked(requestJson).mockReset());
const good = () => runAgentDemo('What is the learning rate in run_047.yaml?');

describe('research-agent evidence and bounded execution', () => {
  it('accepts the actual companion API captures, including nullable trace errors and uninspected search hits', () => {
    expect(serviceCapture.synthetic).toBe(true);
    for (const example of serviceCapture.examples) {
      const result = agentSchema.parse(example);
      expect(result.model).toBe('scripted-extractive-demo-v1');
      for (const source of result.sources) {
        expect(source.excerpt_sha256).toBe(createHash('sha256').update(source.text, 'utf8').digest('hex'));
      }
      for (const citation of result.citations) {
        expect(result.sources.find((source) => source.source_id === citation.source_id)?.inspected).toBe(true);
      }
    }
  });

  it('runs actual local search and source inspection with no language model or network', async () => {
    const result = await good();
    expect(result.status).toBe('complete');
    expect(result.synthetic).toBe(true);
    expect(result.model).toBe('scripted-browser-demo');
    expect(result.notice).toContain('No language model runs');
    expect(result.limits.model_rounds).toBe(0);
    expect(result.trace[0].tool).toBe('search_research');
    expect(result.trace.slice(1).every((step) => step.tool === 'inspect_source')).toBe(true);
    expect(result.trace).toHaveLength(result.sources.length + 1);
    expect(result.limits.tool_calls).toBe(result.trace.length);
    expect(requestJson).not.toHaveBeenCalled();
    for (const source of result.sources) {
      expect(source.inspected).toBe(true);
      expect(source.excerpt_sha256).toBe(createHash('sha256').update(source.text, 'utf8').digest('hex'));
      expect(result.citations.find((citation) => citation.source_id === source.source_id)).toEqual({
        source_id: source.source_id, path: source.path, start_byte: source.start_byte,
        end_byte: source.end_byte, excerpt_sha256: source.excerpt_sha256,
      });
      expect(result.answer).toContain(`[${source.source_id}]`);
    }
  });

  it('abstains when no evidence exists and does not invent a citation', async () => {
    const result = await runAgentDemo('zzunknownunique987');
    expect(result.status).toBe('abstained');
    expect(result.sources).toEqual([]);
    expect(result.citations).toEqual([]);
    expect(result.stop_reason).toBe('no_sources');
    expect(result.limits.tool_calls).toBe(1);
  });

  it.each(['  ', 'x'.repeat(2001)])('rejects a blank or oversized research question', async (question) => {
    await expect(runAgentDemo(question)).rejects.toThrow();
  });

  it('rejects citations to absent or uninspected evidence', async () => {
    const absent = await good();
    absent.citations[0].source_id = 'S999';
    expect(agentSchema.safeParse(absent).success).toBe(false);
    const uninspected = await good();
    uninspected.sources[0].inspected = false;
    expect(agentSchema.safeParse(uninspected).success).toBe(false);
  });

  it('rejects citation fingerprints or physical spans that differ from inspected sources', async () => {
    const fingerprint = await good();
    fingerprint.citations[0].excerpt_sha256 = 'a'.repeat(64);
    expect(agentSchema.safeParse(fingerprint).success).toBe(false);
    const span = await good();
    span.citations[0].start_byte = 1;
    expect(agentSchema.safeParse(span).success).toBe(false);
  });

  it('rejects unknown answer references and a complete answer without source references', async () => {
    const unknown = await good();
    unknown.answer += '\nUnsupported claim [S999]';
    expect(agentSchema.safeParse(unknown).success).toBe(false);
    const missing = await good();
    missing.answer = 'A confident claim without a reference.';
    expect(agentSchema.safeParse(missing).success).toBe(false);
  });

  it('rejects failed citation checks instead of displaying a verified answer', async () => {
    const failed = await good();
    failed.citation_check.passed = false;
    expect(agentSchema.safeParse(failed).success).toBe(false);
    const unknown = await good();
    unknown.citation_check.unknown_ids = ['S999'];
    expect(agentSchema.safeParse(unknown).success).toBe(false);
  });

  it.each([-1, 0.5, 7])('rejects invalid or excessive tool-call count %s', async (count) => {
    const result = await good();
    result.limits.tool_calls = count;
    expect(agentSchema.safeParse(result).success).toBe(false);
  });

  it('rejects excessive model rounds and invalid maxima', async () => {
    const rounds = await good();
    rounds.limits.model_rounds = rounds.limits.max_model_rounds + 1;
    expect(agentSchema.safeParse(rounds).success).toBe(false);
    const maximum = await good();
    maximum.limits.max_tool_calls = 0;
    expect(agentSchema.safeParse(maximum).success).toBe(false);
  });

  it('checks actual excerpt bytes, even when citation and source metadata agree on a false hash', async () => {
    const response = await good();
    response.mode = 'live';
    response.sources[0].excerpt_sha256 = '0'.repeat(64);
    response.citations[0].excerpt_sha256 = '0'.repeat(64);
    vi.mocked(requestJson).mockResolvedValue(response);
    await expect(askAgent('https://agent.example.test', response.question, 'session-token')).rejects.toThrow('fingerprint');
  });

  it('accepts a verified service response and sends the session token only through the request', async () => {
    const response = await good();
    response.mode = 'live';
    vi.mocked(requestJson).mockResolvedValue(response);
    await expect(askAgent('https://agent.example.test', response.question, 'session-token')).resolves.toEqual(response);
    expect(requestJson).toHaveBeenCalledWith('https://agent.example.test', '/agent/ask',
      { question: response.question, mode: 'live' }, 'session-token', 90000);
  });
});

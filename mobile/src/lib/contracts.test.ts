import { describe, expect, it } from 'vitest';
import soccer from '../fixtures/soccer.json';
import serverCapture from '../../../app/mobile/demo-response.json';
import { hitSchema, retrievalSchema, soccerSchema } from './contracts';

const goodSoccer = () => structuredClone(soccer.samples.find((sample) => sample.response.status === 'complete')!.response);

describe('responses stay honest about results', () => {
  it.each([NaN, Infinity, -0.01, 1.01])('rejects invalid outcome probability %s', (value) => {
    const input = goodSoccer();
    input.prediction!.match_outcome.home = value;
    expect(soccerSchema.safeParse(input).success).toBe(false);
  });

  it('rejects an outcome distribution that does not sum to one', () => {
    const input = goodSoccer();
    Object.assign(input.prediction!.match_outcome, { home: 0.6, draw: 0.3, away: 0.3 });
    expect(soccerSchema.safeParse(input).success).toBe(false);
  });

  it('rejects negative expected goals', () => {
    const input = goodSoccer();
    input.prediction!.expected_goals.away = -1;
    expect(soccerSchema.safeParse(input).success).toBe(false);
  });

  it('rejects a prediction attached to a paused operator-review response', () => {
    const input = { ...goodSoccer(), status: 'pending_approval', approval_request: { reason: 'Review required' } };
    expect(soccerSchema.safeParse(input).success).toBe(false);
  });

  it('rejects incompatible research responses instead of constructing empty success', () => {
    expect(retrievalSchema.safeParse({ status: 'ok', hits: [] }).success).toBe(false);
    const input = structuredClone(serverCapture.examples[0].response);
    input.hits[0].score = Infinity;
    expect(retrievalSchema.safeParse(input).success).toBe(false);
  });

  it('rejects backwards physical byte spans', () => {
    const input = { ...serverCapture.examples[0].response.hits[0], start_byte: 20, end_byte: 10 };
    expect(hitSchema.safeParse(input).success).toBe(false);
  });
});

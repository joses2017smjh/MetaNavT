import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import meta from '../fixtures/meta.json';
import soccer from '../fixtures/soccer.json';
import serverCapture from '../../../app/mobile/demo-response.json';
import { searchDemo, replaySoccer, starterQueries } from './demo';
import { retrievalSchema, soccerSchema } from './contracts';

describe('auditable research fixture', () => {
  it('contains exact source bytes with matching SHA-256, rather than invented excerpts', () => {
    expect(meta.synthetic).toBe(true);
    expect(meta.files).toHaveLength(5);
    for (const file of meta.files) {
      const bytes = Buffer.from(file.text, 'utf8');
      const original = readFileSync(resolve(process.cwd(), '../app/mobile/fixtures', file.path));
      expect(bytes.equals(original), file.path).toBe(true);
      expect(createHash('sha256').update(bytes).digest('hex'), file.path).toBe(file.sha256);
    }
  });

  it('returns valid physical UTF-8 spans and never labels local keyword ranking as neural inference', () => {
    for (const query of starterQueries) {
      const result = searchDemo(query);
      expect(result.retrieval_mode).toBe('offline_keyword_demo');
      expect(result.embedding_provider).toBe('none');
      expect(result.hits.length).toBeGreaterThan(0);
      for (const hit of result.hits) {
        const file = meta.files.find((candidate) => candidate.path === hit.path)!;
        const physical = Buffer.from(file.text, 'utf8');
        expect(hit.start_byte).toBe(0);
        expect(hit.end_byte).toBe(physical.byteLength);
        expect(physical.subarray(hit.start_byte!, hit.end_byte!).toString('utf8')).toBe(hit.text);
        expect(hit.chunk_id).toBe(file.sha256);
      }
    }
  });

  it('finds the learning rate and an exact checkpoint path', () => {
    const learning = searchDemo(starterQueries[0]);
    expect(learning.hits.some((hit) => hit.path === 'configs/run_047.yaml' && hit.text.includes('learning_rate: 0.0003'))).toBe(true);
    expect(searchDemo('Open checkpoints/run_047.ckpt.meta.json').hits[0].path).toBe('checkpoints/run_047.ckpt.meta.json');
  });

  it('returns no evidence for absent terms and rejects blank or oversized questions', () => {
    expect(searchDemo('zzunknownunique987').hits).toHaveLength(0);
    expect(() => searchDemo('   ')).toThrow();
    expect(() => searchDemo('x'.repeat(2001))).toThrow();
  });

  it('accepts actual shared-router fixture responses without losing byte references', () => {
    expect(serverCapture.synthetic).toBe(true);
    for (const example of serverCapture.examples) {
      const response = retrievalSchema.parse(example.response);
      expect(response.query).toBe(example.query);
      expect(response.retrieval_mode).toBe('mobile_demo_hash');
      for (const hit of response.hits) {
        const source = meta.files.find((file) => file.path === hit.path)!;
        expect(Buffer.from(source.text).subarray(hit.start_byte!, hit.end_byte!).toString('utf8').slice(0, 800)).toBe(hit.text);
      }
    }
  });
});

describe('recorded agent workflow fixture', () => {
  it('keeps the source revision of the recorded fixture explicit as live code advances', () => {
    expect(soccer.base_commit).toBe('f5699deca4b4be08297293fa1770929f1cd168bc');
    expect(soccer.generator).toBe('python -m scripts.export_mobile_demo');
    for (const source of ['agent/graph.py', 'agent/tooling.py', 'scripts/export_mobile_demo.py']) {
      expect(soccer.source_sha256[source as keyof typeof soccer.source_sha256]).toMatch(/^[a-f0-9]{64}$/);
    }
  });

  it('parses every captured backend response and preserves the tool trail', () => {
    for (const sample of soccer.samples) {
      expect(soccerSchema.parse(sample.response).thread_id).toBe(sample.response.thread_id);
      const replay = replaySoccer(sample.id);
      expect(replay.mode).toBe('demo');
      expect(replay.provenance.notes.join(' ').toLowerCase()).toContain('synthetic');
      if (replay.status === 'complete') {
        expect(replay.tool_calls).toHaveLength(11);
        expect(replay.prediction!.match_outcome.home + replay.prediction!.match_outcome.draw + replay.prediction!.match_outcome.away).toBeCloseTo(1, 5);
      } else {
        expect(replay.prediction == null).toBe(true);
        expect(replay.approval_request).toBeTruthy();
      }
    }
  });

  it('cannot silently substitute a fixture for an unknown matchup', () => {
    expect(() => replaySoccer('unavailable-fixture')).toThrow();
  });
});

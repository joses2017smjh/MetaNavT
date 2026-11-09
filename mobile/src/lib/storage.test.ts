import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { defaultSettings, loadSaved, loadSettings, storeSaved, storeSettings } from './storage';

class MemoryStorage {
  items = new Map<string, string>();
  getItem(key: string) { return this.items.get(key) ?? null; }
  setItem(key: string, value: string) { this.items.set(key, value); }
  removeItem(key: string) { this.items.delete(key); }
  clear() { this.items.clear(); }
  key(index: number) { return [...this.items.keys()][index] ?? null; }
  get length() { return this.items.size; }
}

let storage: MemoryStorage;
beforeEach(() => { storage = new MemoryStorage(); vi.stubGlobal('localStorage', storage); });
afterEach(() => vi.unstubAllGlobals());

describe('local device state', () => {
  it('recovers from corrupted JSON and invalid shapes without crashing launch', () => {
    storage.setItem('agent-field.settings.v1', '{');
    storage.setItem('agent-field.saved.v1', JSON.stringify({ unexpected: true }));
    expect(loadSettings()).toEqual(defaultSettings);
    expect(loadSaved()).toEqual([]);
  });

  it('recovers when the browser refuses storage reads', () => {
    vi.spyOn(storage, 'getItem').mockImplementation(() => { throw new Error('Storage disabled'); });
    expect(loadSettings()).toEqual(defaultSettings);
    expect(loadSaved()).toEqual([]);
  });

  it('persists service choices while stripping accidental token properties', () => {
    const input = { mode: 'connected' as const, metaUrl: 'https://research.example.com', soccerUrl: 'https://soccer.example.com', agentUrl: 'https://agent.example.com', token: 'PRIVATE_SESSION_TOKEN', apiKey: 'PRIVATE_API_KEY', agentToken: 'PRIVATE_AGENT_TOKEN' };
    storeSettings(input);
    expect(loadSettings()).toEqual({ mode: input.mode, metaUrl: input.metaUrl, soccerUrl: input.soccerUrl, agentUrl: input.agentUrl });
    expect([...storage.items.values()].join(' ')).not.toContain('PRIVATE_');
  });

  it('migrates saved connections created before the optional agent URL existed', () => {
    storage.setItem('agent-field.settings.v1', JSON.stringify({ mode: 'connected', metaUrl: 'https://research.example.com', soccerUrl: 'https://soccer.example.com' }));
    expect(loadSettings()).toEqual({ mode: 'connected', metaUrl: 'https://research.example.com', soccerUrl: 'https://soccer.example.com', agentUrl: '' });
  });

  it('preserves a saved evidence excerpt and rejects more than 100 items', () => {
    const item = { id: 'research:sha256', kind: 'research' as const, title: 'run_047.yaml', subtitle: 'learning rate', content: 'learning_rate: 0.0003', savedAt: '2026-10-07T12:00:00Z', provenance: 'Synthetic research fixture' };
    storeSaved([item]);
    expect(loadSaved()).toEqual([item]);
    expect(() => storeSaved(Array.from({ length: 101 }, (_, index) => ({ ...item, id: String(index) })))).toThrow();
    expect(loadSaved()).toEqual([item]);
  });

  it('propagates a failed write so the UI can report session-only storage', () => {
    vi.spyOn(storage, 'setItem').mockImplementation(() => { throw new Error('Quota exceeded'); });
    expect(() => storeSaved([])).toThrow('Quota exceeded');
  });
});

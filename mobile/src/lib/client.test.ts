import { Capacitor, CapacitorHttp } from '@capacitor/core';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import responseFixture from '../../../app/mobile/demo-response.json';
import { requestJson, retrieve, validateBaseUrl } from './client';

vi.mock('@capacitor/core', () => ({
  Capacitor: { isNativePlatform: vi.fn(() => false) },
  CapacitorHttp: { request: vi.fn() },
}));
beforeEach(() => { vi.mocked(Capacitor.isNativePlatform).mockReturnValue(false); });
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('service connection boundaries', () => {
  it.each(['https://user:password@example.com', 'https://example.com?token=private', 'https://example.com#private', 'file:///etc/passwd', 'http://public.example.com'])('rejects an unsafe service URL %s', (url) => {
    expect(() => validateBaseUrl(url)).toThrow();
  });

  it('permits local HTTP only in browser development; native always requires HTTPS', () => {
    expect(validateBaseUrl(' http://localhost:8000/ ')).toBe('http://localhost:8000');
    expect(validateBaseUrl('https://research.example.com/')).toBe('https://research.example.com');
    vi.mocked(Capacitor.isNativePlatform).mockReturnValue(true);
    for (const url of ['http://localhost:8000', 'http://127.0.0.1:8000', 'http://10.0.2.2:8000']) expect(() => validateBaseUrl(url)).toThrow('HTTPS');
    expect(validateBaseUrl('https://research.example.com')).toBe('https://research.example.com');
  });

  it('disables redirects and ambient browser credentials when sending a session token', async () => {
    const fetchSpy = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ok: true }), { status: 200 }));
    vi.stubGlobal('fetch', fetchSpy);
    await requestJson('https://soccer.example.com', '/mobile/predict', { text: 'Arsenal vs City' }, 'session-token');
    const options = fetchSpy.mock.calls[0][1];
    expect(options.redirect).toBe('error');
    expect(options.credentials).toBe('omit');
    expect(options.headers['X-API-Key']).toBe('session-token');
  });

  it('uses native HTTP with redirect suppression and bounded connection/read timeouts', async () => {
    vi.mocked(Capacitor.isNativePlatform).mockReturnValue(true);
    vi.mocked(CapacitorHttp.request).mockResolvedValue({ status: 200, data: { ok: true }, headers: {}, url: 'https://soccer.example.com/mobile/health' });
    await requestJson('https://soccer.example.com', '/mobile/health', undefined, undefined, 10000);
    expect(CapacitorHttp.request).toHaveBeenCalledWith(expect.objectContaining({ disableRedirects: true, connectTimeout: 15000, readTimeout: 10000 }));
  });

  it.each([[401, 'Access was denied'], [429, 'busy'], [503, 'HTTP 503']])('reports HTTP %i instead of treating JSON errors as results', async (status, message) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status: status as number })));
    await expect(requestJson('https://research.example.com', '/health')).rejects.toThrow(message as string);
  });

  it('rejects malformed JSON and incompatible research payloads', async () => {
    const fake = vi.fn().mockResolvedValueOnce(new Response('<html>Wrong server</html>', { status: 200 })).mockResolvedValueOnce(new Response(JSON.stringify({ status: 'ok' }), { status: 200 }));
    vi.stubGlobal('fetch', fake);
    await expect(retrieve('https://research.example.com', 'learning rate')).rejects.toThrow('unreadable');
    await expect(retrieve('https://research.example.com', 'learning rate')).rejects.toThrow('incompatible');
  });

  it('parses the actual backend response and posts a bounded retrieval request', async () => {
    const fake = vi.fn().mockResolvedValue(new Response(JSON.stringify(responseFixture.examples[0].response), { status: 200 }));
    vi.stubGlobal('fetch', fake);
    const result = await retrieve('https://research.example.com', 'learning rate');
    expect(result.hits[0].path).toBe('configs/run_047.yaml');
    expect(JSON.parse(fake.mock.calls[0][1].body)).toEqual({ query: 'learning rate', k: 5 });
  });
});

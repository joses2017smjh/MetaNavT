import { Capacitor, CapacitorHttp } from '@capacitor/core';
import { retrievalSchema, soccerSchema } from './contracts';

export function validateBaseUrl(value: string): string {
  let url: URL;
  try { url = new URL(value.trim()); } catch { throw new Error('Enter a complete service URL, such as https://research.example.com.'); }
  if (url.username || url.password || url.search || url.hash) throw new Error('Service URLs must not contain credentials, query strings, or fragments.');
  const loopback = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && loopback && !Capacitor.isNativePlatform())) throw new Error('Use HTTPS for services. Browser development also supports local HTTP.');
  return url.toString().replace(/\/+$/, '');
}
export async function requestJson(base: string, path: string, body?: unknown, token?: string, timeout = 50000): Promise<unknown> {
  const url = `${validateBaseUrl(base)}${path}`;
  const headers: Record<string, string> = { Accept: 'application/json' };
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  if (token) headers['X-API-Key'] = token;
  let status: number, data: unknown;
  if (Capacitor.isNativePlatform()) {
    const response = await CapacitorHttp.request({ url, method: body === undefined ? 'GET' : 'POST', headers, data: body, responseType: 'json', connectTimeout: 15000, readTimeout: timeout, disableRedirects: true });
    status = response.status; data = response.data;
  } else {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(url, { method: body === undefined ? 'GET' : 'POST', headers, body: body === undefined ? undefined : JSON.stringify(body), signal: controller.signal, redirect: 'error', credentials: 'omit' });
      status = response.status;
      const text = await response.text();
      try { data = JSON.parse(text); } catch { throw new Error('The service returned an unreadable response. Check its URL and API version.'); }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') throw new Error('The service timed out. Please try again.');
      throw error;
    } finally { clearTimeout(timer); }
  }
  if (status < 200 || status >= 300) {
    if (status === 401 || status === 403) throw new Error('Access was denied. Check the session access token in Connections.');
    if (status === 429) throw new Error('The service is busy. Please try again shortly.');
    if (status === 504) throw new Error('The upstream service timed out. Please try again.');
    throw new Error(`The service could not complete this request (HTTP ${status}).`);
  }
  return data;
}
export async function retrieve(base: string, query: string) {
  const raw = await requestJson(base, '/api/retrieve/', { query, k: 5 });
  const result = retrievalSchema.safeParse(raw);
  if (!result.success) throw new Error('The research service returned an incompatible response.');
  return result.data;
}
export async function predict(base: string, text: string, token: string) {
  const raw = await requestJson(base, '/mobile/predict', { text, mode: 'live' }, token);
  const result = soccerSchema.safeParse(raw);
  if (!result.success) throw new Error('The match service returned an incompatible response.');
  return result.data;
}

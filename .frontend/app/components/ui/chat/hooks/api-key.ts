"use client";

import { useSyncExternalStore } from "react";
import { getBackendOrigin } from "./use-config";

// The API key, when the server requires one (METANAVIT_API_KEY or METANAVIT_API_KEYS_FILE). The user types it into
// the page (ApiKeyField) and it is kept in this tab's sessionStorage, which the browser clears when the tab closes.
// It is never part of the build: a static export is readable by anyone who can load the page. apiFetch adds it as
// X-API-Key to the page's requests to the API; links the browser follows by itself (/api/files/...) carry no header.

export const API_KEY_STORAGE = "metanavit.apiKey";
const HEADER = "X-API-Key";

let memoryKey: string | null = null; // only while sessionStorage cannot be used (blocked storage): lost on reload
let version = 0;
const listeners = new Set<() => void>();

function storage(): Storage | null {
  try {
    return typeof window === "undefined" ? null : window.sessionStorage;
  } catch {
    return null;
  }
}

export function getApiKey(): string | null {
  try {
    return storage()?.getItem(API_KEY_STORAGE) || memoryKey;
  } catch {
    return memoryKey;
  }
}

export function setApiKey(key: string | null): void {
  const value = key?.trim() || null;
  memoryKey = null;
  try {
    const store = storage();
    if (!store) throw new Error("no sessionStorage");
    if (value) store.setItem(API_KEY_STORAGE, value);
    else store.removeItem(API_KEY_STORAGE);
  } catch {
    memoryKey = value;
  }
  version += 1;
  listeners.forEach((listener) => listener());
}

// True for a request to the API: a URL under <backend>/api/, where the backend is the page's own origin unless
// NEXT_PUBLIC_CHAT_API names another (use-config.ts). The key goes nowhere else.
function isApiRequest(input: RequestInfo | URL): boolean {
  if (typeof window === "undefined") return false;
  const raw =
    typeof input === "string"
      ? input
      : input instanceof URL
        ? input.href
        : input.url;
  const url = new URL(raw, window.location.href);
  const api = new URL(`${getBackendOrigin()}/api/`, window.location.href);
  return url.href.startsWith(api.href);
}

// fetch, plus X-API-Key on a request to the API when a key is set. Every request the page's code makes goes
// through here (tests/requests.test.ts).
export function apiFetch(
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<Response> {
  const key = getApiKey();
  if (!key || !isApiRequest(input)) return fetch(input, init);
  const headers = new Headers(
    init.headers ?? (input instanceof Request ? input.headers : undefined),
  );
  headers.set(HEADER, key);
  return fetch(input, { ...init, headers });
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

// Whether a key is set (false while prerendering: the static export never knows one).
export function useApiKeySet(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => getApiKey() !== null,
    () => false,
  );
}

// Changes whenever the key is set or forgotten: a component that fetched without it fetches again.
export function useApiKeyVersion(): number {
  return useSyncExternalStore(
    subscribe,
    () => version,
    () => 0,
  );
}

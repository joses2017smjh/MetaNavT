// organize-imports-ignore: "./dom" must stay the first import (it sets up the browser globals the others read)
import "./dom";
import { cleanup } from "@testing-library/react";
import { mock } from "node:test";

// A mocked API: fetch is replaced by a function that records each request and answers it from `answer`.

export interface Call {
  method: string;
  url: URL;
  path: string;
  headers: Headers;
  body: unknown;
}

// Where the page keeps the key the user enters (app/components/ui/chat/hooks/api-key.ts).
export const KEY_SLOT = "metanavit.apiKey";

export function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

export function mockApi(
  answer: (call: Call) => Response | Promise<Response>,
): Call[] {
  const calls: Call[] = [];
  mock.method(
    globalThis,
    "fetch",
    async (input: RequestInfo | URL, init: RequestInit = {}) => {
      const raw =
        typeof input === "string"
          ? input
          : input instanceof URL
            ? input.href
            : input.url;
      const url = new URL(raw, window.location.href);
      const call: Call = {
        method: (init.method ?? "GET").toUpperCase(),
        url,
        path: url.pathname,
        headers: new Headers(init.headers),
        body: typeof init.body === "string" ? JSON.parse(init.body) : undefined,
      };
      calls.push(call);
      return answer(call);
    },
  );
  return calls;
}

// After each test: unmount what it rendered (Testing Library cleans up by itself only under Jest-style globals), put
// fetch back, and forget the key.
export function reset(): void {
  cleanup();
  mock.restoreAll();
  window.sessionStorage.clear();
  window.localStorage.clear();
}

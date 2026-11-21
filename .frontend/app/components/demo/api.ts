import { apiFetch } from "../ui/chat/hooks/api-key";

// The public demo's API (PUBLIC_DEMO=1, app/api/public.py), called on the page's own origin only, never another host,
// and through apiFetch like every request the page makes (tests/requests.test.ts).

// The routes' limits; tests/api/test_public_demo.py checks them against app/api/public.py and the answer route.
export const MAX_QUESTION_CHARS = 500; // PUBLIC_MAX_QUERY_CHARS, and the answer route's MAX_QUESTION_CHARS
export const SEARCH_K = 8; // hits asked of POST /api/retrieve/ (at most PUBLIC_MAX_K)
export const ANSWER_K = 6; // excerpts POST /api/answer/ retrieves and sends to Claude (at most its MAX_K)

export const HEALTH_PATH = "/health";
export const RETRIEVE_PATH = "/api/retrieve/";
export const ANSWER_PATH = "/api/answer/";

// The response fields the page reads (app/api/routers/retrieve.py, app/api/routers/answer.py).
export interface StageScores {
  bm25: number | null;
  dense: number | null;
  path: number | null;
  rrf: number | null;
  rerank: number | null;
}

export interface Hit {
  rank: number;
  chunk_id: string;
  path: string | null;
  start_byte: number | null;
  end_byte: number | null;
  score: number;
  scores: StageScores;
  text: string;
}

export interface RetrieveResponse {
  query: string;
  route: string | null;
  bm25_backend: string | null;
  embedding_provider: string;
  reranker_loaded: boolean; // false on a server without the cross-encoder (RERANKER_MODEL=none): the fused order
  reranker: { loaded: boolean; model: string | null };
  degraded: { component: string; error: string }[];
  latency_ms: Record<string, number>;
  hits: Hit[];
}

export interface Citation {
  chunk_id: string;
  path: string | null;
  cited_text: string;
  start_char: number;
  end_char: number;
  start_byte: number | null;
  end_byte: number | null;
}

export interface AnswerResponse {
  question: string;
  answer: string | null;
  reason: string | null;
  uncited_reply: string | null;
  citations: Citation[];
  model: string | null;
  fallback_used: boolean;
  cached: boolean;
  hits: Hit[];
}

// A request that got no usable answer: its HTTP status (null: no response at all), its Retry-After header, and the
// API's `detail` when that is a text (a 422's is a list, and a proxy's or the CDN's error page is not JSON).
export interface RequestFailure {
  status: number | null;
  retryAfter: string | null;
  detail: string | null;
}

export type Outcome<T> =
  | { ok: true; data: T }
  | { ok: false; failure: RequestFailure };

// What GET /health on this origin says that the page uses: whether the API runs the public demo (public_demo, read
// from its 503 too: an API that is still starting), and whether its reranker is loaded (reranker_loaded; null when
// the answer does not say, as a starting API's does not). Anything else (no API here, a development server's 404
// page, a network error, an API from before the flag) means the chat interface, as before.
export interface HealthInfo {
  publicDemo: boolean;
  rerankerLoaded: boolean | null;
}

export async function readHealth(): Promise<HealthInfo> {
  try {
    const response = await apiFetch(HEALTH_PATH, {
      headers: { accept: "application/json" },
    });
    const body = (await response.json()) as {
      public_demo?: unknown;
      reranker_loaded?: unknown;
    } | null;
    return {
      publicDemo: body?.public_demo === true,
      rerankerLoaded:
        typeof body?.reranker_loaded === "boolean"
          ? body.reranker_loaded
          : null,
    };
  } catch {
    return { publicDemo: false, rerankerLoaded: null };
  }
}

export async function postJson<T>(
  path: string,
  body: unknown,
): Promise<Outcome<T>> {
  let response: Response;
  try {
    response = await apiFetch(path, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "application/json",
      },
      body: JSON.stringify(body),
    });
  } catch {
    return {
      ok: false,
      failure: { status: null, retryAfter: null, detail: null },
    };
  }
  let parsed: unknown = null;
  try {
    parsed = await response.json();
  } catch {
    parsed = null; // not JSON: an error page from a proxy or the CDN
  }
  if (response.ok && parsed !== null && typeof parsed === "object")
    return { ok: true, data: parsed as T };
  const detail = (parsed as { detail?: unknown } | null)?.detail;
  return {
    ok: false,
    failure: {
      status: response.status,
      retryAfter: response.headers.get("retry-after"),
      detail: typeof detail === "string" ? detail : null,
    },
  };
}

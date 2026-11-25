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
// from its 503 too: an API that is still starting), whether its reranker is loaded (reranker_loaded; null when the
// answer does not say, as a starting API's does not), and what it reports about its index and its retrieval stages
// (system: the Dataset and Pipeline sections show it; null when the answer is not a JSON object). Anything else (no
// API here, a development server's 404 page, a network error, an API from before the flag) means the chat interface,
// as before.
export interface HealthInfo {
  publicDemo: boolean;
  rerankerLoaded: boolean | null;
  system: SystemInfo | null;
}

// The fields of /health (main.py) the page shows, each null when the answer does not have it in the type expected:
// the page shows "not reported" then, never a value of its own.
export interface SystemInfo {
  status: string | null;
  rows: number | null; // n_nodes: the rows of the data table, counted for this answer
  files: number | null; // indexing.manifest_at_start.files
  chunks: number | null; // indexing.manifest_at_start.chunks
  pipelines: string[]; // indexing.manifest_at_start.pipelines
  chunker: string | null; // indexing.chunker
  embeddingProvider: string | null;
  embeddingModel: string | null;
  queryInstruction: string | null; // the text put in front of every query ("" = the bare query)
  embeddingDevice: string | null;
  lexicalBackend: string | null; // bm25_backend: the configured backend
  retrievalMode: string | null;
  denseDepth: number | null;
  denseSettings: [string, string][]; // dense_search.settings, as the API names them
  pgvector: string | null;
  reranker: {
    configured: boolean | null;
    loaded: boolean | null; // reranker.loaded, or reranker_loaded when the answer has no reranker entry
    model: string | null;
    device: string | null;
    precision: string | null;
    maxLength: number | null;
    depth: number | null;
    error: string | null;
  };
  degraded: { component: string; error: string }[];
}

type Json = Record<string, unknown>;

function record(value: unknown): Json {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Json)
    : {};
}

function text(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function count(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function flag(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

export function readSystem(body: unknown): SystemInfo | null {
  if (body === null || typeof body !== "object" || Array.isArray(body))
    return null;
  const health = body as Json;
  const indexing = record(health.indexing);
  const manifest = record(indexing.manifest_at_start);
  const dense = record(health.dense_search);
  const reranker = record(health.reranker);
  return {
    status: text(health.status),
    rows: count(health.n_nodes),
    files: count(manifest.files),
    chunks: count(manifest.chunks),
    pipelines: Array.isArray(manifest.pipelines)
      ? manifest.pipelines.filter(
          (name): name is string => typeof name === "string",
        )
      : [],
    chunker: text(indexing.chunker),
    embeddingProvider: text(health.embedding_provider),
    embeddingModel: text(health.embedding_model),
    queryInstruction: text(health.embedding_query_instruction),
    embeddingDevice: text(record(health.devices).embedding),
    lexicalBackend: text(health.bm25_backend),
    retrievalMode: text(health.retrieval_mode),
    denseDepth: count(dense.depth),
    denseSettings: Object.entries(record(dense.settings)).flatMap(
      ([name, value]): [string, string][] =>
        typeof value === "string" || typeof value === "number"
          ? [[name, String(value)]]
          : [],
    ),
    pgvector: text(dense.pgvector),
    reranker: {
      configured: flag(reranker.configured),
      loaded: flag(reranker.loaded) ?? flag(health.reranker_loaded),
      model: text(reranker.model),
      device: text(reranker.device),
      precision: text(reranker.precision),
      maxLength: count(reranker.max_length),
      depth: count(reranker.depth),
      error: text(reranker.error),
    },
    degraded: Array.isArray(health.degraded)
      ? health.degraded.flatMap((entry) => {
          const { component, error } = record(entry);
          return typeof component === "string" && typeof error === "string"
            ? [{ component, error }]
            : [];
        })
      : [],
  };
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
      system: readSystem(body),
    };
  } catch {
    return { publicDemo: false, rerankerLoaded: null, system: null };
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

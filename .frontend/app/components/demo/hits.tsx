import type { Hit, StageScores } from "./api";

// The retrieved excerpts, as both routes return them: each hit's path, byte range, scores and text. Every text from
// the API is rendered as text, never as HTML.

// The per-stage scores the API names, in pipeline order. `bm25` is the lexical leg's score whatever its backend
// (the response's bm25_backend says which: PostgreSQL ts_rank_cd unless BM25 is configured).
const STAGES: [keyof StageScores, string][] = [
  ["bm25", "lexical"],
  ["dense", "dense"],
  ["path", "path"],
  ["rrf", "fused (RRF)"],
  ["rerank", "rerank"],
];
const SNIPPET_CHARS = 280;

// Byte offsets in the file, the end exclusive (app/engine/ingest.py).
export function byteRange(start: number | null, end: number | null): string {
  return start !== null && end !== null
    ? `bytes ${start}–${end}`
    : "byte range not recorded";
}

function number(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toPrecision(3);
}

export function scoreLine(hit: Hit): string {
  const stages = STAGES.filter(
    ([name]) => typeof hit.scores?.[name] === "number",
  ).map(([name, label]) => `${label} ${number(hit.scores[name] as number)}`);
  return [`score ${number(hit.score)}`, ...stages].join(" · ");
}

function Excerpt({ text }: { text: string }) {
  if (text.length <= SNIPPET_CHARS)
    return (
      <p className="whitespace-pre-wrap break-words font-mono text-xs">
        {text}
      </p>
    );
  const space = text.lastIndexOf(" ", SNIPPET_CHARS);
  const head = text.slice(0, space > SNIPPET_CHARS / 2 ? space : SNIPPET_CHARS);
  return (
    <>
      <p className="whitespace-pre-wrap break-words font-mono text-xs">
        {head}…
      </p>
      <details className="text-xs">
        <summary className="cursor-pointer">The whole excerpt</summary>
        <p className="whitespace-pre-wrap break-words font-mono">{text}</p>
      </details>
    </>
  );
}

export default function HitList({ hits }: { hits: Hit[] }) {
  if (hits.length === 0)
    return <p className="text-muted-foreground">No excerpt was retrieved.</p>;
  return (
    <ol className="space-y-2">
      {hits.map((hit) => (
        <li
          key={hit.chunk_id}
          id={`demo-hit-${hit.rank}`}
          tabIndex={-1}
          className="space-y-1 rounded-lg border bg-white p-3"
        >
          <p className="flex flex-wrap items-baseline gap-x-2">
            <span className="font-semibold">{hit.rank}.</span>
            <code className="break-all">{hit.path ?? hit.chunk_id}</code>
            <span className="text-muted-foreground">
              {byteRange(hit.start_byte, hit.end_byte)}
            </span>
          </p>
          <p className="text-xs text-muted-foreground">{scoreLine(hit)}</p>
          <Excerpt text={hit.text} />
        </li>
      ))}
    </ol>
  );
}

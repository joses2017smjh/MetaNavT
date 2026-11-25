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
const EXCERPT =
  "whitespace-pre-wrap break-words rounded-xl bg-peach/70 px-3.5 py-3 font-mono text-xs leading-relaxed text-ink/85";

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
  if (text.length <= SNIPPET_CHARS) return <p className={EXCERPT}>{text}</p>;
  const space = text.lastIndexOf(" ", SNIPPET_CHARS);
  const head = text.slice(0, space > SNIPPET_CHARS / 2 ? space : SNIPPET_CHARS);
  return (
    <>
      <p className={EXCERPT}>{head}…</p>
      <details className="text-xs">
        <summary className="w-fit cursor-pointer rounded text-muted-foreground underline-offset-4 hover:text-ink hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
          The whole excerpt
        </summary>
        <p className={`${EXCERPT} mt-2`}>{text}</p>
      </details>
    </>
  );
}

export default function HitList({ hits }: { hits: Hit[] }) {
  if (hits.length === 0)
    return <p className="text-muted-foreground">No excerpt was retrieved.</p>;
  return (
    <ol className="space-y-3">
      {hits.map((hit) => (
        <li
          key={hit.chunk_id}
          id={`demo-hit-${hit.rank}`}
          tabIndex={-1}
          className="space-y-2 rounded-2xl border border-black/[0.07] bg-white p-4 shadow-pill transition-shadow duration-200 target:border-brand/60 target:shadow-soft focus:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:p-5"
        >
          <p className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <span className="font-serif text-2xl leading-none text-brand-deep">
              {hit.rank}.
            </span>
            <code className="break-all font-mono text-[13px] font-medium text-ink">
              {hit.path ?? hit.chunk_id}
            </code>
            <span className="text-xs text-muted-foreground">
              {byteRange(hit.start_byte, hit.end_byte)}
            </span>
          </p>
          <p className="font-mono text-[11px] leading-relaxed text-muted-foreground">
            {scoreLine(hit)}
          </p>
          <Excerpt text={hit.text} />
        </li>
      ))}
    </ol>
  );
}

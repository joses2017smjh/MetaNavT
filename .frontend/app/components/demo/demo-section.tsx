"use client";

import { Quote, Search } from "lucide-react";
import { type RefObject, useEffect, useRef, useState } from "react";
import { BENCHMARKS, REPOSITORY } from "../site";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import {
  ANSWER_K,
  ANSWER_PATH,
  type AnswerResponse,
  type Citation,
  MAX_QUESTION_CHARS,
  postJson,
  type RequestFailure,
  RETRIEVE_PATH,
  type RetrieveResponse,
  SEARCH_K,
} from "./api";
import HitList, { byteRange } from "./hits";
import { failureText, reasonText } from "./text";

// The public demo's home page (PUBLIC_DEMO=1; home-switch.tsx shows it when GET /health says so): one question box,
// Search (POST /api/retrieve/) and Ask (POST /api/answer/), on the page's own origin only.

const EXAMPLES = [
  "What learning rate did run 47 use?",
  "What encoder did run 40 use?",
];

// Shared looks: a result's heading, a smaller heading inside it, a line of facts about a result, a note about how
// the result was made (as prominent as the result's own facts), a link in running text, and a cited passage's card.
const RESULT_HEADING =
  "break-words font-serif text-[2rem] font-normal leading-[1.05] tracking-[-0.02em] text-brand-deep focus:outline-none sm:text-4xl";
const SUB_HEADING =
  "font-serif text-2xl font-medium tracking-[-0.02em] text-brand-deep";
const FACTS = "font-mono text-[11px] leading-relaxed text-muted-foreground";
const NOTE =
  "rounded-xl border border-black/[0.07] bg-muted px-3.5 py-2.5 text-xs leading-relaxed text-ink/80";
const LINK =
  "font-medium text-brand-deep underline decoration-brand/60 underline-offset-[3px] transition-colors duration-200 hover:decoration-brand focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";
const PASSAGE =
  "space-y-2 rounded-2xl border border-black/[0.07] bg-white p-4 shadow-pill target:border-brand/60 target:shadow-soft focus:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:p-5";

type Action = "search" | "ask";
type Result =
  | { kind: "search"; question: string; data: RetrieveResponse }
  | { kind: "ask"; data: AnswerResponse }
  | { kind: "failure"; action: Action; failure: RequestFailure };

// One entry per cited passage: a citation repeated with the same excerpt and range gets one number.
function distinct(citations: Citation[]): Citation[] {
  const seen = new Set<string>();
  return citations.filter((citation) => {
    const key = `${citation.chunk_id}:${citation.start_char}:${citation.end_char}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function retrievalLine(data: RetrieveResponse): string {
  const lexical =
    data.bm25_backend === "ts_rank_cd"
      ? "ts_rank_cd (PostgreSQL full-text rank, not BM25)"
      : (data.bm25_backend ?? "not reported");
  const parts = [
    `route ${data.route ?? "none"}`,
    `lexical ${lexical}`,
    `embeddings ${data.embedding_provider}`,
    `reranker ${data.reranker_loaded ? (data.reranker?.model ?? "loaded") : "off"}`,
  ];
  const total = data.latency_ms?.total;
  if (typeof total === "number") parts.push(`${Math.round(total)} ms`);
  return parts.join(" · ");
}

// rerankerLoaded: GET /health's reranker_loaded as the page read it at load (null: not said). Search reads its own
// response's instead; Ask's response has no such field.
export default function DemoSection({
  rerankerLoaded = null,
}: {
  rerankerLoaded?: boolean | null;
}) {
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState<Action | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const heading = useRef<HTMLHeadingElement>(null);

  // A new result takes the keyboard focus to its heading, so a screen reader starts there.
  useEffect(() => {
    if (result) heading.current?.focus();
  }, [result]);

  const run = async (action: Action) => {
    const text = question.trim();
    if (!text || busy) return;
    setBusy(action);
    if (action === "search") {
      const outcome = await postJson<RetrieveResponse>(RETRIEVE_PATH, {
        query: text,
        k: SEARCH_K,
      });
      setResult(
        outcome.ok
          ? { kind: "search", question: text, data: outcome.data }
          : { kind: "failure", action, failure: outcome.failure },
      );
    } else {
      const outcome = await postJson<AnswerResponse>(ANSWER_PATH, {
        question: text,
        k: ANSWER_K,
      });
      if (outcome.ok && outcome.data.model) setModel(outcome.data.model);
      setResult(
        outcome.ok
          ? { kind: "ask", data: outcome.data }
          : { kind: "failure", action, failure: outcome.failure },
      );
    }
    setBusy(null);
  };

  const blank = !question.trim();
  return (
    <section
      id="search"
      aria-labelledby="demo-title"
      className="space-y-10 sm:space-y-12"
    >
      <div className="rise mx-auto max-w-3xl space-y-5 text-center">
        <p className="eyebrow">
          <span aria-hidden className="h-1.5 w-1.5 rounded-full bg-brand" />
          Public demo
        </p>
        <h1
          id="demo-title"
          className="display text-[clamp(2.75rem,6.4vw,4.75rem)]"
        >
          Search a research project’s files
        </h1>
        <p className="mx-auto max-w-2xl text-[15px] leading-relaxed text-muted-foreground sm:text-[17px]">
          This public demo searches the fixture this repository benchmarks: a
          small experiment tree of YAML configs, Slurm scripts, training logs,
          an ablation table, paper drafts, Python modules and checkpoint
          metadata. Search lists the excerpts that match best, each with its
          file and byte range; Ask has Claude answer from them and cite them.
        </p>
      </div>

      <div className="rise rise-2 mx-auto max-w-4xl space-y-5 rounded-card border border-white bg-white/90 p-5 shadow-card sm:p-8">
        <form
          className="space-y-3"
          onSubmit={(event) => {
            event.preventDefault();
            void run("search");
          }}
        >
          <label
            htmlFor="demo-question"
            className="block text-[11px] uppercase tracking-[0.08em] text-muted-foreground"
          >
            Question
          </label>
          <div className="flex flex-col gap-2 sm:flex-row">
            <Input
              id="demo-question"
              name="question"
              value={question}
              maxLength={MAX_QUESTION_CHARS}
              autoComplete="off"
              placeholder={EXAMPLES[0]}
              aria-describedby="demo-question-help"
              onChange={(event) => setQuestion(event.target.value)}
              className="h-12 flex-1 text-base"
            />
            <div className="flex gap-2">
              <Button
                type="submit"
                disabled={busy !== null || blank}
                className="h-12 flex-1 sm:flex-none"
              >
                <Search aria-hidden className="h-4 w-4" />
                Search
              </Button>
              <Button
                type="button"
                variant="outline"
                disabled={busy !== null || blank}
                onClick={() => void run("ask")}
                className="h-12 flex-1 sm:flex-none"
              >
                <Quote aria-hidden className="h-4 w-4" />
                Ask
              </Button>
            </div>
          </div>
          <p id="demo-question-help" className="text-xs text-muted-foreground">
            {question.length} of {MAX_QUESTION_CHARS} characters. Search lists
            the {SEARCH_K} best excerpts (Enter searches); Ask sends the{" "}
            {ANSWER_K} best to Claude.
          </p>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="text-muted-foreground">Try:</span>
            {EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                className="rounded-full border border-black/10 bg-white px-3 py-1.5 text-left text-[13px] text-ink/80 shadow-pill transition-colors duration-200 hover:border-brand/50 hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                onClick={() => setQuestion(example)}
              >
                {example}
              </button>
            ))}
            <span role="status" className="ml-auto text-[13px] text-brand-deep">
              {busy === "search"
                ? "Searching…"
                : busy === "ask"
                  ? "Asking Claude…"
                  : ""}
            </span>
          </div>
        </form>

        <aside
          aria-label="About the answers"
          className="rounded-2xl border border-brand/25 bg-peach/70 px-4 py-3.5 text-sm leading-relaxed text-ink/85"
        >
          Answers come from Claude, Anthropic’s model
          {model ? (
            <>
              {" "}
              (the last one from{" "}
              <code className="font-mono text-[13px]">{model}</code>, as the API
              reported it)
            </>
          ) : null}
          , and are not evaluated in this repository: no committed result
          measures their quality. Retrieval is the benchmarked part:{" "}
          <a className={LINK} href={BENCHMARKS}>
            benchmark results in the README
          </a>{" "}
          ·{" "}
          <a className={LINK} href={REPOSITORY}>
            source code
          </a>
          .
        </aside>
      </div>

      {result?.kind === "failure" && (
        <div className="mx-auto max-w-4xl space-y-3">
          <h2 ref={heading} tabIndex={-1} className={RESULT_HEADING}>
            {result.action === "search" ? "Search failed" : "No answer"}
          </h2>
          <p
            role="alert"
            className="rounded-2xl border border-destructive/25 bg-destructive/[0.04] px-4 py-3 text-[15px] leading-relaxed text-destructive"
          >
            {failureText(result.failure)}
          </p>
        </div>
      )}

      {result?.kind === "search" && (
        <div className="mx-auto max-w-4xl space-y-4">
          <h2 ref={heading} tabIndex={-1} className={RESULT_HEADING}>
            {result.data.hits.length === 1
              ? "1 excerpt"
              : `${result.data.hits.length} excerpts`}{" "}
            for “{result.question}”
          </h2>
          <p className={FACTS}>{retrievalLine(result.data)}</p>
          {!result.data.reranker_loaded && <RerankerOff />}
          {result.data.degraded?.length > 0 && (
            <p className={NOTE}>
              Degraded:{" "}
              {result.data.degraded
                .map((entry) => `${entry.component} (${entry.error})`)
                .join("; ")}
            </p>
          )}
          <HitList hits={result.data.hits} />
        </div>
      )}

      {result?.kind === "ask" && (
        <AnswerView
          data={result.data}
          heading={heading}
          rerankerOff={rerankerLoaded === false}
        />
      )}
    </section>
  );
}

// A server without the cross-encoder (RERANKER_MODEL=none): the hits keep the fused order, a fallback the README's
// benchmark tables measure against the reranker (their Neural models on the fixture table, in the section the link
// opens: the table itself sits in a collapsed block, which a link into it would not open).
function RerankerOff() {
  return (
    <p className={NOTE}>
      Reranker: off on this server, so the hits are in the fused order of the
      lexical and dense lists (a fallback).{" "}
      <a className={LINK} href={BENCHMARKS}>
        What the reranker adds: the README’s benchmark tables, Neural models on
        the fixture
      </a>
      .
    </p>
  );
}

// The answer beside the excerpts it was written from (one column on a narrow screen): a citation's link jumps to its
// passage, and a passage's link to the retrieved excerpt it is in.
function AnswerView({
  data,
  heading,
  rerankerOff,
}: {
  data: AnswerResponse;
  heading: RefObject<HTMLHeadingElement | null>;
  rerankerOff: boolean;
}) {
  const citations = distinct(data.citations ?? []);
  const hitRank = (chunk: string) =>
    data.hits.find((hit) => hit.chunk_id === chunk)?.rank;
  return (
    <div className="grid gap-10 lg:grid-cols-2 lg:gap-8">
      <div className="space-y-4 lg:sticky lg:top-28 lg:-m-3 lg:max-h-[calc(100vh-8rem)] lg:self-start lg:overflow-y-auto lg:p-3">
        <h2 ref={heading} tabIndex={-1} className={RESULT_HEADING}>
          {data.answer !== null ? "Answer" : "No answer"}
        </h2>
        {data.answer !== null ? (
          <>
            <p className={FACTS}>
              From Claude, model <code>{data.model ?? "not reported"}</code>
              {data.cached ? " · served from the answer cache" : ""}
              {data.fallback_used ? " · a fallback model answered" : ""} · not
              evaluated in this repository
            </p>
            <p className="whitespace-pre-wrap break-words rounded-2xl border border-black/[0.07] bg-white p-5 text-[17px] leading-relaxed text-ink shadow-soft">
              {data.answer}{" "}
              {citations.map((citation, index) => (
                <sup key={index} className="mx-0.5">
                  <a
                    className={`${LINK} font-mono text-[11px]`}
                    href={`#demo-citation-${index + 1}`}
                    aria-label={`Citation ${index + 1}`}
                  >
                    [{index + 1}]
                  </a>
                </sup>
              ))}
            </p>
            <h3 className={`${SUB_HEADING} pt-2`}>Cited excerpts</h3>
            <ol className="space-y-3">
              {citations.map((citation, index) => {
                const rank = hitRank(citation.chunk_id);
                return (
                  <li
                    key={index}
                    id={`demo-citation-${index + 1}`}
                    tabIndex={-1}
                    className={PASSAGE}
                  >
                    <p className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                      <span className="font-mono text-[13px] font-medium text-brand-deep">
                        [{index + 1}]
                      </span>
                      <code className="break-all font-mono text-[13px] font-medium text-ink">
                        {citation.path ?? citation.chunk_id}
                      </code>
                      <span className="text-xs text-muted-foreground">
                        {byteRange(citation.start_byte, citation.end_byte)}
                      </span>
                    </p>
                    <blockquote className="whitespace-pre-wrap break-words border-l-2 border-brand/70 pl-3.5 font-mono text-xs leading-relaxed text-ink/85">
                      {citation.cited_text}
                    </blockquote>
                    {rank !== undefined && (
                      <a
                        className={`${LINK} inline-block text-xs`}
                        href={`#demo-hit-${rank}`}
                      >
                        In retrieved excerpt {rank}
                      </a>
                    )}
                  </li>
                );
              })}
            </ol>
          </>
        ) : (
          <>
            <p className="text-[15px] leading-relaxed text-ink/85">
              {reasonText(data.reason ?? "not answered")}
            </p>
            {data.uncited_reply && (
              <figure className="space-y-2 rounded-2xl border border-dashed border-black/25 bg-white/60 p-4">
                <figcaption className="text-[11px] uppercase tracking-[0.08em] text-muted-foreground">
                  Unverified reply: no citation supports it
                </figcaption>
                <blockquote className="whitespace-pre-wrap break-words text-[15px] leading-relaxed text-ink/85">
                  {data.uncited_reply}
                </blockquote>
              </figure>
            )}
          </>
        )}
      </div>
      <div className="space-y-4">
        <h3 className={`${SUB_HEADING} lg:pt-2`}>Retrieved excerpts</h3>
        {rerankerOff && <RerankerOff />}
        <HitList hits={data.hits} />
      </div>
    </div>
  );
}

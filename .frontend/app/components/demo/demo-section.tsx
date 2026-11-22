"use client";

import { type RefObject, useEffect, useRef, useState } from "react";
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

const REPOSITORY = "https://github.com/joses2017smjh/MetaNavT";
const BENCHMARKS = `${REPOSITORY}#generated-benchmark-evidence`;
const EXAMPLES = [
  "What learning rate did run 47 use?",
  "What encoder did run 40 use?",
];

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
      aria-labelledby="demo-title"
      className="w-full space-y-4 rounded-xl bg-white/90 p-4 pb-24 text-sm lg:pb-4"
    >
      <div className="space-y-1">
        <h1 id="demo-title" className="text-lg font-semibold">
          Search a research project’s files
        </h1>
        <p>
          This public demo searches the fixture this repository benchmarks: a
          small experiment tree of YAML configs, Slurm scripts, training logs,
          an ablation table, paper drafts, Python modules and checkpoint
          metadata. Search lists the excerpts that match best, each with its
          file and byte range; Ask has Claude answer from them and cite them.
        </p>
      </div>

      <aside
        aria-label="About the answers"
        className="rounded-lg border border-amber-300 bg-amber-50 p-3"
      >
        Answers come from Claude, Anthropic’s model
        {model ? (
          <>
            {" "}
            (the last one from <code>{model}</code>, as the API reported it)
          </>
        ) : null}
        , and are not evaluated in this repository: no committed result measures
        their quality. Retrieval is the benchmarked part:{" "}
        <a className="underline" href={BENCHMARKS}>
          benchmark results in the README
        </a>{" "}
        ·{" "}
        <a className="underline" href={REPOSITORY}>
          source code
        </a>
        .
      </aside>

      <form
        className="space-y-2"
        onSubmit={(event) => {
          event.preventDefault();
          void run("search");
        }}
      >
        <label htmlFor="demo-question" className="block font-medium">
          Question
        </label>
        <Input
          id="demo-question"
          name="question"
          value={question}
          maxLength={MAX_QUESTION_CHARS}
          autoComplete="off"
          placeholder={EXAMPLES[0]}
          aria-describedby="demo-question-help"
          onChange={(event) => setQuestion(event.target.value)}
        />
        <p id="demo-question-help" className="text-xs text-muted-foreground">
          {question.length} of {MAX_QUESTION_CHARS} characters. Search lists the{" "}
          {SEARCH_K} best excerpts (Enter searches); Ask sends the {ANSWER_K}{" "}
          best to Claude.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <Button type="submit" disabled={busy !== null || blank}>
            Search
          </Button>
          <Button
            type="button"
            variant="outline"
            disabled={busy !== null || blank}
            onClick={() => void run("ask")}
          >
            Ask
          </Button>
          <span role="status" className="text-muted-foreground">
            {busy === "search"
              ? "Searching…"
              : busy === "ask"
                ? "Asking Claude…"
                : ""}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-1 text-xs">
          <span>Try:</span>
          {EXAMPLES.map((example) => (
            <Button
              key={example}
              type="button"
              size="sm"
              variant="ghost"
              className="h-auto whitespace-normal py-1 text-left"
              onClick={() => setQuestion(example)}
            >
              {example}
            </Button>
          ))}
        </div>
      </form>

      {result?.kind === "failure" && (
        <div className="space-y-2">
          <h2
            ref={heading}
            tabIndex={-1}
            className="break-words text-base font-semibold focus:outline-none"
          >
            {result.action === "search" ? "Search failed" : "No answer"}
          </h2>
          <p role="alert" className="text-red-700">
            {failureText(result.failure)}
          </p>
        </div>
      )}

      {result?.kind === "search" && (
        <div className="space-y-2">
          <h2
            ref={heading}
            tabIndex={-1}
            className="break-words text-base font-semibold focus:outline-none"
          >
            {result.data.hits.length === 1
              ? "1 excerpt"
              : `${result.data.hits.length} excerpts`}{" "}
            for “{result.question}”
          </h2>
          <p className="text-xs text-muted-foreground">
            {retrievalLine(result.data)}
          </p>
          {!result.data.reranker_loaded && <RerankerOff />}
          {result.data.degraded?.length > 0 && (
            <p className="text-xs">
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
    <p className="text-xs">
      Reranker: off on this server, so the hits are in the fused order of the
      lexical and dense lists (a fallback).{" "}
      <a className="underline" href={BENCHMARKS}>
        What the reranker adds: the README’s benchmark tables, Neural models on
        the fixture
      </a>
      .
    </p>
  );
}

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
    <div className="space-y-2">
      <h2
        ref={heading}
        tabIndex={-1}
        className="break-words text-base font-semibold focus:outline-none"
      >
        {data.answer !== null ? "Answer" : "No answer"}
      </h2>
      {data.answer !== null ? (
        <>
          <p className="text-xs text-muted-foreground">
            From Claude, model <code>{data.model ?? "not reported"}</code>
            {data.cached ? " · served from the answer cache" : ""}
            {data.fallback_used ? " · a fallback model answered" : ""} · not
            evaluated in this repository
          </p>
          <p className="whitespace-pre-wrap break-words rounded-lg border bg-white p-3">
            {data.answer}{" "}
            {citations.map((citation, index) => (
              <sup key={index} className="mx-0.5">
                <a
                  className="underline"
                  href={`#demo-citation-${index + 1}`}
                  aria-label={`Citation ${index + 1}`}
                >
                  [{index + 1}]
                </a>
              </sup>
            ))}
          </p>
          <h3 className="font-semibold">Cited excerpts</h3>
          <ol className="space-y-2">
            {citations.map((citation, index) => {
              const rank = hitRank(citation.chunk_id);
              return (
                <li
                  key={index}
                  id={`demo-citation-${index + 1}`}
                  tabIndex={-1}
                  className="space-y-1 rounded-lg border bg-white p-3"
                >
                  <p className="flex flex-wrap items-baseline gap-x-2">
                    <span className="font-semibold">[{index + 1}]</span>
                    <code className="break-all">
                      {citation.path ?? citation.chunk_id}
                    </code>
                    <span className="text-muted-foreground">
                      {byteRange(citation.start_byte, citation.end_byte)}
                    </span>
                  </p>
                  <blockquote className="whitespace-pre-wrap break-words border-l-2 pl-3 font-mono text-xs">
                    {citation.cited_text}
                  </blockquote>
                  {rank !== undefined && (
                    <a className="text-xs underline" href={`#demo-hit-${rank}`}>
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
          <p>{reasonText(data.reason ?? "not answered")}</p>
          {data.uncited_reply && (
            <figure className="space-y-1 rounded-lg border border-dashed p-3">
              <figcaption className="text-xs font-semibold">
                Unverified reply: no citation supports it
              </figcaption>
              <blockquote className="whitespace-pre-wrap break-words">
                {data.uncited_reply}
              </blockquote>
            </figure>
          )}
        </>
      )}
      <h3 className="font-semibold">Retrieved excerpts</h3>
      {rerankerOff && <RerankerOff />}
      <HitList hits={data.hits} />
    </div>
  );
}

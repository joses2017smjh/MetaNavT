"use client";

import { useState } from "react";
import { ANSWER_K, type SystemInfo } from "../demo/api";

// The stages a question passes through (app/retrieval/router.py, app/engine/retriever.py; the answer route on the
// public demo), each with the settings this server's /health answer reported when the page loaded. One stage is open
// at a time. A setting the answer does not have is shown as "not reported"; a fallback is named as one.

const NOT_REPORTED = "not reported";

interface Stage {
  key: string;
  name: string; // in the list of stages
  title: string; // over the open stage
  summary: string;
  note?: string;
  facts: [string, string | number | null][];
}

// The lexical leg's backend, saying what ts_rank_cd is not (as the search results' own line does).
function lexicalBackend(backend: string | null): string | null {
  return backend === "ts_rank_cd"
    ? "ts_rank_cd (PostgreSQL full-text rank, not BM25)"
    : backend;
}

function queryPrefix(instruction: string | null): string | null {
  if (instruction === null) return null;
  return instruction === "" ? "none (the bare query)" : `“${instruction}”`;
}

function stagesOf(system: SystemInfo, demo: boolean): Stage[] {
  const reranker = system.reranker;
  const rerank: Stage = {
    key: "rerank",
    name: "Rerank",
    title: "Rerank the top",
    summary:
      "A cross-encoder reads the question together with each of the top fused excerpts and reorders them.",
    facts: [["Model", reranker.model]],
  };
  if (reranker.loaded === true)
    rerank.facts.push(
      ["Depth", reranker.depth],
      ["Max length", reranker.maxLength],
      ["Precision", reranker.precision],
      ["Device", reranker.device],
    );
  else if (reranker.configured === true)
    rerank.note = `Configured on this server but not loaded${reranker.error ? ` (${reranker.error})` : ""}: the hits keep the fused order of the lexical and dense lists (a fallback).`;
  else if (reranker.loaded === false) {
    rerank.note =
      "Off on this server: the hits keep the fused order of the lexical and dense lists (a fallback).";
    rerank.facts = [["Model", "none"]];
  }

  return [
    {
      key: "route",
      name: "Route",
      title: "Route the question",
      summary:
        "Regular expressions, no model: the first rule that matches picks the route. A path or file name skips the query embedding and the reranker and adds a path lookup; counting and comparing questions skip the reranker; everything else runs every stage.",
      facts: [["Retrieval mode", system.retrievalMode]],
    },
    {
      key: "lexical",
      name: "Lexical",
      title: "Match the words",
      summary:
        "A keyword list: the chunks that share the question’s words, ranked by the backend below.",
      facts: [["Backend", lexicalBackend(system.lexicalBackend)]],
    },
    {
      key: "dense",
      name: "Dense",
      title: "Match the meaning",
      summary:
        "The question is embedded, and the chunks whose vectors lie nearest to it make a second list.",
      facts: [
        ["Embeddings", system.embeddingProvider],
        ["Model", system.embeddingModel],
        ["Query prefix", queryPrefix(system.queryInstruction)],
        ["Device", system.embeddingDevice],
        ["pgvector", system.pgvector],
        ["List depth", system.denseDepth],
        ...system.denseSettings,
      ],
    },
    {
      key: "fuse",
      name: "Fuse",
      title: "Fuse the lists",
      summary:
        "Reciprocal rank fusion (RRF) merges the lexical and dense lists into one order. On a path route there is no dense list: the path lookup’s list is fused with the lexical one.",
      facts: [],
    },
    rerank,
    ...(demo
      ? [
          {
            key: "answer",
            name: "Cited answer",
            title: "Answer with citations",
            summary: `Ask sends the ${ANSWER_K} best excerpts to Claude, Anthropic’s model. The reply is shown as an answer only when it cites them; a reply without a citation is shown apart, labelled unverified.`,
            note: "Answers are not evaluated in this repository: no committed result measures their quality.",
            facts: [],
          } satisfies Stage,
        ]
      : []),
  ];
}

export default function PipelineSection({
  system,
  demo,
}: {
  system: SystemInfo;
  demo: boolean;
}) {
  const stages = stagesOf(system, demo);
  const [open, setOpen] = useState(stages[0].key);
  const stage = stages.find((entry) => entry.key === open) ?? stages[0];
  return (
    <section
      id="pipeline"
      aria-labelledby="pipeline-title"
      className="space-y-10 rounded-[2rem] bg-sage/35 px-5 py-10 sm:rounded-[2.75rem] sm:px-12 sm:py-16"
    >
      <div className="max-w-2xl space-y-5">
        <p className="eyebrow">Pipeline</p>
        <h2
          id="pipeline-title"
          className="display text-[clamp(2.25rem,4.6vw,3.5rem)] leading-[1.02]"
        >
          From a question to ranked excerpts
        </h2>
        <p className="max-w-xl text-[15px] leading-relaxed text-ink/70">
          The stages a question passes through, each with the settings this
          server’s{" "}
          <code className="font-mono text-[13px] text-ink">/health</code>{" "}
          reported when the page loaded.
        </p>
      </div>
      <div className="grid gap-8 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:gap-16">
        <ol className="flex flex-row flex-wrap gap-x-5 gap-y-1 lg:flex-col lg:gap-y-2">
          {stages.map((entry, index) => {
            const active = entry.key === stage.key;
            return (
              <li key={entry.key}>
                <button
                  type="button"
                  aria-pressed={active}
                  onClick={() => setOpen(entry.key)}
                  className={`group flex items-baseline gap-3 rounded-lg text-left text-[clamp(1.375rem,2.5vw,2rem)] leading-tight tracking-[-0.02em] transition-colors duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                    active ? "text-ink" : "text-ink/45 hover:text-ink"
                  }`}
                >
                  <span
                    aria-hidden
                    className={`hidden font-mono text-xs tracking-normal lg:inline ${
                      active ? "text-brand-deep" : "text-ink/45"
                    }`}
                  >
                    {String(index + 1).padStart(2, "0")}
                  </span>
                  {entry.name}
                </button>
              </li>
            );
          })}
        </ol>
        <div className="space-y-5 rounded-card border border-white bg-white p-6 shadow-card sm:p-8">
          <h3 className="font-serif text-3xl font-medium leading-none tracking-[-0.02em] text-brand-deep">
            {stage.title}
          </h3>
          <p className="text-[15px] leading-relaxed text-ink/80">
            {stage.summary}
          </p>
          {stage.note && (
            <p className="rounded-xl border border-brand/25 bg-peach/70 px-3.5 py-2.5 text-sm leading-relaxed text-ink/85">
              {stage.note}
            </p>
          )}
          {stage.facts.length > 0 && (
            <dl className="divide-y divide-black/10 border-t border-black/10">
              {stage.facts.map(([label, value]) => (
                <div
                  key={label}
                  className="grid gap-1 py-3 sm:grid-cols-[10rem_minmax(0,1fr)] sm:gap-6"
                >
                  <dt className="text-[11px] uppercase leading-5 tracking-[0.08em] text-muted-foreground">
                    {label}
                  </dt>
                  <dd
                    className={
                      value !== null
                        ? "break-words font-mono text-[13px] leading-5 text-ink"
                        : "text-sm leading-5 text-muted-foreground"
                    }
                  >
                    {value ?? NOT_REPORTED}
                  </dd>
                </div>
              ))}
            </dl>
          )}
        </div>
      </div>
    </section>
  );
}

// organize-imports-ignore: "./dom" must stay the first import (it sets up the browser globals the others read)
import "./dom";
import { readSystem } from "@/app/components/demo/api";
import HomeSwitch from "@/app/components/home-switch";
import { fireEvent, render, screen, within } from "@testing-library/react";
import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";
import { json, mockApi, reset, type Call } from "./mock-api";

// The page's sections against a mocked API. The navigation bar links the sections the page shows (the public demo's,
// or the chat interface's). Dataset and Pipeline show what the one GET /health answer says about the index and the
// retrieval stages: a field the answer does not have is "not reported", never a value of the page's own, and a
// fallback (ts_rank_cd for BM25, no reranker) is named as one. Capabilities marks what the API behind the page serves.

const REPOSITORY = "https://github.com/joses2017smjh/MetaNavT";
// A /health answer in the shape main.py gives it.
const HEALTH = {
  status: "ok",
  n_nodes: 1234,
  embedding_provider: "huggingface",
  embedding_model: "BAAI/bge-small-en-v1.5",
  embedding_query_instruction: "Represent this question: ",
  devices: { embedding: "cpu", reranker: null },
  bm25_backend: "ts_rank_cd",
  dense_search: {
    depth: 50,
    settings: { "hnsw.ef_search": 100, "hnsw.iterative_scan": "strict_order" },
    pgvector: "0.8.0",
  },
  retrieval_mode: "sql",
  reranker_loaded: false,
  reranker: {
    configured: false,
    loaded: false,
    model: null,
    device: null,
    precision: "fp32",
    max_length: 512,
    depth: 20,
    error: null,
  },
  degraded: [],
  indexing: {
    indexed: false,
    chunker: "app/chunking (structure-aware)",
    manifest_at_start: {
      files: 987,
      chunks: 1234,
      pipelines: ["app.chunking/auto/v2|BAAI/bge-small-en-v1.5"],
    },
  },
  public_demo: true,
};
const LOADED = {
  configured: true,
  loaded: true,
  model: "BAAI/bge-reranker-v2-m3",
  device: "cuda:0",
  precision: "fp16",
  max_length: 512,
  depth: 20,
  error: null,
};

function region(name: string): HTMLElement {
  return screen.getByRole("region", { name });
}

function links(): [string | null, string | null][] {
  const bar = screen.getByRole("navigation", { name: "Sections" });
  return within(bar)
    .getAllByRole("link")
    .map((link) => [link.textContent, link.getAttribute("href")]);
}

// The value beside a label in a list of facts.
function fact(scope: HTMLElement, label: string): string | null {
  const term = within(scope).getByText(label);
  return term.nextElementSibling?.textContent ?? null;
}

function openStage(name: string): HTMLElement {
  const pipeline = region("From a question to ranked excerpts");
  const button = within(pipeline).getByRole("button", { name });
  fireEvent.click(button);
  assert.equal(button.getAttribute("aria-pressed"), "true");
  return pipeline;
}

async function demo(health: unknown): Promise<Call[]> {
  const calls = mockApi(() => json(health));
  render(<HomeSwitch />);
  await screen.findByLabelText("Question");
  return calls;
}

afterEach(reset);

describe("the page's sections", () => {
  it("links each section of the demo page, every target on the page", async () => {
    const calls = await demo(HEALTH);
    assert.deepEqual(links(), [
      ["MetaNaviT", "#top"],
      ["Search", "#search"],
      ["Dataset", "#dataset"],
      ["Pipeline", "#pipeline"],
      ["Capabilities", "#capabilities"],
      ["Source on GitHub", REPOSITORY],
    ]);
    for (const [, href] of links().slice(0, -1))
      assert.ok(document.getElementById(href!.slice(1)), href!);
    assert.deepEqual(
      calls.map((call) => `${call.method} ${call.path}`),
      ["GET /health"], // the sections read the answer the switch already has
    );
  });

  it("shows the index as /health reports it", async () => {
    await demo(HEALTH);
    const dataset = region("What this server has indexed");
    assert.equal(fact(dataset, "files in the index’s manifest"), "987");
    assert.equal(fact(dataset, "chunks in the index"), "1,234");
    assert.equal(fact(dataset, "Chunker"), "app/chunking (structure-aware)");
    assert.equal(
      fact(dataset, "Indexing pipeline"),
      "app.chunking/auto/v2|BAAI/bge-small-en-v1.5",
    );
    assert.equal(fact(dataset, "Embedding model"), "BAAI/bge-small-en-v1.5");
    assert.equal(fact(dataset, "Service status"), "ok");
    within(dataset).getByRole("list", {
      name: "What the demo’s fixture holds",
    });
  });

  it("names the degraded components beside the status", async () => {
    await demo({
      ...HEALTH,
      status: "degraded",
      degraded: [{ component: "rerank", error: "configured but not loaded" }],
    });
    assert.equal(
      fact(region("What this server has indexed"), "Service status"),
      "degraded: rerank (configured but not loaded)",
    );
  });

  it("says “not reported” for what /health does not say, and shows no number of its own", async () => {
    await demo({ status: "unavailable", public_demo: true }); // an API that is still starting
    const dataset = region("What this server has indexed");
    assert.equal(
      fact(dataset, "files in the index’s manifest (not reported)"),
      "—",
    );
    assert.equal(fact(dataset, "chunks in the index (not reported)"), "—");
    for (const label of ["Chunker", "Indexing pipeline", "Embedding model"])
      assert.equal(fact(dataset, label), "not reported");
    assert.equal(/\d/.test(dataset.textContent ?? ""), false);

    const pipeline = region("From a question to ranked excerpts");
    assert.equal(fact(pipeline, "Retrieval mode"), "not reported");
    assert.equal(fact(openStage("Lexical"), "Backend"), "not reported");
    const rerank = openStage("Rerank");
    assert.equal(fact(rerank, "Model"), "not reported");
    assert.equal(within(rerank).queryByText(/a fallback/) === null, true);
  });

  it("opens one stage at a time, each with the settings /health reports", async () => {
    await demo(HEALTH);
    const pipeline = region("From a question to ranked excerpts");
    assert.deepEqual(
      within(pipeline)
        .getAllByRole("button")
        .map((button) => [
          button.textContent,
          button.getAttribute("aria-pressed"),
        ]),
      [
        ["01Route", "true"],
        ["02Lexical", "false"],
        ["03Dense", "false"],
        ["04Fuse", "false"],
        ["05Rerank", "false"],
        ["06Cited answer", "false"],
      ],
    );
    within(pipeline).getByRole("heading", { name: "Route the question" });
    assert.equal(fact(pipeline, "Retrieval mode"), "sql");

    // ts_rank_cd is PostgreSQL's full-text rank: the page never calls it BM25
    assert.equal(
      fact(openStage("Lexical"), "Backend"),
      "ts_rank_cd (PostgreSQL full-text rank, not BM25)",
    );
    assert.equal(
      within(pipeline).queryByRole("heading", { name: "Route the question" }),
      null,
    );

    const dense = openStage("Dense");
    assert.equal(fact(dense, "Embeddings"), "huggingface");
    assert.equal(fact(dense, "Model"), "BAAI/bge-small-en-v1.5");
    assert.equal(fact(dense, "Query prefix"), "“Represent this question: ”");
    assert.equal(fact(dense, "Device"), "cpu");
    assert.equal(fact(dense, "pgvector"), "0.8.0");
    assert.equal(fact(dense, "List depth"), "50");
    assert.equal(fact(dense, "hnsw.ef_search"), "100");
    assert.equal(fact(dense, "hnsw.iterative_scan"), "strict_order");

    const answer = openStage("Cited answer");
    within(answer).getByText(
      /^Ask sends the 6 best excerpts to Claude, Anthropic’s model\. The reply is shown as an answer only when it cites them/,
    );
    within(answer).getByText(
      "Answers are not evaluated in this repository: no committed result measures their quality.",
    );
  });

  it("names the missing reranker as a fallback, and shows a loaded one's settings", async () => {
    await demo(HEALTH);
    const off = openStage("Rerank");
    within(off).getByText(
      "Off on this server: the hits keep the fused order of the lexical and dense lists (a fallback).",
    );
    assert.equal(fact(off, "Model"), "none");
    assert.equal(within(off).queryByText("Depth"), null); // a reranker that is off has no settings in use
    reset();

    await demo({ ...HEALTH, reranker_loaded: true, reranker: LOADED });
    const loaded = openStage("Rerank");
    assert.equal(fact(loaded, "Model"), "BAAI/bge-reranker-v2-m3");
    assert.equal(fact(loaded, "Depth"), "20");
    assert.equal(fact(loaded, "Max length"), "512");
    assert.equal(fact(loaded, "Precision"), "fp16");
    assert.equal(fact(loaded, "Device"), "cuda:0");
    assert.equal(within(loaded).queryByText(/a fallback/) === null, true);
    reset();

    await demo({
      ...HEALTH,
      reranker: { ...LOADED, loaded: false, error: "out of memory" },
    });
    within(openStage("Rerank")).getByText(
      "Configured on this server but not loaded (out of memory): the hits keep the fused order of the lexical and dense lists (a fallback).",
    );
  });

  it("marks the capabilities the API behind the page serves", async () => {
    await demo(HEALTH);
    const capabilities = () => region("What MetaNaviT does, and where");
    const served = () =>
      within(capabilities())
        .getAllByRole("listitem")
        .filter((card) => within(card).queryByText("(served here)") !== null)
        .map((card) => within(card).getByRole("heading").textContent);
    assert.deepEqual(served(), [
      "Hybrid retrieval",
      "Citations to the byte range",
      "Structure-aware chunking",
      "Guardrails for a public endpoint",
    ]);
    // no measurement on a card: those are in the README
    assert.equal(/\d\s*%|\d\.\d/.test(capabilities().textContent ?? ""), false);
    reset();

    mockApi((call: Call) => {
      if (call.path === "/health")
        return json({ ...HEALTH, public_demo: false });
      if (call.path === "/api/chat/config") return json({});
      return json([]);
    });
    render(<HomeSwitch />);
    await screen.findByText("No plans waiting for a decision.");
    assert.deepEqual(served(), [
      "Hybrid retrieval",
      "Structure-aware chunking",
      "Chat agents with tools",
      "Reviewed file changes",
    ]);
  });

  it("links the chat interface's sections, without the demo's fixture or its answer stage", async () => {
    mockApi((call: Call) => {
      if (call.path === "/health")
        return json({ ...HEALTH, public_demo: false });
      if (call.path === "/api/chat/config") return json({});
      return json([]);
    });
    render(<HomeSwitch />);
    await screen.findByText("No plans waiting for a decision.");
    assert.deepEqual(links(), [
      ["MetaNaviT", "#top"],
      ["Chat", "#chat"],
      ["Files", "#files"],
      ["Dataset", "#dataset"],
      ["Pipeline", "#pipeline"],
      ["Capabilities", "#capabilities"],
      ["Source on GitHub", REPOSITORY],
    ]);
    for (const [, href] of links().slice(0, -1))
      assert.ok(document.getElementById(href!.slice(1)), href!);
    const dataset = region("What this server has indexed");
    assert.equal(fact(dataset, "files in the index’s manifest"), "987");
    assert.equal(within(dataset).queryByRole("list"), null);
    const pipeline = region("From a question to ranked excerpts");
    assert.equal(
      within(pipeline).queryByRole("button", { name: "Cited answer" }),
      null,
    );
  });

  it("shows neither Dataset nor Pipeline, nor their links, when /health is not JSON", async () => {
    mockApi((call: Call) => {
      if (call.path === "/health")
        return new Response("<!doctype html><title>404</title>", {
          status: 404,
          headers: { "content-type": "text/html" },
        });
      if (call.path === "/api/chat/config") return json({});
      return json([]);
    });
    render(<HomeSwitch />);
    await screen.findByText("No plans waiting for a decision.");
    assert.deepEqual(
      links().map(([label]) => label),
      ["MetaNaviT", "Chat", "Files", "Capabilities", "Source on GitHub"],
    );
    assert.equal(
      screen.queryByRole("region", { name: "What this server has indexed" }),
      null,
    );
    assert.equal(document.getElementById("pipeline"), null);
  });
});

describe("readSystem", () => {
  it("takes a field only in the type main.py gives it", () => {
    const system = readSystem({
      status: 200,
      n_nodes: "71",
      embedding_model: null,
      bm25_backend: ["ts_rank_cd"],
      dense_search: { depth: "50", settings: { a: 1, b: "x", c: null, d: {} } },
      reranker: "none",
      degraded: [{ component: "rerank" }, "database", null],
      indexing: {
        manifest_at_start: { files: Number.NaN, pipelines: [1, "p"] },
      },
    });
    assert.ok(system);
    assert.equal(system.status, null);
    assert.equal(system.rows, null);
    assert.equal(system.files, null);
    assert.equal(system.embeddingModel, null);
    assert.equal(system.lexicalBackend, null);
    assert.equal(system.denseDepth, null);
    assert.deepEqual(system.denseSettings, [
      ["a", "1"],
      ["b", "x"],
    ]);
    assert.deepEqual(system.pipelines, ["p"]);
    assert.deepEqual(system.degraded, []);
    assert.equal(system.reranker.loaded, null);
  });

  it("reads the reranker's state from reranker_loaded when there is no reranker entry", () => {
    assert.equal(
      readSystem({ reranker_loaded: false })?.reranker.loaded,
      false,
    );
    assert.equal(
      readSystem({ reranker_loaded: false, reranker: { loaded: true } })
        ?.reranker.loaded,
      true,
    );
  });

  it("is null for an answer that is not a JSON object", () => {
    for (const body of [null, "ok", 3, [1]])
      assert.equal(readSystem(body), null);
  });
});

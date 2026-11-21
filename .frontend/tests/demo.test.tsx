// organize-imports-ignore: "./dom" must stay the first import (it sets up the browser globals the others read)
import "./dom";
import DemoSection from "@/app/components/demo/demo-section";
import HomeSwitch from "@/app/components/home-switch";
import { fireEvent, render, screen, within } from "@testing-library/react";
import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";
import { json, mockApi, reset, type Call } from "./mock-api";

// The home page against a mocked API. The switch reads GET /health's public_demo: true shows the demo page, anything
// else the chat interface, as before. On the demo page, Search and Ask post to the public routes (app/api/public.py)
// with their k, an answer's citations link to the excerpts they cite, and every way the answer route ends without an
// answer, or a request fails, is said in plain words. Every request stays on the page's own origin.

const QUESTION = "What learning rate did run 47 use?";
const REPOSITORY = "https://github.com/joses2017smjh/MetaNavT";
const HIT = {
  rank: 1,
  chunk_id: "n1",
  path: "configs/run_047.yaml",
  start_byte: 40,
  end_byte: 59,
  score: 0.98,
  scores: { bm25: 0.5, dense: 0.81, path: null, rrf: 0.0325, rerank: 0.98 },
  text: "learning_rate: 3e-4",
};
const LONG_HIT = {
  rank: 2,
  chunk_id: "n2",
  path: "logs/run_047.out",
  start_byte: null,
  end_byte: null,
  score: 0.0164,
  scores: { bm25: null, dense: null, path: null, rrf: 0.0164, rerank: null },
  text: "epoch 3 val_rmse=0.0549 ".repeat(20),
};
const RETRIEVED = {
  query: QUESTION,
  k: 8,
  route: "semantic",
  retrieval_mode: "sql",
  bm25_backend: "ts_rank_cd",
  embedding_provider: "huggingface",
  reranker_loaded: true,
  reranker: {
    configured: true,
    loaded: true,
    model: "BAAI/bge-reranker-v2-m3",
  },
  degraded: [],
  staleness: { enabled: true, applied: true, dropped: 0 },
  counts: { returned: 2 },
  latency_ms: { total: 41.6 },
  hits: [HIT, LONG_HIT],
};
const CITED = {
  chunk_id: "n1",
  path: "configs/run_047.yaml",
  cited_text: "learning_rate: 3e-4",
  start_char: 0,
  end_char: 19,
  start_byte: 40,
  end_byte: 59,
};
const CITED_LOG = {
  chunk_id: "n2",
  path: "logs/run_047.out",
  cited_text: "epoch 3",
  start_char: 0,
  end_char: 7,
  start_byte: null,
  end_byte: null,
};

// An answer route response (app/api/routers/answer.py): no answer unless the fields say otherwise.
function answered(fields: Record<string, unknown> = {}) {
  return {
    question: QUESTION,
    answer: null,
    reason: null,
    uncited_reply: null,
    citations: [],
    model: null,
    fallback_used: false,
    cached: false,
    cost_usd_estimate: null,
    evaluated: false,
    hits: [HIT, LONG_HIT],
    ...fields,
  };
}

function send(action: "Search" | "Ask", question = QUESTION): void {
  fireEvent.change(screen.getByLabelText("Question"), {
    target: { value: question },
  });
  fireEvent.click(screen.getByRole("button", { name: action }));
}

function element(id: string): HTMLElement {
  const found = document.getElementById(id);
  assert.ok(found, `#${id}`);
  return found;
}

afterEach(reset);

describe("the home page", () => {
  it("shows the demo page when /health reports public_demo, and asks for nothing else", async () => {
    const calls = mockApi(() => json({ status: "ok", public_demo: true }));
    render(<HomeSwitch />);
    assert.equal(
      screen.getByRole("status").textContent,
      "Connecting to the API…",
    );

    await screen.findByLabelText("Question");
    assert.equal(screen.queryByPlaceholderText("Type a message"), null);
    assert.equal(screen.queryByLabelText("API key"), null);
    assert.deepEqual(
      calls.map((call) => `${call.method} ${call.path}`),
      ["GET /health"], // no starter questions, no plans list: the public API serves neither
    );
  });

  it("reads the flag from the 503 of an API that is still starting", async () => {
    mockApi(() => json({ status: "unavailable", public_demo: true }, 503));
    render(<HomeSwitch />);
    await screen.findByLabelText("Question");
  });

  const chat: [string, () => Response][] = [
    ["/health says public_demo is false", () => json({ public_demo: false })],
    ["/health has no public_demo field", () => json({ status: "ok" })],
    [
      "/health is not JSON (a development server's 404 page)",
      () =>
        new Response("<!doctype html><title>404</title>", {
          status: 404,
          headers: { "content-type": "text/html" },
        }),
    ],
    [
      "/health cannot be reached",
      () => {
        throw new TypeError("fetch failed");
      },
    ],
  ];
  for (const [when, health] of chat)
    it(`shows the chat interface, as before, when ${when}`, async () => {
      const calls = mockApi((call: Call) => {
        if (call.path === "/health") return health();
        if (call.path === "/api/chat/config")
          return json({ starterQuestions: ["What is in the corpus?"] });
        if (call.path === "/api/plans/") return json([]);
        return json({ detail: "not mocked" }, 404);
      });
      render(<HomeSwitch />);

      await screen.findByText("What is in the corpus?");
      await screen.findByText("No plans waiting for a decision.");
      screen.getByPlaceholderText("Type a message");
      screen.getByLabelText("API key");
      assert.equal(screen.queryByLabelText("Question"), null);
      assert.equal(calls[0].path, "/health");
    });
});

describe("the demo page", () => {
  it("searches: POST /api/retrieve/ with k 8, listing each hit's path, byte range, scores and text", async () => {
    const calls = mockApi(() => json(RETRIEVED));
    render(<DemoSection />);
    const box = screen.getByLabelText("Question") as HTMLInputElement;
    assert.equal(box.maxLength, 500);
    const search = screen.getByRole("button", { name: "Search" });
    assert.equal((search as HTMLButtonElement).disabled, true); // nothing typed yet

    send("Search");
    const heading = await screen.findByRole("heading", {
      name: `2 excerpts for “${QUESTION}”`,
    });
    assert.equal(document.activeElement, heading);
    assert.deepEqual(
      calls.map((call) => `${call.method} ${call.path}`),
      ["POST /api/retrieve/"],
    );
    assert.deepEqual(calls[0].body, { query: QUESTION, k: 8 });

    const first = element("demo-hit-1");
    within(first).getByText("configs/run_047.yaml");
    within(first).getByText("bytes 40–59");
    within(first).getByText(
      "score 0.980 · lexical 0.500 · dense 0.810 · fused (RRF) 0.0325 · rerank 0.980",
    );
    within(first).getByText("learning_rate: 3e-4");
    const second = element("demo-hit-2");
    within(second).getByText("byte range not recorded");
    within(second).getByText("score 0.0164 · fused (RRF) 0.0164");
    within(second).getByText("The whole excerpt"); // a long excerpt: a snippet, and the rest on demand
    screen.getByText(
      /lexical ts_rank_cd \(PostgreSQL full-text rank, not BM25\) · embeddings huggingface · reranker BAAI\/bge-reranker-v2-m3 · 42 ms/,
    );
    assert.equal(screen.queryByText(/Reranker: off on this server/), null);
  });

  it("says when the server runs without the reranker, and links its measured gain", async () => {
    mockApi(() =>
      json({
        ...RETRIEVED,
        reranker_loaded: false,
        reranker: { configured: false, loaded: false, model: null },
      }),
    );
    render(<DemoSection />);
    send("Search");

    const note = await screen.findByText(/^Reranker: off on this server/);
    assert.equal(
      within(note).getByRole("link").getAttribute("href"),
      `${REPOSITORY}#bench-neural`,
    );
    screen.getByText(/ · reranker off · /);
  });

  it("asks: POST /api/answer/ with k 6, numbering each cited passage, linked to its excerpt", async () => {
    const calls = mockApi(() =>
      json(
        answered({
          answer: "Run 47 used a learning rate of 3e-4.",
          citations: [CITED, CITED_LOG, CITED],
          model: "claude-haiku-5-5",
        }),
      ),
    );
    render(<DemoSection />);
    const note = screen.getByRole("complementary", {
      name: "About the answers",
    });
    assert.match(
      note.textContent ?? "",
      /^Answers come from Claude, Anthropic’s model, and are not evaluated in this repository/,
    );

    send("Ask");
    const heading = await screen.findByRole("heading", { name: "Answer" });
    assert.equal(document.activeElement, heading);
    assert.deepEqual(
      calls.map((call) => `${call.method} ${call.path}`),
      ["POST /api/answer/"],
    );
    assert.deepEqual(calls[0].body, { question: QUESTION, k: 6 });
    screen.getByText("Run 47 used a learning rate of 3e-4.", { exact: false });

    const markers = screen.getAllByRole("link", { name: /^Citation \d+$/ });
    assert.deepEqual(
      markers.map((marker) => [
        marker.textContent,
        marker.getAttribute("href"),
      ]),
      [
        ["[1]", "#demo-citation-1"],
        ["[2]", "#demo-citation-2"],
      ], // the passage cited twice has one number
    );
    const first = element("demo-citation-1");
    within(first).getByText("configs/run_047.yaml");
    within(first).getByText("learning_rate: 3e-4");
    within(first).getByText("bytes 40–59");
    assert.equal(
      within(first).getByRole("link").getAttribute("href"),
      "#demo-hit-1",
    );
    within(element("demo-citation-2")).getByText("byte range not recorded");

    // the model id is the response's, in the answer and in the note about answers
    assert.equal(screen.getAllByText("claude-haiku-5-5").length, 2);
    assert.match(
      note.textContent ?? "",
      /\(the last one from claude-haiku-5-5, as the API reported it\), and are not evaluated in this repository: no committed result measures their quality\. Retrieval is the benchmarked part/,
    );
    assert.equal(
      within(note)
        .getByRole("link", { name: "benchmark results in the README" })
        .getAttribute("href"),
      `${REPOSITORY}#generated-benchmark-evidence`,
    );
    assert.equal(
      within(note)
        .getByRole("link", { name: "source code" })
        .getAttribute("href"),
      REPOSITORY,
    );
    screen.getByRole("heading", { name: "Retrieved excerpts" });
    within(element("demo-hit-2")).getByText("logs/run_047.out");
  });

  const reasons: [string, RegExp][] = [
    [
      "answers off",
      /^Answers are off on this server: it has no Anthropic API key\./,
    ],
    [
      "daily budget spent",
      /^Today's answer budget is spent, so Claude was not asked\. Answers come back after midnight UTC/,
    ],
    [
      "limits unavailable",
      /^Answers are paused: the server cannot reach the store that keeps its quotas and its budget/,
    ],
    [
      "answer spend limit reached",
      /^Answers are paused: the demo's Anthropic account has reached its spend limit or run out of credit, so Claude did not answer\. Search still works/,
    ],
    [
      "refused (cyber)",
      /^Claude declined to answer this question \(category: cyber\)\.$/,
    ],
    ["refused (no category)", /^Claude declined to answer this question\.$/],
    [
      "upstream timeout",
      /^Claude's API did not answer in time\. Try again in a minute; search still works\.$/,
    ],
    [
      "upstream rate limited",
      /^Claude's API is turning this server's requests away for now \(a rate limit, or the account's monthly spend cap\)\./,
    ],
    [
      "upstream error 529",
      /^Claude's API returned an error \(upstream error 529\)\. Try again in a minute/,
    ],
    [
      "cut at max_tokens",
      /^Claude's reply reached its length limit before it ended, so it is not shown\.$/,
    ],
    [
      "nothing retrieved",
      /^Nothing was retrieved for this question, so Claude was not asked\.$/,
    ],
    [
      "stopped: pause_turn",
      /^Claude's reply stopped early \(pause_turn\), so it is not shown\.$/,
    ],
    [
      "a reason this page does not know",
      /^No answer \(a reason this page does not know\)\.$/,
    ],
  ];
  for (const [reason, said] of reasons)
    it(`says in plain words why there is no answer: ${reason}`, async () => {
      mockApi(() => json(answered({ reason })));
      render(<DemoSection />);
      send("Ask");

      await screen.findByRole("heading", { name: "No answer" });
      screen.getByText(said);
      assert.equal(screen.queryByRole("link", { name: /^Citation/ }), null);
      assert.equal(screen.queryByRole("figure"), null);
      within(element("demo-hit-1")).getByText("configs/run_047.yaml"); // the hits come back on every path
    });

  it("shows an uncited reply apart, labelled unverified", async () => {
    mockApi(() =>
      json(
        answered({
          reason: "uncited",
          uncited_reply: "It was probably 3e-4.",
          model: "claude-haiku-5-5",
        }),
      ),
    );
    render(<DemoSection />);
    send("Ask");

    await screen.findByRole("heading", { name: "No answer" });
    screen.getByText(
      /^Claude replied without citing any of the excerpts, so the reply is not shown as an answer\.$/,
    );
    const reply = screen.getByRole("figure");
    within(reply).getByText("Unverified reply: no citation supports it");
    within(reply).getByText("It was probably 3e-4.");
    assert.equal(screen.queryByRole("link", { name: /^Citation/ }), null);
  });

  it("shows a 429's Retry-After", async () => {
    mockApi(
      () =>
        new Response(
          JSON.stringify({ detail: "answer quota reached for this caller" }),
          {
            status: 429,
            headers: {
              "content-type": "application/json",
              "retry-after": "42",
            },
          },
        ),
    );
    render(<DemoSection />);
    send("Ask");

    const alert = await screen.findByRole("alert");
    assert.equal(
      alert.textContent,
      "Too many requests: this demo limits how often each visitor can ask (the API says: answer quota reached for this caller). Try again in 42 seconds (Retry-After: 42).",
    );
    assert.equal(
      document.activeElement,
      screen.getByRole("heading", { name: "No answer" }),
    );
  });

  it("words a long Retry-After in hours and minutes", async () => {
    mockApi(
      () =>
        new Response("", { status: 429, headers: { "retry-after": "3700" } }),
    );
    render(<DemoSection />);
    send("Search");

    const alert = await screen.findByRole("alert");
    assert.match(
      alert.textContent ?? "",
      /Try again in 1 hour 2 minutes \(Retry-After: 3700\)\.$/,
    );
    screen.getByRole("heading", { name: "Search failed" });
  });

  const refusedInput: [number, unknown][] = [
    [413, { detail: "request body too large" }],
    [422, { detail: [{ type: "string_too_long", loc: ["body", "question"] }] }],
  ];
  for (const [status, body] of refusedInput)
    it(`shows the input limits for a ${status}`, async () => {
      mockApi(() => json(body, status));
      render(<DemoSection />);
      send("Ask");

      const alert = await screen.findByRole("alert");
      assert.match(
        alert.textContent ?? "",
        /: a question has 1 to 500 characters, not only spaces\.$/,
      );
    });

  it("says when the API cannot be reached", async () => {
    mockApi(() => {
      throw new TypeError("fetch failed");
    });
    render(<DemoSection />);
    send("Search");

    const alert = await screen.findByRole("alert");
    assert.match(alert.textContent ?? "", /^The API could not be reached\./);
  });

  it("fills the question box from an example", () => {
    render(<DemoSection />);
    fireEvent.click(
      screen.getByRole("button", { name: "What encoder did run 40 use?" }),
    );
    assert.equal(
      (screen.getByLabelText("Question") as HTMLInputElement).value,
      "What encoder did run 40 use?",
    );
  });

  it("stays on the page's own origin, and shows what the API returns as text, never as HTML", async () => {
    const calls = mockApi((call: Call) => {
      if (call.path === "/health")
        return json({ status: "ok", public_demo: true });
      if (call.path === "/api/retrieve/")
        return json({
          ...RETRIEVED,
          hits: [{ ...HIT, text: "<img src=x onerror=alert(1)>" }],
        });
      return json(
        answered({
          answer: "<b>3e-4</b>",
          citations: [{ ...CITED, cited_text: "<script>alert(1)</script>" }],
          model: "claude-haiku-5-5",
        }),
      );
    });
    render(<HomeSwitch />);
    await screen.findByLabelText("Question");

    send("Search");
    await screen.findByRole("heading", { name: /^1 excerpt for/ });
    screen.getByText("<img src=x onerror=alert(1)>");
    assert.equal(document.querySelector("img"), null);

    send("Ask");
    await screen.findByRole("heading", { name: "Answer" });
    screen.getByText("<b>3e-4</b>", { exact: false });
    screen.getByText("<script>alert(1)</script>");
    assert.equal(document.querySelector("b, script"), null);

    assert.deepEqual(
      calls.map((call) => `${call.method} ${call.path}`),
      ["GET /health", "POST /api/retrieve/", "POST /api/answer/"],
    );
    assert.ok(
      calls.every((call) => call.url.origin === window.location.origin),
    );
  });
});

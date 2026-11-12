// organize-imports-ignore: "./dom" must stay the first import (it sets up the browser globals the others read)
import "./dom";
import ApiKeyField from "@/app/components/api-key-field";
import ChatSection from "@/app/components/chat-section";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";
import { json, KEY_SLOT, mockApi, reset, type Call } from "./mock-api";

// The chat section as the page renders it, against a mocked API: the starter questions (GET /api/chat/config), a
// message (POST /api/chat, answered in the AI SDK's data stream format) and a document upload (POST
// /api/chat/upload) each carry the key the user entered.

const KEY = "test-key-for-the-chat";

function chatApi(): Call[] {
  return mockApi((call) => {
    if (call.method === "GET" && call.path === "/api/chat/config")
      return json({ starterQuestions: ["What is in the corpus?"] });
    if (call.method === "POST" && call.path === "/api/chat")
      return new Response('0:"The answer."\n', {
        headers: { "content-type": "text/plain; charset=utf-8" },
      });
    if (call.method === "POST" && call.path === "/api/chat/upload")
      return json({
        id: "doc-1",
        name: "notes.txt",
        size: 5,
        type: "txt",
        url: "/api/files/data/uploads/notes.txt",
      });
    return json({ detail: "not mocked" }, 404);
  });
}

afterEach(reset);

describe("chat section", () => {
  it("sends a message, and the config and chat requests carry the key", async () => {
    window.sessionStorage.setItem(KEY_SLOT, KEY);
    const calls = chatApi();
    render(<ChatSection />);

    await screen.findByText("What is in the corpus?");
    fireEvent.change(screen.getByPlaceholderText("Type a message"), {
      target: { value: "hello" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));
    await screen.findByText("The answer.");

    assert.deepEqual(
      calls.map((call) => `${call.method} ${call.path}`),
      ["GET /api/chat/config", "POST /api/chat"],
    );
    assert.deepEqual(
      calls.map((call) => call.headers.get("X-API-Key")),
      [KEY, KEY],
    );
    const body = calls[1].body as { messages: { content: string }[] };
    assert.equal(body.messages.at(-1)?.content, "hello");
  });

  it("uploads a document with the key", async () => {
    window.sessionStorage.setItem(KEY_SLOT, KEY);
    const calls = chatApi();
    render(<ChatSection />);

    const input = document.getElementById("fileInput") as HTMLInputElement;
    fireEvent.change(input, {
      target: {
        files: [
          new window.File(["hello"], "notes.txt", { type: "text/plain" }),
        ],
      },
    });
    await waitFor(() =>
      assert.ok(calls.some((call) => call.path === "/api/chat/upload")),
    );
    const upload = calls.find((call) => call.path === "/api/chat/upload")!;
    assert.equal(upload.headers.get("X-API-Key"), KEY);
    assert.deepEqual(upload.body, {
      base64: "data:text/plain;base64,aGVsbG8=",
      name: "notes.txt",
    });
  });

  it("sends no X-API-Key header when no key was entered", async () => {
    const calls = chatApi();
    render(<ChatSection />);
    await screen.findByText("What is in the corpus?");
    assert.equal(calls[0].headers.has("X-API-Key"), false);
  });

  it("asks for the starter questions again once a key is entered", async () => {
    const calls = mockApi((call) =>
      call.headers.get("X-API-Key") === KEY
        ? json({ starterQuestions: ["What is in the corpus?"] })
        : json({ detail: "missing or invalid API key" }, 401),
    );
    render(
      <>
        <ApiKeyField />
        <ChatSection />
      </>,
    );
    await waitFor(() => assert.equal(calls.length, 1));
    assert.equal(screen.queryByText("What is in the corpus?"), null);

    fireEvent.change(screen.getByLabelText("API key"), {
      target: { value: KEY },
    });
    fireEvent.click(screen.getByRole("button", { name: "Use key" }));
    await screen.findByText("What is in the corpus?");
    assert.deepEqual(
      calls.map((call) => `${call.path} ${call.headers.get("X-API-Key")}`),
      ["/api/chat/config null", `/api/chat/config ${KEY}`],
    );
  });
});

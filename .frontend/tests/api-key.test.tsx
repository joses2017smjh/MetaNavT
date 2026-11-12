// organize-imports-ignore: "./dom" must stay the first import (it sets up the browser globals the others read)
import "./dom";
import ApiKeyField from "@/app/components/api-key-field";
import PlansPanel, { type Plan } from "@/app/components/plans-panel";
import { apiFetch, setApiKey } from "@/app/components/ui/chat/hooks/api-key";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";
import { json, KEY_SLOT, mockApi, reset } from "./mock-api";

// The key is typed into the page at run time and kept in this tab's sessionStorage; nothing in the build holds it.

const KEY = "test-key-typed-in";
const REFUSED =
  "missing or invalid API key (send X-API-Key or Authorization: Bearer)";
const pending: Plan = {
  plan_id: "p1",
  src: "notes/a.txt",
  dst: "archive/a.txt",
  status: "pending_approval",
  created_at: 1,
};

afterEach(reset);

describe("the API key field", () => {
  it("keeps the key in sessionStorage only, and the plans list is fetched again with it", async () => {
    const calls = mockApi((call) =>
      call.headers.get("X-API-Key") === KEY
        ? json([pending])
        : json({ detail: REFUSED }, 401),
    );
    render(
      <>
        <ApiKeyField />
        <PlansPanel />
      </>,
    );
    assert.equal((await screen.findByRole("alert")).textContent, REFUSED);

    const field = screen.getByLabelText("API key") as HTMLInputElement;
    assert.equal(field.type, "password");
    fireEvent.change(field, { target: { value: ` ${KEY} ` } });
    fireEvent.click(screen.getByRole("button", { name: "Use key" }));

    await screen.findByLabelText("Approval token for notes/a.txt");
    assert.equal(screen.queryByRole("alert"), null);
    assert.deepEqual(
      calls.map((call) => call.headers.get("X-API-Key")),
      [null, KEY],
    );
    assert.equal(window.sessionStorage.getItem(KEY_SLOT), KEY);
    assert.equal(window.localStorage.length, 0);
    assert.equal(field.value, "");
    assert.ok(!document.body.innerHTML.includes(KEY));
  });

  it("forgets the key, and the list is fetched again without it", async () => {
    setApiKey(KEY);
    const calls = mockApi(() => json([]));
    render(
      <>
        <ApiKeyField />
        <PlansPanel />
      </>,
    );
    await screen.findByText("No plans waiting for a decision.");
    assert.equal(calls.length, 1);
    fireEvent.click(screen.getByRole("button", { name: "Forget key" }));

    await waitFor(() => assert.equal(calls.length, 2));
    assert.deepEqual(
      calls.map((call) => call.headers.get("X-API-Key")),
      [KEY, null],
    );
    assert.equal(window.sessionStorage.getItem(KEY_SLOT), null);
    assert.equal(screen.queryByRole("button", { name: "Forget key" }), null);
  });
});

describe("apiFetch", () => {
  it("adds the key to /api requests on the page's origin only, keeping the caller's headers", async () => {
    setApiKey(KEY);
    const calls = mockApi(() => json({}));
    await apiFetch("/api/chat/config");
    await apiFetch("/api/plans/", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{}",
    });
    await apiFetch("/health");
    await apiFetch("https://elsewhere.example/api/plans/");

    assert.deepEqual(
      calls.map((call) => call.headers.get("X-API-Key")),
      [KEY, KEY, null, null],
    );
    assert.equal(calls[1].headers.get("content-type"), "application/json");
  });

  it("sends no key when none is set", async () => {
    const calls = mockApi(() => json({}));
    await apiFetch("/api/chat/config");
    assert.equal(calls[0].headers.has("X-API-Key"), false);
  });
});

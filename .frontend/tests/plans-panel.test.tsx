// organize-imports-ignore: "./dom" must stay the first import (it sets up the browser globals the others read)
import "./dom";
import PlansPanel, { type Plan } from "@/app/components/plans-panel";
import { fireEvent, render, screen } from "@testing-library/react";
import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";
import { json, KEY_SLOT, mockApi, reset } from "./mock-api";

// The plans panel against a mocked /api/plans: a pending plan is listed, approved with the operator's token, and
// listed again as applied by the identity the server took from the key.

const KEY = "test-key-for-the-panel";
const pending: Plan = {
  plan_id: "p1",
  src: "notes/a.txt",
  dst: "archive/a.txt",
  status: "pending_approval",
  created_at: 1,
  review: { moves: [{ src: "notes/a.txt", dst: "archive/a.txt" }] },
};
const applied: Plan = {
  ...pending,
  status: "applied",
  decided_at: 2,
  actor: "alice",
};

afterEach(reset);

describe("plans panel", () => {
  it("approves a plan against a mocked API, every request carrying the key the user entered", async () => {
    window.sessionStorage.setItem(KEY_SLOT, KEY);
    let approved = false;
    const calls = mockApi((call) => {
      if (call.method === "GET" && call.path === "/api/plans/")
        return json(approved ? [applied] : [pending]);
      if (call.method === "POST" && call.path === "/api/plans/p1/approve") {
        approved = true;
        return json(applied);
      }
      return json({ detail: "not mocked" }, 404);
    });

    render(<PlansPanel />);
    fireEvent.change(
      await screen.findByLabelText("Approval token for notes/a.txt"),
      { target: { value: "signed-capability" } },
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Apply reviewed change" }),
    );

    await screen.findByText("by alice");
    screen.getByText("Pending (0)");
    assert.deepEqual(
      calls.map((call) => `${call.method} ${call.path}`),
      ["GET /api/plans/", "POST /api/plans/p1/approve", "GET /api/plans/"],
    );
    assert.deepEqual(
      calls.map((call) => call.headers.get("X-API-Key")),
      [KEY, KEY, KEY],
    );
    assert.deepEqual(calls[1].body, {
      actor: "web-ui",
      note: null,
      approval_token: "signed-capability",
    });
    assert.ok(!document.body.innerHTML.includes(KEY));
  });

  it("sends no X-API-Key header when no key was entered", async () => {
    const calls = mockApi(() => json([pending]));
    render(<PlansPanel />);
    await screen.findByLabelText("Approval token for notes/a.txt");
    assert.equal(calls.length, 1);
    assert.equal(calls[0].headers.has("X-API-Key"), false);
  });
});

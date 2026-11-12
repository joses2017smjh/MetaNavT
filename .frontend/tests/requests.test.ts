import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, it } from "node:test";

// Every request the page's own code makes goes through apiFetch (app/components/ui/chat/hooks/api-key.ts), which adds
// the key the user entered. A new bare fetch, or a library hook that fetches by itself, would send none.

const APP = path.resolve("app"); // `npm test` runs in .frontend
const HELPER = path.join("components", "ui", "chat", "hooks", "api-key.ts");

function sources(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return sources(full);
    return /\.(ts|tsx)$/.test(entry.name) ? [full] : [];
  });
}

const files = sources(APP).map((file) => ({
  name: path.relative(APP, file),
  text: readFileSync(file, "utf8"),
}));

describe("the page's requests", () => {
  it("read the app's sources", () => {
    assert.ok(files.some((file) => file.name === HELPER));
    assert.ok(files.length > 20, `${files.length} files under ${APP}`);
  });

  it("call fetch only inside apiFetch", () => {
    const bare = files
      .filter((file) => file.name !== HELPER && /\bfetch\s*\(/.test(file.text))
      .map((file) => file.name);
    assert.deepEqual(bare, []);
  });

  it("do not use chat-ui's upload hook, which sends no key", () => {
    const users = files
      .filter((file) =>
        /import\s*{[^}]*\buseFile\b[^}]*}\s*from\s*"@llamaindex\/chat-ui"/.test(
          file.text,
        ),
      )
      .map((file) => file.name);
    assert.deepEqual(users, []);
  });

  it("give the chat hook apiFetch", () => {
    const chat = files.find(
      (file) => file.name === path.join("components", "chat-section.tsx"),
    );
    assert.match(chat?.text ?? "", /useChat\(\{[^]*?\bfetch:\s*apiFetch\b/);
  });

  it("read no build-time variable but the API's address, so no key is baked into the export", () => {
    const variables = new Set(
      files.flatMap((file) =>
        Array.from(file.text.matchAll(/process\.env\.(\w+)/g), (m) => m[1]),
      ),
    );
    assert.deepEqual([...variables], ["NEXT_PUBLIC_CHAT_API"]);
  });
});

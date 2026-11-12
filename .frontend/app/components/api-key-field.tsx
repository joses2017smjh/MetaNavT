"use client";

import { FormEvent, useState } from "react";
import { Button } from "./ui/button";
import { setApiKey, useApiKeySet } from "./ui/chat/hooks/api-key";
import { Input } from "./ui/input";

// Where the user gives the page the API key, when the server requires one. It is kept in this tab's sessionStorage
// (hooks/api-key.ts), never in the build, and the field never shows a stored key.
export default function ApiKeyField() {
  const keySet = useApiKeySet();
  const [draft, setDraft] = useState("");

  const save = (event: FormEvent) => {
    event.preventDefault();
    if (!draft.trim()) return;
    setApiKey(draft);
    setDraft("");
  };

  return (
    <form onSubmit={save} className="flex flex-wrap items-center gap-2 text-sm">
      <Input
        type="password"
        autoComplete="off"
        aria-label="API key"
        placeholder={
          keySet
            ? "Replace the API key"
            : "API key (only if the server requires one)"
        }
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        className="w-72"
      />
      <Button type="submit" size="sm" disabled={!draft.trim()}>
        Use key
      </Button>
      {keySet && (
        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={() => setApiKey(null)}
        >
          Forget key
        </Button>
      )}
      <span role="status" className="text-muted-foreground">
        {keySet
          ? "This tab sends the key with its requests to the API."
          : "No key: requests go without one."}
      </span>
    </form>
  );
}

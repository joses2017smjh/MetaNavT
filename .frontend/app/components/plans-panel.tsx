"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "./ui/button";
import { apiFetch, useApiKeyVersion } from "./ui/chat/hooks/api-key";
import { useClientConfig } from "./ui/chat/hooks/use-config";
import { Input } from "./ui/input";

// Human-in-the-loop file plans. A plan is a proposed move; nothing on disk changes
// until a separately issued, reviewed approval token is verified by the server.
// The UI cannot issue tokens and never stores them in browser persistence.
// Requests go through apiFetch, which adds the API key the user entered (ApiKeyField); with keys
// on, the server records the key's name as the actor, whatever the actor field says.

export interface Plan {
  plan_id: string;
  src: string;
  dst: string;
  status: "pending_approval" | "approved" | "applied" | "rejected";
  created_at: number;
  decided_at?: number | null;
  actor?: string | null;
  note?: string | null;
  review?: Record<string, unknown> | null;
}

async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    return typeof body.detail === "string" ? body.detail : JSON.stringify(body);
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

export default function PlansPanel() {
  const { backend } = useClientConfig();
  const base = `${backend ?? ""}/api/plans`;
  const [plans, setPlans] = useState<Plan[]>([]);
  const [src, setSrc] = useState("");
  const [dst, setDst] = useState("");
  const [actor, setActor] = useState("web-ui");
  const [tokens, setTokens] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const keyVersion = useApiKeyVersion();

  const refresh = useCallback(async () => {
    try {
      const response = await apiFetch(`${base}/`);
      if (!response.ok) throw new Error(await readError(response));
      setPlans((await response.json()) as Plan[]);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [base]);

  // again when the key changes: a list refused without the key is fetched with it
  useEffect(() => {
    void refresh();
  }, [refresh, keyVersion]);

  const post = useCallback(
    async (url: string, body: unknown) => {
      setBusy(url);
      try {
        const response = await apiFetch(url, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!response.ok) throw new Error(await readError(response));
        setError(null);
        await refresh();
        return true;
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        return false;
      } finally {
        setBusy(null);
      }
    },
    [refresh],
  );

  const propose = () => post(`${base}/`, { src, dst });
  const decide = async (plan: Plan, action: "approve" | "reject") => {
    const applied = await post(`${base}/${plan.plan_id}/${action}`, {
      actor,
      note: null,
      ...(action === "approve" ? { approval_token: tokens[plan.plan_id] } : {}),
    });
    if (applied)
      setTokens((current) => {
        const next = { ...current };
        delete next[plan.plan_id];
        return next;
      });
  };
  const downloadReview = (plan: Plan) => {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(plan.review, null, 2)], {
        type: "application/json",
      }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = `review-${plan.plan_id}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const pending = plans.filter((p) => p.status === "pending_approval");
  const decided = plans.filter((p) => p.status !== "pending_approval");

  return (
    <section
      className="w-full space-y-5 rounded-card border border-white bg-white/90 p-5 text-sm shadow-card sm:p-8"
      aria-label="File plans"
    >
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="space-y-4">
          <p className="eyebrow">Files</p>
          <h2 className="display text-[clamp(2rem,4vw,3rem)] leading-[1.02]">
            Review file changes before applying
          </h2>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => void refresh()}
          disabled={busy !== null}
        >
          Refresh
        </Button>
      </div>
      <p className="max-w-2xl text-[15px] leading-relaxed text-muted-foreground">
        Review the manifest, then paste a short-lived approval token from your
        authorized operator. A changed file requires a new plan.
      </p>

      <div className="flex flex-wrap items-center gap-2">
        <Input
          placeholder="source path (relative to DATA_DIR)"
          value={src}
          onChange={(e) => setSrc(e.target.value)}
          className="min-w-[16rem] flex-1"
        />
        <Input
          placeholder="destination path"
          value={dst}
          onChange={(e) => setDst(e.target.value)}
          className="min-w-[16rem] flex-1"
        />
        <Input
          placeholder="your name (logged if the API has no keys)"
          value={actor}
          onChange={(e) => setActor(e.target.value)}
          className="w-full sm:w-72"
        />
        <Button
          onClick={() => void propose()}
          disabled={!src || !dst || busy !== null}
        >
          Propose move
        </Button>
      </div>

      {error && (
        <p
          role="alert"
          className="rounded-2xl border border-destructive/25 bg-destructive/[0.04] px-4 py-3 text-[15px] leading-relaxed text-destructive"
        >
          {error}
        </p>
      )}

      <h3 className="pt-2 font-serif text-2xl font-medium tracking-[-0.02em] text-brand-deep">
        Pending ({pending.length})
      </h3>
      {pending.length === 0 && (
        <p className="text-muted-foreground">
          No plans waiting for a decision.
        </p>
      )}
      <ul className="space-y-2">
        {pending.map((plan) => (
          <li
            key={plan.plan_id}
            className="flex flex-wrap items-center gap-2 rounded-2xl border border-black/[0.07] bg-white p-4 shadow-pill"
          >
            <code className="flex-1 break-all font-mono text-[13px] font-medium text-ink">
              {plan.src} → {plan.dst}
            </code>
            <details className="w-full">
              <summary className="w-fit cursor-pointer rounded text-muted-foreground underline-offset-4 hover:text-ink hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
                Review manifest
              </summary>
              <pre className="my-2 max-h-64 overflow-auto whitespace-pre-wrap rounded-xl bg-peach/70 px-3.5 py-3 font-mono text-xs leading-relaxed text-ink/85">
                {JSON.stringify(plan.review, null, 2)}
              </pre>
              <Button
                size="sm"
                variant="outline"
                onClick={() => downloadReview(plan)}
                disabled={!plan.review}
              >
                Download review
              </Button>
            </details>
            <Input
              type="password"
              autoComplete="off"
              aria-label={`Approval token for ${plan.src}`}
              placeholder="Operator approval token"
              value={tokens[plan.plan_id] ?? ""}
              onChange={(e) =>
                setTokens((current) => ({
                  ...current,
                  [plan.plan_id]: e.target.value,
                }))
              }
              className="min-w-[16rem] flex-1"
            />
            <Button
              size="sm"
              onClick={() => void decide(plan, "approve")}
              disabled={busy !== null || !tokens[plan.plan_id]?.trim()}
            >
              Apply reviewed change
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => void decide(plan, "reject")}
              disabled={busy !== null}
            >
              Reject
            </Button>
          </li>
        ))}
      </ul>

      {decided.length > 0 && (
        <details>
          <summary className="w-fit cursor-pointer rounded font-serif text-2xl font-medium tracking-[-0.02em] text-brand-deep focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
            Decided ({decided.length})
          </summary>
          <ul className="mt-2 space-y-1">
            {decided.map((plan) => (
              <li
                key={plan.plan_id}
                className="flex flex-wrap items-center gap-2"
              >
                <span className="rounded-full border border-black/10 bg-white px-2.5 py-1 text-[11px] uppercase leading-none tracking-[0.08em] text-muted-foreground shadow-pill">
                  {plan.status}
                </span>
                <code className="break-all font-mono text-[13px] text-ink">
                  {plan.src} → {plan.dst}
                </code>
                {plan.actor && (
                  <span className="text-muted-foreground">by {plan.actor}</span>
                )}
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}

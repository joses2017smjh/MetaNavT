"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { useClientConfig } from "./ui/chat/hooks/use-config";

// Human-in-the-loop file plans. A plan is a proposed move; nothing on disk changes
// until someone presses Approve here (POST /api/plans/{id}/approve), and every
// decision is written to Postgres before the move runs (app/api/routers/plans.py).

export interface Plan {
  plan_id: string;
  src: string;
  dst: string;
  status: "pending_approval" | "approved" | "applied" | "rejected";
  created_at: number;
  decided_at?: number | null;
  actor?: string | null;
  note?: string | null;
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
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const response = await fetch(`${base}/`);
      if (!response.ok) throw new Error(await readError(response));
      setPlans((await response.json()) as Plan[]);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [base]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const post = useCallback(
    async (url: string, body: unknown) => {
      setBusy(url);
      try {
        const response = await fetch(url, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!response.ok) throw new Error(await readError(response));
        setError(null);
        await refresh();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(null);
      }
    },
    [refresh],
  );

  const propose = () => post(`${base}/`, { src, dst });
  const decide = (plan: Plan, action: "approve" | "reject") =>
    post(`${base}/${plan.plan_id}/${action}`, { actor, note: null });

  const pending = plans.filter((p) => p.status === "pending_approval");
  const decided = plans.filter((p) => p.status !== "pending_approval");

  return (
    <section className="w-full rounded-xl bg-white/80 dark:bg-black/40 p-4 space-y-3 text-sm" aria-label="File plans">
      <div className="flex items-center justify-between">
        <h2 className="font-semibold">File plans: approve or reject before anything moves</h2>
        <Button variant="outline" size="sm" onClick={() => void refresh()} disabled={busy !== null}>
          Refresh
        </Button>
      </div>

      <div className="flex flex-wrap gap-2 items-center">
        <Input placeholder="source path (relative to DATA_DIR)" value={src} onChange={(e) => setSrc(e.target.value)} className="min-w-[16rem] flex-1" />
        <Input placeholder="destination path" value={dst} onChange={(e) => setDst(e.target.value)} className="min-w-[16rem] flex-1" />
        <Input placeholder="your name (logged)" value={actor} onChange={(e) => setActor(e.target.value)} className="w-40" />
        <Button size="sm" onClick={() => void propose()} disabled={!src || !dst || busy !== null}>
          Propose move
        </Button>
      </div>

      {error && (
        <p role="alert" className="text-red-600 dark:text-red-400">
          {error}
        </p>
      )}

      <h3 className="font-medium">Pending ({pending.length})</h3>
      {pending.length === 0 && <p className="text-muted-foreground">No plans waiting for a decision.</p>}
      <ul className="space-y-2">
        {pending.map((plan) => (
          <li key={plan.plan_id} className="flex flex-wrap items-center gap-2 rounded-lg border p-2">
            <code className="flex-1 break-all">
              {plan.src} → {plan.dst}
            </code>
            <Button size="sm" onClick={() => void decide(plan, "approve")} disabled={busy !== null}>
              Approve
            </Button>
            <Button size="sm" variant="outline" onClick={() => void decide(plan, "reject")} disabled={busy !== null}>
              Reject
            </Button>
          </li>
        ))}
      </ul>

      {decided.length > 0 && (
        <details>
          <summary className="cursor-pointer">Decided ({decided.length})</summary>
          <ul className="mt-2 space-y-1">
            {decided.map((plan) => (
              <li key={plan.plan_id} className="flex flex-wrap items-center gap-2">
                <span className="rounded px-2 py-0.5 border">{plan.status}</span>
                <code className="break-all">
                  {plan.src} → {plan.dst}
                </code>
                {plan.actor && <span className="text-muted-foreground">by {plan.actor}</span>}
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}

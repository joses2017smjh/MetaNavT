"use client";

import { useEffect, useState } from "react";
import ApiKeyField from "./api-key-field";
import ChatSection from "./chat-section";
import { type HealthInfo, readHealth } from "./demo/api";
import DemoSection from "./demo/demo-section";
import PlansPanel from "./plans-panel";

// One static export serves two kinds of API. When GET /health on this origin reports public_demo (PUBLIC_DEMO=1: only
// search and cited answers are served, app/api/public.py), the home page is the demo page; otherwise it is the chat
// interface, as before. The check runs in the browser, and neither renders until it has settled: the chat interface
// would ask at once for its starter questions and the plans list, routes the public demo does not serve.
export default function HomeSwitch() {
  const [health, setHealth] = useState<HealthInfo | null>(null); // null: /health has not answered yet

  useEffect(() => {
    let current = true;
    void readHealth().then((info) => {
      if (current) setHealth(info);
    });
    return () => {
      current = false;
    };
  }, []);

  if (health === null)
    return (
      <p role="status" className="rounded-xl bg-white/80 p-4 text-sm">
        Connecting to the API…
      </p>
    );
  if (health.publicDemo)
    return <DemoSection rerankerLoaded={health.rerankerLoaded} />;
  return (
    <>
      <ApiKeyField />
      <div className="h-[65vh] flex">
        <ChatSection />
      </div>
      <PlansPanel />
    </>
  );
}

"use client";

import { useEffect, useState } from "react";
import ApiKeyField from "./api-key-field";
import ChatSection from "./chat-section";
import { isPublicDemo } from "./demo/api";
import DemoSection from "./demo/demo-section";
import PlansPanel from "./plans-panel";

// One static export serves two kinds of API. When GET /health on this origin reports public_demo (PUBLIC_DEMO=1: only
// search and cited answers are served, app/api/public.py), the home page is the demo page; otherwise it is the chat
// interface, as before. The check runs in the browser, and neither renders until it has settled: the chat interface
// would ask at once for its starter questions and the plans list, routes the public demo does not serve.
export default function HomeSwitch() {
  const [mode, setMode] = useState<"checking" | "demo" | "chat">("checking");

  useEffect(() => {
    let current = true;
    void isPublicDemo().then((demo) => {
      if (current) setMode(demo ? "demo" : "chat");
    });
    return () => {
      current = false;
    };
  }, []);

  if (mode === "checking")
    return (
      <p role="status" className="rounded-xl bg-white/80 p-4 text-sm">
        Connecting to the API…
      </p>
    );
  if (mode === "demo") return <DemoSection />;
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

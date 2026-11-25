"use client";

import { type ReactNode, useEffect, useState } from "react";
import ApiKeyField from "./api-key-field";
import ChatSection from "./chat-section";
import { type HealthInfo, readHealth } from "./demo/api";
import DemoSection from "./demo/demo-section";
import Header, { type SectionLink } from "./header";
import PlansPanel from "./plans-panel";
import SiteFooter from "./site-footer";
import CapabilitiesSection from "./system/capabilities-section";
import DatasetSection from "./system/dataset-section";
import PipelineSection from "./system/pipeline-section";

// One static export serves two kinds of API. When GET /health on this origin reports public_demo (PUBLIC_DEMO=1: only
// search and cited answers are served, app/api/public.py), the home page is the demo page; otherwise it is the chat
// interface, as before. The check runs in the browser, and neither renders until it has settled: the chat interface
// would ask at once for its starter questions and the plans list, routes the public demo does not serve.
//
// Either way the page is laid out in sections, each with a link in the navigation bar: what the API serves first
// (Search and Ask, or Chat and Files), then what the same /health answer says about the index (Dataset) and the
// retrieval stages (Pipeline), then what the project does and where (Capabilities). The page asks /health once.
export default function HomeSwitch({ brand }: { brand?: ReactNode }) {
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
      <div id="top" className="page-wash min-h-screen">
        <Header brand={brand} links={[]} />
        <main className="mx-auto w-[92%] max-w-6xl pt-36">
          <p
            role="status"
            className="mx-auto w-fit rounded-full border border-black/10 bg-white px-4 py-2 text-sm text-muted-foreground shadow-pill"
          >
            Connecting to the API…
          </p>
        </main>
      </div>
    );

  const demo = health.publicDemo;
  const system = health.system;
  const links: SectionLink[] = [
    ...(demo
      ? [{ href: "#search", label: "Search" }]
      : [
          { href: "#chat", label: "Chat" },
          { href: "#files", label: "Files" },
        ]),
    ...(system
      ? [
          { href: "#dataset", label: "Dataset" },
          { href: "#pipeline", label: "Pipeline" },
        ]
      : []),
    { href: "#capabilities", label: "Capabilities" },
  ];

  return (
    <div id="top" className="page-wash min-h-screen">
      <Header brand={brand} links={links} />
      <main className="mx-auto w-[92%] max-w-6xl space-y-24 pt-32 sm:space-y-32 sm:pt-36">
        {demo ? (
          <DemoSection rerankerLoaded={health.rerankerLoaded} />
        ) : (
          <>
            <section
              id="chat"
              aria-labelledby="chat-title"
              className="space-y-8"
            >
              <div className="rise mx-auto max-w-3xl space-y-5 text-center">
                <p className="eyebrow">Chat</p>
                <h1
                  id="chat-title"
                  className="display text-[clamp(2.75rem,6.4vw,4.75rem)]"
                >
                  Chat with your files
                </h1>
                <p className="mx-auto max-w-2xl text-[15px] leading-relaxed text-muted-foreground sm:text-[17px]">
                  The chat agent works with the files this server has indexed. A
                  document attached to the message box is uploaded and indexed
                  with the rest.
                </p>
              </div>
              <div className="rise rise-2 mx-auto max-w-4xl space-y-4">
                <ApiKeyField />
                <div className="flex h-[65vh] min-h-[28rem]">
                  <ChatSection />
                </div>
              </div>
            </section>
            <section id="files" aria-label="Files">
              <PlansPanel />
            </section>
          </>
        )}
        {system && <DatasetSection system={system} demo={demo} />}
        {system && <PipelineSection system={system} demo={demo} />}
        <CapabilitiesSection demo={demo} />
      </main>
      <SiteFooter />
    </div>
  );
}

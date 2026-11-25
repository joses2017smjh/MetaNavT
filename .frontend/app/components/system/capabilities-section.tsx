import {
  Blocks,
  Cloud,
  FlaskConical,
  Gauge,
  Layers,
  type LucideIcon,
  MessagesSquare,
  Plug,
  Quote,
  ShieldCheck,
} from "lucide-react";
import { BENCHMARKS, DEPLOY_GUIDE, REFERENCE } from "../site";

// What the project does, one card per capability, and where each one lives: on the public demo's page, in the full
// application (the chat interface and its API), in both, or in the repository only. The page marks the ones the
// visitor can use where they are. No card carries a measurement: the numbers are in the README, which a link opens.

type Where = "demo" | "app" | "both" | "repository";

interface Capability {
  icon: LucideIcon;
  title: string;
  text: string;
  where: Where;
  link?: [string, string];
}

const CAPABILITIES: Capability[] = [
  {
    icon: Layers,
    title: "Hybrid retrieval",
    text: "Keyword and semantic search over PostgreSQL and pgvector, fused by reciprocal rank fusion, then a cross-encoder reranker where the server loads one.",
    where: "both",
  },
  {
    icon: Quote,
    title: "Citations to the byte range",
    text: "Every excerpt comes back with its file path and byte range, and an answer from Claude is shown only when it cites the excerpts it was written from.",
    where: "demo",
  },
  {
    icon: Blocks,
    title: "Structure-aware chunking",
    text: "Files are cut along their own structure: a Markdown section, a JSON value, a Python definition. Fixed-size windows are the baseline it has to beat.",
    where: "both",
  },
  {
    icon: MessagesSquare,
    title: "Chat agents with tools",
    text: "A chat agent that works with the indexed files through tools. A document attached to a message is uploaded and indexed with the rest.",
    where: "app",
  },
  {
    icon: ShieldCheck,
    title: "Reviewed file changes",
    text: "A file move is proposed as a plan. Nothing on disk changes until the server verifies a separately issued approval token.",
    where: "app",
  },
  {
    icon: Plug,
    title: "MCP server",
    text: "Any MCP client can search, read and list the corpus and propose changes. Applying one needs a separately issued operator capability.",
    where: "repository",
    link: [
      "Engineering reference",
      `${REFERENCE}#mcp-filesystem-with-a-human-interrupt`,
    ],
  },
  {
    icon: FlaskConical,
    title: "Measured, with the losers kept",
    text: "Retrieval is benchmarked on the project’s fixture and on BEIR with paired bootstrap intervals, and the tables keep every configuration.",
    where: "repository",
    link: ["Benchmark results", BENCHMARKS],
  },
  {
    icon: Gauge,
    title: "Guardrails for a public endpoint",
    text: "The public demo serves only its search and answer routes, caps the size of a request, and keeps answers under shared quotas and a daily budget.",
    where: "demo",
  },
  {
    icon: Cloud,
    title: "Deployed on AWS",
    text: "Terraform, CloudFront and one EC2 host running Docker Compose: FastAPI, PostgreSQL with pgvector, and Redis.",
    where: "repository",
    link: ["Deployment guide", DEPLOY_GUIDE],
  },
];

const WHERE: Record<Where, string> = {
  demo: "Public demo",
  app: "Full application",
  both: "Public demo and full application",
  repository: "Repository",
};

export default function CapabilitiesSection({ demo }: { demo: boolean }) {
  // The cards whose capability this page's own API serves.
  const here = (where: Where) =>
    where === "both" || where === (demo ? "demo" : "app");
  return (
    <section
      id="capabilities"
      aria-labelledby="capabilities-title"
      className="space-y-10"
    >
      <div className="mx-auto max-w-2xl space-y-5 text-center">
        <p className="eyebrow">Capabilities</p>
        <h2
          id="capabilities-title"
          className="display text-[clamp(2.25rem,4.6vw,3.5rem)] leading-[1.02]"
        >
          What MetaNaviT does, and where
        </h2>
        <p className="mx-auto max-w-xl text-[15px] leading-relaxed text-muted-foreground">
          Each card says where its capability lives. The ones marked with a dot
          are served by the API behind this page.
        </p>
      </div>
      <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {CAPABILITIES.map((capability) => {
          const Icon = capability.icon;
          const live = here(capability.where);
          return (
            <li
              key={capability.title}
              className="flex flex-col gap-4 rounded-card border border-black/[0.06] bg-white p-6 shadow-pill transition-shadow duration-300 hover:shadow-card"
            >
              <Icon
                aria-hidden
                strokeWidth={1.75}
                className="h-6 w-6 text-stone"
              />
              <div className="space-y-2">
                <h3 className="font-serif text-2xl font-medium leading-tight tracking-[-0.02em] text-brand-deep">
                  {capability.title}
                </h3>
                <p className="text-[15px] leading-relaxed text-ink/75">
                  {capability.text}
                </p>
              </div>
              <div className="mt-auto flex min-h-8 flex-wrap items-center justify-between gap-x-4 gap-y-2 pt-2">
                <p className="inline-flex items-center gap-2 text-[11px] uppercase leading-none tracking-[0.08em] text-muted-foreground">
                  <span
                    aria-hidden
                    className={`h-1.5 w-1.5 rounded-full ${live ? "bg-brand" : "bg-black/15"}`}
                  />
                  {WHERE[capability.where]}
                  {live && <span className="sr-only"> (served here)</span>}
                </p>
                {capability.link && (
                  <a
                    href={capability.link[1]}
                    className="text-[13px] font-medium text-brand-deep underline decoration-brand/60 underline-offset-[3px] transition-colors duration-200 hover:decoration-brand focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    {capability.link[0]}
                  </a>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

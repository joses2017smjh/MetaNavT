import type { SystemInfo } from "../demo/api";

// What this server has indexed, as its /health answer said when the page loaded (home-switch.tsx reads it once).
// Every value here is the API's: a field the answer does not have is shown as "not reported", never filled in.

// What the public demo's fixture holds, as the demo's own introduction lists it.
const FIXTURE_KINDS = [
  "YAML configs",
  "Slurm scripts",
  "Training logs",
  "An ablation table",
  "Paper drafts",
  "Python modules",
  "Checkpoint metadata",
];

const NOT_REPORTED = "not reported";

function Count({ value, label }: { value: number | null; label: string }) {
  return (
    <div className="flex flex-col-reverse justify-end gap-2">
      <dt className="text-sm text-muted-foreground">
        {label}
        {value === null ? ` (${NOT_REPORTED})` : ""}
      </dt>
      <dd className="font-sans text-[clamp(4.5rem,13vw,10rem)] font-normal leading-[0.92] tracking-[-0.045em] text-stone">
        {value !== null ? value.toLocaleString("en-US") : "—"}
      </dd>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string | null }) {
  return (
    <div className="grid gap-1 py-4 sm:grid-cols-[14rem_minmax(0,1fr)] sm:gap-6">
      <dt className="text-[11px] uppercase leading-5 tracking-[0.08em] text-muted-foreground">
        {label}
      </dt>
      <dd
        className={
          value !== null
            ? "break-words font-mono text-[13px] leading-5 text-ink"
            : "text-sm leading-5 text-muted-foreground"
        }
      >
        {value ?? NOT_REPORTED}
      </dd>
    </div>
  );
}

function statusText(system: SystemInfo): string | null {
  if (system.status === null) return null;
  const degraded = system.degraded
    .map((entry) => `${entry.component} (${entry.error})`)
    .join("; ");
  return degraded ? `${system.status}: ${degraded}` : system.status;
}

export default function DatasetSection({
  system,
  demo,
}: {
  system: SystemInfo;
  demo: boolean;
}) {
  return (
    <section
      id="dataset"
      aria-labelledby="dataset-title"
      className="space-y-10"
    >
      <div className="grid items-end gap-10 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:gap-16">
        <div className="space-y-5">
          <p className="eyebrow">Dataset</p>
          <h2
            id="dataset-title"
            className="display text-[clamp(2.25rem,4.6vw,3.5rem)] leading-[1.02]"
          >
            What this server has indexed
          </h2>
          <p className="max-w-md text-[15px] leading-relaxed text-muted-foreground">
            Read from this server’s{" "}
            <code className="font-mono text-[13px] text-ink">/health</code> when
            the page loaded. No value in this section is written into the page.
          </p>
          {demo && (
            <ul
              aria-label="What the demo’s fixture holds"
              className="flex flex-wrap gap-2 pt-1"
            >
              {FIXTURE_KINDS.map((kind) => (
                <li
                  key={kind}
                  className="rounded-full border border-black/10 bg-white px-3 py-1.5 text-[13px] leading-none text-ink/80 shadow-pill"
                >
                  {kind}
                </li>
              ))}
            </ul>
          )}
        </div>
        <dl className="grid grid-cols-2 gap-6 sm:gap-10">
          <Count value={system.files} label="files in the index’s manifest" />
          <Count value={system.rows} label="chunks in the index" />
        </dl>
      </div>
      <dl className="divide-y divide-black/10 border-y border-black/10">
        <Row label="Chunker" value={system.chunker} />
        <Row
          label="Indexing pipeline"
          value={system.pipelines.length ? system.pipelines.join(", ") : null}
        />
        <Row label="Embedding model" value={system.embeddingModel} />
        <Row label="Service status" value={statusText(system)} />
      </dl>
    </section>
  );
}

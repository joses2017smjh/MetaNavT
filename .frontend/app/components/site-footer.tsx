import { BENCHMARKS, DEPLOY_GUIDE, REFERENCE, REPOSITORY } from "./site";

// The foot of the page: the project's name on a dark green ground, and where to read more.

const LINKS: [string, string][] = [
  ["Source code", REPOSITORY],
  ["Benchmark results", BENCHMARKS],
  ["Engineering reference", REFERENCE],
  ["Deployment guide", DEPLOY_GUIDE],
];

export default function SiteFooter() {
  return (
    <footer className="mt-24 rounded-t-[2.5rem] bg-meadow text-white sm:mt-32 sm:rounded-t-[4rem]">
      <div className="mx-auto flex w-[92%] max-w-6xl flex-col gap-10 py-14 sm:py-20 md:flex-row md:items-end md:justify-between">
        <div className="max-w-md space-y-4">
          <p className="font-serif text-5xl leading-none tracking-[-0.03em] sm:text-6xl">
            MetaNaviT
          </p>
          <p className="text-[15px] leading-relaxed text-white/70">
            A hybrid search and RAG service for research files: keyword plus
            semantic search, and answers with citations to the exact file and
            byte range.
          </p>
        </div>
        <ul className="grid grid-cols-2 gap-x-10 gap-y-3 text-[15px]">
          {LINKS.map(([label, href]) => (
            <li key={href}>
              <a
                href={href}
                className="text-white/75 underline-offset-4 transition-colors duration-200 hover:text-white hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/80"
              >
                {label}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </footer>
  );
}

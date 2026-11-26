import type { ReactNode } from "react";
import { REPOSITORY } from "./site";

// The navigation bar: a frosted pane fixed over the top of the page, with the project's name, one link per section the
// page shows (home-switch.tsx chooses them: the public demo's sections, or the chat interface's) and a link to the
// source. `brand` is the mark in front of the name (app/page.tsx passes the logo).

export interface SectionLink {
  href: string;
  label: string;
}

export default function Header({
  brand,
  links,
}: {
  brand?: ReactNode;
  links: SectionLink[];
}) {
  return (
    <header className="fixed inset-x-0 top-3 z-50 px-3 sm:top-4">
      <nav
        aria-label="Sections"
        className="glass mx-auto flex max-w-[46rem] flex-wrap items-center gap-x-2 gap-y-1 rounded-2xl px-3 py-2"
      >
        <a
          href="#top"
          className="flex items-center gap-2 rounded-lg pr-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {brand}
          <span className="font-serif text-[1.375rem] leading-none tracking-[-0.02em] text-ink">
            MetaNaviT
          </span>
        </a>
        {links.length > 0 && (
          <div className="order-last flex w-full items-center justify-between gap-0.5 overflow-x-auto sm:order-none sm:w-auto sm:flex-1 sm:justify-center">
            {links.map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="whitespace-nowrap rounded-lg px-2 py-1.5 text-sm leading-none text-ink/70 sm:px-3 sm:text-[15px] transition-colors duration-200 hover:bg-black/[0.04] hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                {link.label}
              </a>
            ))}
          </div>
        )}
        <a
          href={REPOSITORY}
          className="ml-auto inline-flex h-8 items-center justify-center rounded-lg bg-ink px-3 text-[13px] font-medium leading-none text-white transition-[opacity,transform] duration-200 hover:opacity-90 active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
        >
          Source on GitHub
        </a>
      </nav>
    </header>
  );
}

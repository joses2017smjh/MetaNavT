import type { Metadata } from "next";
import { EB_Garamond, Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

// The page's three families, downloaded at build time and served with the export (no request to a font host from the
// browser): a sans for text, a serif for display headings, a mono for paths, scores and excerpts. tailwind.config.ts
// reads the variables.
const sans = Geist({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});
const serif = EB_Garamond({
  subsets: ["latin"],
  variable: "--font-serif",
  display: "swap",
});
const mono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "MetaNaviT",
  description:
    "Hybrid search and cited answers over a research project's files.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${sans.variable} ${serif.variable} ${mono.variable}`}>
        {children}
      </body>
    </html>
  );
}

// A browser for the tests: jsdom's window as the global scope, the way a jsdom test environment sets it up. Every test
// file imports this module first, before React and Testing Library look for `document`. `npm test` runs the files
// with Node's own test runner through tsx (tests/tsconfig.json compiles JSX with React's automatic runtime).
import { JSDOM } from "jsdom";
import Module from "node:module";

const { window } = new JSDOM(
  "<!doctype html><html><head></head><body></body></html>",
  { url: "http://localhost:3000/", pretendToBeVisual: true },
);
const scope = window as unknown as Record<string, unknown>;

// A name Node already defines keeps Node's value (fetch, Response, Headers, AbortController, URL, timers), except
// these: the page's storage and location, and the file classes, since jsdom's FileReader reads only jsdom's File.
const fromWindow = new Set([
  "window",
  "self",
  "document",
  "navigator",
  "location",
  "localStorage",
  "sessionStorage",
  "Blob",
  "File",
  "FileList",
  "FileReader",
]);
for (const name of Object.getOwnPropertyNames(window)) {
  if (name in globalThis && !fromWindow.has(name)) continue;
  const value = scope[name];
  // window's own functions (getComputedStyle, requestAnimationFrame, ...) need the window as `this`
  const bound =
    typeof value === "function" && !/^[A-Z]/.test(name)
      ? (value as (...args: unknown[]) => unknown).bind(window)
      : undefined;
  Object.defineProperty(globalThis, name, {
    configurable: true,
    enumerable: true,
    get: () => bound ?? scope[name],
  });
}

// Node's module hooks, which @types/node 20 does not describe.
type Resolve = (
  specifier: string,
  context: { conditions?: string[] },
) => unknown;
const loader = Module as unknown as {
  _extensions: Record<string, (module: unknown) => void>;
  registerHooks(hooks: {
    resolve(
      specifier: string,
      context: { conditions?: string[] },
      nextResolve: Resolve,
    ): unknown;
  }): void;
};

// chat-section.tsx imports chat-ui's stylesheets; Node cannot load CSS, and no test reads it.
loader._extensions[".css"] = () => {};

// The page (webpack) uses chat-ui's ES module build. Its CommonJS build, which a require() from the compiled tests
// would load, wraps the ES-module-only react-markdown as if it were CommonJS, so under Node its Markdown component is
// an object and every chat message fails to render. The tests load the ES module build too.
loader.registerHooks({
  resolve(specifier, context, nextResolve) {
    if (/^@llamaindex\/chat-ui(\/|$)/.test(specifier))
      return nextResolve(specifier, {
        ...context,
        conditions: ["import", "node", "default"],
      });
    return nextResolve(specifier, context);
  },
});

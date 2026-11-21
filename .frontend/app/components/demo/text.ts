import { MAX_QUESTION_CHARS, type RequestFailure } from "./api";

// What the demo page says, in plain words, when POST /api/answer/ delivers no answer (its `reason`: the route in
// app/api/routers/answer.py and app/answer/claude.py) and when a request fails.

const UPSTREAM: Record<string, string> = {
  "upstream timeout": "Claude's API did not answer in time.",
  "upstream rate limited":
    "Claude's API is turning this server's requests away for now (a rate limit, or the account's monthly spend cap).",
  "upstream unreachable": "This server could not reach Claude's API.",
  "upstream overloaded": "Claude's API is overloaded.",
  "upstream unavailable": "Claude's API is unavailable.",
};

export function reasonText(reason: string): string {
  switch (reason) {
    case "answers off":
      return "Answers are off on this server: it has no Anthropic API key. Search still works, and the excerpts it found are below.";
    case "daily budget spent":
      return "Today's answer budget is spent, so Claude was not asked. Answers come back after midnight UTC; search still works.";
    case "limits unavailable":
      return "Answers are paused: the server cannot reach the store that keeps its quotas and its budget, and it does not call Claude without them. Search still works.";
    case "uncited":
      return "Claude replied without citing any of the excerpts, so the reply is not shown as an answer.";
    case "nothing retrieved":
      return "Nothing was retrieved for this question, so Claude was not asked.";
    case "cut at max_tokens":
      return "Claude's reply reached its length limit before it ended, so it is not shown.";
  }
  const refused = /^refused \((.*)\)$/.exec(reason);
  if (refused)
    return refused[1] && refused[1] !== "no category"
      ? `Claude declined to answer this question (category: ${refused[1]}).`
      : "Claude declined to answer this question.";
  if (reason.startsWith("stopped: "))
    return `Claude's reply stopped early (${reason.slice("stopped: ".length)}), so it is not shown.`;
  if (reason.startsWith("upstream "))
    return `${UPSTREAM[reason] ?? `Claude's API returned an error (${reason}).`} Try again in a minute; search still works.`;
  return `No answer (${reason}).`;
}

function plural(count: number, unit: string): string {
  return `${count} ${unit}${count === 1 ? "" : "s"}`;
}

// "in 42 seconds", "in 1 hour 2 minutes", "after <date>" from a Retry-After header (seconds, or an HTTP date).
export function waitText(retryAfter: string | null): string | null {
  const raw = retryAfter?.trim();
  if (!raw) return null;
  if (/^\d+$/.test(raw)) {
    const seconds = Number(raw);
    if (seconds < 60) return `in ${plural(seconds, "second")}`;
    const minutes = Math.ceil(seconds / 60);
    if (minutes < 60) return `in ${plural(minutes, "minute")}`;
    const hours = Math.floor(minutes / 60);
    const rest = minutes % 60;
    return `in ${plural(hours, "hour")}${rest ? ` ${plural(rest, "minute")}` : ""}`;
  }
  const date = Date.parse(raw);
  return Number.isNaN(date) ? null : `after ${new Date(date).toUTCString()}`;
}

const LIMITS = `a question has 1 to ${MAX_QUESTION_CHARS} characters, not only spaces`;

export function failureText(failure: RequestFailure): string {
  const { status, detail } = failure;
  if (status === null)
    return "The API could not be reached. Check the connection and try again.";
  if (status === 429) {
    const wait = waitText(failure.retryAfter);
    const said = detail ? ` (the API says: ${detail})` : "";
    return wait
      ? `Too many requests: this demo limits how often each visitor can ask${said}. Try again ${wait} (Retry-After: ${failure.retryAfter?.trim()}).`
      : `Too many requests: this demo limits how often each visitor can ask${said}. Try again later.`;
  }
  if (status === 413)
    return `The request was too large for the API: ${LIMITS}.`;
  if (status === 422) return `The API did not accept the question: ${LIMITS}.`;
  if (status === 401)
    return "This server asks for an API key (HTTP 401), and the demo page has no field for one.";
  if (status === 503)
    return `The search backend is not ready${detail ? ` (${detail})` : ""}. It may still be starting: try again in a minute.`;
  if (status === 504)
    return `The API took too long to answer${detail ? ` (${detail})` : ""}. Try again.`;
  return `The API answered with HTTP ${status}${detail ? `: ${detail}` : ""}.`;
}

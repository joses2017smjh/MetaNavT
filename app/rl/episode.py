"""The episode the search policy is trained and scored in (M44): turns, the environment's replies, the score.

One episode is one assistant turn of a chat model, written in pieces. The policy writes `<search>query</search>`
and stops; the environment appends the results between `<information>` tags (the top INFO_CHUNKS chunks of the
corpus's in-memory hybrid index, app/rl/search_r1.py); the policy continues, and ends with
`<answer>VALUE [path]</answer>`. An episode may search MAX_SEARCHES times, and each piece the policy writes is at
most MAX_NEW_TOKENS tokens: the token budget.

An episode ends when a piece holds a complete answer ("answer"), holds no complete tag ("no_tag": the model
stopped, or ran out of tokens, without closing one), or asks for a search beyond the cap ("search_limit"). Only the
first ends with an answer.

Nothing here needs torch: run_episodes() drives any policy that implements Policy (app/rl/policy.py is the model;
the tests use a scripted one), and score_episode() is the whole reward, read from what an episode stored. The
results file's sidecar is these episodes as dictionaries, so every metric is recomputed from it on a CPU with no
retrieval and no model (app/rl/results.py).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from app.rl.reward import answer_value, bounded_match, cited_paths, search_r1_reward
from app.rl.search_r1 import ANSWER_RE, SEARCH_RE, SearchR1Env, Trajectory

MAX_SEARCHES = 3
MAX_NEW_TOKENS = 48
INFO_CHUNKS = 4
STOP_STRINGS = ("</search>", "</answer>")

SYSTEM_PROMPT = """You answer questions about a directory of research files: run configs, logs, job scripts, result tables, source files and paper drafts. You cannot see the files, but you can search them.

Rules:
- To search, write <search>a short query</search> and stop. The results come back between <information> and </information>. Each result is a file path, a byte range, and the text found there.
- When the results contain the answer, write <answer>VALUE [path]</answer> and stop. VALUE is the answer alone, as short as it can be: a number, a name, a path or a few words, copied as the file writes it. [path] is the file you read the answer in, copied from the results.
- You may search at most {searches}. Never answer before you have searched. Write nothing outside the tags.

Example
Question: what batch size did trial 7 use
<search>trial 7 batch size</search>
<information>
trials/trial_007.toml [0:58] # trial 7
batch_size = 64
optimizer = "adamw"
</information>
<answer>64 [trials/trial_007.toml]</answer>"""


def system_prompt(max_searches: int = MAX_SEARCHES) -> str:
    return SYSTEM_PROMPT.format(searches="once" if max_searches == 1 else f"{max_searches} times")


def user_message(question: str) -> str:
    return f"Question: {question}"


def information_block(information: str) -> str:
    """What the environment appends after a search, tags and line breaks included."""
    return f"\n<information>\n{information}\n</information>\n"


def prompt_sha256(max_searches: int = MAX_SEARCHES) -> str:
    """Identifies the prompt and the reply format a result was produced under."""
    text = "\x1e".join([system_prompt(max_searches), user_message("{question}"), information_block("{information}")])
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class Turn:
    text: str  # what was written in this piece (by the policy, unless forced)
    forced: bool = False  # written by the environment: the single-shot arm's search with the question itself
    query: str | None = None  # the search this piece asked for and got
    information: str | None = None  # the text the environment put between the <information> tags
    shown_paths: list[str] = field(default_factory=list)  # the files of the chunks shown, in order

    def to_dict(self) -> dict:
        return {"text": self.text, "forced": self.forced, "query": self.query, "information": self.information,
                "shown_paths": list(self.shown_paths)}


@dataclass
class Episode:
    question: str
    turns: list[Turn] = field(default_factory=list)
    answer: str | None = None  # the text inside <answer>, None when the episode ended without one
    stop: str = ""  # answer | no_tag | search_limit

    @property
    def searches(self) -> list[str]:
        return [t.query for t in self.turns if t.query is not None]

    @property
    def shown_paths(self) -> list[str]:
        """The files shown over the whole episode, in order of first appearance: what the reward ranks."""
        out: list[str] = []
        for turn in self.turns:
            for path in turn.shown_paths:
                if path not in out:
                    out.append(path)
        return out

    @property
    def raw(self) -> str:
        return "".join(t.text + (information_block(t.information) if t.information is not None else "") for t in self.turns)

    def cited(self) -> bool:
        """The answer names, in brackets, a file this episode's searches showed."""
        shown = set(self.shown_paths)
        return any(path in shown for path in cited_paths(self.answer or ""))

    def trajectory(self) -> Trajectory:
        """The scaffold's Trajectory for this episode, with `cited` by the rule above and the shown files as retrieved."""
        return Trajectory(raw=self.raw, searches=self.searches, informations=[t.information for t in self.turns if t.information is not None],
                          answer=self.answer or "", retrieved_paths=self.shown_paths, cited=self.cited())

    def to_dict(self) -> dict:
        return {"question": self.question, "turns": [t.to_dict() for t in self.turns], "answer": self.answer, "stop": self.stop}

    @classmethod
    def from_dict(cls, raw: dict) -> "Episode":
        return cls(question=raw["question"], turns=[Turn(**t) for t in raw["turns"]], answer=raw["answer"], stop=raw["stop"])


def read_turn(text: str) -> tuple[str, str | None]:
    """What a piece asks for: ("answer", its text), ("search", its query) or ("none", None). The first complete tag
    decides; a tag that is opened and never closed is not a tag."""
    search, answer = SEARCH_RE.search(text or ""), ANSWER_RE.search(text or "")
    if answer and (not search or answer.start() < search.start()):
        return "answer", answer.group(1).strip()
    if search:
        return "search", search.group(1).strip()
    return "none", None


def score_episode(episode: Episode, gold: str, gold_paths: Sequence[str]) -> dict:
    """The reward and its parts (app/rl/reward.py: strict exact match, nDCG@10 of the shown files, the search tax, the
    uncited-or-empty penalty; no jury term), plus the guard columns."""
    out = search_r1_reward(episode.trajectory(), gold=gold, gold_paths=gold_paths)
    value = answer_value(episode.answer or "")
    out.update({"em_bounded": bounded_match(episode.answer or "", gold), "answered": episode.answer is not None,
                "cited": episode.cited(), "answer_chars": len(value), "stop": episode.stop})
    return out


def corpus_env(files_root, embedder=None, *, retrieve_k: int = 50) -> SearchR1Env:
    """The environment over one corpus: the in-memory hybrid index as the benchmark's plain hybrid row builds it
    (BM25 and dense lists fused with RRF; no router, no reranker; a fixture's mtimes from its MANIFEST), with the
    embedder passed in so that many corpora share one model. embedder=None is the hash embedder: the tests'
    stand-in, never a measured run's."""
    from app.eval.index_loader import load_chunks_with_report
    from app.retrieval.hybrid import InMemoryHybridIndex

    chunks, _ = load_chunks_with_report(files_root, mtimes="manifest")
    return SearchR1Env(InMemoryHybridIndex(chunks, embedder=embedder, retrieve_k=retrieve_k, enable_router=False, enable_rerank=False),
                       max_searches=MAX_SEARCHES)


def greedy_episodes(policy: "Policy", questions: Sequence[Any], env: SearchR1Env, *, forced_first_search: bool = False,
                    max_searches: int = MAX_SEARCHES, encode_queries: Callable[[list[str]], Any] | None = None) -> dict[str, dict]:
    """{question id: {"episode", "score"}} for gold questions under greedy decoding. forced_first_search is the
    single-shot arm: one search with the question itself, then the answer (a cap of one search)."""
    episodes, _ = run_episodes(policy, [q.question for q in questions], [env] * len(questions), sample=False,
                               max_searches=1 if forced_first_search else max_searches, forced_first_search=forced_first_search,
                               encode_queries=encode_queries)
    return {q.id: {"episode": ep.to_dict(), "score": score_episode(ep, q.answer, q.relevant_paths)} for q, ep in zip(questions, episodes)}


class Policy(Protocol):
    """What run_episodes needs from a policy. A state is the policy's own record of one episode's text so far."""

    def begin(self, questions: Sequence[str], max_searches: int) -> list[Any]: ...

    def extend(self, state: Any, text: str) -> None:
        """Append text the environment wrote (never trained on)."""

    def generate(self, states: Sequence[Any], *, max_new_tokens: int, sample: bool) -> list[str]:
        """Continue every state by one piece, record it in the state, and return the pieces as text."""


def run_episodes(
    policy: Policy,
    questions: Sequence[str],
    envs: Sequence[SearchR1Env],
    *,
    sample: bool,
    max_searches: int = MAX_SEARCHES,
    max_new_tokens: int = MAX_NEW_TOKENS,
    n_info: int = INFO_CHUNKS,
    forced_first_search: bool = False,
    encode_queries: Callable[[list[str]], Any] | None = None,
) -> tuple[list[Episode], list[Any]]:
    """One episode per question against its own environment; returns (episodes, the policy's states).

    forced_first_search: the environment itself writes `<search>question</search>` as the first piece (single-shot
    RAG: one search with the question, then the policy answers). `encode_queries` embeds a batch of queries at once
    (the indexes share one embedder); without it each index embeds its own query."""
    episodes = [Episode(question=q) for q in questions]
    states = policy.begin(list(questions), max_searches)
    active = list(range(len(episodes)))

    def search(pending: list[tuple[int, Turn]]) -> None:
        vectors = encode_queries([turn.query for _, turn in pending]) if encode_queries is not None and pending else None
        for k, (i, turn) in enumerate(pending):
            vec = vectors[k] if vectors is not None and turn.query.strip() else None
            turn.information, turn.shown_paths = envs[i].search_shown(turn.query, n=n_info, query_vec=vec)
            policy.extend(states[i], information_block(turn.information))

    if forced_first_search:
        pending = []
        for i in active:
            turn = Turn(text=f"<search>{questions[i]}</search>", forced=True, query=questions[i])
            episodes[i].turns.append(turn)
            policy.extend(states[i], turn.text)
            pending.append((i, turn))
        search(pending)

    while active:
        texts = policy.generate([states[i] for i in active], max_new_tokens=max_new_tokens, sample=sample)
        still, pending = [], []
        for i, text in zip(active, texts):
            episode = episodes[i]
            kind, content = read_turn(text)
            turn = Turn(text=text)
            episode.turns.append(turn)
            if kind == "answer":
                episode.answer, episode.stop = content, "answer"
            elif kind == "none":
                episode.stop = "no_tag"
            elif len(episode.searches) >= max_searches:
                episode.stop = "search_limit"
            else:
                turn.query = content
                pending.append((i, turn))
                still.append(i)
        search(pending)
        active = still
    return episodes, states

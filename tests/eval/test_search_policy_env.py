"""M44: the search policy's episode, its reward and its exact match, with a scripted policy (no model, no torch).

The index here uses the hash embedder: these tests check the protocol and the arithmetic, not retrieval quality.
"""

from pathlib import Path

import pytest

from app.eval.gold import load_gold
from app.rl.episode import (
    MAX_SEARCHES,
    Episode,
    Turn,
    corpus_env,
    greedy_episodes,
    information_block,
    prompt_sha256,
    read_turn,
    run_episodes,
    score_episode,
    system_prompt,
)
from app.rl.reward import answer_value, bounded_match, cited_paths, exact_match, normalise

ROOT = Path(__file__).resolve().parents[2]


class Scripted:
    """A policy that writes the next piece of a fixed script per question. Its state is the list of texts so far."""

    def __init__(self, scripts: dict[str, list[str]]):
        self.scripts = scripts
        self.calls: list[dict] = []

    def begin(self, questions, max_searches):
        return [{"question": q, "pieces": [], "seen": []} for q in questions]

    def extend(self, state, text):
        state["seen"].append(text)

    def generate(self, states, *, max_new_tokens, sample):
        self.calls.append({"n": len(states), "max_new_tokens": max_new_tokens, "sample": sample})
        out = []
        for state in states:
            script = self.scripts[state["question"]]
            piece = script[min(len(state["pieces"]), len(script) - 1)]
            state["pieces"].append(piece)
            out.append(piece)
        return out


@pytest.fixture(scope="module")
def env():
    return corpus_env(ROOT / "bench" / "corpus" / "files")


def test_strict_exact_match_is_equality_of_the_value_not_containment():
    assert exact_match("3e-4 [configs/run_047.yaml]", "3e-4") == 1.0
    assert exact_match("[configs/run_047.yaml] 0.0003", "3e-04") == 1.0  # one number, three spellings
    assert exact_match("0.0003.", "3e-4") == 1.0 and exact_match('"Birch"', "birch") == 1.0
    assert exact_match("0.072 [configs/run_043.yaml]", "0.072") == 1.0 and exact_match("0.0720", "0.072") == 1.0
    assert exact_match("46, 47, 48, 52, 53", "46, 47, 48, 52, 53") == 1.0
    # what containment paid for and strict exact match does not
    assert exact_match("run 42 used 3 pairs", "2") == 0.0  # the scaffold's `gold in answer` found "2" in "42"
    assert exact_match("1, 2 or 3 [configs/run_040.yaml]", "2") == 0.0  # a list of candidates
    assert exact_match("the learning rate is 0.0003", "3e-4") == 0.0  # a sentence around the value
    assert exact_match("dinov2", "2") == 0.0 and exact_match("resnet50", "resnet50") == 1.0
    assert exact_match("", "2") == 0.0 and exact_match("[configs/run_040.yaml]", "2") == 0.0  # a citation is not a value
    assert exact_match("2", "") == 0.0
    # the guard column still credits the verbose answer and the list, and still refuses a number inside a longer one
    assert bounded_match("the learning rate is 0.0003 [configs/run_047.yaml]", "3e-4") == 1.0
    assert bounded_match("1, 2 or 3", "2") == 1.0 and bounded_match("run 42", "2") == 0.0
    assert bounded_match("[configs/run_047.yaml]", "configs/run_047.yaml") == 0.0  # the citation alone is not the answer


def test_normalise_leaves_numbers_inside_names_alone():
    assert normalise("  Configs/Run_047.yaml ") == "configs/run_047.yaml"
    assert normalise("dinov2; 3e-4") == "dinov2; 0.0003"
    assert normalise("0.0612 vs 0.0521") == normalise("0.0612  VS 0.05210")
    assert answer_value("  3e-4   [configs/run_047.yaml] ") == "3e-4"
    assert cited_paths("3e-4 [configs/run_047.yaml] [ logs/run_047.out ]") == ["configs/run_047.yaml", "logs/run_047.out"]


def test_read_turn_takes_the_first_complete_tag():
    assert read_turn("<search>run 47 lr</search>") == ("search", "run 47 lr")
    assert read_turn("<answer>3 [a.yaml]</answer>\n") == ("answer", "3 [a.yaml]")
    assert read_turn("<answer>3</answer><search>x</search>") == ("answer", "3")
    assert read_turn("<search>x</search><answer>3</answer>") == ("search", "x")
    assert read_turn("<search>never closed") == ("none", None) and read_turn("the answer is 3") == ("none", None)
    assert read_turn("<search></search>") == ("search", "")


def test_an_episode_searches_reads_and_answers_and_its_reward_is_the_declared_formula(env):
    q = "what learning rate did run 47 use"
    policy = Scripted({q: ["<search>run 47</search>", "<answer>0.0003 [configs/run_047.yaml]</answer>"]})
    (episode,), (state,) = run_episodes(policy, [q], [env], sample=False)
    assert episode.stop == "answer" and episode.answer == "0.0003 [configs/run_047.yaml]" and episode.searches == ["run 47"]
    assert "configs/run_047.yaml" in episode.shown_paths and len(episode.turns[0].shown_paths) <= 4
    assert state["seen"] == [information_block(episode.turns[0].information)]  # the policy saw exactly the stored block
    assert episode.raw == "<search>run 47</search>" + state["seen"][0] + "<answer>0.0003 [configs/run_047.yaml]</answer>"
    score = score_episode(episode, "3e-04", ["configs/run_047.yaml"])
    rank = episode.shown_paths.index("configs/run_047.yaml") + 1
    import math
    assert score["em"] == 1.0 and score["cited"] is True and score["uncited_or_empty"] == 0.0 and score["n_search"] == 1
    assert score["ndcg@10"] == round(1 / math.log2(rank + 1), 6) and score["jury_applied"] is False and score["jury_term"] == 0.0
    assert score["reward"] == round(1.0 + 0.5 * score["ndcg@10"] - 0.1, 6)
    assert Episode.from_dict(episode.to_dict()).to_dict() == episode.to_dict()
    assert score_episode(Episode.from_dict(episode.to_dict()), "3e-04", ["configs/run_047.yaml"]) == score  # rescored from the stored episode


def test_a_citation_counts_only_when_the_episode_showed_that_file(env):
    q = "what bark type did run 47 use"
    made_up = Scripted({q: ["<search>run 47</search>", "<answer>birch [configs/run_999.yaml]</answer>"]})
    (episode,), _ = run_episodes(made_up, [q], [env], sample=False)
    score = score_episode(episode, "birch", ["configs/run_047.yaml"])
    assert score["em"] == 1.0 and score["cited"] is False and score["uncited_or_empty"] == 1.0
    assert score["reward"] == round(1.0 + 0.5 * score["ndcg@10"] - 0.1 - 1.0, 6)
    no_search = Scripted({q: ["<answer>birch [configs/run_047.yaml]</answer>"]})  # nothing was shown, so nothing can be cited
    (episode,), _ = run_episodes(no_search, [q], [env], sample=False)
    score = score_episode(episode, "birch", ["configs/run_047.yaml"])
    assert score["n_search"] == 0 and score["cited"] is False and score["reward"] == 0.0  # 1.0 for the match, -1.0 uncited
    only_citation = Scripted({q: ["<search>run 47</search>", "<answer>[configs/run_047.yaml]</answer>"]})
    (episode,), _ = run_episodes(only_citation, [q], [env], sample=False)
    assert episode.cited() and score_episode(episode, "birch", ["configs/run_047.yaml"])["uncited_or_empty"] == 1.0  # cited, but no value


def test_the_search_cap_and_a_piece_without_a_tag_end_the_episode_unanswered(env):
    q = "which run had the highest val RMSE"
    greedy = Scripted({q: ["<search>val_rmse</search>"]})  # searches for ever
    (episode,), _ = run_episodes(greedy, [q], [env], sample=True)
    assert episode.stop == "search_limit" and episode.answer is None and len(episode.searches) == MAX_SEARCHES
    assert len(episode.turns) == MAX_SEARCHES + 1 and episode.turns[-1].query is None  # the fourth request got nothing
    score = score_episode(episode, "43", ["configs/run_043.yaml"])
    assert score["answered"] is False and score["uncited_or_empty"] == 1.0 and score["em"] == 0.0 and score["n_search"] == MAX_SEARCHES
    assert greedy.calls[0]["sample"] is True and [c["n"] for c in greedy.calls] == [1] * (MAX_SEARCHES + 1)
    chatty = Scripted({q: ["Sure, I will look that up."]})
    (episode,), _ = run_episodes(chatty, [q], [env], sample=False)
    assert episode.stop == "no_tag" and episode.searches == [] and score_episode(episode, "43", [])["reward"] == -1.0
    empty = Scripted({q: ["<search> </search>", "<answer>43 [configs/run_043.yaml]</answer>"]})
    (episode,), _ = run_episodes(empty, [q], [env], sample=False)
    assert episode.searches == [""] and episode.shown_paths == [] and score_episode(episode, "43", ["configs/run_043.yaml"])["n_search"] == 1


def test_single_shot_arm_searches_once_with_the_question_and_may_not_search_again(env):
    gold = [g for g in load_gold(ROOT / "bench" / "gold" / "questions.jsonl") if g.id in ("q036", "q038")]
    policy = Scripted({gold[0].question: ["<answer>0.0003 [configs/run_047.yaml]</answer>"], gold[1].question: ["<search>again</search>"]})
    rows = greedy_episodes(policy, gold, env, forced_first_search=True)
    first, second = rows["q036"], rows["q038"]
    assert first["episode"]["turns"][0] == {"text": f"<search>{gold[0].question}</search>", "forced": True, "query": gold[0].question,
                                            "information": first["episode"]["turns"][0]["information"],
                                            "shown_paths": first["episode"]["turns"][0]["shown_paths"]}
    assert first["score"]["n_search"] == 1 and first["episode"]["stop"] == "answer"
    assert second["episode"]["stop"] == "search_limit" and second["score"]["n_search"] == 1 and second["score"]["answered"] is False
    assert len(policy.calls) == 1 and policy.calls[0]["sample"] is False  # one piece each: the answer


def test_episodes_run_as_a_batch_and_finish_independently(env):
    a, b, c = "what encoder did run 40 use", "what bark type did run 41 use", "what was val RMSE for run 43"
    policy = Scripted({a: ["<answer>resnet50</answer>"], b: ["<search>run 41 bark type</search>", "<answer>oak [logs/run_041.out]</answer>"],
                       c: ["<search>run 43</search>", "<search>run 43 val rmse</search>", "<answer>0.072 [logs/run_043.out]</answer>"]})
    episodes, _ = run_episodes(policy, [a, b, c], [env, env, env], sample=False)
    assert [e.stop for e in episodes] == ["answer"] * 3 and [len(e.searches) for e in episodes] == [0, 1, 2]
    assert [e.cited() for e in episodes] == [False, True, True]
    assert [call["n"] for call in policy.calls] == [3, 2, 1]  # each piece is generated only for the episodes still running


def test_the_prompt_states_the_protocol_and_its_hash_moves_with_it():
    text = system_prompt()
    assert f"at most {MAX_SEARCHES} times" in text and "<answer>VALUE [path]</answer>" in text and "<search>" in text
    assert "run_0" not in text and "dinov2" not in text.lower()  # the example is not from the fixture
    assert "at most once." in system_prompt(1) and "times" not in system_prompt(1).split("at most")[1].split(".")[0]
    assert prompt_sha256() == prompt_sha256(MAX_SEARCHES) and prompt_sha256(1) != prompt_sha256(MAX_SEARCHES)
    assert Turn(text="x").to_dict() == {"text": "x", "forced": False, "query": None, "information": None, "shown_paths": []}

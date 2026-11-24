"""M44: the parts of the search policy's training loop that need no model and no torch: which questions a step
asks, how episodes are cut into backward passes, GRPO's per-question advantage, the warm-up.
"""

from app.rl.corpora import training_set
from app.rl.train import (
    TOKEN_BUDGET,
    WARMUP_STEPS,
    batch_normalised,
    group_advantages,
    lr_at,
    micro_batches,
    question_schedule,
    scoring_step,
    step_questions,
    summarise,
)


def test_a_steps_questions_are_half_lookups_and_half_other_templates_and_depend_only_on_seed_and_step():
    questions, _ = training_set([1, 2, 3])
    schedule = question_schedule(questions, seed=0)
    assert sorted(schedule[0] + schedule[1]) == list(range(len(questions))) and schedule == question_schedule(questions, seed=0)
    assert schedule != question_schedule(questions, seed=1)
    batches = [step_questions(questions, schedule, step, 8) for step in range(60)]
    for batch in batches:
        assert [q.template.startswith("simple.") for q in batch] == [True] * 4 + [False] * 4
    assert batches[7] == step_questions(questions, schedule, 7, 8)  # a resumed run asks the same questions at the same step
    n_other = len(schedule[1])
    seen = [q.key for batch in batches for q in batch[4:]]
    assert len(set(seen[:n_other])) == n_other == len(set(seen))  # every other-template question once before any repeats
    lookups = [q.key for batch in batches for q in batch[:4]]
    assert len(set(lookups)) == len(lookups)  # 240 lookups asked, none twice
    natural = [q for step in range(40) for q in step_questions(questions, schedule, step, 8, "natural")]
    share = sum(q.template.startswith("simple.") for q in natural) / len(natural)
    assert 0.6 < share < 0.85 and len({q.key for q in natural}) == len(natural)


def test_micro_batches_respect_the_token_budget_and_group_advantages_are_per_question():
    lengths = [10, 10, 30, 30, 100, 100]
    assert micro_batches([0, 1, 2, 3, 4, 5], lengths, 120) == [[0, 1, 2, 3], [4], [5]]
    assert micro_batches([0, 1, 2, 3, 4, 5], lengths, 10**6) == [[0, 1, 2, 3, 4, 5]] and micro_batches([], lengths, 100) == []
    assert micro_batches([4], lengths, 5) == [[4]]  # a sequence longer than the budget still runs, alone
    adv = group_advantages([1.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0], 4)
    assert adv[4:] == [0.0, 0.0, 0.0, 0.0] and adv[0] > 0 > adv[1] and abs(sum(adv[:4])) < 1e-6
    assert abs(sum(a * a for a in adv[:4]) / 4 - 1.0) < 1e-6  # unit variance inside the group
    assert all(abs(a - b) < 1e-6 for a, b in zip(batch_normalised([2.0, 4.0]), [-1.0, 1.0]))  # the PPO fallback normalises over the whole batch


def test_the_tune_half_scores_every_eval_every_steps_and_the_last_and_never_step_zero_through_this_rule():
    assert [d for d in range(0, 131) if scoring_step(d, 50, 130)] == [50, 100, 130]
    assert [d for d in range(0, 301) if scoring_step(d, 50, 300)] == [50, 100, 150, 200, 250, 300]


def test_the_learning_rate_warms_up_and_the_token_budget_is_set_per_precision():
    assert lr_at(0, 1e-4) == 1e-4 / WARMUP_STEPS and lr_at(WARMUP_STEPS - 1, 1e-4) == 1e-4 == lr_at(10 * WARMUP_STEPS, 1e-4)
    assert set(TOKEN_BUDGET) == {"bf16", "fp32"} and TOKEN_BUDGET["fp32"] < TOKEN_BUDGET["bf16"]  # fp32 holds twice as much per token


def test_a_steps_summary_is_the_mean_of_its_episodes_scores():
    scores = [{"reward": 1.0, "em": 1.0, "em_bounded": 1.0, "ndcg@10": 1.0, "n_search": 1, "uncited_or_empty": 0.0, "answered": True, "answer_chars": 4, "stop": "answer"},
              {"reward": -1.0, "em": 0.0, "em_bounded": 0.0, "ndcg@10": 0.0, "n_search": 3, "uncited_or_empty": 1.0, "answered": False, "answer_chars": 0, "stop": "search_limit"}]
    out = summarise(scores)
    assert out == {"n": 2, "reward": 0.0, "em": 0.5, "em_bounded": 0.5, "ndcg@10": 0.5, "n_search": 2.0, "uncited_or_empty": 0.5, "answered": 0.5,
                   "answer_chars": 2.0, "stop": {"answer": 1, "search_limit": 1}}

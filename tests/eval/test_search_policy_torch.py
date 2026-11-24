"""M44: the policy's sampler, its loss mask and its update, on a tiny random model (CPU, a few seconds).

Skipped where torch, peft or the cached Qwen tokenizer are missing (CI's eval job installs none of them): the
training venv runs it before every pilot and training job. No weights of the real policy model are loaded.
"""

import math
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from app.rl.episode import information_block, read_turn, run_episodes
from app.rl.policy import LORA, HFPolicy, State
from app.rl.train import micro_batches, policy_gradient_step, ppo_step

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL)
    except Exception as exc:  # not cached on this machine
        pytest.skip(f"the {MODEL} tokenizer is not in the local cache: {exc}")


def tiny(tokenizer, *, lora: bool = True, seed: int = 0) -> HFPolicy:
    torch.manual_seed(seed)
    config = transformers.Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096,
                                      eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    model = transformers.Qwen2ForCausalLM(config)
    model.generation_config.eos_token_id = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    return HFPolicy(MODEL, device="cpu", lora=LORA if lora else None, trainable=True, seed=seed, model=model, tokenizer=tokenizer, batch_size=3)


class Forced(torch.nn.Module):
    """An output head that makes the sampler write given token sequences: row r, step t -> a huge logit at scripts[r][t]."""

    def __init__(self, vocab: int, scripts: list[list[int]]):
        super().__init__()
        self.vocab, self.scripts, self.step = vocab, scripts, 0

    def forward(self, hidden):
        logits = torch.zeros((hidden.shape[0], self.vocab))
        for r in range(hidden.shape[0]):
            script = self.scripts[r]
            logits[r, script[min(self.step, len(script) - 1)]] = 50.0
        self.step += 1
        return logits


class Env:
    def __init__(self, information: str = "notes/a.txt [0:9] value: 7"):
        self.information = information

    def search_shown(self, query, *, n=4, query_vec=None):
        return self.information, ["notes/a.txt"]


def test_a_piece_ends_where_it_closes_a_tag_or_ends_the_turn_and_never_later(tokenizer):
    policy = tiny(tokenizer, lora=False)
    search = policy.encode("<search>value</search>")
    chatter = policy.encode(" and then more text that must never be sampled")
    ended = policy.encode("I do not know") + [tokenizer.eos_token_id]
    long = policy.encode("one two three four five six seven eight nine ten eleven twelve")
    real_head = policy.head
    policy.head = Forced(len(tokenizer), [search + chatter, ended + chatter, long])
    contexts = [policy.encode("a"), policy.encode("a b c d e f"), policy.encode("a b c")]  # unequal lengths: left padding
    new, logprobs = policy._continue(contexts, max_new_tokens=10, sample=False)
    policy.head = real_head
    assert len(search) < 10 and new[0] == search  # stops with the token that closes the tag
    assert tokenizer.decode(new[0]).endswith("</search>") and read_turn(tokenizer.decode(new[0])) == ("search", "value")
    assert new[1] == ended  # the end-of-turn token is the last token of the piece, and is part of it
    assert len(long) > 10 and new[2] == long[:10]  # the token cap
    assert [len(lp) for lp in logprobs] == [len(ids) for ids in new]
    assert all(abs(x) < 1e-6 for lp in logprobs for x in lp)  # a logit of 50 against zeros: probability one


def test_the_training_pass_scores_each_policy_token_as_the_sampler_did_and_masks_everything_else(tokenizer):
    policy = tiny(tokenizer)
    questions = ["what is the value", "what is the other value", "third"]
    episodes, states = run_episodes(policy, questions, [Env(), Env("notes/a.txt [0:9] value: 7\nnotes/b.txt [0:5] x"), Env()],
                                    sample=True, max_new_tokens=6, max_searches=2, forced_first_search=True)
    for state, episode in zip(states, episodes):
        flags = [flag for _, flag in state.segments]
        assert flags[:3] == [False, False, False]  # the prompt, the forced search, its information block: environment text
        assert flags[3] is True and state.n_policy_tokens == len(state.logprobs) == sum(state.mask) > 0
        assert len(state.ids) == len(state.mask)
        text = tokenizer.decode(state.ids, skip_special_tokens=False)
        assert information_block(episode.turns[0].information) in text and f"<search>{episode.question}</search>" in text
    logps = policy.policy_logprobs(states)
    for state, lp in zip(states, logps):
        assert lp.shape == (state.n_policy_tokens,) and lp.requires_grad and lp.dtype == torch.float32
        assert torch.allclose(lp.detach(), torch.tensor(state.logprobs), atol=1e-4)  # the same tokens, the same weights
    # one state alone scores the same as in the batch (right padding in the training pass changes nothing)
    alone = policy.policy_logprobs(states[1:2])[0]
    assert torch.allclose(alone, logps[1], atol=1e-5)
    # retrieved text is context, not a target: changing it changes the policy tokens' scores, never their count
    changed = State(segments=[(list(ids), flag) for ids, flag in states[0].segments], logprobs=list(states[0].logprobs))
    changed.segments[2] = (policy.encode(information_block("notes/z.txt [0:3] no")), False)
    other = policy.policy_logprobs([changed])[0]
    assert other.shape == logps[0].shape and not torch.allclose(other, logps[0], atol=1e-6)


def test_greedy_decoding_is_the_argmax_and_sampling_is_seeded(tokenizer):
    policy = tiny(tokenizer, lora=False)
    context = policy.prompt_ids("what is the value", 3)
    a, lp_a = policy._continue([context], 5, sample=False)
    b, _ = policy._continue([context, context[:-3]], 5, sample=False)  # in a batch with another row
    assert a[0] == b[0]
    with torch.no_grad():
        hidden = policy.body(input_ids=torch.tensor([context + a[0][:-1]])).last_hidden_state
        logp = torch.log_softmax(policy.head(hidden[0, len(context) - 1:]).float(), dim=-1)
    assert logp.argmax(dim=-1).tolist() == a[0]  # no temperature, no top-k, no repetition penalty between logits and choice
    assert torch.allclose(logp.gather(1, torch.tensor(a[0])[:, None]).squeeze(1), torch.tensor(lp_a[0]), atol=1e-4)
    policy.generator.manual_seed(7)
    s1, _ = policy._continue([context], 5, sample=True)
    policy.generator.manual_seed(7)
    s2, _ = policy._continue([context], 5, sample=True)
    policy.generator.manual_seed(8)
    s3, _ = policy._continue([context], 5, sample=True)
    assert s1 == s2 and s1 != s3 and s1 != a


def test_one_step_trains_only_the_adapter_and_follows_the_sign_of_the_advantage(tokenizer):
    policy = tiny(tokenizer)
    base_before = {n: p.detach().clone() for n, p in policy.model.named_parameters() if not p.requires_grad}
    lora_before = {n: p.detach().clone() for n, p in policy.model.named_parameters() if p.requires_grad}
    assert lora_before and all("lora_" in n for n in lora_before) and all(p.dtype == torch.float32 for p in policy.trainable_parameters())
    _, states = run_episodes(policy, ["q one", "q two", "q three", "q four"], [Env()] * 4, sample=True, max_new_tokens=6, max_searches=1)
    advantages = [1.0, -1.0, 0.0, 0.0]
    before = [float(lp.mean()) for lp in policy.policy_logprobs(states)]
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=1e-2, weight_decay=0.0)
    stats = policy_gradient_step(policy, optimizer, states, advantages, token_budget=10**6)
    after = [float(lp.mean()) for lp in policy.policy_logprobs(states)]
    assert stats["n_episodes_with_gradient"] == 2 and stats["grad_norm"] > 0 and math.isfinite(stats["loss"])
    assert stats["logprob_gap_max"] < 1e-3  # before the update, the training pass agrees with the sampler
    assert abs(stats["loss"] - (-(before[0]) + before[1]) / 4) < 1e-4  # sum(-A * mean log-prob) / n episodes
    assert after[0] > before[0] and after[1] < before[1]  # more of what was rewarded, less of what was not
    for n, p in policy.model.named_parameters():
        if n in base_before:
            assert torch.equal(p, base_before[n]), n  # the base model does not move
    assert any(not torch.equal(p, lora_before[n]) for n, p in policy.model.named_parameters() if n in lora_before)
    assert stats["token_budget"] == 10**6
    # a batch with no reward spread anywhere takes no step
    frozen = {n: p.detach().clone() for n, p in policy.model.named_parameters() if p.requires_grad}
    assert policy_gradient_step(policy, optimizer, states, [0.0] * 4)["n_episodes_with_gradient"] == 0
    assert all(torch.equal(p, frozen[n]) for n, p in policy.model.named_parameters() if n in frozen)


def test_ppo_fallback_step_clips_the_ratio_against_the_sampling_log_probabilities(tokenizer):
    policy = tiny(tokenizer)
    _, states = run_episodes(policy, ["q one", "q two"], [Env()] * 2, sample=True, max_new_tokens=6, max_searches=1)
    before = [float(lp.mean()) for lp in policy.policy_logprobs(states)]
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=1e-2, weight_decay=0.0)
    stats = ppo_step(policy, optimizer, states, [1.0, -1.0], epochs=2, clip=0.2, token_budget=10**6)
    after = [float(lp.mean()) for lp in policy.policy_logprobs(states)]
    assert after[0] > before[0] and after[1] < before[1] and 0.0 <= stats["clip_fraction"] <= 1.0 and math.isfinite(stats["loss"])


def test_an_adapter_saved_and_loaded_decodes_the_same(tokenizer, tmp_path):
    policy = tiny(tokenizer)
    _, states = run_episodes(policy, ["q one", "q two"], [Env()] * 2, sample=True, max_new_tokens=6, max_searches=1)
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=1e-2, weight_decay=0.0)
    policy_gradient_step(policy, optimizer, states, [1.0, -1.0])
    policy.save_adapter(tmp_path / "adapter")
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists() and (tmp_path / "adapter" / "adapter_config.json").exists()
    context = policy.prompt_ids("what is the value", 3)
    trained, _ = policy._continue([context], 6, sample=False)
    torch.manual_seed(0)
    config = transformers.Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                                      num_key_value_heads=1, max_position_embeddings=4096, eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.pad_token_id)
    fresh = transformers.Qwen2ForCausalLM(config)  # the same random base (same seed), without the adapter
    fresh.generation_config.eos_token_id = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    loaded = HFPolicy(MODEL, device="cpu", adapter=tmp_path / "adapter", model=fresh, tokenizer=tokenizer)
    again, _ = loaded._continue([context], 6, sample=False)
    assert again == trained and not loaded.trainable_parameters()
    with torch.no_grad():
        h1 = policy.body(input_ids=torch.tensor([context])).last_hidden_state
        h2 = loaded.body(input_ids=torch.tensor([context])).last_hidden_state
    assert torch.allclose(h1, h2, atol=1e-6)


def test_a_backward_pass_that_runs_out_of_memory_restarts_the_step_with_half_the_budget(tokenizer, monkeypatch):
    from app.rl import train

    monkeypatch.setattr(train, "MIN_TOKEN_BUDGET", 1)
    policy = tiny(tokenizer)
    _, states = run_episodes(policy, ["q one", "q two", "q three"], [Env()] * 3, sample=True, max_new_tokens=6, max_searches=1)
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=1e-2, weight_decay=0.0)
    reference = tiny(tokenizer)  # the same weights, stepped without any failure, one episode per backward pass too
    ref_opt = torch.optim.AdamW(reference.trainable_parameters(), lr=1e-2, weight_decay=0.0)
    advantages = [1.0, -1.0, 0.5]
    budget = int(1.5 * max(len(s.ids) for s in states))
    expected = policy_gradient_step(reference, ref_opt, states, advantages, token_budget=budget)
    real, calls = policy.policy_logprobs, {"n": 0}

    def fails_once(batch):
        calls["n"] += 1
        if calls["n"] == 2:  # the second micro-batch: the first has already added its gradient
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return real(batch)

    policy.policy_logprobs = fails_once
    assert [len(m) for m in micro_batches([0, 1, 2], [len(s.ids) for s in states], budget)] == [1, 1, 1]
    assert expected["token_budget"] == budget
    stats = policy_gradient_step(policy, optimizer, states, advantages, token_budget=budget)
    policy.policy_logprobs = real
    assert calls["n"] == 2 + 3 and stats["token_budget"] == budget // 2 and abs(stats["loss"] - expected["loss"]) < 1e-5
    for (n1, p1), (_, p2) in zip(policy.model.named_parameters(), reference.model.named_parameters()):
        if p1.requires_grad:
            assert torch.allclose(p1, p2, atol=1e-6), n1  # the retried step is the whole step once, not one micro-batch twice

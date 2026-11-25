"""The policy model (M44): a causal LM with a LoRA adapter, its sampler, and the log-probabilities the loss trains.

Needs torch, transformers and peft (the `rl` extra); nothing else in app/rl does.

An episode is kept as token ids in segments, each written by the policy or by the environment (the prompt, an
`<information>` block, a forced search). The context the model continues is the plain concatenation of those ids,
and the training pass runs over the same concatenation, so the log-probability the loss trains is the
log-probability the token had when it was sampled, and the mask of policy tokens is exact: the retrieved tokens
are in the context and never in the loss.

Sampling is written out here instead of calling `generate()`: the model's generation_config ships a temperature, a
top-p, a top-k and a repetition penalty, and `generate()` fills every setting a caller leaves unset from it. This
sampler draws from softmax(logits) with nothing in between (temperature 1), or takes the argmax, so the behaviour
policy is exactly the distribution the gradient is taken through. A piece ends when it closes a tag
(app.rl.episode.STOP_STRINGS), when the model writes an end-of-turn token, or at the token cap.

The base weights load in bfloat16 on a GPU that has it natively and the adapter trains in float32 under autocast:
an Adam update at these learning rates is below bfloat16's resolution for a weight of ordinary size. Elsewhere
everything is float32 (`precision`).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from app.rl.episode import STOP_STRINGS, system_prompt, user_message

LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.0, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "bias": "none", "task_type": "CAUSAL_LM"}
STOP_WINDOW = 12  # tokens decoded to see whether a piece has just closed a tag (a closing tag is three or four tokens)


@dataclass
class State:
    """One episode as the model sees it."""

    segments: list[tuple[list[int], bool]] = field(default_factory=list)  # (token ids, written by the policy)
    logprobs: list[float] = field(default_factory=list)  # of each policy token, under the policy that sampled it

    @property
    def ids(self) -> list[int]:
        return [t for seg, _ in self.segments for t in seg]

    @property
    def mask(self) -> list[bool]:
        """True at the tokens the policy wrote: the only ones in the loss."""
        return [flag for seg, flag in self.segments for _ in seg]

    @property
    def n_policy_tokens(self) -> int:
        return sum(len(seg) for seg, flag in self.segments if flag)


class HFPolicy:
    def __init__(self, model_name: str, *, device: str = "cuda", adapter: str | Path | None = None, lora: dict | None = None,
                 trainable: bool = False, seed: int = 0, batch_size: int = 64, model=None, tokenizer=None, precision: str = "auto"):
        """adapter: a saved LoRA adapter to load; lora: a new adapter with this LoraConfig; neither: the base model.
        model / tokenizer: already-built objects (the tests pass a tiny random model).
        precision: bf16 (base weights and autocast in bfloat16; needs a GPU with native bfloat16, compute capability
        8 or later), fp32, or auto: bf16 where it is native, fp32 elsewhere (a CPU, an RTX 8000). `self.precision`
        is what was used; a results file records it."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if precision not in ("auto", "bf16", "fp32"):
            raise ValueError(f"precision must be auto, bf16 or fp32, got {precision!r}")
        self.torch = torch
        self.model_name = model_name
        self.device = torch.device(device)
        native = self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] >= 8
        if precision == "bf16" and not native:
            raise RuntimeError("bf16 needs a GPU with native bfloat16 (compute capability 8 or later)")
        self.precision = "bf16" if native and precision != "fp32" else "fp32"
        self.dtype = torch.bfloat16 if self.precision == "bf16" else torch.float32
        self.tok = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        base = model if model is not None else AutoModelForCausalLM.from_pretrained(model_name, dtype=self.dtype)
        end = base.generation_config.eos_token_id
        self.end_ids = {int(t) for t in (end if isinstance(end, (list, tuple)) else [end]) if t is not None} | {int(self.tok.eos_token_id)}
        self.pad_id = int(self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id)
        if adapter is not None:
            from peft import PeftModel

            base = PeftModel.from_pretrained(base, str(adapter), is_trainable=trainable)
        elif lora is not None:
            from peft import LoraConfig, get_peft_model

            base = get_peft_model(base, LoraConfig(**lora))
        self.model = base.to(self.device)
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.model.eval()  # no dropout anywhere (lora_dropout 0): the training pass sees the sampler's distribution
        causal = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        self.body, self.head = causal.model, causal.lm_head
        self.batch_size = batch_size
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    # ---- the Policy protocol (app/rl/episode.py)

    def prompt_ids(self, question: str, max_searches: int) -> list[int]:
        messages = [{"role": "system", "content": system_prompt(max_searches)}, {"role": "user", "content": user_message(question)}]
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.encode(text)

    def encode(self, text: str) -> list[int]:
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def begin(self, questions: Sequence[str], max_searches: int) -> list[State]:
        return [State(segments=[(self.prompt_ids(q, max_searches), False)]) for q in questions]

    def extend(self, state: State, text: str) -> None:
        state.segments.append((self.encode(text), False))

    def generate(self, states: Sequence[State], *, max_new_tokens: int, sample: bool) -> list[str]:
        texts: list[str] = [""] * len(states)
        order = sorted(range(len(states)), key=lambda i: len(states[i].ids))  # similar lengths share a batch
        for start in range(0, len(order), self.batch_size):
            chunk = order[start:start + self.batch_size]
            new, logprobs = self._continue([states[i].ids for i in chunk], max_new_tokens, sample)
            for i, ids, lps in zip(chunk, new, logprobs):
                states[i].segments.append((ids, True))
                states[i].logprobs.extend(lps)
                texts[i] = self.tok.decode(ids, skip_special_tokens=True)
        return texts

    # ---- sampling and the training pass

    def _autocast(self):
        if self.precision == "bf16":
            return self.torch.autocast("cuda", dtype=self.torch.bfloat16)
        return contextlib.nullcontext()

    def _closed(self, ids: list[int]) -> bool:
        tail = self.tok.decode(ids[-STOP_WINDOW:], skip_special_tokens=False)
        return any(stop in tail for stop in STOP_STRINGS)

    def _continue(self, contexts: list[list[int]], max_new_tokens: int, sample: bool) -> tuple[list[list[int]], list[list[float]]]:
        """One piece per context: (its token ids, the log-probability each had). Left-padded batch, key-value cache."""
        torch = self.torch
        n, width = len(contexts), max(len(c) for c in contexts)
        ids = torch.full((n, width), self.pad_id, dtype=torch.long)
        attn = torch.zeros((n, width), dtype=torch.long)
        for r, context in enumerate(contexts):
            ids[r, width - len(context):] = torch.tensor(context, dtype=torch.long)
            attn[r, width - len(context):] = 1
        ids, attn = ids.to(self.device), attn.to(self.device)
        new: list[list[int]] = [[] for _ in range(n)]
        logprobs: list[list[float]] = [[] for _ in range(n)]
        done = [False] * n
        with torch.no_grad():
            with self._autocast():
                out = self.body(input_ids=ids, attention_mask=attn, use_cache=True)
            past, hidden = out.past_key_values, out.last_hidden_state[:, -1]
            for step in range(max_new_tokens):
                with self._autocast():
                    logits = self.head(hidden)
                logp = torch.log_softmax(logits.float(), dim=-1)
                if sample:
                    nxt = torch.multinomial(logp.exp(), 1, generator=self.generator).squeeze(1)
                else:
                    nxt = logp.argmax(dim=-1)
                chosen = logp.gather(1, nxt[:, None]).squeeze(1).tolist()
                tokens = nxt.tolist()
                for r in range(n):
                    if done[r]:
                        continue
                    new[r].append(tokens[r])
                    logprobs[r].append(chosen[r])
                    if tokens[r] in self.end_ids or self._closed(new[r]):
                        done[r] = True
                if all(done) or step == max_new_tokens - 1:
                    break
                feed = torch.where(torch.tensor(done, device=self.device), torch.full_like(nxt, self.pad_id), nxt)
                attn = torch.cat([attn, torch.ones((n, 1), dtype=torch.long, device=self.device)], dim=1)
                with self._autocast():
                    out = self.body(input_ids=feed[:, None], attention_mask=attn, past_key_values=past, use_cache=True)
                past, hidden = out.past_key_values, out.last_hidden_state[:, -1]
        return new, logprobs

    def policy_logprobs(self, states: Sequence[State]):
        """Per state, a float32 tensor (with gradient) of the log-probability of each policy token given everything
        before it. The output head runs on the policy tokens' positions only: the prompt and the retrieved text are
        context, not targets."""
        torch = self.torch
        seqs = [s.ids for s in states]
        n, width = len(seqs), max(len(s) for s in seqs)
        ids = torch.full((n, width), self.pad_id, dtype=torch.long)
        attn = torch.zeros((n, width), dtype=torch.long)
        rows, cols, targets, counts = [], [], [], []
        for r, (seq, state) in enumerate(zip(seqs, states)):
            ids[r, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn[r, :len(seq)] = 1
            positions = [p for p, flag in enumerate(state.mask) if flag]
            rows += [r] * len(positions)
            cols += [p - 1 for p in positions]  # the position that predicts token p
            targets += [seq[p] for p in positions]
            counts.append(len(positions))
        ids, attn = ids.to(self.device), attn.to(self.device)
        with self._autocast():
            hidden = self.body(input_ids=ids, attention_mask=attn, use_cache=False).last_hidden_state
            logits = self.head(hidden[torch.tensor(rows, device=self.device), torch.tensor(cols, device=self.device)])
        logp = torch.log_softmax(logits.float(), dim=-1)
        picked = logp.gather(1, torch.tensor(targets, device=self.device)[:, None]).squeeze(1)
        return list(torch.split(picked, counts))

    # ---- adapters

    def trainable_parameters(self) -> list:
        return [p for p in self.model.parameters() if p.requires_grad]

    def save_adapter(self, path: str | Path) -> None:
        self.model.save_pretrained(str(path))
